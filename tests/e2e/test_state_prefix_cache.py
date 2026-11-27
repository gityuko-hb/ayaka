"""End-to-end hybrid model tests for block-aligned recurrent state prefix caching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.model import ModelConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import FinishReason, RequestId
from ayaka.execution.runner import ModelRunner
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.kvcache.state_slots import StateCheckpointStore, StateSlotPool
from ayaka.kvcache.storage.pool import KVPool, RecurrentStatePool
from ayaka.kvcache.storage.spec import (
    KVGroup,
    KVSpec,
    RecurrentStateGroup,
    RecurrentStateSpec,
)
from ayaka.layers.attention.dense import Attention
from ayaka.layers.attention.linear import GatedDeltaNetAttention
from ayaka.layers.embedding.token import LMHead, TokenEmbedding
from ayaka.layers.norm import RMSNorm
from ayaka.layers.rope import RotaryEmbedding
from ayaka.models.base import CausalLM
from ayaka.ops.backend import load_backend
from ayaka.ops.kvcache import RAW_CODEC
from ayaka.scheduler.scheduler import Scheduler
from ayaka.utils.dtypes import DType
from ayaka.weights.mapping import WeightMapping

torch: Any = pytest.importorskip("torch")

if TYPE_CHECKING:
    from torch import Tensor

    from ayaka.core.types import KVLayoutKind
    from ayaka.layers.context import ForwardContext

pytestmark = pytest.mark.unit


class _ToyHybridModel(CausalLM):
    """Two-layer hybrid model: layer 0 GatedDeltaNet, layer 1 full attention."""

    def __init__(self, config: ModelConfig, *, ops: Any) -> None:
        super().__init__(config, ops=ops, dtype=DType.FP32, device="cpu")
        dt = torch.float32
        h = config.hidden_size
        self.embed = TokenEmbedding(config.vocab_size, h, ops=ops, dtype=dt, device="cpu")
        self.rotary = RotaryEmbedding(
            rotary_dim=config.head_dim,
            max_positions=config.max_position_embeddings,
            theta=10000.0,
            ops=ops,
            device="cpu",
        )
        self.norm0 = RMSNorm(h, eps=1e-6, ops=ops, dtype=dt, device="cpu")
        self.gdn = GatedDeltaNetAttention(
            hidden_size=h,
            num_k_heads=2,
            num_v_heads=4,
            key_head_dim=8,
            value_head_dim=8,
            conv_kernel_size=4,
            layer_id=0,
            ops=ops,
            dtype=dt,
            device="cpu",
        )
        self.norm1 = RMSNorm(h, eps=1e-6, ops=ops, dtype=dt, device="cpu")
        self.attn = Attention(
            h,
            config.num_attention_heads,
            config.num_kv_heads,
            config.head_dim,
            layer_id=1,
            bias=False,
            rotary=self.rotary,
            ops=ops,
            dtype=dt,
            device="cpu",
        )
        self.final_norm = RMSNorm(h, eps=1e-6, ops=ops, dtype=dt, device="cpu")
        self.head = LMHead(h, config.vocab_size, ops=ops, dtype=dt, device="cpu")
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(mean=0.0, std=0.08)
            self.norm0.weight.fill_(1.0)
            self.norm1.weight.fill_(1.0)
            self.final_norm.weight.fill_(1.0)
            self.gdn.norm_weight.fill_(1.0)
            self.gdn.a_log.fill_(-1.0)
            self.gdn.dt_bias.fill_(0.0)
        self.initialize_buffers()

    @classmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        del config
        return WeightMapping(())

    def kv_specs(
        self,
        *,
        block_size: int,
        layout: KVLayoutKind,
        codec: str = RAW_CODEC,
        kv_scale: float = 1.0,
    ) -> tuple[KVGroup, ...]:
        spec = KVSpec(
            num_layers=1,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            block_size=block_size,
            dtype=self.compute_dtype,
            layout=layout,
            codec=codec,
            kv_scale=kv_scale,
        )
        return (KVGroup(layer_ids=(1,), spec=spec),)

    def state_spec(self) -> RecurrentStateGroup:
        return RecurrentStateGroup(
            layer_ids=(0,),
            spec=RecurrentStateSpec(
                num_layers=1,
                num_v_heads=4,
                key_head_dim=8,
                value_head_dim=8,
                conv_dim=2 * 2 * 8 + 4 * 8,
                conv_kernel_size=4,
                state_dtype=DType.FP32,
                conv_dtype=DType.FP32,
            ),
        )

    def forward(self, token_ids: Tensor, context: ForwardContext) -> Tensor:
        x = self.embed(token_ids)
        x = x + self.gdn(self.norm0(x), context)
        x = x + self.attn(self.norm1(x), context)
        return self.final_norm(x)

    def compute_logits(self, hidden_rows: Tensor) -> Tensor:
        return self.head(hidden_rows)


def _run_stack(
    *,
    block_size: int = 4,
    stride: int = 1,
    ckpt_bytes: int = 10 * 1024 * 1024,
    max_tokens_per_step: int = 64,
    prefix_cache: bool = True,
) -> tuple[_ToyHybridModel, Scheduler, ModelRunner, KVCacheManager]:
    torch.manual_seed(101)
    cfg = ModelConfig(
        architecture="toy_hybrid",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        hidden_act="silu",
        norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model = _ToyHybridModel(cfg, ops=ops)
    kv_group = model.kv_specs(block_size=block_size, layout=ops.kvcache.layout())[0]
    state_group = model.state_spec()
    prefix = (
        HashPrefixCache(
            ChainRoot(model_id="toy-hybrid", dtype="fp32", codec="raw", block_size=block_size)
        )
        if prefix_cache
        else None
    )
    ckpts = StateCheckpointStore(ckpt_bytes, stride_blocks=stride) if prefix_cache else None
    slots = StateSlotPool(4)
    mgr = KVCacheManager(
        [kv_group],
        [32],
        max_requests=3,
        max_blocks_per_request=16,
        prefix=prefix,
        state_slots=slots,
        state_checkpoints=ckpts,
    )
    kv_pool = KVPool(kv_group, 32, device="cpu")
    st_pool = RecurrentStatePool(state_group, 4, device="cpu")
    sched = Scheduler(
        SchedulerConfig(max_running=3, max_tokens_per_step=max_tokens_per_step),
        mgr,
        max_model_len=128,
    )
    runner = ModelRunner(
        model,
        [kv_pool],
        mgr.table,
        max_rows=max_tokens_per_step,
        device="cpu",
        state_pool=st_pool,
        state_checkpoints=ckpts,
    )
    return model, sched, runner, mgr


def _drive(
    sched: Scheduler, runner: ModelRunner, rid: RequestId, prompt: tuple[int, ...], gen: int
) -> tuple[list[int], int]:
    sched.add_request(
        TokenizedRequest(
            request_id=rid,
            prompt_token_ids=prompt,
            sampling=SamplingParams(temperature=0.0, max_tokens=gen),
        )
    )
    out_tokens: list[int] = []
    cached = 0
    while (step := sched.schedule()) is not None:
        out = runner.execute(step)
        sched.update(out)
        for ev in sched.take_events():
            if ev.request_id == rid:
                cached = max(cached, ev.cached_tokens)
                out_tokens.extend(ev.token_ids)
                if ev.finished:
                    assert ev.finish_reason is FinishReason.LENGTH
    return out_tokens, cached


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_hybrid_state_checkpoint_prefix_reuse_matches_cold(stride: int) -> None:
    _, sched, runner, mgr = _run_stack(block_size=4, stride=stride)
    # 19 tokens -> 4 full blocks (tokens 0..16), 3 trailing tokens
    shared = tuple(range(1, 20))
    _cold_tokens, cold_cached = _drive(sched, runner, RequestId("cold"), shared, gen=5)
    assert cold_cached == 0

    # Shared prefix + new suffix (forces recompute from last checkpointed block)
    followup = (*shared[:17], 31, 32, 33)
    warm_tokens, warm_cached = _drive(sched, runner, RequestId("warm"), followup, gen=5)

    # Exact cold comparison with prefix cache disabled for followup!
    _, sched_nc, runner_nc, _ = _run_stack(block_size=4, prefix_cache=False)
    expected_followup, _ = _drive(sched_nc, runner_nc, RequestId("ref"), followup, gen=5)
    assert warm_tokens == expected_followup

    # With 17 shared tokens (4 full blocks indices 0..3):
    # stride=1 -> blocks 0..3 (16 tokens); stride=2 -> idx 3 (16); stride=4 -> idx 3 (16).
    assert warm_cached == 16

    # Matching 12 tokens (3 blocks) with stride=2 aligns back to block 1 (8 tokens).
    prefix13 = (*shared[:13], 41)
    warm13, warm13_cached = _drive(sched, runner, RequestId("w13"), prefix13, gen=5)
    _, sched_nc2, runner_nc2, _ = _run_stack(block_size=4, prefix_cache=False)
    exp13, _ = _drive(sched_nc2, runner_nc2, RequestId("r13"), prefix13, gen=5)
    assert warm13 == exp13
    expected_hit = {1: 12, 2: 8, 4: 0}[stride]
    assert warm13_cached == expected_hit
    mgr.check_integrity()


def test_chunked_prefill_matches_single_chunk_prefill() -> None:
    prompt = tuple(range(2, 24))
    _, sched_full, runner_full, _ = _run_stack(
        block_size=4, max_tokens_per_step=64, prefix_cache=False
    )
    expected, _ = _drive(sched_full, runner_full, RequestId("full"), prompt, gen=4)

    _, sched_chunk, runner_chunk, mgr_chunk = _run_stack(
        block_size=4, max_tokens_per_step=6, prefix_cache=False
    )
    chunked, _ = _drive(sched_chunk, runner_chunk, RequestId("chunk"), prompt, gen=4)
    assert chunked == expected
    mgr_chunk.check_integrity()


def test_state_checkpoint_eviction_falls_back_cleanly() -> None:
    _, sched, runner, mgr = _run_stack(block_size=4, stride=1)
    p1 = tuple(range(1, 18))
    cold_out, _ = _drive(sched, runner, RequestId("r1"), p1, gen=4)

    # Clear checkpoint store to simulate LRU host-budget eviction while KV prefix stays present
    assert mgr.state_checkpoints is not None
    mgr.state_checkpoints.clear()
    warm_after_evict, warm_cached = _drive(sched, runner, RequestId("r2"), p1, gen=4)
    assert warm_cached == 0
    assert warm_after_evict == cold_out
    mgr.check_integrity()
