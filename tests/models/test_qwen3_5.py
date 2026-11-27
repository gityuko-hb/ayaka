"""Unit and transformers parity tests for Qwen3.5 hybrid Gated DeltaNet + full attention."""

from __future__ import annotations

import math
from typing import Any

import pytest

from ayaka.configs.model import ModelConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.core.request import TokenizedRequest
from ayaka.core.types import ForwardMode, KVLayoutKind, RequestId
from ayaka.execution.runner import ModelRunner
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.prefix.hash_cache import HashPrefixCache
from ayaka.kvcache.prefix.hashing import ChainRoot
from ayaka.kvcache.state_slots import StateCheckpointStore, StateSlotPool
from ayaka.kvcache.storage.pool import KVPool, RecurrentStatePool
from ayaka.kvcache.storage.spec import KVGroup, RecurrentStateGroup
from ayaka.layers.context import ForwardContext
from ayaka.ops.backend import load_backend
from ayaka.ops.linear_attention import LinearAttentionMetadata
from ayaka.scheduler.scheduler import Scheduler
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.models.qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402
from gpt2_support import load_hf_state  # noqa: E402

_LINEAR = "linear_attention"
_FULL = "full_attention"


def _tiny_config_dict() -> dict[str, Any]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": False,
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "layer_types": [_LINEAR, _LINEAR, _LINEAR, _FULL],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "attn_output_gate": True,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 128,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            },
        },
    }


def _init_random(model: Any, *, seed: int = 7) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "a_log" in name:
                param.uniform_(-2.0, 0.5)
            elif "norm" in name or "dt_bias" in name:
                param.normal_(0.0, 0.1)
            else:
                param.normal_(0.0, 0.2)


def _collect_all_tokens(
    model: Qwen3_5ForConditionalGeneration,
    prompts: list[list[int]],
    *,
    max_tokens_per_step: int,
    prefix_cache: bool,
    gen_tokens: int = 4,
) -> list[list[int]]:
    block_size = 4
    kv_groups = model.kv_specs(block_size=block_size, layout=model.ops.kvcache.layout())
    state_group = model.state_spec()
    assert state_group is not None
    pools = [KVPool(g, 32, device="cpu") for g in kv_groups]
    state_slots = StateSlotPool(4)
    state_pool = RecurrentStatePool(state_group, 4, device="cpu")
    checkpoints = StateCheckpointStore(1 << 20, stride_blocks=1) if prefix_cache else None
    prefix = (
        HashPrefixCache(ChainRoot(model_id="tiny-q35", dtype="fp32", codec="raw", block_size=4))
        if prefix_cache
        else None
    )
    kv = KVCacheManager(
        kv_groups,
        [32],
        max_requests=3,
        max_blocks_per_request=32,
        prefix=prefix,
        state_slots=state_slots,
        state_checkpoints=checkpoints,
    )
    scheduler = Scheduler(
        SchedulerConfig(max_running=3, max_tokens_per_step=max_tokens_per_step),
        kv,
        max_model_len=128,
        max_chunk_tokens=max_tokens_per_step,
        eos_token_ids=frozenset(),
    )
    runner = ModelRunner(
        model,
        pools,
        kv.table,
        max_rows=ModelRunner.required_rows(
            token_budget=max_tokens_per_step, max_running=3, lookahead=0
        ),
        device="cpu",
        eos_token_ids=frozenset(),
        state_pool=state_pool,
        state_checkpoints=checkpoints,
    )
    results: list[list[int]] = []
    for idx, prompt in enumerate(prompts):
        rid = RequestId(f"r-{idx}")
        scheduler.add_request(
            TokenizedRequest(
                request_id=rid,
                prompt_token_ids=tuple(prompt),
                sampling=SamplingParams(temperature=0.0, max_tokens=gen_tokens),
            )
        )
        req_toks: list[int] = []
        while (step := scheduler.schedule()) is not None:
            out = runner.execute(step)
            scheduler.update(out)
            for row_toks in out.tokens:
                if row_toks:
                    req_toks.extend(int(t) for t in row_toks)
        results.append(req_toks)
    return results


@pytest.mark.unit
def test_qwen3_5_specs_and_chunk_state_prefix_invariance() -> None:
    cfg = ModelConfig.from_hf_config(_tiny_config_dict())
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model = Qwen3_5ForConditionalGeneration(cfg, ops=ops, dtype=DType.FP32)
    _init_random(model)

    kv_groups = model.kv_specs(block_size=4, layout=KVLayoutKind.NHD)
    assert len(kv_groups) == 1
    assert isinstance(kv_groups[0], KVGroup)
    assert kv_groups[0].layer_ids == (3,)

    state_group = model.state_spec()
    assert isinstance(state_group, RecurrentStateGroup)
    assert state_group.layer_ids == (0, 1, 2)

    prefix = [3, 7, 11, 15, 19, 23, 27, 31]  # 2 exact 4-token blocks
    cold = _collect_all_tokens(
        model, [[*prefix, 5, 9, 13]], max_tokens_per_step=64, prefix_cache=False
    )
    chunked = _collect_all_tokens(
        model, [[*prefix, 5, 9, 13]], max_tokens_per_step=3, prefix_cache=False
    )
    warm = _collect_all_tokens(
        model,
        [[*prefix, 1, 2], [*prefix, 5, 9, 13]],
        max_tokens_per_step=6,
        prefix_cache=True,
    )
    assert cold[0] == chunked[0] == warm[1]
    assert len(cold[0]) == 4


@pytest.mark.integration
def test_qwen3_5_matches_transformers_tiny_cpu() -> None:
    transformers: Any = pytest.importorskip("transformers")
    raw = _tiny_config_dict()
    tc = dict(raw["text_config"])
    tc["attn_implementation"] = "eager"
    hf_cfg = transformers.Qwen3_5Config(
        tie_word_embeddings=False,
        text_config=transformers.Qwen3_5TextConfig(**tc),
    )
    ref = transformers.Qwen3_5ForConditionalGeneration(hf_cfg).float().eval()
    torch.manual_seed(17)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "A_log" in name:
                param.uniform_(-1.5, 0.2)
            elif "norm" in name or "dt_bias" in name:
                param.normal_(0.0, 0.1)
            else:
                param.normal_(0.0, 0.2)

    cfg = ModelConfig.from_hf_config(raw)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    ours = Qwen3_5ForConditionalGeneration(cfg, ops=ops, dtype=DType.FP32)
    state = {k: v for k, v in ref.state_dict().items() if not k.startswith("model.visual.")}
    load_hf_state(ours, state)

    tokens = torch.tensor([2, 14, 9, 27, 33, 41, 8], dtype=torch.int64)
    block_size = 4
    kv_group = ours.kv_specs(block_size=block_size, layout=ops.kvcache.layout())[0]
    state_group = ours.state_spec()
    assert state_group is not None
    kv_pool = KVPool(kv_group, 8, device="cpu")
    st_pool = RecurrentStatePool(state_group, 2, device="cpu")
    st_pool.zero_slot(1)

    seq_len = tokens.shape[0]
    pages = math.ceil(seq_len / block_size)
    table = torch.tensor([[1, 2][:pages]], dtype=torch.int32)
    attn_meta = ops.attention.plan(
        ForwardMode.PREFILL,
        torch.tensor([seq_len], dtype=torch.int32),
        torch.tensor([seq_len], dtype=torch.int32),
        table,
        block_size,
    )
    lin_meta = LinearAttentionMetadata(
        is_prefill=True,
        cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32),
        state_slots=torch.tensor([1], dtype=torch.int32),
        rows_per_request=torch.tensor([seq_len], dtype=torch.int32),
    )
    ctx = ForwardContext(
        attention=attn_meta,
        kv_buffers={lid: kv_pool.buffers(lid) for lid in kv_group.layer_ids},
        state_buffers={lid: st_pool.buffers(lid) for lid in state_group.layer_ids},
        linear_attention=lin_meta,
    )
    with torch.no_grad():
        expected = ref(input_ids=tokens[None]).logits[0]
        hidden = ours(tokens, ctx)
        actual = ours.compute_logits(hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
