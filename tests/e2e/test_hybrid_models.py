"""GPU parity, chunk invariance, CUDA-graph, and state-prefix tests for hybrid models."""

from __future__ import annotations

import gc
import importlib
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

from ayaka.configs.cache import CacheConfig  # noqa: E402
from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.engine import EngineConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.configs.scheduler import SchedulerConfig  # noqa: E402
from ayaka.core.params import SamplingParams  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402

pytestmark = pytest.mark.gpu

_PROMPT = "Explain in two clear sentences how paged key-value caching reduces LLM memory waste:"


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _hf_tokens(model_id: str, prompt: str, max_new: int = 12) -> tuple[list[int], list[int]]:
    tok: Any = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    prompt_ids = [int(t) for t in tok.encode(prompt, add_special_tokens=True)]
    cls: Any = (
        transformers.AutoModelForImageTextToText
        if ("Qwen3.5" in model_id or "gemma-4" in model_id)
        else transformers.AutoModelForCausalLM
    )
    extra: dict[str, Any] = {}
    if "gpt-oss" in model_id:
        quant_cfg = importlib.import_module("transformers.utils.quantization_config")
        extra["quantization_config"] = quant_cfg.Mxfp4Config(dequantize=True)
    hf_model: Any = cls.from_pretrained(
        model_id, local_files_only=True, dtype=torch.bfloat16, **extra
    ).to("cuda")
    hf_model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.int64, device="cuda")
    with torch.inference_mode():
        seq = hf_model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new,
            do_sample=False,
        )
    out = [int(t) for t in seq[0, len(prompt_ids) :].tolist()]
    del hf_model
    _clear_cuda()
    return prompt_ids, out


def _cfg(
    model_id: str,
    *,
    max_tokens_per_step: int = 256,
    prefix_cache: bool = True,
    stride: int = 1,
) -> EngineConfig:
    return EngineConfig(
        source=ModelSourceConfig(model_id, offline=True),
        device=DeviceConfig(
            device="cuda",
            dtype=DType.BF16,
            backends=("flashinfer", "torch"),
            cuda_graph=True,
        ),
        cache=CacheConfig(
            block_size=16,
            kv_reserve_tokens=2048,
            prefix_cache=prefix_cache,
            state_checkpoint_stride=stride,
        ),
        scheduler=SchedulerConfig(max_running=4, max_tokens_per_step=max_tokens_per_step),
    )


@pytest.mark.parametrize(
    ("model_id", "expected_groups"),
    [
        ("Qwen/Qwen3.5-9B", 1),
        ("google/gemma-4-E2B-it", 2),
        ("openai/gpt-oss-20b", 2),
    ],
)
def test_real_checkpoint_greedy_parity_and_cuda_graphs(model_id: str, expected_groups: int) -> None:
    prompt_ids, expected = _hf_tokens(model_id, _PROMPT, max_new=12)
    params = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=False)
    with LLM.from_config(_cfg(model_id)) as llm:
        assert "flashinfer" in llm.engine.runner.model.ops.providers.values()
        assert len(llm.engine.kv.groups) == expected_groups
        assert llm.engine.startup is not None
        assert llm.engine.startup.graphs
        out = llm.generate(tuple(prompt_ids), params)
        assert not isinstance(out, list)
        assert out.error is None
        actual = [int(t) for t in out.outputs[0].token_ids]
    _clear_cuda()
    assert actual == expected


def _run_cold_case(
    model_id: str, prompt: tuple[int, ...], *, max_tokens_per_step: int
) -> list[int]:
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    with LLM.from_config(
        _cfg(model_id, max_tokens_per_step=max_tokens_per_step, prefix_cache=False)
    ) as llm:
        res = llm.generate(prompt, params)
        assert not isinstance(res, list)
        out = [int(t) for t in res.outputs[0].token_ids]
    _clear_cuda()
    return out


def _run_warm_case(
    model_id: str, prompt_a: tuple[int, ...], prompt_b: tuple[int, ...]
) -> list[int]:
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    with LLM.from_config(
        _cfg(model_id, max_tokens_per_step=128, prefix_cache=True, stride=2)
    ) as llm:
        store = llm.engine.kv.state_checkpoints
        assert store is not None
        llm.generate(prompt_a, params)
        hits_before = store.hits
        res_warm = llm.generate(prompt_b, params)
        assert not isinstance(res_warm, list)
        warm_ids = [int(t) for t in res_warm.outputs[0].token_ids]
        assert store.hits > hits_before
    _clear_cuda()
    return warm_ids


def test_qwen3_5_9b_chunk_and_state_prefix_exact_parity() -> None:
    model_id = "Qwen/Qwen3.5-9B"
    tok: Any = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    prefix = [
        int(t)
        for t in tok.encode("System prompt: keep answers concise. " * 12, add_special_tokens=True)
    ][:64]
    assert len(prefix) == 64
    suffix_a = [
        int(t) for t in tok.encode(" Describe dynamic batching." * 3, add_special_tokens=False)
    ][:12]
    suffix_b = [
        int(t) for t in tok.encode(" Describe prefix caching." * 3, add_special_tokens=False)
    ][:12]
    prompt_b = (*prefix, *suffix_b)

    single_ids = _run_cold_case(model_id, prompt_b, max_tokens_per_step=512)
    chunk_ids = _run_cold_case(model_id, prompt_b, max_tokens_per_step=24)
    warm_ids = _run_warm_case(model_id, (*prefix, *suffix_a), prompt_b)
    assert single_ids == chunk_ids == warm_ids


def test_deepseek_mla_flashinfer_cuda_graph_gpu(tmp_path: Any) -> None:
    pytest.importorskip("flashinfer")
    import json

    from ayaka.configs.model_source import RequestFormat
    from ayaka.engine.build import plan_engine
    from ayaka.kvcache.storage.spec import LatentKVGroup
    from gpt2_support import tensor_data
    from safetensors_files import write_safetensors
    from tiny_model import tiny_tokenizer

    raw: dict[str, Any] = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "vocab_size": 260,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "q_lora_rank": 256,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "rope_interleave": True,
        "n_routed_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 0,
        "n_shared_experts": 0,
        "first_k_dense_replace": 2,
    }
    hf_cfg = transformers.DeepseekV2Config(**raw, attn_implementation="eager")
    torch.manual_seed(42)
    ref = transformers.DeepseekV2ForCausalLM(hf_cfg).to(torch.bfloat16)
    with torch.no_grad():
        for name, param in ref.named_parameters():
            if "norm" in name:
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.04)

    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    write_safetensors(
        tmp_path / "model.safetensors",
        {name: tensor_data(tensor, torch.bfloat16) for name, tensor in ref.state_dict().items()},
    )
    prompt = tuple(range(5, 25))
    ref_cuda = ref.to("cuda").eval()
    ids_t = torch.tensor([prompt], dtype=torch.int64, device="cuda")
    with torch.inference_mode():
        seq = ref_cuda.generate(
            input_ids=ids_t,
            attention_mask=torch.ones_like(ids_t),
            max_new_tokens=10,
            do_sample=False,
        )
    expected = [int(t) for t in seq[0, len(prompt) :].tolist()]
    del ref_cuda, ref
    _clear_cuda()

    cfg = EngineConfig(
        source=ModelSourceConfig(str(tmp_path), format=RequestFormat.SAFETENSORS, offline=True),
        device=DeviceConfig(
            device="cuda",
            dtype=DType.BF16,
            backends=("flashinfer", "torch"),
            cuda_graph=True,
            graph_buckets=(1, 2),
        ),
        cache=CacheConfig(block_size=16, kv_reserve_tokens=512, prefix_cache=True),
        scheduler=SchedulerConfig(max_running=2, max_tokens_per_step=16),
    )
    plan = plan_engine(cfg)
    assert isinstance(plan.group, LatentKVGroup)
    params = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True)
    with LLM.from_config(cfg, tokenizer=tiny_tokenizer()) as llm:
        out = llm.generate(prompt, params)
        assert not isinstance(out, list)
        assert out.error is None
        actual = [int(t) for t in out.outputs[0].token_ids]
    _clear_cuda()
    assert actual == expected
