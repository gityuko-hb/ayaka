"""GPU parity, chunk invariance, CUDA-graph, and state-prefix tests for hybrid models."""

from __future__ import annotations

import gc
from typing import Any

import pytest

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.params import SamplingParams
from ayaka.engine.llm import LLM
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")
transformers: Any = pytest.importorskip("transformers")

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
        from transformers.utils.quantization_config import Mxfp4Config

        extra["quantization_config"] = Mxfp4Config(dequantize=True)
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
