"""Generation end to end on a CUDA GPU: the tiny GPT-2 and a tiny Llama through the public ``LLM``.

The same requests run in every configuration M1 adds: the ``torch`` backend and ``flashinfer``
(attention) over ``torch`` (everything else), eager and with decode steps in CUDA graphs, fp32,
fp16 and bf16. The oracles:

* in fp32 on ``torch``, the tokens the plain formulas pick (``greedy_reference`` for GPT-2,
  ``llama_reference_hidden`` for Llama), as on the CPU;
* in fp16 and bf16, the tokens of the same backend and dtype run eagerly: a graph replays the
  eager kernels, so it must draw exactly what eager draws, sampled requests included.

After every run the KV pool must be entirely free, and on the GPU the start-up run must have
filled a memory ledger that matches what the allocator holds; the driver's free memory is logged
as a diagnostic, not used for the match.

Every model here has heads of 64 so that flashinfer has kernels for it; the vocabulary is the byte
tokenizer's.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from ayaka.configs.device import DeviceConfig
from ayaka.core.constraints import ConstraintKind, StructuredParams
from ayaka.core.params import SamplingParams
from ayaka.core.request import GenerateRequest
from ayaka.core.types import RequestId
from ayaka.utils.dtypes import DType

torch: Any = pytest.importorskip("torch")

from ayaka.engine.llm import LLM  # noqa: E402
from gpt2_support import write_checkpoint  # noqa: E402
from llama_support import llama_config, llama_reference_hidden, llama_state, write_llama  # noqa: E402
from tiny_model import (  # noqa: E402  # needs the importorskip above
    MAX_POSITIONS,
    assert_cache_released,
    greedy_reference,
    prompt_ids,
    tiny_engine_config,
    tiny_gpt2_config,
    tiny_gpt2_state,
    tiny_tokenizer,
)
from tokenizer_fakes import MODEL_VOCAB  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

    from ayaka.configs.model import ModelConfig

pytestmark = pytest.mark.gpu

PROMPTS = ("hello", "The capital of France is", "x", "a prompt that is longer than several blocks")
NEW_TOKENS = 12


@pytest.fixture(autouse=True)
def _profiles_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the start-up profiles of these tests out of the user's cache."""
    monkeypatch.setenv("AYAKA_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def gpt2(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ModelConfig, dict[str, Any]]:
    # intermediate_size is 4 * hidden: the GPT-2 config.json leaves n_inner out.
    config = tiny_gpt2_config(
        hidden_size=128, num_attention_heads=2, num_kv_heads=2, head_dim=64, intermediate_size=512
    )
    state = tiny_gpt2_state(config)
    directory = tmp_path_factory.mktemp("gpu-gpt2")
    write_checkpoint(directory, state, config)
    return directory, config, state


@pytest.fixture(scope="module")
def llama(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ModelConfig, dict[str, Any]]:
    config = llama_config(
        vocab_size=MODEL_VOCAB,
        hidden_size=256,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=384,
        max_position_embeddings=MAX_POSITIONS,
        tie_word_embeddings=False,
    )
    state = llama_state(config, seed=2)
    # As in the GPT-2 fixture, quiet the rows outside printable ASCII, so text stays readable.
    quiet = torch.tensor([i not in range(32, 127) for i in range(config.vocab_size)])
    state["lm_head.weight"][quiet] *= 0.01
    directory = tmp_path_factory.mktemp("gpu-llama")
    write_llama(directory, state, config)
    return directory, config, state


def _llm(directory: Path, *, backends: tuple[str, ...], dtype: DType, graphs: bool) -> LLM:
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer", reason="the flashinfer extra is not installed")
    config = tiny_engine_config(directory, block_size=16, num_blocks=65, max_running=4)
    config = replace(
        config,
        device=DeviceConfig(device="cuda", dtype=dtype, backends=backends, cuda_graph=graphs),
    )
    return LLM.from_config(config, tokenizer=tiny_tokenizer())


def _generate(llm: LLM, sampling: SamplingParams) -> list[tuple[int, ...]]:
    outs = llm.generate(list(PROMPTS), sampling)
    assert isinstance(outs, list)
    for out in outs:
        assert out.error is None, out.error
    assert_cache_released(llm.engine)
    return [out.outputs[0].token_ids for out in outs]


def _greedy() -> SamplingParams:
    # ignore_eos fixes the length without a min_tokens mask, which would make the steps eager.
    return SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)


def _sampled() -> SamplingParams:
    return SamplingParams(
        temperature=0.8, top_k=40, top_p=0.95, seed=11, max_tokens=NEW_TOKENS, ignore_eos=True
    )


def _llama_greedy(state: dict[str, Any], config: ModelConfig, prompt: list[int]) -> list[int]:
    tokens = list(prompt)
    for _ in range(NEW_TOKENS):
        hidden = llama_reference_hidden(state, config, torch.tensor(tokens))
        tokens.append(int((hidden[-1] @ state["lm_head.weight"].T).argmax()))
    return tokens[len(prompt) :]


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graph"])
def test_gpt2_in_fp32_on_torch_draws_the_tokens_of_the_formulas(
    gpt2: tuple[Path, ModelConfig, dict[str, Any]], graphs: bool
) -> None:
    directory, config, state = gpt2
    with _llm(directory, backends=("torch",), dtype=DType.FP32, graphs=graphs) as llm:
        tokens = _generate(llm, _greedy())
        startup = llm.engine.startup
        assert startup is not None
        assert startup.reconciliation.matches, startup.reconciliation
        if graphs:
            assert startup.graphs
            assert llm.engine.runner.graph is not None
            assert llm.engine.runner.graph.replays > 0

    for prompt, got in zip(PROMPTS, tokens, strict=True):
        assert list(got) == greedy_reference(state, config, prompt_ids(prompt), NEW_TOKENS)


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graph"])
def test_llama_in_fp32_on_torch_draws_the_tokens_of_the_formulas(
    llama: tuple[Path, ModelConfig, dict[str, Any]], graphs: bool
) -> None:
    directory, config, state = llama
    with _llm(directory, backends=("torch",), dtype=DType.FP32, graphs=graphs) as llm:
        tokens = _generate(llm, _greedy())

    for prompt, got in zip(PROMPTS, tokens, strict=True):
        assert list(got) == _llama_greedy(state, config, prompt_ids(prompt))


@pytest.mark.parametrize("model", ["gpt2", "llama"])
@pytest.mark.parametrize("backends", [("torch",), ("flashinfer", "torch")], ids=["torch", "fi"])
@pytest.mark.parametrize("dtype", [DType.FP16, DType.BF16], ids=["fp16", "bf16"])
def test_graphs_draw_what_eager_draws(
    request: pytest.FixtureRequest, model: str, backends: tuple[str, ...], dtype: DType
) -> None:
    directory, _, _ = request.getfixturevalue(model)
    results = []
    for graphs in (False, True):
        with _llm(directory, backends=backends, dtype=dtype, graphs=graphs) as llm:
            if "flashinfer" in backends:
                assert llm.engine.runner.model.ops.providers["attention"] == "flashinfer"
            results.append((_generate(llm, _greedy()), _generate(llm, _sampled())))
            startup = llm.engine.startup
            assert startup is not None
            assert startup.reconciliation.matches, startup.reconciliation

    (eager_greedy, eager_sampled), (graph_greedy, graph_sampled) = results
    assert graph_greedy == eager_greedy
    assert graph_sampled == eager_sampled


@pytest.mark.parametrize("backends", [("torch",), ("flashinfer", "torch")], ids=["torch", "fi"])
def test_several_sequences_draw_the_same_in_graphs_and_eager(
    gpt2: tuple[Path, ModelConfig, dict[str, Any]], backends: tuple[str, ...]
) -> None:
    """The sequence index reaches the sampler through the graph's fixed buffer as well."""
    directory, _, _ = gpt2
    results: list[list[tuple[int, ...]]] = []
    for graphs in (False, True):
        with _llm(directory, backends=backends, dtype=DType.FP16, graphs=graphs) as llm:
            out = llm.generate(PROMPTS[0], replace(_sampled(), n=3))
            assert not isinstance(out, list)
            assert out.error is None, out.error
            results.append([completion.token_ids for completion in out.outputs])
            assert_cache_released(llm.engine)

    assert results[0] == results[1]
    assert len(set(results[0])) > 1  # the sequences drew from independent streams


@pytest.mark.parametrize("backends", [("torch",), ("flashinfer", "torch")], ids=["torch", "fi"])
def test_sampling_controls_run_on_the_gpu_beside_graphs(
    gpt2: tuple[Path, ModelConfig, dict[str, Any]], backends: tuple[str, ...]
) -> None:
    """A request with penalties, a bias, min_tokens and logprobs runs eagerly next to graphs."""
    directory, _, _ = gpt2
    controlled = replace(
        _sampled(),
        presence_penalty=0.5,
        repetition_penalty=1.2,
        logit_bias={66: 2.0},
        logprobs=2,
        max_tokens=4,  # ends early, so the plain request's later steps replay graphs
        min_tokens=3,
    )
    results = []
    for graphs in (False, True):
        with _llm(directory, backends=backends, dtype=DType.FP16, graphs=graphs) as llm:
            outs = llm.generate([PROMPTS[0], PROMPTS[1]], [controlled, _greedy()])
            assert isinstance(outs, list)
            for out in outs:
                assert out.error is None, out.error
            logprobs = outs[0].outputs[0].logprobs
            assert logprobs is not None
            assert [entry.token_id for entry in logprobs] == list(outs[0].outputs[0].token_ids)
            assert len(llm.engine.runner.penalties) == 0
            if graphs:
                assert llm.engine.runner.graph is not None
                assert llm.engine.runner.graph.replays > 0  # the plain request still replays
            results.append([out.outputs[0].token_ids for out in outs])
            assert_cache_released(llm.engine)

    assert results[0] == results[1]


@pytest.mark.parametrize("backends", [("torch",), ("flashinfer", "torch")], ids=["torch", "fi"])
def test_a_json_constraint_holds_on_the_gpu_beside_graphs(
    gpt2: tuple[Path, ModelConfig, dict[str, Any]], backends: tuple[str, ...]
) -> None:
    """A schema-constrained request runs eagerly and always parses; the plain one keeps graphs."""
    pytest.importorskip("llguidance", reason="the structured extra is not installed")
    directory, _, _ = gpt2
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 0, "maximum": 9}},
        "required": ["n"],
        "additionalProperties": False,
    }
    constrained = GenerateRequest(
        request_id=RequestId("json"),
        prompt=PROMPTS[0],
        sampling=SamplingParams(temperature=1.0, seed=3, max_tokens=12),
        structured=StructuredParams(kind=ConstraintKind.JSON_SCHEMA, schema=schema),
    )
    results = []
    for graphs in (False, True):
        with _llm(directory, backends=backends, dtype=DType.FP16, graphs=graphs) as llm:
            outs = llm.generate(
                [
                    constrained,
                    GenerateRequest(
                        request_id=RequestId("plain"), prompt=PROMPTS[1], sampling=_greedy()
                    ),
                ]
            )
            assert isinstance(outs, list)
            for out in outs:
                assert out.error is None, out.error
            assert set(json.loads(outs[0].outputs[0].text)) == {"n"}
            if graphs:
                assert llm.engine.runner.graph is not None
                assert llm.engine.runner.graph.replays > 0
            results.append([out.outputs[0].token_ids for out in outs])
            assert_cache_released(llm.engine)

    assert results[0] == results[1]


def test_flashinfer_and_torch_agree_on_greedy_tokens_in_fp16(
    gpt2: tuple[Path, ModelConfig, dict[str, Any]],
) -> None:
    directory, _, _ = gpt2
    with _llm(directory, backends=("torch",), dtype=DType.FP16, graphs=False) as llm:
        reference = _generate(llm, _greedy())
    with _llm(directory, backends=("flashinfer", "torch"), dtype=DType.FP16, graphs=True) as llm:
        fast = _generate(llm, _greedy())

    # The two attention kernels round differently; the tiny model's top logits are far apart, so
    # the greedy choice is the same. A real model may differ late in a sequence.
    assert fast == reference


def test_the_startup_profile_is_saved_and_read_back(
    gpt2: tuple[Path, ModelConfig, dict[str, Any]], tmp_path: Path
) -> None:
    directory, _, _ = gpt2
    with _llm(directory, backends=("torch",), dtype=DType.FP16, graphs=True) as llm:
        first = llm.engine.startup
        assert first is not None
        assert first.profile.activation_peak_bytes > 0
        assert first.profile.graph_bytes >= 0
    saved = list((tmp_path / "cache" / "profiles").glob("*.json"))
    assert len(saved) == 1

    with _llm(directory, backends=("torch",), dtype=DType.FP16, graphs=True) as llm:
        plan = llm.engine.plan
        startup = llm.engine.startup
        assert startup is not None
    # The second start-up planned with the first one's numbers: its chunk limit and KV share
    # the headroom with the activations and graphs measured before.
    assert plan.graph_bytes == first.profile.graph_bytes or plan.graph_bytes == 0
    assert startup.profile.key == first.profile.key
