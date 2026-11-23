"""A real Llama checkpoint on the GPU against ``transformers``: the same greedy tokens.

The checkpoint is ``HuggingFaceTB/SmolLM2-135M`` (``LlamaForCausalLM``, 135M parameters, grouped
heads, tied embeddings), run through the public ``LLM`` with its real tokenizer, in every GPU
configuration: the ``torch`` backend and ``flashinfer`` attention, eager and with decode graphs,
fp32, fp16 and bf16.

The reference is ``transformers``' greedy ``generate`` on the same GPU in **float32** with eager
attention, whatever the configuration under test uses. Float32 is the more precise source, never
less precise than the code under test, so a token that agrees with it is right; a reference in a
lower precision can instead call a right answer wrong where its own rounding moves the top two
logits (issue #39: TinyLlama-1.1B in fp16 chose 363 over 714 with logits 18.00762 and 17.99535, a
0.0123 gap that fp16 cannot resolve reliably). Every generated token must match, with one
exception: when the first difference for a prompt is at token ``i``, it is accepted as a near tie
if the reference's best two logits there are closer than ``4 * eps(dtype) * |top1|``, the gap that
the configuration's own rounding can close. Such a token is reported as a warning, with the number
of remaining tokens left unchecked: the prefixes have diverged, so nothing after the first
difference is comparable. Any other difference fails with the prompt, the index, both tokens, the
margin and the bound. On the default 135M checkpoint the comparison stayed strict: the six
configurations matched every token against float32, so no near tie needed the exception.

A configuration whose weights do not fit the card is skipped with the engine's own numbers
(``CapacityError``: need, have, short), but only when ``AYAKA_LLAMA_PATH`` names the checkpoint: a
developer chose it, and no numerical question is at stake. The default 135M downloaded from the Hub
always fits and never skips this way; every other capacity error, and any capacity error for the
default checkpoint, still fails.

Expected results on the dev machine (RTX 3050 Laptop, WSL2, transformers 5.22.2, flashinfer
0.7.4): the default 135M matches ``transformers`` in float32 on all 32 tokens of the three
prompts in all six configurations, so a full run reports 6 passed; with ``AYAKA_LLAMA_PATH`` at
TinyLlama-1.1B-Chat-v1.0 on the same 4 GB card, the two fp32 configurations do not fit (their
fp32 weights and fixed memory need 4.40 GB, more than the roughly 3.3-3.7 GB a 4 GB card offers)
and skip with the engine's own need, have and short, so that run reports 4 passed and 2 skipped.
Both were measured; the numbers in a skip reason follow the free memory of the moment.

What it needs, and what it does when something is missing (as ``test_gpt2_parity.py``):

* a CUDA GPU, torch with CUDA, ``transformers`` and ``huggingface_hub`` (``--extra hf``), and
  ``flashinfer`` for its configurations (``--extra flashinfer``);
* the checkpoint: a local directory named by ``AYAKA_LLAMA_PATH``, or else a download of about
  270 MB from the Hugging Face Hub into its usual cache.

A missing piece skips with the reason; ``AYAKA_REQUIRE_REAL_MODEL=1`` turns the skip into a failure.
``AYAKA_REFERENCE_DEVICE=cpu`` runs the float32 reference on the CPU instead of the GPU, for a
checkpoint whose float32 copy does not fit the card (Phi-3-mini-4k, 3.8B, on the 24 GB L4).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Final, NoReturn

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

REPO_ID: Final = "HuggingFaceTB/SmolLM2-135M"
LOCAL_PATH_VARIABLE: Final = "AYAKA_LLAMA_PATH"
REQUIRE_VARIABLE: Final = "AYAKA_REQUIRE_REAL_MODEL"
#: Where the float32 reference runs: ``cuda`` (the default) or ``cpu``, for a checkpoint whose
#: float32 copy does not fit the card next to nothing else (Phi-3-mini on 24 GB).
REFERENCE_DEVICE_VARIABLE: Final = "AYAKA_REFERENCE_DEVICE"
_FILES: Final = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]
NEW_TOKENS: Final = 32
PROMPTS: Final = (
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Once upon a time, in a land far away,",
)
#: Long enough for several full 16-token blocks, so a second run reuses them from the prefix cache.
LONG_PROMPT: Final = (
    "The following is a short history of the printing press. "
    "Johannes Gutenberg introduced movable type in Europe around 1440, and within a few decades "
    "printing shops had opened in most of the large cities of the continent. Books became "
    "cheaper, literacy spread, and the"
)


def _unavailable(reason: str) -> NoReturn:
    if os.environ.get(REQUIRE_VARIABLE) == "1":
        pytest.fail(
            f"{REQUIRE_VARIABLE}=1 but the real Llama test cannot run: {reason}", pytrace=False
        )
    pytest.skip(reason, allow_module_level=True)


def _import(name: str, hint: str) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        _unavailable(f"cannot import {name} ({exc}); {hint}")


torch: Any = _import("torch", "install it with: uv sync --extra cu130")
transformers: Any = _import("transformers", "install it with: uv sync --extra hf")

from ayaka.configs.cache import CacheConfig  # noqa: E402
from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.engine import EngineConfig  # noqa: E402
from ayaka.configs.model import ModelConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.configs.scheduler import SchedulerConfig  # noqa: E402
from ayaka.core.errors import CapacityError  # noqa: E402
from ayaka.core.params import SamplingParams  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402

_TORCH_DTYPES = {DType.FP32: "float32", DType.FP16: "float16", DType.BF16: "bfloat16"}
CONFIGS = [
    pytest.param(("torch",), DType.FP32, False, id="torch-fp32-eager"),
    pytest.param(("torch",), DType.FP32, True, id="torch-fp32-graph"),
    pytest.param(("torch",), DType.FP16, True, id="torch-fp16-graph"),
    pytest.param(("flashinfer", "torch"), DType.FP16, False, id="flashinfer-fp16-eager"),
    pytest.param(("flashinfer", "torch"), DType.FP16, True, id="flashinfer-fp16-graph"),
    pytest.param(("flashinfer", "torch"), DType.BF16, True, id="flashinfer-bf16-graph"),
]

#: How many epsilons of the configuration's dtype a near tie may span, relative to the top logit.
_TIE_FACTOR: Final = 4.0
#: The ``what`` of the capacity error a checkpoint raises when its weights alone do not fit.
_WEIGHTS_WHAT: Final = "weights and fixed memory"


class NearTieWarning(UserWarning):
    """A token chosen where the reference's top two logits are within the dtype's rounding."""


@dataclass(frozen=True, slots=True)
class Reference:
    """What ``transformers`` in float32 says about one prompt.

    Attributes:
        tokens: The ``NEW_TOKENS`` tokens greedy decoding produced.
        top1: The largest logit, in float32, of the distribution each token was drawn from.
        second: The second largest logit of that distribution: the best competing token,
            the only one that matters for a near tie.
    """

    tokens: tuple[int, ...]
    top1: tuple[float, ...]
    second: tuple[float, ...]


def _make_reference(tokenizer: Any, model: Any, prompt: str) -> Reference:
    """Take the argmax ``NEW_TOKENS`` times, then read the top two logits at every step.

    The loop is greedy decoding and nothing else, as Ayaka runs it here with ``ignore_eos``.
    ``generate`` would add the logits processors of the checkpoint's ``generation_config.json``
    even with ``do_sample=False``: a ``repetition_penalty`` (Qwen2.5-Instruct sets 1.05), and,
    with ``min_new_tokens``, a ban on every end-of-sequence id, which changes the token whenever
    the argmax is one of them (Phi-3-mini ends "The capital of France is" after 16 tokens).
    """
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        generated = encoded["input_ids"]
        step = model(input_ids=generated, use_cache=True)
        for _ in range(NEW_TOKENS):
            token = step.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, token], dim=1)
            step = model(input_ids=token, past_key_values=step.past_key_values, use_cache=True)
        # One teacher-forced pass over the prompt plus the generated tokens. Logits at position
        # j predict token j + 1, so the rows that predict the generated tokens start at the last
        # prompt position and stop before the final row: one row per generated token.
        logits = model(input_ids=generated, use_cache=False).logits[0]
    top = logits[prompt_length - 1 : -1].topk(2, dim=-1).values
    return Reference(
        tokens=tuple(int(token) for token in generated[0, prompt_length:].tolist()),
        top1=tuple(float(value) for value in top[:, 0].tolist()),
        second=tuple(float(value) for value in top[:, 1].tolist()),
    )


@pytest.fixture(scope="module")
def llama_dir() -> Path:
    if not torch.cuda.is_available():
        _unavailable("no CUDA device is usable by torch")
    local = os.environ.get(LOCAL_PATH_VARIABLE)
    if local:
        path = Path(local).expanduser()
        if not (path / "config.json").is_file():
            _unavailable(f"{LOCAL_PATH_VARIABLE}={local!r} holds no config.json")
        return path
    hub = _import("huggingface_hub", "install it with: uv sync --extra hf")
    try:
        return Path(hub.snapshot_download(REPO_ID, allow_patterns=_FILES))
    except Exception as exc:  # the Hub client raises many types for "no network" and the like
        _unavailable(
            f"cannot get {REPO_ID} from the Hugging Face Hub ({type(exc).__name__}: {exc}); "
            f"set {LOCAL_PATH_VARIABLE} to a local copy to run without the network"
        )


@pytest.fixture(scope="module")
def expected(llama_dir: Path) -> dict[str, Reference]:
    """``transformers``' greedy tokens and top two logits per prompt, made once on the GPU.

    Float32 is the reference for every configuration, fp32 included: it is the most precise
    source available, therefore never less precise than the code under test, and a token that
    matches it is right. A reference in the configuration's own dtype can instead pick a
    different token where its rounding flips the top two logits (issue #39).
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(llama_dir), local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(llama_dir),
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    model = model.to(_reference_device()).eval()
    found = {prompt: _make_reference(tokenizer, model, prompt) for prompt in PROMPTS}
    del model
    torch.cuda.empty_cache()
    return found


@pytest.fixture(autouse=True)
def _profiles_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AYAKA_CACHE_DIR", str(tmp_path / "cache"))


def _context_bytes(error: CapacityError, key: str) -> int:
    """One byte count from the error's context, asserted to be an int and not a bool."""
    value = error.context.get(key)
    assert isinstance(value, int), (key, value)
    assert not isinstance(value, bool), (key, value)
    return value


def _check_shortfall(need: int, have: int, short: int) -> None:
    """Assert that a capacity error's three byte counts agree with each other."""
    assert short == need - have, (need, have, short)
    assert need > have, (need, have, short)


def _open_llm(config: EngineConfig, *, chosen: bool) -> LLM:
    """Build the engine, skipping a ``chosen`` checkpoint that cannot fit the card.

    Only the weights and fixed-memory shortfall of a checkpoint named by
    ``AYAKA_LLAMA_PATH`` skips: the default 135M always fits, and every other
    failure, capacity or not, is a defect and re-raises.
    """
    try:
        return LLM.from_config(config)
    except CapacityError as exc:
        if not chosen or exc.context.get("what") != _WEIGHTS_WHAT:
            raise
        need = _context_bytes(exc, "need")
        have = _context_bytes(exc, "have")
        short = _context_bytes(exc, "short")
        _check_shortfall(need, have, short)
        pytest.skip(f"{LOCAL_PATH_VARIABLE} chose a checkpoint that does not fit: {exc.message}")


def _reserve_tokens(directory: Path) -> int:
    """2,048 tokens of KV per request, or the model's whole context when it is shorter.

    Phi-3-mini-4k is served up to its 2,047-token sliding window (``ModelConfig.context_length``).
    """
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    return min(2048, ModelConfig.from_hf_config(raw).context_length)


def _reference_device() -> str:
    device = os.environ.get(REFERENCE_DEVICE_VARIABLE, "cuda")
    if device not in ("cuda", "cpu"):
        _unavailable(f"{REFERENCE_DEVICE_VARIABLE}={device!r} is neither 'cuda' nor 'cpu'")
    return device


def _eps(dtype: DType) -> float:
    """The machine epsilon of the configuration's dtype."""
    return float(torch.finfo(getattr(torch, _TORCH_DTYPES[dtype])).eps)


def _assert_greedy_match(
    prompt: str,
    ours: list[int],
    reference: Reference,
    dtype: DType,
) -> None:
    """Compare one prompt, accepting its first difference only at a reference near tie.

    The first difference is the only one that can be excused: once the prefixes differ,
    every later token is generated from a different context, so the remaining ones are
    left unchecked and counted in both the record and the warning.
    """
    gold = reference.tokens
    if len(ours) != len(gold):
        pytest.fail(
            f"{prompt!r}: ayaka made {len(ours)} tokens, the reference made {len(gold)}",
            pytrace=False,
        )
    for index, (mine, theirs) in enumerate(zip(ours, gold, strict=True)):
        if mine == theirs:
            continue
        top1, second = reference.top1[index], reference.second[index]
        margin = top1 - second
        bound = _TIE_FACTOR * _eps(dtype) * abs(top1)
        if margin > bound:
            pytest.fail(
                f"{prompt!r}: token {index}: ayaka {mine}, transformers {theirs}; the "
                f"reference's best two logits are {top1:.6g} and {second:.6g}, a margin of "
                f"{margin:.6g} beyond the {bound:.6g} that {dtype.label} allows",
                pytrace=False,
            )
        unchecked = len(ours) - index - 1
        text = (
            f"{prompt!r}@{index}: margin={margin:.6g} bound={bound:.6g} "
            f"ours={mine} gold={theirs}, {unchecked} later tokens unchecked"
        )
        with pytest.warns(NearTieWarning):
            warnings.warn(text, NearTieWarning, stacklevel=2)
        return


@pytest.mark.parametrize(("backends", "dtype", "graphs"), CONFIGS)
def test_greedy_tokens_match_transformers_on_the_gpu(
    llama_dir: Path,
    expected: dict[str, Reference],
    backends: tuple[str, ...],
    dtype: DType,
    graphs: bool,
) -> None:
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer", reason="the flashinfer extra is not installed")
    config = EngineConfig(
        source=ModelSourceConfig(str(llama_dir), offline=True),
        device=DeviceConfig(device="cuda", dtype=dtype, backends=backends, cuda_graph=graphs),
        cache=CacheConfig(block_size=16, kv_reserve_tokens=_reserve_tokens(llama_dir)),
        scheduler=SchedulerConfig(max_running=4, max_tokens_per_step=256),
    )
    # transformers' reference uses min_new_tokens; ignore_eos gives the same tokens while EOS is
    # not the best token, and keeps the decode steps in CUDA graphs (a min_tokens mask runs
    # eagerly).
    sampling = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)
    chosen = bool(os.environ.get(LOCAL_PATH_VARIABLE))

    with _open_llm(config, chosen=chosen) as llm:
        outs = llm.generate(list(PROMPTS), sampling)
        startup = llm.engine.startup
        assert startup is not None
        assert startup.reconciliation.matches, startup.reconciliation
        if graphs:
            assert llm.engine.runner.graph is not None
            assert llm.engine.runner.graph.replays > 0

    assert isinstance(outs, list)
    for prompt, out in zip(PROMPTS, outs, strict=True):
        assert out.error is None
        _assert_greedy_match(prompt, list(out.outputs[0].token_ids), expected[prompt], dtype)
        assert out.outputs[0].text.strip()


@pytest.fixture(scope="module")
def long_expected(llama_dir: Path) -> Reference:
    """``transformers``' float32 greedy reference for :data:`LONG_PROMPT`."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(llama_dir), local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(llama_dir),
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    model = model.to(_reference_device()).eval()
    found = _make_reference(tokenizer, model, LONG_PROMPT)
    del model
    torch.cuda.empty_cache()
    return found


@pytest.mark.parametrize(("backends", "dtype", "graphs"), CONFIGS)
def test_a_warm_prefix_cache_gives_the_reference_tokens_on_the_gpu(
    llama_dir: Path,
    long_expected: Reference,
    backends: tuple[str, ...],
    dtype: DType,
    graphs: bool,
) -> None:
    """The second run of a long prompt attaches its cached blocks and still matches float32."""
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer", reason="the flashinfer extra is not installed")
    config = EngineConfig(
        source=ModelSourceConfig(str(llama_dir), offline=True),
        device=DeviceConfig(device="cuda", dtype=dtype, backends=backends, cuda_graph=graphs),
        cache=CacheConfig(block_size=16, kv_reserve_tokens=_reserve_tokens(llama_dir)),
        scheduler=SchedulerConfig(max_running=4, max_tokens_per_step=256),
    )
    # transformers' reference uses min_new_tokens; ignore_eos gives the same tokens while EOS is
    # not the best token, and keeps the decode steps in CUDA graphs (a min_tokens mask runs
    # eagerly).
    sampling = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)
    chosen = bool(os.environ.get(LOCAL_PATH_VARIABLE))

    with _open_llm(config, chosen=chosen) as llm:
        cold = llm.generate(LONG_PROMPT, sampling)
        warm = llm.generate(LONG_PROMPT, sampling)

    assert not isinstance(cold, list)
    assert not isinstance(warm, list)
    assert cold.usage is not None
    assert warm.usage is not None
    assert cold.usage.cached_tokens == 0
    assert warm.usage.cached_tokens >= 16
    for out in (cold, warm):
        assert out.error is None
        _assert_greedy_match(LONG_PROMPT, list(out.outputs[0].token_ids), long_expected, dtype)
