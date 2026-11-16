"""Real GPT-2 against ``transformers``: the same greedy tokens, one by one.

The checkpoint is ``openai-community/gpt2`` (124M parameters), run on the CPU in float32 through
the public ``LLM`` with the real tokenizer of the ``hf`` adapter. The reference is
``transformers``' own ``generate`` on the same files with greedy decoding and eager attention.
For each prompt the test compares, in this order: the prompt token ids, the logits of the whole
prompt, and every generated token id.

What it needs, and what it does when something is missing:

* torch, ``transformers`` and ``huggingface_hub`` (``uv sync --extra cpu --extra hf``);
* the checkpoint: a local directory named by ``AYAKA_GPT2_PATH`` (``config.json``,
  ``model.safetensors`` and the tokenizer files), or else a download from the Hugging Face Hub
  into its usual cache, which needs the network the first time.

A missing piece skips the test with the reason in the report, so a machine without them is not
red. Setting ``AYAKA_REQUIRE_REAL_MODEL=1`` turns every such skip into a failure: the CI job that
exists to run this test sets it, so a download that fails cannot pass as a skip.
"""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NoReturn

import pytest

pytestmark = pytest.mark.integration

REPO_ID: Final = "openai-community/gpt2"
LOCAL_PATH_VARIABLE: Final = "AYAKA_GPT2_PATH"
REQUIRE_VARIABLE: Final = "AYAKA_REQUIRE_REAL_MODEL"
#: Only what the test reads: the weights, the config and the tokenizer files.
_FILES: Final = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

NEW_TOKENS: Final = 32
PROMPTS: Final = (
    "The capital of France is",
    "Once upon a time, in a land far away,",
    "def fibonacci(n):\n    ",
    "1, 2, 3, 4, 5, 6, 7,",
    (
        "Large language models are trained on text from many sources. When one is asked a "
        "question, it reads it as a sequence of tokens and then writes the answer one token at a "
        "time, each token chosen from a probability distribution over the vocabulary. The"
    ),
)


def _unavailable(reason: str) -> NoReturn:
    """Skip, with the reason, or fail when the run is required to have the real model."""
    if os.environ.get(REQUIRE_VARIABLE) == "1":
        pytest.fail(
            f"{REQUIRE_VARIABLE}=1 but the real GPT-2 test cannot run: {reason}", pytrace=False
        )
    pytest.skip(reason, allow_module_level=True)


def _import(name: str, hint: str) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        _unavailable(f"cannot import {name} ({exc}); {hint}")


torch: Any = _import("torch", "install it with: uv sync --extra cpu")
transformers: Any = _import("transformers", "install it with: uv sync --extra hf")

from ayaka.configs.cache import CacheConfig  # noqa: E402
from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.engine import EngineConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.configs.scheduler import SchedulerConfig  # noqa: E402
from ayaka.core.params import SamplingParams  # noqa: E402
from ayaka.core.types import FinishReason  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.model_loader.loader import load_model  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402
from gpt2_support import run_paged  # noqa: E402  # needs the imports above

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module")
def gpt2_dir() -> Path:
    """A directory with the GPT-2 checkpoint and its tokenizer files."""
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
def reference(gpt2_dir: Path) -> Any:
    """``transformers``' GPT-2 on the same files, in float32 with eager attention."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(gpt2_dir), dtype=torch.float32, attn_implementation="eager", local_files_only=True
    )
    return model.eval()


@pytest.fixture(scope="module")
def hf_tokenizer(gpt2_dir: Path) -> Any:
    return transformers.AutoTokenizer.from_pretrained(str(gpt2_dir), local_files_only=True)


def _config(gpt2_dir: Path, *, max_running: int, step: int) -> EngineConfig:
    return EngineConfig(
        source=ModelSourceConfig(str(gpt2_dir), offline=True),
        device=DeviceConfig(device="cpu", dtype=DType.FP32),
        cache=CacheConfig(block_size=16),
        scheduler=SchedulerConfig(max_running=max_running, max_tokens_per_step=step),
    )


@pytest.fixture(scope="module")
def llm(gpt2_dir: Path) -> Iterator[LLM]:
    with LLM.from_config(_config(gpt2_dir, max_running=1, step=2048)) as engine:
        yield engine


def _greedy() -> SamplingParams:
    # min_tokens keeps both sides from stopping at the end-of-text token, so each run makes
    # exactly NEW_TOKENS tokens and the comparison is over the same number everywhere.
    return SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, min_tokens=NEW_TOKENS)


def _expected(reference: Any, hf_tokenizer: Any, prompt: str) -> tuple[list[int], list[int]]:
    """The prompt ids and the ``NEW_TOKENS`` ids ``transformers`` generates greedily after it."""
    encoded = hf_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated = reference.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=NEW_TOKENS,
            min_new_tokens=NEW_TOKENS,
            pad_token_id=hf_tokenizer.eos_token_id,
        )
    prompt_ids = encoded["input_ids"][0].tolist()
    return prompt_ids, generated[0, len(prompt_ids) :].tolist()


@pytest.fixture(scope="module")
def expected(reference: Any, hf_tokenizer: Any) -> dict[str, tuple[list[int], list[int]]]:
    return {prompt: _expected(reference, hf_tokenizer, prompt) for prompt in PROMPTS}


def test_the_reference_produced_the_number_of_tokens_asked_for(
    expected: dict[str, tuple[list[int], list[int]]],
) -> None:
    assert [len(new) for _, new in expected.values()] == [NEW_TOKENS] * len(PROMPTS)


@pytest.mark.parametrize("prompt", PROMPTS, ids=lambda p: p[:20])
def test_greedy_tokens_match_transformers_one_prompt_at_a_time(
    llm: LLM,
    hf_tokenizer: Any,
    expected: dict[str, tuple[list[int], list[int]]],
    prompt: str,
) -> None:
    prompt_ids, tokens = expected[prompt]

    out = llm.generate(prompt, _greedy())

    assert not isinstance(out, list)
    assert out.error is None
    assert out.prompt_token_ids == tuple(prompt_ids)  # the same tokenization
    [completion] = out.outputs
    assert completion.finish_reason is FinishReason.LENGTH
    assert len(completion.token_ids) == len(tokens) == NEW_TOKENS
    for position, (ours, theirs) in enumerate(zip(completion.token_ids, tokens, strict=True)):
        assert ours == theirs, (
            f"token {position} of {prompt!r}: ayaka {ours}, transformers {theirs}"
        )
    assert completion.text == hf_tokenizer.decode(tokens, skip_special_tokens=True)


def test_greedy_tokens_match_transformers_in_one_batch_cut_into_small_steps(
    gpt2_dir: Path, expected: dict[str, tuple[list[int], list[int]]]
) -> None:
    config = _config(gpt2_dir, max_running=3, step=32)  # the long prompt takes several steps

    with LLM.from_config(config) as engine:
        outs = engine.generate(list(PROMPTS), _greedy())

        assert isinstance(outs, list)
        for prompt, out in zip(PROMPTS, outs, strict=True):
            prompt_ids, tokens = expected[prompt]
            assert out.error is None
            assert out.prompt_token_ids == tuple(prompt_ids)
            assert out.outputs[0].token_ids == tuple(tokens), prompt
        assert engine.engine.kv.stats().used_blocks == (0,)
        engine.engine.kv.check_integrity()


def test_logits_of_the_whole_prompt_match_transformers(
    gpt2_dir: Path, reference: Any, expected: dict[str, tuple[list[int], list[int]]]
) -> None:
    # Float32 on both sides, but the two compute attention and the layer norms in different
    # orders, so the logits agree to rounding, not bit for bit. The tolerance is far below the
    # gap between the best and second best token that the greedy test relies on.
    prompt_ids, _ = expected[PROMPTS[-1]]
    sequence = torch.tensor(prompt_ids)
    model = load_model(
        ModelSourceConfig(str(gpt2_dir), offline=True), DeviceConfig(device="cpu", dtype=DType.FP32)
    )

    [hidden] = run_paged(model, [sequence], prompt_lengths=[len(prompt_ids)], chunk=16)

    with torch.no_grad():
        theirs = reference(sequence[None]).logits[0]
    ours = model.compute_logits(hidden)
    torch.testing.assert_close(ours, theirs, rtol=1e-3, atol=1e-3)
    assert ours.argmax(dim=-1).tolist() == theirs.argmax(dim=-1).tolist()
