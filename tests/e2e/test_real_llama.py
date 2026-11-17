"""A real Llama checkpoint on the GPU against ``transformers``: the same greedy tokens.

The checkpoint is ``HuggingFaceTB/SmolLM2-135M`` (``LlamaForCausalLM``, 135M parameters, grouped
heads, tied embeddings), run through the public ``LLM`` with its real tokenizer, in every M1
configuration: the ``torch`` backend and ``flashinfer`` attention, eager and with decode graphs,
fp32, fp16 and bf16. The reference is ``transformers``' ``generate`` on the same GPU in the same
dtype, greedy, with eager attention. Every generated token must match.

Measured on the dev machine (RTX 3050 Laptop, WSL2, transformers 5.22.2, flashinfer 0.7.4): all
six configurations matched ``transformers`` on all 32 tokens of the three prompts below.

What it needs, and what it does when something is missing (as ``test_gpt2_parity.py``):

* a CUDA GPU, torch with CUDA, ``transformers`` and ``huggingface_hub`` (``--extra hf``), and
  ``flashinfer`` for its configurations (``--extra flashinfer``);
* the checkpoint: a local directory named by ``AYAKA_LLAMA_PATH``, or else a download of about
  270 MB from the Hugging Face Hub into its usual cache.

A missing piece skips with the reason; ``AYAKA_REQUIRE_REAL_MODEL=1`` turns the skip into a failure.
"""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any, Final, NoReturn

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

REPO_ID: Final = "HuggingFaceTB/SmolLM2-135M"
LOCAL_PATH_VARIABLE: Final = "AYAKA_LLAMA_PATH"
REQUIRE_VARIABLE: Final = "AYAKA_REQUIRE_REAL_MODEL"
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
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.configs.scheduler import SchedulerConfig  # noqa: E402
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
def expected(llama_dir: Path) -> dict[tuple[DType, str], list[int]]:
    """``transformers``' greedy tokens per dtype and prompt, made once on the GPU."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(llama_dir), local_files_only=True)
    found: dict[tuple[DType, str], list[int]] = {}
    for dtype, name in _TORCH_DTYPES.items():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(llama_dir),
            dtype=getattr(torch, name),
            attn_implementation="eager",
            local_files_only=True,
        )
        model = model.cuda().eval()
        for prompt in PROMPTS:
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_new_tokens=NEW_TOKENS,
                    min_new_tokens=NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                )
            found[dtype, prompt] = generated[0, encoded["input_ids"].shape[1] :].tolist()
        del model
        torch.cuda.empty_cache()
    return found


@pytest.fixture(autouse=True)
def _profiles_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AYAKA_CACHE_DIR", str(tmp_path / "cache"))


@pytest.mark.parametrize(("backends", "dtype", "graphs"), CONFIGS)
def test_greedy_tokens_match_transformers_on_the_gpu(
    llama_dir: Path,
    expected: dict[tuple[DType, str], list[int]],
    backends: tuple[str, ...],
    dtype: DType,
    graphs: bool,
) -> None:
    if "flashinfer" in backends:
        pytest.importorskip("flashinfer", reason="the flashinfer extra is not installed")
    config = EngineConfig(
        source=ModelSourceConfig(str(llama_dir), offline=True),
        device=DeviceConfig(device="cuda", dtype=dtype, backends=backends, cuda_graph=graphs),
        cache=CacheConfig(block_size=16, kv_reserve_tokens=2048),
        scheduler=SchedulerConfig(max_running=4, max_tokens_per_step=256),
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, min_tokens=NEW_TOKENS)

    with LLM.from_config(config) as llm:
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
        ours = list(out.outputs[0].token_ids)
        theirs = expected[dtype, prompt]
        assert ours == theirs, f"{prompt!r}: first difference at token " + str(
            next(i for i, (a, b) in enumerate(zip(ours, theirs, strict=True)) if a != b)
        )
        assert out.outputs[0].text.strip()
