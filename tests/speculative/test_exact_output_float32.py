"""Speculation cannot change a token: plain and speculative decoding of a real model, in float32.

In bf16, greedy ids of plain decoding already change with the code path alone: on Qwen3-8B
(NVIDIA A100, 256 tokens), plain decoding of four prompts in one batch differs from the same
prompts one at a time on all four, at positions whose float32 margin between the two tokens is
0.025 to 0.073 nats. A verify step, which runs ``1 + k`` rows through the same matrices, moves
the same near ties. Those differences come from the kernels' reduction order, not from the
acceptance rule, so the rule is checked where the kernels cannot hide it: in float32 every arm
below must emit the ids of plain decoding of one request, at every position.

The arms cover window sizes from 1 to 15, the cost controller, a ragged batch of four prompt
classes of different lengths, and seeded sampling at temperature 0.8, where the counter-based
draws keyed by position make the one-point rule exact as well.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

import pytest

torch: Any = pytest.importorskip("torch")

from ayaka.configs.cache import CacheConfig  # noqa: E402
from ayaka.configs.device import DeviceConfig  # noqa: E402
from ayaka.configs.engine import EngineConfig  # noqa: E402
from ayaka.configs.model_source import ModelSourceConfig  # noqa: E402
from ayaka.configs.scheduler import SchedulerConfig  # noqa: E402
from ayaka.configs.speculative import SpeculativeConfig, SpeculativeController  # noqa: E402
from ayaka.configs.tokenizer import TokenizerConfig  # noqa: E402
from ayaka.core.params import SamplingParams  # noqa: E402
from ayaka.engine.llm import LLM  # noqa: E402
from ayaka.model_loader.source import resolve_source  # noqa: E402
from ayaka.speculative.policy import CostAwareController, FixedController  # noqa: E402
from ayaka.utils.dtypes import DType  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.gpu

_MODEL = "Qwen/Qwen3-1.7B"
_TOKENS = 128

# One prompt per class the speculative benchmarks use; their lengths differ, so a batch of all
# four is ragged.
_PROMPTS = {
    "code_edit": "Add type annotations to this function and keep the logic identical:\n```python\n"
    "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
    "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n"
    "            out.append(b[j]); j += 1\n    out.extend(a[i:]); out.extend(b[j:])\n"
    "    return out\n```\n",
    "prose": "Write a short story about a lighthouse keeper who finds a glass compass.",
    "json_tools": 'Return a JSON object with keys "name", "args" and "reason" for a call to a '
    'tool named read_file that reads "src/main.py" lines 10 to 40.',
    "random": "qz vex plinth 83 orbweaver lacquer +=? manifold kestrel 0x7f umbra drizzle tenet",
}


def _lookup(k: int, controller: SpeculativeController = "fixed") -> SpeculativeConfig:
    return SpeculativeConfig(
        enabled=True, method="lookup", controller=controller, fixed_k=k, max_draft_tokens=k
    )


def _generate(
    snapshot: Path,
    prompts: list[tuple[int, ...]],
    params: SamplingParams,
    speculative: SpeculativeConfig,
    *,
    batched: bool,
) -> tuple[list[tuple[int, ...]], int]:
    """The ids of every prompt, and how many drafted tokens were accepted."""
    config = EngineConfig(
        source=ModelSourceConfig(_MODEL, offline=True),
        tokenizer=TokenizerConfig(tokenizer=str(snapshot), offline=True, skip_tokenizer_init=True),
        device=DeviceConfig(device="cuda", dtype=DType.FP32, backends=("torch",), cuda_graph=True),
        cache=CacheConfig(block_size=16, num_blocks=1024, kv_reserve_tokens=2048),
        scheduler=SchedulerConfig(max_running=len(prompts), max_tokens_per_step=512),
        speculative=speculative,
    )
    with LLM.from_config(config) as llm:
        if batched:
            outputs = llm.generate(prompts, params)
            assert isinstance(outputs, list)
        else:
            outputs = [llm.generate(prompt, params) for prompt in prompts]
        tokens = [output.outputs[0].token_ids for output in outputs]  # type: ignore[union-attr]
        controller = llm.engine.runner.controller
        accepted = (
            int(controller.snapshot()["total_accepted"])
            if isinstance(controller, (CostAwareController, FixedController))
            else 0
        )
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return tokens, accepted


@pytest.fixture(scope="module")
def prompts() -> tuple[Path, list[tuple[int, ...]]]:
    tokenizers: Any = pytest.importorskip("tokenizers")  # tests/tokenizers shadows it for pyright
    snapshot = resolve_source(ModelSourceConfig(_MODEL, offline=True))
    tokenizer = tokenizers.Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    return snapshot, [tuple(tokenizer.encode(text).ids) for text in _PROMPTS.values()]


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(temperature=0.0, max_tokens=_TOKENS, ignore_eos=True),
        SamplingParams(temperature=0.8, seed=1234, max_tokens=_TOKENS, ignore_eos=True),
    ],
    ids=["greedy", "seeded-temperature-0.8"],
)
def test_speculation_emits_the_ids_of_plain_decoding_in_float32(
    prompts: tuple[Path, list[tuple[int, ...]]], params: SamplingParams
) -> None:
    snapshot, ids = prompts
    reference, _ = _generate(snapshot, ids, params, SpeculativeConfig(), batched=False)
    arms: dict[str, tuple[SpeculativeConfig, bool]] = {
        "plain, ragged batch": (SpeculativeConfig(), True),
        "lookup k=1": (_lookup(1), False),
        "lookup k=4": (_lookup(4), False),
        "lookup k=15": (_lookup(15), False),
        "lookup k=4, ragged batch": (_lookup(4), True),
        "lookup, cost controller, ragged batch": (_lookup(8, controller="cost"), True),
    }
    differing: list[str] = []
    for arm, (speculative, batched) in arms.items():
        tokens, accepted = _generate(snapshot, ids, params, speculative, batched=batched)
        if speculative.enabled and accepted == 0:
            differing.append(f"{arm}: no draft was accepted, so the arm checks nothing")
        for name, got, want in zip(_PROMPTS, tokens, reference, strict=True):
            if got != want:
                first = next(i for i, (a, b) in enumerate(zip(got, want, strict=False)) if a != b)
                differing.append(f"{arm} / {name}: first difference at token {first}")

    assert not differing, "\n".join(differing)
