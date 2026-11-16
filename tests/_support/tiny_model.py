"""A tiny GPT-2 the whole engine can run on a CPU in tests, and the pieces around it.

This is the shared fixture of the execution, engine and command-line tests, and of the end-to-end
tests after them. It gives:

* ``tiny_gpt2_config`` and ``tiny_gpt2_state``: a 2-layer GPT-2 with random weights, in the tensor
  names and shapes of a Hugging Face checkpoint. The weights are shaped so that a greedy run only
  ever picks printable ASCII (the token rows outside space to tilde are scaled down), which makes
  the generated text readable and lets a test choose a stop string out of it.
* ``write_tiny_gpt2``: the model as a checkpoint directory (``config.json`` and
  ``model.safetensors``), the thing ``load_model`` and the command line read.
* ``tiny_tokenizer``: a byte-level tokenizer (ids 0-255 are bytes, 256-259 are special) as a
  ``LoadedTokenizer``. ``transformers`` and ``tiktoken`` are optional extras, so no tokenizer files
  are written; ``build_engine`` takes the loaded tokenizer directly, and ``patch_tokenizer_loading``
  makes the default factory return it for code that builds its own engine, such as the command line.
* ``build_tiny_stack``: a model, cache, scheduler and runner wired together without the engine,
  for tests of the execution layer.
* ``greedy_reference``: the tokens plain full-sequence GPT-2 formulas pick greedily, with no cache.
* ``prompt_ids`` and ``assert_cache_released``: the ids the engine makes of a text prompt, and the
  check that an engine holds no request and no cache block once its work is done.

Import this module only from a test that has already done ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.execution.runner import ModelRunner
from ayaka.kvcache.manager import KVCacheManager
from ayaka.kvcache.storage.pool import KVPool
from ayaka.models.gpt2 import GPT2LMHeadModel
from ayaka.ops.backend import load_backend
from ayaka.scheduler.scheduler import Scheduler
from ayaka.tokenizers.factory import DefaultTokenizerFactory
from ayaka.utils.dtypes import DType
from gpt2_support import hf_state, load_hf_state, reference_hidden, tiny_config, write_checkpoint
from tokenizer_fakes import BOS, MODEL_VOCAB, FakeTokenizer, loaded_for

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pytest

    from ayaka.configs.model import ModelConfig
    from ayaka.tokenizers.contracts import LoadedTokenizer

__all__ = [
    "MAX_POSITIONS",
    "TINY_BLOCK_SIZE",
    "TinyCheckpoint",
    "TinyStack",
    "assert_cache_released",
    "build_tiny_stack",
    "greedy_reference",
    "patch_tokenizer_loading",
    "prompt_ids",
    "tiny_engine_config",
    "tiny_gpt2_config",
    "tiny_gpt2_state",
    "tiny_tokenizer",
    "write_tiny_gpt2",
]

#: Positions of the tiny model, and so its context length.
MAX_POSITIONS = 128
#: Tokens per cache block in the fixture; small, so a short prompt spans several blocks.
TINY_BLOCK_SIZE = 4
_PRINTABLE = range(32, 127)


def tiny_gpt2_config(**overrides: object) -> ModelConfig:
    """The architecture of the fixture; its vocabulary is the byte tokenizer's (300 ids)."""
    return tiny_config(vocab_size=MODEL_VOCAB, max_position_embeddings=MAX_POSITIONS, **overrides)


def tiny_gpt2_state(config: ModelConfig, *, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random float32 weights named like a Hugging Face ``GPT2LMHeadModel``.

    Token rows outside printable ASCII are scaled down to a hundredth, so their logits stay near
    zero while the largest printable logit is far above it: greedy decoding never leaves ASCII.
    """
    state = hf_state(config, seed=seed)
    table = state["transformer.wte.weight"]
    quiet = torch.tensor([i not in _PRINTABLE for i in range(config.vocab_size)])
    table[quiet] *= 0.01
    return state


@dataclass(frozen=True, slots=True)
class TinyCheckpoint:
    """A tiny model on disk and the weights it was written from.

    Attributes:
        directory: Holds ``config.json`` and ``model.safetensors``.
        config: The architecture.
        state: The tensors written, under their checkpoint names.
    """

    directory: Path
    config: ModelConfig
    state: dict[str, torch.Tensor]


def write_tiny_gpt2(directory: Path, *, seed: int = 0) -> TinyCheckpoint:
    """Write the fixture model as a checkpoint directory and return what was written."""
    config = tiny_gpt2_config()
    state = tiny_gpt2_state(config, seed=seed)
    write_checkpoint(directory, state, config)
    return TinyCheckpoint(directory=directory, config=config, state=state)


def tiny_tokenizer() -> LoadedTokenizer:
    """A byte-level tokenizer bundle: ids 0-255 are bytes, 256 ``<s>``, 257 ``</s>`` and so on."""
    return loaded_for(FakeTokenizer())


def patch_tokenizer_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``DefaultTokenizerFactory.from_config`` return :func:`tiny_tokenizer`, for this test."""
    monkeypatch.setattr(
        DefaultTokenizerFactory, "from_config", classmethod(lambda cls, config: tiny_tokenizer())
    )


def tiny_engine_config(
    directory: Path,
    *,
    block_size: int = TINY_BLOCK_SIZE,
    num_blocks: int | None = 129,
    max_running: int | None = 4,
    max_tokens_per_step: int = 16,
    **scheduler: object,
) -> EngineConfig:
    """An engine configuration for the fixture on a CPU in float32, with a small fixed cache."""
    return EngineConfig(
        source=ModelSourceConfig(str(directory), offline=True),
        device=DeviceConfig(device="cpu", dtype=DType.FP32),
        cache=CacheConfig(block_size=block_size, num_blocks=num_blocks),
        scheduler=SchedulerConfig(
            max_running=max_running,
            max_tokens_per_step=max_tokens_per_step,
            **scheduler,  # type: ignore[arg-type]  # forwarded options of SchedulerConfig, by design
        ),
    )


@dataclass(frozen=True, slots=True)
class TinyStack:
    """The model, its cache, the scheduler over it and the runner, without an engine."""

    model: GPT2LMHeadModel
    kv: KVCacheManager
    pool: KVPool
    scheduler: Scheduler
    runner: ModelRunner
    state: dict[str, torch.Tensor]
    config: ModelConfig


def build_tiny_stack(
    *,
    num_blocks: int = 129,
    max_running: int = 4,
    max_tokens_per_step: int = 16,
    block_size: int = TINY_BLOCK_SIZE,
    seed: int = 0,
    eos_token_ids: Sequence[int] = (),
) -> TinyStack:
    """Wire the fixture model to a cache, a scheduler and a runner, as the engine builder does."""
    config = tiny_gpt2_config()
    state = tiny_gpt2_state(config, seed=seed)
    ops = load_backend(["torch"], device="cpu", dtype=DType.FP32)
    model = GPT2LMHeadModel(config, ops=ops, dtype=DType.FP32)
    load_hf_state(model, state)
    model.eval()
    [group] = model.kv_specs(block_size=block_size, layout=ops.kvcache.layout())
    kv = KVCacheManager(
        [group],
        [num_blocks],
        max_requests=max_running,
        max_blocks_per_request=-(-MAX_POSITIONS // block_size),
    )
    pool = KVPool(group, num_blocks, device="cpu")
    scheduler = Scheduler(
        SchedulerConfig(max_running=max_running, max_tokens_per_step=max_tokens_per_step),
        kv,
        max_model_len=MAX_POSITIONS,
        eos_token_ids=eos_token_ids,
    )
    runner = ModelRunner(
        model,
        [pool],
        kv.table,
        max_rows=ModelRunner.required_rows(
            token_budget=scheduler.token_budget,
            max_running=max_running,
            lookahead=3,
        ),
        device="cpu",
    )
    return TinyStack(model, kv, pool, scheduler, runner, state, config)


def greedy_reference(
    state: dict[str, torch.Tensor], config: ModelConfig, prompt: Sequence[int], count: int
) -> list[int]:
    """The ``count`` tokens GPT-2's formulas pick greedily after ``prompt``, with no KV cache.

    Every token reruns the whole sequence through ``reference_hidden``, the formulas written out
    in plain float32 torch, so nothing here shares a code path with the engine.
    """
    tokens = list(prompt)
    table = state["transformer.wte.weight"]
    for _ in range(count):
        hidden = reference_hidden(state, config, torch.tensor(tokens))
        tokens.append(int((hidden[-1] @ table.T).argmax()))
    return tokens[len(prompt) :]


def prompt_ids(text: str) -> list[int]:
    """The ids the engine makes of ``text`` with the fixture tokenizer: its BOS, then the bytes."""
    return [BOS, *text.encode()]


def assert_cache_released(engine: Any) -> None:
    """Check that ``engine`` is idle and its KV pool is entirely free, with sound bookkeeping."""
    assert not engine.has_unfinished
    assert engine.scheduler.num_waiting == 0
    stats = engine.kv.stats()
    assert stats.num_requests == 0
    assert stats.used_blocks == (0,) * len(stats.used_blocks)
    assert stats.free_blocks == stats.usable_blocks
    engine.kv.check_integrity()
