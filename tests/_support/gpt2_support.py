"""A tiny GPT-2, its Hugging Face checkpoint layout and a paged runner, for model and loader tests.

Nothing here uses ``transformers``. ``hf_state`` builds random tensors under the names and
shapes of a Hugging Face ``GPT2LMHeadModel`` state dict (``Conv1D`` weights are ``[in, out]``),
``reference_hidden`` computes what GPT-2 does with them from the published formulas in plain
float32 torch, and ``run_paged`` feeds a model its prompt and then one token at a time through
a paged KV cache whose blocks are handed out in scrambled order and start out as NaN.

Import this module only from a test that has already done ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from ayaka.configs.model import ModelConfig
from ayaka.core.types import ForwardMode
from ayaka.kvcache.storage.pool import KVPool
from ayaka.layers.context import ForwardContext
from safetensors_files import write_safetensors

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ayaka.models.base import CausalLM
    from safetensors_files import TensorData

__all__ = [
    "BLOCK_SIZE",
    "hf_config_dict",
    "hf_state",
    "load_hf_state",
    "reference_hidden",
    "run_paged",
    "tensor_data",
    "tiny_config",
    "write_checkpoint",
]

BLOCK_SIZE = 4


def tiny_config(**overrides: object) -> ModelConfig:
    """A 2-layer, 4-head GPT-2 with a 32-wide residual stream and 64 positions."""
    fields: dict[str, object] = {
        "architecture": "GPT2LMHeadModel",
        "vocab_size": 50,
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 8,
        "intermediate_size": 128,
        "max_position_embeddings": 64,
        "hidden_act": "gelu_new",
        "norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }
    fields.update(overrides)
    return ModelConfig(**fields)  # type: ignore[arg-type]  # a dict of the fields, by design


def hf_state(
    config: ModelConfig,
    *,
    seed: int = 0,
    prefix: str = "transformer.",
    with_masks: bool = False,
    with_head: bool = False,
) -> dict[str, torch.Tensor]:
    """Random float32 tensors named and shaped like a Hugging Face GPT-2 state dict.

    ``prefix`` is ``"transformer."`` for ``GPT2LMHeadModel`` and ``""`` for the bare
    ``GPT2Model``. ``with_masks`` adds the per-layer causal-mask buffers old checkpoints
    store; ``with_head`` adds ``lm_head.weight`` (a copy of ``wte``, as a tied checkpoint has).
    """
    generator = torch.Generator().manual_seed(seed)
    hidden, inner = config.hidden_size, config.intermediate_size

    def rand(*shape: int, scale: float = 0.3) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * scale

    def norm_weight() -> torch.Tensor:
        return 1.0 + rand(hidden, scale=0.1)

    state: dict[str, torch.Tensor] = {
        f"{prefix}wte.weight": rand(config.vocab_size, hidden),
        f"{prefix}wpe.weight": rand(config.max_position_embeddings, hidden),
        f"{prefix}ln_f.weight": norm_weight(),
        f"{prefix}ln_f.bias": rand(hidden, scale=0.1),
    }
    for layer in range(config.num_layers):
        block = f"{prefix}h.{layer}."
        state |= {
            f"{block}ln_1.weight": norm_weight(),
            f"{block}ln_1.bias": rand(hidden, scale=0.1),
            f"{block}attn.c_attn.weight": rand(hidden, 3 * hidden),
            f"{block}attn.c_attn.bias": rand(3 * hidden, scale=0.1),
            f"{block}attn.c_proj.weight": rand(hidden, hidden),
            f"{block}attn.c_proj.bias": rand(hidden, scale=0.1),
            f"{block}ln_2.weight": norm_weight(),
            f"{block}ln_2.bias": rand(hidden, scale=0.1),
            f"{block}mlp.c_fc.weight": rand(hidden, inner),
            f"{block}mlp.c_fc.bias": rand(inner, scale=0.1),
            f"{block}mlp.c_proj.weight": rand(inner, hidden),
            f"{block}mlp.c_proj.bias": rand(hidden, scale=0.1),
        }
        if with_masks:
            state[f"{block}attn.bias"] = torch.tril(torch.ones(1, 1, 8, 8))
    if with_head:
        state["lm_head.weight"] = state[f"{prefix}wte.weight"].clone()
    return state


def load_hf_state(model: CausalLM, state: dict[str, torch.Tensor]) -> None:
    """Write a state dict into a model through its own mapping, the way the loader does."""
    mapping = type(model).weight_mapping(model.config)
    for name, tensor in state.items():
        target = mapping.target(name)
        if target is not None:
            model.load_weight(target, tensor)


def reference_hidden(
    state: dict[str, torch.Tensor], config: ModelConfig, token_ids: torch.Tensor
) -> torch.Tensor:
    """GPT-2's final hidden states for one whole sequence ``[T]``, in float32, from the formulas."""
    prefix = "transformer." if "transformer.wte.weight" in state else ""
    steps = token_ids.shape[0]
    heads, head_dim = config.num_attention_heads, config.head_dim

    def norm(x: torch.Tensor, name: str) -> torch.Tensor:
        return functional.layer_norm(
            x,
            (config.hidden_size,),
            state[f"{name}.weight"],
            state[f"{name}.bias"],
            config.norm_eps,
        )

    def conv1d(x: torch.Tensor, name: str) -> torch.Tensor:
        return x @ state[f"{name}.weight"] + state[f"{name}.bias"]

    x = state[f"{prefix}wte.weight"][token_ids] + state[f"{prefix}wpe.weight"][:steps]
    for layer in range(config.num_layers):
        block = f"{prefix}h.{layer}"
        q, k, v = conv1d(norm(x, f"{block}.ln_1"), f"{block}.attn.c_attn").split(
            config.hidden_size, dim=-1
        )
        q, k, v = (t.reshape(steps, heads, head_dim).transpose(0, 1) for t in (q, k, v))
        weights = (q @ k.transpose(-1, -2)) / head_dim**0.5
        weights = weights.masked_fill(
            torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1), float("-inf")
        )
        attended = (weights.softmax(dim=-1) @ v).transpose(0, 1).reshape(steps, -1)
        x = x + conv1d(attended, f"{block}.attn.c_proj")
        hidden = functional.gelu(
            conv1d(norm(x, f"{block}.ln_2"), f"{block}.mlp.c_fc"), approximate="tanh"
        )
        x = x + conv1d(hidden, f"{block}.mlp.c_proj")
    return norm(x, f"{prefix}ln_f")


def run_paged(
    model: CausalLM,
    sequences: Sequence[torch.Tensor],
    *,
    prompt_lengths: Sequence[int],
    chunk: int | None = None,
) -> list[torch.Tensor]:
    """Feed each sequence to ``model`` as a prefill of ``prompt_lengths[i]`` tokens, then decode.

    Step ``s`` carries the ``s``-th piece of every request that has one: the prompt in pieces of
    ``chunk`` rows (the whole prompt when ``chunk`` is ``None``), then one row per decoded token.
    Blocks come from a pool whose every slot starts as NaN and are assigned in reverse order, so
    a wrong block, a wrong slot or a missed store shows up as NaN. Returns, per sequence, the
    ``[T, hidden]`` hidden states of every position, in order.
    """
    ops = model.ops
    [group] = model.kv_specs(block_size=BLOCK_SIZE, layout=ops.kvcache.layout())
    blocks = [-(-len(sequence) // BLOCK_SIZE) for sequence in sequences]
    pool = KVPool(group, sum(blocks) + 1, device="cpu")
    buffers = {layer: pool.buffers(layer) for layer in group.layer_ids}
    for kv in buffers.values():
        kv.k.fill_(float("nan"))
        kv.v.fill_(float("nan"))
    free = list(range(sum(blocks), 0, -1))
    tables = [[free.pop(0) for _ in range(count)] for count in blocks]

    pieces: list[list[int]] = []
    for sequence, prompt in zip(sequences, prompt_lengths, strict=True):
        step = chunk or prompt
        pieces.append(
            [
                *(min(step, prompt - start) for start in range(0, prompt, step)),
                *([1] * (len(sequence) - prompt)),
            ]
        )
    done = [0] * len(sequences)
    outputs: list[list[torch.Tensor]] = [[] for _ in sequences]
    for step in range(max(len(p) for p in pieces)):
        active = [i for i, p in enumerate(pieces) if step < len(p)]
        rows = {i: pieces[i][step] for i in active}
        width = max(len(tables[i]) for i in active)
        meta = ops.attention.plan(
            ForwardMode.MIXED,
            torch.tensor([rows[i] for i in active], dtype=torch.int32),
            torch.tensor([done[i] + rows[i] for i in active], dtype=torch.int32),
            torch.tensor([[*tables[i], *([0] * width)][:width] for i in active], dtype=torch.int32),
            BLOCK_SIZE,
        )
        ids = torch.cat([sequences[i][done[i] : done[i] + rows[i]] for i in active])
        hidden = model(ids, ForwardContext(attention=meta, kv_buffers=buffers))
        start = 0
        for i in active:
            outputs[i].append(hidden[start : start + rows[i]])
            start += rows[i]
            done[i] += rows[i]
    return [torch.cat(parts) for parts in outputs]


_TOKENS = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int8: "I8",
}


def tensor_data(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> TensorData:
    """The safetensors form of ``tensor`` converted to ``dtype``: token, shape, raw bytes."""
    converted = tensor.to(dtype).contiguous()
    return (
        _TOKENS[dtype],
        tuple(converted.shape),
        converted.flatten().view(torch.uint8).numpy().tobytes(),
    )


def hf_config_dict(config: ModelConfig) -> dict[str, object]:
    """The ``config.json`` of a Hugging Face GPT-2 with the shape of ``config``."""
    return {
        "architectures": [config.architecture],
        "model_type": "gpt2",
        "vocab_size": config.vocab_size,
        "n_embd": config.hidden_size,
        "n_layer": config.num_layers,
        "n_head": config.num_attention_heads,
        "n_positions": config.max_position_embeddings,
        "n_inner": None,
        "activation_function": config.hidden_act,
        "layer_norm_epsilon": config.norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
    }


def write_checkpoint(
    directory: Path,
    state: dict[str, torch.Tensor],
    config: ModelConfig | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    shards: int = 1,
    reverse: bool = False,
) -> Path:
    """Write ``state`` as a safetensors checkpoint, with a ``config.json`` if ``config`` is given.

    One file, ``model.safetensors``, or ``shards`` files with the index Hugging Face writes.
    ``reverse`` lays the tensors out in the opposite order to ``state``. Returns ``directory``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    names = list(reversed(state)) if reverse else list(state)
    if config is not None:
        (directory / "config.json").write_text(json.dumps(hf_config_dict(config)), encoding="utf-8")
    if shards == 1:
        write_safetensors(
            directory / "model.safetensors", {n: tensor_data(state[n], dtype) for n in names}
        )
        return directory
    weight_map: dict[str, str] = {}
    for shard in range(shards):
        file = f"model-{shard + 1:05d}-of-{shards:05d}.safetensors"
        members = names[shard::shards]
        write_safetensors(directory / file, {n: tensor_data(state[n], dtype) for n in members})
        weight_map |= dict.fromkeys(members, file)
    index = {"metadata": {}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return directory
