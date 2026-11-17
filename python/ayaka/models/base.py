"""The contract of a decoder-only language model, and what a model owns of its checkpoint.

A model is a :class:`torch.nn.Module` built from a
:class:`~ayaka.configs.model.ModelConfig` and the ops backend it will call. It
has two jobs at run time and two at load time::

    run time    forward(token_ids, context)  -> hidden states of every row of the step
                compute_logits(hidden_rows)  -> logits of the rows that will be sampled
    load time   weight_mapping(config)       -> checkpoint tensor names -> parameter names
                checkpoint_specs()           -> the parameters, in the layout the checkpoint stores
                load_weight(name, tensor)    -> write one checkpoint tensor into its parameter

Where a checkpoint's layout is turned into the model's
----------------------------------------------------------------

``ayaka.weights`` describes tensors and renames them
(:class:`~ayaka.weights.mapping.WeightMapping` only renames and ignores) and
``ayaka.model_loader`` streams the bytes. Neither knows what a parameter is
*for*, so the model owns every change of layout between the two:

* **Names**: :meth:`CausalLM.weight_mapping`.
* **Transposition**: GPT-2 stores its projections as Hugging Face ``Conv1D``
  weights, ``[in, out]``, where a :class:`~ayaka.layers.linear.Linear` is
  ``[out, in]``. A model lists those parameters in :attr:`CausalLM.stored_transposed`;
  :meth:`CausalLM.checkpoint_specs` then reports the stored (transposed) shape, so a
  checkpoint with the wrong shape is caught by ``bind`` before a byte is read,
  and :meth:`CausalLM.load_weight` transposes while it copies.
* **Fusion**: Llama stores ``q_proj``, ``k_proj`` and ``v_proj`` separately
  where the model has one ``qkv_proj``, and ``gate_proj`` and ``up_proj`` where
  it has ``gate_up_proj``. A model lists each fused parameter's parts, in the
  order they are stacked along the first dimension, in :meth:`CausalLM.stored_parts`.
  :meth:`CausalLM.checkpoint_specs` then reports the parts instead of the
  fused parameter, under names of their own (``layers.0.attn.q_proj.weight``),
  and :meth:`CausalLM.load_weight` copies each part into its rows.

The loader never reads a parameter's layout from anywhere but these methods.

Buffers that no checkpoint stores (the rotary table) are computed by
:meth:`CausalLM.initialize_buffers`, which the loader calls once the model has
memory.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from torch import nn

from ayaka.core.errors import ModelLoadError
from ayaka.kvcache.storage.spec import KVGroup, KVSpec
from ayaka.weights.spec import TensorSpec

if TYPE_CHECKING:
    import torch

    from ayaka.configs.model import ModelConfig
    from ayaka.core.types import KVLayoutKind
    from ayaka.layers.context import ForwardContext
    from ayaka.ops.backend import OpsBackend
    from ayaka.utils.dtypes import DType
    from ayaka.weights.mapping import WeightMapping

__all__ = ["CausalLM"]


class CausalLM(nn.Module, ABC):
    """A decoder-only transformer that maps token ids to hidden states and logits.

    Subclasses build their layers in ``__init__``, give every parameter the
    compute dtype and ``device``, and leave the parameters uninitialised: the
    model loader writes the values. Parameter names are the model's own; the
    checkpoint's names are translated to them by :meth:`weight_mapping`.

    A model never touches the KV cache or a block manager. Each attention
    layer reads its cache tensors from the :class:`~ayaka.layers.context.ForwardContext`
    by ``layer_id``, and the layer ids are the ones :meth:`kv_specs` reports.

    Args:
        config: The architecture, from ``config.json``.
        ops: The ops backend every layer calls.
        dtype: Compute dtype of the parameters and of the activations.
        device: Where the parameters live; ``"meta"`` builds a model with no memory
            behind it, which is how the loader starts.

    Attributes:
        config: The architecture this model was built for.
        ops: The ops backend, also needed by whoever plans a step (``ops.attention.plan``).
        compute_dtype: The dtype given at construction.
        stored_transposed: Full-match regular expressions of the names of the
            parameters whose checkpoint tensor is stored with its two dimensions
            swapped. Empty unless the architecture's checkpoints do that.
    """

    stored_transposed: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        config: ModelConfig,
        *,
        ops: OpsBackend,
        dtype: DType,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ops = ops
        self.compute_dtype = dtype

    @classmethod
    @abstractmethod
    def weight_mapping(cls, config: ModelConfig) -> WeightMapping:
        """Return the rules that turn checkpoint tensor names into this model's parameter names.

        It depends on ``config`` because some tensors exist only for some
        configurations, such as an output head that is tied to the embedding.
        """

    @abstractmethod
    def forward(self, token_ids: torch.Tensor, context: ForwardContext) -> torch.Tensor:
        """Run every layer over the rows of one step.

        Args:
            token_ids: ``[N]`` int32 or int64 token ids, the rows of every request of the step
                in the request order of ``context``.
            context: The step's forward context. Models with position embeddings
                read ``context.positions``; attention layers read their cache from it.

        Returns:
            ``[N, hidden_size]`` hidden states after the final normalisation, in the
            compute dtype. Logits are a separate call so that a step samples only the
            rows it needs.
        """

    @abstractmethod
    def compute_logits(self, hidden_rows: torch.Tensor) -> torch.Tensor:
        """Project hidden states onto the vocabulary.

        Args:
            hidden_rows: ``[R, hidden_size]`` rows of :meth:`forward`, usually the last row of
                each request.

        Returns:
            ``[R, vocab_size]`` logits in the compute dtype.
        """

    def tie_weights(self) -> None:
        """Make parameters that share a table the same object again.

        The loader calls this after it materialises a model built on the meta
        device, which gives each tied parameter a tensor of its own. A model
        with nothing tied leaves it as it is.
        """

    def initialize_buffers(self) -> None:
        """Compute the buffers no checkpoint stores, on the device the model now lives on.

        The loader calls this after it materialises a model built on the meta
        device, whose buffers then hold arbitrary memory. The default rebuilds
        the table of every :class:`~ayaka.layers.rope.RotaryEmbedding` in the model.
        """
        from ayaka.layers.rope import RotaryEmbedding

        for module in self.modules():
            if isinstance(module, RotaryEmbedding):
                module.reset_cache()

    def stored_parts(self, name: str) -> tuple[tuple[str, int], ...]:
        """The checkpoint tensors that parameter ``name`` is fused from, with their rows.

        Each entry is ``(part name, rows)``; the parts are stacked along the
        first dimension in this order and their rows add up to the parameter's.
        The default is ``()``: the parameter is stored whole.
        """
        del name
        return ()

    def kv_specs(self, *, block_size: int, layout: KVLayoutKind) -> tuple[KVGroup, ...]:
        """Describe the KV cache this model needs, one group per distinct geometry.

        The default is full attention in every layer: one group of all
        ``num_layers`` layers, ids ``0`` to ``num_layers - 1``, each with
        ``num_kv_heads`` heads of ``head_dim``. A model with other layer kinds
        overrides it.

        Args:
            block_size: Tokens per cache block, from the cache config.
            layout: The layout the ops backend reads fastest (``ops.kvcache.layout()``).
        """
        config = self.config
        spec = KVSpec(
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            dtype=self.compute_dtype,
            layout=layout,
        )
        return (KVGroup(layer_ids=tuple(range(config.num_layers)), spec=spec),)

    def checkpoint_specs(self) -> list[TensorSpec]:
        """List every parameter as the checkpoint stores it.

        Names are parameter names (a tied parameter appears once, under the
        first name it is registered with), and a fused parameter is replaced by
        its parts (:meth:`stored_parts`). Shapes are the stored ones, so
        :meth:`~ayaka.weights.mapping.WeightMapping.bind` compares like with
        like: a parameter in :attr:`stored_transposed` is listed with its
        dimensions swapped. The dtype is the compute dtype; the checkpoint's
        own dtype is converted when the tensor is loaded.
        """
        specs: list[TensorSpec] = []
        for name, parameter in self.named_parameters():
            parts = self.stored_parts(name)
            if parts:
                tail = tuple(parameter.shape[1:])
                specs.extend(
                    TensorSpec(part, (rows, *tail), self.compute_dtype) for part, rows in parts
                )
                continue
            shape = tuple(parameter.shape)
            if self._is_stored_transposed(name):
                shape = tuple(reversed(shape))
            specs.append(TensorSpec(name, shape, self.compute_dtype))
        return specs

    def load_weight(self, name: str, tensor: torch.Tensor) -> None:
        """Copy one checkpoint tensor into the parameter ``name``, converting its dtype.

        Args:
            name: A name as listed by :meth:`checkpoint_specs`: a parameter, or a
                part of a fused one, which fills that part's rows.
            tensor: The tensor in the checkpoint's layout and dtype, on any device.

        Raises:
            ModelLoadError: If the tensor, once laid out for the parameter, does
                not have the parameter's shape. ``bind`` rules this out for a
                checkpoint it accepted; this is the guard for a direct call.
        """
        target = self._part_target(name)
        if target is not None:
            fused, start, rows = target
            parameter = self.get_parameter(fused)
            expected = (rows, *parameter.shape[1:])
            if tuple(tensor.shape) != expected:
                raise ModelLoadError(
                    f"{name} (rows {start}..{start + rows} of {fused}) is {list(expected)} but "
                    f"the checkpoint tensor is {list(tensor.shape)}",
                    context={"parameter": fused, "part": name},
                )
            parameter[start : start + rows].copy_(tensor)
            return
        parameter = self.get_parameter(name)
        if self._is_stored_transposed(name):
            tensor = tensor.t()
        if tensor.shape != parameter.shape:
            raise ModelLoadError(
                f"parameter {name} is {list(parameter.shape)} but the checkpoint tensor "
                f"is {list(tensor.shape)}",
                context={"parameter": name},
            )
        parameter.copy_(tensor)

    def _is_stored_transposed(self, name: str) -> bool:
        return any(re.fullmatch(pattern, name) for pattern in self.stored_transposed)

    def _part_target(self, name: str) -> tuple[str, int, int] | None:
        """``(fused parameter, first row, rows)`` when ``name`` is a part, else ``None``."""
        if name in dict(self.named_parameters()):
            return None
        for fused, _ in self.named_parameters():
            start = 0
            for part, rows in self.stored_parts(fused):
                if part == name:
                    return fused, start, rows
                start += rows
        return None
