"""Embedding ops: the contract a backend implements for ``ayaka.layers.embedding``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["EmbeddingOps"]


class EmbeddingOps(Protocol):
    """Row lookup in an embedding table."""

    def lookup(self, token_ids: Tensor, weight: Tensor, out: Tensor) -> None:
        """Write the rows of ``weight`` selected by ``token_ids`` into ``out``.

        ``token_ids`` is an int32 or int64 tensor of any shape ``[...]``;
        ``weight`` is ``[vocab, hidden]``; ``out`` is ``[..., hidden]`` in the
        dtype of ``weight`` and must not overlap it. Rows are copied, so the
        result is bit-exact.

        Every id must lie in ``[0, vocab)``. The op does not check this: on a
        GPU the check would force a device-to-host synchronisation, and ids are
        validated when the request is admitted.
        """
        ...
