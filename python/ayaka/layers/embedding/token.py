"""The token embedding table and the output projection onto the vocabulary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ayaka.layers.linear import Linear

if TYPE_CHECKING:
    from ayaka.ops.backend import OpsBackend

__all__ = ["LMHead", "TokenEmbedding"]


class TokenEmbedding(nn.Module):
    """Rows of a ``[vocab_size, hidden_size]`` table selected by token id.

    The parameter is named ``weight``, like :class:`torch.nn.Embedding`, and is
    allocated uninitialised.

    Args:
        vocab_size: Number of rows.
        hidden_size: Width of a row.
        ops: The backend whose ``embedding`` op does the lookup.
        dtype: Compute dtype of the table.
        device: Where the table lives.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=dtype, device=device), requires_grad=False
        )
        self._ops = ops.embedding

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return the rows for ``token_ids`` of shape ``[...]`` as ``[..., hidden_size]``.

        Every id must be in ``[0, vocab_size)``; the op does not check it.
        """
        out = torch.empty(
            (*token_ids.shape, self.hidden_size), dtype=self.weight.dtype, device=token_ids.device
        )
        self._ops.lookup(token_ids, self.weight, out)
        return out


class LMHead(Linear):
    """Projects hidden states onto the vocabulary: a :class:`Linear` without a bias.

    Args:
        hidden_size: Width of the input.
        vocab_size: Number of logits.
        ops: The backend whose ``linear`` op does the work.
        dtype: Compute dtype of the weight.
        device: Where the weight lives.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        ops: OpsBackend,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(hidden_size, vocab_size, bias=False, ops=ops, dtype=dtype, device=device)

    def tie_weights(self, embedding: TokenEmbedding) -> None:
        """Share the table of ``embedding`` as this head's weight, as GPT-2 does.

        The parameter becomes the same object, so a checkpoint that stores the
        table once loads both.

        Raises:
            ValueError: If the table does not have this head's shape and dtype.
        """
        if (
            embedding.weight.shape != self.weight.shape
            or embedding.weight.dtype != self.weight.dtype
        ):
            raise ValueError(
                f"cannot tie a {self.weight.dtype} head of shape {list(self.weight.shape)} to "
                f"a {embedding.weight.dtype} embedding of shape {list(embedding.weight.shape)}"
            )
        self.weight = embedding.weight
