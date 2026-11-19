"""The contract of a prefix cache: what a lookup returns and what a cache must answer.

A prefix cache finds the longest run of full prompt blocks whose KV some earlier request already
computed. Looking up changes nothing: the result names blocks, and only
:meth:`~ayaka.kvcache.manager.KVCacheManager.allocate` attaches them to a request, in the same
all-or-nothing call that gives it new blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.kvcache.blocks import BlockPool

__all__ = ["PrefixCache", "PrefixMatch"]


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    """The result of one lookup.

    Attributes:
        num_tokens: Prompt tokens whose KV is reused; a multiple of the block size and always
            shorter than the prompt, so a forward still runs on at least one token.
        num_blocks: Full blocks reused, the same in every group.
        root: The digest the chain starts from (model, format and salt).
        digests: The digest of every full prompt block, matched or not, so registering the
            request's blocks later never hashes the prompt again.
        blocks: Per group, the ids of the matched blocks, in token order.
    """

    num_tokens: int
    num_blocks: int
    root: bytes
    digests: tuple[bytes, ...]
    blocks: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        require_int(self.num_tokens, "num_tokens")
        require_int(self.num_blocks, "num_blocks")
        if any(len(ids) != self.num_blocks for ids in self.blocks):
            raise ValueError("every group must match the same number of blocks")
        if self.num_blocks > len(self.digests):
            raise ValueError("a match cannot reuse more blocks than the prompt has digests")

    @classmethod
    def miss(cls, root: bytes, digests: Sequence[bytes], num_groups: int) -> PrefixMatch:
        """A lookup that reused nothing but still carries the prompt's digests."""
        return cls(0, 0, root, tuple(digests), ((),) * num_groups)

    @property
    def hit(self) -> bool:
        """Whether any block is reused."""
        return self.num_blocks > 0


class PrefixCache(Protocol):
    """Finds reusable blocks for a prompt and names the blocks a request fills."""

    @property
    def block_size(self) -> int:
        """Tokens per block."""
        ...

    def root(self, salt: str | None) -> bytes:
        """The digest every chain of this salt starts from."""
        ...

    def digests(self, parent: bytes, token_ids: Sequence[int]) -> list[bytes]:
        """Digests of the full blocks of ``token_ids``, chained from ``parent``."""
        ...

    def match(
        self, pools: Sequence[BlockPool], token_ids: Sequence[int], salt: str | None
    ) -> PrefixMatch:
        """The longest reusable prefix of ``token_ids`` in ``pools``; read-only."""
        ...
