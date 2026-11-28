"""A prefix cache that matches full blocks by their hash chain.

:class:`HashPrefixCache` keeps no table of its own: each :class:`~ayaka.kvcache.blocks.BlockPool`
already maps a digest to the block registered under it, so a lookup walks the prompt's digests and
asks every group's pool. The first digest some group lacks ends the match.

Only full blocks are ever matched, and when the whole prompt is matched the last block is given
back, so the request still prefills at least one token (its logits produce the first output) and
the first chunk still ends on a block boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayaka.core.errors import EngineError
from ayaka.kvcache.prefix.base import PrefixMatch
from ayaka.kvcache.prefix.hashing import ChainRoot, chain_digests, root_digest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayaka.kvcache.blocks import BlockPool

__all__ = ["HashPrefixCache"]

#: Salts whose roots are remembered; a server sees few distinct salts.
_ROOT_MEMO_LIMIT = 1024


class HashPrefixCache:
    """Matches prompts against the digests registered in the block pools.

    Args:
        identity: What the KV values depend on besides the tokens.
    """

    def __init__(self, identity: ChainRoot) -> None:
        self._identity = identity
        self._roots: dict[str, bytes] = {}

    @property
    def identity(self) -> ChainRoot:
        """The model, dtype, codec and block size every root folds in."""
        return self._identity

    @property
    def block_size(self) -> int:
        """Tokens per block."""
        return self._identity.block_size

    @property
    def lookahead(self) -> int:
        """Trailing tokens beyond each block that its digest depends on."""
        return self._identity.lookahead

    def root(self, salt: str | None) -> bytes:
        """The root digest for ``salt``, memoised."""
        key = salt or ""
        found = self._roots.get(key)
        if found is None:
            found = root_digest(self._identity, salt)
            if len(self._roots) >= _ROOT_MEMO_LIMIT:
                self._roots.clear()
            self._roots[key] = found
        return found

    def digests(self, parent: bytes, token_ids: Sequence[int]) -> list[bytes]:
        """Digests of the full blocks of ``token_ids``, chained from ``parent``."""
        return chain_digests(
            parent,
            token_ids,
            self._identity.block_size,
            lookahead=self._identity.lookahead,
        )

    def match(
        self, pools: Sequence[BlockPool], token_ids: Sequence[int], salt: str | None
    ) -> PrefixMatch:
        """The longest run of full prompt blocks every group still holds.

        Raises:
            EngineError: If a digest is registered in some groups and not in others; groups
                register together, so that is broken bookkeeping, not a miss.
        """
        root = self.root(salt)
        digests = self.digests(root, token_ids)
        matched: list[list[int]] = [[] for _ in pools]
        for digest in digests:
            ids = [block_id for pool in pools if (block_id := pool.lookup(digest)) is not None]
            if len(ids) != len(pools):
                if ids:
                    raise EngineError("a block digest is registered in some KV groups only")
                break
            for group, block_id in enumerate(ids):
                matched[group].append(block_id)
        count = len(matched[0]) if matched else 0
        if count and count * self.block_size >= len(token_ids):
            count -= 1
        return PrefixMatch(
            num_tokens=count * self.block_size,
            num_blocks=count,
            root=root,
            digests=tuple(digests),
            blocks=tuple(tuple(ids[:count]) for ids in matched),
        )
