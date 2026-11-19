"""How the KV cache is sized: block size, block count and the memory it may use.

``CacheConfig`` is a frozen host value. It holds the user's choices only. The
geometry of a block (bytes per token, layers, heads) comes from the model, so
this module knows nothing about ``kvcache``; ``memory.plan_budget`` combines the
two and turns ``memory_ratio`` or ``num_blocks`` into a block count.

Block ``0`` of every pool is padding and never given to a request, which is why
an explicit block count has a floor of two.
"""

from __future__ import annotations

from dataclasses import dataclass

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_bool, require_int, require_number

__all__ = ["CacheConfig"]


@dataclass(frozen=True, slots=True)
class CacheConfig(ConfigMixin):
    """Block size and memory budget of the paged KV cache.

    Attributes:
        block_size: Tokens per cache block. The default of 16 is provisional: the
            final default is an open decision that waits for a measurement.
        num_blocks: Fixed block count per pool, or ``None`` to derive it from
            ``memory_ratio``. When set, the memory division is skipped but the
            total must still fit in ``memory_ratio`` of the free memory. At least
            2: block 0 is padding.
        memory_ratio: Fraction of the free memory measured before anything is
            loaded that the engine may use, in ``(0, 1]``. The rest is headroom
            for activations and graphs. The default of 0.90 is provisional, for
            the same reason as ``block_size``.
        kv_reserve_tokens: Tokens of KV the engine must be able to hold for one
            request, or ``None`` for the model's own minimum (its context
            length). Startup is refused when the plan cannot reach it.
        prefix_cache: Share the KV of full prompt blocks between requests with the same
            prefix (a request can still opt out with ``prefix_cache=False``).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a number is out of range.
        ConfigError: ``INVALID_RANGE`` for ``memory_ratio``, ``NUM_BLOCKS_TOO_SMALL``
            for ``num_blocks < 2``, ``RESERVE_EXCEEDS_BLOCKS`` when an explicit
            ``num_blocks`` cannot hold ``kv_reserve_tokens``.
    """

    block_size: int = 16
    num_blocks: int | None = None
    memory_ratio: float = 0.90
    kv_reserve_tokens: int | None = None
    prefix_cache: bool = True

    def __post_init__(self) -> None:
        require_int(self.block_size, "cache.block_size", minimum=1)
        require_bool(self.prefix_cache, "cache.prefix_cache")
        ratio = require_number(self.memory_ratio, "cache.memory_ratio")
        if not 0.0 < ratio <= 1.0:
            raise ConfigError(
                "cache.memory_ratio",
                "INVALID_RANGE",
                f"must be in (0, 1], got {ratio}",
            )
        object.__setattr__(self, "memory_ratio", ratio)
        if self.kv_reserve_tokens is not None:
            require_int(self.kv_reserve_tokens, "cache.kv_reserve_tokens", minimum=1)
        if self.num_blocks is None:
            return
        require_int(self.num_blocks, "cache.num_blocks", minimum=None)
        if self.num_blocks < 2:
            raise ConfigError(
                "cache.num_blocks",
                "NUM_BLOCKS_TOO_SMALL",
                f"must be >= 2 (block 0 is padding), got {self.num_blocks}",
            )
        if self.kv_reserve_tokens is not None:
            needed = div_ceil(self.kv_reserve_tokens, self.block_size)
            if needed > self.num_blocks - 1:
                raise ConfigError(
                    "cache.kv_reserve_tokens",
                    "RESERVE_EXCEEDS_BLOCKS",
                    f"{self.kv_reserve_tokens} tokens need {needed} blocks of "
                    f"{self.block_size}, but num_blocks={self.num_blocks} leaves "
                    f"{self.num_blocks - 1} after the padding block",
                )
