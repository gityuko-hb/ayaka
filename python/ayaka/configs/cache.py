"""How the KV cache is sized: block size, block count, dtype codec, and state checkpoints.

``CacheConfig`` is a frozen host value. It holds the user's choices only. The
geometry of a block (bytes per token, layers, heads) comes from the model, so
this module knows nothing about ``kvcache``; ``memory.plan_budget`` combines the
two and turns ``memory_ratio`` or ``num_blocks`` into a block count.

Block ``0`` of every pool is padding and never given to a request, which is why
an explicit block count has a floor of two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

from ayaka.configs.base import ConfigMixin
from ayaka.core.errors import ConfigError
from ayaka.utils.math_utils import div_ceil
from ayaka.utils.validation import require_bool, require_int, require_number

__all__ = ["CacheConfig", "KVDtypeMode"]

type KVDtypeMode = Literal["auto", "fp8"]
"""KV cache storage dtype policy: ``"auto"`` (compute dtype) or ``"fp8"`` (float8_e4m3fn)."""

_KV_DTYPE_MODES: Final = ("auto", "fp8")


@dataclass(frozen=True, slots=True)
class CacheConfig(ConfigMixin):
    """Block size, storage dtype codec, and memory budget of the paged KV cache.

    Attributes:
        block_size: Tokens per cache block.
        num_blocks: Fixed block count per pool, or ``None`` to derive it from
            ``memory_ratio``.
        memory_ratio: Fraction of the free memory measured before anything is
            loaded that the engine may use, in ``(0, 1]``.
        kv_reserve_tokens: Tokens of KV the engine must be able to hold for one
            request, or ``None`` for the model's own context length.
        prefix_cache: Share the KV of full prompt blocks between requests with the
            same prefix.
        kv_dtype: Opt-in KV cache storage mode: ``"auto"`` keeps the compute
            dtype unchanged; ``"fp8"`` halves KV cache VRAM using ``float8_e4m3fn``.
        kv_fp8_scale: Static scale applied when storing and reading FP8 KV values.
        state_checkpoint_stride: Store recurrent state checkpoints every N blocks
            on hybrid linear-attention models (>= 1).
        state_checkpoint_max_bytes: Host RAM byte ceiling for recurrent state
            checkpoints (0 disables state checkpointing).
    """

    block_size: int = 16
    num_blocks: int | None = None
    memory_ratio: float = 0.90
    kv_reserve_tokens: int | None = None
    prefix_cache: bool = True
    kv_dtype: KVDtypeMode = "auto"
    kv_fp8_scale: float = 1.0
    state_checkpoint_stride: int = 1
    state_checkpoint_max_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        require_int(self.block_size, "cache.block_size", minimum=1)
        require_bool(self.prefix_cache, "cache.prefix_cache")
        if self.kv_dtype not in _KV_DTYPE_MODES:
            raise ConfigError(
                "cache.kv_dtype",
                "INVALID_CHOICE",
                f"must be one of {list(_KV_DTYPE_MODES)}, got {self.kv_dtype!r}",
            )
        fp8_scale = require_number(self.kv_fp8_scale, "cache.kv_fp8_scale")
        if not math.isfinite(fp8_scale) or fp8_scale <= 0.0:
            raise ConfigError(
                "cache.kv_fp8_scale",
                "INVALID_RANGE",
                f"must be finite and positive, got {fp8_scale}",
            )
        object.__setattr__(self, "kv_fp8_scale", float(fp8_scale))
        require_int(self.state_checkpoint_stride, "cache.state_checkpoint_stride", minimum=1)
        require_int(self.state_checkpoint_max_bytes, "cache.state_checkpoint_max_bytes", minimum=0)
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
