"""The single configuration object that builds an engine.

``EngineConfig`` composes the model source, the tokenizer, the device, the cache
and the scheduler records. ``ayaka.engine.build.build_engine`` takes this and
nothing else, so a run is described, compared and fingerprinted as one value.

The architecture (:class:`~ayaka.configs.model.ModelConfig`) is not part of it:
it is read from the checkpoint after the source is resolved, so it is a result of
building, not an input. Checks that need it happen there.

::

    ModelSourceConfig --+--> tokenizer_config (derived unless given)
    DeviceConfig -------+
    CacheConfig --------+--> EngineConfig --> build_engine
    SchedulerConfig ----+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from ayaka.configs.base import ConfigMixin
from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.host_memory import HostMemoryConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.tokenizer import TokenizerConfig, tokenizer_config_from_source
from ayaka.core.errors import ConfigError

__all__ = ["EngineConfig"]


@dataclass(frozen=True, slots=True)
class EngineConfig(ConfigMixin):
    """Everything ``build_engine`` needs to build an engine.

    Attributes:
        source: Where the model and, by default, its tokenizer come from.
        tokenizer: Tokenizer settings, or ``None`` for the defaults of
            :func:`~ayaka.configs.tokenizer.tokenizer_config_from_source`. Read
            the effective value from :attr:`tokenizer_config`.
        device: Device, dtype and ops backends.
        cache: KV cache sizing.
        scheduler: Scheduling limits.

    Raises:
        TypeError: If a field is not an instance of its record type.
        ConfigError: When fields of different records conflict:
            ``TOKENIZER_TRUST_CONFLICT`` (the tokenizer may run remote code that
            the source forbids), ``TOKENIZER_OFFLINE_CONFLICT`` (the tokenizer may
            use the network although the source is offline),
            ``STEP_SMALLER_THAN_BLOCK`` (a step budget below one cache block can
            never fill a block), ``RUNNING_EXCEEDS_BLOCKS`` (more running requests
            than the explicit block count has blocks for).
    """

    source: ModelSourceConfig
    tokenizer: TokenizerConfig | None = None
    device: DeviceConfig = field(default_factory=DeviceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    host_memory: HostMemoryConfig = field(default_factory=HostMemoryConfig)
    structured_backend: Literal["llguidance", "xgrammar"] = "llguidance"

    def __post_init__(self) -> None:
        if self.structured_backend not in ("llguidance", "xgrammar"):  # pyright: ignore[reportUnnecessaryComparison]
            raise ConfigError(
                "engine.structured_backend",
                "INVALID_CHOICE",
                f"must be llguidance or xgrammar, got {self.structured_backend!r}",
            )
        for name, kind in (
            ("source", ModelSourceConfig),
            ("device", DeviceConfig),
            ("cache", CacheConfig),
            ("scheduler", SchedulerConfig),
            ("host_memory", HostMemoryConfig),
        ):
            if not isinstance(cast("object", getattr(self, name)), kind):
                raise TypeError(f"engine.{name} must be a {kind.__name__}")
        tokenizer = cast("object", self.tokenizer)
        if tokenizer is not None and not isinstance(tokenizer, TokenizerConfig):
            raise TypeError("engine.tokenizer must be a TokenizerConfig or None")

        if self.tokenizer is not None:
            if self.tokenizer.trust_remote_code and not self.source.allows_remote_code:
                raise ConfigError(
                    "engine.tokenizer",
                    "TOKENIZER_TRUST_CONFLICT",
                    "tokenizer.trust_remote_code is set but the model source trust is "
                    "'strict'; set the source trust to allow_remote_code or clear the flag",
                )
            if self.source.offline and not self.tokenizer.offline:
                raise ConfigError(
                    "engine.tokenizer",
                    "TOKENIZER_OFFLINE_CONFLICT",
                    "the model source is offline but the tokenizer may use the network",
                )
        if self.scheduler.max_tokens_per_step < self.cache.block_size:
            raise ConfigError(
                "engine.scheduler",
                "STEP_SMALLER_THAN_BLOCK",
                f"max_tokens_per_step={self.scheduler.max_tokens_per_step} is smaller than "
                f"cache.block_size={self.cache.block_size}",
            )
        blocks = self.cache.num_blocks
        running = self.scheduler.max_running
        if blocks is not None and running is not None and running > blocks - 1:
            raise ConfigError(
                "engine.scheduler",
                "RUNNING_EXCEEDS_BLOCKS",
                f"max_running={running} needs a block each, but cache.num_blocks={blocks} "
                f"leaves {blocks - 1} after the padding block",
            )

    @property
    def tokenizer_config(self) -> TokenizerConfig:
        """The tokenizer settings in effect: ``tokenizer``, or the source's defaults."""
        if self.tokenizer is not None:
            return self.tokenizer
        return tokenizer_config_from_source(self.source)
