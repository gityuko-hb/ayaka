"""Unit tests for HostMemoryConfig validation and EngineConfig integration."""

from __future__ import annotations

import pytest

from ayaka.configs.engine import EngineConfig
from ayaka.configs.host_memory import HostMemoryConfig
from ayaka.configs.model_source import ModelSourceConfig
from ayaka.core.errors import ConfigError

pytestmark = pytest.mark.unit


def test_host_memory_config_defaults_to_disabled() -> None:
    cfg = HostMemoryConfig()
    assert cfg.enabled is False
    assert cfg.ram_keep_fraction == 0.20
    assert cfg.max_host_kv_mb is None
    assert cfg.pinned is True
    assert cfg.allow_pageable is False
    assert cfg.max_inflight_offloads == 8


def test_host_memory_config_validation() -> None:
    with pytest.raises(TypeError):
        HostMemoryConfig(enabled="yes")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        HostMemoryConfig(ram_keep_fraction=0.01)
    with pytest.raises(ConfigError):
        HostMemoryConfig(ram_keep_fraction=0.95)
    with pytest.raises(ValueError, match="max_host_kv_mb"):
        HostMemoryConfig(max_host_kv_mb=0)
    with pytest.raises(ValueError, match="max_inflight_offloads"):
        HostMemoryConfig(max_inflight_offloads=0)


def test_engine_config_carries_host_memory() -> None:
    engine_cfg = EngineConfig(
        source=ModelSourceConfig("org/model"),
        host_memory=HostMemoryConfig(enabled=True, ram_keep_fraction=0.25, max_host_kv_mb=512),
    )
    assert engine_cfg.host_memory.enabled is True
    assert engine_cfg.host_memory.max_host_kv_mb == 512

    with pytest.raises(TypeError):
        EngineConfig(
            source=ModelSourceConfig("org/model"),
            host_memory="bad",  # type: ignore[arg-type]
        )
