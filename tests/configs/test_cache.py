from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ayaka.configs.cache import CacheConfig
from ayaka.core.errors import ConfigError


@pytest.mark.unit
class TestCacheConfig:
    def test_defaults(self) -> None:
        config = CacheConfig()

        assert config.block_size == 16
        assert config.num_blocks is None
        assert config.memory_ratio == 0.90
        assert config.kv_reserve_tokens is None

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            CacheConfig().block_size = 8  # type: ignore[misc]  # frozen dataclass

    def test_serialization_and_fingerprint(self) -> None:
        config = CacheConfig(block_size=4, num_blocks=9)

        assert config.to_dict() == {
            "block_size": 4,
            "num_blocks": 9,
            "memory_ratio": 0.9,
            "kv_reserve_tokens": None,
            "prefix_cache": True,
        }
        assert config.fingerprint != CacheConfig(block_size=4, num_blocks=10).fingerprint

    @pytest.mark.parametrize("value", [0, -1])
    def test_block_size_must_be_positive(self, value: int) -> None:
        with pytest.raises(ValueError, match=r"cache\.block_size must be >= 1"):
            CacheConfig(block_size=value)

    @pytest.mark.parametrize("value", [True, 16.0, "16", None])
    def test_block_size_must_be_an_integer(self, value: object) -> None:
        with pytest.raises(TypeError, match=r"cache\.block_size must be an integer"):
            CacheConfig(block_size=value)  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize("ratio", [0.0, -0.1, 1.01, 2])
    def test_memory_ratio_must_be_in_unit_interval(self, ratio: float) -> None:
        with pytest.raises(ConfigError) as caught:
            CacheConfig(memory_ratio=ratio)

        assert caught.value.code == "INVALID_RANGE"
        assert caught.value.path == "cache.memory_ratio"

    @pytest.mark.parametrize("ratio", [float("nan"), "0.9", True])
    def test_memory_ratio_rejects_non_numbers_and_nan(self, ratio: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            CacheConfig(memory_ratio=ratio)  # type: ignore[arg-type]  # deliberately wrong type

    def test_memory_ratio_one_is_allowed_and_stored_as_float(self) -> None:
        config = CacheConfig(memory_ratio=1)

        assert config.memory_ratio == 1.0
        assert isinstance(config.memory_ratio, float)

    @pytest.mark.parametrize("blocks", [-3, 0, 1])
    def test_explicit_blocks_keep_the_padding_block_free(self, blocks: int) -> None:
        with pytest.raises(ConfigError) as caught:
            CacheConfig(num_blocks=blocks)

        assert caught.value.code == "NUM_BLOCKS_TOO_SMALL"
        assert "padding" in caught.value.message

    def test_two_blocks_is_the_smallest_pool(self) -> None:
        assert CacheConfig(num_blocks=2).num_blocks == 2

    @pytest.mark.parametrize("blocks", [True, 8.0, "8"])
    def test_blocks_must_be_an_integer(self, blocks: object) -> None:
        with pytest.raises(TypeError, match=r"cache\.num_blocks must be an integer"):
            CacheConfig(num_blocks=blocks)  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize("tokens", [0, -5])
    def test_reserve_must_be_positive(self, tokens: int) -> None:
        with pytest.raises(ValueError, match=r"cache\.kv_reserve_tokens must be >= 1"):
            CacheConfig(kv_reserve_tokens=tokens)

    def test_reserve_without_explicit_blocks_is_unchecked(self) -> None:
        assert CacheConfig(kv_reserve_tokens=10**9).kv_reserve_tokens == 10**9

    def test_explicit_blocks_must_hold_the_reserve(self) -> None:
        # 33 tokens at 16 per block need 3 blocks; 4 blocks leave 3 after the padding block.
        assert CacheConfig(block_size=16, num_blocks=4, kv_reserve_tokens=33).num_blocks == 4

        with pytest.raises(ConfigError) as caught:
            CacheConfig(block_size=16, num_blocks=3, kv_reserve_tokens=33)

        assert caught.value.code == "RESERVE_EXCEEDS_BLOCKS"
        assert "3 blocks" in caught.value.message
        assert "leaves 2" in caught.value.message

    def test_reserve_that_exactly_fills_the_pool_is_accepted(self) -> None:
        config = CacheConfig(block_size=4, num_blocks=5, kv_reserve_tokens=16)

        assert config.kv_reserve_tokens == 16
