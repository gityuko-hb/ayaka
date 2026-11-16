from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from ayaka.configs.cache import CacheConfig
from ayaka.configs.device import DeviceConfig
from ayaka.configs.engine import EngineConfig
from ayaka.configs.model_source import ModelSourceConfig, ModelTrustPolicy
from ayaka.configs.scheduler import SchedulerConfig
from ayaka.configs.tokenizer import TokenizerConfig, tokenizer_config_from_source
from ayaka.core.errors import ConfigError
from ayaka.utils.dtypes import DType

SOURCE = ModelSourceConfig("org/model")


@pytest.mark.unit
class TestEngineConfig:
    def test_defaults_compose_the_default_records(self) -> None:
        config = EngineConfig(SOURCE)

        assert config.source is SOURCE
        assert config.device == DeviceConfig()
        assert config.cache == CacheConfig()
        assert config.scheduler == SchedulerConfig()
        assert config.tokenizer is None

    def test_equal_inputs_give_equal_configs(self) -> None:
        assert EngineConfig(SOURCE) == EngineConfig(SOURCE)

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            EngineConfig(SOURCE).cache = CacheConfig()  # type: ignore[misc]  # frozen dataclass

    def test_effective_tokenizer_defaults_to_the_source(self) -> None:
        source = ModelSourceConfig("org/model", tokenizer="org/tok", offline=True)

        assert EngineConfig(source).tokenizer_config == tokenizer_config_from_source(source)

    def test_explicit_tokenizer_is_used_as_given(self) -> None:
        tokenizer = TokenizerConfig("org/model", mode="hf")

        assert EngineConfig(SOURCE, tokenizer=tokenizer).tokenizer_config is tokenizer

    def test_serialization_nests_every_record(self) -> None:
        payload = EngineConfig(
            SOURCE, device=DeviceConfig(dtype=DType.BF16), cache=CacheConfig(block_size=4)
        ).to_dict()

        assert payload["source"]["model"] == "org/model"
        assert payload["device"]["dtype"] == "bf16"
        assert payload["cache"]["block_size"] == 4
        assert payload["scheduler"]["policy"] == "fcfs"
        assert payload["tokenizer"] is None

    def test_fingerprint_changes_with_any_record(self) -> None:
        base = EngineConfig(SOURCE)

        assert base.fingerprint == EngineConfig(SOURCE).fingerprint
        assert base.fingerprint != replace(base, cache=CacheConfig(block_size=4)).fingerprint
        assert (
            base.fingerprint != replace(base, scheduler=SchedulerConfig(max_running=1)).fingerprint
        )
        assert base.fingerprint != replace(base, device=DeviceConfig(dtype=DType.FP16)).fingerprint

    @pytest.mark.parametrize("field", ["source", "device", "cache", "scheduler", "tokenizer"])
    def test_rejects_wrong_record_types(self, field: str) -> None:
        with pytest.raises(TypeError, match=rf"engine\.{field} must be"):
            EngineConfig(**{**{"source": SOURCE}, field: "not a record"})  # type: ignore[arg-type]  # deliberately wrong type

    def test_tokenizer_remote_code_needs_a_trusting_source(self) -> None:
        tokenizer = TokenizerConfig("org/model", trust_remote_code=True)

        with pytest.raises(ConfigError) as caught:
            EngineConfig(SOURCE, tokenizer=tokenizer)

        assert caught.value.code == "TOKENIZER_TRUST_CONFLICT"
        trusting = ModelSourceConfig("org/model", trust=ModelTrustPolicy.ALLOW_REMOTE_CODE)
        assert EngineConfig(trusting, tokenizer=tokenizer).tokenizer is tokenizer

    def test_offline_source_needs_an_offline_tokenizer(self) -> None:
        source = ModelSourceConfig("org/model", offline=True)

        with pytest.raises(ConfigError) as caught:
            EngineConfig(source, tokenizer=TokenizerConfig("org/model"))

        assert caught.value.code == "TOKENIZER_OFFLINE_CONFLICT"
        assert EngineConfig(source, tokenizer=TokenizerConfig("org/model", offline=True))

    def test_step_budget_must_cover_one_block(self) -> None:
        with pytest.raises(ConfigError) as caught:
            EngineConfig(
                SOURCE,
                cache=CacheConfig(block_size=64),
                scheduler=SchedulerConfig(max_tokens_per_step=32),
            )

        assert caught.value.code == "STEP_SMALLER_THAN_BLOCK"
        assert "64" in caught.value.message
        assert "32" in caught.value.message

    def test_step_budget_equal_to_a_block_is_accepted(self) -> None:
        config = EngineConfig(
            SOURCE,
            cache=CacheConfig(block_size=16),
            scheduler=SchedulerConfig(max_tokens_per_step=16),
        )

        assert config.scheduler.max_tokens_per_step == 16

    def test_running_requests_need_a_block_each(self) -> None:
        cache = CacheConfig(num_blocks=5)

        with pytest.raises(ConfigError) as caught:
            EngineConfig(SOURCE, cache=cache, scheduler=SchedulerConfig(max_running=5))

        assert caught.value.code == "RUNNING_EXCEEDS_BLOCKS"
        assert EngineConfig(SOURCE, cache=cache, scheduler=SchedulerConfig(max_running=4))

    def test_derived_limits_are_not_cross_checked(self) -> None:
        # max_running=None and num_blocks=None are derived by plan_budget, not here.
        assert EngineConfig(
            SOURCE, cache=CacheConfig(num_blocks=2), scheduler=SchedulerConfig(max_running=1)
        )
        assert EngineConfig(SOURCE, scheduler=SchedulerConfig(max_running=10**6))
