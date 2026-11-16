from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ayaka.configs.scheduler import SchedulerConfig
from ayaka.core.errors import ConfigError


@pytest.mark.unit
class TestSchedulerConfig:
    def test_defaults(self) -> None:
        config = SchedulerConfig()

        assert config.max_running is None
        assert config.max_tokens_per_step == 2048
        assert config.max_waiting == 1024
        assert config.policy == "fcfs"
        assert config.max_consecutive_prefills == 2

    def test_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            SchedulerConfig().max_waiting = 1  # type: ignore[misc]  # frozen dataclass

    def test_serialization_and_fingerprint(self) -> None:
        config = SchedulerConfig(max_running=4, policy="priority")

        assert config.to_dict()["policy"] == "priority"
        assert config.to_dict()["max_running"] == 4
        assert config.fingerprint != SchedulerConfig(max_running=4).fingerprint

    @pytest.mark.parametrize("policy", ["priority", "fcfs"])
    def test_accepts_both_policies(self, policy: str) -> None:
        assert SchedulerConfig(policy=policy).policy == policy  # type: ignore[arg-type]  # str narrowed by the check

    @pytest.mark.parametrize("policy", ["lifo", "", "FCFS", None])
    def test_rejects_unknown_policy(self, policy: object) -> None:
        with pytest.raises(ConfigError) as caught:
            SchedulerConfig(policy=policy)  # type: ignore[arg-type]  # deliberately wrong type

        assert caught.value.code == "INVALID_CHOICE"
        assert caught.value.path == "scheduler.policy"

    @pytest.mark.parametrize(
        "field", ["max_running", "max_tokens_per_step", "max_waiting", "max_consecutive_prefills"]
    )
    @pytest.mark.parametrize("value", [0, -1])
    def test_limits_must_be_positive(self, field: str, value: int) -> None:
        with pytest.raises(ValueError, match=rf"scheduler\.{field} must be >= 1"):
            SchedulerConfig(**{field: value})  # type: ignore[arg-type]  # keyword names are fields

    @pytest.mark.parametrize(
        "field", ["max_running", "max_tokens_per_step", "max_waiting", "max_consecutive_prefills"]
    )
    @pytest.mark.parametrize("value", [True, 2.0, "2"])
    def test_limits_must_be_integers(self, field: str, value: object) -> None:
        with pytest.raises(TypeError, match=rf"scheduler\.{field} must be an integer"):
            SchedulerConfig(**{field: value})  # type: ignore[arg-type]  # keyword names are fields
