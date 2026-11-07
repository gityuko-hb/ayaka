from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ayaka.configs.base import ConfigError
from ayaka.configs.model_source import ModelSourceConfig, ModelTrustPolicy
from ayaka.configs.tokenizer import (
    DetokenizeParams,
    DetokUpdate,
    EncodeResult,
    TokenizerConfig,
    tokenizer_config_from_source,
)


@pytest.mark.unit
class TestTokenizerConfig:
    def test_defaults(self) -> None:
        config = TokenizerConfig("org/model")

        assert config.mode == "auto"
        assert not config.offline
        assert config.revision is None
        assert config.special_token_policy == "reject"

    def test_accepts_path(self) -> None:
        assert TokenizerConfig(Path("tok")).tokenizer == Path("tok")

    def test_rejects_non_path_tokenizer(self) -> None:
        with pytest.raises(TypeError, match="string or Path"):
            TokenizerConfig(1)  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("mode", "fast"),
            ("truncation_side", "middle"),
            ("detokenize_backend", "x"),
            ("special_token_policy", "maybe"),
        ],
    )
    def test_rejects_invalid_choice(self, field: str, value: str) -> None:
        kwargs: dict[str, Any] = {field: value}
        with pytest.raises(ConfigError) as info:
            TokenizerConfig("t", **kwargs)
        assert info.value.code == "INVALID_CHOICE"
        assert info.value.path == f"tokenizer.{field}"

    @pytest.mark.parametrize(
        "field", ["trust_remote_code", "offline", "skip_tokenizer_init", "enable_session_cache"]
    )
    def test_rejects_non_bool_flags(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: 1}
        with pytest.raises(TypeError, match=f"{field} must be bool"):
            TokenizerConfig("t", **kwargs)

    @pytest.mark.parametrize("field", ["encode_max_batch", "chunk_chars", "session_cache_capacity"])
    def test_rejects_non_positive_sizes(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: 0}
        with pytest.raises(ValueError, match=field):
            TokenizerConfig("t", **kwargs)

    @pytest.mark.parametrize("window", [-1.0, float("inf"), float("nan"), True, "1"])
    def test_rejects_bad_batch_window(self, window: Any) -> None:
        with pytest.raises(ValueError, match="encode_batch_window_ms"):
            TokenizerConfig("t", encode_batch_window_ms=window)

    def test_verify_requires_cache(self) -> None:
        with pytest.raises(ConfigError) as info:
            TokenizerConfig("t", verify_session_cache=True)
        assert info.value.code == "CACHE_DISABLED"

    @pytest.mark.parametrize(
        "kwargs", [{"enable_session_cache": True}, {"detokenize_backend": "bytes"}]
    )
    def test_token_id_only_mode_conflicts(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ConfigError) as info:
            TokenizerConfig("t", skip_tokenizer_init=True, **kwargs)
        assert info.value.code == "TOKENIZER_DISABLED"


@pytest.mark.unit
class TestFromSource:
    def test_copies_source_policy(self) -> None:
        source = ModelSourceConfig(
            "org/model",
            revision="main",
            tokenizer="org/tok",
            tokenizer_revision="tok-rev",
            trust=ModelTrustPolicy.ALLOW_REMOTE_CODE,
            offline=True,
            cache_dir="/cache",
        )

        config = tokenizer_config_from_source(source, mode="hf")

        assert config.tokenizer == "org/tok"
        assert config.mode == "hf"
        assert config.revision == "tok-rev"
        assert config.trust_remote_code
        assert config.offline
        assert config.download_dir == "/cache"

    def test_special_token_policy_is_passed_through(self) -> None:
        source = ModelSourceConfig("org/model")

        assert tokenizer_config_from_source(source).special_token_policy == "reject"
        assert (
            tokenizer_config_from_source(source, special_token_policy="escape").special_token_policy
            == "escape"
        )

    def test_separate_tokenizer_uses_its_own_revision(self) -> None:
        source = ModelSourceConfig(
            "org/model", revision="v1", tokenizer="org/tok", tokenizer_revision="v2"
        )

        assert tokenizer_config_from_source(source).revision == "v2"

    def test_separate_tokenizer_does_not_inherit_model_revision(self) -> None:
        source = ModelSourceConfig("org/model", revision="v1", tokenizer="org/tok")

        assert tokenizer_config_from_source(source).revision is None

    def test_shared_tokenizer_uses_model_revision(self) -> None:
        source = ModelSourceConfig("org/model", revision="v1", tokenizer_revision="v2")

        assert tokenizer_config_from_source(source).revision == "v1"

    def test_empty_source_fields_become_none(self) -> None:
        config = tokenizer_config_from_source(ModelSourceConfig("org/model"))

        assert config.tokenizer == "org/model"
        assert config.revision is None
        assert config.download_dir is None
        assert not config.offline


@pytest.mark.unit
class TestDetokenizeParams:
    def test_max_stop_len(self) -> None:
        assert DetokenizeParams().max_stop_len() == 0
        assert DetokenizeParams(stop=("a", "abc")).max_stop_len() == 3

    @pytest.mark.parametrize("stop", [["a"], ("",), (1,)])
    def test_rejects_bad_stop(self, stop: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            DetokenizeParams(stop=stop)

    def test_rejects_non_bool(self) -> None:
        with pytest.raises(TypeError, match="accumulate_text"):
            DetokenizeParams(accumulate_text=1)  # type: ignore[arg-type]  # deliberately wrong type


@pytest.mark.unit
class TestDetokUpdate:
    def test_defaults(self) -> None:
        assert DetokUpdate().delta == ""

    def test_rejects_empty_stop_matched(self) -> None:
        with pytest.raises(ValueError, match="stop_matched"):
            DetokUpdate(stop_matched="")


@pytest.mark.unit
class TestEncodeResult:
    def test_valid(self) -> None:
        result = EncodeResult((1, 2, 3), reused_prefix_len=2, extra=(("k", 1),))

        assert result.chunks == 1

    def test_rejects_list_ids(self) -> None:
        with pytest.raises(TypeError, match="immutable host values"):
            EncodeResult([1, 2])  # type: ignore[arg-type]  # deliberately wrong type

    def test_rejects_prefix_longer_than_ids(self) -> None:
        with pytest.raises(ValueError, match="reused prefix"):
            EncodeResult((1,), reused_prefix_len=2)

    def test_rejects_duplicate_extra_key(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            EncodeResult((), extra=(("k", 1), ("k", 2)))

    def test_rejects_bad_extra_value(self) -> None:
        with pytest.raises(TypeError, match="string, int or bool"):
            EncodeResult((), extra=(("k", 1.5),))  # type: ignore[arg-type]  # deliberately wrong
