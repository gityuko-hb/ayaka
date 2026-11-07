from __future__ import annotations

import pytest

from ayaka.configs.model_source import (
    ModelSourceConfig,
    ModelTrustPolicy,
    RequestFormat,
)


@pytest.mark.unit
class TestModelSourceConfig:
    def test_defaults_and_derived_values(self) -> None:
        config = ModelSourceConfig("org/model")

        assert config.format is RequestFormat.AUTO
        assert config.trust is ModelTrustPolicy.STRICT
        assert config.tokenizer_path == "org/model"
        assert not config.is_dummy
        assert not config.allows_remote_code

    def test_enum_values_are_canonicalized(self) -> None:
        config = ModelSourceConfig(
            "org/model",
            format=RequestFormat.DUMMY,
            trust=ModelTrustPolicy.ALLOW_REMOTE_CODE,
        )

        assert config.is_dummy
        assert config.allows_remote_code
        assert config.to_dict()["format"] == "dummy"
        assert config.to_dict()["trust"] == "allow_remote_code"

    @pytest.mark.parametrize("model", ["", "  ", None, 1])
    def test_rejects_empty_or_non_string_model(self, model: object) -> None:
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ModelSourceConfig(model)  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize("revision", ["", "main", "v1.0", "abcdef0", "g" * 40])
    def test_reproducible_requires_full_commit_sha(self, revision: str) -> None:
        with pytest.raises(ValueError, match="full 40-character commit SHA"):
            ModelSourceConfig("org/model", revision=revision, reproducible=True)

    def test_reproducible_accepts_full_commit_sha(self) -> None:
        config = ModelSourceConfig("org/model", revision="a" * 40, reproducible=True)

        assert config.reproducible

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("org/model", True),
            ("model", False),
            ("./org/model", False),
            ("C:/models/model", False),
            (r"C:\models\model", False),
            ("org/", False),
            ("/model", False),
            ("../model", False),
            ("~/model", False),
            ("a/b/c", False),
            ("org/mo del", False),
            ("org/Model-3.1_x", True),
        ],
    )
    def test_looks_like_hub_id(self, model: str, expected: bool) -> None:
        assert ModelSourceConfig(model).looks_like_hub_id() is expected

    @pytest.mark.parametrize("field", ["revision", "tokenizer", "tokenizer_revision", "cache_dir"])
    def test_rejects_non_string_text_fields(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be a string"):
            ModelSourceConfig("org/model", **{field: 1})  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize("field", ["format", "trust"])
    def test_rejects_raw_strings_for_enums(self, field: str) -> None:
        with pytest.raises(TypeError, match=field):
            ModelSourceConfig("org/model", **{field: "dummy"})  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize("field", ["offline", "reproducible"])
    def test_rejects_non_bool_flags(self, field: str) -> None:
        with pytest.raises(TypeError, match=f"{field} must be bool"):
            ModelSourceConfig("org/model", **{field: 1})  # type: ignore[arg-type]  # deliberately wrong type

    def test_tokenizer_path_prefers_explicit_tokenizer(self) -> None:
        assert ModelSourceConfig("org/model", tokenizer="org/tok").tokenizer_path == "org/tok"

    def test_fingerprint_tracks_every_field(self) -> None:
        base = ModelSourceConfig("org/model")
        assert base.fingerprint == ModelSourceConfig("org/model").fingerprint
        assert base.fingerprint != ModelSourceConfig("org/model", offline=True).fingerprint
        assert base.fingerprint != ModelSourceConfig("org/model", revision="main").fingerprint

    def test_effective_tokenizer_revision(self) -> None:
        shared = ModelSourceConfig("org/m", revision="v1", tokenizer_revision="v2")
        same = ModelSourceConfig("org/m", revision="v1", tokenizer="org/m", tokenizer_revision="v2")
        other = ModelSourceConfig(
            "org/m", revision="v1", tokenizer="org/t", tokenizer_revision="v2"
        )

        assert not shared.has_separate_tokenizer
        assert shared.effective_tokenizer_revision == "v1"
        assert same.effective_tokenizer_revision == "v1"
        assert other.has_separate_tokenizer
        assert other.effective_tokenizer_revision == "v2"

    def test_reproducible_requires_tokenizer_sha_for_separate_repo(self) -> None:
        sha = "a" * 40
        with pytest.raises(ValueError, match="tokenizer_revision"):
            ModelSourceConfig("org/m", revision=sha, tokenizer="org/t", reproducible=True)

        config = ModelSourceConfig(
            "org/m", revision=sha, tokenizer="org/t", tokenizer_revision=sha, reproducible=True
        )
        assert config.effective_tokenizer_revision == sha

    def test_reproducible_ignores_tokenizer_revision_when_shared(self) -> None:
        ModelSourceConfig("org/m", revision="a" * 40, reproducible=True)
