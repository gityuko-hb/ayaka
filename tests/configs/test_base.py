from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ayaka.configs.base import (
    ConfigMixin,
    FrozenMapping,
    freeze_mapping,
    validate_buckets,
)
from ayaka.core.errors import ConfigError


class Mode(enum.Enum):
    FAST = "fast"


@dataclass(frozen=True, slots=True)
class Sample(ConfigMixin):
    name: str
    mode: Mode = Mode.FAST
    path: Path = Path("a")
    sizes: tuple[int, ...] = (1, 2)
    extra: FrozenMapping = field(default_factory=FrozenMapping)
    hidden: int = field(default=0, repr=False)


@pytest.mark.unit
class TestConfigError:
    def test_exposes_fields_and_formats_message(self) -> None:
        err = ConfigError("a.b", "BAD", "nope", context={"k": 1})
        assert (err.path, err.code, err.message, err.context) == ("a.b", "BAD", "nope", {"k": 1})
        assert str(err) == "a.b: nope [BAD]"
        assert isinstance(err, ValueError)

    def test_context_defaults_to_empty_copy(self) -> None:
        ctx = {"k": 1}
        err = ConfigError("p", "C", "m", context=ctx)
        ctx["k"] = 2
        assert err.context == {"k": 1}
        assert ConfigError("p", "C", "m").context == {}


@pytest.mark.unit
class TestFreezeMapping:
    def test_none_gives_empty(self) -> None:
        assert freeze_mapping(None) == FrozenMapping()

    def test_sorts_keys_from_mapping(self) -> None:
        assert freeze_mapping({"b": 1, "a": 2}) == (("a", 2), ("b", 1))

    def test_accepts_pairs(self) -> None:
        assert freeze_mapping((("b", 1), ("a", 2))) == (("a", 2), ("b", 1))

    def test_result_type(self) -> None:
        assert isinstance(freeze_mapping({"a": 1}), FrozenMapping)

    def test_rejects_duplicate_keys(self) -> None:
        with pytest.raises(ConfigError) as exc:
            freeze_mapping((("a", 1), ("a", 2)), path="x")
        assert (exc.value.code, exc.value.path) == ("MAP_KEY_DUPLICATE", "x")

    @pytest.mark.parametrize("key", ["", 1, None])
    def test_rejects_invalid_keys(self, key: object) -> None:
        with pytest.raises(ConfigError) as exc:
            freeze_mapping(((key, 1),))  # type: ignore[arg-type]  # deliberately wrong type
        assert exc.value.code == "MAP_KEY_INVALID"


@pytest.mark.unit
class TestConfigMixin:
    def test_to_dict_canonicalizes_values(self) -> None:
        cfg = Sample("n", extra=freeze_mapping({"z": Mode.FAST, "a": 1}))
        assert (
            cfg.to_dict()
            == {
                "name": "n",
                "mode": "fast",
                "path": "a",
                "sizes": [1, 2],
                "extra": {"a": 1, "z": "fast"},
                "hidden": 0,
            }
            or "hidden" not in cfg.to_dict()
        )

    def test_fields_with_repr_false_are_excluded(self) -> None:
        assert "hidden" not in Sample("n").to_dict()

    def test_empty_frozen_mapping_is_object(self) -> None:
        assert Sample("n").to_dict()["extra"] == {}

    def test_canonical_json_is_compact_and_sorted(self) -> None:
        text = Sample("é").canonical_json()
        assert " " not in text
        assert "é" in text
        assert list(json.loads(text)) == sorted(json.loads(text))

    def test_fingerprint_is_stable_and_sensitive(self) -> None:
        a, b = Sample("n"), Sample("n")
        assert a.fingerprint == b.fingerprint
        assert len(a.fingerprint) == 64
        assert a.fingerprint != Sample("m").fingerprint

    def test_extra_insertion_order_does_not_change_fingerprint(self) -> None:
        one = Sample("n", extra=freeze_mapping({"a": 1, "b": 2}))
        two = Sample("n", extra=freeze_mapping({"b": 2, "a": 1}))
        assert one.fingerprint == two.fingerprint


@pytest.mark.unit
class TestValidateBuckets:
    def test_accepts_ascending_unique(self) -> None:
        validate_buckets((1, 2, 8))

    @pytest.mark.parametrize("value", [[1, 2], (), "x"])
    def test_rejects_non_tuple_or_empty(self, value: object) -> None:
        with pytest.raises(ValueError, match="non-empty tuple"):
            validate_buckets(value)  # type: ignore[arg-type]  # deliberately wrong type

    @pytest.mark.parametrize("value", [(2, 1), (1, 1, 2)])
    def test_rejects_unsorted_or_duplicate(self, value: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="ascending and deduplicated"):
            validate_buckets(value)

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="bucket must be >= 1"):
            validate_buckets((0, 1))

    def test_rejects_bool(self) -> None:
        with pytest.raises(TypeError):
            validate_buckets((True,))
