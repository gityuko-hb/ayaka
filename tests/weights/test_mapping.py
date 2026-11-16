"""Name translation and the match between a checkpoint and a model's declared parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ayaka.core.errors import ModelLoadError
from ayaka.utils.dtypes import DType
from ayaka.weights.manifest import WeightManifest
from ayaka.weights.mapping import WeightMapping
from ayaka.weights.spec import TensorSpec
from safetensors_files import filled, write_safetensors

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from safetensors_files import TensorData

pytestmark = pytest.mark.unit


def _manifest(tmp_path: Path, tensors: Mapping[str, TensorData]) -> WeightManifest:
    write_safetensors(tmp_path / "model.safetensors", tensors)
    return WeightManifest.from_directory(tmp_path)


def _spec(name: str, *shape: int, dtype: DType = DType.FP32) -> TensorSpec:
    return TensorSpec(name, shape, dtype)


class TestTarget:
    def test_a_name_with_no_rule_is_its_own_target(self) -> None:
        assert WeightMapping().target("layers.0.weight") == "layers.0.weight"

    def test_rename_rewrites_with_groups(self) -> None:
        mapping = WeightMapping(renames=((r"model\.layers\.(\d+)\.(.*)", r"blocks.\1.\2"),))

        assert mapping.target("model.layers.12.mlp.up.weight") == "blocks.12.mlp.up.weight"

    def test_rename_supports_named_groups(self) -> None:
        mapping = WeightMapping(renames=((r"(?P<n>\d+)\.w", r"layer\g<n>.weight"),))

        assert mapping.target("7.w") == "layer7.weight"

    def test_a_pattern_must_match_the_whole_name(self) -> None:
        mapping = WeightMapping(renames=((r"layers", "blocks"),), ignore=(r"bias",))

        assert mapping.target("layers.0.weight") == "layers.0.weight"
        assert mapping.target("layers.0.bias") == "layers.0.bias"

    def test_the_first_matching_rename_wins(self) -> None:
        mapping = WeightMapping(renames=((r"a\..*", "first"), (r"a\.b", "second")))

        assert mapping.target("a.b") == "first"

    def test_a_renamed_name_is_not_renamed_again(self) -> None:
        mapping = WeightMapping(renames=((r"a", "b"), (r"b", "c")))

        assert mapping.target("a") == "b"

    def test_ignore_beats_rename(self) -> None:
        mapping = WeightMapping(renames=((r"x\..*", "y"),), ignore=(r"x\.mask",))

        assert mapping.target("x.mask") is None
        assert mapping.target("x.other") == "y"

    def test_an_invalid_pattern_fails_when_the_mapping_is_built(self) -> None:
        with pytest.raises(Exception, match="unterminated character set"):
            WeightMapping(ignore=("[",))
        with pytest.raises(Exception, match="missing \\)"):
            WeightMapping(renames=(("(", "x"),))

    def test_a_replacement_naming_a_missing_group_says_which_rule(self) -> None:
        mapping = WeightMapping(renames=((r"a", r"\1"),))

        with pytest.raises(ValueError, match=r"rename 'a' -> '\\\\1' cannot rewrite 'a'"):
            mapping.target("a")

    def test_is_hashable_and_compares_by_value(self) -> None:
        assert WeightMapping(ignore=("a",)) == WeightMapping(ignore=("a",))
        assert len({WeightMapping(ignore=("a",)), WeightMapping(ignore=("a",))}) == 1


class TestBind:
    def test_returns_the_stored_tensors_in_the_order_of_the_specs(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, {"b": filled("F32", (2,)), "a": filled("F32", (3,))})

        bound = WeightMapping().bind(manifest, [_spec("a", 3), _spec("b", 2)])

        assert list(bound) == ["a", "b"]
        assert bound["a"] is manifest["a"]

    def test_applies_renames_and_ignores(self, tmp_path: Path) -> None:
        manifest = _manifest(
            tmp_path,
            {
                "transformer.w": filled("F32", (2,)),
                "transformer.attn.bias": filled("F32", (4,)),
                "lm_head.weight": filled("F32", (2,)),
            },
        )
        mapping = WeightMapping(
            renames=((r"transformer\.(.*)", r"\1"),),
            ignore=(r".*attn\.bias", r"lm_head\.weight"),
        )

        bound = mapping.bind(manifest, [_spec("w", 2)])

        assert bound["w"] is manifest["transformer.w"]

    def test_a_different_stored_dtype_is_not_a_mismatch(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, {"w": filled("F16", (2,))})

        bound = WeightMapping().bind(manifest, [_spec("w", 2, dtype=DType.FP32)])

        assert bound["w"].dtype is DType.FP16

    def test_a_model_with_no_parameters_matches_only_an_ignored_checkpoint(
        self, tmp_path: Path
    ) -> None:
        manifest = _manifest(tmp_path, {"w": filled("F32", (1,))})

        assert WeightMapping(ignore=("w",)).bind(manifest, []) == {}
        with pytest.raises(ModelLoadError, match="unexpected 1 tensors"):
            WeightMapping().bind(manifest, [])

    def test_every_problem_of_every_kind_is_reported_at_once(self, tmp_path: Path) -> None:
        manifest = _manifest(
            tmp_path,
            {
                "ok": filled("F32", (2,)),
                "wrong": filled("F32", (2, 3)),
                "extra": filled("F32", (1,)),
                "also_extra": filled("F32", (1,)),
                "dup_a": filled("F32", (1,)),
                "dup_b": filled("F32", (1,)),
            },
        )
        mapping = WeightMapping(renames=((r"dup_[ab]", "dup"),))
        specs = [_spec("ok", 2), _spec("wrong", 3, 2), _spec("absent", 1), _spec("dup", 1)]

        with pytest.raises(ModelLoadError) as excinfo:
            mapping.bind(manifest, specs)

        message = str(excinfo.value)
        assert "missing 1 parameters: absent" in message
        assert "unexpected 2 tensors that no parameter claims: also_extra, extra" in message
        assert "1 tensors with the wrong shape: wrong is [2, 3], expected [3, 2]" in message
        assert "1 parameters claimed by several tensors: dup <- dup_a, dup_b" in message
        assert excinfo.value.context == {
            "missing": ["absent"],
            "unexpected": ["also_extra", "extra"],
            "wrong_shape": {"wrong": ((3, 2), (2, 3))},
            "ambiguous": {"dup": ["dup_a", "dup_b"]},
        }
        assert str(tmp_path) in message

    def test_missing_parameters_are_listed_in_the_order_the_model_declares_them(
        self, tmp_path: Path
    ) -> None:
        manifest = _manifest(tmp_path, {"w": filled("F32", (1,))})
        specs = [_spec("z", 1), _spec("w", 1), _spec("a", 1)]

        with pytest.raises(ModelLoadError) as excinfo:
            WeightMapping().bind(manifest, specs)

        assert excinfo.value.context["missing"] == ["z", "a"]

    def test_a_long_list_is_cut_in_the_message_but_complete_in_the_context(
        self, tmp_path: Path
    ) -> None:
        manifest = _manifest(tmp_path, {"w": filled("F32", (1,))})
        specs = [_spec("w", 1), *(_spec(f"p{index:02d}", 1) for index in range(30))]

        with pytest.raises(
            ModelLoadError, match=r"missing 30 parameters: p00.*p11 and 18 more"
        ) as e:
            WeightMapping().bind(manifest, specs)

        assert len(cast("list[str]", e.value.context["missing"])) == 30

    def test_an_unexpected_tensor_is_named_by_its_checkpoint_name(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, {"old.name": filled("F32", (1,))})
        mapping = WeightMapping(renames=((r"old\.(.*)", r"new.\1"),))

        with pytest.raises(ModelLoadError, match=r"no parameter claims: old\.name"):
            mapping.bind(manifest, [])

    def test_a_scalar_parameter(self, tmp_path: Path) -> None:
        manifest = _manifest(tmp_path, {"s": filled("F32", ())})

        assert WeightMapping().bind(manifest, [_spec("s")])["s"].shape == ()
