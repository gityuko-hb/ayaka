"""How checkpoint tensor names map to the parameters of a model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

from ayaka.core.errors import ModelLoadError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ayaka.weights.manifest import WeightManifest
    from ayaka.weights.spec import TensorSpec
    from ayaka.weights.tensor import TensorRef

__all__ = ["WeightMapping"]

#: Names listed in full in an error message; all of them are in ``error.context``.
_LISTED = 12


@cache
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


@dataclass(frozen=True, slots=True)
class WeightMapping:
    """A model's rules for reading a checkpoint, as data.

    A checkpoint tensor name becomes a parameter name in three steps:

    1. A name that fully matches a pattern in ``ignore`` is dropped on purpose.
       Use it for tensors the model does not have (cached attention masks, a
       tied output projection).
    2. Otherwise the first pair in ``renames`` whose pattern fully matches the
       name rewrites it; the replacement may use the pattern's groups
       (``\\1``, ``\\g<name>``).
    3. Otherwise the name is the parameter name.

    A name that ends in one of the stored suffixes of ``suffixes`` (a quantized
    checkpoint's ``q_proj.weight_scale_inv``) is renamed as the ``weight`` it sits
    next to, then given the parameter suffix (``attn.q_proj.weight_scale``), so a
    model's renames, written for ``weight`` and ``bias``, cover its scales too.

    Patterns are Python regular expressions. A name that is neither ignored nor
    claimed by a parameter is an error, never skipped silently.

    Attributes:
        renames: ``(pattern, replacement)`` pairs, tried in order.
        ignore: Patterns of checkpoint names to drop.
        suffixes: ``(stored suffix, parameter suffix)`` pairs of tensors stored next to a
            ``weight`` under the same prefix.

    Raises:
        ValueError: If a pattern is not a valid regular expression.

    Example:
        >>> mapping = WeightMapping(
        ...     renames=((r"transformer\\.(.*)", r"\\1"),), ignore=(r".*\\.attn\\.bias",)
        ... )
        >>> mapping.target("transformer.h.0.mlp.c_fc.weight")
        'h.0.mlp.c_fc.weight'
        >>> mapping.target("transformer.h.0.attn.bias") is None
        True
    """

    renames: tuple[tuple[str, str], ...] = ()
    ignore: tuple[str, ...] = ()
    suffixes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for pattern in (*(pattern for pattern, _ in self.renames), *self.ignore):
            try:
                _compiled(pattern)
            except re.error as exc:
                raise ValueError(f"invalid pattern {pattern!r}: {exc}") from exc

    def target(self, name: str) -> str | None:
        """Return the parameter name for a checkpoint tensor, or ``None`` if it is ignored.

        Raises:
            ValueError: If a replacement refers to a group its pattern lacks.
        """
        if any(_compiled(pattern).fullmatch(name) for pattern in self.ignore):
            return None
        for stored, parameter in self.suffixes:
            if name.endswith(f".{stored}"):
                renamed = self._rename(f"{name.removesuffix(stored)}weight")
                if renamed.endswith(".weight"):
                    return f"{renamed.removesuffix('weight')}{parameter}"
                return name
        return self._rename(name)

    def _rename(self, name: str) -> str:
        for pattern, replacement in self.renames:
            if (match := _compiled(pattern).fullmatch(name)) is not None:
                try:
                    return match.expand(replacement)
                except (re.error, IndexError) as exc:
                    raise ValueError(
                        f"rename {pattern!r} -> {replacement!r} cannot rewrite {name!r}: {exc}"
                    ) from exc
        return name

    def bind(
        self, manifest: WeightManifest, specs: Iterable[TensorSpec]
    ) -> Mapping[str, TensorRef]:
        """Match a checkpoint to the parameters a model declares, before reading any tensor.

        Every checkpoint tensor is translated with :meth:`target` and the result
        is compared with ``specs``. Names and shapes must agree; dtypes need not,
        because converting a stored dtype to the model's is the loader's job.

        Args:
            manifest: The checkpoint.
            specs: The parameters of the model.

        Returns:
            Parameter name to the stored tensor that fills it, in the order of ``specs``.

        Raises:
            ayaka.core.errors.ModelLoadError: If the two do not agree. One error
                lists every problem of every kind, so a wrong checkpoint is
                diagnosed in one run: parameters with no tensor (``missing``),
                tensors no parameter claims (``unexpected``), parameters whose
                stored shape differs (``wrong_shape``) and parameters claimed by
                several tensors (``ambiguous``). The same lists are in
                ``error.context``, in full.
        """
        wanted = {spec.name: spec for spec in specs}
        bound: dict[str, TensorRef] = {}
        claimed_by: dict[str, list[str]] = {}
        unexpected: list[str] = []
        for key, ref in manifest.items():
            parameter = self.target(key)
            if parameter is None:
                continue
            if parameter not in wanted:
                unexpected.append(key)
                continue
            claimed_by.setdefault(parameter, []).append(key)
            bound.setdefault(parameter, ref)

        missing = [name for name in wanted if name not in bound]
        ambiguous = {name: keys for name, keys in claimed_by.items() if len(keys) > 1}
        wrong_shape = {
            name: (wanted[name].shape, ref.shape)
            for name, ref in bound.items()
            if ref.shape != wanted[name].shape
        }
        if missing or unexpected or wrong_shape or ambiguous:
            raise _disagreement(manifest, missing, sorted(unexpected), wrong_shape, ambiguous)
        return {name: bound[name] for name in wanted}


def _disagreement(
    manifest: WeightManifest,
    missing: list[str],
    unexpected: list[str],
    wrong_shape: dict[str, tuple[tuple[int, ...], tuple[int, ...]]],
    ambiguous: dict[str, list[str]],
) -> ModelLoadError:
    parts: list[str] = []
    if missing:
        parts.append(f"missing {len(missing)} parameters: {_listed(missing)}")
    if unexpected:
        parts.append(
            f"unexpected {len(unexpected)} tensors that no parameter claims: {_listed(unexpected)}"
        )
    if wrong_shape:
        shapes = [
            f"{name} is {list(found)}, expected {list(expected)}"
            for name, (expected, found) in wrong_shape.items()
        ]
        parts.append(f"{len(shapes)} tensors with the wrong shape: {_listed(shapes)}")
    if ambiguous:
        tensors = [f"{name} <- {', '.join(keys)}" for name, keys in ambiguous.items()]
        parts.append(f"{len(tensors)} parameters claimed by several tensors: {_listed(tensors)}")
    return ModelLoadError(
        f"checkpoint {manifest.root} does not match the model: " + "; ".join(parts),
        context={
            "missing": missing,
            "unexpected": unexpected,
            "wrong_shape": wrong_shape,
            "ambiguous": ambiguous,
        },
    )


def _listed(items: list[str]) -> str:
    shown = ", ".join(items[:_LISTED])
    return shown if len(items) <= _LISTED else f"{shown} and {len(items) - _LISTED} more"
