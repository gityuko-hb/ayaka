"""The architecture registry: lookup by the name ``config.json`` gives, and its errors."""

from __future__ import annotations

from typing import Any

import pytest

from ayaka.core.errors import ModelLoadError
from ayaka.models import registry

torch: Any = pytest.importorskip("torch")

from ayaka.models.base import CausalLM  # noqa: E402  # needs the importorskip above
from ayaka.models.gpt2 import GPT2LMHeadModel  # noqa: E402
from ayaka.models.registry import register_model, resolve_model  # noqa: E402

pytestmark = pytest.mark.unit


class _Stub(CausalLM):
    @classmethod
    def weight_mapping(cls, config: Any) -> Any:
        raise NotImplementedError

    def forward(self, token_ids: Any, context: Any) -> Any:
        raise NotImplementedError

    def compute_logits(self, hidden_rows: Any) -> Any:
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_registered", {})


def test_gpt2_is_resolved_by_its_architecture_name() -> None:
    assert resolve_model("GPT2LMHeadModel") is GPT2LMHeadModel


def test_an_unknown_architecture_lists_what_is_supported() -> None:
    with pytest.raises(
        ModelLoadError, match=r"'LlamaForCausalLM' is not supported.*GPT2LMHeadModel"
    ) as error:
        resolve_model("LlamaForCausalLM")

    assert error.value.context["architecture"] == "LlamaForCausalLM"
    assert "GPT2LMHeadModel" in error.value.context["supported"]  # type: ignore[operator]  # a list


def test_a_registered_class_is_resolved_and_listed_in_the_error() -> None:
    register_model("StubForCausalLM", "StubAlias")(_Stub)

    assert resolve_model("StubForCausalLM") is _Stub
    assert resolve_model("StubAlias") is _Stub
    with pytest.raises(ModelLoadError, match=r"StubAlias.*StubForCausalLM"):
        resolve_model("Nope")


def test_register_model_works_as_a_decorator_and_returns_the_class() -> None:
    @register_model("DecoratedForCausalLM")
    class Decorated(_Stub):
        pass

    assert resolve_model("DecoratedForCausalLM") is Decorated


def test_a_name_cannot_be_registered_twice() -> None:
    register_model("StubForCausalLM")(_Stub)

    with pytest.raises(ValueError, match=r"already registered.*StubForCausalLM"):
        register_model("StubForCausalLM")(_Stub)


def test_a_name_of_a_shipped_model_cannot_be_taken() -> None:
    with pytest.raises(ValueError, match=r"already registered.*GPT2LMHeadModel"):
        register_model("GPT2LMHeadModel")(_Stub)


def test_a_failed_registration_registers_none_of_its_names() -> None:
    register_model("Taken")(_Stub)

    with pytest.raises(ValueError, match="already registered"):
        register_model("Fresh", "Taken")(_Stub)

    with pytest.raises(ModelLoadError):
        resolve_model("Fresh")


def test_only_a_causal_lm_can_be_registered() -> None:
    with pytest.raises(ValueError, match="not a CausalLM subclass"):
        register_model("Bad")(dict)  # type: ignore[type-var]  # that is the test


def test_a_name_is_required_and_must_not_be_blank() -> None:
    with pytest.raises(ValueError, match="at least one"):
        register_model()
    with pytest.raises(ValueError, match="architecture"):
        register_model("  ")
