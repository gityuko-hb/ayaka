"""The models package: lazy exports that resolve, and no torch until a model is used."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_every_public_name_resolves_and_is_listed_by_dir() -> None:
    pytest.importorskip("torch")
    package = importlib.import_module("ayaka.models")

    missing = [name for name in package.__all__ if not hasattr(package, name)]

    assert missing == []
    assert set(package.__all__) <= set(dir(package))


def test_the_public_names_are_the_ones_the_tree_lists() -> None:
    package = importlib.import_module("ayaka.models")

    assert sorted(package.__all__) == [
        "CausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "GPT2LMHeadModel",
        "Gemma4ForConditionalGeneration",
        "GptOssForCausalLM",
        "LlamaForCausalLM",
        "MiMoForCausalLM",
        "OlmoeForCausalLM",
        "Phi3ForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "register_model",
        "resolve_model",
    ]


@pytest.mark.parametrize("module", ["ayaka", "ayaka.models", "ayaka.models.registry"])
def test_importing_does_not_import_torch_or_transformers(module: str) -> None:
    program = (
        "import sys\n"
        f"import {module}\n"
        "import ayaka.models as models\n"
        "models.__all__\n"
        "dir(models)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'transformers' not in sys.modules, 'transformers was imported'\n"
    )

    result = subprocess.run(  # noqa: S603  # fixed argv, no input from outside
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
