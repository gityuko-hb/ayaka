"""Tests marked ``gpu`` are skipped, with the reason in the report, where no CUDA GPU is usable.

Hosted CI deselects them with ``-m "not gpu"``; this covers a developer machine without a GPU
running the whole suite. Setting ``AYAKA_REQUIRE_GPU=1`` turns that skip into a failure, so a run
that is meant to exercise the GPU cannot pass by skipping it.
"""

from __future__ import annotations

import os
from functools import cache

import pytest

_REQUIRE = "AYAKA_REQUIRE_GPU"


@cache
def _gpu_unavailable() -> str | None:
    """Why ``gpu`` tests cannot run here, or ``None`` when they can."""
    try:
        import torch
    except ImportError as exc:
        return f"torch is not installed ({exc})"
    if not torch.cuda.is_available():
        return f"no CUDA device is usable by torch {torch.__version__}"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    gpu_items = [item for item in items if item.get_closest_marker("gpu") is not None]
    if not gpu_items:
        return
    reason = _gpu_unavailable()
    if reason is None:
        return
    if os.environ.get(_REQUIRE) == "1":
        for item in gpu_items:
            item.add_marker(pytest.mark.usefixtures("_fail_without_gpu"))
        return
    for item in gpu_items:
        item.add_marker(pytest.mark.skip(reason=f"gpu test: {reason}"))


@pytest.fixture
def _fail_without_gpu() -> None:
    reason = _gpu_unavailable()
    if reason is not None:
        pytest.fail(f"{_REQUIRE}=1 but {reason}", pytrace=False)
