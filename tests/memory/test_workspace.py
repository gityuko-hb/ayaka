"""The step workspace: aligned slices of one buffer, reset per step, refused when it runs out."""

from __future__ import annotations

from typing import Any

import pytest

from ayaka.core.errors import CapacityError

torch: Any = pytest.importorskip("torch")

from ayaka.memory.workspace import WORKSPACE_ALIGNMENT, Workspace  # noqa: E402

pytestmark = pytest.mark.unit


def test_slices_are_aligned_and_do_not_overlap() -> None:
    workspace = Workspace(4096, device="cpu")

    a = workspace.take((3,), torch.float32)
    b = workspace.take((2, 5), torch.int64)

    assert a.shape == (3,)
    assert b.shape == (2, 5)
    assert b.dtype == torch.int64
    assert (b.data_ptr() - a.data_ptr()) % WORKSPACE_ALIGNMENT == 0
    assert b.data_ptr() >= a.data_ptr() + 12
    assert workspace.used_bytes == WORKSPACE_ALIGNMENT + 80


def test_reset_hands_out_the_same_addresses_again() -> None:
    workspace = Workspace(4096, device="cpu")
    first = workspace.take((8,), torch.float16).data_ptr()

    workspace.reset()

    assert workspace.take((8,), torch.float16).data_ptr() == first
    assert workspace.used_bytes == 16


def test_the_high_water_mark_survives_a_reset() -> None:
    workspace = Workspace(4096, device="cpu")
    workspace.take((256,), torch.float32)
    workspace.reset()
    workspace.take((4,), torch.float32)

    assert workspace.high_water_bytes == 1024


def test_running_out_is_a_capacity_error_with_the_numbers() -> None:
    workspace = Workspace(1024, device="cpu")
    workspace.take((100,), torch.float32)

    with pytest.raises(CapacityError, match=r"need 4000 bytes, have 512, short 3488"):
        workspace.take((1000,), torch.float32)


def test_the_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="nbytes"):
        Workspace(0, device="cpu")
