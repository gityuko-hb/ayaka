from __future__ import annotations

import threading

import pytest

from ayaka.core.errors import CancelledError
from ayaka.utils.cancellation import CancellationToken


@pytest.mark.unit
class TestCancellationToken:
    def test_starts_active(self) -> None:
        token = CancellationToken()

        assert not token.is_cancelled
        assert token.reason is None
        token.raise_if_cancelled()

    def test_cancel_records_the_reason_and_raises(self) -> None:
        token = CancellationToken()
        token.cancel("client disconnected")

        assert token.is_cancelled
        assert token.reason == "client disconnected"
        with pytest.raises(CancelledError, match="client disconnected"):
            token.raise_if_cancelled()

    def test_first_reason_wins(self) -> None:
        token = CancellationToken()
        token.cancel("first")
        token.cancel("second")

        assert token.reason == "first"

    def test_default_reason(self) -> None:
        token = CancellationToken()
        token.cancel()

        assert token.reason == "cancelled"

    def test_concurrent_cancels_keep_exactly_one_reason(self) -> None:
        token = CancellationToken()
        barrier = threading.Barrier(8)

        def cancel(index: int) -> None:
            barrier.wait()
            token.cancel(f"reason-{index}")

        threads = [threading.Thread(target=cancel, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert token.reason is not None
        assert token.reason.startswith("reason-")
