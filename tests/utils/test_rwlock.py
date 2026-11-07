from __future__ import annotations

import asyncio

import pytest

from ayaka.utils.rwlock import RWLock

pytestmark = pytest.mark.unit


class TestReaders:
    def test_readers_run_concurrently(self) -> None:
        async def scenario() -> int:
            lock = RWLock()
            active = 0
            peak = 0

            async def reader() -> None:
                nonlocal active, peak
                async with lock.reader_lock:
                    active += 1
                    peak = max(peak, active)
                    await asyncio.sleep(0.01)
                    active -= 1

            await asyncio.gather(reader(), reader(), reader())
            return peak

        assert asyncio.run(scenario()) == 3

    def test_reader_reentrancy(self) -> None:
        async def scenario() -> None:
            lock = RWLock()
            current = asyncio.current_task()
            assert current is not None
            async with lock.reader_lock, lock.reader_lock:
                assert lock._reader_tasks[current] == 2
            assert not lock._reader_tasks

        asyncio.run(scenario())

    def test_reader_context_manager_releases_on_error(self) -> None:
        async def scenario() -> None:
            lock = RWLock()
            with pytest.raises(RuntimeError, match="boom"):
                async with lock.reader_lock:
                    raise RuntimeError("boom")
            async with lock.writer_lock:
                pass

        asyncio.run(scenario())


class TestWriters:
    def test_writer_excludes_readers_and_other_writers(self) -> None:
        async def scenario() -> int:
            lock = RWLock()
            active = 0
            peak = 0

            async def writer() -> None:
                nonlocal active, peak
                async with lock.writer_lock:
                    active += 1
                    peak = max(peak, active)
                    await asyncio.sleep(0.01)
                    active -= 1

            await asyncio.gather(writer(), writer())
            return peak

        assert asyncio.run(scenario()) == 1

    def test_writer_reentrancy_and_nested_reader(self) -> None:
        async def scenario() -> None:
            lock = RWLock()
            current = asyncio.current_task()
            assert current is not None
            async with lock.writer_lock, lock.writer_lock, lock.reader_lock:
                assert lock._writer_depth == 2
                assert lock._reader_tasks[current] == 1
            assert lock._writer_depth == 0
            assert not lock._reader_tasks

        asyncio.run(scenario())

    def test_reader_cannot_upgrade_to_writer(self) -> None:
        async def scenario() -> None:
            lock = RWLock()
            async with lock.reader_lock:
                with pytest.raises(RuntimeError, match="upgrade"):
                    await lock.acquire_writer()

        asyncio.run(scenario())

    def test_release_without_ownership_raises(self) -> None:
        async def scenario() -> None:
            lock = RWLock()
            with pytest.raises(RuntimeError, match="does not hold"):
                await lock.release_writer()
            with pytest.raises(RuntimeError, match="does not hold"):
                await lock.release_reader()

        asyncio.run(scenario())


class TestWriterPreference:
    def test_a_queued_writer_blocks_later_readers(self) -> None:
        async def scenario() -> list[str]:
            lock = RWLock()
            events: list[str] = []
            release_first = asyncio.Event()

            async def first_reader() -> None:
                async with lock.reader_lock:
                    events.append("r1-enter")
                    await release_first.wait()
                    events.append("r1-exit")

            async def writer() -> None:
                async with lock.writer_lock:
                    events.append("w-enter")
                    events.append("w-exit")

            async def second_reader() -> None:
                async with lock.reader_lock:
                    events.append("r2-enter")

            first = asyncio.create_task(first_reader())
            await asyncio.sleep(0)
            queued_writer = asyncio.create_task(writer())
            await asyncio.sleep(0)
            second = asyncio.create_task(second_reader())
            await asyncio.sleep(0)
            release_first.set()
            await asyncio.gather(first, queued_writer, second)
            return events

        assert asyncio.run(scenario()) == [
            "r1-enter",
            "r1-exit",
            "w-enter",
            "w-exit",
            "r2-enter",
        ]


class TestCancellation:
    def test_cancelling_a_queued_writer_does_not_strand_waiting_writers(self) -> None:
        async def scenario() -> None:
            lock = RWLock()
            await lock.acquire_writer()
            queued = asyncio.create_task(lock.acquire_writer())
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert lock._waiting_writers == 0
            await lock.release_writer()
            async with lock.reader_lock:
                pass

        asyncio.run(scenario())
