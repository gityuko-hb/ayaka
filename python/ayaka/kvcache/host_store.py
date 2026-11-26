"""Host-RAM second-tier KV cache with asynchronous write-behind and race-free promotion."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ayaka.core.types import KVLayoutKind
from ayaka.utils.torch_memory import empty_host_tensor
from ayaka.utils.torch_utils import require_torch
from ayaka.utils.validation import require_int

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from ayaka.kvcache.storage.pool import KVPool
    from ayaka.kvcache.storage.spec import KVGroup

__all__ = ["HostKVStats", "HostKVStore"]


@dataclass(frozen=True, slots=True)
class HostKVStats:
    """Snapshot of the host-RAM KV block store."""

    capacity_blocks: int
    stored_blocks: int
    offloads: int
    promotions: int
    host_evictions: int
    inflight_offloads: int
    pinned_memory: bool


@dataclass(slots=True)
class _InFlightOffload:
    slot: int
    future: Future[None] | None = None
    event: Any = None


class HostKVStore:
    """Fixed-capacity LRU host-RAM tier for evicted KV cache blocks.

    Args:
        groups: KV cache groups matching the device pools.
        capacity_blocks: Maximum logical KV blocks cached in host RAM (identical across groups).
        pinned: Whether to request page-locked (pinned) host memory.
        allow_pageable: Allow falling back to pageable host memory if pinning fails.
        max_inflight_offloads: Maximum concurrent background offload tasks before blocking.
    """

    def __init__(
        self,
        groups: Sequence[KVGroup],
        capacity_blocks: int,
        *,
        pinned: bool = True,
        allow_pageable: bool = False,
        max_inflight_offloads: int = 8,
    ) -> None:
        if not groups:
            raise ValueError("HostKVStore requires at least one KV group")
        cap = require_int(capacity_blocks, "capacity_blocks", minimum=1)
        inflight_cap = require_int(max_inflight_offloads, "max_inflight_offloads", minimum=1)
        torch = require_torch(capability="host_kv_store")

        self._torch = torch
        self._groups = tuple(groups)
        self._capacity = cap
        self._max_inflight = inflight_cap
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ayaka-host-kv")
        self._streams: dict[int, Any] = {}

        k_hosts: list[Tensor] = []
        v_hosts: list[Tensor] = []
        all_pinned = True
        for group in self._groups:
            spec = group.spec
            if spec.layout is KVLayoutKind.NHD:
                shape = (
                    spec.num_layers,
                    cap,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_dim,
                )
            else:
                shape = (
                    spec.num_layers,
                    cap,
                    spec.num_kv_heads,
                    spec.block_size,
                    spec.head_dim,
                )
            k_buf, k_pin = empty_host_tensor(
                shape, spec.dtype.torch_dtype, pinned=pinned, allow_pageable=allow_pageable
            )
            v_buf, v_pin = empty_host_tensor(
                shape, spec.dtype.torch_dtype, pinned=pinned, allow_pageable=allow_pageable
            )
            k_hosts.append(k_buf)
            v_hosts.append(v_buf)
            all_pinned = all_pinned and k_pin and v_pin

        self._k_hosts = tuple(k_hosts)
        self._v_hosts = tuple(v_hosts)
        self._pinned = all_pinned
        self._free_slots: tuple[deque[int], ...] = tuple(deque(range(cap)) for _ in self._groups)
        self._slots: tuple[OrderedDict[bytes, int], ...] = tuple(
            OrderedDict() for _ in self._groups
        )
        self._inflight: tuple[dict[bytes, _InFlightOffload], ...] = tuple({} for _ in self._groups)
        self._offloads = 0
        self._promotions = 0
        self._host_evictions = 0
        self._closed = False

    @property
    def capacity_blocks(self) -> int:
        """Maximum blocks held in host RAM per group."""
        return self._capacity

    @property
    def pinned_memory(self) -> bool:
        """Whether all host KV backing tensors are page-locked."""
        return self._pinned

    def _cuda_stream(self, device: Any) -> Any:
        index = device.index or 0
        stream = self._streams.get(index)
        if stream is None:
            stream = self._torch.cuda.Stream(device=device)
            self._streams[index] = stream
        return stream

    def _reap_completed_locked(self, group_idx: int) -> None:
        done = [
            digest
            for digest, entry in self._inflight[group_idx].items()
            if (entry.future is not None and entry.future.done())
            or (entry.event is not None and entry.event.query())
        ]
        for digest in done:
            entry = self._inflight[group_idx].pop(digest)
            if entry.future is not None:
                entry.future.result()
            self._slots[group_idx][digest] = entry.slot
            self._slots[group_idx].move_to_end(digest)

    def _acquire_slot_locked(self, group_idx: int) -> int:
        self._reap_completed_locked(group_idx)
        free = self._free_slots[group_idx]
        if free:
            return free.popleft()
        slots = self._slots[group_idx]
        if slots:
            _, evicted_slot = slots.popitem(last=False)
            self._host_evictions += 1
            return evicted_slot
        # Every slot is currently inflight: wait on the oldest inflight entry.
        oldest_digest, oldest = next(iter(self._inflight[group_idx].items()))
        self._inflight[group_idx].pop(oldest_digest)
        self._wait_entry(oldest)
        self._host_evictions += 1
        return oldest.slot

    @staticmethod
    def _wait_entry(entry: _InFlightOffload) -> None:
        if entry.future is not None:
            entry.future.result()
        if entry.event is not None:
            entry.event.synchronize()

    def offload(self, group_idx: int, block_id: int, digest: bytes, pool: KVPool) -> bool:
        """Schedule asynchronous write-behind of an evicted block to host RAM."""
        wait_oldest: _InFlightOffload | None = None
        wait_digest: bytes | None = None
        with self._lock:
            if self._closed:
                return False
            self._reap_completed_locked(group_idx)
            if digest in self._slots[group_idx]:
                self._slots[group_idx].move_to_end(digest)
                return False
            if digest in self._inflight[group_idx]:
                return False
            total_inflight = sum(len(m) for m in self._inflight)
            if total_inflight >= self._max_inflight and self._inflight[group_idx]:
                wait_digest, wait_oldest = next(iter(self._inflight[group_idx].items()))
                self._inflight[group_idx].pop(wait_digest)
        if wait_oldest is not None and wait_digest is not None:
            self._wait_entry(wait_oldest)
            with self._lock:
                self._slots[group_idx][wait_digest] = wait_oldest.slot
                self._slots[group_idx].move_to_end(wait_digest)

        k_src, v_src = pool.block_tensors(block_id)
        is_cuda = pool.device.type == "cuda"

        with self._lock:
            if self._closed:
                return False
            if digest in self._slots[group_idx] or digest in self._inflight[group_idx]:
                return False
            slot = self._acquire_slot_locked(group_idx)
            if is_cuda:
                stream = self._cuda_stream(pool.device)
                stream.wait_stream(self._torch.cuda.current_stream(pool.device))
                with self._torch.cuda.stream(stream):
                    self._k_hosts[group_idx][:, slot].copy_(k_src, non_blocking=True)
                    self._v_hosts[group_idx][:, slot].copy_(v_src, non_blocking=True)
                event = stream.record_event()
                self._inflight[group_idx][digest] = _InFlightOffload(slot=slot, event=event)
            else:
                k_snap = k_src.detach().clone()
                v_snap = v_src.detach().clone()

                def _write(
                    g: int = group_idx,
                    target_slot: int = slot,
                    k_copy: Tensor = k_snap,
                    v_copy: Tensor = v_snap,
                ) -> None:
                    self._k_hosts[g][:, target_slot].copy_(k_copy)
                    self._v_hosts[g][:, target_slot].copy_(v_copy)

                future = self._executor.submit(_write)
                self._inflight[group_idx][digest] = _InFlightOffload(slot=slot, future=future)
            self._offloads += 1
        return True

    def has(self, digest: bytes) -> bool:
        """Whether ``digest`` is stored or currently completing offload across every group."""
        with self._lock:
            for g in range(len(self._groups)):
                self._reap_completed_locked(g)
                if digest not in self._slots[g] and digest not in self._inflight[g]:
                    return False
            return True

    def promote(self, digest: bytes, dst_block_ids: Sequence[int], pools: Sequence[KVPool]) -> bool:
        """Promote a cached block from host RAM back into target device blocks.

        Waits for any in-flight write-behind copy for ``digest`` before copying back,
        preventing read-before-write races.
        """
        if len(dst_block_ids) != len(self._groups) or len(pools) != len(self._groups):
            raise ValueError("dst_block_ids and pools must match number of groups")
        resolved_slots: list[int] = [-1] * len(self._groups)
        pending: list[tuple[int, _InFlightOffload]] = []
        with self._lock:
            for g in range(len(self._groups)):
                self._reap_completed_locked(g)
                if digest in self._slots[g]:
                    resolved_slots[g] = self._slots[g][digest]
                    self._slots[g].move_to_end(digest)
                elif digest in self._inflight[g]:
                    entry = self._inflight[g].pop(digest)
                    pending.append((g, entry))
                    resolved_slots[g] = entry.slot
                else:
                    return False
        for g, entry in pending:
            self._wait_entry(entry)
            with self._lock:
                self._slots[g][digest] = entry.slot
                self._slots[g].move_to_end(digest)
        for g, (dst_block_id, pool) in enumerate(zip(dst_block_ids, pools, strict=True)):
            slot = resolved_slots[g]
            pool.copy_block_from(
                dst_block_id,
                self._k_hosts[g][:, slot],
                self._v_hosts[g][:, slot],
                non_blocking=(pool.device.type == "cuda"),
            )
        with self._lock:
            self._promotions += 1
        return True

    def flush(self) -> None:
        """Wait for all in-flight asynchronous host offloads to complete."""
        with self._lock:
            pending = [
                (g, digest, entry)
                for g, inflight_map in enumerate(self._inflight)
                for digest, entry in list(inflight_map.items())
            ]
            for inflight_map in self._inflight:
                inflight_map.clear()
        for g, digest, entry in pending:
            self._wait_entry(entry)
            with self._lock:
                self._slots[g][digest] = entry.slot
                self._slots[g].move_to_end(digest)

    def close(self) -> None:
        """Drain pending offloads and shut down the background worker."""
        self.flush()
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True)

    def stats(self) -> HostKVStats:
        """Return a consistent snapshot of host store occupancy and activity."""
        with self._lock:
            for g in range(len(self._groups)):
                self._reap_completed_locked(g)
            stored = len(self._slots[0]) + len(self._inflight[0])
            inflight = sum(len(m) for m in self._inflight)
            return HostKVStats(
                capacity_blocks=self._capacity,
                stored_blocks=stored,
                offloads=self._offloads,
                promotions=self._promotions,
                host_evictions=self._host_evictions,
                inflight_offloads=inflight,
                pinned_memory=self._pinned,
            )
