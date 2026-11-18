# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opt-in save pipeline: one sender thread per worker makes every native call.

A save's copy thread packs batches into transfer slots while the sender puts
the previous one; ``capacity`` batches at most are preparing or in flight. A
save reports its terminal only after :meth:`Sender.drain_request` saw every
put of it return.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from atom.kv_transfer.offload.kv_store.client import CallRefused, KVStoreClient

if TYPE_CHECKING:
    from atom.kv_transfer.offload.kv_store.pool import Slot, TransferSlotPool
    from atom.kv_transfer.offload.kv_store.worker import KVStoreOffloadConnector
    from atom.kv_transfer.offload.metadata import LMCacheReqMeta

MAX_BATCH_CHUNKS = 64


class StreamStopped(RuntimeError):
    """The sender is closed, or the save was cancelled or already failed."""


class DuplicateSave(RuntimeError):
    """A save operation is already active on this sender."""


def check_batch_chunks(batch_chunks: int) -> None:
    if type(batch_chunks) is not int or not 1 <= batch_chunks <= MAX_BATCH_CHUNKS:
        raise ValueError(f"Batch chunks must be an integer in [1, {MAX_BATCH_CHUNKS}]")


def stream_chunks(segments, batch_chunks: int) -> Iterator:
    """Split each ``(run, held, keys, lookup_ms)`` into batches, highest first.

    The lookup's answer is reused: no second ``exists`` call.
    """
    check_batch_chunks(batch_chunks)
    for segment, held, keys, lookup_ms in segments:
        high = segment.stop
        while high > segment.start:
            low = max(segment.start, high - batch_chunks)
            offset = low - segment.start
            yield range(low, high), held, keys[offset : offset + high - low], lookup_ms
            lookup_ms = None
            high = low


class SettlementTransaction:
    """The releases and quarantines a settlement hook asks for, applied together.

    The slots stay leased until the hook has returned; ``commit`` then applies
    every action. A slot whose action did not apply is in :meth:`remaining`.
    """

    def __init__(self, pool: TransferSlotPool, slots: list[Slot]) -> None:
        self.pool = pool
        self.original = list(slots)
        self.actions: list[tuple[str, list[Slot], str]] = []
        self.applied: set[int] = set()

    def release(self, slots: list[Slot]) -> None:
        self.actions.append(("release", list(slots), ""))

    def quarantine(self, slots: list[Slot], *, reason: str = "") -> None:
        self.actions.append(("quarantine", list(slots), reason))

    def commit(self) -> None:
        touched = [id(slot) for _, slots, _ in self.actions for slot in slots]
        if len(touched) != len(set(touched)) or set(touched) != {
            id(slot) for slot in self.original
        }:
            raise RuntimeError("settlement must partition every leased slot once")
        for method, slots, reason in self.actions:
            for slot in slots:
                if method == "release":
                    self.pool.release([slot])
                else:
                    self.pool.quarantine([slot], reason=reason)
                self.applied.add(id(slot))

    def remaining(self) -> list[Slot]:
        return [slot for slot in self.original if id(slot) not in self.applied]


@dataclass
class Ticket:
    request: LMCacheReqMeta
    run: range
    keys: list[str]
    slots: list[Slot]
    problem: str | None = None
    native_ms: float = 0.0
    stored_chunks: int = 0
    returned_chunks: int = 0
    transaction: SettlementTransaction | None = None


@dataclass
class RequestResult:
    batch_chunks: int
    outstanding: int = 0
    returned_chunks: int = 0
    stored_chunks: int = 0
    native_ms: float = 0.0
    problem: str | None = None


@dataclass
class Lookup:
    client: KVStoreClient
    keys: list[str]
    done: threading.Event = field(default_factory=threading.Event)
    result: list[int] | None = None
    error: Exception | None = None


class _PreparingFlag(threading.local):
    """Whether this thread holds a credit for a batch it is preparing."""

    active = False


class Sender:
    def __init__(self, owner: KVStoreOffloadConnector, capacity: int) -> None:
        self.owner = owner
        self.capacity = capacity
        self.credits = threading.BoundedSemaphore(capacity)
        self.jobs: queue.Queue[Ticket | Lookup | None] = queue.Queue()
        self.local = _PreparingFlag()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.pending = 0
        self.closed = False
        self.stop_sent = False
        self.failed: dict[Any, str] = {}
        self.cancelled: set[Any] = set()
        self.results: dict[Any, RequestResult] = {}
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="store-save-sender"
        )
        self.thread.start()

    @staticmethod
    def _key(req: LMCacheReqMeta) -> Any:
        return req.save_operation

    def _stop_reason(self, key: Any, *, closing: bool = True) -> str | None:
        """Why batches of ``key`` stop; ``closing=False`` ignores a close, which
        still lets an accepted batch drain."""
        with self.lock:
            if closing and self.closed:
                return "closed"
            if key in self.failed:
                return self.failed[key]
            return "cancelled" if key in self.cancelled else None

    def begin_request(self, req: LMCacheReqMeta, batch_chunks: int) -> None:
        with self.lock:
            if self._key(req) in self.results:
                raise DuplicateSave("Save operation already active")
            self.results[self._key(req)] = RequestResult(batch_chunks)

    def batch_size(self, req: LMCacheReqMeta) -> int:
        with self.lock:
            return self.results[self._key(req)].batch_chunks

    def start_lookup(self, client: KVStoreClient, keys: list[str]) -> Lookup:
        """Queue ``keys``' lookup behind earlier jobs. Lookups take no credit."""
        lookup = Lookup(client, list(keys))
        with self.lock:
            if self.closed:
                raise StreamStopped("closed")
            self.jobs.put(lookup)
        return lookup

    @staticmethod
    def finish_lookup(lookup: Lookup) -> list[int]:
        lookup.done.wait()
        if lookup.error is not None:
            raise lookup.error
        assert lookup.result is not None
        return lookup.result

    def reserve(self, req: LMCacheReqMeta, nslots: int) -> None:
        """Take one credit for a batch the calling thread is about to prepare."""
        assert not self.local.active
        assert 1 <= nslots <= self.batch_size(req)
        while True:
            reason = self._stop_reason(self._key(req))
            if reason is not None:
                raise StreamStopped(reason)
            if self.credits.acquire(timeout=0.01):
                break
        with self.lock:
            self.local.active = True
            self.pending += 1
        reason = self._stop_reason(self._key(req))
        if reason is not None:
            self.release_preparing()
            raise StreamStopped(reason)

    def release_preparing(self) -> None:
        """Give back the credit of a batch that will not be submitted."""
        if self.local.active:
            self.local.active = False
            self._release_credit()

    def _release_credit(self) -> None:
        with self.condition:
            self.pending -= 1
            self.condition.notify_all()
        self.credits.release()

    def submit(
        self, req: LMCacheReqMeta, run: range, keys: list[str], slots: list[Slot]
    ) -> None:
        """Queue a packed batch; its GPU copy has returned, so a close may not drop it."""
        assert self.local.active and len(run) == len(slots)
        ticket = Ticket(req, run, keys, slots)
        with self.lock:
            self.results[self._key(req)].outstanding += 1
            self.local.active = False
            # A close may begin during the pack. The sender still drains this
            # accepted, GPU-complete batch before its stop sentinel, but does
            # not put it.
            if self.closed:
                self.cancelled.add(self._key(req))
            self.jobs.put(ticket)

    def _safe_distrust(self, why: str, ticket: Ticket) -> None:
        # `_distrust_store` refuses local loads before it publishes the taint;
        # whatever it raises must not keep the sender from draining.
        try:
            self.owner._distrust_store(why)
        except BaseException as exc:  # noqa: BLE001 - preserve native drain
            ticket.problem = f"{ticket.problem or why}; distrust failed: {exc!r}"

    def _run(self) -> None:
        while True:
            item = self.jobs.get()
            try:
                if item is None:
                    return
                if isinstance(item, Lookup):
                    try:
                        item.result = item.client.exists(item.keys)
                    except Exception as exc:  # noqa: BLE001 - returned to the waiter
                        item.error = exc
                    finally:
                        item.done.set()
                else:
                    self._put(item)
            finally:
                self.jobs.task_done()

    def _put(self, t: Ticket) -> None:
        pool, client, _ = self.owner._registered()
        try:
            reason = self._stop_reason(self._key(t.request), closing=False)
            if reason is not None:
                # Submitted only after the GPU copy returned: nothing reads these.
                pool.release(t.slots)
                t.problem = reason
            else:
                self._put_native(t, pool, client)
        except BaseException as exc:  # noqa: BLE001 - retain unsettled slots
            t.problem = t.problem or f"settlement exception: {exc!r}"
            self._quarantine_remaining(t, pool)
            self._safe_distrust(repr(exc), t)
        finally:
            with self.condition:
                state = self.results[self._key(t.request)]
                state.outstanding -= 1
                state.returned_chunks += t.returned_chunks
                state.stored_chunks += t.stored_chunks
                state.native_ms += t.native_ms
                if t.problem:
                    self.failed[self._key(t.request)] = t.problem
                    state.problem = state.problem or t.problem
                self.condition.notify_all()
            self._release_credit()

    def _put_native(
        self, t: Ticket, pool: TransferSlotPool, client: KVStoreClient
    ) -> None:
        call_start = time.perf_counter()
        try:
            answer = self.owner._put_chunks(client, t.request, t.run, t.keys, t.slots)
        except CallRefused as exc:
            pool.release(t.slots)
            t.problem = f"refused: {exc}"
            return
        except Exception as exc:  # noqa: BLE001 - the NIC may still use the buffers
            t.problem = f"native exception: {exc!r}"
            pool.quarantine(t.slots, reason="native put raised")
            self._safe_distrust(repr(exc), t)
            return
        finally:
            t.native_ms = (time.perf_counter() - call_start) * 1000
        t.returned_chunks = len(t.slots)
        t.transaction = SettlementTransaction(pool, t.slots)
        t.problem = self.owner._settle_put(t.transaction, t.slots, answer)
        t.transaction.commit()
        if t.problem is None:
            t.stored_chunks = len(t.slots)

    def _quarantine_remaining(self, t: Ticket, pool: TransferSlotPool) -> None:
        # The hook itself failed: do not infer that the NIC is idle.
        remaining = t.transaction.remaining() if t.transaction else t.slots
        try:
            pool.quarantine(remaining, reason="settlement hook raised")
        except BaseException as exc:  # noqa: BLE001
            t.problem = f"{t.problem}; quarantine recovery failed: {exc!r}"
            with self.lock:
                self.closed = True

    def drain_request(self, req: LMCacheReqMeta) -> RequestResult:
        """Wait until every put of ``req`` has returned and been settled."""
        with self.condition:
            state = self.results[self._key(req)]
            while state.outstanding:
                self.condition.wait()
            return state

    def forget_request(self, req: LMCacheReqMeta) -> None:
        with self.lock:
            key = self._key(req)
            self.results.pop(key, None)
            self.failed.pop(key, None)
            self.cancelled.discard(key)

    def cancel(self, req: LMCacheReqMeta) -> None:
        with self.lock:
            self.cancelled.add(self._key(req))

    def close(self, timeout: float = 2) -> bool:
        """Stop the thread once its queue is empty; False if native work persists.

        The caller may release the pool and client only after True.
        """
        with self.lock:
            self.closed = True
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                # ``pending`` includes batches still preparing, so no save can
                # submit behind the stop sentinel.
                if self.pending == 0 and self.jobs.unfinished_tasks == 0:
                    if not self.stop_sent:
                        self.jobs.put(None)
                        self.stop_sent = True
                    break
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        self.thread.join(max(0, deadline - time.monotonic()))
        return not self.thread.is_alive()
