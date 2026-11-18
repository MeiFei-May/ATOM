# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The opt-in save pipeline on the real worker, pool and Store client (CPU fakes)."""

import threading
import time
from functools import partial

import pytest
import test_mooncake_store_offload as offload

from atom.kv_transfer.offload.chunked_scheduler import (
    DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL,
    DENSE_PAGE_SOURCE_SAFE_CHANNEL,
    DENSE_PAGE_STORE_CHANNEL,
)
from atom.kv_transfer.offload.kv_store import worker as worker_mod
from atom.kv_transfer.offload.kv_store.save_pipeline import (
    StreamStopped,
    stream_chunks,
)

aiter_arch = offload.aiter_arch
cluster = offload.cluster
make_worker = offload.make_worker
KEYS = 256  # the pipeline's key size, in tokens


@pytest.fixture(autouse=True)
def pipeline_chunks(monkeypatch):
    # The registered-buffer fake transport with the production key size.
    monkeypatch.setattr(offload, "CHUNK", KEYS)
    monkeypatch.setattr(offload, "CHUNK_BYTES", 1024)
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "1")


def _channels(worker):
    return worker.get_finished().connector_completions


def _packed(gpu):
    return sorted(s for kind, starts in gpu.calls if kind == "from" for s in starts)


def _puts(worker):
    return [c[1] for c in worker._client._store.calls if c[0] == "put"]


def _reports(worker, gpu):
    """Every source-safe report, by chunk index, in the order they arrive."""
    reported = []
    report = worker._source_group_safe

    def counting(identity):
        [(first, _end)] = identity.ranges
        reported.append(first // KEYS)
        report(identity)

    worker._source_group_safe = counting
    gpu.source_safe = counting
    return reported


# --- configuration ---------------------------------------------------------


@pytest.mark.parametrize(
    "tokens, value, expected",
    [(256, None, 0), (256, "", 0), (256, "0", 0), (256, "1", 1), (256, "64", 64)]
    + [(256, "-1", ValueError), (256, "65", ValueError), (8, "1", ValueError)],
)
def test_pipeline_is_opt_in_and_validated(monkeypatch, tokens, value, expected):
    monkeypatch.delenv("ATOM_STORE_SAVE_BATCH_CHUNKS")
    if value is not None:
        monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", value)
    config = offload._config(extra=offload._extra(chunk_tokens=tokens))
    if expected is ValueError:
        with pytest.raises(ValueError):
            offload.MooncakeStoreOffloadConnector(config)
        return
    worker = offload.MooncakeStoreOffloadConnector(config)
    try:
        assert worker._save_batch_chunks == expected
        assert (worker._sender is not None) == bool(expected)
    finally:
        worker.close()
    assert worker._sender is None or not worker._sender.thread.is_alive()


def test_batches_reuse_the_window_lookup_tail_first():
    segments = [
        (range(4, 9), False, list("45678"), 3.5),
        (range(4), True, list("0123"), None),
    ]
    got = [(list(r), h, k, ms) for r, h, k, ms in stream_chunks(segments, 2)]
    assert got == [
        ([7, 8], False, ["7", "8"], 3.5),
        ([5, 6], False, ["5", "6"], None),
        ([4], False, ["4"], None),
        ([2, 3], True, ["2", "3"], None),
        ([0, 1], True, ["0", "1"], None),
    ]
    for value in (True, 0, 65, 1.5):
        with pytest.raises(ValueError):
            list(stream_chunks([], value))


# --- the sender, on the real pool and client --------------------------------


def _sender_of(make_worker, name, *, batch=2):
    worker, _ = make_worker(save_slots=32)
    req = offload._save_req(name, 8 * KEYS)
    worker._sender.begin_request(req, batch)
    return worker, worker._sender, req


def _submit(worker, req, *, first=0, n=2):
    worker._sender.reserve(req, n)
    slots = worker._pool.acquire("save", n)
    run = range(first, first + n)
    worker._sender.submit(req, run, worker._window_keys(req, run), slots)
    return slots


def _block_native_put(monkeypatch, worker):
    store = worker._client._store
    entered, release = threading.Event(), threading.Event()
    put = store.batch_put_from

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release the native put"
        return put(*args, **kwargs)

    monkeypatch.setattr(store, "batch_put_from", blocked)
    return entered, release


def test_two_credits_bound_preparation_while_a_put_runs(monkeypatch, make_worker):
    worker, sender, req = _sender_of(make_worker, "credits")
    entered, release = _block_native_put(monkeypatch, worker)
    _submit(worker, req)
    assert entered.wait(1)
    _submit(worker, req, first=2)
    acquired = threading.Event()

    def third():
        sender.reserve(req, 2)
        acquired.set()
        sender.release_preparing()

    thread = threading.Thread(target=third)
    thread.start()
    try:
        assert not acquired.wait(0.05) and sender.pending == 2
    finally:
        release.set()
        thread.join(2)
    assert acquired.is_set()
    result = sender.drain_request(req)
    assert (result.stored_chunks, result.problem, sender.pending) == (4, None, 0)
    assert worker._pool.quarantined("save") == 0


def test_a_failed_settlement_hook_quarantines_what_it_did_not_apply(
    monkeypatch, make_worker
):
    worker, sender, req = _sender_of(make_worker, "settle-raises")

    def hook(pool, slots, answer):
        pool.release(slots[:1])
        raise RuntimeError("hook blew up")

    monkeypatch.setattr(worker, "_settle_put", hook)
    _submit(worker, req)
    assert "hook blew up" in sender.drain_request(req).problem
    # The hook's release was only recorded, never applied: both slots stay out.
    assert worker._pool.quarantined("save") == 2


def test_mixed_settlement_releases_only_the_settled_slots(monkeypatch, make_worker):
    worker, sender, req = _sender_of(make_worker, "mixed-settle")

    def hook(pool, slots, answer):
        pool.release(slots[:1])
        pool.quarantine(slots[1:], reason="the native side still owns it")
        return "partial settlement"

    monkeypatch.setattr(worker, "_settle_put", hook)
    slots = _submit(worker, req)
    assert sender.drain_request(req).problem == "partial settlement"
    assert [slot.leased for slot in slots] == [False, False]
    assert worker._pool.quarantined("save") == 1


@pytest.mark.parametrize("how", ["cancel", "close"])
def test_a_batch_cancelled_or_closed_while_preparing_is_never_put(make_worker, how):
    worker, sender, req = _sender_of(make_worker, "cancelled")
    sender.reserve(req, 2)
    slots = worker._pool.acquire("save", 2)
    # A close cannot finish while a batch is preparing, but it is marked.
    sender.cancel(req) if how == "cancel" else sender.close(timeout=0.01)
    sender.submit(req, range(2), worker._window_keys(req, range(2)), slots)
    assert sender.drain_request(req).problem == "cancelled"
    assert _puts(worker) == [] and not any(slot.leased for slot in slots)


# --- saves through the worker ----------------------------------------------


def test_held_chunks_are_never_packed_or_put(cluster, make_worker):
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("mixed", 4 * KEYS)
    held = {offload._key(req.chunk_hashes, i) for i in (0, 2)}
    objects = cluster.objects.setdefault(offload.MASTER, {})
    for key in held:
        objects[key] = b"already present"
    worker._do_save_req(req)
    assert _packed(gpu) == [KEYS, 3 * KEYS]
    puts = {key for keys in _puts(worker) for key in keys}
    assert puts == {offload._key(req.chunk_hashes, i) for i in (1, 3)}
    # The bytes stored are the packed chunks, and the held ones are untouched.
    assert {len(objects[key]) for key in puts} == {offload.CHUNK_BYTES}
    assert all(objects[key] == b"already present" for key in held)
    done = _channels(worker)
    assert {offload._safe(req.save_operation, i) for i in range(4)} <= done
    assert offload._store(req.save_operation) in done
    assert worker._sender.pending == 0


def test_lookup_failure_puts_every_chunk(make_worker, caplog):
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("lookup-failure", 3 * KEYS)
    worker._client._store.raise_on = "exists"
    worker._do_save_req(req)
    assert _packed(gpu) == [0, KEYS, 2 * KEYS]
    assert offload._store(req.save_operation) in _channels(worker)
    assert "lookup-failure" in caplog.text and "lookup failed" in caplog.text


def test_a_stopped_sender_is_not_a_lookup_failure(make_worker, caplog):
    worker, gpu = make_worker(save_slots=32)
    sender = worker._sender
    req = offload._save_req("stopped", 3 * KEYS)
    assert sender.close(timeout=1)
    worker._do_save_req(req)
    # The window is not handed over for repacking; the save ends as a failure.
    assert gpu.calls == []
    assert offload._store(req.save_operation, False) in _channels(worker)
    assert "lookup failed" not in caplog.text
    with pytest.raises(StreamStopped):
        start = partial(sender.start_lookup, worker._client)
        list(worker._save_segments(start, sender.finish_lookup, req, [range(3)]))


@pytest.mark.parametrize("fence_fails", [False, True])
def test_copy_failure_drains_native_before_the_terminal(
    make_worker, monkeypatch, caplog, fence_fails
):
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("copy-failure", 4 * KEYS)
    entered, release = _block_native_put(monkeypatch, worker)
    copy_failed = threading.Event()
    copy = gpu.batched_from_gpu
    reported = _reports(worker, gpu)

    def fail_second_copy(*args, **kwargs):
        if gpu.calls:
            assert entered.wait(2)
            copy_failed.set()
            raise RuntimeError("second pack failed")
        return copy(*args, **kwargs)

    def fence(device):
        if fence_fails:
            raise RuntimeError("copy fence failed")

    monkeypatch.setattr(gpu, "batched_from_gpu", fail_second_copy)
    monkeypatch.setattr(worker_mod, "_synchronize", fence)
    errors = []

    def save():
        try:
            worker._do_save_req(req)
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            errors.append(exc)

    thread = threading.Thread(target=save)
    thread.start()
    try:
        assert copy_failed.wait(2)
        before = _channels(worker)
        assert before == {offload._safe(req.save_operation, 3)}
        assert thread.is_alive() and worker._sender.pending > 0
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    after = before | _channels(worker)
    stores = [c for c in after if c.channel == DENSE_PAGE_STORE_CHANNEL]
    assert stores == [offload._store(req.save_operation, False)]
    safe = {c for c in after if c.channel == DENSE_PAGE_SOURCE_SAFE_CHANNEL}
    quiescent = {c for c in after if c.channel == DENSE_PAGE_SOURCE_QUIESCENT_CHANNEL}
    if fence_fails:
        # Only chunk 3, reported when its copy finished; nothing is claimed
        # for the chunks a failed fence cannot vouch for.
        assert safe == {offload._safe(req.save_operation, 3)} and not quiescent
        assert worker._pool.quarantined("save") == 1
    else:
        assert safe == {offload._safe(req.save_operation, i) for i in range(4)}
        assert quiescent == {offload._quiescent(req.save_operation)}
        assert worker._pool.quarantined("save") == 0
        # Chunk 3 was reported when its copy finished and is not reported again.
        assert sorted(reported) == [0, 1, 2, 3]
    assert worker._sender.pending == 0
    assert "copy-failure" in caplog.text and "second pack failed" in caplog.text


def test_a_put_failure_ends_the_save_once_after_native_returns(make_worker, caplog):
    worker, _ = make_worker(save_slots=32)
    req = offload._save_req("put-failure", KEYS)
    worker._client._store.raise_on = "put"
    worker._do_save_req(req)
    done = _channels(worker)
    assert {c for c in done if c.channel == DENSE_PAGE_STORE_CHANNEL} == {
        offload._store(req.save_operation, False)
    }
    assert worker._pool.quarantined("save") == 1
    assert worker._store_distrusted is not None and worker._sender.pending == 0
    assert "put-failure" in caplog.text and "put blew up" in caplog.text


def test_an_earlier_put_failure_stops_preparation_and_reports_every_chunk_once(
    make_worker, monkeypatch, caplog
):
    worker, gpu = make_worker(save_slots=32)
    req = offload._save_req("reserve-stopped", 2 * KEYS)
    worker._client._store.raise_on = "put"
    sender = worker._sender
    reserve = sender.reserve
    reported = _reports(worker, gpu)
    calls = 0

    def reserve_after_failure(request, nslots):
        nonlocal calls
        calls += 1
        if calls == 2:
            with sender.condition:
                assert sender.condition.wait_for(
                    lambda: request.save_operation in sender.failed, timeout=2
                )
        return reserve(request, nslots)

    monkeypatch.setattr(sender, "reserve", reserve_after_failure)
    worker._do_save_req(req)
    assert calls == 2
    assert gpu.calls == [("from", [KEYS])]
    done = _channels(worker)
    assert {c for c in done if c.channel == DENSE_PAGE_STORE_CHANNEL} == {
        offload._store(req.save_operation, False)
    }
    assert {offload._safe(req.save_operation, i) for i in range(2)} <= done
    assert offload._quiescent(req.save_operation) in done
    # Chunk 1 was packed and reported; the failed save reports only chunk 0 more.
    assert sorted(reported) == [0, 1]
    assert sender.pending == 0
    assert "reserve-stopped" in caplog.text and "put blew up" in caplog.text


def test_next_window_lookup_does_not_wait_behind_the_current_put(
    make_worker, monkeypatch
):
    """At the default batch limit window N+1 packs while window N's put runs."""
    monkeypatch.setenv("ATOM_STORE_SAVE_BATCH_CHUNKS", "64")
    worker, gpu = make_worker(save_slots=32)
    window = worker._pool.window("save", worker.save_workers)
    assert window < 64, "each window must be one put at the default batch size"
    req = offload._save_req("overlap", 3 * window * KEYS)
    store = worker._client._store
    events = []
    entered, release = threading.Event(), threading.Event()
    put, copy, exists = worker._put_chunks, gpu.batched_from_gpu, store.batch_is_exist

    def blocked_put(*args, **kwargs):
        events.append("put")
        entered.set()
        assert release.wait(5), "test did not release the native put"
        return put(*args, **kwargs)

    def logged_copy(*args, **kwargs):
        events.append("pack")
        return copy(*args, **kwargs)

    def logged_exists(keys):
        events.append("exists")
        return exists(keys)

    monkeypatch.setattr(worker, "_put_chunks", blocked_put)
    monkeypatch.setattr(gpu, "batched_from_gpu", logged_copy)
    monkeypatch.setattr(store, "batch_is_exist", logged_exists)
    thread = threading.Thread(target=worker._do_save_req, args=(req,))
    thread.start()
    try:
        assert entered.wait(2)
        deadline = time.monotonic() + 2
        while events.count("pack") < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        # Pack #2 overlapped the first put. The producer may pack between the
        # two lookups (the first is answered before the second runs): only
        # their order against the put is fixed.
        assert not release.is_set() and events.count("pack") >= 2, events
        assert events[: events.index("put")].count("exists") == 2, events
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert offload._store(req.save_operation) in _channels(worker)


# --- close() ---------------------------------------------------------------


def _short_close(monkeypatch, worker):
    close = worker._sender.close
    monkeypatch.setattr(worker._sender, "close", lambda: close(timeout=0.02))


def _blocked_save(worker, req):
    def save():
        worker._sender.begin_request(req, 1)
        _submit(worker, req, n=1)
        return worker._sender.drain_request(req)

    return worker._save_executor.submit(save)


def test_close_keeps_the_owners_while_a_native_put_is_blocked(
    monkeypatch, cluster, make_worker
):
    worker, gpu = make_worker(save_slots=32)
    pool, client, _ = worker._registered()
    store = cluster.stores[-1]
    entered, release = _block_native_put(monkeypatch, worker)
    _short_close(monkeypatch, worker)
    future = _blocked_save(worker, offload._save_req("blocked-close", KEYS))
    assert entered.wait(1)
    started = time.monotonic()
    worker.close()  # the one close the exit path makes
    assert time.monotonic() - started < 1, "close must give up, not wait"
    assert not future.done()
    assert worker._pool is pool and worker._client is client
    assert not gpu.closed and not store.closed
    with pytest.raises(RuntimeError, match="shutdown"):
        worker._save_executor.submit(lambda: None)
    release.set()
    assert future.result(timeout=2).stored_chunks == 1
    worker.close()  # a second close releases everything
    assert gpu.closed and store.closed
    assert not worker._sender.thread.is_alive()
