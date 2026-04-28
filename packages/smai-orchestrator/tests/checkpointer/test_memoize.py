"""Tests for the opt-in :func:`memoized` helper and :func:`stable_hash`
default convention per `05-orchestrator.md` §2 / §9 #3.
"""

from __future__ import annotations

from datetime import UTC, datetime

from _helpers import FakeClock  # type: ignore[import-not-found]
from pydantic import BaseModel
from smai_orchestrator.checkpointer import (
    InMemoryCheckpointBackend,
    MetadataStoreCheckpointer,
    memoized,
    stable_hash,
)


class _Inputs(BaseModel):
    cg_id: str
    review_attempt: int


class _Outputs(BaseModel):
    verdict: str
    findings: list[str]


def _serialize(out: _Outputs) -> bytes:
    return out.model_dump_json().encode("utf-8")


def _deserialize(raw: bytes) -> _Outputs:
    return _Outputs.model_validate_json(raw)


def test_stable_hash_deterministic() -> None:
    a = _Inputs(cg_id="cg_1", review_attempt=1)
    b = _Inputs(cg_id="cg_1", review_attempt=1)
    assert stable_hash(a) == stable_hash(b)


def test_stable_hash_distinguishes_inputs() -> None:
    a = _Inputs(cg_id="cg_1", review_attempt=1)
    b = _Inputs(cg_id="cg_1", review_attempt=2)
    assert stable_hash(a) != stable_hash(b)


async def test_memoized_first_call_runs_work_and_caches() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    call_count = 0

    async def work() -> _Outputs:
        nonlocal call_count
        call_count += 1
        return _Outputs(verdict="pass", findings=["nit"])

    inputs = _Inputs(cg_id="cg_1", review_attempt=1)
    out = await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="code_review_v1",
        inputs=inputs,
        work=work,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    assert out.verdict == "pass"
    assert call_count == 1


async def test_memoized_second_call_returns_cached_without_running() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    call_count = 0

    async def work() -> _Outputs:
        nonlocal call_count
        call_count += 1
        return _Outputs(verdict="pass", findings=[])

    inputs = _Inputs(cg_id="cg_1", review_attempt=1)
    await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="code_review_v1",
        inputs=inputs,
        work=work,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    out2 = await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="code_review_v1",
        inputs=inputs,
        work=work,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    assert out2.verdict == "pass"
    assert call_count == 1  # work() invoked exactly once across both calls


async def test_memoized_step_id_versioning_invalidates() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    call_count = 0

    async def work() -> _Outputs:
        nonlocal call_count
        call_count += 1
        return _Outputs(verdict="pass", findings=[])

    inputs = _Inputs(cg_id="cg_1", review_attempt=1)
    await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="code_review_v1",
        inputs=inputs,
        work=work,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    # Bump step_id; expect cache miss → work runs again.
    await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="code_review_v2",
        inputs=inputs,
        work=work,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    assert call_count == 2


async def test_memoized_distinct_inputs_distinct_memos() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    invocations: list[int] = []

    inputs_1 = _Inputs(cg_id="cg_1", review_attempt=1)
    inputs_2 = _Inputs(cg_id="cg_1", review_attempt=2)

    async def work_1() -> _Outputs:
        invocations.append(1)
        return _Outputs(verdict="pass", findings=[])

    async def work_2() -> _Outputs:
        invocations.append(2)
        return _Outputs(verdict="fail", findings=["x"])

    out_1 = await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="step_v1",
        inputs=inputs_1,
        work=work_1,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    out_2 = await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="step_v1",
        inputs=inputs_2,
        work=work_2,
        serialize=_serialize,
        deserialize=_deserialize,
    )
    assert out_1.verdict == "pass"
    assert out_2.verdict == "fail"
    assert invocations == [1, 2]


async def test_memoized_custom_compute_hash() -> None:
    """Handlers can override the hashing convention — e.g., hashing
    only a subset of the inputs to keep memo hits high across non-
    load-bearing changes."""
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    call_count = 0

    async def work() -> _Outputs:
        nonlocal call_count
        call_count += 1
        return _Outputs(verdict="pass", findings=[])

    def hash_only_cg_id(inp: _Inputs) -> str:
        return f"cg-{inp.cg_id}"

    inputs_1 = _Inputs(cg_id="cg_1", review_attempt=1)
    inputs_2 = _Inputs(cg_id="cg_1", review_attempt=99)  # only attempt differs
    await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="step",
        inputs=inputs_1,
        work=work,
        serialize=_serialize,
        deserialize=_deserialize,
        compute_hash=hash_only_cg_id,
    )
    await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="step",
        inputs=inputs_2,
        work=work,
        serialize=_serialize,
        deserialize=_deserialize,
        compute_hash=hash_only_cg_id,
    )
    # Second call hits the cache because the custom hash ignored
    # ``review_attempt``.
    assert call_count == 1


async def test_memoized_wall_clock_pins_created_at() -> None:
    cp = MetadataStoreCheckpointer(InMemoryCheckpointBackend())
    fake_clock = FakeClock(start=datetime(2026, 1, 1, tzinfo=UTC))

    async def work() -> _Outputs:
        return _Outputs(verdict="pass", findings=[])

    inputs = _Inputs(cg_id="cg_1", review_attempt=1)
    await memoized(
        checkpointer=cp,
        thread_id="cg_1",
        step_id="step_v1",
        inputs=inputs,
        work=work,
        serialize=_serialize,
        deserialize=_deserialize,
        wall_clock=fake_clock,
    )
    # Confirm the saved checkpoint carries the fake wall clock's now.
    from smai_orchestrator.checkpointer import CheckpointKey

    key = CheckpointKey(thread_id="cg_1", step_id="step_v1", input_hash=stable_hash(inputs))
    loaded = await cp.load(key)
    assert loaded is not None
    assert loaded.created_at == datetime(2026, 1, 1, tzinfo=UTC)
