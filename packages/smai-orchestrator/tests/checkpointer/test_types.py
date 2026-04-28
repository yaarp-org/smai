"""Shape tests for :class:`CheckpointKey` / :class:`Checkpoint` /
:class:`Checkpointer` Protocol per ``05-orchestrator.md`` §2.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from smai_orchestrator.checkpointer import (
    Checkpoint,
    Checkpointer,
    CheckpointKey,
    InMemoryCheckpointBackend,
    MetadataStoreCheckpointer,
)


def test_checkpoint_key_round_trip() -> None:
    key = CheckpointKey(thread_id="cg_1", step_id="code_review_v3", input_hash="abc")
    assert key.thread_id == "cg_1"
    assert key.step_id == "code_review_v3"
    assert key.input_hash == "abc"


def test_checkpoint_key_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        CheckpointKey.model_validate(
            {"thread_id": "cg_1", "step_id": "x", "input_hash": "h", "extra": True}
        )


def test_checkpoint_round_trip() -> None:
    ckpt = Checkpoint(
        key=CheckpointKey(thread_id="t", step_id="s", input_hash="h"),
        result=b"payload",
        created_at=datetime(2026, 4, 28, 12, tzinfo=UTC),
    )
    assert ckpt.result == b"payload"
    assert ckpt.created_at.tzinfo is UTC


def test_checkpointer_protocol_runtime_checkable() -> None:
    """:class:`Checkpointer` is a runtime-checkable Protocol; both
    shipped flavors satisfy ``isinstance`` per `05` §2."""
    backend = InMemoryCheckpointBackend()
    instance = MetadataStoreCheckpointer(backend)
    assert isinstance(instance, Checkpointer)
