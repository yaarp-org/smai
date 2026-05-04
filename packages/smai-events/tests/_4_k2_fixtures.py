"""Test-fixture helpers for ``packages/smai-events/tests/``.

Per CLAUDE.md "Per-task fixture filename hygiene": this module's name
is unique within the workspace under ``--import-mode=importlib`` so it
will not collide with sibling fixture files in other packages.
"""

from __future__ import annotations

from datetime import UTC, datetime

from smai_api_spec.events import StateChangeEvent, WorkerHeartbeatEvent


def make_state_change(
    *,
    kind: str = "comparison_group",
    id: str = "cg-test-001",
    from_state: str = "draft",
    to_state: str = "implementing",
) -> StateChangeEvent:
    """Construct a deterministic :class:`StateChangeEvent` for tests."""
    return StateChangeEvent.model_validate(
        {
            "kind": kind,
            "id": id,
            "from": from_state,
            "to": to_state,
            "ts": datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC),
        }
    )


def make_heartbeat(
    *,
    cycle_id: int = 1,
    cycles_processed: int = 1,
) -> WorkerHeartbeatEvent:
    """Construct a deterministic :class:`WorkerHeartbeatEvent` for tests."""
    return WorkerHeartbeatEvent(
        cycle_id=cycle_id,
        cycles_processed=cycles_processed,
        ts=datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC),
    )


__all__ = ["make_heartbeat", "make_state_change"]
