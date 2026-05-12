"""``smai status`` proposal / paper support + enriched snapshot —
observability follow-up, item 4.

Pass A generalized ``smai status`` from CG-only to "CG | proposal | paper"
via :meth:`StatusService.resolve_status` (probing the three id namespaces
in order), added :class:`EntityStatusSnapshot` / :class:`AgentStatusSnippet`
(attempt counters, ``last_error``, ``seconds_since_updated``, and the
per-turn ``status.json`` snippet for in-progress agent states), and rewrote
``_emit_status``. These tests exercise the service-surface call path the
CLI verb adapts over (full subprocess invocation isn't testable here —
same shape as :mod:`test_run_status`), plus ``_emit_status`` directly.
"""

from __future__ import annotations

import io
import json
from contextlib import AbstractAsyncContextManager, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _cli_fakes import (  # type: ignore[import-not-found]
    EXPERIMENT_YAML,
    FakeCompute,
    StubLlmProvider,
    make_registries_with_technique,
)
from smai_artifacts_localfs import LocalFsStore
from smai_cli.main import _emit_status
from smai_cli.runtime import (
    AgentStatusSnippet,
    EntityNotFoundError,
    EntityStatusSnapshot,
    Runtime,
)
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig
from smai_orchestrator.entities.tracking import PaperRecord, ProposalRecord

_NOW = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)


def _make_config() -> RuntimeConfig:
    return RuntimeConfig(
        engine=EngineConfig(poll_interval_seconds=10),
        plugins=PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
        ),
        pipelines=["smai_cg_execution", "smai_cg_entries"],
    )


def _runtime(tmp_path: Path) -> AbstractAsyncContextManager[Runtime]:
    overrides = PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )
    return Runtime.start_in_band(
        _make_config(),
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=False,
    )


def _proposal(proposal_id: str, *, state: str, **kw: object) -> ProposalRecord:
    return ProposalRecord(
        id=proposal_id,
        submission_kind="novel_technique",
        state=state,  # type: ignore[arg-type]
        created_at=_NOW,
        updated_at=_NOW,
        **kw,  # type: ignore[arg-type]
    )


def _paper(arxiv_id: str, *, state: str, **kw: object) -> PaperRecord:
    return PaperRecord(
        arxiv_id=arxiv_id,
        state=state,  # type: ignore[arg-type]
        created_at=_NOW,
        updated_at=_NOW,
        **kw,  # type: ignore[arg-type]
    )


# === resolve_status: the three id namespaces ================================


@pytest.mark.asyncio
async def test_resolve_status_cg_still_works(tmp_path: Path) -> None:
    async with _runtime(tmp_path) as runtime:
        runtime.experiments._registries_factory = make_registries_with_technique  # type: ignore[attr-defined]
        await runtime.experiments.submit_text(EXPERIMENT_YAML)

        snap = await runtime.status.resolve_status("cg_example")
        assert snap.kind == "cg"
        assert snap.entity_id == "cg_example"
        assert snap.state == "draft"
        assert snap.is_terminal is False
        assert snap.attempts == {"code_review_attempt": 0}
        assert snap.seconds_since_updated >= 0
        assert snap.agent_status == []  # draft is not an in-progress agent state


@pytest.mark.asyncio
async def test_resolve_status_resolves_proposal(tmp_path: Path) -> None:
    async with _runtime(tmp_path) as runtime:
        await runtime.plugins.metadata_store.create_proposal(
            _proposal(
                "prop-stat-1",
                state="designed",
                design_attempt=2,
                registration_attempt=1,
                last_error="registration plugin timeout",
            )
        )
        snap = await runtime.status.resolve_status("prop-stat-1")
        assert snap.kind == "proposal"
        assert snap.entity_id == "prop-stat-1"
        assert snap.state == "designed"
        assert snap.is_terminal is False
        assert snap.attempts == {"design_attempt": 2, "registration_attempt": 1}
        assert snap.last_error == "registration plugin timeout"
        # `designed` is not an in-progress agent state — no status.json read.
        assert snap.agent_status == []


@pytest.mark.asyncio
async def test_resolve_status_resolves_paper(tmp_path: Path) -> None:
    async with _runtime(tmp_path) as runtime:
        await runtime.plugins.metadata_store.create_paper(
            _paper(
                "2401.00001",
                state="screening",
                screening_attempt=1,
                planning_attempt=0,
                registration_attempt=0,
            )
        )
        snap = await runtime.status.resolve_status("2401.00001")
        assert snap.kind == "paper"
        assert snap.entity_id == "2401.00001"
        assert snap.state == "screening"
        assert snap.is_terminal is False
        assert snap.attempts == {
            "screening_attempt": 1,
            "planning_attempt": 0,
            "registration_attempt": 0,
        }


@pytest.mark.asyncio
async def test_resolve_status_terminal_markers(tmp_path: Path) -> None:
    async with _runtime(tmp_path) as runtime:
        await runtime.plugins.metadata_store.create_proposal(
            _proposal("prop-term", state="registered")
        )
        await runtime.plugins.metadata_store.create_paper(_paper("2401.00002", state="failed"))
        assert (await runtime.status.resolve_status("prop-term")).is_terminal is True
        assert (await runtime.status.resolve_status("2401.00002")).is_terminal is True


@pytest.mark.asyncio
async def test_resolve_status_unknown_id_raises(tmp_path: Path) -> None:
    async with _runtime(tmp_path) as runtime:
        with pytest.raises(EntityNotFoundError) as ei:
            await runtime.status.resolve_status("does-not-exist")
        # The message names all three namespaces it probed.
        msg = str(ei.value)
        assert "CG" in msg and "proposal" in msg and "paper" in msg


# === resolve_status: in-progress agent state surfaces status.json ===========


@pytest.mark.asyncio
async def test_resolve_status_designing_proposal_surfaces_planner_status(tmp_path: Path) -> None:
    async with _runtime(tmp_path) as runtime:
        await runtime.plugins.metadata_store.create_proposal(
            _proposal("prop-designing", state="designing")
        )
        # Stage the per-turn planner status.json (item-1 enriched payload).
        await runtime.plugins.artifact_store.put(
            "proposals/prop-designing/planner_status.json",
            json.dumps(
                {
                    "role": "planner",
                    "turn_count": 80,
                    "last_tool_call": "draft_comparison",
                    "last_tool_error": None,
                    "wall_clock_utc": "2026-05-11T11:42:00+00:00",
                    "attempt_index": None,
                }
            ).encode("utf-8"),
        )
        snap = await runtime.status.resolve_status("prop-designing")
        assert snap.kind == "proposal"
        assert len(snap.agent_status) == 1
        a = snap.agent_status[0]
        assert a.label == "planner"
        assert a.role == "planner"
        assert a.turn_count == 80
        assert a.last_tool_call == "draft_comparison"
        assert a.last_tool_error is None
        assert a.wall_clock_utc == "2026-05-11T11:42:00+00:00"


@pytest.mark.asyncio
async def test_resolve_status_missing_status_artifact_omits_section(tmp_path: Path) -> None:
    """A `designing` proposal with no staged planner_status.json yields an
    empty `agent_status` rather than erroring."""
    async with _runtime(tmp_path) as runtime:
        await runtime.plugins.metadata_store.create_proposal(
            _proposal("prop-no-artifact", state="designing")
        )
        snap = await runtime.status.resolve_status("prop-no-artifact")
        assert snap.kind == "proposal"
        assert snap.agent_status == []


@pytest.mark.asyncio
async def test_resolve_status_unparseable_status_artifact_omits_section(tmp_path: Path) -> None:
    async with _runtime(tmp_path) as runtime:
        await runtime.plugins.metadata_store.create_proposal(
            _proposal("prop-bad-artifact", state="designing")
        )
        await runtime.plugins.artifact_store.put(
            "proposals/prop-bad-artifact/planner_status.json", b"this is not json"
        )
        snap = await runtime.status.resolve_status("prop-bad-artifact")
        assert snap.agent_status == []


# === _emit_status rendering =================================================


def _snapshot(**kw: object) -> EntityStatusSnapshot:
    base: dict[str, object] = {
        "kind": "proposal",
        "entity_id": "prop-x",
        "state": "designing",
        "is_terminal": False,
        "updated_at": _NOW,
        "seconds_since_updated": 1080.0,  # 18 minutes
        "last_error": None,
        "attempts": {"design_attempt": 1, "registration_attempt": 0},
        "agent_status": [],
    }
    base.update(kw)
    return EntityStatusSnapshot(**base)  # type: ignore[arg-type]


def test_emit_status_text_renders_state_attempts_and_agent_line() -> None:
    snap = _snapshot(
        agent_status=[
            AgentStatusSnippet(
                label="planner",
                role="planner",
                turn_count=80,
                last_tool_call="draft_comparison",
                last_tool_error=None,
                wall_clock_utc="2026-05-11T11:42:00+00:00",
            )
        ],
        last_error=None,
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_status(snap, "text")
    out = buf.getvalue()
    assert "prop-x [proposal]: designing" in out
    assert "18m ago" in out
    assert "design_attempt=1" in out
    # registration_attempt=0 is omitted (only non-zero attempts shown).
    assert "registration_attempt" not in out
    assert "planner: turn ~80, last action draft_comparison, no error" in out


def test_emit_status_text_shows_last_error_and_terminal_marker() -> None:
    snap = _snapshot(
        kind="cg",
        entity_id="cg-x",
        state="implementation_failed",
        is_terminal=True,
        last_error="harness build exhausted retries",
        attempts={"code_review_attempt": 0},
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_status(snap, "text")
    out = buf.getvalue()
    assert "cg-x [cg]: implementation_failed (terminal)" in out
    assert "last_error: harness build exhausted retries" in out


def test_emit_status_json_includes_all_new_fields() -> None:
    snap = _snapshot(
        agent_status=[
            AgentStatusSnippet(
                label="planner", role="planner", turn_count=12, last_tool_call="set_conditions"
            )
        ],
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_status(snap, "json")
    payload = json.loads(buf.getvalue())
    assert payload["kind"] == "proposal"
    assert payload["id"] == "prop-x"
    assert payload["state"] == "designing"
    assert payload["is_terminal"] is False
    assert payload["seconds_since_updated"] == 1080.0
    assert payload["last_error"] is None
    assert payload["attempts"] == {"design_attempt": 1, "registration_attempt": 0}
    assert payload["agent_status"][0]["role"] == "planner"
    assert payload["agent_status"][0]["turn_count"] == 12
    assert payload["agent_status"][0]["last_tool_call"] == "set_conditions"
