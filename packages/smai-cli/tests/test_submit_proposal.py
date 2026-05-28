"""Tests for ``smai submit-proposal`` + :meth:`ProposalsService.submit`.

Two surfaces:

* Service level — :meth:`ProposalsService.submit` persists the submitted
  novel-technique input to the submission artifact. Per DEC-032 a
  proposal's primary input is a free-text ``description`` the planner
  drafts the technique *from*; the typed ``technique_description`` is an
  optional pre-structured path (the planner's / ingestion's output
  shape). Both persist to the same key so the planner's JSON-or-text
  loader round-trips either. Exercised against a real LocalFsStore +
  in-memory SQLite.
* CLI verb — the prose positional argument and the ``--technique-json``
  typed path are wired to the matching ``submit(...)`` parameter; the
  mutual-exclusivity parse happens before any plugin is instantiated, so
  those paths are exercised through the CLI with a mocked runtime.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _cli_fakes import (  # type: ignore[import-not-found]
    FakeCompute,
    StubLlmProvider,
)
from smai_artifacts_localfs import LocalFsStore
from smai_cli import main
from smai_cli.runtime import Runtime
from smai_inline_agents.planner import TechniqueDescription
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
    PluginSelection,
    RuntimeConfig,
)
from smai_orchestrator.engine.config import EngineConfig
from typer.testing import CliRunner

_PROSE = (
    "Add cutout augmentation to the CIFAR-10 training transforms and "
    "compare top-1 accuracy against an unaugmented baseline."
)

_FLAG_NOTE = "Submit-path test probe; field intentionally left unset."
_VALID_TECHNIQUE_DESCRIPTION: dict[str, object] = {
    "name": "cutout_probe",
    "summary": "Cutout augmentation probe for the submit-path test.",
    "motivation": (
        "The submit-path test needs a minimal valid typed TechniqueDescription "
        "body that round-trips through the schema introduced at Step 2."
    ),
    "problem_setting": "Unit test of the submit path; no real ML problem setting.",
    "algorithm": {
        "summary": "Placeholder algorithm summary that satisfies the minimum length rule.",
    },
    "context_kind": "proposal",
    "source_proposal_id": "submit-test-probe",
    "confidence_flags": [
        {"field_path": "/limitations", "severity": "unknown", "note": _FLAG_NOTE},
        {"field_path": "/loss_function", "severity": "unknown", "note": _FLAG_NOTE},
        {"field_path": "/training_recipe", "severity": "unknown", "note": _FLAG_NOTE},
        {"field_path": "/evaluation_protocol", "severity": "unknown", "note": _FLAG_NOTE},
    ],
}


def _make_runtime_config() -> RuntimeConfig:
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


def _overrides(tmp_path: Path) -> PluginOverrides:
    return PluginOverrides(
        llm_providers={role: StubLlmProvider() for role in DEFAULT_TASK_ROLES},
        artifact_store=LocalFsStore(tmp_path / "artifacts"),
        compute=FakeCompute(),
    )


# === Service level ==========================================================


@pytest.mark.asyncio
async def test_submit_persists_prose_description_verbatim(tmp_path: Path) -> None:
    """``submit(description=...)`` persists the raw prose (utf-8) at the
    submission artifact key — the format the planner loader reads back
    verbatim (DEC-032 primary path)."""
    async with Runtime.start_in_band(
        _make_runtime_config(),
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=_overrides(tmp_path),
        run_worker=False,
    ) as runtime:
        submission = await runtime.proposals.submit(
            proposal_id="p-prose",
            description=_PROSE,
        )
        assert submission.technique_description_artifact_key
        raw = await runtime.plugins.artifact_store.get(
            submission.technique_description_artifact_key
        )
        assert raw == _PROSE.encode("utf-8")


@pytest.mark.asyncio
async def test_submit_persists_typed_description_as_json(tmp_path: Path) -> None:
    """``submit(technique_description=...)`` persists ``model_dump_json``
    at the same key (optional pre-structured path)."""
    td = TechniqueDescription.model_validate(_VALID_TECHNIQUE_DESCRIPTION)
    async with Runtime.start_in_band(
        _make_runtime_config(),
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=_overrides(tmp_path),
        run_worker=False,
    ) as runtime:
        submission = await runtime.proposals.submit(
            proposal_id="p-typed",
            technique_description=td,
        )
        assert submission.technique_description_artifact_key
        raw = await runtime.plugins.artifact_store.get(
            submission.technique_description_artifact_key
        )
        assert raw == td.model_dump_json(indent=2).encode("utf-8")


@pytest.mark.asyncio
async def test_submit_requires_exactly_one_input_form(tmp_path: Path) -> None:
    """Zero or multiple input forms are rejected at the service boundary."""
    async with Runtime.start_in_band(
        _make_runtime_config(),
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=_overrides(tmp_path),
        run_worker=False,
    ) as runtime:
        with pytest.raises(ValueError, match="exactly one"):
            await runtime.proposals.submit(proposal_id="p-none")
        with pytest.raises(ValueError, match="exactly one"):
            await runtime.proposals.submit(
                proposal_id="p-two",
                description=_PROSE,
                reproduce_paper_arxiv_id="2401.12345",
            )


# === CLI verb ===============================================================


def _install_mock_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``Runtime.start_in_band`` + ``load_runtime_config`` with
    stubs that record the kwargs the verb passes to ``proposals.submit``.

    Returns the dict the recorded submit kwargs land in.
    """
    captured: dict[str, Any] = {}

    class _FakeProposals:
        async def submit(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(proposal_id=kwargs["proposal_id"])

    class _FakeRuntime:
        def __init__(self) -> None:
            self.proposals = _FakeProposals()

    def _fake_start_in_band(config: Any, *, run_worker: bool = True, **_: Any) -> Any:
        @asynccontextmanager
        async def _cm() -> AsyncIterator[_FakeRuntime]:
            yield _FakeRuntime()

        return _cm()

    monkeypatch.setattr(main, "load_runtime_config", lambda *a, **k: object())
    monkeypatch.setattr(main.Runtime, "start_in_band", _fake_start_in_band)
    return captured


def test_cli_submit_proposal_prose_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positional argument is prose, wired to ``submit(description=...)``."""
    captured = _install_mock_runtime(monkeypatch)
    result = CliRunner().invoke(main.app, ["submit-proposal", _PROSE, "--id", "p1"])
    assert result.exit_code == 0, result.output
    assert captured["description"] == _PROSE
    assert captured["technique_description"] is None
    assert captured["reproduce_paper_arxiv_id"] is None
    assert captured["submission_kind"] == "novel_technique"


def test_cli_submit_proposal_technique_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--technique-json`` validates the file as a TechniqueDescription and
    wires it to ``submit(technique_description=...)``."""
    captured = _install_mock_runtime(monkeypatch)
    td_path = tmp_path / "technique.json"
    td_path.write_text(json.dumps(_VALID_TECHNIQUE_DESCRIPTION), encoding="utf-8")

    result = CliRunner().invoke(
        main.app, ["submit-proposal", "--technique-json", str(td_path), "--id", "p2"]
    )
    assert result.exit_code == 0, result.output
    assert captured["description"] is None
    assert isinstance(captured["technique_description"], TechniqueDescription)
    assert captured["technique_description"].name == "cutout_probe"


def test_cli_submit_proposal_requires_one_form() -> None:
    """No input form supplied → a clear parse-time error."""
    result = CliRunner().invoke(main.app, ["submit-proposal"])
    assert result.exit_code != 0
    assert "requires one of" in result.output


def test_cli_submit_proposal_prose_and_technique_json_mutually_exclusive(
    tmp_path: Path,
) -> None:
    """Prose argument + ``--technique-json`` is rejected before any plugin."""
    td_path = tmp_path / "technique.json"
    td_path.write_text(json.dumps(_VALID_TECHNIQUE_DESCRIPTION), encoding="utf-8")
    result = CliRunner().invoke(
        main.app, ["submit-proposal", _PROSE, "--technique-json", str(td_path)]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cli_submit_proposal_invalid_technique_json(tmp_path: Path) -> None:
    """A ``--technique-json`` file that is not a valid TechniqueDescription
    is rejected with a TechniqueDescription-shaped error."""
    td_path = tmp_path / "technique.json"
    td_path.write_text(json.dumps({"name": "incomplete"}), encoding="utf-8")
    result = CliRunner().invoke(main.app, ["submit-proposal", "--technique-json", str(td_path)])
    assert result.exit_code != 0
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "not a valid TechniqueDescription" in combined
