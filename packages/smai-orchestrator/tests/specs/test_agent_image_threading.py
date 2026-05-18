"""Round 12 — ``EngineConfig.agent_image`` threads through to the two
containerized-agent dispatch factories.

CG-execution dispatches exactly two agent jobs as containers: the
harness builder (CG-level :func:`build_cg_execution_spec`) and the
technique implementer (entry-level :func:`build_cg_entries_spec`). Both
submit to :class:`Compute` with ``image=agent_image``. Round 12 made
that image configurable via :attr:`EngineConfig.agent_image`; round 11
had only ``runtime_image`` / ``runtime_cpu_image`` configurable.

These tests assert the configured value reaches each dispatch factory:
directly via the spec builders and through the
:func:`register_smai_specs` convenience helper the CLI bootstrap calls.
The four ``DEFAULT_AGENT_IMAGE`` copies (one per package, kept distinct
to avoid pulling the heavy smai-agents module into the foundational
``EngineConfig`` import) are pinned to the same string by a unit test
here rather than by a cross-package import.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
import smai_agents.agents.harness_builder as harness_builder_mod
import smai_agents.agents.technique_implementer as technique_implementer_mod
import smai_compute_localgpu as localgpu_mod
from _specs_fakes import StubLlmProvider  # type: ignore[import-not-found]
from smai_orchestrator.engine import DEFAULT_AGENT_IMAGE, EngineConfig
from smai_orchestrator.specs import register_smai_specs
from smai_orchestrator.specs.cg_entries import build_cg_entries_spec
from smai_orchestrator.specs.cg_execution import build_cg_execution_spec

_OVERRIDE_IMAGE = "registry.example.com/your-org/smai-agent:v2"


async def _noop_handler(ctx: Any) -> Any:  # pragma: no cover - never invoked
    """Stand-in dispatch handler returned by the patched factories."""
    raise AssertionError("the patched dispatch factory's handler must not run")


def _recording_factory(captured: dict[str, Any]) -> Callable[..., Callable[[Any], Awaitable[Any]]]:
    """Build a fake dispatch-factory that records its kwargs."""

    def _factory(**kwargs: Any) -> Callable[[Any], Awaitable[Any]]:
        captured.update(kwargs)
        return _noop_handler

    return _factory


def test_agent_image_reaches_harness_builder_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """``build_cg_execution_spec(agent_image=...)`` passes the value
    straight into :func:`make_dispatch_harness_build`."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        harness_builder_mod, "make_dispatch_harness_build", _recording_factory(captured)
    )

    build_cg_execution_spec(
        workspace_root=Path("/tmp/round12-cg"),
        llm_for_code_reviewer=StubLlmProvider([]),
        llm_for_contextual_evaluator=StubLlmProvider([]),
        agent_image=_OVERRIDE_IMAGE,
    )

    assert captured["agent_image"] == _OVERRIDE_IMAGE


def test_agent_image_reaches_technique_implementer_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_cg_entries_spec(agent_image=...)`` passes the value
    straight into :func:`make_dispatch_technique_implementation`."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        technique_implementer_mod,
        "make_dispatch_technique_implementation",
        _recording_factory(captured),
    )

    build_cg_entries_spec(
        workspace_root=Path("/tmp/round12-entries"),
        agent_image=_OVERRIDE_IMAGE,
    )

    assert captured["agent_image"] == _OVERRIDE_IMAGE


def test_register_smai_specs_threads_engine_config_agent_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full CLI-bootstrap path: ``EngineConfig.agent_image`` →
    :func:`register_smai_specs` → *both* dispatch factories."""
    harness_captured: dict[str, Any] = {}
    technique_captured: dict[str, Any] = {}
    monkeypatch.setattr(
        harness_builder_mod, "make_dispatch_harness_build", _recording_factory(harness_captured)
    )
    monkeypatch.setattr(
        technique_implementer_mod,
        "make_dispatch_technique_implementation",
        _recording_factory(technique_captured),
    )

    engine_config = EngineConfig(agent_image=_OVERRIDE_IMAGE)
    register_smai_specs(
        workspace_root=Path("/tmp/round12-register"),
        llm_for_code_reviewer=StubLlmProvider([]),
        llm_for_contextual_evaluator=StubLlmProvider([]),
        agent_image=engine_config.agent_image,
    )

    assert harness_captured["agent_image"] == _OVERRIDE_IMAGE
    assert technique_captured["agent_image"] == _OVERRIDE_IMAGE


def test_default_agent_image_used_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconfigured ``EngineConfig`` threads the built-in default
    constant — the orchestrator path always passes a value explicitly
    (never relies on the agent-module factory default)."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        harness_builder_mod, "make_dispatch_harness_build", _recording_factory(captured)
    )

    build_cg_execution_spec(
        workspace_root=Path("/tmp/round12-default"),
        llm_for_code_reviewer=StubLlmProvider([]),
        llm_for_contextual_evaluator=StubLlmProvider([]),
        agent_image=EngineConfig().agent_image,
    )

    assert EngineConfig().agent_image == DEFAULT_AGENT_IMAGE
    assert captured["agent_image"] == DEFAULT_AGENT_IMAGE


def test_default_agent_image_constants_agree_across_packages() -> None:
    """All four ``DEFAULT_AGENT_IMAGE`` copies hold the same string.

    ``smai_orchestrator.engine.config`` keeps a local copy rather than
    importing smai-agents' (a foundational-import / lazy-import
    discipline call, see the reconciliation comment in
    ``engine/config.py``). This test is the reconciliation that comment
    references — it fails loudly if a copy drifts.
    """
    assert DEFAULT_AGENT_IMAGE == harness_builder_mod.DEFAULT_AGENT_IMAGE
    assert DEFAULT_AGENT_IMAGE == technique_implementer_mod.DEFAULT_AGENT_IMAGE
    assert DEFAULT_AGENT_IMAGE == localgpu_mod.DEFAULT_AGENT_IMAGE
