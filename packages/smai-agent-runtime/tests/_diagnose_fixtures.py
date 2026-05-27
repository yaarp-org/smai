"""Per-task fixtures for the sub-PR C2 diagnose-step tests.

Per the SMAI test convention (``tests/_<task>_<purpose>.py``).

The :class:`DiagnoseFakeAgentRunner` mirrors the body-generation
:class:`FakeAgentRunner` but routes :class:`Diagnosis` outputs through
the same monkeypatch-on-``_run_agent_sync`` surface. Distinct module
so the two test surfaces' fixtures stay isolated under
``--import-mode=importlib``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from smai_agent_runtime.schemas import Diagnosis


@dataclass
class CapturedDiagnoseCall:
    """One captured agent-call invocation for a diagnose step."""

    user_message: str
    output_type: str


@dataclass
class DiagnoseFakeAgentRunner:
    """Callable test fake replacing ``_run_agent_sync`` for diagnose tests.

    Queues per-call :class:`Diagnosis` outputs; the runner pops the next
    one per call. If the queue exhausts the runner re-uses the last
    entry (lets a test author one canned response that covers every
    diagnose-retry attempt).
    """

    responses: list[BaseModel]
    calls: list[CapturedDiagnoseCall] = field(default_factory=list)

    def __call__(
        self,
        agent: Any,
        user_message: str,
        output_type: type[BaseModel],
        *,
        workspace: Any = None,
        trace_step_name: str | None = None,
        agent_runner: Any = None,
        usage_accumulator: Any = None,
    ) -> BaseModel:
        # Sub-PR D's resume-prep surface (architectural_decisions §12 #4):
        # accept and ignore the trace-persistence kwargs. The fake does
        # not exercise the message-history dump path (no AgentRunResult
        # to serialize); the unit test for that path lives in
        # ``test_entry.py``'s _persist_message_history coverage.
        # ``agent_runner`` is the thread-5 DI seam — also ignored (the
        # fake IS the runner). ``usage_accumulator`` is the Round-22
        # Bug-3 fix's seam — also ignored (the fake has no RunUsage).
        del workspace, trace_step_name, agent_runner, usage_accumulator
        self.calls.append(
            CapturedDiagnoseCall(
                user_message=user_message,
                output_type=output_type.__name__,
            ),
        )
        if not self.responses:
            raise RuntimeError("DiagnoseFakeAgentRunner exhausted its response queue")
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def make_diagnosis(
    *,
    diagnosis: str | None = None,
    fix_target: str = "harness",
    fix_payload: str | None = None,
) -> Diagnosis:
    """Build a conforming :class:`Diagnosis` for tests.

    Defaults are valid-against-the-schema (``diagnosis >= 20 chars``,
    ``fix_payload >= 1 char``) and produce a working harness body when
    ``fix_target == "harness"`` so the integration-style diagnose tests
    can drive the apply-then-revalidate loop end-to-end.
    """
    if diagnosis is None:
        diagnosis = (
            "Stub diagnosis: the failing call site is harness/__init__.py "
            "line 42; the test fix payload replaces it with a working body."
        )
    if fix_payload is None:
        # Working harness body that passes the experiment-shim's "always
        # succeed" semantics; the diagnose-step revalidation re-runs the
        # shim which always writes a passing validation_results.json.
        fix_payload = (
            '"""Diagnose-applied harness body for tests."""\n'
            "\n"
            "def build_harness(config):\n"
            "    return {}\n"
            "\n"
            "def run_training_loop(harness, technique):\n"
            "    return {}\n"
            "\n"
            "def evaluate(harness, result):\n"
            "    return {}\n"
        )
    return Diagnosis(
        diagnosis=diagnosis,
        fix_target=fix_target,  # type: ignore[arg-type]
        fix_payload=fix_payload,
    )


__all__ = [
    "CapturedDiagnoseCall",
    "DiagnoseFakeAgentRunner",
    "make_diagnosis",
]
