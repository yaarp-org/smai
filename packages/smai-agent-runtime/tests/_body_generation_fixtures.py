"""Per-task fake-LLM fixtures for sub-PR C1 body-generation step tests.

Per the SMAI test convention (``tests/_<task>_<purpose>.py``); the
per-task prefix keeps this module isolated from sibling-package fakes
under ``--import-mode=importlib``.

The fixtures here patch :func:`smai_agent_runtime.harness_builder._main._run_agent_sync`
so a test can assert against the bundle the dispatcher built without
calling a real LLM. Cross-process subprocess tests use the env-var
fake-LLM toggle instead (``SMAI_AGENT_RUNTIME_FAKE_LLM=1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from smai_agent_runtime.schemas import (
    HarnessBuilderBodyGenerationOutput,
    TechniqueBodyOutput,
)


@dataclass
class CapturedAgentCall:
    """One captured agent-call invocation."""

    user_message: str
    output_type: str
    workspace: Any = None
    trace_step_name: str | None = None


@dataclass
class FakeAgentRunner:
    """Callable test fake replacing ``_run_agent_sync``.

    ``responses`` queues per-call outputs; the runner pops the next one
    per call. If the queue exhausts, the runner reuses the last entry
    (lets a test author one canned response that covers every retry).
    ``calls`` records the user_message + output_type for each invocation
    so the test asserts against the bundle the dispatcher built.
    """

    responses: list[BaseModel]
    calls: list[CapturedAgentCall] = field(default_factory=list)

    def __call__(
        self,
        agent: Any,
        user_message: str,
        output_type: type[BaseModel],
        *,
        workspace: Any = None,
        trace_step_name: str | None = None,
        agent_runner: Any = None,
    ) -> BaseModel:
        # ``workspace`` + ``trace_step_name`` are sub-PR D thread 1's
        # resume-prep surface (architectural_decisions §12 #4); the
        # fake records them so a test can assert the dispatcher passed
        # the right step-name + workspace for the conversation-trace
        # persistence. ``agent_runner`` is sub-PR D thread 5's DI seam;
        # the fake accepts and ignores it (it IS the runner already).
        del agent_runner
        self.calls.append(
            CapturedAgentCall(
                user_message=user_message,
                output_type=output_type.__name__,
                workspace=workspace,
                trace_step_name=trace_step_name,
            ),
        )
        if self.responses:
            return self.responses[0] if len(self.responses) == 1 else self.responses.pop(0)
        raise RuntimeError("FakeAgentRunner exhausted its response queue")


def make_body_generation_output(
    *,
    function_name: str = "build_harness",
    module_source: str | None = None,
    reasoning: str = "test reasoning",
) -> HarnessBuilderBodyGenerationOutput:
    """Build a conforming :class:`HarnessBuilderBodyGenerationOutput`.

    Default ``module_source`` is a stub passing ``ruff check`` so the
    happy-path tests do not depend on real model output. Pass a bad
    source to force a lint failure for retry-path tests.
    """
    if module_source is None:
        module_source = (
            '"""Test stub harness body."""\n'
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
    return HarnessBuilderBodyGenerationOutput(
        function_name=function_name,  # type: ignore[arg-type]
        module_source=module_source,
        reasoning=reasoning,
    )


def make_technique_body_output(
    *,
    technique_py_source: str | None = None,
    reasoning: str | None = "test reasoning",
) -> TechniqueBodyOutput:
    """Build a conforming :class:`TechniqueBodyOutput`.

    Default ``technique_py_source`` passes ``ruff check``; pass a bad
    body to drive the lint-retry path in tests.
    """
    if technique_py_source is None:
        technique_py_source = (
            '"""Test baseline technique module."""\n'
            "\n"
            "def baseline(*args, **kwargs):\n"
            "    return None\n"
        )
    return TechniqueBodyOutput(
        technique_py_source=technique_py_source,
        reasoning=reasoning,
    )


__all__ = [
    "CapturedAgentCall",
    "FakeAgentRunner",
    "make_body_generation_output",
    "make_technique_body_output",
]
