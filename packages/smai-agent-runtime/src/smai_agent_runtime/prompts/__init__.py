"""Per-step prompt YAMLs for the sandboxed roles.

Lives sandbox-side per architectural_decisions.md §3 + §12 #2: each
agent-reasoning step has its own prompt file rather than one monolithic
prompt per role. The mini-orchestrator loads these via
:func:`smai_agent_runtime.prompts._loader.load_step_prompt` at dispatch
time.

Sub-PR C1 lands two harness-builder prompts:

* ``harness_builder/step_2_fill_init_py.yaml`` — body-generation for
  one ABI function of ``harness/__init__.py`` (instantiated N times
  per ABI function in the contract).
* ``harness_builder/step_4_fill_baseline.yaml`` — body-generation for
  ``techniques/baseline.py``.

Sub-PR C2 relocates ``step_7_diagnose.yaml`` for the diagnose step.
"""

from __future__ import annotations

from smai_agent_runtime.prompts._loader import StepPrompt, load_step_prompt

__all__ = ["StepPrompt", "load_step_prompt"]
