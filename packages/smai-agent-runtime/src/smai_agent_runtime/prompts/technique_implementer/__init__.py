"""Per-step prompt bundle for the sandboxed ``technique_implementer``
mini-orchestrator (D8b).

Loaded via :func:`smai_agent_runtime.prompts.load_step_prompt` at
mini-orchestrator step dispatch time. The single per-step YAML in this
package is :mod:`step_2_fill_technique_body` — the technique
body-generation step. The validation and diagnose-on-failure steps
reuse harness_builder/step_7_diagnose.yaml (they're role-agnostic
scripted-and-agent steps shared across both sandboxed roles per
architectural_decisions §12 #1 expanded 2026-05-25).
"""
