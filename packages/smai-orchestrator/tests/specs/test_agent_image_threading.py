"""Round 14 — ``EngineConfig.agent_image`` is retained but unconsumed.

Round 12 added :attr:`EngineConfig.agent_image` and threaded it into a
containerized harness-builder / technique-implementer dispatch. Round 14
removed that containerized path: both agents now run **in-process in the
worker** (mirroring the planner), so nothing submits a :class:`Compute`
job under the agent image. The spec builders no longer accept an
``agent_image`` argument and the agent dispatch factories no longer take
one.

The field is deliberately kept — reserved for a possible future
re-containerization of the agent fleet — so these tests pin that it
still exists, still defaults to :data:`DEFAULT_AGENT_IMAGE`, and that
the spec builders / dispatch factories no longer carry the threading.
"""

from __future__ import annotations

import inspect

import smai_agents.agents.harness_builder as harness_builder_mod
import smai_agents.agents.technique_implementer as technique_implementer_mod
import smai_compute_localgpu as localgpu_mod
from smai_orchestrator.engine import DEFAULT_AGENT_IMAGE, EngineConfig
from smai_orchestrator.specs import register_smai_specs
from smai_orchestrator.specs.cg_entries import build_cg_entries_spec
from smai_orchestrator.specs.cg_execution import build_cg_execution_spec


def test_agent_image_field_retained_and_defaulted() -> None:
    """The field survives round 14 (reserved for re-containerization)
    and still defaults to the built-in constant."""
    assert EngineConfig().agent_image == DEFAULT_AGENT_IMAGE
    assert DEFAULT_AGENT_IMAGE == "smai-agent:dev"


def test_default_agent_image_constants_agree_across_packages() -> None:
    """The two surviving ``DEFAULT_AGENT_IMAGE`` copies hold the same
    string. Round 14 removed the smai-agents copies (the dispatch
    factories no longer reference an agent image); ``engine.config``
    keeps the canonical reserved constant and ``smai_compute_localgpu``
    keeps its own for the (currently unexercised) local image-build
    machinery. A local literal — not an import — per the foundational-
    import discipline noted in ``engine/config.py``; this test is the
    drift guard that reconciliation comment references."""
    assert DEFAULT_AGENT_IMAGE == localgpu_mod.DEFAULT_AGENT_IMAGE


def test_spec_builders_dropped_agent_image_param() -> None:
    """``build_cg_execution_spec`` / ``build_cg_entries_spec`` /
    ``register_smai_specs`` no longer accept ``agent_image`` — round 14
    removed the threading along with the containerized dispatch."""
    for fn in (build_cg_execution_spec, build_cg_entries_spec, register_smai_specs):
        assert "agent_image" not in inspect.signature(fn).parameters, fn.__name__


def test_dispatch_factories_dropped_agent_image_param() -> None:
    """The harness-builder / technique-implementer dispatch factories no
    longer take an ``agent_image`` — they run the agent loop in-process,
    never submitting a Compute job under an agent image."""
    assert (
        "agent_image"
        not in inspect.signature(harness_builder_mod.make_dispatch_harness_build).parameters
    )
    assert (
        "agent_image"
        not in inspect.signature(
            technique_implementer_mod.make_dispatch_technique_implementation
        ).parameters
    )
