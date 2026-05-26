"""Tests for :class:`RuntimeConfig` + :class:`PluginSelection`.

Per ``09-cli.md`` §3 — the typed model the CLI's config-layering
pipeline produces. The layering itself is Task 2.D2; this module
covers shape, defaults, and the small ``with_overrides`` helper.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from smai_orchestrator.engine import EngineConfig
from smai_orchestrator.runtime import PluginSelection, RuntimeConfig


def _minimal_selection() -> PluginSelection:
    return PluginSelection(
        llm_provider="bedrock",
        metadata_store="sqlite",
        artifact_store="localfs",
        compute="localgpu",
    )


def test_minimal_runtime_config_constructs() -> None:
    cfg = RuntimeConfig(
        engine=EngineConfig(),
        plugins=_minimal_selection(),
        pipelines=["smai_cg_execution"],
    )
    assert cfg.plugins.llm_provider == "bedrock"
    assert cfg.engine.poll_interval_seconds == 30
    assert cfg.pipelines == ["smai_cg_execution"]


def test_engine_runtime_images_have_defaults_and_are_overridable() -> None:
    """Round-4 friction (A): the GPU / CPU experiment-run images are
    config knobs on :class:`EngineConfig` (default to the reference
    ``smai-runtime:dev`` / ``smai-runtime-cpu:dev`` tags). Step 3 of
    the agent-layer refactor (D4 §5) adds the third image —
    ``smai-agent-runtime:dev`` — with the same default-and-overridable
    posture."""
    assert EngineConfig().runtime_image == "smai-runtime:dev"
    assert EngineConfig().runtime_cpu_image == "smai-runtime-cpu:dev"
    assert EngineConfig().agent_runtime_image == "smai-agent-runtime:dev"
    custom = EngineConfig(
        runtime_image="myorg/smai-runtime:1.2.3",
        runtime_cpu_image="myorg/smai-runtime-cpu:1.2.3",
        agent_runtime_image="myorg/smai-agent-runtime:1.2.3",
    )
    assert custom.runtime_image == "myorg/smai-runtime:1.2.3"
    assert custom.runtime_cpu_image == "myorg/smai-runtime-cpu:1.2.3"
    assert custom.agent_runtime_image == "myorg/smai-agent-runtime:1.2.3"


def test_plugin_selection_config_dicts_default_empty() -> None:
    sel = _minimal_selection()
    assert sel.llm_provider_config == {}
    assert sel.metadata_store_config == {}
    assert sel.artifact_store_config == {}
    assert sel.compute_config == {}


def test_plugin_selection_passes_through_config_dicts() -> None:
    sel = PluginSelection(
        llm_provider="bedrock",
        metadata_store="sqlite",
        artifact_store="localfs",
        compute="localgpu",
        llm_provider_config={"region": "us-east-1", "model_id": "test"},
        metadata_store_config={"uri": "sqlite+aiosqlite:///:memory:"},
        artifact_store_config={"root": "/tmp/test"},
        compute_config={"skip_preflight": True},
    )
    assert sel.llm_provider_config["region"] == "us-east-1"
    assert sel.metadata_store_config["uri"] == "sqlite+aiosqlite:///:memory:"
    assert sel.compute_config["skip_preflight"] is True


def test_extra_fields_rejected_on_plugin_selection() -> None:
    with pytest.raises(ValidationError):
        PluginSelection(
            llm_provider="bedrock",
            metadata_store="sqlite",
            artifact_store="localfs",
            compute="localgpu",
            unknown="oops",  # type: ignore[call-arg]
        )


def test_extra_fields_rejected_on_runtime_config() -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig(
            engine=EngineConfig(),
            plugins=_minimal_selection(),
            pipelines=["x"],
            dashboard={},  # type: ignore[call-arg] — DashboardConfig not yet shipped
        )


def test_pipelines_must_be_list() -> None:
    """The plural ``pipelines`` shape per `09` §3 — single-spec
    deployments pass a one-element list, not a bare string."""
    with pytest.raises(ValidationError):
        RuntimeConfig(
            engine=EngineConfig(),
            plugins=_minimal_selection(),
            pipelines="not_a_list",  # type: ignore[arg-type]
        )


def test_engine_and_plugins_are_orthogonal() -> None:
    """DEC-028 / `09` §3: changing engine config doesn't affect plugin
    selection and vice-versa.
    """
    cfg = RuntimeConfig(
        engine=EngineConfig(poll_interval_seconds=10, fair_scheduling="round_robin"),
        plugins=_minimal_selection(),
        pipelines=["x"],
    )
    flipped_engine = cfg.with_overrides(
        engine=EngineConfig(poll_interval_seconds=60, fair_scheduling="off")
    )
    assert flipped_engine.plugins == cfg.plugins  # same selection
    assert flipped_engine.engine.poll_interval_seconds == 60

    flipped_plugins = cfg.with_overrides(
        plugins=PluginSelection(
            llm_provider="anthropic",
            metadata_store="postgres",
            artifact_store="s3",
            compute="modal",
        )
    )
    assert flipped_plugins.engine == cfg.engine  # same engine config
    assert flipped_plugins.plugins.metadata_store == "postgres"


# ---- D3 / sub-PR D thread 2: per_role_runtime + nested role_models ---------


def test_per_role_runtime_defaults_sandbox_for_both_sandboxed_roles() -> None:
    """D3 / 2026-05-25 settled decision: both sandboxed roles default to
    ``"sandbox"`` uniformly across deployments. Pydantic-encoded so the
    default cannot drift through a code-only change."""
    from smai_orchestrator.engine import PerRoleRuntime

    runtime = EngineConfig().per_role_runtime
    assert runtime.harness_builder == "sandbox"
    assert runtime.technique_implementer == "sandbox"
    # Independent instance constructs the same defaults (not just a
    # shared reference).
    assert PerRoleRuntime().harness_builder == "sandbox"


def test_per_role_runtime_rejects_inline_role_keys() -> None:
    """D3 / per_role_policy.md Position 5: misconfiguring an inline role
    under ``per_role_runtime`` is Pydantic-visible. The ``extra="forbid"``
    on :class:`PerRoleRuntime` keeps the operator-footgun where a typo
    on ``planner`` / ``code_reviewer`` would silently fall through to
    the inline-by-architecture default out of bounds."""
    from smai_orchestrator.engine import PerRoleRuntime

    with pytest.raises(ValidationError):
        PerRoleRuntime(planner="sandbox")  # type: ignore[call-arg]


def test_role_models_accepts_flat_round7_shape() -> None:
    """Backward-compat drift-guard: round-7's flat
    ``{role: model_id}`` configs parse cleanly under the widened
    union type. No migration is required for existing users
    (per_role_policy.md Position 6)."""
    cfg = EngineConfig(
        role_models={
            "planner": "us.anthropic.claude-opus-4-6-v1",
            "harness_builder": "us.anthropic.claude-sonnet-4-6",
        }
    )
    assert cfg.role_models["planner"] == "us.anthropic.claude-opus-4-6-v1"


def test_role_models_accepts_nested_d3_shape() -> None:
    """D3 nested shape parses cleanly. ``_default`` sentinel + step-keyed
    entries co-exist on the same role; mixed flat + nested co-exist
    across different roles."""
    cfg = EngineConfig(
        role_models={
            "planner": "us.anthropic.claude-opus-4-6-v1",  # flat
            "harness_builder": {
                "_default": "us.anthropic.claude-opus-4-6-v1",
                "diagnose_on_failure": "us.anthropic.claude-sonnet-4-6",
            },  # nested
        }
    )
    planner_value = cfg.role_models["planner"]
    assert isinstance(planner_value, str)
    harness_value = cfg.role_models["harness_builder"]
    assert isinstance(harness_value, dict)
    assert harness_value["_default"] == "us.anthropic.claude-opus-4-6-v1"
    assert harness_value["diagnose_on_failure"] == "us.anthropic.claude-sonnet-4-6"


def test_with_overrides_is_immutable() -> None:
    """``with_overrides`` returns a new instance; the original is
    unchanged (Pydantic's ``model_copy`` semantics)."""
    cfg = RuntimeConfig(
        engine=EngineConfig(poll_interval_seconds=10),
        plugins=_minimal_selection(),
        pipelines=["x"],
    )
    cfg2 = cfg.with_overrides(pipelines=["y"])
    assert cfg.pipelines == ["x"]
    assert cfg2.pipelines == ["y"]
