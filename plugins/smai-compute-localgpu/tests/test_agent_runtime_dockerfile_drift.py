"""Drift guards for the agent-runtime Dockerfile (Step 3 of the
agent-layer refactor; see ``designs/smai/agent_refactor/design_notes/
agent_image_release.md``).

The sandbox-side ``smai-agent-runtime:dev`` image hosts the mini-
orchestrator that runs the harness_builder / technique_implementer
roles. The mini-orchestrator imports :mod:`smai_runtime.runner` for the
validation smoke and :class:`smai_runtime.manifest.HarnessAPIManifest`
for agent-reasoning input bundles, plus :mod:`smai_core` types
transitively. All three workspace packages must be pip-installed into
the image in dep-graph order because the workspace packages are not
published to PyPI (Round 19 discipline; D4 §3 of the refactor).

These tests pin the install in place so a future Dockerfile edit can't
silently drop a workspace-package layer. Same drift-guard discipline as
the sibling ``test_dockerfile_drift.py`` — assertion is on the
Dockerfile text, not on the built image (the pytest suite does not
build Docker images; the real test is the operator rebuild plus the
Step 3 ``docker run`` smoke).
"""

from __future__ import annotations

from pathlib import Path

_DOCKERFILES_DIR = Path(__file__).resolve().parents[1] / "dockerfiles"
_AGENT_DOCKERFILE = _DOCKERFILES_DIR / "agent-runtime.Dockerfile"


def test_agent_runtime_dockerfile_exists() -> None:
    """Acceptance criterion (Step 3): the Dockerfile is co-located with
    the existing runtime Dockerfiles per D4 §1."""
    assert _AGENT_DOCKERFILE.exists(), (
        f"missing Dockerfile at {_AGENT_DOCKERFILE} — D4 §1 placement"
    )


def test_agent_runtime_installs_workspace_packages_in_dep_order() -> None:
    """Per Round 19 + D4 §3 the agent-runtime image must end up with
    smai-core, smai-runtime, AND smai-agent-runtime all baked in.
    Round-21 (2026-05-26) refactored the Dockerfile to ``FROM
    smai-runtime-cpu:dev``, which means the first two are inherited
    from the base image (whose sibling drift-guards pin them in place);
    only smai-agent-runtime is added here directly.

    Guard:
    * ``FROM smai-runtime-cpu:dev`` is present (the base carries
      smai-core + smai-runtime per its own drift-guards).
    * ``COPY packages/smai-agent-runtime/ /tmp/smai-agent-runtime/`` is
      present (the layer added here).
    * ``pip install --no-cache-dir /tmp/smai-agent-runtime`` is present
      and follows the COPY line.
    """
    content = _AGENT_DOCKERFILE.read_text()
    assert "FROM smai-runtime-cpu:dev" in content, (
        "agent-runtime.Dockerfile: must inherit from smai-runtime-cpu:dev "
        "so smai-core + smai-runtime + the ML stack land transitively "
        "(round-21 composition)"
    )
    assert "COPY packages/smai-agent-runtime/ /tmp/smai-agent-runtime/" in content, (
        "agent-runtime.Dockerfile: missing COPY for smai-agent-runtime "
        "(D4 §3 — the agent runtime package itself must land in this layer)"
    )
    copy_pos = content.find("COPY packages/smai-agent-runtime/ /tmp/smai-agent-runtime/")
    install_pos = content.find("pip install --no-cache-dir /tmp/smai-agent-runtime")
    assert install_pos != -1, (
        "agent-runtime.Dockerfile: missing smai-agent-runtime pip install (D4 §3)"
    )
    assert copy_pos < install_pos, (
        "agent-runtime.Dockerfile: COPY must precede pip install for smai-agent-runtime"
    )


def test_agent_runtime_does_not_leave_source_behind() -> None:
    """The COPY-into-/tmp pattern is only acceptable if the same RUN
    layer removes the source — otherwise the final image carries the
    package twice (once in /tmp, once installed). Pin the ``rm -rf``
    cleanup so a future edit that splits the RUN into separate layers
    can't silently inflate the image. Round-21: only smai-agent-runtime
    is COPY'd into this layer (smai-core + smai-runtime are inherited
    from the base); the rm targets only what this layer added."""
    content = _AGENT_DOCKERFILE.read_text()
    assert "rm -rf /tmp/smai-agent-runtime" in content, (
        "agent-runtime.Dockerfile: workspace source must be removed in "
        "the same RUN as the pip install (Round 19, image-size hygiene)"
    )


def test_agent_runtime_pins_provider_sdks() -> None:
    """D4 §3 pins the four substrate deps. The exact pin values are
    operator-controlled and bumps land in this Dockerfile, but the
    presence of an ``==<version>`` pin is the discipline we lock here
    (matching the sibling runtime Dockerfiles' torch / numpy pins)."""
    content = _AGENT_DOCKERFILE.read_text()
    for package in ("pydantic-ai==", "anthropic==", "openai==", "boto3=="):
        assert package in content, (
            f"agent-runtime.Dockerfile: substrate dep {package!r} must be pinned (D4 §3)"
        )


def test_agent_runtime_uses_cpu_only_base() -> None:
    """Per D4 §1, the agent image is CPU-only and runs on a lean
    multi-arch CPU base — no CUDA runtime. Round-21 (2026-05-26)
    refactored the agent-runtime image to inherit from
    ``smai-runtime-cpu:dev``, which itself is on the lean
    ``python:3.11-slim-bookworm`` base with CPU-only torch wheels (its
    own drift-guard pins that). A future edit pointing at the
    nvidia/cuda base for either image would silently 5-8x the image
    size for no consumer (validation smokes are CPU per the §6
    invariant; GPU seed runs go to the separate runtime.Dockerfile).
    """
    content = _AGENT_DOCKERFILE.read_text()
    # Round-21 composition: agent-runtime is FROM smai-runtime-cpu, not
    # FROM python:3.11-slim-bookworm directly. The base image carries
    # the python:3.11-slim-bookworm bottom of the stack per its own
    # drift-guard test (``test_dockerfile_drift.py``).
    assert "FROM smai-runtime-cpu:dev" in content, (
        "agent-runtime.Dockerfile: must inherit from smai-runtime-cpu:dev "
        "(round-21 composition; the base carries the python:3.11-slim-bookworm "
        "+ CPU-only torch stack)"
    )
    assert "nvidia/cuda" not in content, (
        "agent-runtime.Dockerfile: agent sandbox is CPU-only by the §6 invariant; no CUDA base"
    )
