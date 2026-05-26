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
    """Per Round 19 + D4 §3 the agent-runtime image must COPY + pip
    install smai-core, then smai-runtime, then smai-agent-runtime.
    Order is load-bearing: pip resolves each workspace-only dep locally
    only when the earlier package is already installed (the workspace
    packages are not published to PyPI).
    """
    content = _AGENT_DOCKERFILE.read_text()
    for package_path in (
        "COPY packages/smai-core/ /tmp/smai-core/",
        "COPY packages/smai-runtime/ /tmp/smai-runtime/",
        "COPY packages/smai-agent-runtime/ /tmp/smai-agent-runtime/",
    ):
        assert package_path in content, (
            f"agent-runtime.Dockerfile: missing {package_path!r} (Round 19 + D4 §3)"
        )
    core_install_pos = content.find("pip install --no-cache-dir /tmp/smai-core")
    runtime_install_pos = content.find("pip install --no-cache-dir /tmp/smai-runtime")
    agent_install_pos = content.find("pip install --no-cache-dir /tmp/smai-agent-runtime")
    assert core_install_pos != -1, (
        "agent-runtime.Dockerfile: missing smai-core pip install (Round 19)"
    )
    assert runtime_install_pos != -1, (
        "agent-runtime.Dockerfile: missing smai-runtime pip install (Round 19)"
    )
    assert agent_install_pos != -1, (
        "agent-runtime.Dockerfile: missing smai-agent-runtime pip install (D4 §3)"
    )
    assert core_install_pos < runtime_install_pos < agent_install_pos, (
        "agent-runtime.Dockerfile: install order must be smai-core → "
        "smai-runtime → smai-agent-runtime (pip resolves the workspace-"
        "only deps locally only when each earlier package is already "
        "installed)"
    )


def test_agent_runtime_does_not_leave_source_behind() -> None:
    """The COPY-into-/tmp pattern is only acceptable if the same RUN
    layer removes the source — otherwise the final image carries each
    package twice (once in /tmp, once installed). Pin the ``rm -rf``
    cleanup so a future edit that splits the RUN into separate layers
    can't silently inflate the image."""
    content = _AGENT_DOCKERFILE.read_text()
    assert "rm -rf /tmp/smai-core /tmp/smai-runtime /tmp/smai-agent-runtime" in content, (
        "agent-runtime.Dockerfile: workspace sources must be removed in "
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
    """Per D4 §1, the agent image is CPU-only and runs on the lean
    multi-arch python:3.11-slim-bookworm base — no CUDA runtime. A future
    edit pointing at the nvidia/cuda base would silently 5-8x the image
    size for no consumer (validation smokes are CPU per the §6
    invariant; GPU seed runs go to the separate runtime images).
    """
    content = _AGENT_DOCKERFILE.read_text()
    assert "FROM python:3.11-slim-bookworm" in content, (
        "agent-runtime.Dockerfile: must use python:3.11-slim-bookworm "
        "base (D4 §1 — CPU-only, multi-arch)"
    )
    assert "nvidia/cuda" not in content, (
        "agent-runtime.Dockerfile: agent sandbox is CPU-only by the §6 invariant; no CUDA base"
    )
