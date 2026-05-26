# Sandbox-side image for smai-compute-localgpu — agent reasoning
# sessions (CPU-only, multi-arch).
#
# Used when the orchestrator dispatches a sandboxed agent role
# (currently harness_builder, technique_implementer) through the unified
# ``make_compute_dispatcher`` factory (Step 4 of the agent-layer
# refactor; see ``designs/smai/agent_refactor/implementation_plan.md``).
# The container entry point is ``python -m smai_agent_runtime --role
# <role> --cg-id <id>``; the mini-orchestrator inside runs PydanticAI
# Agent calls plus scripted workflow steps plus a ``subprocess.run``
# validation smoke. CPU-only by the §6 invariant — validation is a CPU
# smoke; seed runs are dispatched separately against the runtime
# images.
#
# Sibling image: ``runtime-cpu.Dockerfile`` (``smai-runtime-cpu:dev``,
# the lean multi-arch CPU image hosting experiment seed runs); ditto
# ``runtime.Dockerfile`` (``smai-runtime:dev``, the GPU variant). This
# is the third image in the smai-compute-localgpu lineup per D4 of the
# agent-layer refactor design notes.
#
# Build with::
#
#     docker build -t smai-agent-runtime:dev \
#         -f plugins/smai-compute-localgpu/dockerfiles/agent-runtime.Dockerfile .
#
# Smoke-test the entry point::
#
#     docker run --rm smai-agent-runtime:dev \
#         python -m smai_agent_runtime --role harness_builder --cg-id test
#
# At Step 3 of the refactor the role bodies are stubs that exit with
# code 64 (EXIT_NOT_IMPLEMENTED); a non-stub exit (0 success or 70
# crash) indicates a downstream wiring bug.
#
# Linux bind-mount caveat (shared with the other two images): the
# container runs as the non-root ``smai`` user (uid 1000). On Linux a
# workspace directory bind-mounted at ``/workspace`` must be writable by
# uid 1000 (``chmod -R a+rwX`` it, or run smai as uid 1000); on macOS /
# Docker Desktop the virtiofs mount is writable regardless of host uid.
#
# Round-19 install discipline applies verbatim: smai-core then
# smai-runtime then smai-agent-runtime, pip-installed in dep-graph
# order because the workspace packages are not on PyPI. The
# smai-runtime dep is load-bearing — the mini-orchestrator imports
# ``smai_runtime.runner`` for the validation smoke and the
# ``HarnessAPIManifest`` shape for the agent-reasoning input bundles
# (per D4 §3).

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates=\* \
        curl=\* \
        git=\* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pin pip / setuptools / wheel so workspace-package + PydanticAI wheel
# resolution is reproducible across rebuilds.
RUN pip install --no-cache-dir --upgrade \
    pip==24.3.1 \
    setuptools==75.6.0 \
    wheel==0.45.1

# Agent substrate + provider SDKs. Pinned per D4 §3 — PydanticAI is
# pre-1.0-numbered but moving fast; provider SDK majors do break
# compatibly only across long horizons. Bumps land in this Dockerfile
# and are reviewed (same discipline as the runtime Dockerfiles'
# torch / numpy / etc. pins).
# hadolint ignore=DL3013
RUN pip install --no-cache-dir \
        pydantic-ai==1.102.0 \
        anthropic==0.97.0 \
        openai==2.33.0 \
        boto3==1.42.97

# smai-core + smai-runtime + smai-agent-runtime per Round 19. Required
# because the mini-orchestrator inside this image imports
# ``smai_runtime.runner`` (validation smoke) and the ``HarnessAPIManifest``
# shape (agent-reasoning input bundles), and pulls workspace types from
# smai-core. Order matters: pip resolves the workspace-only deps locally
# only when the earlier package is already installed (the workspace
# packages are not published to PyPI). Non-editable install (smaller
# layer, no ``__editable__`` indirection); source removed in the same
# RUN so it does not survive in the image layer. Build context is the
# repo root (see header build command).
COPY packages/smai-core/ /tmp/smai-core/
COPY packages/smai-runtime/ /tmp/smai-runtime/
COPY packages/smai-agent-runtime/ /tmp/smai-agent-runtime/
RUN pip install --no-cache-dir /tmp/smai-core \
    && pip install --no-cache-dir /tmp/smai-runtime \
    && pip install --no-cache-dir /tmp/smai-agent-runtime \
    && rm -rf /tmp/smai-core /tmp/smai-runtime /tmp/smai-agent-runtime

# Workspace mount-point per ``10-runtime-and-templates.md`` §8.5 —
# Step 4 of the refactor will bind-mount the agent's per-session
# workspace here via the host-side ``stage_workspace`` Protocol.
WORKDIR /workspace

# Default to a non-root user — same rationale as the runtime images:
# an agent's writes into ``/workspace`` produce files owned by a UID
# that maps cleanly to the host.
RUN useradd --create-home --shell /bin/bash smai
USER smai

# No CMD / ENTRYPOINT — the substrate sets the command via
# ``docker run <image> <command>``. The unified compute dispatcher
# passes ``["python", "-m", "smai_agent_runtime", "--role", <role>, ...]``
# explicitly (Step 4 of the refactor).
