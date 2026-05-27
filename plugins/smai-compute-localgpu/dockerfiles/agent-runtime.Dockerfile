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
# Round-21 (2026-05-26) composition: the agent-runtime image is now a
# strict superset of ``smai-runtime-cpu:dev`` — it inherits the ML stack
# (torch / torchvision / numpy / scipy / einops / timm) plus the
# baked-in smai-core + smai-runtime packages, and adds the agent
# substrate on top (PydanticAI + provider SDKs + ruff + smai-agent-
# runtime). This matches architectural_decisions.md §1's design intent
# ("validation subprocess inside the agent sandbox"): the validation
# subprocess ``python experiment.py --mode validation`` imports the
# harness module, which transitively imports ``torch`` — so the
# sandbox must carry the ML deps the harness expects. Prior to round 21
# this image was a lean PydanticAI-only base and the validation
# subprocess crashed with ``ModuleNotFoundError: No module named
# 'torch'`` on the first attempt with a real ML harness.
#
# Build with::
#
#     # The smai-runtime-cpu:dev base must be built first.
#     docker build -t smai-runtime-cpu:dev \
#         -f plugins/smai-compute-localgpu/dockerfiles/runtime-cpu.Dockerfile .
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
# Linux bind-mount caveat (inherited from the base image): the
# container runs as the non-root ``smai`` user (uid 1000). On Linux a
# workspace directory bind-mounted at ``/workspace`` must be writable by
# uid 1000 (``chmod -R a+rwX`` it, or run smai as uid 1000); on macOS /
# Docker Desktop the virtiofs mount is writable regardless of host uid.
#
# Round-19 install discipline applies verbatim: smai-core, smai-runtime
# (both inherited from the base) plus smai-agent-runtime (added here)
# are pip-installed in dep-graph order because the workspace packages
# are not on PyPI. The smai-runtime dep is load-bearing — the mini-
# orchestrator imports ``smai_runtime.runner`` for the validation smoke
# and the ``HarnessAPIManifest`` shape for the agent-reasoning input
# bundles (per D4 §3).

FROM smai-runtime-cpu:dev

# Inherited from smai-runtime-cpu:dev: python:3.11-slim-bookworm base,
# pip 24.3.1 / setuptools 75.6.0 / wheel 0.45.1, the ML stack
# (torch 2.5.1 / torchvision 0.20.1 / numpy 2.1.3 / scipy 1.14.1 /
# einops 0.8.0 / timm 1.0.12), smai-core, smai-runtime, the ``smai``
# user, and ``WORKDIR /workspace``. We add the agent substrate on top.
#
# Switch back to root temporarily for the pip installs; the base image
# ends with ``USER smai`` so the layer additions need root.
USER root

# Agent substrate + provider SDKs. Pinned per D4 §3 — PydanticAI is
# pre-1.0-numbered but moving fast; provider SDK majors do break
# compatibly only across long horizons. Bumps land in this Dockerfile
# and are reviewed (same discipline as the runtime Dockerfiles'
# torch / numpy / etc. pins).
#
# ``ruff`` is the lint-on-write gate the mini-orchestrator runs after
# each body-generation step (per architectural_decisions §6 / Step 4
# task 7); without it on PATH the workflow loops on
# "ruff invocation error: ruff binary not on PATH" until the lint-retry
# budget exhausts (round-21 finding 2026-05-26).
# hadolint ignore=DL3013
RUN pip install --no-cache-dir \
        pydantic-ai==1.102.0 \
        anthropic==0.97.0 \
        openai==2.33.0 \
        boto3==1.42.97 \
        ruff==0.15.12

# smai-agent-runtime on top of the inherited smai-core + smai-runtime.
# Workspace mount-point and ``smai`` user already set by the base.
COPY packages/smai-agent-runtime/ /tmp/smai-agent-runtime/
RUN pip install --no-cache-dir /tmp/smai-agent-runtime \
    && rm -rf /tmp/smai-agent-runtime

# Restore the non-root user the base image established.
USER smai

# No CMD / ENTRYPOINT — the substrate sets the command via
# ``docker run <image> <command>``. The unified compute dispatcher
# passes ``["python", "-m", "smai_agent_runtime", "--role", <role>, ...]``
# explicitly (Step 4 of the refactor).
