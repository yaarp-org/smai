# Agent-side image for smai-compute-localgpu (CPU-only, ~500 MB).
#
# Used when ``LocalGpuCompute.submit(..., gpu=False)`` — i.e., the
# agent loop's harness builder, technique implementer, code reviewer,
# and contextual evaluator. Mac hosts get this image and only this
# image; Docker Desktop on Mac cannot pass GPU through, so experiment
# runs (gpu=True) must point at smai-compute-modal / smai-compute-runpod.
#
# Per Task 2.A4 / ``10-runtime-and-templates.md`` §8.5: the runtime
# expects POSIX + Python 3.11+ + the ML stack. Agents don't need the
# ML stack (they generate code, they don't train models), so this
# image is intentionally lean.
#
# Build with::
#
#     docker build -t smai-agent:dev \
#         -f plugins/smai-compute-localgpu/dockerfiles/agent.Dockerfile .
#
# SMAI does NOT publish prebuilt images for the OSS plugin in v1
# (commit a9e57bd / ``07-plugin-interfaces.md`` §7.4).

FROM python:3.11-slim-bookworm

# Hadolint: pin the apt cache update + install in a single layer; clean
# up afterwards. ``--no-install-recommends`` keeps the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates=\* \
        curl=\* \
        git=\* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pin pip / setuptools / wheel before installing project deps so the
# image is reproducible across rebuilds. Lint-on-write per
# ``04-agents.md`` §3.2: agents run ``ruff check`` on every ``.py``
# write. Pinned to match the workspace's ``[tool.ruff]`` target.
RUN pip install --no-cache-dir --upgrade \
        pip==24.3.1 \
        setuptools==75.6.0 \
        wheel==0.45.1 \
    && pip install --no-cache-dir \
        ruff==0.15.12 \
        pydantic==2.10.4 \
        pyyaml==6.0.2 \
        httpx==0.28.1

# Workspace mount-point per ``10-runtime-and-templates.md`` §8.5.
WORKDIR /workspace

# Default to a non-root user so an agent's ``write_file`` tool produces
# files owned by a UID that maps cleanly to the host. The harness
# builder agent often runs as root only when it has to, but the
# baseline is non-root.
RUN useradd --create-home --shell /bin/bash smai
USER smai

# No CMD / ENTRYPOINT — the substrate (Docker) sets the command via
# ``docker run <image> <command>``. ``LocalGpuCompute.submit`` always
# passes a command list explicitly.
