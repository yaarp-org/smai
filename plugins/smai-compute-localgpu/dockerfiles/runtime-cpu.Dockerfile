# CPU-only runtime-side image for smai-compute-localgpu (~1-2 GB, multi-arch).
#
# Used when ``LocalGpuCompute.submit(..., gpu=False)`` for *experiment
# seed runs* — i.e., the orchestrator's run-record dispatcher hands this
# image to a CG whose ``controlled_conditions.compute.gpu`` is ``false``
# (a methodology smoke run, a kNN comparison, a small MLP, any
# comparison that isn't GPU-bound). Same ML stack as ``runtime.Dockerfile``
# (torch / torchvision / numpy / scipy / einops / timm) but on the lean
# ``python:3.11-slim-bookworm`` base with CPU-only torch wheels, so it is
# small everywhere and — because the base is multi-arch — runs natively
# on Apple Silicon (no amd64-under-Rosetta emulation, unlike the
# ``nvidia/cuda``-based ``runtime.Dockerfile``).
#
# It is selected automatically by the run-record dispatcher when
# ``compute.gpu=false``; override the name via ``engine.runtime_cpu_image``
# in ``smai.yaml``.
#
# Build with::
#
#     docker build -t smai-runtime-cpu:dev \
#         -f plugins/smai-compute-localgpu/dockerfiles/runtime-cpu.Dockerfile .
#
# No NVIDIA Container Toolkit needed (no GPU). SMAI does NOT publish
# prebuilt images for the OSS plugin in v1 (commit a9e57bd /
# ``07-plugin-interfaces.md`` §7.4).
#
# Linux bind-mount caveat (shared with the other two images): the
# container runs as the non-root ``smai`` user (uid 1000). On Linux a
# workspace directory bind-mounted at ``/workspace`` must be writable by
# uid 1000 (``chmod -R a+rwX`` it, or run smai as uid 1000); on macOS /
# Docker Desktop the virtiofs mount is writable regardless of host uid.
#
# Round 19: smai-core + smai-runtime are baked into the image. The
# fixed-template ``experiment.py`` (the agent's ``run_experiment`` tool
# entrypoint, hash-checked and locked) and the seed-run dispatcher's
# ``python -m smai_runtime.runner`` both import ``smai_runtime``, so the
# image must carry it. Rebuild this image when either package changes;
# see ``packages/smai-cli/OPERATIONS.md`` §7.1.

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

# Pin pip / setuptools / wheel so torch wheel resolution is
# reproducible across rebuilds.
RUN pip install --no-cache-dir --upgrade \
    pip==24.3.1 \
    setuptools==75.6.0 \
    wheel==0.45.1

# ML stack per ``10-runtime-and-templates.md`` §8.3 / §8.5 — same pins
# as ``runtime.Dockerfile`` but the CPU torch wheel index (``/whl/cpu``,
# which carries both x86_64 and aarch64 wheels). Bumps land in this
# Dockerfile and are reviewed.
# hadolint ignore=DL3013
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 \
        torchvision==0.20.1 \
    && pip install --no-cache-dir \
        numpy==2.1.3 \
        scipy==1.14.1 \
        einops==0.8.0 \
        timm==1.0.12

# smai-core + smai-runtime per Round 19. Required because the locked
# fixed-template ``experiment.py`` (smai-runtime/templates/_files/) does
# ``from smai_runtime.runner import run`` at module top, and the seed-run
# dispatcher submits ``python -m smai_runtime.runner ...`` against this
# image. Both consumers run inside this image, not on the host, so the
# image must carry the packages. Order matters: smai-core is
# smai-runtime's only first-party dependency (see
# ``tools/check_deps.py``); install it first so pip resolves the dep
# locally rather than reaching for PyPI (where these workspace packages
# are not published). Non-editable install (smaller layer, no
# ``__editable__`` indirection). Source is removed in the same RUN so it
# does not survive in the image layer. Build context is the repo root
# (see header build command).
COPY packages/smai-core/ /tmp/smai-core/
COPY packages/smai-runtime/ /tmp/smai-runtime/
RUN pip install --no-cache-dir /tmp/smai-core \
    && pip install --no-cache-dir /tmp/smai-runtime \
    && rm -rf /tmp/smai-core /tmp/smai-runtime

# Workspace mount-point per ``10-runtime-and-templates.md`` §8.5.
WORKDIR /workspace

# Default to a non-root user — same rationale as the other two images:
# an agent's / runner's writes into ``/workspace`` produce files owned
# by a UID that maps cleanly to the host.
RUN useradd --create-home --shell /bin/bash smai
USER smai

# No CMD / ENTRYPOINT — the substrate (Docker) sets the command via
# ``docker run <image> <command>``. ``LocalGpuCompute.submit`` always
# passes a command list explicitly.
