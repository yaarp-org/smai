# Runtime-side image for smai-compute-localgpu — GPU experiment runs
# (~5-8 GB, amd64-only).
#
# Used for *experiment seed runs* whose CG requests GPU
# (``controlled_conditions.compute.gpu=true``, the default) — i.e., the
# pipeline-runtime substrate that hosts harness pipelines and technique
# modules per ``10-runtime-and-templates.md`` §8.5. Includes the ML
# stack the runtime expects (torch, torchvision, numpy, scipy, einops,
# timm) on top of CUDA + cuDNN.
#
# Sibling image: ``runtime-cpu.Dockerfile`` (``smai-runtime-cpu:dev``,
# the lean multi-arch CPU variant the run-record dispatcher picks when
# ``compute.gpu=false``).
#
# Mac caveat: Docker Desktop on macOS / Apple Silicon cannot pass GPU
# through. This image will build on Mac but ``LocalGpuCompute`` rejects
# ``gpu=true`` jobs on Darwin upstream and points the user at
# smai-compute-modal / smai-compute-runpod; ``compute.gpu=false``
# experiments use ``smai-runtime-cpu:dev`` instead.
#
# Build with::
#
#     docker build -t smai-runtime:dev \
#         -f plugins/smai-compute-localgpu/dockerfiles/runtime.Dockerfile .
#
# Requires the NVIDIA Container Toolkit on the host
# (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
# SMAI does NOT publish prebuilt images for the OSS plugin in v1
# (commit a9e57bd / ``07-plugin-interfaces.md`` §7.4).
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

# Pinned digest-equivalent: CUDA 12.4.1 + cuDNN 9 + Ubuntu 22.04. The
# runtime variant excludes development headers; nothing in the runtime
# substrate compiles CUDA kernels at run time (torch's wheels ship
# precompiled).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Hadolint: single layer, pinned packages, cache cleanup.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates=\* \
        curl=\* \
        git=\* \
        python3.11=\* \
        python3.11-venv=\* \
        python3-pip=\* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pin pip / setuptools / wheel so torch wheel resolution is
# reproducible across rebuilds.
RUN pip install --no-cache-dir --upgrade \
    pip==24.3.1 \
    setuptools==75.6.0 \
    wheel==0.45.1

# ML stack per ``10-runtime-and-templates.md`` §8.3 / §8.5. Torch wheels
# index points at PyPI's CUDA 12.4 build (cu124). Pinned to a known-
# compatible set; bumps land in this Dockerfile and are reviewed.
# hadolint ignore=DL3013
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
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

# Default to a non-root user so an agent's ``write_file`` tool produces
# files owned by a UID that maps cleanly to the host. CUDA + Docker GPU
# passthrough work fine for non-root containers given the NVIDIA
# Container Toolkit.
RUN useradd --create-home --shell /bin/bash smai
USER smai

# No CMD / ENTRYPOINT — the substrate sets the command via
# ``docker run <image> <command>``. ``LocalGpuCompute.submit`` always
# passes a command list explicitly.
