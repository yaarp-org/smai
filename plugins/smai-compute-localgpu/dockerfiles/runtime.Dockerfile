# Runtime-side image for smai-compute-localgpu (GPU, ~5-8 GB).
#
# Used when ``LocalGpuCompute.submit(..., gpu=True)`` — i.e., the
# pipeline-runtime substrate that hosts harness pipelines and technique
# modules per ``10-runtime-and-templates.md`` §8.5. Includes the ML
# stack the runtime expects (torch, torchvision, numpy, scipy, einops,
# timm) on top of CUDA + cuDNN.
#
# Mac caveat: Docker Desktop on macOS / Apple Silicon cannot pass GPU
# through. This image will build on Mac but ``LocalGpuCompute`` rejects
# ``gpu=True`` jobs on Darwin upstream and points the user at
# smai-compute-modal / smai-compute-runpod.
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

# Workspace mount-point per ``10-runtime-and-templates.md`` §8.5.
WORKDIR /workspace

# Default to a non-root user — same rationale as agent.Dockerfile.
# CUDA + Docker GPU passthrough work fine for non-root containers
# given the NVIDIA Container Toolkit.
RUN useradd --create-home --shell /bin/bash smai
USER smai

# No CMD / ENTRYPOINT — the substrate sets the command via
# ``docker run <image> <command>``. ``LocalGpuCompute.submit`` always
# passes a command list explicitly.
