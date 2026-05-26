"""Compute plugin: local Docker reference implementation.

Per ``07-plugin-interfaces.md`` §7 / §7.4 / §7.5 and the Task 2.A4 brief
in ``designs/smai/implementation_plan.md`` §3.3. Substrate is **Docker**
per commit a9e57bd; compatible OCI runtimes (Podman, Apple ``container``,
containerd via ``nerdctl``) typically alias the ``docker`` binary on
``$PATH`` and work transparently.

Registered via the ``smai.computes`` entry-point group::

    [project.entry-points."smai.computes"]
    localgpu = "smai_compute_localgpu:LocalGpuCompute"

Tier A integrators (the in-tree CLI / hosted backend) instantiate the
plugin through the entry-point discovery flow owned by ``smai-cli`` (see
Task 2.C3); Tier B integrators import :class:`LocalGpuCompute` directly.
"""

from smai_compute_localgpu._compute import (
    DEFAULT_RUNTIME_CPU_IMAGE,
    DEFAULT_RUNTIME_IMAGE,
    LocalGpuCompute,
)

__all__ = [
    "DEFAULT_RUNTIME_CPU_IMAGE",
    "DEFAULT_RUNTIME_IMAGE",
    "LocalGpuCompute",
]
