"""Compute plugin: RunPod REST-API implementation.

Per ``designs/smai/07-plugin-interfaces.md`` §7 and the Task 3.F4 brief
in ``designs/smai/implementation_plan.md`` §3.4.

Pairs with :class:`smai_compute_localgpu.LocalGpuCompute` (single-host
Docker reference) and :class:`smai_compute_modal.ModalCompute` (Modal
Sandboxes) as the third v1 :class:`Compute` plugin — RunPod is the
GPU-cloud option for operators who want a pay-per-pod surface without
running their own GPU host.

Registered via the ``smai.computes`` entry-point group::

    [project.entry-points."smai.computes"]
    runpod = "smai_compute_runpod:RunPodCompute"

Tier A integrators (the in-tree CLI / hosted backend) instantiate the
plugin through entry-point discovery; Tier B integrators import
:class:`RunPodCompute` directly.
"""

from smai_compute_runpod._compute import (
    DEFAULT_API_BASE,
    DEFAULT_GPU_TYPE_ID,
    GPU_DISPATCH,
    RunPodCompute,
)

__all__ = [
    "DEFAULT_API_BASE",
    "DEFAULT_GPU_TYPE_ID",
    "GPU_DISPATCH",
    "RunPodCompute",
]
