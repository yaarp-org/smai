""":class:`ComputeRequirements` — the per-experiment compute-substrate
hints lifted onto :class:`ControlledConditions`.

Hung off :class:`ControlledConditions` because compute hardware is a
ceteris-paribus property: across the entries of a single CG the same
hardware is held constant (running one entry on GPU and another on CPU
is itself a confound for almost any performance-relevant metric). The
field therefore lives next to ``dataset`` / ``optimization`` / ``seeds``
rather than as a top-level deployment-knob.

v2 Phase 1 (this revision) ships a single ``gpu: bool`` hint with a
``True`` default that preserves pre-Phase-1 dispatch behavior — every
``Compute.submit`` was hardcoded ``gpu=True`` before this surface
existed, so the default keeps existing experiment YAMLs and
already-emitted :class:`HarnessContract` artifacts working unchanged.

Phase 2 (post-M5 backlog item) widens this to ``gpu_type`` / ``cpu`` /
``memory_mb`` / ``timeout_seconds`` once the LocalGpu / Modal / RunPod
plugin-surface translation is scoped — the same field grows, the
dispatcher's read site doesn't move.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ComputeRequirements(BaseModel):
    """Per-experiment compute-substrate hints (`02-dsl-and-contracts.md` §7.4).

    ``gpu``: whether the per-(entry × seed) training jobs require GPU
    passthrough. Defaults to ``True`` (the pre-Phase-1 hardcoded
    behavior). Set to ``False`` for CPU-only experiments — methodology
    smoke runs, kNN comparisons, small MLPs, anything where the GPU
    isn't load-bearing — which is also the unblock for the
    ``smai-compute-localgpu`` plugin on macOS (Docker Desktop has no
    GPU passthrough on Darwin and refuses ``gpu=True`` eagerly).
    """

    model_config = ConfigDict(extra="forbid")

    gpu: bool = True


__all__ = ["ComputeRequirements"]
