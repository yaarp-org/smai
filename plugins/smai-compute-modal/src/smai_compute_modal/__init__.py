"""Compute plugin: Modal Sandboxes implementation.

Per ``07-plugin-interfaces.md`` §7 / §7.4 / §7.5, DEC-021 (Modal
Sandboxes carry-forward), and ``modal_migration.md`` (the v1 reference).
The Phase 3 production plugin per the implementation_plan §3.4 Task
3.F3.

Registered via the ``smai.computes`` entry-point group::

    [project.entry-points."smai.computes"]
    modal = "smai_compute_modal:ModalCompute"

Tier A integrators (the in-tree CLI / hosted backend) instantiate the
plugin through the entry-point discovery flow owned by ``smai-cli`` (see
Task 3.G3); Tier B integrators import :class:`ModalCompute` directly.
"""

from smai_compute_modal._compute import ModalCompute

__all__ = ["ModalCompute"]
