"""Re-export shim for the shared row ↔ record helpers.

Per Task 3.H2: the helpers live in
:mod:`smai_orchestrator.migrations.serde` so the Postgres plugin can
import them without cross-importing this plugin. This shim preserves
the prior ``from smai_store_sqlite._serde import row_to_record``
import surface for any external caller.
"""

from __future__ import annotations

from smai_orchestrator.migrations.serde import record_to_row, row_to_record

__all__ = ["record_to_row", "row_to_record"]
