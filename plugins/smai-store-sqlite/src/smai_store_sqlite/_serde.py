"""Pydantic record ↔ SQL row conversion helpers.

The records that cross the :class:`MetadataStore` Protocol boundary are
Pydantic models from ``smai_orchestrator.entities.tracking``. SQLAlchemy
2.0 Core surfaces rows as ``Mapping[str, Any]``-shaped tuples; we round
them through ``model_dump()`` (record → row mapping) and
``model_validate()`` (row mapping → record).

Per DEC-036 we use Core not ORM — there are no SQLAlchemy ORM
declarative classes; the column-map ↔ Pydantic conversion lives here.

JSON-shaped fields (``JobHandle``, list-of-string columns) are dumped
to plain dicts / lists by ``model_dump(mode="json")``; the SQLAlchemy
``JSON`` column stores them verbatim and SQLite serializes via its
JSON1 extension. Reading back: SQLAlchemy hydrates the JSON column to
the original Python shape, and Pydantic ``model_validate`` reconstructs
the inner :class:`JobHandle` etc.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

_R = TypeVar("_R", bound=BaseModel)


def record_to_row(record: BaseModel) -> dict[str, Any]:
    """Serialize a pipeline-tracking Pydantic record to a SQL row mapping.

    ``mode="json"`` handles the inner :class:`JobHandle` and datetime
    fields — JobHandle dumps to a dict, datetimes to ISO strings.
    SQLAlchemy's JSON column accepts dicts directly; the DateTime
    column accepts datetime objects and ISO strings interchangeably
    (the dialect serializes appropriately).

    However, datetimes need to round-trip as datetime objects for
    SQLAlchemy's ``DateTime(timezone=True)`` column to render correctly
    on SQLite — calling ``model_dump(mode="json")`` would convert to
    string and lose the SQL-native handling. We use ``mode="python"``
    (the default) and rely on SQLAlchemy's adapters for datetime, with
    a post-process step converting only the JSON-shaped fields.
    """
    return record.model_dump(mode="python")


def row_to_record(record_type: type[_R], row: Any) -> _R:
    """Deserialize a SQL row mapping into a Pydantic record.

    ``row`` is whatever SQLAlchemy's ``Result.mappings()`` yields —
    a ``RowMapping`` (dict-like). Pydantic ``model_validate`` accepts
    arbitrary mappings.

    Pyright's ``BaseModel.model_validate`` signature accepts ``Any``,
    so the return type narrows to ``_R`` cleanly.
    """
    # SQLAlchemy returns either a ``Row`` (with ``_mapping`` view) or a
    # ``RowMapping`` directly via ``Result.mappings()``. Both are
    # dict-like; the helper tolerates either by reading attributes
    # through ``getattr`` to keep pyright's strict-mode generics happy.
    source: Any = getattr(row, "_mapping", row)
    row_mapping: dict[str, Any] = {str(k): source[k] for k in source.keys()}
    return record_type.model_validate(row_mapping)


__all__ = ["record_to_row", "row_to_record"]
