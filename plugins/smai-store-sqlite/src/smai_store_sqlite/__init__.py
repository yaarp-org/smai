"""MetadataStore plugin: SQLite reference local implementation.

Per ``designs/smai/07-plugin-interfaces.md`` §5, DEC-030 (SQL-shape
only), DEC-036 (SQLAlchemy 2.0 async Core; Alembic deferred to Task
3.H2). The plugin implements the 47-method :class:`MetadataStore`
Protocol surface plus the 10-method :class:`Transaction` Protocol;
schema covers six pipeline-tracking tables (per ``01-data-model.md``
§5), one methodology lookup mirror, and three normalized cost tables
per DEC-033.

Entry point::

    [project.entry-points."smai.metadata_stores"]
    sqlite = "smai_store_sqlite:SqliteStore"
"""

from smai_store_sqlite._store import SqliteStore

__all__ = ["SqliteStore"]
