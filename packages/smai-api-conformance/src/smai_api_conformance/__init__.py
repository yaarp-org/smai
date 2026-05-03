"""Parameterizable API-contract conformance suite for ``smai-api-spec``.

Per DEC-037 / ``designs/smai/11-api.md`` §10: any HTTP API that claims
to implement the ``smai-api-spec`` contract proves it by subclassing
:class:`APIConformanceBase`, overriding the ``client`` fixture to
point at the implementation, and running pytest. The inherited
contract methods exercise every endpoint in ``11-api.md`` §4 plus the
cross-cutting concerns (errors / pagination / auth / SSE).

See the package README for the full opt-in pattern, the scope
boundary (shape vs lifecycle), and the configuration knobs.
"""

from __future__ import annotations

from smai_api_conformance._base import APIConformanceBase

__all__ = ["APIConformanceBase"]
