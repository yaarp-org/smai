"""Per-resource APIRouter modules per ``designs/smai/11-api.md`` §4.

One module per resource; :func:`smai_api.make_api_app` includes each
router. Per-resource handlers translate HTTP into Runtime service
calls and return ``smai_api_spec`` response models.
"""

from __future__ import annotations
