"""Fixed-template integration layer.

Per ``10-runtime-and-templates.md`` §3.3. The two template files
(``experiment.py``, ``techniques/__init__.py``) ship as resources in
:mod:`smai_runtime.templates._files`; they are byte-stable on disk and
hash-checked at run start (§7) against the package's expected hashes.

The resource files use a ``.template`` suffix so pyright does not type-check
them as part of smai-runtime's source tree — they are not importable Python
modules; they are byte-blobs the runtime drops into the workspace.

This module exposes nothing at import time; access the template bytes via
:func:`smai_runtime.read_template_bytes`.
"""
