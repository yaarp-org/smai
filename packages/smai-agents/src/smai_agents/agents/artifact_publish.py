"""Workspace -> :class:`ArtifactStore` publishing for in-process agents.

Round 14: the harness-builder and technique-implementer agents run
in-process in the worker (no container). Their workspace lives on the
worker's local disk; the orchestrator gates that decide
``implementing -> implemented`` / ``implementation_failed`` read the
agent's outputs from :class:`ArtifactStore` keys. The container model
was always meant to land those outputs in the store; nothing ever did
(the only writers were the ``emit_harness_manifest`` tool for the
manifest and the loop's conversation-trace / status writes). This
module closes that gap: after an in-process session the dispatch
handler calls :func:`publish_workspace_outputs` to copy the agent's
output files from the local workspace into the store at their
canonical keys.

The mapping is purely mechanical — a workspace-relative path ``r``
under one of the requested ``roots`` is published at
``f"{key_prefix}/{r}"`` — so the keys line up with what the existing
consumers already read:

* harness builder: ``harness/**`` + ``techniques/baseline.py`` +
  ``validation_results.json`` under
  ``comparison-groups/{cg_id}/harness/`` — the prefix
  :func:`smai_agents.agents.technique_implementer._load_harness_outputs`
  drains and the code reviewer scans for ``.py`` files.
* technique implementer: ``techniques/**`` + ``validation_results.json``
  under ``comparison-groups/{cg_id}/entries/{entry_id}/code/`` — the
  keys the entry-spec validation gate + the code reviewer read.

Fixed runtime templates and ``contracts/`` are deliberately NOT
published: the technique-implementer runner regenerates the templates
(:func:`smai_runtime.write_template_files`) and is handed the contracts
as in-memory objects, so transferring them through the store would be
redundant.
"""

from __future__ import annotations

from pathlib import Path

from smai_core.plugins import ArtifactStore

_CONTENT_TYPE_BY_SUFFIX: dict[str, str] = {
    ".json": "application/json",
    ".py": "text/x-python",
    ".txt": "text/plain",
}


async def publish_workspace_outputs(
    *,
    artifact_store: ArtifactStore,
    workspace_path: Path,
    key_prefix: str,
    roots: list[str],
) -> list[str]:
    """Publish selected workspace files to :class:`ArtifactStore`.

    For each entry in ``roots`` (a workspace-relative file or directory
    name): a file is published directly; a directory is walked
    recursively and every file under it is published. The store key is
    ``f"{key_prefix}/{workspace_relative_posix_path}"``.

    Publishing is best-effort per file — a single failed
    :meth:`ArtifactStore.put` is swallowed so a partial agent run still
    lands whatever it did produce (debugging artifacts), and the
    downstream completeness check / gates decide success from what is
    actually present. Returns the list of keys successfully published.
    """
    published: list[str] = []
    for root in roots:
        base = workspace_path / root
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = sorted(p for p in base.rglob("*") if p.is_file())
        else:
            continue
        for file_path in files:
            rel = file_path.relative_to(workspace_path).as_posix()
            key = f"{key_prefix}/{rel}"
            content_type = _CONTENT_TYPE_BY_SUFFIX.get(file_path.suffix, "application/octet-stream")
            try:
                body = file_path.read_bytes()
                await artifact_store.put(key, body, content_type=content_type)
            except Exception:  # noqa: BLE001 — best-effort; a bad file must not abort the rest
                continue
            published.append(key)
    return published


__all__ = ["publish_workspace_outputs"]
