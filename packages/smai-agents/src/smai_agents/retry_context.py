"""DEC-023 retry-context helper.

Per ``04-agents.md`` §8 / DEC-023. When the orchestrator routes an
entry back through implementation after a code-review failure, the
retry agent receives the previous attempt's ``conversation-trace.json``
as a workspace file at ``prev-conversation-trace.json``.

The implementer agent's prompt-config variant (in 2.B2 / 2.B3) directs
the agent to read the file selectively; this module ships only the
file-copy mechanism. The agent doesn't manage the wait; the loop /
dispatch handler does.
"""

from __future__ import annotations

from pathlib import Path

from smai_core.plugins import ArtifactNotFound, ArtifactStore

PREV_CONVERSATION_TRACE_FILENAME = "prev-conversation-trace.json"


async def load_retry_context(
    *,
    workspace_path: Path,
    artifact_store: ArtifactStore,
    artifact_path: str,
) -> Path | None:
    """Copy the previous attempt's conversation trace into the workspace.

    Reads ``artifact_path`` from :class:`ArtifactStore` and writes its
    bytes to ``<workspace_path>/prev-conversation-trace.json``. Returns
    the written path on success, ``None`` if the artifact does not
    exist (no prior attempt, no retry context to load).

    Other I/O errors propagate — a corrupt or unreadable trace is a
    real fault the dispatch handler should surface, not a "fall back to
    no retry context" condition (per DEC-018's no-silent-fallback
    discipline; the principle generalizes beyond structured_call).

    The destination filename is fixed by §8's contract — the prompt
    variant for retry attempts hardcodes the ``prev-conversation-trace.json``
    name when nudging the agent to read the file.
    """
    try:
        data = await artifact_store.get(artifact_path)
    except ArtifactNotFound:
        return None

    workspace_path.mkdir(parents=True, exist_ok=True)
    dest = workspace_path / PREV_CONVERSATION_TRACE_FILENAME
    dest.write_bytes(data)
    return dest


__all__ = ["PREV_CONVERSATION_TRACE_FILENAME", "load_retry_context"]
