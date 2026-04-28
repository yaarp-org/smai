"""``emit_harness_manifest`` — harness-builder-only structured tool.

Per ``04-agents.md`` §9 and ``10-runtime-and-templates.md`` §5. The
harness builder agent calls this tool *after* it has written the harness
code and run a few-epoch validation that passed; the tool's handler
validates the manifest the agent declared, runs the runtime's manifest
conformance check, freezes the ``content_hash`` deterministically, and
writes the result to :class:`ArtifactStore`.

Per §9 and §10 OQ #10, the tool is registered ONLY for the harness
builder role — the technique implementer never emits a manifest. The
factory shape (``make_emit_harness_manifest_tool(...)``) lets the
dispatch handler bind the per-CG anchors (the loaded
:class:`HarnessContract`, the :class:`ArtifactStore` key the manifest is
written under, the workspace-path harness directory the agent's writes
landed in) at registration time, so the handler stays purely
agent-facing.

Validation discipline (§9):

1. ``harness_version_hash`` must equal
   :func:`smai_runtime.compute_harness_version_hash` over the workspace's
   ``harness/`` directory contents. Mismatch — the agent regenerated the
   hash from memory rather than from the actual files — is a tool error
   so the agent rebuilds it from the on-disk surface.

2. ``parent_harness_contract_hash`` must equal the loaded
   :class:`HarnessContract`'s ``envelope.content_hash``. Mismatch means
   the manifest was emitted against a stale contract; the tool surfaces
   the divergence rather than letting downstream consumers read a
   misaligned manifest.

3. ``runtime_template_version`` must equal
   :data:`smai_runtime.RUNTIME_TEMPLATE_VERSION`. The runtime that
   later consumes the manifest (§5.4) refuses a too-old / too-new
   template version anyway; the tool catches the mismatch upstream so
   the agent self-corrects on its next turn rather than at run start
   after compute is consumed.

4. ``manifest_schema_version`` must equal
   :data:`smai_runtime.MANIFEST_SCHEMA_VERSION` (§5.1).

5. ``validation_results.json`` must exist in the workspace and report
   ``passed: true`` (§9 step 3 / §2.2). The tool reads the file and
   inspects the canonical ``passed`` boolean key — the validation
   protocol writes it via :func:`smai_runtime.write_validation_results`.

6. The runtime's manifest-conformance check fires
   :func:`smai_runtime.assert_additive_baseline_has_no_required_extension_points`
   when the harness contract's factor type is ``additive`` (§9.1) — an
   additive factor that declares a required extension point is
   internally inconsistent and must surface to the agent rather than be
   accepted into the artifact store.

On every validation failure the handler returns a tool result with
``is_error=True`` carrying a brief diagnostic; the agent observes the
error on its next turn and self-corrects. On success the handler runs
:func:`smai_runtime.freeze_manifest` to recompute ``content_hash``
deterministically and writes the JSON to :class:`ArtifactStore` at the
configured key.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel
from smai_core import HarnessContract
from smai_core.plugins import ToolResultContent
from smai_runtime import (
    HARNESS_API_MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_TEMPLATE_VERSION,
    VALIDATION_RESULTS_FILENAME,
    HarnessAPIManifest,
    assert_additive_baseline_has_no_required_extension_points,
    compute_harness_version_hash,
    freeze_manifest,
)

from smai_agents.tools import Tool, ToolContext

EMIT_HARNESS_MANIFEST_TOOL_NAME = "emit_harness_manifest"

# Subdirectory the harness builder is expected to write its harness Python
# modules into. The ``compute_harness_version_hash`` input is the bytes of
# every file under this directory, recursively (per §5.3 canonicalization).
_HARNESS_SUBDIR = "harness"


def _read_harness_files(workspace: Path) -> dict[str, bytes]:
    """Collect ``{relative_path: bytes}`` for every regular file under
    ``<workspace>/harness/``.

    Paths are relative to the harness directory so the digest is stable
    across workspaces. Sorted iteration is the caller's job
    (:func:`compute_harness_version_hash` sorts internally).
    """
    base = workspace / _HARNESS_SUBDIR
    if not base.is_dir():
        return {}
    files: dict[str, bytes] = {}
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        files[rel] = path.read_bytes()
    return files


def _read_validation_results(workspace: Path) -> tuple[bool, str]:
    """Read ``validation_results.json`` and decide pass / fail.

    Returns ``(passed, diagnostic)``. ``passed`` is ``True`` only when
    the file exists, parses as a JSON object, and carries a top-level
    ``"passed": true`` key; every other shape is a structural failure
    the tool surfaces verbatim.
    """
    target = workspace / VALIDATION_RESULTS_FILENAME
    if not target.is_file():
        return (
            False,
            f"{VALIDATION_RESULTS_FILENAME} not found in workspace; the agent "
            "must run a passing validation via run_experiment before emitting "
            "the manifest (see 04-agents.md §9).",
        )
    try:
        raw = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"failed to read {VALIDATION_RESULTS_FILENAME}: {exc}"
    try:
        body: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            False,
            f"{VALIDATION_RESULTS_FILENAME} is not valid JSON: {exc}",
        )
    if not isinstance(body, dict):
        return (
            False,
            f"{VALIDATION_RESULTS_FILENAME} root must be an object, got {type(body).__name__}",
        )
    passed: object = body.get("passed")  # type: ignore[reportUnknownMemberType]
    if passed is not True:
        return (
            False,
            f"{VALIDATION_RESULTS_FILENAME} reports passed != True; the tool "
            "only emits the manifest after a passing validation run.",
        )
    return True, ""


def _error(message: str) -> ToolResultContent:
    return ToolResultContent(tool_use_id="", content=message, is_error=True)


def make_emit_harness_manifest_tool(
    *,
    harness_contract: HarnessContract,
    artifact_path: str,
) -> Tool:
    """Build the :func:`emit_harness_manifest` tool (§9).

    Args:
        harness_contract: The loaded :class:`HarnessContract` for the CG
            this harness builder is dispatched against. The tool's
            handler reads ``envelope.content_hash`` to validate the
            manifest's ``parent_harness_contract_hash`` and reads
            ``body.factor.type`` to fire the additive-factor sanity
            check.
        artifact_path: :class:`ArtifactStore` key under which the
            handler writes the frozen manifest JSON. The orchestrator's
            dispatch handler establishes this path; the tool itself
            stays storage-shape-agnostic.

    Returns:
        A :class:`Tool` whose ``input_schema`` is
        :class:`smai_runtime.HarnessAPIManifest`. The tool is registered
        ONLY for the harness builder role per §9 / §10 OQ #10.
    """
    expected_contract_hash = harness_contract.envelope.content_hash

    async def _handler(
        parsed_input: BaseModel,
        context: ToolContext,
    ) -> ToolResultContent:
        if not isinstance(parsed_input, HarnessAPIManifest):
            raise TypeError(
                "emit_harness_manifest handler expected HarnessAPIManifest, "
                f"got {type(parsed_input).__name__}"
            )

        # === Step 4 — schema-version compatibility (§5.1) ====================
        if parsed_input.manifest_schema_version != MANIFEST_SCHEMA_VERSION:
            return _error(
                f"manifest_schema_version {parsed_input.manifest_schema_version} "
                f"does not match runtime expectation {MANIFEST_SCHEMA_VERSION}; "
                "the runtime that consumes this manifest would reject it."
            )

        # === Step 3 — runtime template version (§5.4 / §7) ===================
        if parsed_input.runtime_template_version != RUNTIME_TEMPLATE_VERSION:
            return _error(
                f"runtime_template_version {parsed_input.runtime_template_version!r} "
                f"does not match the runtime's template version "
                f"{RUNTIME_TEMPLATE_VERSION!r}; the manifest would be rejected at "
                "run start before any training compute is consumed."
            )

        # === Step 2 — parent_harness_contract_hash (§9 step 2) ===============
        if parsed_input.parent_harness_contract_hash != expected_contract_hash:
            return _error(
                f"parent_harness_contract_hash {parsed_input.parent_harness_contract_hash!r} "
                f"does not match the loaded HarnessContract.envelope.content_hash "
                f"{expected_contract_hash!r}; the manifest was emitted against a "
                "stale or wrong contract."
            )

        # === Step 1 — harness_version_hash (§9 step 1 / §5.3) ================
        harness_files = _read_harness_files(context.workspace_path)
        if not harness_files:
            return _error(
                f"workspace harness/ directory at "
                f"{context.workspace_path / _HARNESS_SUBDIR} contains no files; "
                "the harness builder must write at least one harness module before "
                "emitting the manifest."
            )
        actual_harness_hash = compute_harness_version_hash(harness_files)
        if parsed_input.harness_version_hash != actual_harness_hash:
            return _error(
                f"harness_version_hash {parsed_input.harness_version_hash!r} does not "
                f"match the actual SHA-256 of harness/ contents "
                f"{actual_harness_hash!r}; regenerate the hash from the on-disk "
                "files (do not derive it from memory)."
            )

        # === Step 5 — validation_results.json reflects pass (§9 step 3) ======
        passed, diagnostic = _read_validation_results(context.workspace_path)
        if not passed:
            return _error(diagnostic)

        # === Step 6 — manifest-conformance (§9 step 4) =======================
        try:
            assert_additive_baseline_has_no_required_extension_points(
                harness_contract,
                parsed_input,
            )
        except ValueError as exc:
            return _error(f"manifest conformance check failed: {exc}")

        # === Freeze + write (§9 step 5 / §5.3) ===============================
        frozen = freeze_manifest(parsed_input)
        body = frozen.model_dump_json(indent=2).encode("utf-8")

        store = context.artifact_store
        if store is None:
            return _error(
                "emit_harness_manifest requires an ArtifactStore on the agent "
                "session; the dispatch handler did not provide one."
            )
        await store.put(artifact_path, body, content_type="application/json")

        # Mirror the manifest into the workspace's contracts/ directory so
        # the runtime's `experiment.py` (and downstream technique implementer
        # workspaces materialized from this harness) can read it from the
        # canonical workspace layout per ``10-runtime-and-templates.md`` §2.
        contracts_dir = context.workspace_path / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)
        (contracts_dir / HARNESS_API_MANIFEST_FILENAME).write_text(
            frozen.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return ToolResultContent(
            tool_use_id="",
            content=(
                f"manifest written: artifact_path={artifact_path!r}, "
                f"content_hash={frozen.content_hash!r}, "
                f"extension_points={len(frozen.extension_points)}"
            ),
            is_error=False,
        )

    return Tool(
        name=EMIT_HARNESS_MANIFEST_TOOL_NAME,
        description=(
            "Emit the HarnessAPIManifest after the harness has been written "
            "and validation has passed. Inputs follow the HarnessAPIManifest "
            "Pydantic schema (see smai_runtime.manifest). The handler validates "
            "harness_version_hash against harness/ contents, "
            "parent_harness_contract_hash against the loaded HarnessContract, "
            "runtime_template_version against the runtime, and "
            "validation_results.json reflects pass. On any mismatch the tool "
            "returns is_error=True and the agent self-corrects. On success the "
            "manifest is written to ArtifactStore with content_hash frozen."
        ),
        input_schema=HarnessAPIManifest,
        handler=_handler,
    )


__all__ = [
    "EMIT_HARNESS_MANIFEST_TOOL_NAME",
    "make_emit_harness_manifest_tool",
]
