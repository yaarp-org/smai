"""Tests for the harness-builder-only ``emit_harness_manifest`` tool.

Per ``04-agents.md`` §9 and ``10-runtime-and-templates.md`` §5. The
tool's handler is the production handoff: validate the agent's manifest
against the on-disk harness state, the loaded HarnessContract, and the
runtime version pins; freeze ``content_hash``; write to ArtifactStore.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _agent_fakes import StubArtifactStore  # type: ignore[import-not-found]
from _b3_fakes import (  # type: ignore[import-not-found]
    SAMPLE_HARNESS_FILES,
    make_harness_contract,
    make_minimal_manifest,
)
from smai_agents import (
    EMIT_HARNESS_MANIFEST_TOOL_NAME,
    AgentLoopConfig,
    AgentSession,
    ToolContext,
    ToolRegistry,
    make_emit_harness_manifest_tool,
    make_finish_tool,
)
from smai_core.plugins import LlmCapabilities, NormalizedMessage, TextContent
from smai_runtime import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_TEMPLATE_VERSION,
    VALIDATION_RESULTS_FILENAME,
    HarnessAPIManifest,
)


class _NoopProvider:
    """Minimal :class:`LlmProvider` stub — emit_manifest tests don't drive
    the loop, so the provider only needs to type-check."""

    def __init__(self) -> None:
        self.name = "noop"
        self.capabilities = LlmCapabilities(
            supports_caching=False,
            context_window=200_000,
            max_output_tokens=4_096,
            supports_tool_use=True,
            model_id="noop:test",
        )

    async def call(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("emit_manifest tests do not exercise LLM calls")


def _make_workspace_with_harness(workspace: Path, files: dict[str, bytes]) -> None:
    """Lay down ``harness/<file>`` for every entry in ``files``."""
    harness_dir = workspace / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        target = harness_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _write_validation_passed(workspace: Path) -> None:
    (workspace / VALIDATION_RESULTS_FILENAME).write_text(
        json.dumps({"passed": True, "notes": "ok"}),
        encoding="utf-8",
    )


def _make_context(
    workspace: Path,
    artifact_store: StubArtifactStore,
    tool_registry: ToolRegistry,
) -> ToolContext:
    """Build a :class:`ToolContext` for direct handler invocation."""
    session = AgentSession(
        system_prompt="test",
        messages=[
            NormalizedMessage(
                role="user",
                content=[TextContent(text="x")],
            )
        ],
        tools=tool_registry,
        llm_providers={"harness_builder": _NoopProvider()},  # type: ignore[dict-item]
        current_role="harness_builder",
        workspace_path=workspace,
        artifact_store=artifact_store,
        config=AgentLoopConfig(status_write_every_turns=0),
    )
    return ToolContext(
        workspace_path=workspace,
        artifact_store=artifact_store,
        compute=None,
        session=session,
    )


# === Happy path =============================================================


@pytest.mark.asyncio
async def test_emit_manifest_happy_path_writes_to_artifact_store(
    tmp_path: Path,
) -> None:
    """A correctly-shaped manifest passes every check and is written
    to ArtifactStore with content_hash frozen."""
    contract = make_harness_contract(factor_type="additive")
    _make_workspace_with_harness(tmp_path, SAMPLE_HARNESS_FILES)
    _write_validation_passed(tmp_path)
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )

    store = StubArtifactStore()
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    registry = ToolRegistry()
    registry.register(tool)
    registry.register(make_finish_tool())
    ctx = _make_context(tmp_path, store, registry)

    result = await tool.handler(manifest, ctx)
    assert result.is_error is False
    assert "manifest written" in result.content

    # ArtifactStore round-trip: the manifest body parses back to a
    # frozen HarnessAPIManifest with a populated content_hash.
    written_body = await store.get("cg-1/manifest.json")
    revived = HarnessAPIManifest.model_validate_json(written_body)
    assert revived.content_hash != ""
    assert revived.parent_harness_contract_hash == contract.envelope.content_hash

    # The manifest is also mirrored into workspace/contracts/ so the
    # runtime's experiment.py finds it from the canonical layout.
    on_disk = tmp_path / "contracts" / "harness_api_manifest.json"
    assert on_disk.is_file()


# === Hash-mismatch failures =================================================


@pytest.mark.asyncio
async def test_emit_manifest_rejects_mismatched_harness_version_hash(
    tmp_path: Path,
) -> None:
    """The agent declared ``harness_version_hash`` but the on-disk
    files don't match; tool returns is_error=True."""
    contract = make_harness_contract(factor_type="additive")
    _make_workspace_with_harness(tmp_path, SAMPLE_HARNESS_FILES)
    _write_validation_passed(tmp_path)

    # Build a manifest whose harness_version_hash is computed against a
    # *different* set of files.
    bogus_files = {"trainer.py": b"# different bytes\n"}
    bad_manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        harness_files=bogus_files,
        factor_type="additive",
    )

    store = StubArtifactStore()
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    registry = ToolRegistry()
    registry.register(tool)
    ctx = _make_context(tmp_path, store, registry)

    result = await tool.handler(bad_manifest, ctx)
    assert result.is_error is True
    assert "harness_version_hash" in result.content
    # No write happened — the ArtifactStore is empty.
    assert "cg-1/manifest.json" not in store._data  # noqa: SLF001 — fake exposes _data


@pytest.mark.asyncio
async def test_emit_manifest_rejects_mismatched_contract_hash(
    tmp_path: Path,
) -> None:
    """``parent_harness_contract_hash`` doesn't match the loaded contract."""
    contract = make_harness_contract(factor_type="additive")
    _make_workspace_with_harness(tmp_path, SAMPLE_HARNESS_FILES)
    _write_validation_passed(tmp_path)
    base_manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )
    bad_manifest = base_manifest.model_copy(
        update={"parent_harness_contract_hash": "deadbeef"},
    )

    store = StubArtifactStore()
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    registry = ToolRegistry()
    registry.register(tool)
    ctx = _make_context(tmp_path, store, registry)

    result = await tool.handler(bad_manifest, ctx)
    assert result.is_error is True
    assert "parent_harness_contract_hash" in result.content


# === Validation-results gate ================================================


@pytest.mark.asyncio
async def test_emit_manifest_rejects_when_validation_results_missing(
    tmp_path: Path,
) -> None:
    """No ``validation_results.json`` at workspace root → tool error."""
    contract = make_harness_contract(factor_type="additive")
    _make_workspace_with_harness(tmp_path, SAMPLE_HARNESS_FILES)
    # No write_validation_passed() — the file is absent.
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )

    store = StubArtifactStore()
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    registry = ToolRegistry()
    registry.register(tool)
    ctx = _make_context(tmp_path, store, registry)

    result = await tool.handler(manifest, ctx)
    assert result.is_error is True
    assert VALIDATION_RESULTS_FILENAME in result.content


@pytest.mark.asyncio
async def test_emit_manifest_rejects_when_validation_failed(
    tmp_path: Path,
) -> None:
    """``validation_results.json`` reports passed=false → tool error."""
    contract = make_harness_contract(factor_type="additive")
    _make_workspace_with_harness(tmp_path, SAMPLE_HARNESS_FILES)
    (tmp_path / VALIDATION_RESULTS_FILENAME).write_text(
        json.dumps({"passed": False, "notes": "loss didn't decrease"}),
        encoding="utf-8",
    )
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )

    store = StubArtifactStore()
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    registry = ToolRegistry()
    registry.register(tool)
    ctx = _make_context(tmp_path, store, registry)

    result = await tool.handler(manifest, ctx)
    assert result.is_error is True
    assert "passed" in result.content


# === Version pinning ========================================================


@pytest.mark.asyncio
async def test_emit_manifest_rejects_wrong_runtime_template_version(
    tmp_path: Path,
) -> None:
    contract = make_harness_contract(factor_type="additive")
    _make_workspace_with_harness(tmp_path, SAMPLE_HARNESS_FILES)
    _write_validation_passed(tmp_path)
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )
    bad = manifest.model_copy(update={"runtime_template_version": "9.9.9"})

    store = StubArtifactStore()
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    registry = ToolRegistry()
    registry.register(tool)
    ctx = _make_context(tmp_path, store, registry)

    result = await tool.handler(bad, ctx)
    assert result.is_error is True
    assert RUNTIME_TEMPLATE_VERSION in result.content


@pytest.mark.asyncio
async def test_emit_manifest_rejects_wrong_schema_version(
    tmp_path: Path,
) -> None:
    contract = make_harness_contract(factor_type="additive")
    _make_workspace_with_harness(tmp_path, SAMPLE_HARNESS_FILES)
    _write_validation_passed(tmp_path)
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="additive",
    )
    bad = manifest.model_copy(
        update={"manifest_schema_version": MANIFEST_SCHEMA_VERSION + 1},
    )

    store = StubArtifactStore()
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    registry = ToolRegistry()
    registry.register(tool)
    ctx = _make_context(tmp_path, store, registry)

    result = await tool.handler(bad, ctx)
    assert result.is_error is True
    assert "manifest_schema_version" in result.content


# === Conformance check (factor-type-aware) ==================================


@pytest.mark.asyncio
async def test_emit_manifest_rejects_additive_with_required_extension_point(
    tmp_path: Path,
) -> None:
    """Per §9.1: an additive factor cannot declare optional=False
    extension points. The runtime conformance check fires."""
    contract = make_harness_contract(factor_type="additive")
    _make_workspace_with_harness(tmp_path, SAMPLE_HARNESS_FILES)
    _write_validation_passed(tmp_path)

    # Build a substitutive-shaped manifest (one mandatory extension point)
    # but for an additive contract — internally inconsistent.
    manifest = make_minimal_manifest(
        parent_harness_contract_hash=contract.envelope.content_hash,
        factor_type="substitutive",
    )

    store = StubArtifactStore()
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    registry = ToolRegistry()
    registry.register(tool)
    ctx = _make_context(tmp_path, store, registry)

    result = await tool.handler(manifest, ctx)
    assert result.is_error is True
    assert "manifest conformance" in result.content


# === Tool registration discipline (§9 / §10 OQ #10) =========================


def test_emit_manifest_tool_uses_canonical_name() -> None:
    """The tool name matches the literal :data:`EMIT_HARNESS_MANIFEST_TOOL_NAME`."""
    contract = make_harness_contract(factor_type="additive")
    tool = make_emit_harness_manifest_tool(
        harness_contract=contract,
        artifact_path="cg-1/manifest.json",
    )
    assert tool.name == EMIT_HARNESS_MANIFEST_TOOL_NAME == "emit_harness_manifest"
    assert tool.input_schema is HarnessAPIManifest
