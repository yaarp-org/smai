"""Pydantic model tests for plugin Protocol supporting types.

Per Task 1.8 — validate hand-crafted fixtures, JSON round-trip, reject
malformed input. Covers normalized types, capability flags, error
classes, and the shared :class:`CursorPage` / :class:`LeaseToken`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from smai_core.entities import Entry, Level
from smai_core.plugins import (
    ArtifactStoreCapabilities,
    CacheConfig,
    ComputeCapabilities,
    CursorPage,
    JobHandle,
    JobStatus,
    LeaseToken,
    LlmCapabilities,
    MetadataStoreCapabilities,
    ModelResponse,
    NormalizedMessage,
    TextContent,
    TokenUsage,
    ToolDefinition,
    ToolResultContent,
    ToolUseContent,
)

# ---------- LlmProvider normalized types -------------------------------------


def test_text_content_round_trip() -> None:
    msg = TextContent(text="hello")
    assert msg.type == "text"
    raw = msg.model_dump_json()
    parsed = TextContent.model_validate_json(raw)
    assert parsed == msg


def test_tool_use_content_round_trip() -> None:
    msg = ToolUseContent(id="abc", name="search", input={"q": "foo"})
    assert msg.type == "tool_use"
    raw = msg.model_dump_json()
    parsed = ToolUseContent.model_validate_json(raw)
    assert parsed == msg
    assert parsed.input == {"q": "foo"}


def test_tool_result_content_default_is_error_false() -> None:
    msg = ToolResultContent(tool_use_id="abc", content="42")
    assert msg.is_error is False


def test_normalized_message_user_role_with_text_only() -> None:
    msg = NormalizedMessage(
        role="user",
        content=[TextContent(text="hi")],
    )
    assert msg.role == "user"
    assert len(msg.content) == 1


def test_normalized_message_assistant_with_mixed_content() -> None:
    """Assistant turns can mix text and tool_use blocks."""
    msg = NormalizedMessage(
        role="assistant",
        content=[
            TextContent(text="searching..."),
            ToolUseContent(id="t1", name="search", input={"q": "x"}),
        ],
    )
    raw = msg.model_dump_json()
    parsed = NormalizedMessage.model_validate_json(raw)
    assert parsed == msg


def test_normalized_message_rejects_bad_role() -> None:
    with pytest.raises(ValidationError):
        NormalizedMessage(role="system", content=[])  # type: ignore[arg-type]


def test_token_usage_defaults_cache_to_zero() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    assert usage.cache_read_tokens == 0
    assert usage.cache_write_tokens == 0


def test_model_response_round_trip() -> None:
    resp = ModelResponse(
        message=NormalizedMessage(role="assistant", content=[TextContent(text="ok")]),
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1, output_tokens=2),
    )
    raw = resp.model_dump_json()
    parsed = ModelResponse.model_validate_json(raw)
    assert parsed == resp


def test_model_response_rejects_bad_stop_reason() -> None:
    with pytest.raises(ValidationError):
        ModelResponse(
            message=NormalizedMessage(role="assistant", content=[]),
            stop_reason="goodbye",  # type: ignore[arg-type]
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )


def test_tool_definition_input_schema_is_dict() -> None:
    td = ToolDefinition(
        name="search",
        description="search the web",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    assert td.input_schema["type"] == "object"


# ---------- Capability flags -------------------------------------------------


def test_llm_capabilities_round_trip() -> None:
    caps = LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=4096,
        model_id="us.anthropic.claude-opus-4-7-v1",
    )
    assert caps.supports_tool_use is True  # default
    raw = caps.model_dump_json()
    parsed = LlmCapabilities.model_validate_json(raw)
    assert parsed == caps


def test_llm_capabilities_rejects_extra_fields() -> None:
    """``extra="forbid"`` is the smai-core convention — typo guard."""
    with pytest.raises(ValidationError):
        LlmCapabilities.model_validate(
            {
                "supports_caching": True,
                "context_window": 1,
                "max_output_tokens": 1,
                "model_id": "x",
                "supports_caching_typo": True,
            },
        )


def test_metadata_store_capabilities_defaults() -> None:
    caps = MetadataStoreCapabilities(is_tenant_aware=False)
    assert caps.supports_transactions is True


def test_artifact_store_capabilities_unbounded_size() -> None:
    caps = ArtifactStoreCapabilities(
        supports_presigned_urls=False,
        max_object_size_bytes=None,
    )
    assert caps.max_object_size_bytes is None


def test_compute_capabilities_default_no_log_streaming() -> None:
    caps = ComputeCapabilities(supports_gpu=True, max_timeout_seconds=3600)
    assert caps.supports_log_streaming is False


# ---------- Cache config -----------------------------------------------------


def test_cache_config_defaults_all_off() -> None:
    cc = CacheConfig()
    assert cc.cache_static_prefix is False
    assert cc.cache_initial_message is False
    assert cc.rolling_cache_count == 0


def test_cache_config_round_trip() -> None:
    cc = CacheConfig(
        cache_static_prefix=True,
        cache_initial_message=True,
        rolling_cache_count=3,
    )
    raw = cc.model_dump_json()
    parsed = CacheConfig.model_validate_json(raw)
    assert parsed == cc


# ---------- JobHandle / JobStatus --------------------------------------------


def test_job_handle_default_metadata_independent_per_instance() -> None:
    """Sanity: ``Field(default_factory=dict)`` shouldn't share state."""
    a = JobHandle(plugin="x", handle="1")
    b = JobHandle(plugin="x", handle="2")
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_job_handle_round_trip() -> None:
    h = JobHandle(plugin="modal", handle="sb_abc", metadata={"region": "us-east-1"})
    raw = h.model_dump_json()
    parsed = JobHandle.model_validate_json(raw)
    assert parsed == h


def test_job_status_round_trip_minimal() -> None:
    s = JobStatus(
        state="submitted",
        exit_code=None,
        started_at=None,
        finished_at=None,
        failure_reason=None,
    )
    raw = s.model_dump_json()
    parsed = JobStatus.model_validate_json(raw)
    assert parsed == s


def test_job_status_round_trip_with_iso_timestamps() -> None:
    s = JobStatus(
        state="failed",
        exit_code=137,
        started_at="2026-04-28T12:00:00Z",
        finished_at="2026-04-28T12:05:00Z",
        failure_reason="OOM",
    )
    raw = s.model_dump_json()
    parsed = JobStatus.model_validate_json(raw)
    assert parsed == s


def test_job_status_rejects_invalid_state() -> None:
    with pytest.raises(ValidationError):
        JobStatus(
            state="cooking",  # type: ignore[arg-type]
            exit_code=0,
            started_at=None,
            finished_at=None,
            failure_reason=None,
        )


# ---------- LeaseToken -------------------------------------------------------


def test_lease_token_round_trip_preserves_nonce() -> None:
    """DEC-035 #2: ``nonce`` is the canonical CAS gate; survives JSON
    round-trip exactly."""
    now = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
    token = LeaseToken(
        entity_kind="cg",
        entity_id="cg-123",
        acquired_at=now,
        expires_at=now,
        lease_holder_id="worker-host-pid",
        nonce="9d2a1b3c-7f4e-4d6a-8c1b-2e3f4a5b6c7d",
    )
    raw = token.model_dump_json()
    parsed = LeaseToken.model_validate_json(raw)
    assert parsed.nonce == token.nonce
    assert parsed == token


def test_lease_token_rejects_bad_entity_kind() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        LeaseToken(
            entity_kind="bogus",  # type: ignore[arg-type]
            entity_id="x",
            acquired_at=now,
            expires_at=now,
            lease_holder_id="w",
            nonce="n",
        )


# ---------- CursorPage[T] generic --------------------------------------------


def test_cursor_page_empty() -> None:
    page: CursorPage[str] = CursorPage(items=[])
    assert page.items == []
    assert page.next_cursor is None
    assert page.total is None


def test_cursor_page_str_round_trip() -> None:
    page: CursorPage[str] = CursorPage(items=["a", "b", "c"], next_cursor="opaque", total=42)
    raw = page.model_dump_json()
    parsed = CursorPage[str].model_validate_json(raw)
    assert parsed.items == page.items
    assert parsed.next_cursor == "opaque"
    assert parsed.total == 42


def test_cursor_page_parameterizes_with_methodology_entity() -> None:
    """Per Task 1.8 deliverable #5: ``CursorPage[Entry]`` parameterizes
    cleanly. ``Entry`` is a real methodology entity (the CG-execution
    pipeline doesn't ship records yet, so the test uses a real type
    that does)."""
    entry = Entry(
        id="e1",
        is_baseline=True,
        level=Level(factor="augmentation", name="none"),
    )
    page: CursorPage[Entry] = CursorPage(items=[entry])
    assert page.items[0].id == "e1"
    raw = page.model_dump_json()
    parsed = CursorPage[Entry].model_validate_json(raw)
    assert parsed.items[0].id == "e1"


def test_cursor_page_total_is_optional() -> None:
    """Per DEC-035 #1: ``total`` is optional; plugins MAY populate it,
    consumers don't depend on it."""
    page: CursorPage[str] = CursorPage(items=["a"], next_cursor=None)
    raw = page.model_dump_json()
    parsed = CursorPage[str].model_validate_json(raw)
    assert parsed.total is None


def test_cursor_page_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CursorPage[str].model_validate(
            {"items": [], "page_number": 3},
        )
