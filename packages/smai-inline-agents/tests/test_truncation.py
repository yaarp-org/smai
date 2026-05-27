""":mod:`smai_inline_agents.truncation` — middle-truncation and threshold checks."""

from __future__ import annotations

from smai_core.plugins import (
    LlmCapabilities,
    NormalizedMessage,
    TextContent,
)
from smai_inline_agents import TruncationPolicy
from smai_inline_agents.truncation import (
    effective_limit,
    estimate_total_tokens,
    should_truncate,
    truncate_messages,
    truncation_threshold,
)


def _msg(role: str, text: str) -> NormalizedMessage:
    return NormalizedMessage(role=role, content=[TextContent(text=text)])  # type: ignore[arg-type]


# --- TruncationPolicy defaults ----------------------------------------------


def test_truncation_policy_defaults_match_v1() -> None:
    """§7: ``threshold_fraction=0.90``, ``truncate_at_fraction=0.85``,
    ``keep_head_messages=2``, ``keep_tail_messages=8``."""
    policy = TruncationPolicy()
    assert policy.threshold_fraction == 0.90
    assert policy.truncate_at_fraction == 0.85
    assert policy.keep_head_messages == 2
    assert policy.keep_tail_messages == 8


# --- Threshold arithmetic ----------------------------------------------------


def test_effective_limit_is_window_times_threshold_fraction() -> None:
    """For Opus (200K window): effective limit = 180K."""
    capabilities = LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=4096,
        model_id="opus",
    )
    assert effective_limit(capabilities, TruncationPolicy()) == 180_000


def test_truncation_threshold_fires_at_85_percent_of_effective() -> None:
    """For Opus: truncation fires at 153K (= 180K * 0.85)."""
    capabilities = LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=4096,
        model_id="opus",
    )
    assert truncation_threshold(capabilities, TruncationPolicy()) == 153_000


def test_truncation_scales_with_smaller_context_windows() -> None:
    """For 128K-window models: effective ~115.2K, fires at ~98K."""
    capabilities = LlmCapabilities(
        supports_caching=False,
        context_window=128_000,
        max_output_tokens=4096,
        model_id="qwen",
    )
    policy = TruncationPolicy()
    assert effective_limit(capabilities, policy) == 115_200
    assert truncation_threshold(capabilities, policy) == 97_920


# --- should_truncate gate ----------------------------------------------------


def test_should_truncate_false_when_well_below_threshold() -> None:
    capabilities = LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=4096,
        model_id="opus",
    )
    messages = [_msg("user", "small")]
    assert not should_truncate(
        messages=messages, capabilities=capabilities, policy=TruncationPolicy()
    )


def test_should_truncate_fires_when_estimated_tokens_cross_threshold() -> None:
    """Synthesize a large enough conversation to cross 153K tokens."""
    capabilities = LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=4096,
        model_id="opus",
    )
    big = "x" * 20_000  # ~5K tokens via the 4-char heuristic
    messages = [_msg("user", big) for _ in range(35)]  # ~175K tokens
    assert should_truncate(messages=messages, capabilities=capabilities, policy=TruncationPolicy())


def test_should_truncate_includes_system_prompt_overhead() -> None:
    """A short message list with a giant system prompt should still fire."""
    capabilities = LlmCapabilities(
        supports_caching=True,
        context_window=200_000,
        max_output_tokens=4096,
        model_id="opus",
    )
    huge_system = "S" * 700_000  # ~175K token estimate
    assert should_truncate(
        messages=[_msg("user", "tiny")],
        capabilities=capabilities,
        policy=TruncationPolicy(),
        system=huge_system,
    )


# --- Middle-truncation -------------------------------------------------------


def test_truncate_keeps_head_and_tail_drops_middle() -> None:
    """§7: 'middle-truncation only ... preserve the head + the tail.'"""
    messages = [_msg("user", f"m{i}") for i in range(20)]
    policy = TruncationPolicy(keep_head_messages=2, keep_tail_messages=3)
    result = truncate_messages(messages, policy)
    assert len(result) == 5
    # Head preserved (m0, m1).
    assert result[0].content[0].text == "m0"  # type: ignore[union-attr]
    assert result[1].content[0].text == "m1"  # type: ignore[union-attr]
    # Tail preserved (m17, m18, m19).
    assert result[2].content[0].text == "m17"  # type: ignore[union-attr]
    assert result[-1].content[0].text == "m19"  # type: ignore[union-attr]


def test_truncate_no_op_when_conversation_fits() -> None:
    """If head + tail >= len, nothing is dropped."""
    messages = [_msg("user", f"m{i}") for i in range(5)]
    policy = TruncationPolicy(keep_head_messages=2, keep_tail_messages=8)
    result = truncate_messages(messages, policy)
    assert len(result) == len(messages)


def test_truncate_with_zero_tail_keeps_only_head() -> None:
    """A pathological policy with no tail still works."""
    messages = [_msg("user", f"m{i}") for i in range(10)]
    policy = TruncationPolicy(keep_head_messages=2, keep_tail_messages=0)
    result = truncate_messages(messages, policy)
    assert len(result) == 2
    assert result[0].content[0].text == "m0"  # type: ignore[union-attr]


# --- estimate_total_tokens ---------------------------------------------------


def test_estimate_scales_with_message_count() -> None:
    """The char-count heuristic should be roughly proportional."""
    short = [_msg("user", "x" * 40) for _ in range(1)]
    long = [_msg("user", "x" * 40) for _ in range(10)]
    assert estimate_total_tokens(long) >= estimate_total_tokens(short) * 9
