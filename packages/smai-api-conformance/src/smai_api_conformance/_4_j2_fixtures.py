"""Shared fixture helpers for the API-conformance test suite.

Per ``designs/smai/11-api.md`` §10.2 the suite is shape-only: it does
not exercise lifecycle correctness. The helpers here exist to keep
per-resource test files terse — sample request bodies, the SSE
event-line parser, the assert-error-envelope helper, and a small set
of well-known fixture identifiers.

Per the per-task fixture filename hygiene convention
(``CLAUDE.md`` "Per-task fixture filename hygiene"), helper modules in
this package use the ``_4_j2_<purpose>.py`` pattern so they don't
collide with sibling-conformance fixture files under
``--import-mode=importlib``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from httpx import Response
from smai_api_spec import ErrorCode, ErrorEnvelope

# === Sample request bodies ===================================================
#
# Minimal-but-spec-conformant payloads. Real implementations may
# override the corresponding fixture in their subclass to provide a
# payload that exercises the implementation's compilation path; the
# defaults here are sufficient for shape-only conformance against any
# implementation that accepts the SubmitProposalRequest / SubmitPaperRequest
# / SubmitExperimentRequest schemas.


def sample_proposal_request_body() -> dict[str, Any]:
    """A minimal valid ``SubmitProposalRequest`` body.

    Per planner-refactor Step 2 / ``upstream_requirements §2`` the
    proposal-submit body is now a typed :class:`TechniqueDescription`
    dict (the freeform ``technique_description_text`` path was dropped
    per D10). The shape below is the minimum valid ``proposal`` variant:
    one paraphrased prose set plus an empty ``algorithm.source_excerpts``
    list (paper-only fields are absent), plus the
    ``source_proposal_id`` provenance pointer and the loss /
    training-recipe ``ConfidenceFlag`` annotations the
    ``_none_requires_flag`` validator demands.
    """
    return {
        "submission_kind": "novel_technique",
        "technique_description": {
            "name": "conformance_suite_probe",
            "summary": (
                "Placeholder novel-technique submission used by "
                "smai-api-conformance to exercise the POST /api/v1/proposals "
                "shape contract; not a real technique."
            ),
            "motivation": (
                "Conformance suites need a minimum-viable body that round-trips "
                "through Pydantic without depending on the implementation's "
                "methodology DSL or compiler; this proposal variant is the "
                "smallest valid shape."
            ),
            "problem_setting": (
                "API shape conformance; no real ML setting. Implementations "
                "override this fixture in their subclass if compilation is "
                "needed."
            ),
            "algorithm": {
                "summary": (
                    "Placeholder algorithm summary that exists only to satisfy "
                    "the TechniqueDescription schema's minimum length rule on "
                    "AlgorithmSpec.summary; not a real algorithm description."
                ),
            },
            "context_kind": "proposal",
            "source_proposal_id": "conformance-suite-probe",
            "confidence_flags": [
                {
                    "field_path": "/limitations",
                    "severity": "unknown",
                    "note": "Conformance probe; field intentionally unset.",
                },
                {
                    "field_path": "/loss_function",
                    "severity": "unknown",
                    "note": "Conformance probe; field intentionally unset.",
                },
                {
                    "field_path": "/training_recipe",
                    "severity": "unknown",
                    "note": "Conformance probe; field intentionally unset.",
                },
                {
                    "field_path": "/evaluation_protocol",
                    "severity": "unknown",
                    "note": "Conformance probe; field intentionally unset.",
                },
            ],
        },
    }


def sample_paper_request_body(arxiv_id: str = "2501.00001") -> dict[str, Any]:
    """A minimal valid ``SubmitPaperRequest`` body."""
    return {"arxiv_id": arxiv_id}


def sample_experiment_definition_text() -> str:
    """A placeholder experiment-definition body.

    The conformance suite cannot ship a one-size-fits-all DSL document
    (the methodology compiler is implementation-coupled — different
    builds may have different factor-type plugin sets). Real
    implementations should override the
    ``sample_experiment_definition_text`` fixture in their subclass to
    supply an actually-compilable YAML document; the default here is
    accepted by the self-test mock but will be rejected by a real
    methodology compiler.
    """
    return "# smai-api-conformance placeholder — override in your subclass\n"


# === Error-envelope assertion ===============================================


def assert_error_envelope(response: Response, expected_code: ErrorCode | None = None) -> APIError:
    """Assert ``response`` carries a valid :class:`ErrorEnvelope` and
    return the parsed :class:`smai_api_spec.APIError` for further
    inspection.

    Per ``11-api.md`` §6.1: every non-2xx body MUST parse cleanly into
    the envelope, and the ``error.code`` value MUST be a member of the
    documented :data:`ErrorCode` literal. Pydantic ``extra="forbid"``
    catches unknown keys at parse time.

    When ``expected_code`` is supplied the test additionally asserts
    the parsed code matches.
    """
    payload: object = response.json()
    envelope = ErrorEnvelope.model_validate(payload)
    if expected_code is not None:
        assert envelope.error.code == expected_code, (
            f"expected error.code={expected_code!r}, got {envelope.error.code!r}; "
            f"message={envelope.error.message!r}"
        )
    return envelope.error


# === SSE event parsing ======================================================


@dataclass(frozen=True)
class SSEEvent:
    """One event parsed off the SSE wire.

    Per ``11-api.md`` §8.1: each event is a sequence of ``id:`` /
    ``event:`` / ``data:`` lines terminated by a blank line. Comment
    lines (``:keepalive``) are skipped here — they're a transport
    concern (defeating idle-timeout proxies), not a contract event.
    """

    id: str | None
    event: str | None
    data: str

    def parse_data(self) -> dict[str, Any]:
        """Parse ``data`` as JSON. Raises ``ValueError`` on malformed input."""
        parsed: object = json.loads(self.data)
        if not isinstance(parsed, dict):
            raise ValueError(f"SSE data is not a JSON object: {self.data!r}")
        # Pyright: cast through the runtime check above.
        return parsed  # type: ignore[return-value]


async def read_one_sse_event(lines: AsyncIterator[str]) -> SSEEvent | None:
    """Read one SSE event from ``lines``.

    Returns ``None`` when the iterator ends before a complete event is
    received. Multi-line ``data:`` payloads are joined with ``"\\n"``
    per the SSE spec; comment lines (``:`` prefix) are skipped.
    """
    event_id: str | None = None
    event_kind: str | None = None
    data_lines: list[str] = []
    received_any = False
    async for raw in lines:
        # httpx returns lines without their trailing newline.
        line = raw.rstrip("\r")
        if line == "":
            # Event terminator. Emit only when we accumulated something.
            if received_any:
                return SSEEvent(id=event_id, event=event_kind, data="\n".join(data_lines))
            # Spurious blank line — keep looking.
            continue
        if line.startswith(":"):
            # SSE comment / keepalive — ignore.
            continue
        received_any = True
        if line.startswith("id:"):
            event_id = line[len("id:") :].lstrip()
        elif line.startswith("event:"):
            event_kind = line[len("event:") :].lstrip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
        # Other field names are valid per the SSE spec but unused here.
    # Iterator exhausted without a terminator.
    if received_any:
        return SSEEvent(id=event_id, event=event_kind, data="\n".join(data_lines))
    return None


# === Re-exports =============================================================
#
# Pulled into ``__init__`` so the ``APIError`` type alias used in
# annotations resolves without an extra import in callers.
from smai_api_spec import APIError  # noqa: E402 — re-export at module bottom

__all__ = [
    "APIError",
    "SSEEvent",
    "assert_error_envelope",
    "read_one_sse_event",
    "sample_experiment_definition_text",
    "sample_paper_request_body",
    "sample_proposal_request_body",
]
