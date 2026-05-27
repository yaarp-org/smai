"""End-to-end test for the canonical SMAI v2 API user journey.

Drives the full canonical journey via the HTTP API + the in-process
:class:`Runtime` worker:

    POST /api/v1/proposals (novel-technique submission)
    → wait for SSE state_change: proposal_submitted → designing
    → wait for SSE state_change: designing → designed
    POST /api/v1/proposals/{id}/approve
    → wait for SSE state_change events on the resulting CGs through to complete
    GET /api/v1/comparison-groups/{cg_id}/artifacts/evaluation_result.json
    → assert artifact is fetchable + has the right shape

Per Task 4.N3 (M5 gate). Runs the API stack in-process via
``httpx.AsyncClient(transport=ASGITransport(app=app))`` against a
:meth:`Runtime.start_in_band` instance with ``run_worker=True``. Uses
fake compute / LLM stubs (per
:mod:`tests.integration._4_n3_helpers`) so the test is deterministic
and completes within the brief's ~30s wall-clock budget.

Live-update channel: instead of reading from the
``/api/v1/events`` SSE endpoint over HTTP (httpx's ASGI transport
does not stream SSE chunks cleanly across versions), the test
subscribes directly to :attr:`Runtime.event_broker` — the same
:class:`smai_events.EventBroker` the SSE handler in
:mod:`smai_api.routers.events` drains. This is the brief's
"fall back to direct broker subscription if needed" path, and
exercises the same fire-on-transition wiring (per Task 4.K2) that
the SSE endpoint reads — only the wire transport differs.

Out of scope: the credentialed Case-B variant (Postgres + S3 +
optional Modal compute) is below; it carries the
``@pytest.mark.credentialed`` + ``skipif`` envelope so the lane runs
locally pre-merge for the credential-holder and skips cleanly on CI
(no-credentials-in-CI rule per ``CLAUDE.md`` "Project conventions").
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from _4_n3_helpers import (  # type: ignore[import-not-found]
    FakeAlwaysSucceededCompute,
    StubLlmProvider,
    default_n3_shape,
    make_canned_contextual_promising,
    make_canned_review_pass,
    make_n3_runtime_config,
    make_planner_responses_novel_technique,
    pre_stage_cg_artifacts,
    wait_for_state_change,
)
from httpx import ASGITransport, AsyncClient
from smai_agents import AgentOutcome, AgentSession
from smai_api import make_api_app
from smai_artifacts_localfs import LocalFsStore
from smai_cli.runtime import Runtime
from smai_core import EvaluationResult
from smai_orchestrator import (
    DEFAULT_TASK_ROLES,
    PluginOverrides,
)


def _build_per_role_stubs(
    *,
    planner_responses: Sequence[object],
) -> dict[str, StubLlmProvider]:
    """One :class:`StubLlmProvider` per task role.

    Roles invoked in the canonical novel-technique journey:

    * ``planner`` — the proposal pipeline's design step. Consumes the
      full canned response sequence from
      :func:`make_planner_responses_novel_technique`.
    * ``code_reviewer`` — the CG-execution spec's
      ``implemented → running`` review-pass gate. One canned pass
      response.
    * ``contextual_evaluator`` — the CG-execution spec's
      ``evaluating → complete`` evaluation gate. One canned promising
      verdict.

    Every other role gets an empty-queue stub that ``AssertionError``s
    if anything calls it — a tripwire matching the smoke test's
    pattern (``test_smoke_e2e._build_per_role_stubs``) so an
    unintended agent dispatch surfaces immediately.
    """
    role_to_stub: dict[str, StubLlmProvider] = {}
    for role in DEFAULT_TASK_ROLES:
        if role == "planner":
            role_to_stub[role] = StubLlmProvider(
                list(planner_responses),  # pyright: ignore[reportArgumentType]
                name=f"stub-{role}",
            )
        elif role == "code_reviewer":
            role_to_stub[role] = StubLlmProvider(
                [make_canned_review_pass()],
                name=f"stub-{role}",
            )
        elif role == "contextual_evaluator":
            role_to_stub[role] = StubLlmProvider(
                [make_canned_contextual_promising()],
                name=f"stub-{role}",
            )
        else:
            role_to_stub[role] = StubLlmProvider([], name=f"stub-{role}")
    return role_to_stub


async def _noop_agent_runner(session: AgentSession) -> AgentOutcome:
    """No-op session runner for the round-14 in-process harness-builder
    / technique-implementer dispatches. The agents' artifacts are
    pre-staged by :func:`pre_stage_cg_artifacts` before the worker
    boots, so the runner just reports a successful outcome."""
    return AgentOutcome(
        kind="finished",
        turn_count=0,
        usage_total=session.usage_total,
        finish_success=True,
    )


@pytest.mark.asyncio
async def test_full_user_journey(tmp_path: Path) -> None:
    """The canonical user journey end-to-end via the HTTP API.

    Submits a novel-technique proposal, waits for the worker-driven
    ``proposal_submitted → designing → designed`` transitions via
    :class:`EventBroker` subscription, approves via the API, waits
    for the CG (created by the proposal-pipeline registration handler)
    to traverse ``draft → ... → complete``, then fetches the
    ``evaluation_result.json`` artifact via the
    ``/api/v1/comparison-groups/{cg_id}/artifacts/{path}`` endpoint
    and asserts it round-trips through :class:`EvaluationResult`.

    Pre-staging strategy (mirrors :mod:`tests.integration.test_smoke_e2e`):
    the harness manifest, validation, technique source, and per-(entry,
    seed=0) metrics that a real harness-builder / technique-implementer
    container would write to ArtifactStore are written upfront against
    the deterministic CG namespace before the proposal is submitted.
    The CG record does not yet exist — the writes are orphaned in the
    store until the proposal-pipeline registration handler creates the
    CG with the matching ``cg_id``. From that point onward the
    pre-staged artifacts are visible to the worker's gate bodies.
    """
    shape = default_n3_shape()
    artifact_store = LocalFsStore(tmp_path / "artifacts")

    # Pre-stage the artifact side effects normally written by the
    # harness-builder / technique-implementer / runtime containers.
    # FakeCompute reports succeeded immediately; the gate bodies'
    # artifact-presence predicates need these in place by the cycle
    # AFTER the CG enters ``implementing``.
    await pre_stage_cg_artifacts(artifact_store=artifact_store, shape=shape)

    role_stubs = _build_per_role_stubs(
        planner_responses=make_planner_responses_novel_technique(shape=shape),
    )
    fake_compute = FakeAlwaysSucceededCompute()
    overrides = PluginOverrides(
        # Cast through ``LlmProvider`` so the type checker sees the
        # Protocol-conformant shape; StubLlmProvider duck-types against
        # ``LlmProvider`` per the smoke test's pattern.
        llm_providers={role: provider for role, provider in role_stubs.items()},
        artifact_store=artifact_store,
        compute=fake_compute,
    )
    sqlite_path = tmp_path / "n3_state.db"
    config = make_n3_runtime_config(
        poll_interval_seconds=1,
        sqlite_path=str(sqlite_path),
    )

    async with Runtime.start_in_band(
        config,
        workspace_root=tmp_path / "workspaces",
        plugin_overrides=overrides,
        run_worker=True,
        # Sub-PR E + Step-7 cutover: both harness-builder and
        # technique-implementer dispatches are sandboxed now;
        # FakeAlwaysSucceededCompute's status-always-succeeded shape
        # carries them both.
        # Round-9 added a test-only knob to pin deterministic CG ids.
        # Production generates ULID-shaped ids by default; the N3
        # fixture pre-stages artifacts at ``comparison-groups/<shape.cg_id>``
        # paths before submission, so the registration handler MUST land
        # the CG at exactly ``shape.cg_id`` for the harness/runs gates to
        # find the pre-staged outputs.
        proposal_cg_id_for=lambda proposal_id, draft_cg_id: f"{proposal_id}--{draft_cg_id}",
    ) as runtime:
        app = make_api_app(runtime)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            # Subscribe to the broker FIRST — events fired between
            # ``Runtime.start_in_band`` entry and our subscription are
            # missed, which is fine for the journey we drive: no entity
            # exists yet so the only events would be worker-heartbeats
            # we ignore.
            async with runtime.event_broker.subscribe() as events:
                # === Step 1: submit proposal =================================
                submit_response = await client.post(
                    "/api/v1/proposals",
                    json={
                        "submission_kind": "novel_technique",
                        "technique_description_text": "n3 e2e test technique",
                        "proposal_id": shape.proposal_id,
                    },
                )
                assert submit_response.status_code == 202, submit_response.text
                submit_body = submit_response.json()
                assert submit_body["id"] == shape.proposal_id
                assert submit_body["state"] == "proposal_submitted"

                # === Step 2: wait for designing → designed ===================
                # The worker's proposal-spec advances proposal_submitted →
                # designing (auto-advance edge, on-entry dispatch fires the
                # planner) and designing → designed (after the planner
                # finalizes the buffer). With poll=1s and inline planner +
                # stubbed LLM, both transitions land within a few cycles.
                await wait_for_state_change(
                    events,
                    kind="proposal",
                    id=shape.proposal_id,
                    target_state="designing",
                    timeout=20.0,
                )
                await wait_for_state_change(
                    events,
                    kind="proposal",
                    id=shape.proposal_id,
                    target_state="designed",
                    timeout=20.0,
                )

                # === Step 3: approve via API =================================
                approve_response = await client.post(
                    f"/api/v1/proposals/{shape.proposal_id}/approve",
                )
                assert approve_response.status_code == 200, approve_response.text
                approve_body = approve_response.json()
                assert approve_body["id"] == shape.proposal_id
                # Approve writes user_decision; the proposal advances
                # designed → registered on the next worker cycle, and
                # the registration handler creates the CG records in
                # ``draft`` from the planner buffer. The CG's id is
                # pinned by the ``proposal_cg_id_for`` resolver wired
                # into ``Runtime.start_in_band`` above (round-9): the
                # test resolver returns ``f"{proposal_id}--{draft_cg_id}"``
                # so the pre-staged artifact paths line up. Production's
                # default resolver instead generates a fresh ULID-shaped
                # id per CG (so a long symbolic name no longer blows past
                # the 64-char id-format cap).

                # === Step 4: wait for CG complete ============================
                # Skip the explicit proposal=registered wait — the
                # state_change event fires AFTER the CAS UPDATE but
                # BEFORE the on-entry-dispatch registration handler
                # runs (per ``engine/_metadata_ops.py``'s ordering),
                # so observing ``registered`` does not guarantee CG
                # creation has landed. Instead wait directly for the
                # CG to traverse to ``complete``. The wait_for helper
                # silently skips intermediate transitions
                # (proposal=registered, comparison_group=implementing,
                # etc.) and returns on the matching triple.
                #
                # 5-state CG-execution traversal (draft → implementing →
                # implemented → running → evaluating → complete) at 1s
                # poll typically completes in 5–7 cycles. 45s is a
                # generous timeout that covers slow-CI variance + the
                # initial proposal-pipeline cycles before the CG even
                # exists.
                await wait_for_state_change(
                    events,
                    kind="comparison_group",
                    id=shape.cg_id,
                    target_state="complete",
                    timeout=45.0,
                )

                # Cross-check the CG materialized at the deterministic id.
                proposal_cgs = await runtime.proposals.list_cgs(shape.proposal_id)
                assert len(proposal_cgs) == 1
                assert proposal_cgs[0].id == shape.cg_id

            # === Step 6: fetch evaluation_result.json ========================
            # The artifact endpoint streams bytes for LocalFsStore (the
            # store's url_for returns a file:// URL; the router's
            # HTTP-scheme check falls through to the streaming branch).
            artifact_response = await client.get(
                f"/api/v1/comparison-groups/{shape.cg_id}/artifacts/evaluation_result.json",
            )
            assert artifact_response.status_code == 200, artifact_response.text
            evaluation = EvaluationResult.model_validate_json(artifact_response.content)
            assert evaluation is not None

            # The convenience endpoint also serves the same evaluation
            # data — projected through the API spec's
            # :class:`EvaluationResultResponse` shape rather than
            # streaming the raw JSON. We assert on the shape's key
            # fields (``artifact_key``, ``cg_id``) rather than a
            # specific top-level key.
            evaluation_response = await client.get(
                f"/api/v1/comparison-groups/{shape.cg_id}/evaluation",
            )
            assert evaluation_response.status_code == 200, evaluation_response.text
            evaluation_payload = evaluation_response.json()
            assert evaluation_payload["cg_id"] == shape.cg_id
            assert (
                evaluation_payload["artifact_key"]
                == f"comparison-groups/{shape.cg_id}/evaluation_result.json"
            )

            # The CG record reports terminal complete via the API.
            cg_status = await client.get(
                f"/api/v1/comparison-groups/{shape.cg_id}/status",
            )
            assert cg_status.status_code == 200, cg_status.text
            assert cg_status.json()["state"] == "complete"
            assert cg_status.json()["is_terminal"] is True


@pytest.mark.credentialed
@pytest.mark.skipif(
    not (os.getenv("SMAI_TEST_POSTGRES_URL") and os.getenv("AWS_TEST_BUCKET")),
    reason="requires SMAI_TEST_POSTGRES_URL + AWS_TEST_BUCKET",
)
@pytest.mark.asyncio
async def test_full_user_journey_remote_data(tmp_path: Path) -> None:
    """Same canonical journey against postgres + s3 plugins.

    Skipped on CI per ``CLAUDE.md`` "no credentials in CI" — the
    composite ``skipif`` checks BOTH env vars so the lane skips when
    either is missing. Documents the Case-B
    (`12-ui-process.md` §5.2 — remote-data, in-process API)
    deployment shape.

    Implementation deferred — the credentialed wiring requires a
    ``smai-store-postgres`` plugin instance against the supplied
    Postgres URL + a ``smai-artifacts-s3`` plugin instance against
    the supplied bucket, plus a stable ``run_worker`` strategy
    against multi-worker leasing semantics (Task 3.G1). Tracked as
    an open question for the supervisor — see the N3 status note's
    OQ list. The body :func:`pytest.skip`s so the lane is a
    structural placeholder that does NOT silently pass against real
    creds; the supervisor adjudicates implementation depth before
    activating.
    """
    del tmp_path
    pytest.skip(
        "Case-B credentialed wiring is deferred; see N3 OQ list for "
        "supervisor adjudication. Body intentionally skips even when "
        "env vars are set so the lane does not silently pass."
    )
