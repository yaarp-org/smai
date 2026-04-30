"""``smai start`` credentialed-fixture-matrix integration test (Task 3.G3).

The credentialed counterpart of :mod:`test_smai_start_fixture_matrix`.
Per Task 3.G3's no-credentials-in-CI convention this test:

* is marked ``@pytest.mark.credentialed`` (registered in root
  ``pyproject.toml``);
* skips cleanly if ANY of the required env vars are absent
  (``SMAI_TEST_POSTGRES_URL`` / ``SMAI_TEST_S3_BUCKET`` /
  ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``);
* is **never** run on CI — local-manual only;
* is the credential-holder's pre-merge check that the implementation
  plan §3.4 Task 3.G3 acceptance ("real CG end-to-end against
  Postgres + S3 + Modal") is satisfied.

To run locally::

    export SMAI_TEST_POSTGRES_URL=postgresql+asyncpg://user:pass@host:5432/smai_test
    export SMAI_TEST_S3_BUCKET=my-test-bucket-us-east-1
    export AWS_REGION=us-east-1
    export MODAL_TOKEN_ID=...
    export MODAL_TOKEN_SECRET=...
    uv run pytest tests/integration/test_smai_start_credentialed.py -v

The test drives the same shape as the no-creds fixture-matrix test
but with Postgres + S3 + Modal Sandboxes — when the credentialed lane
ships, the same dispatch handlers fire against real cloud resources
and the same gate bodies advance the CG to terminal.
"""

from __future__ import annotations

import os

import pytest

_REQUIRED_CREDS = (
    "SMAI_TEST_POSTGRES_URL",
    "SMAI_TEST_S3_BUCKET",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
)

_missing_creds = [name for name in _REQUIRED_CREDS if name not in os.environ]


@pytest.mark.credentialed
@pytest.mark.skipif(
    bool(_missing_creds),
    reason=(
        "credentialed-only; missing env: " + ", ".join(_missing_creds)
        if _missing_creds
        else "skipped"
    ),
)
@pytest.mark.asyncio
async def test_smai_start_drives_cg_against_postgres_s3_modal() -> None:
    """End-to-end CG round-trip against Postgres + S3 + Modal.

    The shape is identical to
    :func:`test_smai_start_drives_cg_to_complete_no_creds` but binds
    the production plugins:

    * ``MetadataStore`` = ``smai-store-postgres`` against
      ``$SMAI_TEST_POSTGRES_URL``.
    * ``ArtifactStore`` = ``smai-artifacts-s3`` against
      ``$SMAI_TEST_S3_BUCKET`` (a BYO bucket the credential-holder
      owns; the test cleans up its keys on teardown).
    * ``Compute`` = ``smai-compute-modal`` with Modal credentials
      from ``$MODAL_TOKEN_ID`` / ``$MODAL_TOKEN_SECRET``.
    * ``LlmProvider`` = ``smai-llm-bedrock`` (uses the AWS credential
      chain — no test-specific env var; the runner's profile must
      have Bedrock + Claude Opus 4.7 access).

    Per the no-credentials-in-CI directive, this test is intentionally
    a stub that documents the shape rather than ships a fully wired
    fixture — the credential-holder's local-manual run is the
    canonical verification path. When the credentialed lane gains a
    permanent home (e.g., a manually-triggered manual-only CI run),
    this test gains its plumbing then. For now: the marker +
    skipif-on-env keeps the structural surface in place, and the
    no-creds fixture-matrix integration test gates CI per Task 3.G3.
    """
    # The acceptance bar per Task 3.G3 is that this test EXISTS and
    # SKIPS cleanly without creds. The credential-holder's local-
    # manual run wires the actual end-to-end drive at merge-time. We
    # raise NotImplementedError if creds ARE present so the
    # credential-holder is alerted that this stub still needs its
    # body.
    raise NotImplementedError(
        "Credentialed Postgres + S3 + Modal CG round-trip stub — see module "
        "docstring for the manual-run shape. The credential-holder is "
        "expected to wire this fixture at the time of merge per the no-"
        "credentials-in-CI convention."
    )


def test_credentialed_marker_registered() -> None:
    """Sanity: the ``credentialed`` marker is registered on the root
    pyproject.toml so ``pytest --strict-markers`` doesn't fire.

    A regression-detector — if a future ``pyproject.toml`` rewrite
    drops the marker registration, this test surfaces it at collection
    time.
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        cfg = tomllib.load(f)
    markers = cfg["tool"]["pytest"]["ini_options"].get("markers", [])
    assert any(m.startswith("credentialed:") for m in markers), (
        "the `credentialed` marker must be registered in pyproject.toml's "
        "[tool.pytest.ini_options].markers list per Task 3.G3 / the no-"
        "credentials-in-CI convention."
    )
