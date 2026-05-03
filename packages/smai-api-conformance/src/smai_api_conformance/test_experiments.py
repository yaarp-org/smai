"""Conformance tests for the ``/api/v1/experiments`` endpoints.

Per ``designs/smai/11-api.md`` §4.3.

The conformance suite cannot ship a one-size-fits-all DSL document
(the methodology compiler is implementation-coupled). The default
``sample_experiment_definition_text`` from
:mod:`smai_api_conformance._4_j2_fixtures` is a placeholder accepted
by the self-test mock; real implementations should override that
fixture in their subclass to supply an actually-compilable YAML
document. When a real implementation rejects the default body, the
``compile`` test asserts the rejection arrives as a well-formed
``VALIDATION_ERROR`` (so the rejection itself is contract-conformant
even when no compile happens).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from smai_api_spec import CompileExperimentResponse, SubmitExperimentResponse
from smai_api_spec.paths import EXPERIMENTS, EXPERIMENTS_COMPILE

from smai_api_conformance._4_j2_fixtures import (
    assert_error_envelope,
    sample_experiment_definition_text,
)


class ExperimentsConformanceTests:
    """Mixin: ``/api/v1/experiments`` endpoint conformance tests."""

    # The conformance suite supplies a default placeholder definition
    # text. Real implementations override the fixture below to supply an
    # actually-compilable YAML document; the placeholder is accepted by
    # the self-test mock but typically rejected by a real methodology
    # compiler — see test_compile_experiment_smoke for the rejection
    # branch.

    @pytest.fixture
    def experiment_definition_text(self) -> str:
        """Override to supply a methodology-compilable YAML body."""
        return sample_experiment_definition_text()

    # ---- POST /api/v1/experiments/compile ---------------------------------

    async def test_compile_experiment_smoke(
        self, client: AsyncClient, experiment_definition_text: str
    ) -> None:
        """``POST /experiments/compile`` returns either:

        * 200 with a valid ``CompileExperimentResponse`` (when the
          implementation accepts the supplied definition), OR
        * 400 with a ``VALIDATION_ERROR`` envelope (when the
          implementation rejects it — the placeholder body in
          :mod:`_4_j2_fixtures` is intentionally rejected by real
          methodology compilers).

        Both branches are spec-conformant; the test asserts only the
        shape of whichever branch fires.
        """
        response = await client.post(
            EXPERIMENTS_COMPILE,
            json={"definition_text": experiment_definition_text},
        )
        if response.status_code == 200:
            CompileExperimentResponse.model_validate(response.json())
        elif response.status_code == 400:
            assert_error_envelope(response, expected_code="VALIDATION_ERROR")
        else:
            pytest.fail(
                f"unexpected status {response.status_code} from "
                f"POST {EXPERIMENTS_COMPILE}; body: {response.text}"
            )

    async def test_compile_experiment_validation_error(self, client: AsyncClient) -> None:
        """Empty body → 400 + VALIDATION_ERROR (definition_text required)."""
        response = await client.post(EXPERIMENTS_COMPILE, json={})
        assert response.status_code == 400, response.text
        assert_error_envelope(response, expected_code="VALIDATION_ERROR")

    # ---- POST /api/v1/experiments -----------------------------------------

    async def test_submit_experiment_smoke(
        self, client: AsyncClient, experiment_definition_text: str
    ) -> None:
        """``POST /experiments`` returns 202 + SubmitExperimentResponse OR
        400 + VALIDATION_ERROR — same dual-branch logic as compile."""
        response = await client.post(
            EXPERIMENTS,
            json={"definition_text": experiment_definition_text},
        )
        if response.status_code == 202:
            SubmitExperimentResponse.model_validate(response.json())
        elif response.status_code == 400:
            assert_error_envelope(response, expected_code="VALIDATION_ERROR")
        else:
            pytest.fail(
                f"unexpected status {response.status_code} from "
                f"POST {EXPERIMENTS}; body: {response.text}"
            )

    async def test_submit_experiment_validation_error(self, client: AsyncClient) -> None:
        """Empty body → 400 + VALIDATION_ERROR."""
        response = await client.post(EXPERIMENTS, json={})
        assert response.status_code == 400, response.text
        assert_error_envelope(response, expected_code="VALIDATION_ERROR")


__all__ = ["ExperimentsConformanceTests"]
