"""Parameterizable conformance test base classes for the four plugin Protocols.

Per ``designs/smai/07-plugin-interfaces.md`` §8 (cross-cutting discipline) and
§4.7 / §5.8 / §6.4 / §7.5 (per-Protocol contract enumerations); per Task 1.9
of ``designs/smai/implementation_plan.md`` §3.2.

A plugin author validates conformance by writing one small subclass per
interface and overriding the ``make_<interface>()`` factory method::

    # In smai-llm-bedrock/tests/test_conformance.py
    from smai_core.plugins.conformance import LlmProviderConformance
    from smai_llm_bedrock import BedrockProvider

    class TestBedrockConformance(LlmProviderConformance):
        def make_provider(self) -> LlmProvider:
            return BedrockProvider(region="us-east-1", model_id="...")

The subclass inherits every contract assertion the base ships. When the
subclass is collected by pytest, the inherited tests run against the
plugin's instance. When the base class itself is collected (no override),
it skip-collects with a clear message — the contract surface is visible
in the test report, but no spurious failures fire.

**Why this is shipped as an optional dependency.**

Conformance test bases are :mod:`pytest` classes. They use ``@pytest.fixture``,
``@pytest.mark.parametrize``, and ``pytest.skip`` — pytest is required at
module-import time of this package. But pytest is NOT in ``smai-core``'s
runtime allowlist (only Pydantic + ``jsonschema`` + stdlib per DEC-029).

Resolution: ``smai-core`` declares pytest as an **optional dependency
group** in its ``pyproject.toml``::

    [project.optional-dependencies]
    conformance = ["pytest>=8.0"]

Tier B integrators who want to run conformance against their plugin
install with::

    pip install smai-core[conformance]

The default ``pip install smai-core`` install stays minimal.
``tools/check_deps.py`` rule 1 walks ``[project] dependencies`` only and
ignores ``[project.optional-dependencies]``, so the dep allowlist isn't
tripped.

**Wall-clock seam (deferred to Task 2.C1).**

Per ``07-plugin-interfaces.md`` §5.6.10 / §12 OQ18 / implementation plan
§6 Q4: a few conformance contracts are wall-clock-timing-sensitive (lease
expiry, retry backoff). The bases ship with a :func:`time_provider`
pytest fixture that defaults to :func:`time.monotonic`. Subclasses can
override it once Task 2.C1 ships its ``EngineConfig``-driven fake-clock
seam. The Task 1.9 bases just leave the override hook in place — the
seam shape itself lives in 2.C1.
"""

from __future__ import annotations

from smai_core.plugins.conformance.test_artifact_store import ArtifactStoreConformance
from smai_core.plugins.conformance.test_compute import ComputeConformance
from smai_core.plugins.conformance.test_llm_provider import LlmProviderConformance
from smai_core.plugins.conformance.test_metadata_store import MetadataStoreConformance

__all__ = [
    "ArtifactStoreConformance",
    "ComputeConformance",
    "LlmProviderConformance",
    "MetadataStoreConformance",
]
