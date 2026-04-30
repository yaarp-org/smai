"""Conformance suite for :class:`RunPodCompute`.

Subclasses :class:`smai_core.plugins.conformance.ComputeConformance` per
Task 3.F4 — runs the universal contract suite against the RunPod
plugin instance, with an in-process :class:`FakeRunPodBackend`
standing in for real RunPod so the suite runs offline in CI and
deterministically locally.

The fake backend is constructed once per test (``fake_runpod_backend``
fixture) and shared across :meth:`make_compute` and
:meth:`make_fresh_compute` so the §7.5 reconnection contract — a fresh
:class:`Compute` instance polling a serialized handle observes the
same substrate state — actually exercises shared substrate state.
"""

from __future__ import annotations

import httpx
from _runpod_fakes import FakeRunPodBackend
from smai_compute_runpod import RunPodCompute
from smai_core.plugins import Compute
from smai_core.plugins.conformance import ComputeConformance


class TestRunPodComputeConformance(ComputeConformance):
    """Run the 8-method :class:`ComputeConformance` contract suite
    against :class:`RunPodCompute`.

    Per Task 3.F4: ``make_compute()`` returns a real
    :class:`RunPodCompute` instance wired to the fake backend;
    ``make_fresh_compute()`` returns a separately-constructed instance
    pointed at the *same* backend so the cross-instance reconnection
    contract (§7.5) genuinely exercises a fresh in-process state
    against shared substrate state.

    The fake backend is captured on the test instance via
    :meth:`autouse_backend` so both factory hooks see the same
    backend; the fixture cannot be shared via constructor injection
    because :class:`ComputeConformance` instantiates the test class
    via pytest's normal collection path.
    """

    _backend: FakeRunPodBackend

    @staticmethod
    def _build_compute(backend: FakeRunPodBackend) -> RunPodCompute:
        transport = backend.transport()
        client = httpx.AsyncClient(transport=transport, base_url="https://rest.runpod.io")
        return RunPodCompute(
            api_key="rpa_fake_test_key",
            api_base="https://rest.runpod.io/v1",
            client=client,
        )

    def make_compute(self) -> Compute:
        # ``ComputeConformance`` resolves ``compute`` via the ``compute``
        # fixture, which calls ``make_compute()`` exactly once per test
        # — we capture the backend on the instance here so
        # ``make_fresh_compute()`` (called inside
        # ``test_job_handle_reconnection``) re-uses the same substrate.
        backend = FakeRunPodBackend()
        self._backend = backend
        return self._build_compute(backend)

    def make_fresh_compute(self) -> Compute:
        # Re-use the substrate state from ``make_compute()`` so a
        # serialized JobHandle round-tripped to a fresh instance can
        # still be polled — same shape as
        # :meth:`smai_compute_localgpu.LocalGpuCompute.make_fresh_compute`'s
        # "Docker daemon is process-global" trick.
        return self._build_compute(self._backend)
