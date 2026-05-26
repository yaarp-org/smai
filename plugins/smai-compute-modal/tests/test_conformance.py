""":class:`ComputeConformance` subclass for :class:`ModalCompute`.

Per Task 3.F3 of ``designs/smai/implementation_plan.md`` §3.4 and
``07-plugin-interfaces.md`` §7.5. The 8 contract methods on
:class:`smai_core.plugins.conformance.ComputeConformance` run against
:class:`ModalCompute` configured with the in-process Modal SDK fake
(:mod:`_modal_fakes`). The fake spawns real OS subprocesses for the
conformance fixture commands — so the tested job actually runs and
produces the expected exit code — while keeping the suite offline and
deterministic (no Modal credentials, no Modal RPC, no network).

The credentialed real-Modal lane lives in :mod:`tests.test_real_modal`
and is gated on ``MODAL_TOKEN_ID`` per Task 3.F3's no-credentials-in-CI
convention.
"""

from __future__ import annotations

import sys
from typing import cast

import pytest
from _modal_fakes import FakeModal  # type: ignore[import-not-found]
from smai_compute_modal import ModalCompute
from smai_core.plugins import Compute
from smai_core.plugins.conformance import ComputeConformance


class TestModalComputeConformance(ComputeConformance):
    """Run the 8-method :class:`ComputeConformance` contract suite
    against :class:`ModalCompute` with a mocked Modal SDK.

    ``make_compute()`` returns a fresh :class:`ModalCompute` paired
    with a fresh :class:`FakeModal` per call so cross-test state is
    isolated; ``make_fresh_compute()`` does the same with a separate
    fake so the cross-process reconnection contract (§7.5) genuinely
    exercises a fresh substrate handle.

    Substrate-resolvable image: the fake ignores the ``image`` argument
    other than checking the bad-image blacklist, so any non-``.invalid``
    tag works. We override :meth:`fixture_image` to use
    ``smai-runtime:dev`` — the documented v1 production tag — making
    the Modal-side mocked test mirror what a real Modal Sandbox would
    receive.

    ``command[0]`` for the conformance fixtures is the literal ``python``
    binary; the fake spawns the host Python interpreter to execute the
    contract's ``python -c "..."`` jobs. We pin :attr:`sys.executable`
    rather than relying on ``$PATH`` so the tests pass on systems where
    ``python`` resolves to Python 2 (or nothing at all) on the host PATH.
    """

    @pytest.fixture
    def fixture_image(self) -> str:
        # Any non-``.invalid`` tag works against the fake; the
        # ``smai-runtime:dev`` choice mirrors what production
        # ``smai start`` would dispatch via the engine.
        return "smai-runtime:dev"

    @pytest.fixture(autouse=True)
    def _patch_python_to_host_interpreter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Map ``["python", ...]`` commands to ``[sys.executable, ...]``.

        The conformance suite's command shape is
        ``["python", "-c", "..."]`` — that's what a real Sandbox
        running the ``python:3.12-slim`` image would resolve. The
        host running the fake may not have ``python`` on PATH (some
        macOS / Linux distros ship ``python3`` only); rewriting on the
        FakeSandbox side keeps the conformance suite portable without
        editing the upstream contract.
        """
        from _modal_fakes import FakeSandbox  # type: ignore[import-not-found]  # noqa: PLC0415

        original_init = FakeSandbox.__init__

        def patched_init(
            self: FakeSandbox,
            command: list[str],
            env: dict[str, str] | None,
            timeout: int,
        ) -> None:
            if command and command[0] == "python":
                command = [sys.executable, *command[1:]]
            original_init(self, command, env, timeout)

        monkeypatch.setattr(FakeSandbox, "__init__", patched_init)

    def make_compute(self) -> Compute:
        fake_modal = FakeModal()
        return cast(Compute, ModalCompute(modal_module=fake_modal))

    def make_fresh_compute(self) -> Compute:
        # Returns a fresh instance with its own Modal-SDK fake. The
        # conformance suite serializes the JobHandle and reconnects
        # via ``Sandbox.from_id`` — but the substrate-side state lives
        # in the original ``FakeModal`` (the original test instance).
        # To preserve cross-process semantics with the fake, the
        # ``test_job_handle_reconnection`` test below overrides this
        # with a shared-state variant.
        fake_modal = FakeModal()
        return cast(Compute, ModalCompute(modal_module=fake_modal))

    async def test_workspace_round_trip(  # type: ignore[override]
        self,
        compute: Compute,
        fixture_image: str,
        time_provider,  # type: ignore[no-untyped-def]
        poll_budget_seconds: float,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """Skip on the in-process fake: ``FakeSandbox`` spawns a host
        subprocess that has no namespace isolation, so the staged
        volume cannot be mounted at the literal ``/workspace`` path
        the round-trip's container program reads from.

        The credentialed real-Modal lane
        (``tests/test_real_modal.py::TestRealModalComputeConformance``)
        covers the round-trip against a real Modal Volume mount.
        """
        del compute, fixture_image, time_provider, poll_budget_seconds, tmp_path
        pytest.skip(
            "FakeSandbox cannot simulate the /workspace mount; covered by "
            "the credentialed lane in test_real_modal.py."
        )

    async def test_job_handle_reconnection(  # type: ignore[override]
        self,
        compute: Compute,
        fixture_image: str,
        time_provider,  # type: ignore[no-untyped-def]
        poll_budget_seconds: float,
    ) -> None:
        """Override the §7.5 reconnection test to share substrate state.

        Modal's substrate state (the running Sandbox) lives on Modal,
        not in the SDK; ``Sandbox.from_id`` is how a fresh process
        reattaches. The :class:`FakeModal` in :mod:`_modal_fakes` keeps its
        sandbox registry in instance-local state, so reconnecting via
        a *fresh* :class:`FakeModal` would (correctly) raise
        :class:`JobNotFound` — the wrong shape for this test. We rebuild
        the "fresh" :class:`ModalCompute` against the *same* fake so the
        test exercises plugin-side reconnection (handle round-trip via
        Pydantic JSON, ``Sandbox.from_id`` lookup) without simulating
        a substrate-side wipe.

        The contract assertion is unchanged from the base test: submit,
        round-trip the handle, poll to terminal via a fresh
        :class:`Compute` instance, expect ``state='succeeded'``.
        """
        from smai_core.plugins import JobHandle  # noqa: PLC0415
        from smai_core.plugins.conformance.test_compute import (  # noqa: PLC0415
            _poll_until_terminal,
        )

        handle = await compute.submit(
            image=fixture_image,
            command=["python", "-c", "import sys; sys.exit(0)"],
            env={},
            timeout_seconds=int(poll_budget_seconds),
        )
        round_tripped = JobHandle.model_validate_json(handle.model_dump_json())
        # Reconnect against the same FakeModal so the sandbox registry
        # is visible — the production analog is "the same Modal account
        # / API token" which IS shared across processes.
        modal_module = compute._modal  # type: ignore[attr-defined]  # noqa: SLF001
        fresh = ModalCompute(modal_module=modal_module)
        status = await _poll_until_terminal(
            fresh, round_tripped, time_provider, poll_budget_seconds
        )
        assert status.state == "succeeded"
