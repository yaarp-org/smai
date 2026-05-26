""":class:`ComputeConformance` — parameterizable test base for
:class:`Compute` plugins.

Per ``designs/smai/07-plugin-interfaces.md`` §7.5 (the per-Protocol
conformance enumeration) and §8 (cross-cutting discipline).

Subclass and override :meth:`ComputeConformance.make_compute` to run
the suite against your plugin::

    from smai_core.plugins.conformance import ComputeConformance
    from smai_compute_localgpu import LocalGpuCompute

    class TestLocalGpuConformance(ComputeConformance):
        def make_compute(self) -> Compute:
            return LocalGpuCompute(...)

The wall-clock-sensitive tests (cancel propagation, timeout firing) use
the :func:`time_provider` fixture (defaults to :func:`time.monotonic`)
— Task 2.C1 will replace it with an ``EngineConfig``-driven fake clock
seam.
"""

from __future__ import annotations

import pathlib
import time

import pytest

from smai_core.plugins import (
    Compute,
    ComputeCapabilities,
    JobHandle,
    JobImageInvalid,
    JobStatus,
    WorkspaceHandle,
    WorkspaceNotFound,
)
from smai_core.plugins.conformance._common import (
    TimeProvider,
    skip_no_factory_override,
)

# Module-level fixture tree exercised by ``test_workspace_round_trip``.
# Three regular files across two directory levels: enough to exercise
# tar/zip boundary handling on substrates that upload-and-download
# (Modal Volumes) while small enough that round-trip latency stays
# under the conformance suite's poll_budget_seconds.
_WORKSPACE_FIXTURE_FILES: tuple[tuple[str, bytes], ...] = (
    ("README.md", b"# smai conformance round-trip\nfixture tree.\n"),
    ("data/config.json", b'{"smai_conformance": "round-trip"}\n'),
    ("data/sub/payload.bin", bytes(range(256))),
)


class ComputeConformance:
    """Universal contract suite for :class:`Compute` implementations.

    The contract methods correspond 1:1 to the rules in
    ``07-plugin-interfaces.md`` §7.5:

    * :meth:`test_submit_and_poll_succeeds` — submit, poll until
      ``state='succeeded'``, ``exit_code == 0``
    * :meth:`test_submit_and_fail_nonzero_exit` — submit a job that
      exits non-zero, ``state='failed'``, ``exit_code != 0``
    * :meth:`test_timeout_fires` — submit a job that sleeps longer than
      its timeout, ``state='timeout'``
    * :meth:`test_cancel_terminates` — submit a long job, cancel
      mid-run, eventual ``state='cancelled'``
    * :meth:`test_job_handle_reconnection` — submit, serialize handle,
      deserialize from a fresh :class:`Compute` instance, poll status
    * :meth:`test_logs_returns_stdout` — submit a job that prints a
      known string, retrieve logs, assert the string is present
    * :meth:`test_invalid_image_raises` — submit with a non-existent
      image, raises :class:`JobImageInvalid` with the bad image in the
      payload
    """

    # ---- factory hooks -----------------------------------------------------

    def make_compute(self) -> Compute:
        """Return a configured :class:`Compute` instance.

        Override in your subclass. The default skip-collects.
        """
        skip_no_factory_override(type(self).__name__, "make_compute")
        raise AssertionError("unreachable")  # pragma: no cover

    def make_fresh_compute(self) -> Compute:
        """Return a *fresh* :class:`Compute` instance with the same
        configuration as :meth:`make_compute` but no shared in-process
        state.

        Used by :meth:`test_job_handle_reconnection` to verify that a
        :class:`JobHandle` round-trips across processes (the test
        constructs a second instance and re-polls).

        Default: returns :meth:`make_compute`'s instance — many plugins
        are stateless from the substrate's perspective and the same
        instance is fine. Plugins with in-process state (e.g., a
        background poller cache) override this to construct a genuinely
        fresh instance.
        """
        return self.make_compute()

    # ---- fixtures ----------------------------------------------------------

    @pytest.fixture
    def compute(self) -> Compute:
        return self.make_compute()

    @pytest.fixture
    def time_provider(self) -> TimeProvider:
        """Wall-clock provider (deferred per Task 2.C1 — see
        ``07-plugin-interfaces.md`` §12 OQ18 / implementation plan §6 Q4).

        Defaults to :func:`time.monotonic`. Subclasses override this to
        inject a fake clock once Task 2.C1 ships its seam.
        """
        return time.monotonic

    @pytest.fixture
    def fixture_image(self) -> str:
        """Substrate-resolvable image reference (per §7.5: a tiny Python
        runtime image).

        Default: ``python:3.12-slim`` — sufficient for the
        :class:`smai-compute-localgpu` reference. Substrate-specific
        plugins override (e.g., Modal-internal image references)."""
        return "python:3.12-slim"

    @pytest.fixture
    def poll_budget_seconds(self) -> float:
        """Max wall-clock the conformance suite waits for terminal state."""
        return 60.0

    # ---- §7.5 contract assertions ------------------------------------------

    async def test_submit_and_poll_succeeds(
        self,
        compute: Compute,
        fixture_image: str,
        time_provider: TimeProvider,
        poll_budget_seconds: float,
    ) -> None:
        """Submit a job, poll until ``state='succeeded'``, assert
        ``exit_code == 0``."""
        handle = await compute.submit(
            image=fixture_image,
            command=["python", "-c", "import sys; sys.exit(0)"],
            env={},
            timeout_seconds=int(poll_budget_seconds),
        )
        status = await _poll_until_terminal(compute, handle, time_provider, poll_budget_seconds)
        assert status.state == "succeeded"
        assert status.exit_code == 0

    async def test_submit_and_fail_nonzero_exit(
        self,
        compute: Compute,
        fixture_image: str,
        time_provider: TimeProvider,
        poll_budget_seconds: float,
    ) -> None:
        """Job exits non-zero → ``state='failed'``, ``exit_code != 0``."""
        handle = await compute.submit(
            image=fixture_image,
            command=["python", "-c", "import sys; sys.exit(2)"],
            env={},
            timeout_seconds=int(poll_budget_seconds),
        )
        status = await _poll_until_terminal(compute, handle, time_provider, poll_budget_seconds)
        assert status.state == "failed"
        assert status.exit_code is not None and status.exit_code != 0

    async def test_timeout_fires(
        self,
        compute: Compute,
        fixture_image: str,
        time_provider: TimeProvider,
        poll_budget_seconds: float,
    ) -> None:
        """A job sleeping longer than its timeout transitions to
        ``state='timeout'`` on the next ``.status()`` call."""
        handle = await compute.submit(
            image=fixture_image,
            command=["python", "-c", "import time; time.sleep(60)"],
            env={},
            timeout_seconds=2,
        )
        status = await _poll_until_terminal(compute, handle, time_provider, poll_budget_seconds)
        assert status.state == "timeout"

    async def test_cancel_terminates(
        self,
        compute: Compute,
        fixture_image: str,
        time_provider: TimeProvider,
        poll_budget_seconds: float,
    ) -> None:
        """Submit a long job, cancel mid-run, eventual
        ``state='cancelled'``.

        Cancellation is documented as eventually-consistent — the next
        ``.status()`` call will return ``cancelled`` once the substrate
        confirms.
        """
        handle = await compute.submit(
            image=fixture_image,
            command=["python", "-c", "import time; time.sleep(60)"],
            env={},
            timeout_seconds=int(poll_budget_seconds),
        )
        await compute.cancel(handle)
        status = await _poll_until_terminal(compute, handle, time_provider, poll_budget_seconds)
        assert status.state == "cancelled"
        # Idempotent — calling cancel on an already-terminal job is a no-op.
        await compute.cancel(handle)

    async def test_job_handle_reconnection(
        self,
        compute: Compute,
        fixture_image: str,
        time_provider: TimeProvider,
        poll_budget_seconds: float,
    ) -> None:
        """Submit job, serialize handle, deserialize from a fresh
        :class:`Compute` instance, poll status — must work across
        processes (per §7.5: "handle's reconnection state is encoded
        in :attr:`JobHandle.handle` / :attr:`JobHandle.metadata` so a
        fresh process can resume polling")."""
        handle = await compute.submit(
            image=fixture_image,
            command=["python", "-c", "import sys; sys.exit(0)"],
            env={},
            timeout_seconds=int(poll_budget_seconds),
        )
        # Round-trip through Pydantic JSON to simulate cross-process
        # serialization.
        round_tripped = JobHandle.model_validate_json(handle.model_dump_json())
        fresh = self.make_fresh_compute()
        status = await _poll_until_terminal(
            fresh, round_tripped, time_provider, poll_budget_seconds
        )
        assert status.state == "succeeded"

    async def test_logs_returns_stdout(
        self,
        compute: Compute,
        fixture_image: str,
        time_provider: TimeProvider,
        poll_budget_seconds: float,
    ) -> None:
        """Submit a job that prints a known string; ``logs()`` returns
        a string with that token present."""
        token = "smai-conformance-marker-7c3a"
        handle = await compute.submit(
            image=fixture_image,
            command=["python", "-c", f"print({token!r})"],
            env={},
            timeout_seconds=int(poll_budget_seconds),
        )
        await _poll_until_terminal(compute, handle, time_provider, poll_budget_seconds)
        logs = await compute.logs(handle)
        assert token in logs

    async def test_invalid_image_raises(self, compute: Compute) -> None:
        """Submit with a non-existent image; plugin raises
        :class:`JobImageInvalid` with the bad image in the payload.

        Per §7.4: "eager fail on bad image references, not lazy fail
        mid-run." Plugins that defer image validation to the substrate
        (no eager check) skip this test — the substrate-specific
        --integration mode covers them.
        """
        bad_image = "not-a-real-registry.invalid/never-pushed:nope"
        try:
            handle = await compute.submit(
                image=bad_image,
                command=["python", "-c", "pass"],
                env={},
                timeout_seconds=30,
            )
        except JobImageInvalid as e:
            assert e.image == bad_image
            return
        # Plugin accepted the submit — its substrate is responsible for
        # surfacing the failure later. The conformance suite skips
        # rather than asserts here; --integration covers the lazy path.
        await compute.cancel(handle)
        pytest.skip("plugin defers image validation to substrate; not eager-checked")

    def test_capabilities_are_capabilities_model(self, compute: Compute) -> None:
        """Capability attribute is the right Pydantic model."""
        assert isinstance(compute.capabilities, ComputeCapabilities)

    # ---- workspace distribution (agent-refactor Step 1) --------------------

    async def test_workspace_round_trip(
        self,
        compute: Compute,
        fixture_image: str,
        time_provider: TimeProvider,
        poll_budget_seconds: float,
        tmp_path: pathlib.Path,
    ) -> None:
        """Stage a directory tree, run a container that reads it and
        writes derived outputs, harvest, assert the host sees what the
        container wrote (§2 agent_refactor/compute_dispatch_decisions.md).

        The round-trip exercises three substrate behaviors at once:
        (a) ``stage_workspace`` materializes the source tree at the
        container's ``/workspace/`` mount point; (b) the container can
        read the staged files and write to the same location; (c)
        ``harvest_workspace`` returns the container-side writes to the
        host. Real ``submit`` is used (not a no-op) because (b) is the
        load-bearing assertion — a stage/harvest pair that bypasses the
        container cannot catch substrate-side mount-direction bugs.

        Plugins with ``workspace_distribution == "none"`` skip cleanly;
        substrates whose fake test seam cannot simulate the
        in-container mount may override this test to skip with the
        rationale documented locally.
        """
        if compute.capabilities.workspace_distribution == "none":
            pytest.skip(
                f"plugin {compute.name!r} declares workspace_distribution='none'; "
                "round-trip test does not apply."
            )

        source = tmp_path / "source"
        for relpath, payload in _WORKSPACE_FIXTURE_FILES:
            target = source / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)

        # Stage; round-trip the handle through Pydantic JSON to mirror
        # the cross-process persistence contract used by JobHandle in
        # ``test_job_handle_reconnection``.
        handle = await compute.stage_workspace(source)
        round_tripped = WorkspaceHandle.model_validate_json(handle.model_dump_json())

        # Container reads /workspace/README.md and writes a derived
        # marker file plus an exact-copy directory. Reading proves the
        # stage direction worked; writing proves the harvest direction
        # will have something to harvest. Command is shell-free (list
        # argv, no expansion) per the existing conformance fixtures'
        # style.
        program = (
            "import pathlib, shutil;"
            "ws = pathlib.Path('/workspace');"
            "src = (ws / 'README.md').read_bytes();"
            "(ws / 'echo.txt').write_bytes(src);"
            "shutil.copytree(ws / 'data', ws / 'data_copy')"
        )
        job_handle = await compute.submit(
            image=fixture_image,
            command=["python", "-c", program],
            env={},
            timeout_seconds=int(poll_budget_seconds),
            workspace=round_tripped,
        )
        status = await _poll_until_terminal(compute, job_handle, time_provider, poll_budget_seconds)
        assert status.state == "succeeded", (
            f"in-container worker exited non-success: {status!r}; "
            f"logs={await compute.logs(job_handle)!r}"
        )

        harvest = tmp_path / "harvest"
        await compute.harvest_workspace(round_tripped, harvest)

        expected: dict[str, bytes] = {
            relpath: payload for relpath, payload in _WORKSPACE_FIXTURE_FILES
        }
        expected["echo.txt"] = _WORKSPACE_FIXTURE_FILES[0][1]
        for relpath, payload in _WORKSPACE_FIXTURE_FILES:
            if relpath.startswith("data/"):
                expected[f"data_copy/{relpath[len('data/') :]}"] = payload

        harvested = {
            str(p.relative_to(harvest)): p.read_bytes() for p in harvest.rglob("*") if p.is_file()
        }
        assert harvested == expected, (
            f"workspace round-trip mismatch: "
            f"missing={set(expected) - set(harvested)!r}, "
            f"extra={set(harvested) - set(expected)!r}"
        )

    async def test_workspace_harvest_missing_handle_raises(
        self, compute: Compute, tmp_path: pathlib.Path
    ) -> None:
        """``harvest_workspace`` with an unknown handle raises
        :class:`WorkspaceNotFound` — the symmetric obligation to
        :class:`JobNotFound` on ``status``."""
        if compute.capabilities.workspace_distribution == "none":
            pytest.skip(f"plugin {compute.name!r} declares workspace_distribution='none'.")
        bogus = WorkspaceHandle(
            plugin=compute.name,
            handle="smai-conformance-unknown-workspace-handle-7c3a",
        )
        with pytest.raises(WorkspaceNotFound):
            await compute.harvest_workspace(bogus, tmp_path / "harvest")


# ---- private helpers --------------------------------------------------------


async def _poll_until_terminal(
    compute: Compute,
    handle: JobHandle,
    time_provider: TimeProvider,
    budget_seconds: float,
    poll_interval_seconds: float = 1.0,
) -> JobStatus:
    """Poll ``compute.status(handle)`` until terminal or budget exhausted."""
    import asyncio

    terminal_states = frozenset({"succeeded", "failed", "cancelled", "timeout"})
    start = time_provider()
    while True:
        status = await compute.status(handle)
        if status.state in terminal_states:
            return status
        if time_provider() - start > budget_seconds:
            raise TimeoutError(
                f"job {handle.handle!r} did not reach terminal state within "
                f"{budget_seconds}s; last state={status.state!r}"
            )
        await asyncio.sleep(poll_interval_seconds)
