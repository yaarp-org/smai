""":class:`ComputeConformance` subclass for :class:`LocalGpuCompute`.

Per Task 2.A4 of ``designs/smai/implementation_plan.md`` §3.3 and
``07-plugin-interfaces.md`` §7.5. The 8 contract methods on
:class:`smai_core.plugins.conformance.ComputeConformance` run against a
real :class:`LocalGpuCompute` whenever a Docker daemon is reachable; on
hosts without Docker (CI, dev machines without Docker installed) the
suite skips cleanly.

Skip-if-no-Docker rationale: the conformance tests genuinely shell out
to ``docker``. Recorded fixtures would have to mock the CLI's full
state surface — image pulls, container lifecycle, log retrieval — and
the resulting tape would be longer than the implementation it's testing.
The right boundary is "real Docker if available, skip otherwise"; the
``--integration`` shape per ``07`` §7.5 is implicit here (this whole
file is the integration shape for LocalGpu).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from smai_compute_localgpu import LocalGpuCompute
from smai_core.plugins import Compute, ComputeUnavailable
from smai_core.plugins.conformance import ComputeConformance


def _docker_daemon_reachable() -> bool:
    """Return True iff ``docker info`` succeeds in under 5 seconds.

    Used to skip-collect the whole class when Docker isn't available.
    Mirrors :class:`LocalGpuCompute`'s preflight; we duplicate the
    check here rather than catching :class:`ComputeUnavailable` from
    the constructor because pytest's ``pytest.skip`` only works inside
    test bodies / fixtures, and ``pytest.importorskip``-style
    module-level skipping requires this gate up front.
    """
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


_DOCKER_AVAILABLE = _docker_daemon_reachable()

# Apply the skip at class collection time. Each contract method picks
# up the marker via inheritance.
pytestmark = pytest.mark.skipif(
    not _DOCKER_AVAILABLE,
    reason=(
        "Docker daemon not reachable; skipping LocalGpuCompute conformance. "
        "Install Docker Desktop (or a compatible OCI runtime) and start the "
        "daemon to run this suite."
    ),
)


class TestLocalGpuComputeConformance(ComputeConformance):
    """Run the 8-method :class:`ComputeConformance` contract suite
    against :class:`LocalGpuCompute`.

    Per Task 2.A4: ``make_compute()`` returns a real
    :class:`LocalGpuCompute` instance with the SMAI agent default tag;
    ``make_fresh_compute()`` returns a separately-constructed instance
    so the cross-process reconnection contract (§7.5) genuinely
    exercises a fresh in-process state.
    """

    def make_compute(self) -> Compute:
        try:
            return LocalGpuCompute(agent_image="smai-agent:dev")
        except ComputeUnavailable as exc:  # pragma: no cover — guarded by pytestmark
            pytest.skip(f"Docker preflight failed: {exc}")

    def make_fresh_compute(self) -> Compute:
        # Returns a fresh instance with the same configuration — the
        # plugin holds no in-process state shared with the original
        # (every call is a fresh subprocess) but the contract is
        # cleaner with a separately-constructed object.
        try:
            return LocalGpuCompute(agent_image="smai-agent:dev")
        except ComputeUnavailable as exc:  # pragma: no cover
            pytest.skip(f"Docker preflight failed: {exc}")
