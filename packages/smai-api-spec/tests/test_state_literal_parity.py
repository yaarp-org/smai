"""Defends against silent drift between ``smai-api-spec``'s state Literals
and ``smai_orchestrator.entities.tracking``'s.

Per DEC-037 / ``smai_api_spec._common``: this package depends only on
Pydantic, so it can't import the orchestrator at runtime. The state
Literals (``ProposalState``, ``PaperState``, ``CGState``, ``EntryState``,
``RunState``) are duplicated. This test imports both sides at TEST time
(where ``smai-orchestrator`` is available as a workspace dev-dep) and
asserts the literal value sets are identical. Any state added to either
side without updating the other breaks this test loudly.
"""

from __future__ import annotations

from typing import get_args

import smai_api_spec._common as spec_common
from smai_orchestrator.entities.tracking import (
    CGState as OrchCGState,
)
from smai_orchestrator.entities.tracking import (
    EntryState as OrchEntryState,
)
from smai_orchestrator.entities.tracking import (
    PaperState as OrchPaperState,
)
from smai_orchestrator.entities.tracking import (
    ProposalState as OrchProposalState,
)
from smai_orchestrator.entities.tracking import (
    RunState as OrchRunState,
)


def _values(literal: object) -> tuple[object, ...]:
    return get_args(literal)


def test_proposal_state_parity() -> None:
    assert _values(spec_common.ProposalState) == _values(OrchProposalState)


def test_paper_state_parity() -> None:
    assert _values(spec_common.PaperState) == _values(OrchPaperState)


def test_cg_state_parity() -> None:
    assert _values(spec_common.CGState) == _values(OrchCGState)


def test_entry_state_parity() -> None:
    assert _values(spec_common.EntryState) == _values(OrchEntryState)


def test_run_state_parity() -> None:
    assert _values(spec_common.RunState) == _values(OrchRunState)
