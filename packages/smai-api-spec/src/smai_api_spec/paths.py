"""URL path constants for the SMAI v1 HTTP API.

Per ``designs/smai/11-api.md`` §3 / §9.1: every API endpoint URL is
exported here as a module-level constant. Both SMAI's ``smai-api`` and
Yaarp v2's hosted API register routes by importing from this module —
URL drift between the two implementations is impossible by construction.

URL templates use Python ``str.format``-style placeholders
(``{proposal_id}``, ``{arxiv_id}``, etc.). Route registration code calls
``.replace("{proposal_id}", "{proposal_id:str}")`` (or the equivalent
FastAPI path-converter syntax) as needed.

Per ``11`` §9.2 the URL prefix (``/api/v1``) is the **wire** version;
the spec package's semver is the **schema** version. They evolve
independently — minor schema bumps do not change the URL prefix; a major
URL bump (``/api/v2``) is a separate documented event.
"""

from __future__ import annotations

# ---- Versioned root --------------------------------------------------------

API_V1_PREFIX = "/api/v1"

# ---- Proposals (primary input per DEC-032) — `11` §4.1 --------------------

PROPOSALS = f"{API_V1_PREFIX}/proposals"
PROPOSAL_DETAIL = f"{API_V1_PREFIX}/proposals/{{proposal_id}}"
PROPOSAL_APPROVE = f"{API_V1_PREFIX}/proposals/{{proposal_id}}/approve"
PROPOSAL_REJECT = f"{API_V1_PREFIX}/proposals/{{proposal_id}}/reject"

# ---- Papers (supporting utility per DEC-032) — `11` §4.2 ------------------

PAPERS = f"{API_V1_PREFIX}/papers"
PAPER_DETAIL = f"{API_V1_PREFIX}/papers/{{arxiv_id}}"
PAPER_PROMOTE_PARTIAL = f"{API_V1_PREFIX}/papers/{{arxiv_id}}/promote-partial"

# ---- Experiments (the `smai run` adapter) — `11` §4.3 ---------------------

EXPERIMENTS = f"{API_V1_PREFIX}/experiments"
EXPERIMENTS_COMPILE = f"{API_V1_PREFIX}/experiments/compile"

# ---- Comparison Groups — `11` §4.4 ----------------------------------------

COMPARISON_GROUPS = f"{API_V1_PREFIX}/comparison-groups"
COMPARISON_GROUP_DETAIL = f"{API_V1_PREFIX}/comparison-groups/{{cg_id}}"
COMPARISON_GROUP_STATUS = f"{API_V1_PREFIX}/comparison-groups/{{cg_id}}/status"
COMPARISON_GROUP_AGENT_STATUS = f"{API_V1_PREFIX}/comparison-groups/{{cg_id}}/agent-status"
COMPARISON_GROUP_ENTRIES = f"{API_V1_PREFIX}/comparison-groups/{{cg_id}}/entries"
COMPARISON_GROUP_ENTRY_DETAIL = f"{API_V1_PREFIX}/comparison-groups/{{cg_id}}/entries/{{entry_id}}"
COMPARISON_GROUP_EVALUATION = f"{API_V1_PREFIX}/comparison-groups/{{cg_id}}/evaluation"
COMPARISON_GROUP_ARTIFACTS = f"{API_V1_PREFIX}/comparison-groups/{{cg_id}}/artifacts"
COMPARISON_GROUP_ARTIFACT = f"{API_V1_PREFIX}/comparison-groups/{{cg_id}}/artifacts/{{path}}"

# ---- Runs — `11` §4.5 -----------------------------------------------------

RUNS = f"{API_V1_PREFIX}/runs"
RUN_DETAIL = f"{API_V1_PREFIX}/runs/{{run_id}}"

# ---- System (cross-cutting reads) — `11` §4.6 -----------------------------

SYSTEM_VERSION = f"{API_V1_PREFIX}/system/version"
SYSTEM_CONFIG = f"{API_V1_PREFIX}/system/config"
SYSTEM_PLUGINS = f"{API_V1_PREFIX}/system/plugins"
SYSTEM_VERIFY = f"{API_V1_PREFIX}/system/verify"
SYSTEM_DASHBOARD = f"{API_V1_PREFIX}/system/dashboard"
SYSTEM_MIGRATE_STATUS = f"{API_V1_PREFIX}/system/migrate-status"
SYSTEM_HEALTH = f"{API_V1_PREFIX}/system/health"

# ---- Live updates (SSE) — `11` §4.7 / §8 ----------------------------------

EVENTS = f"{API_V1_PREFIX}/events"


__all__ = [
    "API_V1_PREFIX",
    "COMPARISON_GROUPS",
    "COMPARISON_GROUP_AGENT_STATUS",
    "COMPARISON_GROUP_ARTIFACT",
    "COMPARISON_GROUP_ARTIFACTS",
    "COMPARISON_GROUP_DETAIL",
    "COMPARISON_GROUP_ENTRIES",
    "COMPARISON_GROUP_ENTRY_DETAIL",
    "COMPARISON_GROUP_EVALUATION",
    "COMPARISON_GROUP_STATUS",
    "EVENTS",
    "EXPERIMENTS",
    "EXPERIMENTS_COMPILE",
    "PAPERS",
    "PAPER_DETAIL",
    "PAPER_PROMOTE_PARTIAL",
    "PROPOSALS",
    "PROPOSAL_APPROVE",
    "PROPOSAL_DETAIL",
    "PROPOSAL_REJECT",
    "RUNS",
    "RUN_DETAIL",
    "SYSTEM_CONFIG",
    "SYSTEM_DASHBOARD",
    "SYSTEM_HEALTH",
    "SYSTEM_MIGRATE_STATUS",
    "SYSTEM_PLUGINS",
    "SYSTEM_VERIFY",
    "SYSTEM_VERSION",
]
