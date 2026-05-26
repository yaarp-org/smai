# smai-agent-runtime

Sandbox-side agent runtime: the mini-orchestrator plus PydanticAI Agents
plus provider SDKs that run *inside* the agent sandbox container image
(`smai-agent-runtime:dev`), invoked by the host worker's unified
`make_compute_dispatcher` factory.

Per `designs/smai/agent_refactor/architectural_decisions.md` §3 and
`compute_dispatch_decisions.md` §6 (DEC-038): this package is consumed
via container image, not by the host process. It has its own dependency
policy (PydanticAI + provider SDKs) distinct from the methodology-layer
allowlist.

## Status

Step 3 of the agent-layer refactor (this commit) ships the package
skeleton only:

- `python -m smai_agent_runtime --role <role> --cg-id <id>` dispatches
  to a per-role mini-orchestrator.
- Role bodies (`harness_builder`, `technique_implementer`) raise
  `RoleNotImplementedError` and exit with `EXIT_NOT_IMPLEMENTED` (64).
  Step 4 of the refactor lands `harness_builder`; Step 7 lands
  `technique_implementer`.

## Roles

| Role | Status | Filled in step |
|---|---|---|
| `harness_builder` | stub | Step 4 |
| `technique_implementer` | stub | Step 7 |
| `planner` | rejected (inline role) | n/a |

The `planner` role is in the dispatch table only so a typo on `--role`
gets an argparse "invalid choice" rejection rather than a softer "wrong
place" message; the planner itself runs in the host worker process per
`architectural_decisions.md` §6.

## Dependencies

- `smai-core` (methodology layer)
- `smai-runtime` (the mini-orchestrator imports `smai_runtime.runner`
  for the validation smoke + the `HarnessAPIManifest` shape per D4 §3)
- `pydantic-ai`, `anthropic`, `openai`, `boto3` — pinned in the
  sandbox image's Dockerfile; loose floors on the host so `uv sync`
  resolves cleanly for IDE / type-checking purposes
