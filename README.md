# smai

Private; under active development. Not yet ready for external use.

## Local development

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync           # install all workspace members and dev dependencies
uv run pytest     # run the test suite
```

The repo is a `uv` workspace; packages live under `packages/` and `plugins/`. See `designs/smai/implementation_plan.md` (in the upstream Yaarp repo) for the implementation roadmap.
