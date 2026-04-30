# smai-llm-openai

`LlmProvider` plugin: OpenAI Chat Completions adapter (`OpenAIProvider`).

Per `07-plugin-interfaces.md` §4 and Phase 3 Task 3.F5. Wraps the
official `openai` SDK's `AsyncOpenAI.chat.completions.create` surface.

## Install

```
pip install smai-llm-openai
```

The OpenAI SDK reads `OPENAI_API_KEY` from the environment at
client-construction time; the plugin does not take an `api_key`
constructor argument so credentials never enter shell history.
`OPENAI_BASE_URL` is honored by the SDK as well (proxy / Azure-OpenAI /
self-hosted deployments).

## Discovery

Registered via the `smai.llm_providers` entry-point group:

```toml
[project.entry-points."smai.llm_providers"]
openai = "smai_llm_openai:OpenAIProvider"
```

Tier A integrators (the in-tree CLI / hosted backend) instantiate the
plugin through entry-point discovery; Tier B integrators import
`OpenAIProvider` directly.

## Caching

OpenAI's chat-completions API does **not** expose explicit prompt-cache
markers (the platform applies automatic caching server-side based on
prefix length). The plugin reports `supports_caching=False`, so any
`cache_config` passed by the agent loop is silently ignored per the
`07-plugin-interfaces.md` §4.3 contract.
