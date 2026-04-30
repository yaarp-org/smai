# smai-llm-anthropic

`LlmProvider` plugin: native Anthropic API adapter (`AnthropicProvider`).

Per `07-plugin-interfaces.md` §4 and Phase 3 Task 3.F5. Wraps the
official `anthropic` SDK's `AsyncAnthropic.messages.create` surface.

## Install

```
pip install smai-llm-anthropic
```

The Anthropic SDK reads `ANTHROPIC_API_KEY` from the environment at
client-construction time; the plugin does not take an `api_key`
constructor argument so credentials never enter shell history.
`ANTHROPIC_BASE_URL` is honored by the SDK as well (proxy / self-hosted
deployments).

## Discovery

Registered via the `smai.llm_providers` entry-point group:

```toml
[project.entry-points."smai.llm_providers"]
anthropic = "smai_llm_anthropic:AnthropicProvider"
```

Tier A integrators (the in-tree CLI / hosted backend) instantiate the
plugin through entry-point discovery; Tier B integrators import
`AnthropicProvider` directly.
