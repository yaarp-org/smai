# smai-llm-anthropic

`LlmProvider` plugin: native Anthropic API adapter (`AnthropicProvider`).
Wraps the official `anthropic` SDK's `AsyncAnthropic.messages.create` surface.

## Install

```
pip install smai-llm-anthropic
```

## Configuration

```yaml
plugins:
  llm_provider: anthropic
  llm_provider_config:
    model_id: claude-opus-4-7    # default
```

The `model_id` key is passed as `**kwargs` to `AnthropicProvider(model_id=...)`.

## Credentials

The Anthropic SDK reads `ANTHROPIC_API_KEY` from the environment at
client-construction time; the plugin does not take an `api_key`
constructor argument so credentials never enter shell history.
`ANTHROPIC_BASE_URL` is honored by the SDK as well (proxy / self-hosted
deployments).

## Caching

The plugin applies `cache_control: ephemeral` markers when the agent loop
passes a `cache_config`. `AnthropicProvider.capabilities.supports_caching`
is `True` for known Claude model IDs.

## Discovery

Registered via the `smai.llm_providers` entry-point group:

```toml
[project.entry-points."smai.llm_providers"]
anthropic = "smai_llm_anthropic:AnthropicProvider"
```

Tier A integrators (the `smai` CLI) instantiate the plugin through entry-point
discovery; Tier B integrators import `AnthropicProvider` directly.
