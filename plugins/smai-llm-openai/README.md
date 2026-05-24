# smai-llm-openai

`LlmProvider` plugin: OpenAI Chat Completions adapter (`OpenAIProvider`).
Wraps the official `openai` SDK's `AsyncOpenAI.chat.completions.create`
surface.

## Install

```
pip install smai-llm-openai
```

## Configuration

```yaml
plugins:
  llm_provider: openai
  llm_provider_config:
    model_id: gpt-4o    # default
```

The `model_id` key is passed as `**kwargs` to `OpenAIProvider(model_id=...)`.

## Credentials

The OpenAI SDK reads `OPENAI_API_KEY` from the environment at
client-construction time; the plugin does not take an `api_key`
constructor argument so credentials never enter shell history.
**The SDK requires `OPENAI_API_KEY` at construction**, so even `smai verify`
fails without it. `OPENAI_BASE_URL` is honored by the SDK as well (proxy /
Azure-OpenAI / self-hosted deployments).

## Caching

OpenAI's chat-completions API does not expose explicit prompt-cache markers
(the platform applies automatic caching server-side based on prefix length).
The plugin reports `supports_caching=False`, so any `cache_config` passed by
the agent loop is silently ignored per the `LlmProvider` Protocol contract.

## Discovery

Registered via the `smai.llm_providers` entry-point group:

```toml
[project.entry-points."smai.llm_providers"]
openai = "smai_llm_openai:OpenAIProvider"
```

Tier A integrators (the `smai` CLI) instantiate the plugin through entry-point
discovery; Tier B integrators import `OpenAIProvider` directly.
