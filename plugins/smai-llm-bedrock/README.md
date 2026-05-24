# smai-llm-bedrock

`LlmProvider` plugin: AWS Bedrock Converse adapter (`BedrockProvider`). The
`smai dev` default LLM provider. Wraps the `bedrock-runtime` Converse API
via `boto3` + `asyncio.to_thread` so the `call` method stays an `async def`
per the `LlmProvider` Protocol.

## Install

```
pip install smai-llm-bedrock
```

## Configuration

```yaml
plugins:
  llm_provider: bedrock
  llm_provider_config:
    region: us-east-1                              # default
    model_id: us.anthropic.claude-opus-4-6-v1     # required
```

Both keys are passed verbatim as `**kwargs` to `BedrockProvider(region, model_id)`.
The `region` default (`us-east-1`) is applied by `smai dev` when no config is
present; there is no in-plugin default, so `smai start` and `smai verify`
require it explicitly.

## Credentials

The plugin reads AWS credentials from the boto3 default chain (`~/.aws/`,
`AWS_PROFILE`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, IAM instance
role, etc.). There is no `api_key` constructor argument, so credentials never
enter shell history.

A bare credential chain is not enough. **Bedrock model access must be granted
explicitly in the AWS console for the configured model ID and region.** An
ungranted model fails at call time with
`AccessDeniedException: ... is not available for this account`. Grant access
at **AWS Console > Amazon Bedrock > Model access** before running
`smai verify`.

## Caching

For caching-eligible models (Claude tiers, Amazon Nova), the plugin applies
Bedrock `cachePoint` markers when the agent loop passes a `cache_config`.
`BedrockProvider.capabilities.supports_caching` is `True` for known
cache-eligible model IDs and `False` for any unrecognized model ID (a
conservative default; override by passing `capabilities=...` to the
constructor).

## Discovery

Registered via the `smai.llm_providers` entry-point group:

```toml
[project.entry-points."smai.llm_providers"]
bedrock = "smai_llm_bedrock:BedrockProvider"
```

Tier A integrators (the `smai` CLI) instantiate the plugin through entry-point
discovery; Tier B integrators import `BedrockProvider` directly.

## Tests

The always-on test suite runs against an in-process fake Bedrock client (no AWS
credentials required):

```bash
uv run pytest plugins/smai-llm-bedrock/
```

The credentialed lane (`tests/test_live.py`) exercises a real Bedrock
round-trip. It is skipped unless `BEDROCK_LIVE_TESTS=1` is set. Override the
model via `BEDROCK_LIVE_MODEL_ID` (default: `us.anthropic.claude-haiku-4-5-20251001-v1:0`,
the cheapest Claude tier). Never runs in CI:

```bash
BEDROCK_LIVE_TESTS=1 AWS_REGION=us-east-1 \
    uv run pytest plugins/smai-llm-bedrock/tests/test_live.py -v
```
