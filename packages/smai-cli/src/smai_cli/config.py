"""Config layering — defaults → ``smai.yaml`` → ``SMAI_*`` env → flag overrides.

Per ``designs/smai/09-cli.md`` §2 / §5.1. The CLI is the only consumer
of :class:`smai_orchestrator.RuntimeConfig` that owns the layering
pipeline (the typed model itself is shipped by Task 2.C3 — see
``smai_orchestrator.runtime.config``).

Layering precedence (later wins, field-granular):

1. **Defaults** — :func:`dev_defaults` for ``smai dev``;
   :func:`base_defaults` for non-``dev`` callers (tests, programmatic
   integrators).
2. **File** — ``smai.yaml`` from ``--config PATH`` → ``$SMAI_CONFIG``
   → ``./smai.yaml`` → ``$XDG_CONFIG_HOME/smai/config.yaml``. First hit
   wins; absent files fall through silently.
3. **Environment** — ``SMAI_*`` vars per `09` §2.3 conventions.
   Double-underscore separates nested fields (``SMAI_ENGINE__POLL_INTERVAL_SECONDS``).
4. **Flags** — per-verb CLI flag overrides (passed in as a dict by the
   verb implementation in :mod:`smai_cli.main`).

The output is a frozen :class:`RuntimeConfig` — there is no
hot-reload (`09` §3 / `05-orchestrator.md` §9 Q6). Re-reading config
means restarting the worker.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError
from smai_orchestrator import RuntimeConfig

CG_EXECUTION_SPEC_NAME = "smai_cg_execution"
CG_ENTRIES_SPEC_NAME = "smai_cg_entries"

# `09` §5.1 default pipelines: Phase 2 ships only the two CG-execution
# specs; Phase 3 adds the proposal + paper-ingestion specs (Tasks 3.E1
# / 3.E2). The CLI lists what's registered today.
PHASE_2_DEFAULT_PIPELINES: tuple[str, ...] = (
    CG_EXECUTION_SPEC_NAME,
    CG_ENTRIES_SPEC_NAME,
)


# === File search order (`09` §2.4) ==========================================

_CONFIG_FILE_BASENAME = "smai.yaml"
_USER_CONFIG_FILENAME = "config.yaml"


class ConfigFileError(RuntimeError):
    """Raised when ``smai.yaml`` is malformed or unreadable.

    Crashes happen at config-load time so misspellings don't show up
    deep in plugin instantiation (`09` §2.2).
    """


class ConfigValidationError(ValueError):
    """Raised when the layered config fails :class:`RuntimeConfig`
    validation.

    Wraps Pydantic's :class:`ValidationError` so the CLI can render a
    friendly error without the user having to read a Pydantic
    traceback.
    """


def _candidate_config_paths(*, explicit_path: Path | None, env: Mapping[str, str]) -> list[Path]:
    """Build the search order per `09` §2.4."""
    out: list[Path] = []
    if explicit_path is not None:
        out.append(explicit_path)
    env_path = env.get("SMAI_CONFIG")
    if env_path:
        out.append(Path(env_path))
    out.append(Path.cwd() / _CONFIG_FILE_BASENAME)
    xdg_home = env.get("XDG_CONFIG_HOME")
    if xdg_home:
        out.append(Path(xdg_home) / "smai" / _USER_CONFIG_FILENAME)
    else:
        out.append(Path.home() / ".config" / "smai" / _USER_CONFIG_FILENAME)
    return out


def _find_config_file(*, explicit_path: Path | None, env: Mapping[str, str]) -> Path | None:
    for candidate in _candidate_config_paths(explicit_path=explicit_path, env=env):
        if candidate.is_file():
            return candidate
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigFileError(f"could not read config file {path}: {exc}") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"malformed YAML in config file {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigFileError(
            f"config file {path} must contain a YAML mapping at the top level; "
            f"got {type(loaded).__name__}"
        )
    return _ensure_str_keys(loaded, where=str(path))


def _ensure_str_keys(value: Any, *, where: str) -> Any:
    """Recursively reject non-string keys.

    YAML allows non-string mapping keys; our :class:`RuntimeConfig`
    surface assumes strings throughout. Rejecting non-string keys at
    the load boundary surfaces config typos clearly.
    """
    if isinstance(value, dict):
        raw: dict[Any, Any] = value  # type: ignore[type-arg]
        out: dict[str, Any] = {}
        for k_obj, v_obj in raw.items():
            if not isinstance(k_obj, str):
                raise ConfigFileError(
                    f"non-string mapping key {k_obj!r} in {where}; YAML mapping keys "
                    f"must be strings throughout"
                )
            out[k_obj] = _ensure_str_keys(v_obj, where=where)
        return out
    if isinstance(value, list):
        items: list[Any] = value  # type: ignore[type-arg]
        return [_ensure_str_keys(item, where=where) for item in items]
    return value


# === Defaults (`09` §5.1) ====================================================


def base_defaults() -> dict[str, Any]:
    """Conservative defaults — `09` §3 minimum surface, no plugin defaults.

    Programmatic / test callers without a config file or env get an
    empty plugins block (so :class:`RuntimeConfig` validation will
    require either config-file values or env overrides). For
    ``smai dev`` use :func:`dev_defaults` instead.
    """
    return {
        "engine": {},
        "plugins": {},
        "pipelines": list(PHASE_2_DEFAULT_PIPELINES),
    }


def dev_defaults() -> dict[str, Any]:
    """The ``smai dev`` zero-config defaults per `09` §5.1.

    Lower-latency :attr:`EngineConfig.poll_interval_seconds=10` (vs
    ``30`` in production), single worker, sqlite + localfs + localgpu
    + bedrock.
    """
    return {
        "engine": {
            "poll_interval_seconds": 10,
            "worker_count": 1,
            "fair_scheduling": "off",
        },
        "plugins": {
            "llm_provider": "bedrock",
            "metadata_store": "sqlite",
            "artifact_store": "localfs",
            "compute": "localgpu",
            "llm_provider_config": {
                "region": "us-east-1",
                "model_id": "us.anthropic.claude-opus-4-6-v1",
            },
            "metadata_store_config": {
                # Default in-memory; CLI's `smai dev` overrides with a
                # file-backed URI rooted at ``$SMAI_HOME/state.db``.
            },
            "artifact_store_config": {
                # Default ``LocalFsStore`` root from $SMAI_ARTIFACTS_ROOT
                # or ~/.smai/artifacts, applied by the plugin itself.
            },
            "compute_config": {
                # LocalGpuCompute uses smai-runtime:dev / smai-runtime-cpu:dev
                # by default per `10-runtime-and-templates.md` §8.5.
            },
        },
        "pipelines": list(PHASE_2_DEFAULT_PIPELINES),
    }


# === Deep-merge =============================================================


def _deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Field-granular deep merge — ``over`` wins on conflicts.

    Lists are replaced (not concatenated); dicts are merged
    recursively. Per `09` §2: layering precedence is field-granular,
    not document-level.
    """
    out: dict[str, Any] = dict(base)
    for key, value in over.items():
        existing: Any = out.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            existing_dict = cast("dict[str, Any]", existing)
            value_map = cast("Mapping[str, Any]", value)
            out[key] = _deep_merge(existing_dict, value_map)
        else:
            out[key] = value
    return out


# === Environment-variable layering (`09` §2.3) ==============================

_ENV_PREFIX = "SMAI_"
_NESTED_SEPARATOR = "__"

# Env vars that don't map to RuntimeConfig fields — read elsewhere.
_RESERVED_ENV_VARS: frozenset[str] = frozenset(
    {
        "SMAI_CONFIG",
        # Plugin-internal (read by plugin constructors, not CLI):
        "SMAI_ARTIFACTS_ROOT",
        "SMAI_HOME",
    }
)

# Per-role model env vars (`09` §2.3 / DEC-022) — surfaced through
# :mod:`smai_cli.per_role_llm`; layered config doesn't see them.
_MODEL_ENV_VAR_RE = re.compile(r"^SMAI_MODEL_[A-Z_]+$")


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Project ``SMAI_*`` vars onto the :class:`RuntimeConfig` shape.

    Non-prefixed and reserved vars are skipped silently. Values are
    parsed as JSON if they parse cleanly (``{...}`` / ``[...]`` /
    booleans / numbers); otherwise retained as raw strings.
    """
    out: dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(_ENV_PREFIX):
            continue
        if raw_key in _RESERVED_ENV_VARS:
            continue
        if _MODEL_ENV_VAR_RE.match(raw_key):
            # Per-role model overrides flow through smai_cli.per_role_llm.
            continue
        path = _env_path(raw_key)
        value: Any = _parse_env_value(raw_value)
        _assign_nested(out, path, value)
    return out


def _env_path(raw_key: str) -> list[str]:
    """``SMAI_ENGINE__POLL_INTERVAL_SECONDS`` → ``["engine", "poll_interval_seconds"]``.

    Single-underscore-within-segment is preserved (so
    ``SMAI_ENGINE__POLL_INTERVAL_SECONDS`` works as expected); double-
    underscore separates nesting levels per `09` §2.3.
    """
    body = raw_key[len(_ENV_PREFIX) :]
    return [segment.lower() for segment in body.split(_NESTED_SEPARATOR) if segment]


def _parse_env_value(raw: str) -> Any:
    """Best-effort scalar parse for env values.

    Accepts ``true`` / ``false`` (case-insensitive), JSON-shaped
    objects / arrays / numbers, and falls through to raw string.
    Strings that look like integers ("30") become ints; floats stay
    floats; everything else stays a string.
    """
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    # Try int / float before JSON to avoid the "30" → 30 ambiguity (
    # JSON parses "30" the same way, but an explicit fast path is clearer).
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith(("{", "[", '"')):
        import json  # noqa: PLC0415 — small surface, lazy import is fine

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _assign_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    """Place ``value`` at ``path`` inside ``target`` (creating intermediate dicts)."""
    if not path:
        return
    cursor: dict[str, Any] = target
    for segment in path[:-1]:
        nxt: Any = cursor.get(segment)
        if isinstance(nxt, dict):
            cursor = cast("dict[str, Any]", nxt)
        else:
            new_node: dict[str, Any] = {}
            cursor[segment] = new_node
            cursor = new_node
    cursor[path[-1]] = value


# === Public API =============================================================


def load_runtime_config(
    *,
    config_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    flag_overrides: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> RuntimeConfig:
    """Layer the four config sources into a frozen :class:`RuntimeConfig`.

    ``defaults`` defaults to :func:`base_defaults`; pass
    :func:`dev_defaults` for ``smai dev``. ``env`` defaults to
    :data:`os.environ` (for testability — pass a fixture dict in tests
    to avoid leaking real env vars). ``flag_overrides`` is the per-verb
    CLI flag layer (deepest precedence).

    Raises :class:`ConfigFileError` if the file is malformed and
    :class:`ConfigValidationError` if the merged config doesn't satisfy
    :class:`RuntimeConfig`.
    """
    env_map: Mapping[str, str] = env if env is not None else os.environ
    layered = dict(defaults if defaults is not None else base_defaults())

    config_file = _find_config_file(explicit_path=config_path, env=env_map)
    if config_file is not None:
        file_overrides = _load_yaml(config_file)
        layered = _deep_merge(layered, file_overrides)

    env_layer = _env_overrides(env_map)
    if env_layer:
        layered = _deep_merge(layered, env_layer)

    if flag_overrides:
        layered = _deep_merge(layered, dict(flag_overrides))

    engine_block: dict[str, Any] = layered.get("engine") or {}
    plugins_block: dict[str, Any] = layered.get("plugins") or {}
    pipelines_block: list[Any] = layered.get("pipelines") or []
    # The ``api`` block is optional (Task 4.L1 / `12-ui-process.md` §9.1).
    # ``smai dev`` / ``smai start`` ignore it; ``smai ui`` reads it.
    # Empty dict here lets :class:`ApiConfig`'s field defaults take over.
    api_block: dict[str, Any] = layered.get("api") or {}

    try:
        return RuntimeConfig.model_validate(
            {
                "engine": engine_block,
                "plugins": plugins_block,
                "pipelines": pipelines_block,
                "api": api_block,
            }
        )
    except ValidationError as exc:
        raise ConfigValidationError(_format_validation_error(exc, config_file)) from exc


def _format_validation_error(exc: ValidationError, config_file: Path | None) -> str:
    """Render a Pydantic :class:`ValidationError` as a friendly message.

    The default ``str(exc)`` is technically informative but
    visually noisy; the CLI wraps it with the config-file pointer
    (when present) so the user knows where to fix.
    """
    where = f" (loaded from {config_file})" if config_file else ""
    bullets = "\n".join(f"  - {err['loc']}: {err['msg']}" for err in exc.errors())
    return f"invalid SMAI runtime config{where}:\n{bullets}"


__all__ = [
    "CG_ENTRIES_SPEC_NAME",
    "CG_EXECUTION_SPEC_NAME",
    "ConfigFileError",
    "ConfigValidationError",
    "PHASE_2_DEFAULT_PIPELINES",
    "base_defaults",
    "dev_defaults",
    "load_runtime_config",
]
