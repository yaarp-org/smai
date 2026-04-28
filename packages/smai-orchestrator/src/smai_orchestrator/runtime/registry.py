"""Process-local :class:`PipelineSpec` registry.

Per ``designs/smai/05-orchestrator.md`` §5.1 / §7.1: a pipeline-spec is
constructed once at worker startup and frozen for the duration of the
run. The registry is the lookup-by-name surface that
:class:`RuntimeConfig.pipelines` resolves against.

There is no central registry shared across processes — each worker
process maintains its own. Spec authors call
:func:`register_pipeline_spec` at import time (or at CLI bootstrap
time) and the engine resolves the names at startup. The
:func:`reset_registry` helper exists for test isolation; production
code never calls it.
"""

from __future__ import annotations

from threading import Lock

from smai_orchestrator.runtime.spec import PipelineSpec


class DuplicateSpecError(ValueError):
    """Raised when :func:`register_pipeline_spec` is called twice with
    the same :attr:`PipelineSpec.name`.

    Re-registration is rejected to surface accidental name collisions
    across spec modules; tests that legitimately re-register call
    :func:`reset_registry` first.
    """


class SpecNotRegisteredError(LookupError):
    """Raised when :func:`get_pipeline_spec` is called with a name
    that has no registered :class:`PipelineSpec`.

    Carries the registered names in :attr:`available` so the CLI's
    error surface can produce a helpful diagnostic without re-walking
    the registry.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(
            f"no PipelineSpec registered as {name!r}; "
            f"registered: {sorted(available)}"
        )


_lock = Lock()
_registry: dict[str, PipelineSpec] = {}


def register_pipeline_spec(spec: PipelineSpec) -> None:
    """Register ``spec`` under its :attr:`PipelineSpec.name`.

    Raises :class:`DuplicateSpecError` if a spec with the same name is
    already registered; tests use :func:`reset_registry` to clear
    between runs.
    """
    with _lock:
        if spec.name in _registry:
            raise DuplicateSpecError(
                f"PipelineSpec {spec.name!r} is already registered"
            )
        _registry[spec.name] = spec


def get_pipeline_spec(name: str) -> PipelineSpec:
    """Return the registered :class:`PipelineSpec` for ``name``.

    Raises :class:`SpecNotRegisteredError` (a :class:`LookupError`
    subclass) if no spec with that name is registered.
    """
    with _lock:
        if name not in _registry:
            raise SpecNotRegisteredError(name, list(_registry.keys()))
        return _registry[name]


def list_registered_specs() -> list[str]:
    """Return the names of all currently registered specs.

    Used by ``smai plugins`` (Task 2.D2) to print the pipeline-spec
    inventory alongside the plugin inventory.
    """
    with _lock:
        return sorted(_registry.keys())


def reset_registry() -> None:
    """Clear the registry — test isolation only.

    Production code never calls this. Tests that exercise registration
    flows call this in their teardown so cross-test ordering doesn't
    leak state.
    """
    with _lock:
        _registry.clear()


__all__ = [
    "DuplicateSpecError",
    "SpecNotRegisteredError",
    "get_pipeline_spec",
    "list_registered_specs",
    "register_pipeline_spec",
    "reset_registry",
]
