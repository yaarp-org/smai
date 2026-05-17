""":class:`ModalCompute` — :class:`Compute` adapter for Modal Sandboxes.

Per ``07-plugin-interfaces.md`` §7 (the full Protocol surface), DEC-021
(Modal Sandboxes carry-forward), and ``modal_migration.md`` (the v1
reference for "how Modal worked in production"). Implementation plan
§3.4 Task 3.F3.

Substrate is **Modal Sandboxes** — Modal's per-job ephemeral container
primitive. Each :meth:`submit` creates a fresh Sandbox; :meth:`status`
polls it; :meth:`cancel` terminates it; :meth:`logs` reads the
Sandbox's stdout / stderr StreamReaders.

Driver choice: synchronous Modal SDK wrapped with :func:`asyncio.to_thread`
at the plugin's async surface — the same pattern as ``smai-llm-bedrock``
and ``smai-artifacts-s3``. Modal does expose an async surface via
``.aio`` method-attribute shims (e.g., ``await Sandbox.create.aio(...)``)
in its current ``synchronicity``-backed SDK; the sync-plus-``to_thread``
pattern was chosen because:

* The mocking shape stays simple — a duck-typed module fake exposes
  plain ``MagicMock``-compatible callables instead of async coroutines
  with both ``.aio`` and sync entrypoints.
* Pattern parity with the two other production plugins (Bedrock + S3)
  keeps the codebase coherent for plugin authors writing the next one.
* Per-call latency is dominated by the Modal RPC, not in-process
  scheduling — the thread-pool dispatch adds negligible overhead at v1
  scale.

**Image distribution:** Modal Sandboxes accept either a Modal-built
``Image`` or a registry-hosted reference via ``Image.from_registry(tag)``.
The plugin uses :func:`modal.Image.from_registry` so the ``image``
parameter passed to :meth:`submit` is a substrate-resolvable Docker tag
the operator has already published to a registry Modal can pull from.
The plugin does **not** auto-build or auto-publish images — operators
own image publication out-of-band per ``07`` §7.4.

**Authentication:** Modal's SDK reads ``MODAL_TOKEN_ID`` and
``MODAL_TOKEN_SECRET`` from the environment by default. The plugin does
not accept token args at construction (same hygiene as
``smai-llm-bedrock`` + ``smai-artifacts-s3``) — credentials never enter
shell history.

**Not in scope** for this plugin:

* Multi-app routing — single ``app_name`` at construction.
* Auto image build / publication — operator responsibility.
* Modal Function deployment (vs Sandbox-per-job) — out of scope per
  DEC-021 (v1 used Sandboxes; v2 follows the same pattern).
* Cost accounting (per ``07`` §7.4).
* Pricing / concurrency limits.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from typing import Any

from smai_core.plugins import (
    ComputeCapabilities,
    ComputeUnavailable,
    JobHandle,
    JobImageInvalid,
    JobNotFound,
    JobState,
    JobStatus,
)

# === Constants ==============================================================

# Modal Sandbox 24-hour cap per ``modal_migration.md`` §3 / Modal docs.
# Surfaced as :attr:`ComputeCapabilities.max_timeout_seconds` so callers
# can plan around it (the engine's lease loop honors this when sizing
# its own timeout policy).
_MODAL_MAX_TIMEOUT_SECONDS = 24 * 60 * 60

# Default Modal app name when caller does not supply one. Matches the
# convention in the credentialed integration test
# (``tests/integration/test_smai_start_credentialed.py``) — a single
# shared app namespace per deployment.
_DEFAULT_APP_NAME = "smai"

# Default GPU type when ``gpu=True`` is requested without a ``gpu_type``
# plugin option. T4 is the v1 reference per DEC-021 / ``modal_migration.md``
# §13 — sufficient for the classification workloads in scope and the
# cheapest GPU offering on Modal.
_DEFAULT_GPU_TYPE = "T4"


# === Implementation =========================================================


class ModalCompute:
    """Modal-Sandbox reference implementation of :class:`Compute`.

    Constructor::

        ModalCompute()  # app_name="smai", default GPU="T4"
        ModalCompute(
            app_name="my-deployment",
            default_gpu_type="A100",
            modal_module=fake_modal,  # tests only
        )

    Modal credentials are picked up from the SDK's default chain
    (``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` env vars or
    ``~/.modal.toml``) — the constructor takes no token args.

    The ``modal_module`` keyword argument is a test seam — production
    callers leave it ``None`` and the plugin lazy-imports the real
    ``modal`` package. Tests pass a duck-typed fake exposing the
    surfaces the plugin actually calls (``App.lookup``,
    ``Image.from_registry``, ``Sandbox.create`` / ``.from_id``).
    """

    name: str = "modal"
    capabilities: ComputeCapabilities

    def __init__(
        self,
        *,
        app_name: str = _DEFAULT_APP_NAME,
        default_gpu_type: str = _DEFAULT_GPU_TYPE,
        max_timeout_seconds: int = _MODAL_MAX_TIMEOUT_SECONDS,
        modal_module: Any = None,
    ) -> None:
        self._app_name = app_name
        self._default_gpu_type = default_gpu_type
        self._modal: Any = modal_module if modal_module is not None else _import_modal()
        # Cached after the first :meth:`submit` so repeat calls within
        # the same plugin instance reuse the App handle. ``None`` until
        # the first :meth:`_ensure_app` call.
        self._app: Any = None
        self.capabilities = ComputeCapabilities(
            supports_gpu=True,
            max_timeout_seconds=int(max_timeout_seconds),
            supports_log_streaming=False,
            # Modal pulls the job image from a registry via
            # ``Image.from_registry`` — the operator must publish it.
            requires_published_image=True,
        )

    # --- public surface (Compute Protocol) ---------------------------------

    async def submit(
        self,
        image: str,
        command: list[str],
        env: dict[str, str],
        gpu: bool = False,
        timeout_seconds: int = 3600,
        **plugin_options: object,
    ) -> JobHandle:
        """Submit a Sandbox job (§7.2).

        ``plugin_options`` accepts:

        * ``gpu_type: str`` — Modal GPU spec (``"T4"`` / ``"L4"`` /
          ``"A100"`` / ``"A100-80GB"`` / ``"H100"`` etc.). When ``gpu``
          is ``True`` and ``gpu_type`` is omitted, the constructor's
          ``default_gpu_type`` is used.
        * ``cpu: float`` — CPU request (Modal's ``cpu`` arg).
        * ``memory_mb: int`` — Memory request in MiB (Modal's ``memory``
          arg).

        Image validation is implicitly eager: Modal's
        ``Sandbox.create`` is what triggers the registry pull, so a
        non-existent image surfaces here as a Modal exception that the
        plugin translates to :class:`JobImageInvalid` per §7.4.
        """
        gpu_type = self._resolve_gpu_spec(gpu, plugin_options)
        cpu = _coerce_optional_float(plugin_options.get("cpu"), "cpu")
        memory_mb = _coerce_optional_int(plugin_options.get("memory_mb"), "memory_mb")

        # Cap the substrate-imposed maximum so a misconfigured caller
        # cannot ask Modal for a 48-hour Sandbox.
        effective_timeout = min(int(timeout_seconds), self.capabilities.max_timeout_seconds)

        image_obj = await _to_thread(self._modal.Image.from_registry, image)
        app = await self._ensure_app()

        create_kwargs: dict[str, Any] = {
            "app": app,
            "image": image_obj,
            "timeout": int(effective_timeout),
        }
        if env:
            create_kwargs["env"] = dict(env)
        if gpu_type is not None:
            create_kwargs["gpu"] = gpu_type
        if cpu is not None:
            create_kwargs["cpu"] = cpu
        if memory_mb is not None:
            create_kwargs["memory"] = memory_mb

        try:
            sandbox = await _to_thread(
                self._modal.Sandbox.create,
                *list(command),
                **create_kwargs,
            )
        except Exception as exc:
            if _is_image_pull_error(exc):
                raise JobImageInvalid(image, str(exc)) from exc
            raise ComputeUnavailable(f"modal.Sandbox.create failed: {exc!r}") from exc

        sandbox_id = _read_attr(sandbox, "object_id")
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise ComputeUnavailable(
                "modal.Sandbox.create returned a sandbox with no object_id; "
                "Modal SDK state is unexpected"
            )

        submitted_at = _utc_now_iso()
        metadata: dict[str, Any] = {
            "sandbox_id": sandbox_id,
            "submitted_at": submitted_at,
            "timeout_seconds": int(effective_timeout),
            "image": image,
            "gpu": bool(gpu),
            "command": list(command),
        }
        if gpu_type is not None:
            metadata["gpu_type"] = gpu_type

        return JobHandle(
            plugin=self.name,
            handle=sandbox_id,
            metadata=metadata,
        )

    async def status(self, handle: JobHandle) -> JobStatus:
        """Poll the Sandbox's state.

        Per ``07`` §7.2: MUST raise :class:`JobNotFound` if the
        substrate has no record of the handle; MUST NOT block waiting
        for state changes. Modal's ``Sandbox.poll`` returns ``None``
        while running and the integer return code once terminal — we
        read it without waiting.
        """
        sandbox = await self._reconnect(handle)
        try:
            returncode_raw = await _to_thread(sandbox.poll)
        except Exception as exc:
            if _is_not_found_error(exc):
                raise JobNotFound(handle) from exc
            raise ComputeUnavailable(f"modal.Sandbox.poll failed: {exc!r}") from exc

        cancel_requested = bool(handle.metadata.get("cancel_requested"))
        timeout_seconds = int(handle.metadata.get("timeout_seconds", 3600))
        submitted_at_raw = handle.metadata.get("submitted_at")
        submitted_at = submitted_at_raw if isinstance(submitted_at_raw, str) else None

        if returncode_raw is None:
            # Still running. Defensive timeout enforcement: Modal already
            # enforces ``timeout=`` on the Sandbox, but if the next poll
            # arrives after Modal's timer fired but before Modal has
            # marked the sandbox terminal, we'd return ``running``
            # forever — which the orchestrator would treat as a stuck
            # job. Trip the local timeout one wall-clock check above
            # the substrate's, so the caller sees ``timeout`` rather
            # than waiting on a hung Sandbox.
            if submitted_at is not None and _elapsed_seconds(submitted_at) > timeout_seconds:
                with contextlib.suppress(Exception):
                    await _to_thread(sandbox.terminate)
                return JobStatus(
                    state="timeout",
                    exit_code=None,
                    started_at=submitted_at,
                    finished_at=_utc_now_iso(),
                    failure_reason=(
                        f"job exceeded timeout_seconds={timeout_seconds}; "
                        "sandbox terminated by plugin"
                    ),
                )
            return JobStatus(
                state="running",
                exit_code=None,
                started_at=submitted_at,
                finished_at=None,
                failure_reason=None,
            )

        # Terminal. Modal's ``poll`` returned an int — translate per the
        # state-mapping matrix (see module docstring of ``__init__``):
        #
        #   - exit 0                                  -> succeeded
        #   - cancel_requested + non-zero exit        -> cancelled
        #   - elapsed > timeout_seconds + non-zero    -> timeout
        #   - other non-zero                          -> failed
        try:
            exit_code: int | None = int(returncode_raw)
        except (TypeError, ValueError):
            exit_code = None

        finished_at = _utc_now_iso()

        job_state: JobState
        failure_reason: str | None = None
        if exit_code == 0:
            job_state = "succeeded"
        elif cancel_requested:
            job_state = "cancelled"
        elif submitted_at is not None and _elapsed_seconds(submitted_at) >= timeout_seconds:
            job_state = "timeout"
            failure_reason = (
                f"sandbox exceeded timeout_seconds={timeout_seconds} (exit_code={exit_code})"
            )
        else:
            job_state = "failed"
            failure_reason = f"sandbox exited with non-zero status {exit_code}"

        return JobStatus(
            state=job_state,
            exit_code=exit_code,
            started_at=submitted_at,
            finished_at=finished_at,
            failure_reason=failure_reason,
        )

    async def logs(self, handle: JobHandle) -> str:
        """Return ``stdout`` + ``stderr`` concatenated (§7.2).

        Modal's ``Sandbox.stdout`` / ``.stderr`` properties yield
        :class:`StreamReader` objects whose ``.read()`` drains the
        currently-buffered output. Long logs are returned in full at v1
        — Modal's per-stream buffer is bounded by the platform.
        """
        sandbox = await self._reconnect(handle)
        try:
            stdout = await _to_thread(sandbox.stdout.read)
            stderr = await _to_thread(sandbox.stderr.read)
        except Exception as exc:
            if _is_not_found_error(exc):
                raise JobNotFound(handle) from exc
            raise ComputeUnavailable(f"modal Sandbox log read failed: {exc!r}") from exc
        return _coerce_str(stdout) + _coerce_str(stderr)

    async def cancel(self, handle: JobHandle) -> None:
        """Send terminate to the Sandbox (§7.2).

        Per ``07`` §7.2: eventually-consistent — the next ``status``
        call will return ``state='cancelled'`` once Modal confirms the
        terminal state. Idempotent: calling cancel on an
        already-terminal sandbox is a no-op (Modal's ``terminate``
        succeeds with no-op semantics on terminated sandboxes).
        """
        # Mark intent on the handle so :meth:`status` can distinguish
        # "user-cancelled" from "naturally failed". The exit code
        # alone is ambiguous — Modal terminate / Modal timeout / a
        # genuine non-zero crash all surface as non-zero return codes.
        try:
            handle.metadata["cancel_requested"] = True
        except (TypeError, AttributeError):  # pragma: no cover - defensive
            pass

        sandbox = await self._reconnect(handle)
        try:
            await _to_thread(sandbox.terminate)
        except Exception as exc:
            if _is_not_found_error(exc):
                # Idempotent — the sandbox is already gone.
                return
            raise ComputeUnavailable(f"modal.Sandbox.terminate failed: {exc!r}") from exc

    # --- internal helpers --------------------------------------------------

    async def _ensure_app(self) -> Any:
        """Look up (or create) the Modal App on first use.

        ``modal.App.lookup(name, create_if_missing=True)`` is idempotent
        from Modal's side; we cache the App handle locally so repeat
        ``submit`` calls within the same plugin instance avoid the
        round-trip.
        """
        if self._app is not None:
            return self._app
        self._app = await _to_thread(
            self._modal.App.lookup,
            self._app_name,
            create_if_missing=True,
        )
        return self._app

    async def _reconnect(self, handle: JobHandle) -> Any:
        """Reconnect to a Sandbox by id, translating not-found.

        Per ``07`` §7.2's reconnection contract: a fresh process must
        be able to resume polling using only the persisted
        :class:`JobHandle`. Modal's ``Sandbox.from_id`` is the seam.
        """
        sandbox_id = self._sandbox_id(handle)
        try:
            return await _to_thread(self._modal.Sandbox.from_id, sandbox_id)
        except Exception as exc:
            if _is_not_found_error(exc):
                raise JobNotFound(handle) from exc
            raise ComputeUnavailable(f"modal.Sandbox.from_id failed: {exc!r}") from exc

    def _sandbox_id(self, handle: JobHandle) -> str:
        """Read the Modal sandbox id off a :class:`JobHandle`.

        Falls back to ``handle.handle`` when ``metadata['sandbox_id']``
        is absent — round-tripping through Pydantic JSON preserves
        metadata, but ``handle.handle`` is the canonical reference."""
        meta_id = handle.metadata.get("sandbox_id")
        if isinstance(meta_id, str) and meta_id:
            return meta_id
        return handle.handle

    def _resolve_gpu_spec(
        self,
        gpu: bool,
        plugin_options: dict[str, object],
    ) -> str | None:
        """Return the Modal ``gpu`` argument value, or ``None`` for CPU."""
        if not gpu:
            return None
        gpu_type = plugin_options.get("gpu_type")
        if gpu_type is None:
            return self._default_gpu_type
        if not isinstance(gpu_type, str) or not gpu_type:
            raise ValueError(
                f"plugin_options['gpu_type'] must be a non-empty str (e.g., 'T4'), "
                f"got {type(gpu_type).__name__}"
            )
        return gpu_type


# === Module-level helpers ===================================================


def _import_modal() -> Any:
    """Lazy-import the Modal SDK with a clear missing-dep message.

    Mirrors the pattern in :mod:`smai_artifacts_s3` for ``boto3`` —
    the plugin's import surface stays light enough that ``pyright``
    can analyze it on a CI runner that hasn't synced workspace deps.
    """
    try:
        import modal  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - declared dep
        raise RuntimeError(
            "smai-compute-modal requires the `modal` SDK; install with "
            "`pip install smai-compute-modal`"
        ) from exc
    return modal


async def _to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    """:func:`asyncio.to_thread` with widened typing for the duck-typed
    ``modal`` callables.

    A thin wrapper so the call sites stay readable. ``modal``'s sync
    surface is ``synchronicity``-driven and not statically typed, so
    we accept ``Any`` and let the runtime contract carry the load.
    """
    import asyncio  # noqa: PLC0415

    return await asyncio.to_thread(func, *args, **kwargs)


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC, always with explicit ``Z`` suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _elapsed_seconds(submitted_at_iso: str) -> float:
    """Return wall-clock seconds since ``submitted_at_iso``.

    Tolerant of the trailing ``Z`` form (Python 3.11's
    :func:`datetime.fromisoformat` accepts ``Z`` only on 3.12+) — strip
    it and supply a UTC offset.
    """
    iso = submitted_at_iso.replace("Z", "+00:00")
    submitted = datetime.fromisoformat(iso)
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=UTC)
    return (datetime.now(UTC) - submitted).total_seconds()


def _coerce_str(value: Any) -> str:
    """Coerce stream-read output to ``str``.

    Modal's ``StreamReader[str].read`` returns ``str`` in production;
    duck-typed fakes may return ``bytes`` or ``None``. Normalize both
    into the empty-string fallback rather than raising.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _read_attr(obj: Any, name: str) -> Any:
    """Property-safe attribute read.

    Modal's ``Sandbox.object_id`` is a ``synchronicity``-wrapped
    property; reading it can raise if the sandbox isn't fully
    hydrated. We treat any exception as "no id" so the caller's
    error path fires the right :class:`ComputeUnavailable`.
    """
    try:
        return getattr(obj, name)
    except Exception:  # noqa: BLE001 - duck-typed boundary
        return None


def _coerce_optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; ``True``/``False`` would
        # otherwise pass through as 1.0/0.0 without warning.
        raise ValueError(f"plugin_options[{name!r}] must be a number, got bool")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"plugin_options[{name!r}] must be int|float|None, got {type(value).__name__}")


def _coerce_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"plugin_options[{name!r}] must be an int, got bool")
    if isinstance(value, int):
        return value
    raise ValueError(f"plugin_options[{name!r}] must be int|None, got {type(value).__name__}")


# Substrings Modal surfaces in image-pull failures. Modal's exception
# class hierarchy (``modal.exception.NotFoundError`` /
# ``InvalidError``) does not include a dedicated image-pull subtype, so
# message-pattern matching is the cleanest seam — same approach as
# :mod:`smai_compute_localgpu` uses against ``docker pull`` stderr.
_IMAGE_PULL_MARKERS: tuple[str, ...] = (
    "image",
    "manifest",
    "registry",
    "pull",
)


def _is_image_pull_error(exc: BaseException) -> bool:
    """True iff an exception from ``Sandbox.create`` looks like a
    pull failure on a non-existent / unreachable image.

    Examined primarily through the message contents: the SDK's typed
    exception hierarchy has shifted across versions (``NotFoundError``
    vs ``InvalidError`` vs internal wrapping in :class:`modal.Error`),
    and registry-side errors ("manifest unknown", "no such image",
    "pull access denied") sometimes surface through wrapped
    ``RuntimeError`` / ``Error`` types. String-matching on the
    message is the most stable seam — same approach as
    :mod:`smai_compute_localgpu` against ``docker pull`` stderr.

    A ``Sandbox.create`` ``NotFoundError`` whose message references an
    image / manifest / registry / pull is unambiguously an image-pull
    failure; the only other Modal entities ``Sandbox.create`` looks
    up are App (handled by ``_ensure_app``) and the image itself.

    Image *build* failures are folded in here too: when an operator
    points :attr:`smai_orchestrator.engine.config.EngineConfig.runtime_image`
    at a tag Modal cannot resolve, ``Sandbox.create`` surfaces a
    ``RemoteError: Image build ... failed`` — class name ``RemoteError``,
    message containing "image" + "build" but none of pull / registry /
    manifest. The permissive fallback recognizes "image" co-present
    with "build" so that lands as :class:`JobImageInvalid` (an
    actionable image-config error) rather than :class:`ComputeUnavailable`
    (a transient-substrate error the caller would pointlessly retry).
    """
    msg = str(exc).lower()
    # Strong signals: the registry-side error vocabulary surfaces
    # verbatim from the underlying engine.
    if any(marker in msg for marker in ("manifest unknown", "no such image", "pull access denied")):
        return True
    # Type-name-plus-marker fallback: a Modal-typed not-found / invalid
    # exception whose message references the image marker words.
    cls_name = type(exc).__name__
    if cls_name.endswith(("NotFoundError", "InvalidError")) and any(
        marker in msg for marker in _IMAGE_PULL_MARKERS
    ):
        return True
    # Permissive fallback: any exception whose message simultaneously
    # references "image" and either "pull" / "registry" / "manifest" /
    # "build" — covers wrapped ``RuntimeError`` shapes that lose the
    # typed parentage on their way to the plugin, and the
    # ``RemoteError: Image build ... failed`` shape. "build" is scoped
    # to require "image" co-present so an unrelated build-step error
    # does not over-match.
    if "image" in msg and any(
        marker in msg for marker in ("pull", "registry", "manifest", "build")
    ):
        return True
    return False


def _is_not_found_error(exc: BaseException) -> bool:
    """True iff an exception from a Sandbox lookup means "no such sandbox".

    Two shapes are covered:

    * :class:`modal.exception.NotFoundError` — Modal raises this when
      ``Sandbox.from_id`` is given a well-formed id the substrate has
      no record of.
    * :class:`modal.exception.InvalidError` whose message says the id
      isn't a valid Sandbox ID — ``Sandbox.from_id`` rejects malformed
      ids (anything not shaped like ``sb-...``) client-side, before any
      round-trip. The substrate has, by definition, no record of a
      syntactically invalid handle, so this satisfies the `07` §7.2
      ``JobNotFound`` contract — and it lets ``smai verify``'s
      read-only ``status`` probe (which uses a deliberately-bogus
      handle) pass against valid Modal credentials instead of failing
      with a wrapped :class:`ComputeUnavailable`.

    A trailing ``"not found"`` substring match stays as a permissive
    belt — the SDK's typed exception hierarchy is version-sensitive and
    string-matching is the safest fallback. Class-name checks use a
    suffix match (same approach as :func:`_is_image_pull_error`) so
    test fakes (``FakeNotFoundError`` / ``FakeInvalidError``) and any
    wrapped subclass are recognized too.
    """
    cls_name = type(exc).__name__
    if cls_name.endswith("NotFoundError"):
        return True
    msg = str(exc).lower()
    if cls_name.endswith("InvalidError") and "not a valid" in msg and "sandbox" in msg:
        return True
    return "not found" in msg or "no such sandbox" in msg


__all__ = ["ModalCompute"]
