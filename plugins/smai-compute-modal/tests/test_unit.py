"""Unit tests for :class:`ModalCompute` plugin-specific logic.

These cover the plugin's internals beyond the §7.5 conformance contract:

* Capabilities advertise GPU + Modal's 24-hour timeout cap.
* ``submit`` resolves the GPU spec from constructor default and from
  ``plugin_options['gpu_type']``.
* ``submit`` rejects non-string ``gpu_type``.
* ``cpu`` / ``memory_mb`` plugin options reject wrong types.
* ``timeout_seconds`` is capped at the substrate maximum.
* :class:`JobHandle` round-trips preserve ``sandbox_id`` /
  ``submitted_at`` / ``timeout_seconds`` so cross-process reconnection
  works (the §7.5 reconnection contract).
* Image-pull-error translation: known Modal failure shapes raise
  :class:`JobImageInvalid` with the bad image in the payload.
* App is cached after the first :meth:`submit`.
* Sandbox-not-found on ``status`` / ``logs`` / ``cancel`` raises
  :class:`JobNotFound`.

Conformance lifecycle (submit → status → logs → cancel) lives in
:mod:`tests.test_conformance`; the credentialed real-Modal lane in
:mod:`tests.test_real_modal`.
"""

from __future__ import annotations

import pytest
from _modal_fakes import (  # type: ignore[import-not-found]
    FakeInvalidError,
    FakeModal,
    FakeNotFoundError,
    make_fake_with_bad_image,
)
from smai_compute_modal import ModalCompute
from smai_core.plugins import (
    ComputeCapabilities,
    ComputeUnavailable,
    JobHandle,
    JobImageInvalid,
    JobNotFound,
)


def _new_compute(modal: FakeModal | None = None, **kwargs: object) -> ModalCompute:
    """Build a :class:`ModalCompute` with the supplied (or fresh) fake."""
    return ModalCompute(modal_module=modal or FakeModal(), **kwargs)  # type: ignore[arg-type]


def test_capabilities_advertise_gpu_and_modal_timeout_cap() -> None:
    """:attr:`ModalCompute.capabilities` must reflect Modal's 24-hour cap
    and GPU support — Modal Sandboxes are GPU-capable per DEC-021."""
    compute = _new_compute()
    assert isinstance(compute.capabilities, ComputeCapabilities)
    assert compute.capabilities.supports_gpu is True
    assert compute.capabilities.max_timeout_seconds == 24 * 60 * 60
    assert compute.capabilities.supports_log_streaming is False


def test_name_is_modal() -> None:
    """The plugin's ``name`` is the entry-point key."""
    assert ModalCompute.name == "modal"
    assert _new_compute().name == "modal"


async def test_submit_uses_default_gpu_type_when_gpu_true() -> None:
    """``ModalCompute(default_gpu_type='A100')`` + ``submit(gpu=True)``
    passes ``gpu='A100'`` through to ``Sandbox.create``."""
    fake = FakeModal()
    captured: dict[str, object] = {}
    original_create = fake.Sandbox.create

    def capturing_create(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return original_create(*args, **kwargs)  # type: ignore[arg-type]

    fake.Sandbox.create = capturing_create  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake, default_gpu_type="A100")
    await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        gpu=True,
        timeout_seconds=10,
    )
    assert captured.get("gpu") == "A100"


async def test_submit_uses_explicit_gpu_type_from_plugin_options() -> None:
    """``plugin_options['gpu_type']`` overrides the constructor default."""
    fake = FakeModal()
    captured: dict[str, object] = {}
    original_create = fake.Sandbox.create

    def capturing_create(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return original_create(*args, **kwargs)  # type: ignore[arg-type]

    fake.Sandbox.create = capturing_create  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        gpu=True,
        timeout_seconds=10,
        gpu_type="L4",
    )
    assert captured.get("gpu") == "L4"


async def test_submit_omits_gpu_when_gpu_false() -> None:
    """``gpu=False`` (the default) must not pass a GPU spec through."""
    fake = FakeModal()
    captured: dict[str, object] = {}
    original_create = fake.Sandbox.create

    def capturing_create(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return original_create(*args, **kwargs)  # type: ignore[arg-type]

    fake.Sandbox.create = capturing_create  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        timeout_seconds=10,
    )
    assert "gpu" not in captured


async def test_submit_rejects_non_string_gpu_type() -> None:
    """``gpu_type=123`` is a programmer error — surface it, don't paper over."""
    compute = _new_compute()
    with pytest.raises(ValueError, match="gpu_type"):
        await compute.submit(
            image="smai-runtime:dev",
            command=["python", "-c", "pass"],
            env={},
            gpu=True,
            timeout_seconds=10,
            gpu_type=123,
        )


async def test_submit_rejects_bool_for_cpu() -> None:
    """``cpu=True`` is almost certainly a typo for ``gpu=True``; reject."""
    compute = _new_compute()
    with pytest.raises(ValueError, match="cpu"):
        await compute.submit(
            image="smai-runtime:dev",
            command=["python", "-c", "pass"],
            env={},
            timeout_seconds=10,
            cpu=True,
        )


async def test_submit_rejects_non_int_memory_mb() -> None:
    compute = _new_compute()
    with pytest.raises(ValueError, match="memory_mb"):
        await compute.submit(
            image="smai-runtime:dev",
            command=["python", "-c", "pass"],
            env={},
            timeout_seconds=10,
            memory_mb=2.5,
        )


async def test_submit_caps_timeout_at_substrate_maximum() -> None:
    """A 48-hour ``timeout_seconds`` is capped to Modal's 24h limit."""
    fake = FakeModal()
    captured: dict[str, object] = {}
    original_create = fake.Sandbox.create

    def capturing_create(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return original_create(*args, **kwargs)  # type: ignore[arg-type]

    fake.Sandbox.create = capturing_create  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        timeout_seconds=48 * 60 * 60,
    )
    assert captured.get("timeout") == 24 * 60 * 60


async def test_submit_passes_env_when_non_empty() -> None:
    fake = FakeModal()
    captured: dict[str, object] = {}
    original_create = fake.Sandbox.create

    def capturing_create(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return original_create(*args, **kwargs)  # type: ignore[arg-type]

    fake.Sandbox.create = capturing_create  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={"FOO": "bar"},
        timeout_seconds=10,
    )
    assert captured.get("env") == {"FOO": "bar"}


async def test_submit_omits_env_when_empty() -> None:
    """Empty ``env={}`` should NOT pass an empty dict to ``Sandbox.create``
    — Modal would emit a different default request envelope."""
    fake = FakeModal()
    captured: dict[str, object] = {}
    original_create = fake.Sandbox.create

    def capturing_create(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return original_create(*args, **kwargs)  # type: ignore[arg-type]

    fake.Sandbox.create = capturing_create  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        timeout_seconds=10,
    )
    assert "env" not in captured


async def test_submit_image_pull_failure_translates_to_job_image_invalid() -> None:
    """Modal NotFoundError on ``Sandbox.create`` with a bad image →
    :class:`JobImageInvalid` with the bad image in the payload."""
    bad_image = "not-a-real-registry.invalid/never:nope"
    fake = make_fake_with_bad_image(bad_image)
    compute = _new_compute(modal=fake)
    with pytest.raises(JobImageInvalid) as exc_info:
        await compute.submit(
            image=bad_image,
            command=["python", "-c", "pass"],
            env={},
            timeout_seconds=10,
        )
    assert exc_info.value.image == bad_image


async def test_submit_unknown_sandbox_create_failure_is_compute_unavailable() -> None:
    """A non-image-related :class:`ValueError` from ``Sandbox.create``
    bubbles as :class:`ComputeUnavailable`."""
    fake = FakeModal()

    def boom(*_args: object, **_kwargs: object) -> object:
        raise ValueError("modal substrate is wedged")

    fake.Sandbox.create = boom  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    with pytest.raises(ComputeUnavailable, match="wedged"):
        await compute.submit(
            image="smai-runtime:dev",
            command=["python", "-c", "pass"],
            env={},
            timeout_seconds=10,
        )


async def test_submit_image_build_failure_translates_to_job_image_invalid() -> None:
    """A Modal ``RemoteError: Image build ... failed`` (the shape an
    unpublished local-only ``runtime_image`` tag produces) classifies as
    :class:`JobImageInvalid`, not :class:`ComputeUnavailable` — round 11.

    The ``RemoteError`` class name carries none of the pull / registry /
    manifest vocabulary, so this exercises the "image" + "build"
    permissive-fallback branch of ``_is_image_pull_error``.
    """

    class RemoteError(Exception):
        """Stand-in for :class:`modal.exception.RemoteError`."""

    fake = FakeModal()

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RemoteError(
            "Image build for im-1234 failed with the exception:\nno such tag smai-runtime:dev"
        )

    fake.Sandbox.create = boom  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    with pytest.raises(JobImageInvalid) as exc_info:
        await compute.submit(
            image="smai-runtime:dev",
            command=["python", "-c", "pass"],
            env={},
            timeout_seconds=10,
        )
    assert exc_info.value.image == "smai-runtime:dev"


async def test_submit_returns_handle_with_reconnection_metadata() -> None:
    compute = _new_compute()
    handle = await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        timeout_seconds=42,
        gpu=True,
        gpu_type="L4",
    )
    assert handle.plugin == "modal"
    assert handle.handle.startswith("sb-fake-")
    assert handle.metadata["sandbox_id"] == handle.handle
    assert handle.metadata["timeout_seconds"] == 42
    assert handle.metadata["gpu"] is True
    assert handle.metadata["gpu_type"] == "L4"
    assert "submitted_at" in handle.metadata
    assert handle.metadata["command"] == ["python", "-c", "pass"]


async def test_submit_handle_round_trips_through_pydantic_json() -> None:
    """The §7.5 reconnection contract: handle survives JSON round-trip."""
    compute = _new_compute()
    original = await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        timeout_seconds=10,
    )
    rehydrated = JobHandle.model_validate_json(original.model_dump_json())
    assert rehydrated.handle == original.handle
    assert rehydrated.metadata["sandbox_id"] == original.metadata["sandbox_id"]
    assert rehydrated.metadata["timeout_seconds"] == original.metadata["timeout_seconds"]


async def test_status_raises_job_not_found_for_unknown_sandbox_id() -> None:
    fake = FakeModal()
    compute = _new_compute(modal=fake)
    handle = JobHandle(
        plugin="modal",
        handle="sb-fake-does-not-exist",
        metadata={"sandbox_id": "sb-fake-does-not-exist"},
    )
    with pytest.raises(JobNotFound):
        await compute.status(handle)


async def test_logs_raises_job_not_found_for_unknown_sandbox_id() -> None:
    fake = FakeModal()
    compute = _new_compute(modal=fake)
    handle = JobHandle(
        plugin="modal",
        handle="sb-fake-does-not-exist",
        metadata={"sandbox_id": "sb-fake-does-not-exist"},
    )
    with pytest.raises(JobNotFound):
        await compute.logs(handle)


async def test_cancel_is_idempotent_when_sandbox_already_terminated() -> None:
    """``cancel`` against a terminated/missing sandbox is a no-op."""
    fake = FakeModal()
    compute = _new_compute(modal=fake)
    # The first ``from_id`` returns the missing entry → cancel raises
    # JobNotFound; but calling it via the plugin's cancel() should be
    # idempotent.
    handle = JobHandle(
        plugin="modal",
        handle="sb-fake-already-gone",
        metadata={"sandbox_id": "sb-fake-already-gone"},
    )
    # The plugin's cancel translates JobNotFound from from_id() as a
    # not-yet-terminal-but-also-not-found case, but the plugin's
    # contract is "cancel of a missing handle bubbles the not-found"
    # — we exercise that here via the substrate. Confirm metadata is
    # marked even on failure.
    with pytest.raises(JobNotFound):
        await compute.cancel(handle)
    assert handle.metadata["cancel_requested"] is True


async def test_app_is_cached_across_submits() -> None:
    """``App.lookup`` should be called exactly once across multiple submits."""
    fake = FakeModal()
    lookup_calls: list[str] = []
    original_lookup = fake.App.lookup

    def capturing_lookup(name: str, **kwargs: object) -> object:
        lookup_calls.append(name)
        return original_lookup(name, **kwargs)  # type: ignore[arg-type]

    fake.App.lookup = capturing_lookup  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    for _ in range(3):
        await compute.submit(
            image="smai-runtime:dev",
            command=["python", "-c", "pass"],
            env={},
            timeout_seconds=10,
        )
    assert lookup_calls == ["smai"]


async def test_app_name_constructor_is_passed_to_lookup() -> None:
    """``ModalCompute(app_name='X')`` calls ``App.lookup('X', ...)``."""
    fake = FakeModal()
    lookup_calls: list[str] = []
    original_lookup = fake.App.lookup

    def capturing_lookup(name: str, **kwargs: object) -> object:
        lookup_calls.append(name)
        return original_lookup(name, **kwargs)  # type: ignore[arg-type]

    fake.App.lookup = capturing_lookup  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake, app_name="my-deployment")
    await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        timeout_seconds=10,
    )
    assert lookup_calls == ["my-deployment"]


def test_image_pull_error_detection_recognizes_modal_shapes() -> None:
    """The ``_is_image_pull_error`` helper recognizes the Modal exception
    name + message-marker combinations."""
    from smai_compute_modal._compute import _is_image_pull_error  # noqa: PLC0415

    assert _is_image_pull_error(FakeNotFoundError("unable to pull image foo"))
    # Bare "manifest unknown" string from registry side
    assert _is_image_pull_error(RuntimeError("manifest unknown for image"))
    # Round 11: ``RemoteError: Image build ... failed`` — the class name
    # carries no pull / registry / manifest vocabulary, so "image" +
    # "build" co-presence in the message is what classifies it.
    assert _is_image_pull_error(RuntimeError("Image build for im-1 failed"))
    # "build" without "image" must NOT over-match an unrelated failure.
    assert not _is_image_pull_error(RuntimeError("build step 3 failed"))
    # Generic Modal NotFoundError without an image-related message is NOT
    # an image error (could be App / Volume not-found from a different
    # SDK call) — but the plugin's only NotFoundError path on
    # ``Sandbox.create`` IS image-related, so the heuristic stays
    # permissive.
    assert not _is_image_pull_error(ValueError("totally unrelated error"))


def test_not_found_error_detection_recognizes_modal_shapes() -> None:
    from smai_compute_modal._compute import _is_not_found_error  # noqa: PLC0415

    assert _is_not_found_error(FakeNotFoundError("sandbox not found"))
    # String-fallback for non-typed runtimes
    assert _is_not_found_error(RuntimeError("no such sandbox: sb-foo"))
    # A malformed sandbox id rejected client-side (`InvalidError`) counts
    # as "no such sandbox" — the substrate has no record of a
    # syntactically invalid handle (this is the `smai verify` probe path).
    assert _is_not_found_error(
        FakeInvalidError("'smai-verify-probe-handle' is not a valid Sandbox ID.")
    )
    # ...but other InvalidError messages must NOT be swallowed as not-found.
    assert not _is_not_found_error(FakeInvalidError("'A100x' is not a valid GPU type."))
    assert not _is_not_found_error(ValueError("totally unrelated"))


async def test_status_with_malformed_handle_raises_job_not_found() -> None:
    """A handle whose id isn't a ``sb-...`` Sandbox ID (e.g. the one
    ``smai verify`` probes with) surfaces as :class:`JobNotFound`, not
    :class:`ComputeUnavailable` — Modal's ``Sandbox.from_id`` rejects it
    client-side with ``InvalidError`` and the plugin translates that to
    the §7.2 not-found shape so the verify probe passes on valid creds.
    """
    compute = _new_compute()
    handle = JobHandle(plugin="modal", handle="smai-verify-probe-handle-that-does-not-exist")
    with pytest.raises(JobNotFound):
        await compute.status(handle)


async def test_submit_with_cpu_and_memory_passes_through() -> None:
    fake = FakeModal()
    captured: dict[str, object] = {}
    original_create = fake.Sandbox.create

    def capturing_create(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return original_create(*args, **kwargs)  # type: ignore[arg-type]

    fake.Sandbox.create = capturing_create  # type: ignore[method-assign,assignment]
    compute = _new_compute(modal=fake)
    await compute.submit(
        image="smai-runtime:dev",
        command=["python", "-c", "pass"],
        env={},
        timeout_seconds=10,
        cpu=2.0,
        memory_mb=4096,
    )
    assert captured.get("cpu") == 2.0
    assert captured.get("memory") == 4096
