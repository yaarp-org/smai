"""In-process Modal SDK fake — the conformance suite's test seam.

The conformance suite must run deterministically without Modal
credentials (per Task 3.F3's no-credentials-in-CI directive). This
module exposes a duck-typed fake of the surfaces :class:`ModalCompute`
actually calls — ``App.lookup``, ``Image.from_registry``,
``Sandbox.create`` / ``.from_id`` / ``.poll`` / ``.terminate`` /
``.stdout.read`` / ``.stderr.read`` — and runs the user's
``command`` in a fresh subprocess so the conformance fixtures (which
``python -c "..."``) execute the way they would on a real Sandbox.

The ``commands`` interface is intentionally narrow: each fake Sandbox
spawns a real OS subprocess, captures its stdout/stderr, and answers
:meth:`poll` with the integer return code (or ``None`` while running).
This keeps the conformance suite both honest (the tested job actually
ran and produced the expected exit code) and offline (no Modal RPC).

The fake mirrors the production :class:`Sandbox` shape closely enough
that the plugin code path is exercised in full — image-pull error
translation, ``object_id`` round-tripping, ``terminate`` idempotence,
and ``from_id`` reconnection all run through real plugin code with
the fake on the substrate side.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import uuid
from typing import Any


class FakeNotFoundError(Exception):
    """Stand-in for :class:`modal.exception.NotFoundError`."""


class FakeInvalidError(Exception):
    """Stand-in for :class:`modal.exception.InvalidError`."""


class _FakeStreamReader:
    """Stand-in for :class:`modal.io_streams.StreamReader[str]`.

    Wraps an already-captured ``str`` payload — :meth:`read` drains
    it. Repeated calls return the empty string after the first drain
    (matches Modal's StreamReader semantics: read once until EOF).
    """

    def __init__(self, payload: str) -> None:
        self._buf = payload
        self._drained = False

    def read(self) -> str:
        if self._drained:
            return ""
        self._drained = True
        return self._buf


class FakeSandbox:
    """Stand-in for :class:`modal.Sandbox`.

    Spawns a real OS subprocess on construction so the conformance
    suite's fixture jobs (``python -c "exit(0)"``, etc.) actually run
    and produce real stdout / exit codes. The fake honors the
    ``timeout`` Modal parameter by attaching a timer thread that sends
    the configured signal when the timeout elapses.
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None,
        timeout: int,
    ) -> None:
        self.object_id: str = f"sb-fake-{uuid.uuid4().hex[:12]}"
        self._timeout = timeout
        self._timer: threading.Timer | None = None
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        try:
            self._proc: subprocess.Popen[str] | None = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
                text=True,
            )
        except FileNotFoundError as exc:
            # Treat missing executable as Modal would — an "image pull"
            # failure in the fake's vocabulary, since the substrate has
            # no record of the executable to run. Conformance suite's
            # ``test_invalid_image_raises`` exercises the upstream
            # path (image-pull failure on ``Sandbox.create``) so the
            # OS-level FileNotFoundError here is genuinely unexpected.
            raise FakeNotFoundError(f"FakeSandbox could not exec {command!r}: {exc!r}") from exc
        self._stdout_buf: str | None = None
        self._stderr_buf: str | None = None
        self._terminated_by_caller = False
        self._timed_out = False

        if timeout > 0:
            self._timer = threading.Timer(timeout, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()

    def _on_timeout(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._timed_out = True
            try:
                self._proc.send_signal(signal.SIGTERM)
            except Exception:  # noqa: BLE001 - process race
                pass

    def _drain_if_done(self) -> None:
        """Capture stdout/stderr the first time the process is terminal."""
        if self._proc is None:
            return
        if self._proc.poll() is None:
            return
        if self._stdout_buf is not None:
            return
        try:
            stdout, stderr = self._proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            self._proc.kill()
            stdout, stderr = self._proc.communicate()
        self._stdout_buf = stdout or ""
        self._stderr_buf = stderr or ""

    def poll(self) -> int | None:
        if self._proc is None:
            return -1
        rc = self._proc.poll()
        if rc is not None:
            self._drain_if_done()
        return rc

    def wait(self) -> int:
        if self._proc is None:
            return -1
        rc = self._proc.wait()
        self._drain_if_done()
        return rc

    def terminate(self, *, wait: bool = False) -> int | None:
        if self._proc is None:
            return None
        if self._proc.poll() is None:
            self._terminated_by_caller = True
            try:
                self._proc.send_signal(signal.SIGTERM)
            except Exception:  # noqa: BLE001 - process race
                pass
            # Wait briefly so a follow-up ``poll`` can return the exit
            # code (matches Modal's eventual-consistency contract: the
            # next ``status`` call will see the terminal state).
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._drain_if_done()
        if self._timer is not None:
            self._timer.cancel()
        return self._proc.returncode

    @property
    def returncode(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.returncode

    @property
    def stdout(self) -> _FakeStreamReader:
        # The first read after termination should drain the captured
        # buffer; before termination, return empty (matches Modal's
        # behavior of buffering until the sandbox is done).
        self._drain_if_done()
        return _FakeStreamReader(self._stdout_buf or "")

    @property
    def stderr(self) -> _FakeStreamReader:
        self._drain_if_done()
        return _FakeStreamReader(self._stderr_buf or "")

    def __del__(self) -> None:  # pragma: no cover - cleanup safety
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001 - process race
                pass
        if self._timer is not None:
            self._timer.cancel()


class _SandboxFactory:
    """Stand-in for :class:`modal.Sandbox` (the class itself).

    ``modal.Sandbox.create(*command, **kwargs)`` creates a Sandbox;
    ``modal.Sandbox.from_id(id)`` reconnects to one. Both are class
    methods on the production Sandbox; we mirror that with a class
    that exposes them as plain callables.
    """

    def __init__(self, fake_modal: FakeModal) -> None:
        self._fake = fake_modal

    def create(
        self,
        *command: str,
        app: Any = None,
        image: Any = None,
        env: dict[str, str] | None = None,
        gpu: str | None = None,
        timeout: int = 300,
        cpu: float | None = None,
        memory: int | None = None,
        **_extra: Any,
    ) -> FakeSandbox:
        # Surface the bad-image failure here (eager image-validation
        # path per ``07`` §7.4): if the image looks bogus or has been
        # explicitly registered as bad, raise the Modal-equivalent
        # NotFoundError. The plugin's ``submit`` translates this to
        # ``JobImageInvalid``. ``.invalid`` is a reserved TLD per
        # RFC 2606 — any production image with that substring is
        # genuinely unreachable, so the heuristic is safe.
        image_tag = getattr(image, "tag", None)
        if isinstance(image_tag, str) and (
            image_tag in self._fake.bad_images or ".invalid" in image_tag
        ):
            raise FakeNotFoundError(
                f"unable to pull image {image_tag!r}: registry says no such image"
            )
        sandbox = FakeSandbox(list(command), env, timeout)
        self._fake._sandboxes[sandbox.object_id] = sandbox  # noqa: SLF001
        return sandbox

    def from_id(self, sandbox_id: str, client: Any = None) -> FakeSandbox:
        # Modal's real ``Sandbox.from_id`` validates the id shape
        # client-side and raises ``InvalidError`` for anything that
        # isn't a ``sb-...`` id — well before any round-trip. Mirror
        # that so the plugin's malformed-handle → ``JobNotFound``
        # translation is exercised by tests.
        if not sandbox_id.startswith("sb-"):
            raise FakeInvalidError(f"{sandbox_id!r} is not a valid Sandbox ID.")
        try:
            return self._fake._sandboxes[sandbox_id]  # noqa: SLF001
        except KeyError as exc:
            raise FakeNotFoundError(f"sandbox not found: {sandbox_id!r}") from exc


class _FakeImage:
    """Stand-in for :class:`modal.Image`."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _ImageFactory:
    """Stand-in for the ``modal.Image`` namespace."""

    @staticmethod
    def from_registry(tag: str, *_args: Any, **_kwargs: Any) -> _FakeImage:
        return _FakeImage(tag=tag)


class _FakeApp:
    """Stand-in for :class:`modal.App`."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.app_id = f"ap-fake-{uuid.uuid4().hex[:8]}"


class _AppFactory:
    """Stand-in for the ``modal.App`` namespace."""

    @staticmethod
    def lookup(name: str, *, create_if_missing: bool = False, **_kwargs: Any) -> _FakeApp:
        # ``create_if_missing=True`` is required by ``ModalCompute._ensure_app``;
        # the fake honors it implicitly by always returning a fresh App.
        del create_if_missing  # always-create semantics in the fake
        return _FakeApp(name=name)


class _Exception:
    """Stand-in for the ``modal.exception`` namespace."""

    NotFoundError = FakeNotFoundError
    InvalidError = FakeInvalidError


class _FakeFileEntryType:
    """Stand-in for the ``modal.volume.FileEntryType`` enum values.

    Mirrors the string-surface check :func:`_entry_is_dir` performs —
    the production enum's ``str(value)`` yields ``"FileEntryType.FILE"``
    / ``"FileEntryType.DIRECTORY"``.
    """

    FILE = "FileEntryType.FILE"
    DIRECTORY = "FileEntryType.DIRECTORY"


class _FakeFileEntry:
    """Stand-in for ``modal.volume.FileEntry``."""

    def __init__(self, path: str, *, is_dir: bool) -> None:
        self.path = path
        self.type = _FakeFileEntryType.DIRECTORY if is_dir else _FakeFileEntryType.FILE


class _FakeBatchUpload:
    """Stand-in for ``Volume.batch_upload()``'s context manager.

    Mirrors the production ``put_directory(local, remote)`` API: walks
    the local directory and stores files in the parent fake volume's
    in-memory map under their relative remote path.
    """

    def __init__(self, volume: _FakeVolume) -> None:
        self._volume = volume

    def __enter__(self) -> _FakeBatchUpload:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type, exc, tb

    def put_directory(self, local_dir: str, remote_dir: str) -> None:
        import os  # noqa: PLC0415
        import pathlib  # noqa: PLC0415

        base = pathlib.Path(local_dir)
        if not base.is_dir():
            raise FakeNotFoundError(f"local directory not found: {local_dir!r}")
        remote_root = remote_dir.rstrip("/") or "/"
        for root, _dirs, files in os.walk(base):
            root_path = pathlib.Path(root)
            for fname in files:
                local_file = root_path / fname
                rel = local_file.relative_to(base)
                rel_posix = str(rel).replace(os.sep, "/")
                if remote_root == "/":
                    remote_path = "/" + rel_posix
                else:
                    remote_path = remote_root + "/" + rel_posix
                self._volume._files[remote_path] = local_file.read_bytes()  # noqa: SLF001


class _FakeVolume:
    """Stand-in for :class:`modal.Volume`.

    Stores file contents in an in-memory ``dict[str, bytes]`` keyed by
    absolute volume path. Supports the four surfaces
    :class:`ModalCompute.stage_workspace` /
    :meth:`harvest_workspace` actually call: ``batch_upload``,
    ``iterdir``, ``read_file``, plus reattachment via
    :meth:`_VolumeFactory.from_name`.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._files: dict[str, bytes] = {}

    def batch_upload(self) -> _FakeBatchUpload:
        return _FakeBatchUpload(self)

    def iterdir(self, remote_path: str) -> list[_FakeFileEntry]:
        prefix = remote_path.rstrip("/") or "/"
        # Compute the children of ``prefix``. Children are files whose
        # parent path is exactly ``prefix``, plus implicit directories
        # synthesized from any descendant file's path components.
        seen_dirs: set[str] = set()
        files_here: list[str] = []
        for path in self._files:
            parent = path.rsplit("/", 1)[0] or "/"
            if parent == prefix:
                files_here.append(path)
                continue
            # Walk up the chain to find a synthesized directory child.
            cursor = parent
            while cursor and cursor != prefix:
                next_parent = cursor.rsplit("/", 1)[0] or "/"
                if next_parent == prefix:
                    seen_dirs.add(cursor)
                    break
                cursor = next_parent
        entries: list[_FakeFileEntry] = []
        for d in sorted(seen_dirs):
            entries.append(_FakeFileEntry(d, is_dir=True))
        for f in sorted(files_here):
            entries.append(_FakeFileEntry(f, is_dir=False))
        return entries

    def read_file(self, remote_path: str) -> list[bytes]:
        if remote_path not in self._files:
            raise FakeNotFoundError(f"no such file in volume: {remote_path!r}")
        return [self._files[remote_path]]


class _VolumeFactory:
    """Stand-in for the ``modal.Volume`` namespace.

    Volumes registered here persist across :meth:`from_name` calls so
    a fresh ``ModalCompute`` instance can reattach to a previously
    staged volume — the production reattachment contract for workspace
    harvest from a fresh worker.
    """

    _volumes: dict[str, _FakeVolume]

    def __init__(self) -> None:
        self._volumes = {}

    def from_name(
        self, name: str, *, create_if_missing: bool = False, **_kwargs: Any
    ) -> _FakeVolume:
        if name not in self._volumes:
            if not create_if_missing:
                raise FakeNotFoundError(f"no such volume: {name!r}")
            self._volumes[name] = _FakeVolume(name)
        return self._volumes[name]


class FakeModal:
    """Top-level stand-in for the ``modal`` module.

    The plugin imports ``modal`` and reaches for ``modal.App``,
    ``modal.Image``, ``modal.Sandbox`` only — those are the namespaces
    we mirror. Tests construct a fresh :class:`FakeModal` per test (no
    cross-test state) and pass it via ``ModalCompute(modal_module=fake)``.

    ``bad_images`` is the seam by which the conformance suite's
    ``test_invalid_image_raises`` queues an image-pull failure for a
    known-bad tag. Production :class:`ModalCompute` does NOT see this
    attribute — it only consumes the standard module surface.
    """

    def __init__(self) -> None:
        self.App = _AppFactory()
        self.Image = _ImageFactory()
        self.Sandbox = _SandboxFactory(self)
        self.Volume = _VolumeFactory()
        self.exception = _Exception()
        self.bad_images: set[str] = set()
        self._sandboxes: dict[str, FakeSandbox] = {}


def make_fake_with_bad_image(bad_image: str) -> FakeModal:
    """Convenience factory used by ``test_invalid_image_raises``."""
    fake = FakeModal()
    fake.bad_images.add(bad_image)
    return fake


__all__ = [
    "FakeInvalidError",
    "FakeModal",
    "FakeNotFoundError",
    "FakeSandbox",
    "make_fake_with_bad_image",
]
