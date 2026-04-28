# smai-artifacts-localfs

`ArtifactStore` plugin: local-filesystem reference implementation. The
default backing for `smai dev` — stores artifact bytes under a
configurable root directory.

## Configuration

- Pass `LocalFsStore(root=...)` to set the root explicitly.
- Otherwise, reads the `SMAI_ARTIFACTS_ROOT` env var.
- Falls back to `~/.smai/artifacts`.

## Behavior

- `put` is atomic (temp file + `os.replace`) — concurrent writers cannot
  leave a partially written value at a key.
- `delete` is idempotent on missing keys.
- `url_for` returns a `file://` URL for the absolute path; `expires_in`
  is accepted for Protocol parity but ignored.

See `designs/smai/07-plugin-interfaces.md` §6 (in the upstream Yaarp
repo) for the Protocol contract and `designs/smai/implementation_plan.md`
§3.3 Task 2.A3 for scope.
