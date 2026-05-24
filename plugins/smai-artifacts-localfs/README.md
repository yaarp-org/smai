# smai-artifacts-localfs

`ArtifactStore` plugin: local-filesystem reference implementation. The
default backing for `smai dev`. Stores artifact bytes under a configurable
root directory.

## Configuration

The single constructor key is `root` (a path or string):

```yaml
plugins:
  artifact_store: localfs
  artifact_store_config:
    root: ~/.smai/artifacts    # default
```

Resolution order: explicit `root` arg, then the `SMAI_ARTIFACTS_ROOT` env var,
then `~/.smai/artifacts`. `smai dev` and `smai ui` inject the path
automatically; an empty `artifact_store_config: {}` is fine under those verbs.

## Credentials

None required.

## Behavior

- `put` is atomic (temp file + `os.replace`); concurrent writers cannot
  leave a partially written value at a key.
- `delete` is idempotent on missing keys.
- `url_for` returns a `file://` URL for the absolute path; `expires_in`
  is accepted for Protocol parity but ignored.

## Tests

The conformance suite and local-fs tests run with no external dependencies:

```bash
uv run pytest plugins/smai-artifacts-localfs/
```
