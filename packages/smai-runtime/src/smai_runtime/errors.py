"""Runtime-mechanical error taxonomy.

Per ``10-runtime-and-templates.md`` §6.2 (technique-output contract failures)
and §7.4 (no-go-zone hash failures). Errors are JSON-serializable so the
orchestrator's run-completion logic can route the entry back to the
implementer agent with the failure as part of the retry context (the same
retry-with-context pattern v1 uses for code-review feedback per DEC-023).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypeAlias

TechniqueOutputErrorCode: TypeAlias = Literal[
    "apply_raised",
    "missing_required_key",
    "unknown_key",
    "type_signature_mismatch",
]
"""§6.2 error taxonomy."""

NoGoZoneErrorCode: TypeAlias = Literal["file_missing", "hash_mismatch"]
"""§7.4 error taxonomy."""

CONTRACT_ERROR_FILENAME: str = "contract_error.json"
"""§6.2 — runtime writes to this stable workspace path on technique-output failure."""

NO_GO_ZONE_ERROR_FILENAME: str = "no_go_zone_error.json"
"""§7.4 — runtime writes to this stable workspace path on hash-check failure."""


class TechniqueOutputContractError(Exception):
    """Raised when ``apply()`` raised, omitted a required key, returned an
    unknown key, or produced a value that fails the structural type-check
    against the manifest's declared signature (§6.2)."""

    def __init__(
        self,
        *,
        code: TechniqueOutputErrorCode,
        message: str,
        extension_point_key: str | None,
        expected_signature: str | None,
        actual_value_repr: str | None,
        manifest_hash: str = "",
        technique_id: str | None = None,
        entry_id: str = "",
    ) -> None:
        super().__init__(message)
        self.code: TechniqueOutputErrorCode = code
        self.message: str = message
        self.extension_point_key: str | None = extension_point_key
        self.expected_signature: str | None = expected_signature
        self.actual_value_repr: str | None = actual_value_repr
        self.manifest_hash: str = manifest_hash
        self.technique_id: str | None = technique_id
        self.entry_id: str = entry_id

    def with_identity(
        self,
        *,
        manifest_hash: str,
        technique_id: str | None,
        entry_id: str,
    ) -> TechniqueOutputContractError:
        """Return a copy carrying the run's identity fields (§6.2)."""
        return TechniqueOutputContractError(
            code=self.code,
            message=self.message,
            extension_point_key=self.extension_point_key,
            expected_signature=self.expected_signature,
            actual_value_repr=self.actual_value_repr,
            manifest_hash=manifest_hash,
            technique_id=technique_id,
            entry_id=entry_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "extension_point_key": self.extension_point_key,
            "expected_signature": self.expected_signature,
            "actual_value_repr": self.actual_value_repr,
            "manifest_hash": self.manifest_hash,
            "technique_id": self.technique_id,
            "entry_id": self.entry_id,
        }


class NoGoZoneHashError(Exception):
    """Raised when a fixed-template no-go-zone file is missing or modified (§7.4)."""

    def __init__(
        self,
        *,
        code: NoGoZoneErrorCode,
        path: str,
        expected_hash: str | None,
        actual_hash: str | None,
        runtime_template_version: str,
    ) -> None:
        if code == "file_missing":
            super().__init__(f"no-go-zone file missing: {path}")
        else:
            super().__init__(
                f"no-go-zone hash mismatch at {path}: expected={expected_hash} actual={actual_hash}"
            )
        self.code: NoGoZoneErrorCode = code
        self.path: str = path
        self.expected_hash: str | None = expected_hash
        self.actual_hash: str | None = actual_hash
        self.runtime_template_version: str = runtime_template_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "runtime_template_version": self.runtime_template_version,
        }


class MetricsContractError(Exception):
    """Raised when ``metrics.json`` is missing a required-metric key (§10.1)."""

    def __init__(
        self,
        *,
        missing_keys: list[str],
        present_keys: list[str],
    ) -> None:
        super().__init__(f"metrics.json missing required keys: {sorted(missing_keys)}")
        self.missing_keys: list[str] = missing_keys
        self.present_keys: list[str] = present_keys

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_keys": sorted(self.missing_keys),
            "present_keys": sorted(self.present_keys),
        }


def write_contract_error(workspace: Path, err: TechniqueOutputContractError) -> Path:
    """Serialize ``err`` to ``workspace/contract_error.json`` (§6.2)."""
    target = workspace / CONTRACT_ERROR_FILENAME
    target.write_text(json.dumps(err.to_dict(), sort_keys=True, indent=2))
    return target


def write_no_go_zone_error(workspace: Path, err: NoGoZoneHashError) -> Path:
    """Serialize ``err`` to ``workspace/no_go_zone_error.json`` (§7.4)."""
    target = workspace / NO_GO_ZONE_ERROR_FILENAME
    target.write_text(json.dumps(err.to_dict(), sort_keys=True, indent=2))
    return target
