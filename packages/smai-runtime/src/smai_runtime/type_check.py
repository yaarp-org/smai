"""Manifest-driven runtime type check on technique ``apply()`` output.

Per ``10-runtime-and-templates.md`` §6. Structural-only assertions against the
manifest's declared ``type_signature`` strings. Intentionally conservative
(callable / list / dict / class allowlist with degrade-to-True fallback);
finer correctness is agent-validated by code review and the validation run
(§6.1).
"""

from __future__ import annotations

import re
from typing import Any

from smai_runtime.errors import TechniqueOutputContractError
from smai_runtime.manifest import HarnessAPIManifest, HarnessExtensionPoint

# Closed allowlist for class-name resolution (§6.1, open question 7). Maps
# the bare type name in a manifest type_signature to a tuple of qualified
# import paths to try; an unknown name degrades to ``True`` per §6.1's rule.
_CLASS_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "nn.Module": ("torch.nn.Module",),
    "Module": ("torch.nn.Module",),
    "Tensor": ("torch.Tensor",),
    "Dataset": ("torch.utils.data.Dataset",),
    "DataLoader": ("torch.utils.data.DataLoader",),
    "ndarray": ("numpy.ndarray",),
}

_PRIMITIVE_NAMES: frozenset[str] = frozenset(
    {"int", "float", "bool", "str", "bytes", "Any", "object", "None"}
)


def _normalize_signature(sig: str) -> str:
    """Strip whitespace once for matcher convenience.

    The manifest's canonicalization rule (§5.3 #2) preserves the harness
    builder's emitted whitespace; the structural matcher tolerates it.
    """
    return sig.strip()


def _match_callable(sig: str) -> bool:
    return _normalize_signature(sig).startswith("Callable")


_LIST_RE = re.compile(r"^list\s*\[(?P<inner>.+)\]$", re.DOTALL)
_DICT_RE = re.compile(r"^dict\s*\[(?P<key>[^,]+),\s*(?P<value>.+)\]$", re.DOTALL)


def _match_list(sig: str) -> str | None:
    m = _LIST_RE.match(_normalize_signature(sig))
    return m.group("inner").strip() if m else None


def _match_dict(sig: str) -> tuple[str, str] | None:
    m = _DICT_RE.match(_normalize_signature(sig))
    if m is None:
        return None
    return m.group("key").strip(), m.group("value").strip()


def _resolve_class(name: str) -> type | None:
    targets = _CLASS_ALLOWLIST.get(name)
    if targets is None:
        return None
    for target in targets:
        module_name, _, attr = target.rpartition(".")
        try:
            module = __import__(module_name, fromlist=[attr])
        except ImportError:
            continue
        cls = getattr(module, attr, None)
        if isinstance(cls, type):
            return cls
    return None


def _check_value(value: Any, signature: str) -> bool:
    """Structural check against a single ``type_signature`` string (§6.1).

    Closed set of shapes; unknown shapes degrade to ``True``.
    """
    sig = _normalize_signature(signature)

    if _match_callable(sig):
        return callable(value)

    inner = _match_list(sig)
    if inner is not None:
        if not isinstance(value, list):
            return False
        return all(_check_value(item, inner) for item in value)  # type: ignore[arg-type]

    pair = _match_dict(sig)
    if pair is not None:
        key_sig, value_sig = pair
        if not isinstance(value, dict):
            return False
        if _normalize_signature(key_sig) != "str":
            return True  # non-str-keyed dicts not in v1; degrade.
        return all(  # type: ignore[arg-type]
            isinstance(k, str) and _check_value(v, value_sig) for k, v in value.items()
        )

    if sig in _PRIMITIVE_NAMES:
        if sig == "Any" or sig == "object":
            return True
        if sig == "None":
            return value is None
        primitives: dict[str, type | tuple[type, ...]] = {
            "int": int,
            "float": (float, int),
            "bool": bool,
            "str": str,
            "bytes": bytes,
        }
        return isinstance(value, primitives[sig])

    cls = _resolve_class(sig)
    if cls is not None:
        return isinstance(value, cls)

    return True


def check_technique_output(
    technique_output: dict[str, Any],
    manifest: HarnessAPIManifest,
) -> None:
    """Raise :class:`TechniqueOutputContractError` on contract violations (§6.1).

    Three failure modes:
      1. ``optional=False`` extension-point key absent → ``missing_required_key``
      2. Returned key not declared in the manifest → ``unknown_key``
      3. Value at a declared key fails structural type check → ``type_signature_mismatch``
    """
    declared_keys: dict[str, HarnessExtensionPoint] = {
        ep.key: ep for ep in manifest.extension_points
    }

    # 2. Unknown keys first — this catches typos before false-positive
    # signature errors derived from a wrong-key cascade.
    for returned_key in technique_output:
        if returned_key not in declared_keys:
            raise TechniqueOutputContractError(
                code="unknown_key",
                message=(
                    f"technique returned key {returned_key!r} which is not "
                    f"declared on the harness API manifest"
                ),
                extension_point_key=returned_key,
                expected_signature=None,
                actual_value_repr=_safe_repr(technique_output[returned_key]),
                manifest_hash=manifest.content_hash,
            )

    # 1. Required-key absence.
    for key, ep in declared_keys.items():
        if not ep.optional and key not in technique_output:
            raise TechniqueOutputContractError(
                code="missing_required_key",
                message=(
                    f"manifest declares extension point {key!r} as required "
                    f"(optional=False) but technique did not return it"
                ),
                extension_point_key=key,
                expected_signature=ep.type_signature,
                actual_value_repr=None,
                manifest_hash=manifest.content_hash,
            )

    # 3. Per-key structural type check.
    for key, ep in declared_keys.items():
        if key not in technique_output:
            continue
        value: Any = technique_output[key]
        if not _check_value(value, ep.type_signature):
            raise TechniqueOutputContractError(
                code="type_signature_mismatch",
                message=(
                    f"technique value at {key!r} fails structural check "
                    f"against signature {ep.type_signature!r}"
                ),
                extension_point_key=key,
                expected_signature=ep.type_signature,
                actual_value_repr=_safe_repr(value),
                manifest_hash=manifest.content_hash,
            )


def _safe_repr(value: Any) -> str:
    """Bounded repr for error payloads — caps runaway sizes (§6.2)."""
    try:
        text = repr(value)
    except Exception as exc:  # pragma: no cover - defensive
        return f"<unrepr-able: {type(exc).__name__}>"
    return text if len(text) <= 200 else text[:197] + "..."
