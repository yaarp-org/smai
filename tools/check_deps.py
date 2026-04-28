"""Dependency-allowlist lint.

Mechanically enforces the methodology-atomic / pipeline-composable boundary at the
package level. See DEC-029 and ``designs/smai/00-vision.md`` §4 principle #2.

Three rules:
    1. ``smai-core``'s declared runtime dependencies stay on the allowlist
       (``pydantic``, ``jsonschema``, plus the standard library — which is not
       declared).
    2. No ``.py`` file under ``packages/smai-core/src/`` imports a pipeline
       package (``smai_agents``, ``smai_orchestrator``, ``smai_runtime``,
       ``smai_cli``, or the ``smai`` umbrella — which transitively pulls all
       four) or any plugin package (``smai_llm_*``, ``smai_store_*``,
       ``smai_artifacts_*``, ``smai_compute_*``).
    3. No ``plugins/smai-*`` package declares a dependency on a pipeline package,
       and no ``.py`` file under its ``src/`` imports one.

Usage::

    python tools/check_deps.py             # exit 0 on pass, 1 on fail
    python tools/check_deps.py --verbose   # also log what was checked
    python tools/check_deps.py --root PATH # check a different workspace root

This script imports nothing beyond the standard library so it can run before any
workspace dependency is installed.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# ---------- constants ---------------------------------------------------------

# Module names (i.e. importable identifiers, with underscores) for the four
# pipeline packages plus the ``smai`` umbrella (which depends on all of them
# and re-exports the Tier A surface — importing it transitively breaks
# atomicity the same way importing any of the four directly would).
# Forbidden in both smai-core and plugin sources.
PIPELINE_MODULES: frozenset[str] = frozenset(
    {"smai", "smai_agents", "smai_orchestrator", "smai_runtime", "smai_cli"}
)

# Distribution names (i.e. PyPI / pyproject names, with hyphens) for the four
# pipeline packages plus the ``smai`` umbrella. Used when scanning
# ``[project] dependencies``.
PIPELINE_DISTS: frozenset[str] = frozenset(
    {"smai", "smai-agents", "smai-orchestrator", "smai-runtime", "smai-cli"}
)

# Module-name prefixes for plugin packages. The four interface namespaces in
# §2.4 of the implementation plan are llm_providers / metadata_stores /
# artifact_stores / computes; the corresponding module prefixes are these four.
PLUGIN_MODULE_PREFIXES: tuple[str, ...] = (
    "smai_llm_",
    "smai_store_",
    "smai_artifacts_",
    "smai_compute_",
)

# Allowed runtime deps for smai-core. PEP 503-normalized.
SMAI_CORE_ALLOWED_DEPS: frozenset[str] = frozenset({"pydantic", "jsonschema"})

# Reasons surfaced in violation messages.
REASON_CORE_DEP = (
    "smai-core is the methodology layer (DEC-029); only Pydantic and a JSON "
    "Schema validator are allowed as runtime deps. Adding others breaks the "
    "package-boundary atomicity invariant (00-vision.md §4 principle #2)."
)
REASON_CORE_IMPORT = (
    "smai-core is the methodology layer (DEC-029); imports of pipeline "
    "packages (smai_agents, smai_orchestrator, smai_runtime, smai_cli, "
    "the smai umbrella, or any plugin) break the package-boundary "
    "atomicity invariant (00-vision.md §4 principle #2)."
)
REASON_PLUGIN_DEP = (
    "Plugin packages must depend only on smai-core and their own provider SDK "
    "(implementation_plan.md §2.3); pulling in pipeline packages "
    "(smai-agents, smai-orchestrator, smai-runtime, smai-cli, or the smai "
    "umbrella) bloats the plugin install footprint and creates accidental "
    "cross-package coupling."
)
REASON_PLUGIN_IMPORT = (
    "Plugin packages must not import pipeline packages "
    "(smai_agents, smai_orchestrator, smai_runtime, smai_cli, or the smai "
    "umbrella) per implementation_plan.md §2.3; doing so couples the plugin "
    "to the pipeline layer it is meant to plug into."
)

# ---------- data types --------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One rule failure."""

    rule: str
    location: str
    detail: str
    reason: str

    def render(self) -> str:
        return f"FAIL [{self.rule}]: {self.detail}\n  {self.location}\n  Reason: {self.reason}"


# ---------- helpers -----------------------------------------------------------


_DEP_TERMINATORS = re.compile(r"[\[\(<>=~!;\s]")
_NORMALIZE_SEP = re.compile(r"[-_.]+")


def normalize_name(name: str) -> str:
    """PEP 503-normalized package name (lowercase, ``-`` separators)."""
    return _NORMALIZE_SEP.sub("-", name.strip().lower())


def parse_dep_name(spec: str) -> str:
    """Extract the base distribution name from a PEP 508 requirement string."""
    s = spec.strip()
    m = _DEP_TERMINATORS.search(s)
    if m is not None:
        s = s[: m.start()]
    return normalize_name(s)


def read_pyproject_dependencies(pyproject: Path) -> list[str]:
    """Return the raw strings in ``[project] dependencies`` (empty if absent)."""
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    if not isinstance(deps, list):
        raise ValueError(f"{pyproject}: [project] dependencies must be a list")
    return [str(d) for d in deps]


def iter_python_files(root: Path) -> Iterator[Path]:
    """Yield every ``*.py`` file under ``root``, skipping ``__pycache__``."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def iter_top_level_imports(py_file: Path) -> Iterator[tuple[int, str, str]]:
    """Yield ``(lineno, top_level_module, source_snippet)`` for each import.

    Top-level module = the first segment of the dotted module path. Relative
    imports (``from . import x``) yield no result — they target the current
    package.
    """
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{py_file}: failed to parse: {exc}") from exc
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                snippet = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
                yield (node.lineno, top, snippet)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import
            if node.module is None:
                continue
            top = node.module.split(".", 1)[0]
            snippet = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) else ""
            yield (node.lineno, top, snippet)


def is_plugin_module(module: str) -> bool:
    return any(module.startswith(prefix) for prefix in PLUGIN_MODULE_PREFIXES)


def is_pipeline_module(module: str) -> bool:
    return module in PIPELINE_MODULES


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------- rules -------------------------------------------------------------


def check_smai_core_deps(root: Path) -> list[Violation]:
    """Rule 1: smai-core's runtime deps stay on the allowlist."""
    pyproject = root / "packages" / "smai-core" / "pyproject.toml"
    if not pyproject.exists():
        return []
    violations: list[Violation] = []
    for dep_spec in read_pyproject_dependencies(pyproject):
        name = parse_dep_name(dep_spec)
        if not name:
            continue
        if name in SMAI_CORE_ALLOWED_DEPS:
            continue
        violations.append(
            Violation(
                rule="smai-core-deps",
                location=relpath(pyproject, root),
                detail=f"forbidden runtime dependency: {dep_spec!r}",
                reason=REASON_CORE_DEP,
            )
        )
    return violations


def check_smai_core_imports(root: Path) -> list[Violation]:
    """Rule 2: smai-core sources don't import pipeline or plugin packages."""
    src = root / "packages" / "smai-core" / "src"
    if not src.exists():
        return []
    violations: list[Violation] = []
    for py_file in iter_python_files(src):
        for lineno, module, snippet in iter_top_level_imports(py_file):
            if is_pipeline_module(module) or is_plugin_module(module):
                violations.append(
                    Violation(
                        rule="smai-core-imports",
                        location=f"{relpath(py_file, root)}:{lineno}",
                        detail=(f"smai-core imports forbidden package {module!r}\n  > {snippet}"),
                        reason=REASON_CORE_IMPORT,
                    )
                )
    return violations


def check_plugin_boundaries(root: Path) -> list[Violation]:
    """Rule 3: plugins don't pull in pipeline packages (deps or imports)."""
    plugins_dir = root / "plugins"
    if not plugins_dir.exists():
        return []
    violations: list[Violation] = []
    for plugin in sorted(plugins_dir.iterdir()):
        if not plugin.is_dir() or not plugin.name.startswith("smai-"):
            continue
        pyproject = plugin / "pyproject.toml"
        if pyproject.exists():
            for dep_spec in read_pyproject_dependencies(pyproject):
                name = parse_dep_name(dep_spec)
                if name in PIPELINE_DISTS:
                    violations.append(
                        Violation(
                            rule="plugin-deps",
                            location=relpath(pyproject, root),
                            detail=(
                                f"plugin {plugin.name!r} declares forbidden "
                                f"dependency: {dep_spec!r}"
                            ),
                            reason=REASON_PLUGIN_DEP,
                        )
                    )
        src = plugin / "src"
        for py_file in iter_python_files(src):
            for lineno, module, snippet in iter_top_level_imports(py_file):
                if is_pipeline_module(module):
                    violations.append(
                        Violation(
                            rule="plugin-imports",
                            location=f"{relpath(py_file, root)}:{lineno}",
                            detail=(
                                f"plugin {plugin.name!r} imports forbidden "
                                f"package {module!r}\n  > {snippet}"
                            ),
                            reason=REASON_PLUGIN_IMPORT,
                        )
                    )
    return violations


# ---------- entry point -------------------------------------------------------


def run_all_checks(root: Path) -> list[Violation]:
    """Run every rule against ``root``. Returns the combined violation list."""
    return (
        check_smai_core_deps(root) + check_smai_core_imports(root) + check_plugin_boundaries(root)
    )


def _count_files(root: Path) -> int:
    return sum(1 for _ in iter_python_files(root))


def _emit_verbose_report(root: Path, out: object) -> None:
    # Narrow type for write() — duck-typed file-like.
    write = getattr(out, "write")  # noqa: B009 - defensive duck-typing
    core_pyproject = root / "packages" / "smai-core" / "pyproject.toml"
    core_src = root / "packages" / "smai-core" / "src"
    plugins_dir = root / "plugins"
    plugin_count = (
        sum(1 for p in plugins_dir.iterdir() if p.is_dir() and p.name.startswith("smai-"))
        if plugins_dir.exists()
        else 0
    )
    write(f"check_deps OK (root={root})\n")
    write(f"  rule 1 (smai-core deps): {core_pyproject}\n")
    write(f"  rule 2 (smai-core imports): {_count_files(core_src)} .py files under {core_src}\n")
    write(f"  rule 3 (plugin boundaries): {plugin_count} plugins under {plugins_dir}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_deps",
        description=(
            "Lint the smai workspace against the methodology / pipeline "
            "dependency boundary (DEC-029)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root (default: current working directory).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Log what was checked even on success.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    root: Path = args.root.resolve()
    violations = run_all_checks(root)
    if violations:
        for v in violations:
            print(v.render(), file=sys.stderr)
        print(
            f"\n{len(violations)} violation(s); see DEC-029 / 00-vision.md §4 "
            f"principle #2 for context.",
            file=sys.stderr,
        )
        return 1
    if args.verbose:
        _emit_verbose_report(root, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
