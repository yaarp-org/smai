"""Dependency-allowlist lint.

Mechanically enforces the methodology-atomic / pipeline-composable boundary at the
package level. See DEC-029 and ``designs/smai/00-vision.md`` §4 principle #2.

Three rules:
    1. ``smai-core``'s declared runtime dependencies stay on the allowlist
       (``pydantic``, ``jsonschema``, plus the standard library — which is not
       declared).
    2. No ``.py`` file under ``packages/smai-core/src/`` imports a pipeline
       package (``smai_inline_agents``, ``smai_agent_runtime``, ``smai_orchestrator``,
       ``smai_runtime``, ``smai_cli``, or the ``smai`` umbrella) or any plugin
       package (``smai_llm_*``, ``smai_store_*``, ``smai_artifacts_*``,
       ``smai_compute_*``).
    3. No ``plugins/smai-*`` package declares a dependency on a pipeline package,
       and no ``.py`` file under its ``src/`` imports one.

``smai_agent_runtime`` is sandbox-side per DEC-038 (proposed): the package is
consumed via container image, not by the host process. It carries its own
dep policy (PydanticAI + provider SDKs) distinct from the methodology-layer
allowlist. Treating it as a pipeline package here is the cheapest way to
enforce "no host code imports it" — if a plugin or smai-core source ever
tries ``import smai_agent_runtime``, the lint catches it.

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

# Module names (i.e. importable identifiers, with underscores) for the
# pipeline packages plus the ``smai`` umbrella (which depends on all of them
# and re-exports the Tier A surface — importing it transitively breaks
# atomicity the same way importing any of the four directly would).
# Forbidden in both smai-core and plugin sources, EXCEPT smai-orchestrator
# in plugin sources (see ``PLUGIN_PIPELINE_ALLOWED`` below).
#
# ``smai_agent_runtime`` is included so the lint mechanically enforces
# "consumed via image, not via host process" — no plugin or smai-core
# source may import it (per DEC-038, the sandbox-side runtime packages
# DEC noted in agent_refactor/compute_dispatch_decisions.md §6). Step 8
# Wave 1 split the old ``smai_agents`` into the host-side
# ``smai_inline_agents`` and the sandbox-side ``smai_agent_runtime``;
# Wave 2 relocated the residual sandboxed-dispatcher wirings into
# ``smai_orchestrator.sandboxed_dispatch`` and deleted ``smai_agents``.
PIPELINE_MODULES: frozenset[str] = frozenset(
    {
        "smai",
        "smai_agent_runtime",
        "smai_inline_agents",
        "smai_orchestrator",
        "smai_runtime",
        "smai_cli",
    }
)

# Distribution names (i.e. PyPI / pyproject names, with hyphens) for the
# pipeline packages plus the ``smai`` umbrella. Used when scanning
# ``[project] dependencies``.
PIPELINE_DISTS: frozenset[str] = frozenset(
    {
        "smai",
        "smai-agent-runtime",
        "smai-inline-agents",
        "smai-orchestrator",
        "smai-runtime",
        "smai-cli",
    }
)

# Pipeline packages that plugins ARE allowed to depend on / import. Per
# Task 2.A2's MetadataStore-plugin work (DEC-036): the pipeline-tracking
# record types (``ComparisonGroupRecord`` etc.) live in
# ``smai-orchestrator.entities.tracking`` per ``01-data-model.md`` §5.1
# / DEC-029, and ``MetadataStore`` plugins MUST construct + return these
# at runtime (every ``create_*`` / ``get_*`` / ``transition_*`` method
# round-trips a record). Forbidding the import would force plugins to
# either re-declare the record types (drift risk) or push the
# round-tripping responsibility back into smai-core (would break
# DEC-029's "smai-core IS methodology, plugins are the seam" line).
#
# The allowance is narrow: only ``smai-orchestrator`` (which carries
# record types). The other pipeline packages — ``smai-inline-agents``,
# ``smai-runtime``, ``smai-cli``, the ``smai`` umbrella — remain
# forbidden because depending on them would couple the plugin to the
# agent-loop / runtime / CLI surfaces it is meant to plug behind.
#
# Same hygiene applies bidirectionally: smai-orchestrator does NOT
# depend on plugins (per its ``pyproject.toml``), so the dependency
# graph stays acyclic.
PLUGIN_PIPELINE_ALLOWED_DISTS: frozenset[str] = frozenset({"smai-orchestrator"})
PLUGIN_PIPELINE_ALLOWED_MODULES: frozenset[str] = frozenset({"smai_orchestrator"})

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
    "smai-core is the methodology layer (DEC-029); runtime imports of "
    "pipeline packages (smai_inline_agents, smai_agent_runtime, smai_orchestrator, "
    "smai_runtime, smai_cli, the smai umbrella, or any plugin) break the "
    "package-boundary atomicity invariant (00-vision.md §4 principle #2). "
    "Pure typing references are allowed under `if TYPE_CHECKING:` (per "
    "`01-data-model.md` §5.1 / `07-plugin-interfaces.md` §3.1: the "
    "MetadataStore Protocol type-references pipeline-tracking record types "
    "from smai-orchestrator)."
)
REASON_PLUGIN_DEP = (
    "Plugin packages must depend only on smai-core and their own provider SDK "
    "(implementation_plan.md §2.3); pulling in pipeline packages "
    "(smai-inline-agents, smai-agent-runtime, smai-orchestrator, smai-runtime, "
    "smai-cli, or the smai umbrella) bloats the plugin install footprint "
    "and creates accidental cross-package coupling."
)
REASON_PLUGIN_IMPORT = (
    "Plugin packages must not import pipeline packages "
    "(smai_inline_agents, smai_agent_runtime, smai_orchestrator, smai_runtime, "
    "smai_cli, or the smai umbrella) at runtime per implementation_plan.md "
    "§2.3; doing so couples the plugin to the pipeline layer it is meant "
    "to plug into. Pure typing references are allowed under "
    "`if TYPE_CHECKING:`."
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


def _is_type_checking_test(test: ast.expr) -> bool:
    """Return True if the ``If.test`` expression is ``TYPE_CHECKING`` /
    ``typing.TYPE_CHECKING`` (in any module-prefixed form).

    Recognized shapes:

    * ``if TYPE_CHECKING:`` (after ``from typing import TYPE_CHECKING``)
    * ``if typing.TYPE_CHECKING:`` (after ``import typing``)
    * ``if t.TYPE_CHECKING:`` (after ``import typing as t``)

    Explicitly NOT recognized — these are runtime branches and remain
    subject to the import boundary:

    * ``if not TYPE_CHECKING:`` — inverse; imports inside this block run
      everywhere except a static checker.
    * The ``else:`` branch of an ``if TYPE_CHECKING:`` ``If`` — that is
      the runtime fallback.
    """
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


class _ImportCollector(ast.NodeVisitor):
    """AST visitor that yields ``(lineno, top_module, snippet, in_type_checking)``
    for every ``Import`` and ``ImportFrom`` node, tracking whether each
    import is nested inside an ``if TYPE_CHECKING:`` block (whose
    ``orelse`` branch is correctly *not* counted as type-checking).

    A flat ``ast.walk`` cannot do this because it loses the parent-chain
    context. We descend the tree explicitly, flipping a flag in the
    body of ``if TYPE_CHECKING:`` ``If`` nodes only.
    """

    def __init__(self, source_lines: list[str]) -> None:
        self._lines = source_lines
        self._in_type_checking = False
        self.results: list[tuple[int, str, str, bool]] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(node.test):
            saved = self._in_type_checking
            self._in_type_checking = True
            for child in node.body:
                self.visit(child)
            self._in_type_checking = saved
            # ``orelse`` is the runtime fallback (``else:`` branch of the
            # TYPE_CHECKING ``If``) — visit it WITHOUT the flag set.
            for child in node.orelse:
                self.visit(child)
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        snippet = self._lines[node.lineno - 1].strip() if node.lineno - 1 < len(self._lines) else ""
        for alias in node.names:
            top = alias.name.split(".", 1)[0]
            self.results.append((node.lineno, top, snippet, self._in_type_checking))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            return  # relative import — no top-level module to record
        if node.module is None:
            return
        top = node.module.split(".", 1)[0]
        snippet = self._lines[node.lineno - 1].strip() if node.lineno - 1 < len(self._lines) else ""
        self.results.append((node.lineno, top, snippet, self._in_type_checking))


def iter_top_level_imports(py_file: Path) -> Iterator[tuple[int, str, str, bool]]:
    """Yield ``(lineno, top_level_module, source_snippet, in_type_checking)``
    for each import.

    Top-level module = the first segment of the dotted module path. Relative
    imports (``from . import x``) yield no result — they target the current
    package.

    ``in_type_checking`` is ``True`` when the import is nested inside an
    ``if TYPE_CHECKING:`` block (per `01-data-model.md` §5.1 /
    `07-plugin-interfaces.md` §3.1: imports made solely for typing
    purposes are exempt from the methodology / pipeline boundary). The
    rule-evaluation site is responsible for honoring the flag.
    """
    source = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{py_file}: failed to parse: {exc}") from exc
    collector = _ImportCollector(source.splitlines())
    collector.visit(tree)
    yield from collector.results


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


# Subpath under ``packages/smai-core/src/`` that is exempt from rule 2.
# The conformance test suite (``smai_core.plugins.conformance``) is opt-in
# test infrastructure gated behind the ``[conformance]`` extra (per
# ``smai-core/pyproject.toml``); it intentionally crosses the methodology
# / pipeline boundary because it tests the contract between plugin
# implementations and the pipeline-tracking record types they accept and
# return. Plugin authors who subclass these bases install smai-orchestrator
# anyway. Tier B users never reach this subpackage — its ``__init__.py``
# requires pytest, also opt-in.
_RULE2_EXEMPT_SUBPATHS: frozenset[str] = frozenset(
    {"smai_core/plugins/conformance"},
)


def _is_rule2_exempt_path(py_file: Path, src_root: Path) -> bool:
    rel = py_file.relative_to(src_root).as_posix()
    return any(rel.startswith(prefix + "/") for prefix in _RULE2_EXEMPT_SUBPATHS)


def check_smai_core_imports(root: Path) -> list[Violation]:
    """Rule 2: smai-core sources don't import pipeline or plugin packages
    at runtime.

    Two exemptions apply:

    * Imports nested inside ``if TYPE_CHECKING:`` blocks are exempted (per
      `01-data-model.md` §5.1 / `07-plugin-interfaces.md` §3.1: the
      ``MetadataStore`` Protocol type-references record types from
      ``smai_orchestrator.entities.tracking``). Imports in ``not
      TYPE_CHECKING`` branches or in the ``else:`` of a TYPE_CHECKING
      ``If`` remain subject to the rule.
    * Sources under :data:`_RULE2_EXEMPT_SUBPATHS` are exempted entirely
      — opt-in test infrastructure that intentionally crosses the
      package boundary.
    """
    src = root / "packages" / "smai-core" / "src"
    if not src.exists():
        return []
    violations: list[Violation] = []
    for py_file in iter_python_files(src):
        if _is_rule2_exempt_path(py_file, src):
            continue
        for lineno, module, snippet, in_type_checking in iter_top_level_imports(py_file):
            if in_type_checking:
                continue
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
    """Rule 3: plugins don't pull in pipeline packages (deps or imports).

    Per Task 2.A2 (DEC-036): ``smai-orchestrator`` is exempt because
    that's where the pipeline-tracking record types live, and every
    ``MetadataStore`` plugin needs to construct + return them at
    runtime. The other pipeline packages remain forbidden — see
    :data:`PLUGIN_PIPELINE_ALLOWED_DISTS` for the full reasoning.
    """
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
                if name in PIPELINE_DISTS and name not in PLUGIN_PIPELINE_ALLOWED_DISTS:
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
            for lineno, module, snippet, in_type_checking in iter_top_level_imports(py_file):
                if in_type_checking:
                    continue
                if is_pipeline_module(module) and module not in PLUGIN_PIPELINE_ALLOWED_MODULES:
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
