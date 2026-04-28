"""Tests for ``tools/check_deps.py`` — the dependency-allowlist lint.

Each rule has a pass test (clean fixture passes) and a fail test (injected
violation is detected with a clear message). One end-to-end test runs the lint
against the real workspace.

Tests call the lint's functions directly against tempdir fixtures (option (b)
in the task spec) rather than shelling out, so failures point at exact lines.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the lint module by file path — ``tools/`` is not a package and is not on
# sys.path under the workspace's pyright/pytest config.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LINT_PATH = _REPO_ROOT / "tools" / "check_deps.py"


def _load_lint_module():
    spec = importlib.util.spec_from_file_location("check_deps", _LINT_PATH)
    assert spec is not None and spec.loader is not None, "failed to locate check_deps.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_deps"] = module
    spec.loader.exec_module(module)
    return module


check_deps = _load_lint_module()


# ---------- fixture helpers ---------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_smai_core(
    root: Path,
    *,
    deps: list[str] | None = None,
    sources: dict[str, str] | None = None,
) -> None:
    pyproject = (
        "[project]\n"
        'name = "smai-core"\n'
        'version = "0.1.0"\n'
        f"dependencies = {deps if deps is not None else []!r}\n"
    )
    _write(root / "packages" / "smai-core" / "pyproject.toml", pyproject)
    _write(
        root / "packages" / "smai-core" / "src" / "smai_core" / "__init__.py",
        '"""smai-core stub."""\n',
    )
    if sources:
        for relpath, body in sources.items():
            _write(root / "packages" / "smai-core" / "src" / "smai_core" / relpath, body)


def _make_plugin(
    root: Path,
    name: str,
    *,
    module: str | None = None,
    deps: list[str] | None = None,
    sources: dict[str, str] | None = None,
) -> None:
    module = module or name.replace("-", "_")
    pyproject = (
        "[project]\n"
        f'name = "{name}"\n'
        'version = "0.1.0"\n'
        f"dependencies = {deps if deps is not None else []!r}\n"
    )
    _write(root / "plugins" / name / "pyproject.toml", pyproject)
    _write(root / "plugins" / name / "src" / module / "__init__.py", '""""""\n')
    if sources:
        for relpath, body in sources.items():
            _write(root / "plugins" / name / "src" / module / relpath, body)


# ---------- end-to-end against the real workspace ----------------------------


def test_real_workspace_passes() -> None:
    violations = check_deps.run_all_checks(_REPO_ROOT)
    assert violations == [], "\n".join(v.render() for v in violations)


def test_main_exit_code_zero_on_real_workspace(capsys: pytest.CaptureFixture[str]) -> None:
    rc = check_deps.main(["--root", str(_REPO_ROOT), "--verbose"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "check_deps OK" in captured.out


# ---------- rule 1: smai-core deps -------------------------------------------


def test_rule1_pass_with_allowed_deps(tmp_path: Path) -> None:
    _make_smai_core(tmp_path, deps=["pydantic>=2.0", "jsonschema>=4.0"])
    assert check_deps.check_smai_core_deps(tmp_path) == []


def test_rule1_pass_with_extras_and_markers(tmp_path: Path) -> None:
    _make_smai_core(
        tmp_path,
        deps=["pydantic[email]>=2.0", "jsonschema (>=4) ; python_version >= '3.11'"],
    )
    assert check_deps.check_smai_core_deps(tmp_path) == []


def test_rule1_fail_on_forbidden_dep(tmp_path: Path) -> None:
    _make_smai_core(tmp_path, deps=["pydantic>=2.0", "requests>=2.30"])
    violations = check_deps.check_smai_core_deps(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule == "smai-core-deps"
    assert "requests" in v.detail
    assert "pyproject.toml" in v.location
    assert "DEC-029" in v.reason
    assert "FAIL" in v.render()


# ---------- rule 2: smai-core imports ----------------------------------------


def test_rule2_pass_with_allowed_imports(tmp_path: Path) -> None:
    sources = {
        "model.py": (
            "import os\n"
            "from pathlib import Path\n"
            "import pydantic\n"
            "from jsonschema import validate\n"
            "from smai_core.helpers import thing\n"
            "from . import sibling\n"
        )
    }
    _make_smai_core(tmp_path, sources=sources)
    assert check_deps.check_smai_core_imports(tmp_path) == []


def test_rule2_fail_on_pipeline_import(tmp_path: Path) -> None:
    sources = {"bad.py": "from smai_agents import Planner\n"}
    _make_smai_core(tmp_path, sources=sources)
    violations = check_deps.check_smai_core_imports(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule == "smai-core-imports"
    assert "smai_agents" in v.detail
    assert v.location.endswith("bad.py:1")
    rendered = v.render()
    assert "from smai_agents import Planner" in rendered
    assert "DEC-029" in rendered


def test_rule2_fail_on_umbrella_import(tmp_path: Path) -> None:
    sources = {"bad.py": "from smai import Runtime\n"}
    _make_smai_core(tmp_path, sources=sources)
    violations = check_deps.check_smai_core_imports(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule == "smai-core-imports"
    assert "'smai'" in v.detail
    assert v.location.endswith("bad.py:1")


def test_rule2_fail_on_plugin_import(tmp_path: Path) -> None:
    sources = {
        "alias.py": "import smai_llm_bedrock as bedrock\n",
        "deep.py": ("import os\n\nfrom smai_store_sqlite.driver import open_db\n"),
    }
    _make_smai_core(tmp_path, sources=sources)
    violations = check_deps.check_smai_core_imports(tmp_path)
    assert {v.location.rsplit(":", 1)[0].rsplit("/", 1)[-1] for v in violations} == {
        "alias.py",
        "deep.py",
    }
    for v in violations:
        assert v.rule == "smai-core-imports"
        assert "DEC-029" in v.reason


# ---------- rule 3: plugin boundaries ----------------------------------------


def test_rule3_pass_with_clean_plugin(tmp_path: Path) -> None:
    _make_plugin(
        tmp_path,
        "smai-llm-bedrock",
        module="smai_llm_bedrock",
        deps=["smai-core", "boto3>=1.34"],
        sources={"client.py": "import boto3\nfrom smai_core import LlmProvider\n"},
    )
    assert check_deps.check_plugin_boundaries(tmp_path) == []


def test_rule3_fail_on_plugin_pipeline_dep(tmp_path: Path) -> None:
    _make_plugin(
        tmp_path,
        "smai-store-sqlite",
        module="smai_store_sqlite",
        deps=["smai-core", "smai-orchestrator>=0.1"],
    )
    violations = check_deps.check_plugin_boundaries(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule == "plugin-deps"
    assert "smai-orchestrator" in v.detail
    assert "smai-store-sqlite" in v.detail
    assert "implementation_plan.md" in v.reason


def test_rule3_fail_on_plugin_pipeline_import(tmp_path: Path) -> None:
    _make_plugin(
        tmp_path,
        "smai-compute-modal",
        module="smai_compute_modal",
        deps=["smai-core"],
        sources={
            "runner.py": ("import os\nfrom smai_runtime.harness import HarnessAPI  # forbidden\n")
        },
    )
    violations = check_deps.check_plugin_boundaries(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule == "plugin-imports"
    assert "smai_runtime" in v.detail
    assert "smai-compute-modal" in v.detail
    assert v.location.endswith("runner.py:2")


def test_rule3_fail_on_plugin_umbrella_dep(tmp_path: Path) -> None:
    _make_plugin(
        tmp_path,
        "smai-store-sqlite",
        module="smai_store_sqlite",
        deps=["smai-core", "smai>=0.1"],
    )
    violations = check_deps.check_plugin_boundaries(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule == "plugin-deps"
    assert "'smai>=0.1'" in v.detail


def test_rule3_fail_on_plugin_umbrella_import(tmp_path: Path) -> None:
    _make_plugin(
        tmp_path,
        "smai-llm-bedrock",
        module="smai_llm_bedrock",
        deps=["smai-core"],
        sources={"client.py": "from smai import Runtime\n"},
    )
    violations = check_deps.check_plugin_boundaries(tmp_path)
    assert len(violations) == 1
    v = violations[0]
    assert v.rule == "plugin-imports"
    assert "'smai'" in v.detail
    assert v.location.endswith("client.py:1")


def test_rule3_allows_plugin_to_plugin_import(tmp_path: Path) -> None:
    _make_plugin(
        tmp_path,
        "smai-llm-bedrock",
        module="smai_llm_bedrock",
        deps=["smai-core"],
        sources={"x.py": "from smai_store_sqlite import helper\n"},
    )
    _make_plugin(
        tmp_path,
        "smai-store-sqlite",
        module="smai_store_sqlite",
        deps=["smai-core"],
    )
    assert check_deps.check_plugin_boundaries(tmp_path) == []


# ---------- main() exit code on failure --------------------------------------


def test_main_exit_code_one_on_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_smai_core(tmp_path, deps=["requests>=2"])
    rc = check_deps.main(["--root", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAIL [smai-core-deps]" in err
    assert "requests" in err
