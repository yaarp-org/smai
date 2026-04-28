"""Tests for the process-local prompt-config cache (§10.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from smai_agents.prompts import (
    clear_prompt_config_cache,
    load_prompt_config,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_prompt_config_cache()


def test_cache_returns_same_instance_for_identical_keys() -> None:
    """Two loads with the same key return the cached instance."""
    cfg1 = load_prompt_config("harness_builder")
    cfg2 = load_prompt_config("harness_builder")
    assert cfg1 is cfg2


def test_cache_distinguishes_variants() -> None:
    """The cache key includes variant_name — different variants don't collide."""
    base = load_prompt_config("harness_builder")
    variant = load_prompt_config("harness_builder", variant_name="lint_first")
    assert base is not variant
    # Both stay cached.
    assert load_prompt_config("harness_builder") is base
    assert (
        load_prompt_config("harness_builder", variant_name="lint_first") is variant
    )


def test_cache_distinguishes_roles() -> None:
    """Different roles share no cache entry."""
    hb = load_prompt_config("harness_builder")
    ti = load_prompt_config("technique_implementer")
    assert hb is not ti
    assert hb.role != ti.role


def test_cache_invalidates_on_overrides_dir_mtime_change(tmp_path: Path) -> None:
    """Editing the override YAML causes the next load to re-read."""
    override = {
        "layer": "override",
        "name": "v1",
        "role": "harness_builder",
        "system_prompt": "first version",
    }
    target = tmp_path / "harness_builder.yaml"
    target.write_text(yaml.safe_dump(override))

    first = load_prompt_config("harness_builder", overrides_dir=tmp_path)
    assert "first version" in first.system_prompt

    # Mutate the file and bump its mtime.
    override["system_prompt"] = "second version"
    target.write_text(yaml.safe_dump(override))
    # Bump mtime so the cache key changes — write() may not always
    # advance mtime within the same second on every filesystem.
    new_mtime = target.stat().st_mtime + 5
    target.touch()
    import os

    os.utime(target, (new_mtime, new_mtime))

    second = load_prompt_config("harness_builder", overrides_dir=tmp_path)
    assert "second version" in second.system_prompt
    assert second is not first


def test_cache_distinguishes_overrides_dirs(tmp_path: Path) -> None:
    """Two distinct overrides_dirs cache as separate entries."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    (dir_a / "harness_builder.yaml").write_text(
        yaml.safe_dump(
            {
                "layer": "override",
                "name": "a",
                "role": "harness_builder",
                "system_prompt": "from-a",
            }
        )
    )
    (dir_b / "harness_builder.yaml").write_text(
        yaml.safe_dump(
            {
                "layer": "override",
                "name": "b",
                "role": "harness_builder",
                "system_prompt": "from-b",
            }
        )
    )

    cfg_a = load_prompt_config("harness_builder", overrides_dir=dir_a)
    cfg_b = load_prompt_config("harness_builder", overrides_dir=dir_b)
    assert cfg_a is not cfg_b
    assert "from-a" in cfg_a.system_prompt
    assert "from-b" in cfg_b.system_prompt


def test_clear_cache_forces_reload() -> None:
    """clear_prompt_config_cache() drops cache entries."""
    first = load_prompt_config("harness_builder")
    clear_prompt_config_cache()
    second = load_prompt_config("harness_builder")
    # The Pydantic models compare by value but we want ID inequality
    # to confirm the cache was cleared (the loader constructs a new
    # instance on every miss).
    assert first is not second


def test_cache_preserves_no_overrides_after_override_added(tmp_path: Path) -> None:
    """A cached load with no overrides_dir doesn't get clobbered by a
    later load that uses an overrides_dir."""
    bare = load_prompt_config("harness_builder")
    (tmp_path / "harness_builder.yaml").write_text(
        yaml.safe_dump(
            {
                "layer": "override",
                "name": "x",
                "role": "harness_builder",
                "system_prompt": "with override",
            }
        )
    )
    overridden = load_prompt_config("harness_builder", overrides_dir=tmp_path)
    assert overridden is not bare
    # Bare load is still cached and unchanged.
    re_bare = load_prompt_config("harness_builder")
    assert re_bare is bare
    assert "with override" not in re_bare.system_prompt
