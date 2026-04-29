"""DEC-013 / DEC-017 — additive-baseline skip pattern.

Per `03-state-machine.md` §3.4 and the brief, additive baselines
(``technique_id is None`` and ``is_baseline=True``) skip implementation
dispatch. The skip is double-enforced:

1. **MetadataStore-level pre-filter:** ``get_ready_to_implement_entry``
   excludes baselines at the SQL layer (per `07` §5.6.3 / DEC-013), so
   the entry-spec's phase-2 discovery never returns them.
2. **Dispatch-handler-level short-circuit:** Task 2.B3's
   ``make_dispatch_technique_implementation`` inspects
   :func:`smai_runtime.is_additive_baseline` and returns
   ``DispatchOutcome(submitted_handles=[], error=None)`` for baselines.

Phase 2 ships both layers belt-and-braces; the engine's
worker loop sees an entry that's already ``state="implemented"`` from
the proposal-pipeline registration transaction (DEC-013 §6 of `08`
§3.3) and the entry-spec spec validators don't see baselines at all.

These tests exercise the SqliteStore-level filter to verify baselines
don't appear in the entry spec's phase-2 candidate set.
"""

from __future__ import annotations

from _specs_fakes import make_cg, make_entry  # type: ignore[import-not-found]
from smai_store_sqlite import SqliteStore


async def test_get_ready_to_implement_entry_excludes_additive_baselines(
    sqlite_store: SqliteStore,
) -> None:
    """Per `07` §5.6.3 / DEC-013 — the scheduling query excludes
    entries with ``technique_id IS NULL`` (additive baselines).

    Post-R3 the SqliteStore filter is ``technique_id IS NOT NULL AND
    state='pending'``; this test verifies the additive-baseline part of
    that filter (the substitutive-baseline counterpart is asserted in
    :func:`test_get_ready_to_implement_entry_includes_substitutive_baselines`).
    """
    cg_id = "cg-baseline"
    cg = make_cg(cg_id=cg_id, state="implementing")
    await sqlite_store.create_cg(cg)

    # An additive baseline (technique_id=None, is_baseline=True) starts
    # in state ``implemented`` per the proposal-pipeline registration
    # transaction (DEC-013).
    additive_baseline = make_entry(
        "entry-additive-baseline",
        cg_id=cg_id,
        state="implemented",
        technique_id=None,
        is_baseline=True,
    )
    await sqlite_store.create_entry(additive_baseline)

    # A regular pending entry should be returned.
    treatment = make_entry(
        "entry-treatment",
        cg_id=cg_id,
        state="pending",
        technique_id="tq-1",
        is_baseline=False,
    )
    await sqlite_store.create_entry(treatment)

    page = await sqlite_store.get_ready_to_implement_entry(limit=100)
    returned_ids = {e.id for e in page.items}

    # Treatment returned; additive baseline NOT returned.
    assert "entry-treatment" in returned_ids
    assert "entry-additive-baseline" not in returned_ids


async def test_get_ready_to_implement_entry_includes_substitutive_baselines(
    sqlite_store: SqliteStore,
) -> None:
    """Per DEC-013 / DEC-017 — substitutive baselines (``is_baseline=True``
    AND ``technique_id IS NOT NULL``, e.g. a VGG reference in an
    architecture CG) DO need implementation and must appear in the
    scheduling-query result.

    Pre-R3 (the C4 carry-forward) the SqliteStore filter was
    ``is_baseline=False AND technique_id NOT NULL`` which silently
    excluded substitutive baselines. R3 / F3 dropped the
    ``is_baseline=False`` clause; this test pins the corrected
    behavior so a future regression surfaces immediately.
    """
    cg_id = "cg-sub-baseline-fixed"
    cg = make_cg(cg_id=cg_id, state="implementing")
    await sqlite_store.create_cg(cg)
    sub_baseline = make_entry(
        "entry-vgg",
        cg_id=cg_id,
        state="pending",
        technique_id="tq-vgg",
        is_baseline=True,
    )
    await sqlite_store.create_entry(sub_baseline)
    treatment = make_entry(
        "entry-treatment-fixed",
        cg_id=cg_id,
        state="pending",
        technique_id="tq-1",
        is_baseline=False,
    )
    await sqlite_store.create_entry(treatment)

    page = await sqlite_store.get_ready_to_implement_entry(limit=100)
    returned_ids = {e.id for e in page.items}

    # Both substitutive baseline AND treatment are returned.
    assert "entry-vgg" in returned_ids, (
        "substitutive baseline must be returned by get_ready_to_implement_entry (R3 / F3 fix)"
    )
    assert "entry-treatment-fixed" in returned_ids
