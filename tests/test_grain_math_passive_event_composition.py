#!/usr/bin/env python3
"""Finite event/mention identities, null observations and distinct occasions."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_passive_event_composition as M

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads(
    (ROOT / "configs/grain_math_passive_event_composition_v1.json").read_text()
)
ATOM = {(0, 0): 0.1, (0, 1): 0.9, (1, 0): 0.2, (1, 1): 0.8}


def channel(regime, family=0, polarity=1, order=0):
    """Score design metadata with artificial isolated observations only."""
    row = M.account(SPEC, "extraversion", family, polarity, order, regime)
    return M.channels(row["history"], ATOM)


def test_full_design_without_omitted_histories():
    rows = M.documents(SPEC)
    assert len(rows) == 408
    assert sum(r["role"] == "atom" for r in rows) == 24
    assert {r["regime"] for r in rows if r["role"] == "account"} == set(M.REGIMES)
    assert [r["unit"] for r in rows] == list(range(408))


@pytest.mark.parametrize("family", [0, 1])
@pytest.mark.parametrize("polarity", [0, 1])
def test_alias_is_not_a_new_behavior_event(family, polarity):
    base, alias, distinct = (
        channel(k, family, polarity)
        for k in ("balanced", "alias_focus", "distinct_focus")
    )
    assert alias["event_profile"] == base["event_profile"]
    assert alias["own_events"] == 2 and alias["own_mentions"] == 4
    assert distinct["own_events"] == 4
    expected = 0.25 * (ATOM[family, polarity] - ATOM[1 - family, 1 - polarity])
    assert distinct["event_profile"] - base["event_profile"] == pytest.approx(expected)
    assert alias["mention_profile"] == distinct["mention_profile"]


@pytest.mark.parametrize("regime", M.REGIMES)
def test_event_and_mention_profiles_do_not_depend_on_display_order(regime):
    assert channel(regime, order=0) == channel(regime, order=1)


def test_other_person_is_retained_but_not_counted_as_own_action():
    base, other = channel("balanced"), channel("other_focus")
    assert base["event_profile"] == other["event_profile"]
    assert other["total_reports"] == 4
    assert channel("other_only")["event_profile"] is None
    assert channel("other_only")["mention_profile"] is None


def test_revision_replaces_claim_without_erasing_narrative_history():
    r, alias = channel("revision"), channel("revision_alias")
    assert r["event_profile"] == alias["event_profile"] == pytest.approx(0.15)
    assert r["mention_profile"] != alias["mention_profile"]
    assert alias["stale_derived"] == ["r3", "r4"]
    assert channel("retracted")["event_profile"] == 0.2


def test_unknown_event_not_fabricated_from_semantic_similarity():
    row = M.account(SPEC, "extraversion", 0, 1, 0, "distinct_focus")
    r = copy.deepcopy(row["history"])
    r[1]["event_id"] = "K1"
    with pytest.raises(ValueError, match="independent new event"):
        M.channels(r, ATOM)


def test_finite_atom_values_required():
    bad = {**ATOM, (0, 1): float("nan")}
    row = M.account(SPEC, "extraversion", 0, 1, 0, "balanced")
    with pytest.raises(ValueError, match="Finite"):
        M.channels(row["history"], bad)


def test_alias_and_distinct_words_differ_only_in_occasion_identifiers():
    for order in (0, 1):
        a, b = (
            M.account(SPEC, "openness", 1, 0, order, k)["body"]
            for k in ("alias_focus", "distinct_focus")
        )
        assert b.replace("K3", "K1").replace("K4", "K1") == a


def test_endpoint_geometry_preserves_zero_and_opposite_displacements():
    assert M.displacement(np.zeros(3), np.zeros(3), np.ones(3))["cosine"] is None
    assert M.displacement(np.zeros(3), np.ones(3), np.zeros(3))[
        "cosine"
    ] == pytest.approx(-1)


def test_descriptive_summary_excludes_unobserved_profile():
    summary = M.summarize(
        [
            {
                "score": 0.1,
                "log_odds": 1,
                "event_profile": None,
                "mention_profile": None,
                "oriented_score": None,
            }
        ]
    )
    assert summary["event_profile_rms"] is None
    assert summary["oriented_cases"] == 0
