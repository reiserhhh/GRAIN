#!/usr/bin/env python3
"""Known relation/code worlds and exact material-action invariants for HM58."""

import itertools

import numpy as np
import pytest

from grain_research.construction.math_event_probability_pooling import pools
from grain_research.construction.math_event_relation_renaming import (
    actual_atom_weights,
    collision,
    orbit_decomposition,
    pair_design,
    relation_signature,
    renamed,
    transform,
)
from grain_research.construction.math_passive_event_composition import record


@pytest.mark.parametrize("g", range(4))
def test_action_preserves_equality_actor_order_and_claim_identity(g):
    """Consistent text and metadata action must not alter relational information."""
    history = [
        record("r1", "K1", 0, 1),
        record("r2", "K2", 1, 0),
        record("r3", "K1", 0, 1, kind="alias", parent="r1"),
        record("r4", "K4", 0, 1, actor="B"),
    ]
    text = "On occasion K1, A acts.\nOn occasion K2, A waits.\nOn occasion K1, A acts.\nOn occasion K4, B acts."
    result, changed = transform(text, history, g)
    assert relation_signature(changed) == relation_signature(history)
    assert result.count(renamed("K1", g)) == 2
    assert [r["actor"] for r in changed] == ["A", "A", "A", "B"]
    assert changed[2]["parent_id"] == "r1"
    assert history[0]["event_id"] == "K1"


@pytest.mark.parametrize("g,h", itertools.product(range(4), repeat=2))
def test_action_composition_and_inverse(g, h):
    """The material transformation is an exact C4 action, not assumed neural symmetry."""
    text, history = "On occasion K1, A acts.", [record("r1", "K1", 0, 1)]
    a, b = transform(text, history, g)
    x, y = transform(a, b, h)
    assert (x, y) == transform(text, history, (g + h) % 4)
    assert transform(a, b, (-g) % 4) == (text, history)


def test_true_relation_world_has_only_invariant_contrast():
    """A relation-only positive effect survives every code without forced fitting."""
    result = orbit_decomposition([0.2] * 4)
    assert result["orbit_mean"] == pytest.approx(0.2)
    assert result["code_variance"] == 0


def test_pure_code_world_has_zero_invariant_contrast():
    """The same method must permit a zero relation effect."""
    result = orbit_decomposition([0.2, -0.2, 0.2, -0.2])
    assert result["orbit_mean"] == 0
    assert result["code_variance"] == pytest.approx(0.04)


def test_mixed_world_keeps_both_components():
    """Nonzero group mean and code response coexist rather than erase one another."""
    result = orbit_decomposition([0.3, 0.1, 0.3, 0.1])
    assert result["orbit_mean"] == pytest.approx(0.2)
    assert result["code_variance"] == pytest.approx(0.01)
    assert result["orbit_energy"] == pytest.approx(0.05)
    assert result["identity_error"] < 1e-15


def test_orbit_average_invariance_is_tautology_not_a_data_result():
    """Reindexing a full orbit leaves its mean unchanged, regardless of mechanism."""
    x = np.array([0.11, -0.2, 0.03, 0.41])
    for g in range(4):
        assert orbit_decomposition(np.roll(x, g))["orbit_mean"] == pytest.approx(
            x.mean()
        )


def test_actual_atom_weights_preserve_code_and_do_not_misown_other_events():
    """Each event must use its actual-code atom; other-person actions stay contextual."""
    h = [
        record("r1", "K3", 0, 1),
        record("r2", "K4", 1, 0),
        record("r3", "K3", 0, 1, kind="alias", parent="r1"),
        record("r4", "K1", 0, 1, actor="B"),
    ]
    keys, w, n = actual_atom_weights(h, "event")
    assert keys == [(0, 1, "K3"), (1, 0, "K4")]
    assert w == [0.5, 0.5] and n == 2
    assert actual_atom_weights(h, "mention")[1] == [2 / 3, 1 / 3]


def test_mixture_code_orbit_cancels_assignments_but_retains_content_multiplicity():
    """After all code rotations, same content multiplicities give equal mixture means."""
    z = np.arange(12).reshape(4, 3) / 11
    focal = [pools([x], [1])[0] for x in z]
    alias, distinct = [], []
    for g in range(4):
        alias.append(focal[g])
        distinct.append((focal[g] + focal[(g + 2) % 4] + focal[(g + 3) % 4]) / 3)
    np.testing.assert_allclose(
        np.mean(alias, axis=0), np.mean(distinct, axis=0), atol=1e-15
    )


def test_geometric_pool_need_not_commute_with_orbit_average():
    """Do not impose linear code cancellation on a nonlinear normalized product."""
    z = np.array([[1.3, -0.2, -0.1], [-1.0, 1.5, 0.7], [0.1, 0.4, 1.3], [2, 0, -1]])
    alias = [pools([x], [1])[1] for x in z]
    distinct = [
        pools(z[[g, (g + 2) % 4, (g + 3) % 4]], [1 / 3] * 3)[1] for g in range(4)
    ]
    assert np.max(np.abs(np.mean(alias, axis=0) - np.mean(distinct, axis=0))) > 1e-3


def test_collision_probability_equality_and_midpoint_floor():
    """Identical declared summaries can still leave a positive observation floor."""
    r = collision([0.3, 0.7], [0.3, 0.7], [0.1, 0.9], [0.7, 0.3], [0, 1], 1e-12)
    assert r["same_summary_changed_observation"]
    assert r["paired_score_RMS_floor"] == pytest.approx(0.3)


def test_different_summaries_do_not_receive_a_fake_collision_floor():
    """Do not claim an exact obstruction when code-conditioned summaries differ."""
    r = collision([0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.7, 0.3], [0, 1], 1e-12)
    assert not r["same_summary"]
    assert r["paired_score_RMS_floor"] is None


def test_complete_fixed_pair_roster():
    """All three regime and four order contrasts remain in the experiment."""
    result = pair_design(["O", "C", "E", "A", "N", "anthro"])
    assert len(result) == 240
    assert [r["pair"] for r in result] == list(range(240))
    assert sum(r["contrast"] == "distinct_minus_alias" for r in result) == 48
    assert sum(r["contrast"] == "order_balanced" for r in result) == 24


@pytest.mark.parametrize("x", [[], [1, 2, 3], [1, 2, 3, np.nan]])
def test_missing_or_invalid_orbits_fail(x):
    """No incomplete orbit or invalid readout is interpreted as zero."""
    with pytest.raises(ValueError):
        orbit_decomposition(x)


@pytest.mark.parametrize("code,g", [("K0", 0), ("K5", 1), ("K1", 4)])
def test_invalid_group_or_code_fails(code, g):
    """Reject transformations outside the declared scope."""
    with pytest.raises(ValueError):
        renamed(code, g)


def test_text_metadata_code_mismatch_fails():
    """Undeclared text references must not be silently rewritten."""
    with pytest.raises(ValueError):
        transform("On occasion K2, A acts.", [record("r1", "K1", 0, 1)], 1)


def test_alternate_audit_boolean_and_null_handling():
    """The independent CSV audit must preserve Boolean and missing floor fields."""
    from scripts.verify_grain_math_event_relation_renaming import compare

    compare("True", np.bool_(True), 1e-11)
    compare("False", False, 1e-11)
    compare("", None, 1e-11)
    with pytest.raises(ValueError):
        compare("0", None, 1e-11)


def test_alternate_audit_rejects_duplicate_table_keys(tmp_path):
    """Counts and unique keys are checked separately, not a dictionary overwrite."""
    from scripts.verify_grain_math_event_relation_renaming import table_check

    path = tmp_path / "03"
    path.mkdir()
    (path / "test.csv").write_text("unit,value\n0,0.1\n0,0.1\n")
    with pytest.raises(ValueError):
        table_check(
            tmp_path,
            "test",
            [{"unit": 0, "value": 0.1}, {"unit": 1, "value": 0.2}],
            ("unit",),
            1e-11,
        )
