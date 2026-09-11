#!/usr/bin/env python3
"""Source block algebra, conditional innovations and same-function calculus."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import conditional_teaching as f
from grain_research.vector_mechanisms import evidence_coordinate_reader as coord


def toy():
    """Mixed Gaussian observations in three fixed source groups."""
    rng = np.random.default_rng(860)
    x, v = rng.normal(size=(2, 9, 4))
    y = rng.normal(size=(9, 5))
    groups = np.repeat(np.arange(3), 3)
    m = f.base.fit(x, v, groups > 0, y, groups, np.zeros(5), np.zeros(5))
    m.update(target=y, groups=groups)
    return rng, m


def test_block_conditioning_matches_direct_and_keeps_full_model():
    _, original = toy()
    models = f.nested_models(original)
    kernel = f.base.gram(
        original["x"],
        original["v"],
        original["kinds"],
        original["x"],
        original["v"],
        original["kinds"],
    )
    for m in models:
        use = m["selected"]
        a = kernel[np.ix_(use, use)] + np.diag(
            float(m["regularizer"]) / m["weights"][use] ** 2
        )
        expected = np.linalg.solve(a, m["target"][use])
        np.testing.assert_allclose(m["beta"][use], expected, atol=1e-8)
        np.testing.assert_array_equal(m["beta"][~use], 0)
        np.testing.assert_array_equal(m["weights"], original["weights"])
        assert m["regularizer"] == original["regularizer"]
    np.testing.assert_allclose(models[-1]["beta"], original["beta"], atol=1e-8)
    np.testing.assert_allclose(
        sum(m["innovation_beta"] for m in models), original["beta"], atol=1e-8
    )


def test_final_conditioning_is_order_invariant_but_increments_are_not():
    _, source = toy()
    forward = f.nested_models(source)
    reverse = f.nested_models(source, (2, 1, 0))
    np.testing.assert_allclose(forward[-1]["beta"], reverse[-1]["beta"], atol=1e-8)
    assert not np.allclose(forward[0]["beta"], reverse[0]["beta"])
    assert not np.allclose(
        forward[0]["beta"][source["groups"] == 0],
        forward[-1]["beta"][source["groups"] == 0],
    )
    with pytest.raises(ValueError):
        f.nested_models(source, (0, 0, 1))


def test_unassimilated_targets_do_not_enter_subset_fit_with_fixed_weights():
    _, source = toy()
    original = f.nested_models(source)
    changed = source | {"target": source["target"].copy()}
    changed["target"][source["groups"] == 2] += 123
    other = f.nested_models(changed)
    for i in (0, 1):
        np.testing.assert_array_equal(original[i]["beta"], other[i]["beta"])
    assert not np.allclose(original[2]["beta"], other[2]["beta"])


def test_all_nested_functions_and_lifts_keep_directional_derivatives():
    rng, source = toy()
    x, dx = rng.normal(size=(2, 4))
    r, dr = rng.normal(size=(2, 5, 9))
    db, ddb, d0 = rng.normal(size=(3, 9))
    eps = 1e-5
    for m in f.nested_models(source):
        c, dc = f.base.field(m, x, dx)
        values = [f.base.field(m, x + sign * eps * dx) for sign in (-1, 1)]
        np.testing.assert_allclose(
            dc[0], (values[1] - values[0]) / (2 * eps), atol=1e-8
        )
        _, dd, _ = coord.lift(db, ddb[None], r, dr[None], c, dc, d0)
        lifted = [
            coord.lift_value(db + sign * eps * ddb, r + sign * eps * dr, value, d0)
            for sign, value in zip((-1, 1), values, strict=True)
        ]
        np.testing.assert_allclose(
            dd[0], (lifted[1] - lifted[0]) / (2 * eps), atol=1e-7
        )


def test_signed_change_is_not_sum_of_positive_energies():
    value = f.error_change(
        np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.5, 1.0])
    )
    assert value["signed_cross_energy"] < 0
    assert value["error_change_over_target"] == pytest.approx(-0.75)
    assert value["after_error2"] == pytest.approx(1.25)


def test_complete_nested_summary_inventory_preserves_prior_tables():
    import pandas as pd

    from scripts.run_grain_conditional_teaching import COUNTS, p82, summarize

    rng, source = toy()
    r = rng.normal(size=(5, 9))
    models = []
    for moving in (False, True):
        ms = f.nested_models(source)
        models.extend(
            m | {"fixed_frame": r, "moving": moving, "full_beta_relative": 0.0}
            for m in ms
        )
    rows = []
    for _ in range(8):
        rr = rng.normal(size=(6, 2, 9))
        dr = rng.normal(size=(6, 2, 4, 9))
        p = rng.normal(size=(2, 4, 9))
        v = rng.normal(size=(2, 4, 3, 4, 9))
        packets = [p82.projections(p, v, a, d) for a, d in zip(rr, dr, strict=True)]
        rows.append(
            {k: np.stack([a[k] for a in packets]) for k in packets[0]}
            | {
                "readers": rr,
                "frame": r,
                "coordinate_values": rng.normal(size=(6, 5)),
                "raw_group_parts": rng.normal(size=(2, 3, 5)),
                "native_r": rng.normal(size=(2, 9)),
                "native_primal": rng.normal(size=(2, 9)),
                "native_vectors": rng.normal(size=(2, 3, 4, 9)),
                "target_scalar": rng.normal(size=(2, 2)),
                "target_actions": rng.normal(size=(2, 3, 2, 4)),
            }
        )
    previous = [
        pd.DataFrame({"variant": ["prior"] * n, "authors": [8] * n})
        for n in (288, 864, 3456, 156, 12, 1152)
    ]
    tables = summarize(rows, models, {"d0": np.zeros(9)}, previous)
    assert [len(t) for t in tables] == COUNTS
    for old, new in zip(previous, tables, strict=False):
        pd.testing.assert_frame_equal(
            old, new.iloc[: len(old)][list(old)].reset_index(drop=True)
        )
    assert set(tables[8].variant) == set(f.VARIANTS)
    assert set(tables[9]["update"]) == {"V_to_VB", "VB_to_VBC"}
