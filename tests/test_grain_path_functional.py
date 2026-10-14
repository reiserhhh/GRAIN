#!/usr/bin/env python3
"""Synthetic failure-capable tests of path-functional construction, not validity."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_path_functional as M


def full_table():
    """Return a fully positive example with all five measured binary profiles."""
    return np.full((32, 2, 2, 2), 1 / 256)


def record():
    """Return one distinct-event chronology, not three independent subjects."""
    return {
        "author_key": "opaque",
        "events": [
            {"event_id": str(i), "timestamp": i, "cell": c}
            for i, c in enumerate((0, 0, 0, 1, 1))
        ],
    }


def direct_score(f, h):
    """Compute all target functionals directly from a full law."""
    r = f.sum(0)
    return np.einsum("gq,gabc,abc->q", M.profile_bits(), f, h / r)


def test_collapsed_atom_bounds_contain_full_law():
    """All eight binary cells must remain nonnegative at both extreme masses."""
    rng = np.random.default_rng(236)
    f = rng.uniform(0.1, 1, (32, 3, 3, 3))
    f /= f.sum()
    r, joint, low, high = M.collapsed_bounds(f)
    assert np.all(low <= joint + 1e-12)
    assert np.all(joint <= high + 1e-12)
    for q, a, b, c in np.ndindex(5, 3, 3, 3):
        p = joint[q, :, b].sum()
        x, z = r[a, b].sum(), r[:, b, c].sum()
        px, pz, xz = joint[q, a, b].sum(), joint[q, :, b, c].sum(), r[a, b, c]
        t = r[:, b].sum()
        for value in (low[q, a, b, c], high[q, a, b, c]):
            atoms = [
                value,
                px - value,
                pz - value,
                xz - value,
                p - px - pz + value,
                x - px - xz + value,
                z - pz - xz + value,
                t - p - x - z + px + pz + xz - value,
            ]
            assert min(atoms) >= -1e-12


def test_positive_circuit_preserves_every_declared_margin_and_other_targets():
    """Equal complete constraints can still give a different target reading."""
    f = full_table()
    law = M.compile_law(f)
    h = np.zeros((2, 2, 2))
    h[0, 0, 0] = 1
    scored = M.measure_mass(law, h)
    for q, row in enumerate(scored["targets"]):
        i = row["witness_index"]
        plus, minus = M.witness_laws(
            f, law["circuit_index"][i], law["circuit_amplitude"][i]
        )
        assert np.min(plus) > 0 and np.min(minus) > 0
        for axis in (0, 1, 3):
            np.testing.assert_allclose(plus.sum(axis), minus.sum(axis), atol=1e-14)
        gap = direct_score(plus, h) - direct_score(minus, h)
        assert gap[q] == pytest.approx(row["signed_witness_plus_minus_gap"])
        assert gap[q] != 0
        np.testing.assert_allclose(np.delete(gap, q), 0, atol=1e-14)
        assert row["outer_low"] <= min(
            row["witness_plus_score"], row["witness_minus_score"]
        )
        assert row["outer_high"] >= max(
            row["witness_plus_score"], row["witness_minus_score"]
        )


def test_additive_query_is_fixed_without_claiming_complete_fiber_identification():
    """A genuinely additive density gives zero circuit response and remainder."""
    f = full_table()
    law = M.compile_law(f)
    d = np.array([1, 2])[:, None, None] + np.array([3, 4])[None, None, :]
    h = np.broadcast_to(d, (2, 2, 2)).copy() * law["reference"]
    h /= h.sum()
    rows = M.measure_mass(law, h)["targets"]
    for row in rows:
        assert row["fixed_additive_account"] == pytest.approx(row["empirical_point"])
        assert row["certified_feasible_width"] < 1e-14
        assert row["status"] == "NO_TESTED_WITNESS_NOT_IDENTIFICATION"


def test_unsupported_history_not_neutral_or_smoothed():
    """Unobserved source support produces missing measurement rather than zero."""
    f = np.zeros((32, 2, 2, 2))
    f[:, 0, 0, 0] = 1 / 32
    law = M.compile_law(f)
    h = np.zeros((2, 2, 2))
    h[1, 1, 1] = 1
    result = M.measure_mass(law, h)
    assert result == {"supported": False, "coverage": 0.0, "targets": None}
    assert len(law["circuit_index"]) == 0


@pytest.mark.parametrize(
    "defect", ["duplicate", "time", "label", "fractional", "event_label"]
)
def test_record_rejects_bad_provenance_or_labels(defect):
    """Duplicated events, malformed chronology and target fields are disallowed."""
    value = copy.deepcopy(record())
    if defect == "duplicate":
        value["events"][1]["event_id"] = value["events"][0]["event_id"]
    elif defect == "time":
        value["events"][1]["timestamp"] = 0
    elif defect == "label":
        value["Y"] = [1, 0, 1, 0, 1]
    elif defect == "fractional":
        value["events"][0]["cell"] = 0.5
    else:
        value["events"][0]["label"] = 1
    with pytest.raises(ValueError):
        M.history_mass(value, 2)


def test_histories_count_overlap_without_multiplying_likelihoods():
    """Five ordered events give three empirical triple observations."""
    h = M.history_mass(record(), 2)
    assert h.sum() == pytest.approx(1)
    assert np.count_nonzero(h) == 3


def test_malformed_mass_and_table_fail_explicitly():
    """Signed laws and insufficiently normalized histories are not repaired."""
    with pytest.raises(ValueError):
        M.compile_law(-full_table())
    with pytest.raises(ValueError):
        M.additive_account(np.ones((2, 2, 2)), np.full((2, 2, 2), 1 / 8))
