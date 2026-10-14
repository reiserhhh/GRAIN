#!/usr/bin/env python3
"""Small exhaustive conditional orbits and honest reference-law boundaries."""

import itertools

import numpy as np
import pytest

from grain_research.construction import math_anchor_law_disintegration as M


def fixture():
    """One unbalanced target stratum; other strata have one author per side."""
    g = np.concatenate((np.arange(32), [0]))
    k = np.random.default_rng(217).uniform(0.2, 1, (3, 2, 2, 2, len(g)))
    return k, g


def test_exact_orbit_and_shared_sample_product():
    k, g = fixture()
    # Every other-bit stratum changes, but expectation is linear in cell laws;
    # enumerate each local orbit separately instead of 3*2^15 global states.
    mean = M.orbit_density(k, g, 0)
    for c in range(0, 32, 2):
        ix = np.flatnonzero((g & ~1) == c)
        rows = []
        for high in itertools.combinations(ix, int(np.sum(g[ix] == c + 1))):
            assigned = g.copy()
            assigned[ix] = c
            assigned[list(high)] = c + 1
            rows.append(M.densities(k, assigned)[..., [c, c + 1]])
        np.testing.assert_allclose(np.mean(rows, 0), mean[..., [c, c + 1]], atol=1e-14)
    assert np.max(abs(mean[..., 2, :, 0] - mean[..., 2, :, 1])) > 0


def test_uniform_cancellation_is_constructed_not_observed():
    k, g = fixture()
    baseline = M.orbit_density(k, g, 0)
    a = M.action(baseline, M.priors(g)[0], 0)
    np.testing.assert_allclose(a[..., :2], 0, atol=1e-14)
    assert np.max(abs(a[..., 2])) > 0
    assert np.max(abs(M.action(baseline, M.priors(g)[1], 0))) > 0


def test_all_parts_sum_and_label_reassignment_preserves_reference():
    k, g = fixture()
    a = M.accounts(k, g)
    np.testing.assert_allclose(a[..., 0, :] - a[..., 1, :], a[..., 2, :])
    gg = g.copy()
    gg[[0, 1]] = gg[[1, 0]]
    np.testing.assert_allclose(M.orbit_density(k, g, 0), M.orbit_density(k, gg, 0))
    assert np.max(abs(M.accounts(k, gg)[..., 0] - a[..., 0])) > 0


def test_other_only_kernel_retains_association_without_target_binding():
    k, g = fixture()
    for c in range(0, 32, 2):
        ix = np.flatnonzero((g & ~1) == c)
        k[..., ix] = k[..., ix[0], None]
    a = M.accounts(k, g)
    np.testing.assert_allclose(a[..., 2, 0], 0, atol=2e-14)
    assert np.max(abs(a[1, ..., 1, 0])) > 0


def test_rank_translation_and_degenerate_ties():
    a = M.rank(3, np.array([0, 1, 2, 4]))
    b = M.rank(13, np.array([10, 11, 12, 14]))
    assert a["two_sided_rank"] == b["two_sided_rank"]
    assert M.rank(1, np.ones(4))["two_sided_rank"] == 1


def test_cold_scoring_uses_only_addressed_vectors():
    g = np.arange(32)
    rng = np.random.default_rng(219)
    bank = {"source_means": rng.normal(0, 0.1, (32, 2, 3)), "codes": g, "variance": 1}
    record = {
        "frame_key": "cold",
        "windows": [
            [
                {
                    "event_id": f"{v}:{i}",
                    "vector": rng.normal(0, 0.1, 3).tolist(),
                    "condition": 0,
                }
                for i in range(12)
            ]
            for v in (0, 1)
        ],
        "edit": {
            "view": 1,
            "selected": [f"1:{i}" for i in range(4)],
            "control": [f"1:{i}" for i in range(4, 8)],
        },
    }
    a = np.asarray(M.cold(bank, record)["actions"])
    assert a.shape == (2, 3, 3, 5)
    np.testing.assert_allclose(a[:, :, 0] - a[:, :, 1], a[:, :, 2])
    record["labels"] = [0] * 5
    with pytest.raises(ValueError):
        M.cold(bank, record)


@pytest.mark.parametrize("change", ["missing", "nonpositive", "shape"])
def test_invalid_observations_fail(change):
    k, g = fixture()
    if change == "missing":
        g[g == 31] = 30
    elif change == "nonpositive":
        k[..., 0] = 0
    else:
        k = k[:, :1]
    with pytest.raises(ValueError):
        M.accounts(k, g)
