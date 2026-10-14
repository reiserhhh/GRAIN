#!/usr/bin/env python3
"""SOURCE-reference composition controls, including a world that can reverse."""

import numpy as np
import pytest

from grain_research.construction import math_reference_mixture as M


def test_interior_reversal_and_naive_average():
    """Two endpoint orders agree while evidence-weighted interior order reverses."""
    u, v = M.toy()
    r = M.compare(u, v)[0]
    assert r["endpoint_consensus_interior_reversal"]
    assert r["status"] == "crossing"
    assert len(r["crossing_weights"]) == 2
    assert min(r["endpoint_deltas"]) > 0.09
    assert M.mix(u, 0.5)["means"][0] < M.mix(v, 0.5)["means"][0]
    np.testing.assert_allclose(
        M.moments(u)["endpoint_means"][:, 0] - M.moments(v)["endpoint_means"][:, 0],
        0.1,
        atol=3e-16,
    )
    assert M.mix(u, 0.5)["source1_responsibility"] > 0.99
    assert M.mix(v, 0.5)["source1_responsibility"] < 0.01


def test_same_sources_and_equal_evidence_ratio():
    """Without differential responsibilities, endpoint consensus is preserved."""
    u, v = M.toy()
    same_u, same_v = np.stack([u[0]] * 2), np.stack([v[0]] * 2)
    assert M.compare(same_u, same_v)[0]["status"] == "u_above_all"
    lu, lv = (a - np.log(np.exp(a).sum(-1))[:, None] for a in (u, v))
    assert M.compare(lu, lv)[0]["status"] == "u_above_all"


def test_common_density_scaling():
    """One common factor per observation preserves source responsibility and order."""
    u, v = M.toy()
    a, b = M.compare(u, v), M.compare(u - 80, v + 20)
    for x, y in zip(a, b, strict=True):
        assert x["status"] == y["status"]
        for key in ("endpoint_deltas", "delta_min", "delta_max", "crossing_weights"):
            np.testing.assert_allclose(x[key], y[key], atol=2e-14)


def test_author_and_source_swaps():
    """Reverse evidence addresses changes sign; source reversal changes lambda."""
    u, v = M.toy()
    a, b, c = M.compare(u, v)[0], M.compare(v, u)[0], M.compare(u[::-1], v[::-1])[0]
    np.testing.assert_allclose(
        a["endpoint_deltas"], -np.array(b["endpoint_deltas"]), atol=1e-15
    )
    np.testing.assert_allclose(a["delta_min"], -b["delta_max"], atol=1e-15)
    np.testing.assert_allclose(
        a["crossing_weights"], sorted(1 - np.array(c["crossing_weights"])), atol=1e-14
    )
    assert all(x["status"] == "tie_within_tolerance" for x in M.compare(u, u))


@pytest.mark.parametrize("coeff", [[1, -4, 4], [0.2, -1, 0], [1, 0, 0], [0, 0, 0]])
def test_polynomial_roots(coeff):
    """Handle quadratic tangency, linear roots and constant/tie cases explicitly."""
    for t in M.quadratic_roots(coeff):
        np.testing.assert_allclose(
            np.polynomial.polynomial.polyval(t, coeff), 0, atol=1e-15
        )


def test_exact_probability_extrema():
    """Analytic extrema cover a toy interior effect without scanning a grid."""
    u, v = M.toy()
    r = M.compare(u, v)[0]
    for key in ("min", "max"):
        t = r[f"delta_{key}_weight"]
        actual = M.mix(u, t)["means"][0] - M.mix(v, t)["means"][0]
        np.testing.assert_allclose(actual, r[f"delta_{key}"], atol=4e-16)


def test_source_likelihood_replay():
    """Recover common-unit masses whose normalized law equals inherited scoring."""
    from grain_research.construction.math_relational_diagnostic_update import read

    rng = np.random.default_rng(71)
    bank = {
        "prototypes": rng.normal(size=(64, 2, 3)) / 5,
        "codes": np.tile(np.arange(32), 2),
        "variance": 0.3,
    }
    x = rng.normal(size=(3, 2, 3)) / 5
    logs = M.source_log_mass(bank, x)
    expected = read(bank, x)["log_profile"]
    actual = logs - M.logsumexp(logs, axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, expected, atol=2e-15)


def test_profile_relabeling():
    """Changing profile coordinates together with target statistics keeps moments."""
    u, _ = M.toy()
    perm = np.random.default_rng(4).permutation(32)
    a, b = M.moments(u), M.moments(u[:, perm], M.BITS[perm])
    for key in ("numerator", "denominator", "endpoint_means"):
        np.testing.assert_allclose(a[key], b[key], atol=4e-15)


def test_cold_contract():
    """Cold numeric comparison cannot accept author questionnaires or model answers."""
    u, v = M.toy()
    record = {
        "evidence_keys": ["u/full", "v/full"],
        "source_contract": "toy/common-density",
        "log_joint_mass": np.stack([u, v]).tolist(),
    }
    assert M.cold(record)["comparisons"][0]["status"] == "crossing"
    for key in ("answer", "application_label"):
        with pytest.raises(ValueError):
            M.cold({**record, key: 1})
    for bad in (np.zeros((2, 31)), np.full((2, 32), np.nan)):
        with pytest.raises(ValueError):
            M.moments(bad)


def test_invalid_weight():
    """Do not turn an invalid mixture into a clipped reference."""
    u, _ = M.toy()
    for value in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            M.mix(u, value)
