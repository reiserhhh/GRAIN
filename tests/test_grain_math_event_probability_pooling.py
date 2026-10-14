#!/usr/bin/env python3
"""Positive and failing worlds for probability pooling and contextual omissions."""

import ast
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction.math_event_probability_pooling import (
    barycenter_errors,
    collision,
    curve_derivatives,
    decomposition,
    kl,
    law,
    pools,
    probabilities,
    weights,
)
from grain_research.construction.math_passive_event_composition import record
from scripts.run_grain_math_event_probability_pooling import qualified_weights


@pytest.mark.parametrize("w", ([0, 1], [0.25, 0.75], [0.5, 0.5], [1, 0]))
def test_true_pooling_world_has_zero_reference_remainder(w):
    """A correct geometric-pooling world must be accepted rather than always fail."""
    a = np.array([[1, -2, 3, 0], [-1, 2, 0, 1]])
    m, g = pools(a, w)
    r = decomposition(g, a, w, np.linspace(0, 1, 4))
    assert r["fixed_reference_residual"] == pytest.approx(0)
    assert r["native_geometric_tv"] == 0
    assert r["score_energy_error"] < 1e-14
    assert np.sum(m) == pytest.approx(1)


def test_known_multiclass_nonlinearity_can_leave_scalar_interval():
    """Outside scalar endpoints does not alone establish contextual composition."""
    a = np.log([[0.7, 0.2, 0.1], [0.601, 0.398, 0.001]])
    v = np.array([0, 0.5, 1])
    ends = probabilities(a) @ v
    m, g = pools(a, [0.5, 0.5])
    assert ends == pytest.approx([0.2, 0.2])
    assert m @ v == pytest.approx(0.2)
    assert g @ v < 0.17


def test_binary_exponential_path_stays_between_endpoints():
    """The multiclass counterexample must not be generalized to binary logits."""
    a = np.array([[-3, 1], [2, -1]])
    ends = probabilities(a) @ np.array([0, 1])
    for w in ([0.25, 0.75], [0.5, 0.5], [0.75, 0.25]):
        _, g = pools(a, w)
        assert min(ends) < g[1] < max(ends)


def test_both_kl_objective_identities():
    """Mixture and reverse-KL barycenters are different, exact reference objects."""
    a = np.array([[1, -2, 3, 0], [-1, 2, 0, 1], [1, 1, -1, 2]])
    r = barycenter_errors(a, [0.2, 0.3, 0.5], probabilities([0.1, 1, 2, -3]))
    assert r["mixture_kl_identity_error"] < 1e-14
    assert r["geometric_kl_identity_error"] < 1e-14
    assert r["mixture_objective_gap"] > 0
    assert r["geometric_objective_gap"] > 0


def test_logit_gauge_does_not_change_pool():
    """Per-event constants are unobservable candidate-logit gauge changes."""
    a = np.array([[1, -2, 3, 0], [-1, 2, 0, 1]])
    first = pools(a, [0.25, 0.75])
    second = pools(a + np.array([[100], [-30]]), [0.25, 0.75])
    for x, y in zip(first, second):
        np.testing.assert_allclose(x, y, atol=1e-14)


def test_answer_coordinate_permutation_is_not_a_mechanism():
    """A common permutation of logits and answer values preserves every score."""
    a = np.array([[1, -2, 3, 0], [-1, 2, 0, 1]])
    n, v, ix = probabilities([2, 1, 0, -1]), np.linspace(0, 1, 4), [2, 0, 3, 1]
    r, s = (
        decomposition(n, a, [0.3, 0.7], v),
        decomposition(n[ix], a[:, ix], [0.3, 0.7], v[ix]),
    )
    for k in ("native", "mixture", "geometric", "probability_cross"):
        assert r[k] == pytest.approx(s[k])


@pytest.mark.parametrize("t", [0.2, 0.5, 0.75])
def test_derivatives_match_an_independent_finite_difference(t):
    """Path curvature is a categorical moment, not an optimized predictor."""
    x, y, v = np.array([1, -2, 3, 0]), np.array([-1, 2, 0, 1]), np.linspace(0, 1, 4)
    d = curve_derivatives(x, y, t, v)
    step = 1e-4
    plus, minus = (
        float(probabilities(s * x + (1 - s) * y) @ v) for s in (t + step, t - step)
    )
    assert d["first"] == pytest.approx((plus - minus) / (2 * step), abs=1e-8)
    assert d["second"] == pytest.approx(
        (plus - 2 * d["score"] + minus) / step**2, abs=1e-7
    )


def test_signed_cross_term_cannot_be_discarded():
    """Reference discrepancies are not disjoint explained-variance percentages."""
    a = np.log([[0.7, 0.2, 0.1], [0.601, 0.398, 0.001]])
    r = decomposition(probabilities([0, 0, 0]), a, [0.5, 0.5], [0, 0.5, 1])
    assert r["score_cross"] < 0
    assert r["score_energy_error"] < 1e-14
    assert r["probability_energy_error"] < 1e-14


def test_context_shift_has_same_full_summary_and_nonzero_floor():
    """Even a nonlinear readout of an identical entire law cannot distinguish it."""
    p = np.array([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])
    n = np.array([[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]])
    r = collision(n, p, p, p, [0, 0.5, 1], 1e-13)
    assert r["same_summary"]
    assert r["minimum_common_profile_rms"] > 0
    midpoint = (n + p) / 2
    v = np.array([0, 0.5, 1])
    for z in (midpoint, p, n, np.full((2, 3), 1 / 3)):
        loss = np.sqrt((np.sum(((n - z) @ v) ** 2) + np.sum(((p - z) @ v) ** 2)) / 2)
        assert loss >= r["minimum_common_profile_rms"] - 1e-14
    assert np.linalg.norm((n - midpoint) @ v) == pytest.approx(
        r["minimum_common_profile_rms"]
    )


def test_noncollision_does_not_get_a_fake_lower_bound():
    """Different summaries may support different outputs; do not reject them."""
    p, q = np.array([[0.2, 0.5, 0.3]]), np.array([[0.4, 0.5, 0.1]])
    r = collision(p, q, p, q, [0, 0.5, 1], 1e-13)
    assert not r["same_summary"]
    assert r["minimum_common_profile_rms"] is None


def test_identical_summary_and_output_have_zero_floor():
    """The collision certificate can genuinely accept an invariant world."""
    p = np.array([[0.2, 0.5, 0.3]])
    r = collision(p, p, p, p, [0, 0.5, 1], 1e-13)
    assert r["same_summary"]
    assert r["minimum_common_profile_rms"] == 0


def test_alias_new_and_other_have_different_evidence_meanings():
    """Preserve event identity and narrative emphasis rather than deleting context."""
    first, other = record("r1", "K1", 0, 1), record("r2", "K2", 1, 0)
    alias = record("r3", "K1", 0, 1, kind="alias", parent="r1")
    baseline = {"history": [first, other]}
    repeated = {"history": [first, alias, other]}
    added_other = {"history": [first, record("r3", "K3", 0, 1, "B"), other]}
    assert qualified_weights(baseline, "event") == qualified_weights(repeated, "event")
    assert qualified_weights(baseline, "event") == qualified_weights(
        added_other, "event"
    )
    assert qualified_weights(baseline, "mention") != qualified_weights(
        repeated, "mention"
    )
    assert qualified_weights(repeated, "mention") == qualified_weights(
        {"history": list(reversed(repeated["history"]))}, "mention"
    )


def test_no_neural_loader_in_cpu_runner():
    """The new runner cannot silently perform another inference or fitting stage."""
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_grain_math_event_probability_pooling.py"
    )
    tree = ast.parse(path.read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    }
    assert not any(
        n and n.split(".")[0] in {"torch", "transformers", "sentence_transformers"}
        for n in imports
    )


@pytest.mark.parametrize(
    "bad", [[], [float("nan"), 1], [[1]], [[float("inf"), 0]], [1000, -1000]]
)
def test_invalid_logits_are_not_clipped_to_fake_evidence(bad):
    """Missing, nonfinite and underflowing laws fail visibly."""
    with pytest.raises(ValueError):
        probabilities(bad)


@pytest.mark.parametrize("bad", [[-0.1, 1.1], [0.2, 0.2], [float("nan"), 1], [1]])
def test_invalid_weights_are_not_fitted_or_renormalized(bad):
    """A broken design is not repaired by choosing favorable empirical weights."""
    with pytest.raises(ValueError):
        weights(bad, 2)


def test_invalid_probability_and_alphabet_fail():
    """KL requires positive common support and normalized laws."""
    with pytest.raises(ValueError):
        law([0, 1])
    with pytest.raises(ValueError):
        kl([0.5, 0.5], [0.2, 0.3, 0.5])
    with pytest.raises(ValueError):
        curve_derivatives([0, 1], [1, 0], 2, [0, 1])
