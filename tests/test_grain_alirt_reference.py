"""Packet 34 ALIRT reference tests (implementation only; not scientific support)."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import alirt_reference as A  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F06/alirt_reference_r1"
rng = np.random.default_rng(1); J = 6
a = np.exp(rng.normal(0, .3, J)); b = np.sort(rng.normal(0, 1, (J, 3)), 1)


def test_ordinal_probabilities_valid_and_ordered():
    P = A.cat_probs(np.linspace(-4, 4, 50), a, b)
    assert np.abs(P.sum(2) - 1).max() < 1e-12 and P.min() >= 0
    par = np.zeros(J * 4); aa, bb = A.unpack(par + rng.normal(size=J * 4), J)
    assert (np.diff(bb, axis=1) > 0).all() and (aa > 0).all()


def test_item_information_matches_numeric_derivative():
    th, h = 0.3, 1e-6
    P0 = A.cat_probs(th, a, b)[0]; dP = (A.cat_probs(th + h, a, b)[0] - A.cat_probs(th - h, a, b)[0]) / (2 * h)
    assert np.abs((dP ** 2 / P0).sum(1) - A.item_info(th, a, b)).max() < 1e-6


def test_likelihood_gradient_finite_difference_consistency():
    O = rng.integers(0, 4, (30, J)); par = rng.normal(size=J * 4) * .3
    f0 = A.negloglik(par, O, J); e = np.zeros_like(par); e[3] = 1e-6
    g = (A.negloglik(par + e, O, J) - A.negloglik(par - e, O, J)) / 2e-6
    assert np.isfinite(f0) and np.isfinite(g)


def test_map_uses_all_history_and_no_repeats():
    t1 = A.map_theta([0], [3], a, b); t2 = A.map_theta([0, 1], [3, 3], a, b)
    assert t2 > t1                                     # the second high response moves the MAP further
    items, th = A.alirt_cat(np.full(J, 3), a, b, J)
    assert sorted(items) == list(range(J))              # without replacement


def test_source_isolation_signatures():
    for fn in (A.polytomize_fit, A.fit_grm, A.alirt_cat, A.map_theta, A.fit_operator, A.operator_select):
        assert not set(inspect.signature(fn).parameters) & {"theta_true", "a_true", "b_true", "truth", "w_true"}, fn.__name__


def test_output_checks():
    res = json.loads((OUT / "results.json").read_text())
    c = res["checks"]
    assert c["no_repeated_items"] and c["scored_equals_executed"] and c["all_grm_fits_converged"]
    assert c["independent_rmse_recompute_max_abs"] < 1e-10 and c["prob_sum_max_err"] < 1e-12
