"""Packet 33 algebraic fixtures and output checks (implementation only; not scientific support)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import ratio_sets as R  # noqa: E402

OUT = ROOT / "results/grain_gptpro_foundation_v3/F01/ratio_inference_r1"
T = stats.t.ppf(0.975, 29)


def indep_set(a, b, Vaa, Vab, Vbb, t):
    """Independent calculation: numpy roots + sign sampling of f(r) = (a-rb)^2 - t^2(Vaa-2rVab+r^2Vbb)."""
    f = lambda r: (a - r * b) ** 2 - t * t * (Vaa - 2 * r * Vab + r * r * Vbb)
    coef = [b * b - t * t * Vbb, -2 * (a * b - t * t * Vab), a * a - t * t * Vaa]
    rts = sorted(float(x.real) for x in np.roots(coef) if abs(x.imag) < 1e-9) if any(abs(c) > 0 for c in coef) else []
    pts = [-math.inf] + rts + [math.inf]
    out = []
    for lo, hi in zip(pts[:-1], pts[1:]):
        mid = (lo + hi) / 2 if math.isfinite(lo) and math.isfinite(hi) else (hi - 1e6 if math.isfinite(hi) else (lo + 1e6 if math.isfinite(lo) else 0.0))
        if f(mid) <= 0:
            out.append((lo, hi))
    for x in rts:
        if abs(f(x)) < 1e-9 and not any(lo <= x <= hi for lo, hi in out):
            out.append((x, x))
    return R._norm(out)


def same(U, W, tol=1e-7):
    return len(U) == len(W) and all((lo1 == lo2 or abs(lo1 - lo2) <= tol * max(1, abs(lo1))) and (hi1 == hi2 or abs(hi1 - hi2) <= tol * max(1, abs(hi1))) for (lo1, hi1), (lo2, hi2) in zip(U, W))


def test_bounded_case():
    U = R.fieller(2.0, 1.0, 0.01, 0.0, 0.01, T)
    assert R.classify(U) == "bounded" and R.contains(U, 2.0)


def test_weak_denominator_two_rays():
    U = R.fieller(1.0, 0.05, 0.01, 0.0, 0.04, T)            # |t_b| = .25 < T, strong numerator
    assert R.classify(U) == "two_rays"


def test_whole_line():
    U = R.fieller(0.05, 0.05, 0.04, 0.0, 0.04, T)           # both weak
    assert R.classify(U) == "whole_line"


def test_degenerate_point_linear_empty():
    assert R.fieller(2.0, 1.0, 0.0, 0.0, 0.0, T) == [(2.0, 2.0)]
    Vbb = 1.0 / (T * T)                                        # A = 1 - t^2 Vbb = 0 exactly
    U = R.fieller(1.0, 1.0, 0.0, 0.0, Vbb, T)
    assert R.classify(U) == "ray"
    assert R.fieller(1.0, 0.0, 0.0, 0.0, 0.0, T) == []
    assert R.fieller(0.0, 0.0, 0.0, 0.0, 0.0, T) == [(-math.inf, math.inf)]


def test_calibration_negative_and_zero():
    U = [(1.0, 3.0)]
    th, fl = R.map_affine(U, -2.0, 1.0)
    assert fl == "c_negative" and th == [(-1.0, 0.0)]
    th2, fl2 = R.map_affine([(-math.inf, 0.0), (2.0, math.inf)], -1.0, 0.0)
    assert R.classify(th2) == "two_rays" and th2 == [(-math.inf, -2.0), (0.0, math.inf)]
    assert R.map_affine(U, 0.0, 2.0) == ([(-math.inf, math.inf)], "c_zero") and R.map_affine(U, 0.0, 5.0) == ([], "c_zero")


def test_union_intersection():
    two = [(-math.inf, -1.0), (1.0, math.inf)]
    assert R.intersect(two, [(-3.0, 3.0)]) == [(-3.0, -1.0), (1.0, 3.0)]
    assert R.intersect(two, [(-0.5, 0.5)]) == []


def test_membership_and_independent_agreement_random():
    rng = np.random.default_rng(0)
    for _ in range(400):
        a, b = rng.normal(size=2) * rng.choice([0.1, 1, 3]); L = rng.normal(size=(2, 2)) * rng.choice([0.05, 0.3, 1]); V = L @ L.T
        U = R.fieller(a, b, V[0, 0], V[0, 1], V[1, 1], T)
        W = indep_set(a, b, V[0, 0], V[0, 1], V[1, 1], T)
        assert same(U, W), (a, b, V, U, W)
        for r in rng.normal(size=20) * 5:
            assert R.contains(U, r) == R.in_fieller(r, a, b, V[0, 0], V[0, 1], V[1, 1], T) or min(abs(r - x) for iv in U for x in iv if math.isfinite(x)) < 1e-6


def test_output_checks():
    res = json.loads((OUT / "results.json").read_text())
    c = res["checks"]
    assert c["independent_set_agreement_failures"] == 0 and c["membership_disagreements"] == 0
    assert c["all_authors_retained"] and c["provenance_metric_max_abs"] < 1e-12
    S = pd.read_csv(OUT / "per_author_sets.csv")
    assert S.groupby(["seed", "pair", "calibration", "endpoint"]).size().eq(400).all()
