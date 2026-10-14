import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cov", ROOT / "scripts/run_grain_r29_covariance_identification.py")
cov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cov)


def test_blind_fit_refuses_author_variance_without_assumption():
    mom = cov.population_moments(cov.params("W2"), anchor=False)
    with pytest.raises(cov.NonIdentified):
        cov.blind_fit(mom, request=("sa2",))
    out = cov.blind_fit(mom)
    assert "sa2" not in out and "sa2_A0" not in out


def test_anchor_fit_requires_anchor_channel():
    with pytest.raises(cov.NonIdentified):
        cov.anchor_fit(cov.population_moments(cov.params("W1"), anchor=False))


def test_anchor_recovers_exact_population_parameters():
    for w in ("W1", "W2", "W3"):
        th = cov.params(w)
        f = cov.anchor_fit(cov.population_moments(th))
        assert abs(f["sa2"] - th["sa2"]) < 1e-12 and abs(f["k1"] - th["k"][0]) < 1e-12 and abs(f["k12"] - th["k12"]) < 1e-12


def test_two_parameterizations_same_moments_different_author_variance():
    th1 = cov.params("W1")
    th2 = {"sa2": 0.8, "sb": [0.3, 0.3], "k": [0.1, 0.1], "k12": 0.0, "se": [0.5, 0.5], "se0": 0.5}
    m1, m2 = cov.population_moments(th1, anchor=False), cov.population_moments(th2, anchor=False)
    assert all(abs(m1[k] - m2[k]) < 1e-12 for k in m1) and th1["sa2"] != th2["sa2"]


def test_event_aware_counts_duplicate_once_naive_does_not():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(50, 2, 2))
    dup = np.zeros_like(Y)
    dup[:, 0, 0] = 1.0  # duplicate every first-occasion reader-1 event
    aware = cov.person_stats(Y, None)
    naive = cov.person_stats(Y, None, dup)
    assert aware["k"][:, 0].max() == 2 and naive["k"][:, 0].min() == 3
    assert abs(cov.moments_from(aware)["V1"] - cov.moments_from(naive)["V1"]) > 1e-6
