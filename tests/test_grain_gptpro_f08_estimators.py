"""Repaired F08 updates versus original estimators and weighted enumeration."""

import numpy as np

from grain_research.gptpro_foundation_v3.f08_estimators import (
    FEATURES,
    QualifiedHistory,
)
from grain_research.measurement_discovery.estimators import (
    fit_b1,
    fit_b2,
    fit_m,
    predict_one,
)


def fixture():
    """Independent synthetic source observations, used only to construct a scorer."""
    rng = np.random.default_rng(78)
    rows = {
        str(i): [{"observation": int(x)} for x in rng.integers(0, 2, 48)]
        for i in range(12)
    }
    return {"b1": fit_b1(rows), "b2": fit_b2(rows), "m": fit_m(rows)}


def test_complete_visible_replays_original_all_prefixes():
    """All three repaired estimators reduce exactly to the original visible history."""
    fitted = fixture()
    rng = np.random.default_rng(90)
    for _ in range(10):
        history = QualifiedHistory(fitted)
        prefix = []
        for y in rng.integers(0, 2, 48):
            if len(prefix) >= 2:
                values = history.predict()["probabilities"]
                state = tuple(r["observation"] for r in prefix[-2:])
                for candidate in ("b1", "b2", "m"):
                    np.testing.assert_allclose(
                        values[candidate],
                        predict_one(candidate, state, prefix, fitted),
                        atol=1e-12,
                    )
            history.observe(float(y), 1.0)
            prefix.append({"observation": int(y)})


def test_missing_adds_no_personal_statistics():
    """Transition interpolation may change predicted state but not evidence count."""
    history = QualifiedHistory(fixture())
    for p in (0.3, 0.9, 0.1, 0.8):
        history.observe(p, 0)
    assert history.predict()["status"] == "prior_only"
    np.testing.assert_array_equal(history.gram, np.zeros((4, 4)))
    assert history.counts.sum() == history.mass == 0


def test_soft_gram_retains_second_moments():
    """Expected feature outer products differ from outer product of expected features."""
    history = QualifiedHistory(fixture())
    history.observe(0.3, 1)
    history.observe(0.7, 1)
    history.observe(0.8, 0.4)
    weights = history.counts.sum(axis=1)
    expected = sum(w * np.outer(f, f) for w, f in zip(weights, FEATURES, strict=True))
    np.testing.assert_allclose(history.gram, expected)
    mean = weights @ FEATURES / weights.sum()
    assert not np.allclose(expected, weights.sum() * np.outer(mean, mean))


def test_independent_complete_exact_four_state_expectation():
    """Complete soft channel uses independent reader marginals, not source filtering."""
    history = QualifiedHistory(fixture(), mode="independent")
    for p in (0.2, 0.8, 0.7):
        history.observe(p, 1)
    expected = np.outer([0.8, 0.2], [0.2, 0.8]).ravel()
    np.testing.assert_allclose(history.counts, expected[:, None] * [0.3, 0.7])
    np.testing.assert_allclose(
        history.gram, FEATURES.T @ (expected[:, None] * FEATURES)
    )
    np.testing.assert_allclose(history.pair, np.outer([0.2, 0.8], [0.3, 0.7]))
