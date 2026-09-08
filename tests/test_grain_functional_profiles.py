#!/usr/bin/env python3
"""CPU-only mathematical checks for the M25 reusable construction."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.functional_profiles import (
    FixedAuthorCoordinates,
    FunctionalQuery,
    MatchedQueryProjection,
    profile_pair_auc,
    relative_rms_error,
)


def fixture_values():
    """Use a deterministic small matrix with both token and final coordinates."""
    r = np.random.default_rng(123)
    return (
        r,
        r.normal(size=(6, 8)),
        r.normal(size=8),
        r.normal(size=(6, 8)),
        r.normal(size=8),
    )


def test_full_function_and_parts():
    r, p0, q0, g, h = fixture_values()
    p, q = r.normal(size=p0.shape), r.normal(size=q0.shape)
    f = FunctionalQuery(p0, q0, g, h, 0.2)
    out = f.parts(p, q)
    assert out["score"] == pytest.approx(0.2 + np.sum(g * (p - p0)) + h @ (q - q0))
    assert out["score"] == pytest.approx(
        0.2 + out["mean"] + out["configuration"] + out["final"]
    )


def test_shared_mean_and_config_are_distinct():
    r, p0, q0, g, h = fixture_values()
    f = FunctionalQuery(p0, q0, g, h, 0.2)
    d = r.normal(size=8)
    out = f.parts(p0 + d, q0)
    assert out["configuration"] == pytest.approx(0, abs=1e-12)
    assert out["mean"] == pytest.approx(g.sum(0) @ d)
    d = r.normal(size=p0.shape)
    out = f.parts(p0 + d - d.mean(0), q0)
    assert out["mean"] == pytest.approx(0, abs=1e-12)


def test_compact_preserves_mean_and_final():
    r, p0, q0, g, h = fixture_values()
    p, q = r.normal(size=p0.shape), r.normal(size=q0.shape)
    full, low = (
        FunctionalQuery(p0, q0, g, h, 0.2),
        FunctionalQuery(p0, q0, g, h, 0.2, 2),
    )
    assert low.parts(p, q)["mean"] == pytest.approx(full.parts(p, q)["mean"])
    assert low.parts(p, q)["final"] == pytest.approx(full.parts(p, q)["final"])
    assert np.max(np.abs(low.config_coef.mean(0))) < 1e-12


def test_frozen_batch_invariance_and_constant_dimensions():
    r = np.random.default_rng(4)
    ref = r.normal(size=(12, 9))
    ref[:, 2] = 1
    norms = FixedAuthorCoordinates().fit(ref)
    new = r.normal(size=(10, 9))
    np.testing.assert_array_equal(norms.transform(new[:1]), norms.transform(new)[:1])
    assert np.all(norms.transform(new)[:, 2] == 0)


def test_projection_accounting_and_no_self_predictor():
    r = np.random.default_rng(9)
    z = r.normal(size=(48, 3))
    z[:, 1] += z[:, 0]
    c = [str(j % 4) for j in range(48)]
    model = MatchedQueryProjection().fit(z, c)
    values, shared, private = model.transform(z, c)
    np.testing.assert_allclose(values, shared + private)
    np.testing.assert_array_equal(np.diag(model.coef), 0)
    for q in range(3):
        assert abs(shared[:, q] @ private[:, q]) < 1e-10
    before = model.coef.copy()
    model.transform(z * 20, c)
    np.testing.assert_array_equal(model.coef, before)


def test_query_projection_condition_centering():
    x = np.arange(60.0).reshape(20, 3)
    c = ["a"] * 10 + ["b"] * 10
    m = MatchedQueryProjection().fit(x, c)
    z, _, _ = m.transform(x, c)
    np.testing.assert_allclose(z[:10].mean(0), 0, atol=1e-12)
    with pytest.raises(ValueError, match="Uncalibrated"):
        m.transform(x[:1], ["c"])


def test_author_pair_score_and_error():
    x = np.eye(4)
    assert profile_pair_auc(x, x)[0] == 1
    assert profile_pair_auc(x, np.ones_like(x))[0] == 0.5
    assert relative_rms_error(x, x) == 0
    assert relative_rms_error(x, np.zeros_like(x)) == 1


@pytest.mark.parametrize("rank", [0, 7])
def test_invalid_rank(rank):
    _, p0, q0, g, h = fixture_values()
    with pytest.raises(ValueError):
        FunctionalQuery(p0, q0, g, h, 0, rank)


def test_nonfinite_rejected():
    with pytest.raises(ValueError):
        FixedAuthorCoordinates().fit([[1, np.nan], [2, 3], [4, 5]])
