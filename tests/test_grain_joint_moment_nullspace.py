#!/usr/bin/env python3
"""Current-moment null spaces, paired gradient geometry and cohort reporting."""

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms import joint_moment_nullspace as f
from grain_research.vector_mechanisms import material_reader_field as features


def toy(seed=900):
    """Generate finite fixed-pattern BODY fields without real author material."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(17, 4))
    ds, j = rng.normal(size=(4, 17, 7)), rng.normal(size=(2, 17, 7))
    u = np.linalg.qr(rng.normal(size=(7, 3)))[0]
    return rng, g, ds, j, u


def test_projection_is_orthogonal_idempotent_and_nested():
    _, g, _, j, u = toy()
    a, q, geom = f.basis(g)
    energies = []
    for method in f.MAPS:
        v = f.project(j[0], q, u, method)
        np.testing.assert_allclose(f.project(v, q, u, method), v, atol=1e-12)
        assert np.sum(v * (j[0] - v)) == pytest.approx(0, abs=1e-12)
        energies.append(np.sum(v**2))
    assert np.all(np.diff(energies) >= 0)
    np.testing.assert_allclose(a.T @ (j[0] - v), 0, atol=1e-12)
    assert geom["rank"] == 5 and geom["null_token_dimension"] == 12


def test_null_change_preserves_actual_current_features_at_any_amplitude():
    rng, g, _, j, u = toy()
    _, q, _ = f.basis(g)
    null = j[0] - f.project(j[0], q, u, "grain_moments")
    body = rng.normal(size=(17, 7))
    states = [np.vstack((body, rng.normal(size=7))) for _ in range(2)]
    data = [{"state": s, "positions": np.arange(17), "grain": g} for s in states]
    original = features.features(data)
    for amplitude in (-3.0, 0.2, 5.0):
        changed = [
            d | {"state": d["state"] + np.vstack((amplitude * null, np.zeros(7)))}
            for d in data
        ]
        for a, b in zip(original, features.features(changed), strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12)


def test_nonlinearity_after_the_map_cannot_recover_a_null_direction():
    rng, g, _, j, u = toy()
    a, q, _ = f.basis(g)
    null = j[0] - f.project(j[0], q, u, "grain_moments")
    x, weight = rng.normal(size=(17, 7)), rng.normal(size=(5, 7))

    def score(body):
        return np.tanh((a.T @ body) * weight).sum()

    assert score(x + null) == pytest.approx(score(x - null), abs=1e-11)
    assert np.sum(j[0] * null) == pytest.approx(np.sum(null**2), abs=1e-11)
    assert np.linalg.norm(null) > 0


def test_cached_gradient_null_witness_is_target_dependent_optimum():
    rng, g, _, j, u = toy()
    _, q, _ = f.basis(g)
    residual = j[0] - f.project(j[0], q, u, "grain_moments")
    optimum = np.linalg.norm(residual)
    assert np.sum(j[0] * (residual / optimum)) == pytest.approx(optimum)
    d = rng.normal(size=residual.shape)
    d -= f.project(d, q, u, "grain_moments")
    assert abs(np.sum(j[0] * d / np.linalg.norm(d))) <= optimum + 1e-12


def test_all_signed_action_contractions_match_explicit_direction_projection():
    _, g, ds, j, u = toy()
    rows, checks = f.analyze(g, ds, j, u)
    _, q, _ = f.basis(g)
    for row in rows["responses"]:
        qi = f.VIEWS.index(row["query"])
        grad = (j[0], j[1], j.mean(axis=0), j[0] - j[1])[qi]
        d = ds[int(row["factor"][1:]) - 1]
        visible = f.project(d, q, u, row["map"])
        assert row["visible"] == pytest.approx(np.sum(grad * visible), abs=1e-11)
        assert row["null"] == pytest.approx(np.sum(grad * (d - visible)), abs=1e-11)
    assert max(checks.values()) < 1e-11


@pytest.mark.parametrize("sign,expected", [(1, 1), (-1, 0)])
def test_query_common_fraction_distinguishes_same_and_opposing_fields(sign, expected):
    _, g, ds, j, u = toy()
    rows, _ = f.analyze(g, ds, np.stack((j[0], sign * j[0])), u)
    for row in rows["query_geometry"]:
        assert row["common_fraction"] == pytest.approx(expected, abs=1e-12)
        assert row["cosine"] == pytest.approx(sign, abs=1e-12)


def test_rank_deficiency_and_zero_fields_are_retained():
    _, g, ds, j, u = toy()
    rows, _ = f.analyze(np.zeros_like(g), ds, np.zeros_like(j), u)
    assert rows["input_geometry"][0]["rank"] == 1
    assert all(np.isnan(r["null_fraction"]) for r in rows["gradient_geometry"])
    assert all(np.isnan(r["cosine"]) for r in rows["query_geometry"])
    assert all(r["optimal_null_response"] == 0 for r in rows["gradient_geometry"])


def test_zero_linear_moment_does_not_force_zero_power_association():
    # z is orthogonal to both the intercept and g, while z squared is not.
    g = np.column_stack(([-2.0, -1.0, 1.0, 2.0], np.zeros((4, 3))))
    z = np.array([1.0, -1.0, -2.0, 2.0])
    a, q, _ = f.basis(g)
    z -= q @ (q.T @ z)
    np.testing.assert_allclose(a.T @ z, 0, atol=1e-12)
    assert abs(f.cosine(g[:, 0], z**2 - np.mean(z**2))) > 0.1


def test_cohort_tables_keep_four_factors_and_author_not_token_counts():
    raw = {k: [] for k in f.TABLES}
    for i in range(12):
        _, g, ds, j, u = toy(900 + i)
        rows, _ = f.analyze(g, ds, j, u)
        for k, values in rows.items():
            raw[k].extend(
                {"cohort": "source4" if i < 4 else "new8", "author_key": f"unit{i}"} | v
                for v in values
            )
    tables = f.aggregate({k: pd.DataFrame(v) for k, v in raw.items()})
    assert [len(tables[k]) for k in f.TABLES] == f.COUNTS
    for t in tables.values():
        assert set(t.authors) == {4, 8} and "author_key" not in t
    actions = tables["responses"]
    assert set(actions.factor) == {"all4", "F01", "F02", "F03", "F04"}
    np.testing.assert_allclose(
        actions.null_squared_ratio
        + actions.visible_squared_ratio
        + actions.signed_cross_ratio,
        1,
        atol=1e-10,
    )


def test_nonfinite_and_misaligned_inputs_rejected():
    _, g, ds, j, u = toy()
    with pytest.raises(ValueError):
        f.basis(np.full_like(g, np.nan))
    with pytest.raises(ValueError):
        f.analyze(g, ds[:3], j, u)
    with pytest.raises(AssertionError):
        f.analyze(g, ds, j, 2 * u)
