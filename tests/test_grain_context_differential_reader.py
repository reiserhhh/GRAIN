#!/usr/bin/env python3
"""Conditional reader algebra, source isolation, full derivatives and range bound."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import context_differential_reader as f


def toy():
    rng = np.random.default_rng(79)
    bank = {i: (rng.normal(size=9), rng.normal(size=(2, 9))) for i in range(4)}
    return bank, f.build(bank, [0, 1, 2]), rng


def test_source_membership_and_uniform_replay():
    bank, m, rng = toy()
    bank[3] = object()
    n = f.build(bank, [0, 1, 2])
    for key in m:
        np.testing.assert_array_equal(m[key], n[key])
    r = np.mean([bank[i][1] for i in (0, 1, 2)], axis=0)
    np.testing.assert_allclose(f.read(m, rng.normal(size=9), False), r, atol=1e-15)
    with pytest.raises(ValueError):
        f.build(bank, [1, 1])


def test_context_derivative_and_complete_bilinear_product():
    _, m, rng = toy()
    c, dc = rng.normal(size=(2, 9))
    p, dp = rng.normal(size=(2, 2, 9))
    eps = 1e-5
    a, b = [f.read(m, c + e * dc) for e in (-eps, eps)]
    dr = f.derivative(m, c, dc)
    np.testing.assert_allclose((b - a) / (2 * eps), dr, atol=1e-10)
    r = f.read(m, c)
    expected = np.sum(r * dp + dr * p, axis=1)
    actual = (
        np.sum(b * (p + eps * dp), axis=1) - np.sum(a * (p - eps * dp), axis=1)
    ) / (2 * eps)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_array_equal(f.derivative(m, c, np.zeros(9)), np.zeros((2, 9)))
    assert np.linalg.norm(dr) > 0


def test_degenerate_context_and_extreme_weights():
    bank, _, rng = toy()
    m = f.build({i: (np.zeros(9), bank[i][1]) for i in range(3)}, range(3))
    assert f.diagnostics(m, np.ones(9))["degenerate_support"] == 1
    np.testing.assert_allclose(f.weights(m, np.ones(9))[0], np.ones(3) / 3)
    np.testing.assert_array_equal(
        f.derivative(m, np.ones(9), np.ones(9)), np.zeros((2, 9))
    )
    _, m, _ = toy()
    p, _ = f.weights(m, rng.normal(size=9) * 1e5)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1)


def test_rotation_equivariance_and_query_swap():
    bank, m, rng = toy()
    o = np.linalg.qr(rng.normal(size=(9, 9)))[0]
    c = rng.normal(size=9)
    n = f.build({i: (x @ o, r @ o) for i, (x, r) in bank.items()}, [0, 1, 2])
    np.testing.assert_allclose(f.read(n, c @ o), f.read(m, c) @ o, atol=1e-12)
    n = f.build({i: (x, r[::-1]) for i, (x, r) in bank.items()}, [0, 1, 2])
    np.testing.assert_allclose(f.read(n, c), f.read(m, c)[::-1], atol=1e-12)


def test_range_bound_and_projected_error_need_not_track():
    _, m, rng = toy()
    c, target = rng.normal(size=(2, 9))
    pair = f.read(m, c)
    p = pair[0] - pair[1]
    a = f.range_account(m, target, p)
    assert a["rank"] == 2
    assert a["orthogonal_lower_bound2"] <= a["vector_error2"]
    assert a["vector_error2"] == pytest.approx(
        a["orthogonal_lower_bound2"] + a["in_span_error2"]
    )
    u = rng.normal(size=9)
    error = p - target
    u -= (u @ error) / (error @ error) * error
    assert abs(u @ error) < 1e-12 and a["vector_error2"] > 0


def test_projection_modes_and_term_identity():
    _, m, rng = toy()
    p, u = rng.normal(size=(2, 4, 9)), rng.normal(size=(2, 4, 3, 4, 9))
    r = f.read(m, rng.normal(size=9))
    x = f.project(p, u, r)
    for mode, term in ((0, 2), (1, 0)):
        np.testing.assert_allclose(
            x["scalar"][0, :, mode] - x["scalar"][1, :, mode],
            x["scalar_terms"][:, term],
        )
        np.testing.assert_allclose(
            x["actions"][0, :, :, mode] - x["actions"][1, :, :, mode],
            x["action_terms"][:, :, term],
        )


def test_complete_aggregate_grid():
    from scripts.run_grain_context_differential_reader import COUNTS, summarize

    _, m, rng = toy()
    groups = {}
    for cohort, n in (("source4", 4), ("new8", 8)):
        rows = []
        for _ in range(n):
            c = rng.normal(size=9)
            r = np.stack([f.read(m, c, bool(v)) for v in range(2)])
            p, u = rng.normal(size=(2, 4, 9)), rng.normal(size=(2, 4, 3, 4, 9))
            packets = [f.project(p, u, rv) for rv in r]
            row = {k: np.stack([x[k] for x in packets]) for k in packets[0]}
            rc = rng.normal(size=(2, 9))
            target = f.project(p[:, :1], u[:, :1], rc)
            row.update(
                {
                    "target_" + k: (v[:, 0] if k in ("scalar", "actions") else v[0])
                    for k, v in target.items()
                }
            )
            rd = r[:, 0] - r[:, 1]
            row["projected_scalar"] = rd @ p.mean(axis=0)[0]
            row["projected_actions"] = np.einsum("cfh,vh->vcf", u.mean(axis=0)[0], rd)
            row["ranges"] = [f.range_account(m, rc[0] - rc[1], d) for d in rd]
            row["diagnostics"] = f.diagnostics(m, c)
            rows.append(row)
        groups[cohort] = rows
    tables = summarize(groups)
    assert [len(t) for t in tables] == COUNTS
    assert not any({"text", "author_key", "user_id"} & set(t.columns) for t in tables)
    assert tables[4].raw_identity_error.max() < 1e-10
