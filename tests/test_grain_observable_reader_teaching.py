#!/usr/bin/env python3
"""Independent coefficient solves, functional rows, calculus and signed accounts."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import material_reader_field as base
from grain_research.vector_mechanisms import observable_reader_teaching as f


def toy():
    """Finite mixed value/derivative source with redundant output coordinates."""
    rng = np.random.default_rng(830)
    x, v = rng.normal(size=(2, 9, 4))
    groups = np.repeat(np.arange(3), 3)
    y = rng.normal(size=(9, 5)) @ rng.normal(size=(5, 11))
    m = base.fit(x, v, groups > 0, y, groups, rng.normal(size=11), rng.normal(size=11))
    m.update(target=y, groups=groups)
    return rng, m, rng.normal(size=(9, 5, 11))


def test_vector_control_matches_old_field_and_penalty():
    rng, m, r = toy()
    n = f.fit(m, r, False)
    x, dx = rng.normal(size=(2, 4))
    for a, b in zip(base.field(m, x, dx), base.field(n, x, dx), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-9)
    k = base.gram(m["x"], m["v"], m["kinds"], m["x"], m["v"], m["kinds"])
    np.testing.assert_allclose(
        np.sum(n["theta"] ** 2), np.sum(n["beta"] * (k @ n["beta"])), atol=1e-9
    )
    assert n["output_basis"].shape[1] == 5


def test_functional_solve_matches_augmented_least_squares():
    _, m, r = toy()
    n = f.fit(m, r, True)
    a, u = n["a"], n["output_basis"]
    z = [np.kron(a * m["weights"][:, None], np.eye(u.shape[1]))]
    b = [((m["target"] @ u) * m["weights"][:, None]).ravel()]
    for rr in (r[:, :1], r[:, 1:]):
        zz, bb, _ = f.functional_rows(a, u, m["target"], rr, m["groups"])
        z.append(zz)
        b.append(bb)
    z.append(np.sqrt(m["regularizer"]) * np.eye(n["theta"].size))
    b.append(np.zeros(n["theta"].size))
    theta, *_ = np.linalg.lstsq(np.concatenate(z), np.concatenate(b), rcond=None)
    np.testing.assert_allclose(n["theta"].ravel(), theta, atol=1e-9)
    old = f.fit(m, r, False)
    objective = lambda q: q["source_losses"].sum() + q["regularization_penalty"]
    assert objective(n) <= objective(old) + 1e-10


def test_zero_rulers_are_omitted_without_amplification():
    _, m, r = toy()
    r[:3] = 0
    n = f.fit(m, r, True)
    assert n["omitted_functional_groups"][:, 0].all()
    assert np.isfinite(n["theta"]).all()
    with pytest.raises(ValueError):
        f.functional_rows(
            n["a"], n["output_basis"], m["target"], r * np.nan, m["groups"]
        )


def test_functional_field_derivative_and_range_account():
    rng, m, r = toy()
    n = f.fit(m, r, True)
    x, dx = rng.normal(size=(2, 4))
    d, dd = base.field(n, x, dx)
    eps = 1e-5
    np.testing.assert_allclose(
        (base.field(n, x + eps * dx) - base.field(n, x - eps * dx)) / (2 * eps),
        dd[0],
        atol=1e-8,
    )
    target = rng.normal(size=(1, 11))
    inside, outside = f.split_error(d[None], target, n["d0"], n["output_basis"])
    vector = f.error_account(inside, outside)
    assert abs(vector["signed_cross"]) < 1e-10
    assert vector["closure_error"] < 1e-10
    ruler = rng.normal(size=11)
    projected = f.error_account(inside @ ruler, outside @ ruler)
    assert abs(projected["signed_cross"]) > 1e-5
    assert projected["closure_error"] < 1e-10


def test_observation_author_binding():
    from scripts.run_grain_observable_reader_teaching import observation_authors

    a = observation_authors(np.repeat(np.arange(3), [8, 16, 4]))
    np.testing.assert_array_equal(np.bincount(a), [7] * 4)
    np.testing.assert_array_equal(a[:8], np.repeat(np.arange(4), 2))
    np.testing.assert_array_equal(a[8:24], np.repeat(np.arange(4), 4))
    np.testing.assert_array_equal(a[24:], np.arange(4))


def test_full_eight_field_inventory_and_signed_range():
    from scripts.run_grain_observable_reader_teaching import COUNTS, previous, summarize

    rng, m, r = toy()
    ms = [f.fit(m, r, flag) for flag in (False, True)]
    rows = []
    for _ in range(8):
        readers = rng.normal(size=(8, 2, 11))
        for vi, model in enumerate(ms, 6):
            d = (
                model["d0"]
                + rng.normal(size=model["output_basis"].shape[1])
                @ model["output_basis"].T
            )
            readers[vi] = np.stack((model["m"] + d / 2, model["m"] - d / 2))
        dr = rng.normal(size=(8, 2, 4, 11))
        ps = rng.normal(size=(2, 4, 11))
        vs = rng.normal(size=(2, 4, 3, 4, 11))
        packets = [
            previous.projections(ps, vs, rr, dd)
            for rr, dd in zip(readers, dr, strict=True)
        ]
        row = {k: np.stack([p[k] for p in packets]) for k in packets[0]}
        row.update(
            readers=readers,
            reader_derivatives=dr,
            native_r=rng.normal(size=(2, 11)),
            native_primal=rng.normal(size=(2, 11)),
            native_vectors=rng.normal(size=(2, 3, 4, 11)),
            target_scalar=rng.normal(size=(2, 2)),
            target_actions=rng.normal(size=(2, 3, 2, 4)),
        )
        rows.append(row)
    tables = summarize(rows, ms)
    assert [len(t) for t in tables] == COUNTS
    assert tables[-1].closure_error.max() < 1e-8
    assert set(tables[2]["query"]) == {
        "neuroticism",
        "confirmation_seeking",
        "N_minus_confirmation",
    }
