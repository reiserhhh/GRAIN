#!/usr/bin/env python3
"""Endpoint projection, complete derivatives, source exclusion and accounting."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import source_endpoint_assembly as f
from grain_research.vector_mechanisms.selection_endpoint import source_reader
from tests.test_grain_source_value_channel import setup


def toy():
    """Attach a rectangular output matrix and two endpoint readers to old toys."""
    data, d, w, sm, vm, old = setup(77)
    rng = np.random.default_rng(777)
    w["o"] = rng.normal(size=(11, np.prod(old["g"].shape[1:])))
    r = f.readers(rng.normal(size=(2, 11)))[0]
    return data, d, w, sm, vm, r


def test_projection_matches_original_grouped_head_readout():
    data, d, w, sm, vm, r = toy()
    v, dv = f.value.apply(data, d, w, vm)
    for s in f.value.SELECTORS:
        ell, dell = f.value.selector(data, d, w, sm, s)
        for i in range(2):
            p = f.vectors(ell, dell, data["mask"], v[i], dv[i], w["o"])
            a = f.contract(p, r)
            g = (r @ w["o"]).reshape(2, ell.shape[1], v.shape[-1])
            b = f.value.readout(ell, dell, data["mask"], v[i], dv[i], {"g": g})
            for k in a:
                np.testing.assert_allclose(a[k], b[k], atol=1e-10)


def test_full_scalar_finite_differences_and_vector_sum():
    data, d, w, sm, vm, r = toy()
    p = f.predict(data, d, w, sm, vm, r)
    for i in range(4):
        s = [
            f.predict(dict(data, state=data["state"] + e * d[i]), d, w, sm, vm, r)[
                "scalar"
            ]
            for e in (-1e-5, 1e-5)
        ]
        np.testing.assert_allclose(
            (s[1] - s[0]) / 2e-5, p["actions"][:, 2, :, i], atol=1e-8
        )
    np.testing.assert_array_equal(
        p["vectors"][:, 2], p["vectors"][:, 0] + p["vectors"][:, 1]
    )
    np.testing.assert_allclose(
        p["actions"][:, 2], p["actions"][:, 0] + p["actions"][:, 1], atol=1e-10
    )


def test_source_exclusion_and_fixed_query_mean():
    queries = ("neuroticism", "confirmation_seeking")
    bank = {
        (i, q): np.full(5, i + qi) for i in range(4) for qi, q in enumerate(queries)
    }
    for i in range(4):
        r, members = source_reader(bank, "source4", i, queries[0])
        assert i not in members
        poison = dict(bank)
        poison[i, queries[0]] = np.full(5, np.nan)
        np.testing.assert_array_equal(
            source_reader(poison, "source4", i, queries[0])[0], r
        )
    p = np.array([[1, 3], [3, 7]])
    r = f.readers(p)
    np.testing.assert_array_equal(r[:, 0], p)
    np.testing.assert_array_equal(r[:, 1], [[2, 5], [2, 5]])
    with pytest.raises(ValueError):
        f.readers(p[:1])


def test_exact_reader_transport_error_with_negative_cross_term():
    y, p, c = np.array([1.0, 2]), np.array([2.0, 4]), np.array([1.5, 3])
    r = f.error_metrics(y, p, c)
    assert r["identity_error"] < 1e-12
    assert r["signed_cross_mse"] < 0
    rng = np.random.default_rng(77)
    u, uh, r, rb = rng.normal(size=(4, 9))
    y, p, c = u @ r, uh @ r, uh @ rb
    np.testing.assert_allclose(f.account(y, p, c)["reader_error"], uh @ (rb - r))


def test_public_inventory():
    from scripts.run_grain_source_endpoint_assembly import summarize

    rng = np.random.default_rng(771)
    groups = {}
    for cohort, n in (("source4", 4), ("new8", 8)):
        groups[cohort] = []
        for _ in range(n):
            a = rng.normal(size=(2, 6, 3, 2, 4))
            a[:, :, 2] = a[:, :, 0] + a[:, :, 1]
            y = rng.normal(size=(2, 3, 2, 4))
            y[:, 2] = y[:, 0] + y[:, 1]
            groups[cohort].append(
                {
                    "actions": a,
                    "transported": a * 1.1,
                    "scalar": rng.normal(size=(2, 6, 2)),
                    "scalar_transported": rng.normal(size=(2, 6, 2)),
                    "target": y,
                    "target_scalar": rng.normal(size=(2, 2)),
                    "vectors": rng.normal(size=(2, 6, 3, 4, 11)),
                    "target_vectors": rng.normal(size=(2, 3, 4, 11)),
                }
            )
    tables = summarize(groups)
    assert [len(t) for t in tables] == [144, 576, 72, 72, 72, 288, 108]
    assert not any({"author_key", "text", "token"} & set(t.columns) for t in tables)
