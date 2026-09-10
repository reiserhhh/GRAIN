#!/usr/bin/env python3
"""Bilinear identity, source invariance, derivatives and author-centered accounts."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import endpoint_query_contrast as f
from grain_research.vector_mechanisms import source_endpoint_assembly as previous
from tests.test_grain_source_endpoint_assembly import toy


def test_symmetric_query_terms_and_degenerate_limits():
    rng = np.random.default_rng(78)
    p, r = rng.normal(size=(2, 4, 11)), rng.normal(size=(2, 11))
    t = f.decompose(p, r)
    np.testing.assert_allclose(t[2], p[0] @ r[0] - p[1] @ r[1])
    np.testing.assert_allclose(f.decompose(p[::-1], r[::-1]), -t)
    assert np.max(abs(f.decompose(np.stack((p[0], p[0])), r)[0])) == 0
    assert np.max(abs(f.decompose(p, np.stack((r[0], r[0])))[1])) == 0
    with pytest.raises(ValueError):
        f.decompose(p[:1], r)


def test_shared_rotation_preserves_both_terms():
    rng = np.random.default_rng(780)
    p, r = rng.normal(size=(2, 3, 11)), rng.normal(size=(2, 11))
    o = np.linalg.qr(rng.normal(size=(11, 11)))[0]
    np.testing.assert_allclose(f.decompose(p @ o, r @ o), f.decompose(p, r), atol=1e-12)


def test_term_derivatives_are_from_same_local_scalars():
    data, d, w, sm, vm, r = toy()
    qs = [sm, dict(sm, anchor=sm["anchor"] * 1.1)]

    def predict(state):
        rows = [previous.predict(dict(data, state=state), d, w, s, vm, r) for s in qs]
        return f.split_packets(
            np.stack([p["primal"] for p in rows]),
            np.stack([p["vectors"] for p in rows]),
            r,
        )

    p = predict(data["state"])
    for i in range(4):
        a, b = [
            predict(data["state"] + eps * d[i])["scalar_terms"] for eps in (-1e-5, 1e-5)
        ]
        np.testing.assert_allclose(
            (b - a) / 2e-5, p["action_terms"][:, 2, :, i], atol=1e-8
        )


def test_author_centering_does_not_confuse_factor_offsets_with_variance():
    rng = np.random.default_rng(781)
    x = rng.normal(size=(8, 3, 4))
    x[:, 2] = x[:, 0] + x[:, 1]
    shift = np.array([[100, 200, 300, 400], [1, 2, 3, 4], [101, 202, 303, 404]])
    a, b = f.moments(x), f.moments(x + shift)
    for k in a:
        if "author" in k or "projection" in k:
            assert a[k] == pytest.approx(b[k], abs=1e-10)
    assert a["evidence_change_mean"] != b["evidence_change_mean"]
    assert (
        abs(a["evidence_centered_projection"] + a["reader_centered_projection"] - 1)
        < 1e-12
    )


def test_cancelling_error_and_geometry():
    x = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]], float)
    p = x + np.array([2, -1, 1])
    m = f.error_account(x, p)
    assert m["signed_cross_mse"] < 0
    assert m["total_centered_mse"] == 0
    assert m["raw_identity_error"] < 1e-12
    g = f.geometry(np.array([[1, 0], [0, 1]], float))
    assert g["reader_rank"] == 2
    assert g["mean_difference_cosine"] == 0


def test_aggregate_inventory():
    from scripts.run_grain_endpoint_query_contrast import summarize

    rng = np.random.default_rng(782)
    groups = {}
    for cohort, n in (("source4", 4), ("new8", 8)):
        rows = []
        for _ in range(n):
            s = rng.normal(size=(6, 3))
            s[:, 2] = s[:, :2].sum(axis=1)
            a = rng.normal(size=(6, 3, 3, 4))
            a[:, :, 2] = a[:, :, :2].sum(axis=2)
            ys = rng.normal(size=3)
            ys[2] = ys[:2].sum()
            ya = rng.normal(size=(3, 3, 4))
            ya[:, 2] = ya[:, :2].sum(axis=1)
            rows.append(
                {
                    "scalar_terms": s,
                    "action_terms": a,
                    "target_scalar_terms": ys,
                    "target_action_terms": ya,
                    "source_geometry": f.geometry(rng.normal(size=(2, 11))),
                }
            )
        groups[cohort] = rows
    tables = summarize(groups)
    assert [len(t) for t in tables] == [144, 432, 56, 48, 2]
    assert not any({"author_key", "text", "user_id"} & set(t.columns) for t in tables)
