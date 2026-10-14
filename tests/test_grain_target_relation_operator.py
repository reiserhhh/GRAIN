#!/usr/bin/env python3
"""Target relation construction tests using synthetic paired vector observations."""

import copy

import numpy as np
import pytest

from grain_research.construction import math_target_relation_operator as M


def source(seed=242):
    """Provide every complete profile with three paired observations."""
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(32), 3)
    return rng.normal(size=(96, 2, 4)), codes


def dense_total(x, codes):
    """Use explicit centered target groups instead of the dual representation."""
    result = [
        (x[:, 0] - x[:, 0].mean(0)).T @ (x[:, 1] - x[:, 1].mean(0)) / (len(x) - 1)
    ]
    for q in range(5):
        c = []
        for a in (0, 1):
            y = x[((codes >> q) & 1) == a]
            c.append(
                (y[:, 0] - y[:, 0].mean(0)).T
                @ (y[:, 1] - y[:, 1].mean(0))
                / (len(y) - 1)
            )
        result.append(c[1] - c[0])
    return np.asarray(result)


def test_dense_and_component_closure():
    x, codes = source()
    b = M.compile_bank(x, codes)
    full = np.einsum("ia,mqij,jb->mqab", x[:, 0], b["dual"], x[:, 1], optimize=True)
    np.testing.assert_allclose(full[0], dense_total(x, codes), atol=1e-14)
    np.testing.assert_allclose(full[0], full[1] + full[2], atol=1e-14)
    gram = M.operator_gram(b, b)
    np.testing.assert_allclose(
        gram, np.einsum("mqab,nrab->mqnr", full, full), atol=1e-13
    )


def test_zero_singleton_means_but_target_relation_exists():
    codes = np.repeat(np.arange(32), 3)
    r = np.tile([-1.0, 0.0, 1.0], 32)
    x = np.zeros((96, 2, 2))
    x[:, 0, 0] = r
    x[:, 1, 1] = r * (2 * (codes & 1) - 1)
    b = M.compile_bank(x, codes)
    for g in range(32):
        np.testing.assert_allclose(x[codes == g].mean(0), 0, atol=0)
    full = np.einsum("ia,mqij,jb->mqab", x[:, 0], b["dual"], x[:, 1], optimize=True)
    assert full[0, 1, 0, 1] > 1
    np.testing.assert_allclose(full[0, 2:], 0, atol=1e-14)
    np.testing.assert_allclose(full[2], 0, atol=1e-14)


def test_relink_preserves_between_but_changes_within():
    x, codes = source()
    b = M.compile_bank(x, codes)
    changed = x.copy()
    changed[:, 1] = x[M.cyclic_relink(codes), 1]
    c = M.compile_bank(changed, codes)
    fields = lambda bank: np.einsum(
        "ia,mqij,jb->mqab",
        bank["means"][:, 0],
        bank["dual"],
        bank["means"][:, 1],
        optimize=True,
    )
    np.testing.assert_allclose(fields(b)[2], fields(c)[2], atol=1e-13)
    assert np.max(abs(fields(b)[1] - fields(c)[1])) > 0.01


def test_translation_and_orthogonal_frame():
    x, codes = source()
    y, _ = source(243)
    b, c = M.compile_bank(x, codes), M.compile_bank(y, codes)
    rng = np.random.default_rng(5)
    rot = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    bb, cc = M.compile_bank(x @ rot + 3, codes), M.compile_bank(y @ rot + 3, codes)
    np.testing.assert_allclose(
        M.operator_gram(b, c), M.operator_gram(bb, cc), atol=1e-12
    )
    point = x[:2]
    np.testing.assert_allclose(
        M.score_points(b, point), M.score_points(bb, point @ rot + 3), atol=1e-12
    )


def test_singleton_profile_is_recorded_not_population_variance():
    x, codes = source()
    keep = np.r_[0, np.arange(3, 96)]
    b = M.compile_bank(x[keep], codes[keep])
    assert b["profile_counts"][0] == 1
    assert b["dual"].shape == (3, 6, 94, 94)


def test_invalid_groups_rejected():
    x, codes = source()
    with pytest.raises(ValueError, match="at least two"):
        M.compile_bank(x, codes * 0)


def test_cold_direct_and_guards():
    x, codes = source()
    b = M.compile_bank(x, codes)
    record = {
        "frame_key": "new",
        "windows": [
            [
                {"event_id": f"{v}-{t}", "condition": t % 2, "vector": x[t, v].tolist()}
                for t in range(3)
            ]
            for v in (0, 1)
        ],
    }
    output = M.cold_record([b, b], record)
    point = x[:3].mean(0)
    np.testing.assert_allclose(output["scores"][0], M.score_points(b, point), atol=0)
    bad = copy.deepcopy(record)
    bad["windows"][1][0]["event_id"] = "0-0"
    with pytest.raises(ValueError, match="Distinct"):
        M.cold_record([b], bad)
    bad = copy.deepcopy(record)
    bad["label"] = 1
    with pytest.raises(ValueError, match="Only frame"):
        M.cold_record([b], bad)
