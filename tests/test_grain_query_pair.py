#!/usr/bin/env python3
"""Source-only composition and target-denied numerical interface tests."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.vector_mechanisms.complete_pair import ARMS, BLOCKS
from grain_research.vector_mechanisms.query_baseline import source_affine
from grain_research.vector_mechanisms.query_pair import (
    fit_query_pair,
    predict_query_pair,
    scores,
)


def fixture():
    """Create an entirely synthetic five-candidate conditional observer."""
    rng = np.random.default_rng(73)
    blocks = {k: rng.normal(size=(4, 6 if k != "L" else 2)) for k in BLOCKS}
    jm, jr = rng.normal(size=(4, 5, 2, 3)), rng.normal(size=(4, 5, 2, 3))
    z0 = rng.normal(size=(4, 5))
    affine = source_affine(
        blocks["M"].reshape(4, 2, 3), blocks["R"].reshape(4, 2, 3), z0, jm, jr
    )
    phi = {a: rng.normal(size=(4, 3, 7)) for a in ARMS}
    inc, mixed = rng.normal(size=(4, 3, 5)), rng.normal(size=(4, 2, 5))
    model = fit_query_pair(blocks, phi, z0, inc, mixed, affine, np.linspace(0, 1, 5))
    return blocks, phi, model


def test_source_affine_and_complete_logits_are_composed():
    blocks, phi, model = fixture()
    p = predict_query_pair(blocks, phi, model)
    expected = (
        model["affine_intercept"]
        + blocks["M"] @ model["affine_jm"].reshape(5, -1).T
        + blocks["R"] @ model["affine_jr"].reshape(5, -1).T
    )
    np.testing.assert_allclose(
        p["affine_base"], expected - expected.mean(1, keepdims=True)
    )
    for arm in ARMS:
        np.testing.assert_allclose(
            p[arm + "_affine_full"] - p[arm + "_full"],
            np.repeat((p["affine_base"] - p["base"])[:, None], 2, 1),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            p[arm + "_affine_score"], scores(p[arm + "_affine_full"], model["values"])
        )


def test_zero_configuration_reduces_exactly_to_baseline():
    blocks, phi, model = fixture()
    p = predict_query_pair(blocks, {k: np.zeros_like(v) for k, v in phi.items()}, model)
    for arm in ARMS:
        assert not p[arm + "_singleton"].any()
        assert not p[arm + "_mixed"].any()
        np.testing.assert_allclose(
            p[arm + "_affine_full"],
            np.repeat(p["affine_base"][:, None], 2, 1),
            atol=1e-14,
        )


def test_changing_evaluation_batch_does_not_recenter_other_authors():
    blocks, phi, model = fixture()
    p = predict_query_pair(blocks, phi, model)
    altered = {k: v.copy() for k, v in blocks.items()}
    altered["M"][0] += 100
    q = predict_query_pair(altered, phi, model)
    for k in p:
        np.testing.assert_array_equal(p[k][1:], q[k][1:])
    assert not np.allclose(p["affine_base"][0], q["affine_base"][0])


def test_exchanging_partner_comments_exchanges_pair_outputs():
    blocks, phi, model = fixture()
    p = predict_query_pair(blocks, phi, model)
    q = predict_query_pair(blocks, {k: v[:, [0, 2, 1]] for k, v in phi.items()}, model)
    for arm in ARMS:
        np.testing.assert_allclose(
            p[arm + "_affine_full"][:, ::-1], q[arm + "_affine_full"]
        )


def test_source_geometry_ignores_all_evaluation_values():
    from scripts.run_grain_query_pair import source_geometry

    authors = [{"index": i} for i in range(6)]
    packets = {
        i: {
            f"{k}_{v}": np.full((2 + i, 2), i + 1.0)
            for k in ("c", "d")
            for v in range(3)
        }
        for i in range(6)
    }
    mask = np.arange(6) < 4
    expected = source_geometry(authors, packets, mask)
    for i in (4, 5):
        for v in packets[i].values():
            v[:] = np.nan
    actual = source_geometry(authors, packets, mask)
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(a, b)


def test_author_roles_not_old_cv_folds_define_membership():
    from scripts.run_grain_query_pair import membership

    authors = [{"index": i, "author_id": str(i), "fold": 0} for i in range(24)]
    design = [
        {**r, "role": "reference" if r["index"] % 6 == 0 else "evaluation"}
        for r in authors
    ]
    assert np.array_equal(np.flatnonzero(membership(authors, design)), [0, 6, 12, 18])
    with pytest.raises(ValueError):
        membership(authors[:-1] + [authors[0]], design)


def test_scores_reject_wrong_alphabet():
    with pytest.raises(ValueError):
        scores(np.zeros((2, 5)), np.ones(4))


def test_independent_runtime_opens_only_declared_numerical_files(tmp_path):
    blocks, phi, model = fixture()
    mp, ip, op = [tmp_path / n for n in ("model.npz", "inputs.npz", "output.npz")]
    np.savez_compressed(mp, **model)
    np.savez_compressed(ip, **blocks, **phi)
    forbidden = tmp_path / "withheld_endpoint.npz"
    np.savez(forbidden, do_not_read=np.array([777]))
    script = Path(__file__).resolve().parents[1] / "scripts/replay_grain_query_pair.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(mp),
            "--inputs",
            str(ip),
            "--output",
            str(op),
            "--forbid",
            str(forbidden),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "'source_denial_checks': 1" in result.stdout
    assert "'extra_denials': 0" in result.stdout
    with np.load(op) as z:
        for k, v in predict_query_pair(blocks, phi, model).items():
            np.testing.assert_array_equal(z[k], v)
