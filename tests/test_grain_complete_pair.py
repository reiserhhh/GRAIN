#!/usr/bin/env python3
"""Numerical and source-denial tests for the complete conditional interface."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.vector_mechanisms.complete_pair import (
    ARMS,
    context_kernel,
    error_account,
    fit_complete,
    fit_context,
    predict_complete,
    singleton_kernel,
)
from grain_research.vector_mechanisms.joint_readout import centered_logits, fit_joint


def fixture():
    """Create small source data without text, users or private endpoints."""
    r = np.random.default_rng(2601)
    blocks = {k: r.normal(size=(8, d)) for k, d in (("M", 7), ("R", 9), ("L", 4))}
    phi = {a: r.normal(size=(8, 3, 6)) for a in ARMS}
    z0, inc = r.normal(size=(8, 5)), r.normal(size=(8, 3, 5))
    mixed = {
        a: fit_joint(p[:, 0], p[:, 1], centered_logits(r.normal(size=(8, 5))))
        for a, p in phi.items()
    }
    return (
        blocks,
        phi,
        fit_complete(blocks, phi, z0, inc, mixed, values=np.linspace(0, 1, 5)),
    )


def test_context_source_center_and_scale():
    b, _, _ = fixture()
    m = fit_context(b)
    k = context_kernel(b, m)
    np.testing.assert_allclose(k.mean(0), 0, atol=1e-15)
    np.testing.assert_allclose(np.trace(k) / len(k), 1)
    frozen = {key: np.array(v, copy=True) for key, v in m.items()}
    context_kernel({key: value + 100 for key, value in b.items()}, m)
    for key in m:
        np.testing.assert_array_equal(m[key], frozen[key])


def test_constant_blocks():
    b = {key: np.full((6, 2), 0.1) for key in ("M", "R", "L")}
    m = fit_context(b)
    np.testing.assert_array_equal(context_kernel(b, m), np.zeros((6, 6)))
    assert all(bool(m[f"{k}_constant"]) for k in b)


def test_zero_singleton():
    k = np.ones((2, 3))
    out = singleton_kernel(k, np.zeros((2, 4)), np.ones((3, 4)))
    np.testing.assert_array_equal(out, np.zeros((2, 3)))
    b, p, m = fixture()
    out = predict_complete(b, {a: np.zeros_like(v) for a, v in p.items()}, m)
    for a in ARMS:
        np.testing.assert_array_equal(out[f"{a}_singleton"], 0)
        np.testing.assert_array_equal(out[f"{a}_mixed"], 0)
        np.testing.assert_allclose(out[f"{a}_full"][:, 0], out["base"])


def test_swap_and_target_algebra():
    b, p, m = fixture()
    one = predict_complete(b, p, m)
    two = predict_complete(b, {a: v[:, [1, 0, 2]] for a, v in p.items()}, m)
    for a in ARMS:
        np.testing.assert_allclose(
            one[f"{a}_full"][:, 0], two[f"{a}_full"][:, 0], atol=1e-14
        )
        np.testing.assert_allclose(
            one[f"{a}_full"], one[f"{a}_additive"] + one[f"{a}_mixed"]
        )
        z = one[f"{a}_full"]
        probabilities = np.exp(z - z.max(-1, keepdims=True))
        expected = (probabilities @ np.linspace(0, 1, 5)) / probabilities.sum(-1)
        np.testing.assert_allclose(one[f"{a}_score"], expected)


def test_signed_error_account():
    r = np.random.default_rng(1)
    terms = error_account(
        {k: r.normal(size=(5, 6)) for k in ("base", "i", "j", "mixed")}
    )
    np.testing.assert_allclose(terms["identity_error"], 0, atol=2e-15)
    assert any(np.any(v < 0) for k, v in terms.items() if k.startswith("cross"))


def test_bad_context():
    b, _, m = fixture()
    b["L"] = np.ones((3, 2))
    with pytest.raises(ValueError):
        context_kernel(b, m)


def test_source_author_partition():
    from scripts.run_grain_complete_pair import source_partition

    rows = [{"author_id": str(i), "fold": i % 4} for i in range(24)]
    for fold in range(4):
        mask = source_partition(rows, fold)
        assert mask.sum() == 18
        assert not any(mask[i] for i, row in enumerate(rows) if row["fold"] == fold)
    with pytest.raises(ValueError):
        source_partition(rows + rows[:1], 0)


def test_source_denied_replay(tmp_path):
    b, p, m = fixture()
    np.savez(tmp_path / "model.npz", **m)
    np.savez(tmp_path / "input.npz", **b, **p)
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/replay_grain_complete_pair.py"),
        "--model",
        str(tmp_path / "model.npz"),
        "--inputs",
        str(tmp_path / "input.npz"),
        "--output",
        str(tmp_path / "out.npz"),
        "--forbid",
        str(__file__),
    ]
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "'source_denial_checks': 1" in run.stdout
    with np.load(tmp_path / "out.npz") as z:
        for k, v in predict_complete(b, p, m).items():
            np.testing.assert_array_equal(z[k], v)
