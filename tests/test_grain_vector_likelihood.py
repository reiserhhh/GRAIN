#!/usr/bin/env python3
"""Finite likelihood construction, source-coordinate and cold evidence tests."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_vector_likelihood as M

ROOT = Path(__file__).resolve().parents[1]


def world():
    """Return noncollinear observed prototypes and five nonconstant anchors."""
    z = np.array([[-1.0, 0], [1, 0], [0, -2], [0, 2], [1, 1], [-1, -1]])
    y = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
        ]
    )
    return z, y, 1.3


def test_positive_joint_profile_and_marginals():
    z, y, s = world()
    r = M.read(z, y, s, z / 2)
    assert np.all((r["posterior"] > 0) & (r["posterior"] < 1))
    assert np.allclose(r["profile"].sum(1), 1)
    bits = (np.arange(32)[:, None] // (2 ** np.arange(5))) % 2
    assert np.allclose(r["profile"] @ bits, r["posterior"])
    assert r["sigmoid_error"] < 1e-14


def test_literal_posterior_agreement():
    z, y, s = world()
    x = np.array([[0.3, -0.1], [-2, 3]])
    assert np.allclose(
        M.read(z, y, s, x)["posterior"], [M.literal(z, y, s, a) for a in x]
    )


def test_field_is_gradient_not_global_covariance():
    z, y, s = world()
    x, h = np.array([[0.2, 0.7]]), 1e-5
    r = M.read(z, y, s, x)
    for j in range(2):
        dx = np.eye(2)[j][None, :] * h
        a, b = M.read(z, y, s, x + dx), M.read(z, y, s, x - dx)
        assert np.allclose(
            (a["log_odds"] - b["log_odds"]) / (2 * h),
            r["target_log_odds_field"][:, :, j],
            atol=1e-9,
        )
        assert np.allclose(
            (a["posterior"] - b["posterior"]) / (2 * h),
            r["target_probability_field"][:, :, j],
            atol=1e-9,
        )
        assert np.allclose(
            (a["log_density"] - b["log_density"]) / (2 * h),
            r["shared_density_score"][:, j],
            atol=1e-9,
        )


def test_translation_and_rotation_covariance():
    z, y, s = world()
    x, shift = np.array([[0.2, 0.7]]), np.array([5, -3])
    q = np.array([[0, -1], [1, 0]])
    a = M.read(z, y, s, x)
    b = M.read((z + shift) @ q, y, s, (x + shift) @ q)
    assert np.allclose(a["posterior"], b["posterior"])
    assert np.allclose(a["target_log_odds_field"] @ q, b["target_log_odds_field"])


def test_target_complement_reverses_field_not_shared_density():
    z, y, s = world()
    a, b = M.read(z, y, s, z), M.read(z, 1 - y, s, z)
    assert np.allclose(a["posterior"], 1 - b["posterior"])
    assert np.allclose(a["target_log_odds_field"], -b["target_log_odds_field"])
    assert np.allclose(a["shared_density_score"], b["shared_density_score"])


def test_null_coincident_prototypes_are_constant():
    z, y, s = world()
    r = M.read(np.zeros_like(z), y, s, z)
    assert np.allclose(r["posterior"], y.mean(0))
    assert np.max(np.abs(r["target_log_odds_field"])) == 0
    with pytest.raises(ValueError, match="variance"):
        M.source_variance(np.zeros_like(z))


def test_source_variance_is_label_free_and_symmetric():
    z, _, _ = world()
    assert M.source_variance(z) == M.source_variance(z[::-1])
    assert M.source_variance(z + 4) == pytest.approx(M.source_variance(z))


@pytest.mark.parametrize("variance", [0, -1, np.nan, np.inf])
def test_invalid_variance_rejected(variance):
    z, y, _ = world()
    with pytest.raises(ValueError):
        M.read(z, y, variance, z)


def test_cold_units_reject_duplicate_extra_labels_and_dimensions():
    r = {"frame_key": "u0", "events": [{"event_id": "a", "vector": [1, 2]}]}
    assert np.array_equal(M.cold_point(r, 2), [1, 2])
    with pytest.raises(ValueError):
        M.cold_point(r | {"label": 1}, 2)
    with pytest.raises(ValueError):
        M.cold_point(r | {"events": r["events"] * 2}, 2)
    with pytest.raises(ValueError):
        M.cold_point(r, 3)


def test_subset_action_and_group_axes_match_native_convention():
    from scripts import run_grain_vector_likelihood_hm204 as R

    a = np.arange(24 * 2 * 2 * 5 * 2 * 5, dtype=float).reshape(24, 2, 2, 5, 2, 5)
    v = R.action(a)
    assert v.shape == (24, 2, 2, 2, 2, 5)
    assert np.array_equal(v[:, :, :, 0], a[:, :, :, 1] - a[:, :, :, 2])
    assert np.array_equal(v[:, :, :, 1], a[:, :, :, 3] - a[:, :, :, 4])
    splits = np.array(["discovery"] * 12 + ["held_author"] * 12)
    x, y = R.group_values(v[:, 0, :, 0, 0], v[:, 0, :, 1, 0], splits, "held_author")
    assert x.shape == y.shape == (12, 5)
    assert np.array_equal(x, v[12:, 0, :, 0, 0].mean(1))


def test_actual_cold_cli_hash_binding_and_all_target_reading(tmp_path):
    z, y, variance = world()
    kernel = tmp_path / "kernel.npz"
    np.savez(kernel, prototypes=np.stack([z, z + 0.1]), labels=y)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "version": M.VERSION,
                "targets": list(M.TARGETS),
                "kernel": kernel.name,
                "kernel_sha256": digest(kernel),
                "code_sha256": digest(Path(M.__file__)),
                "variance": variance,
                "reference": y.mean(0).tolist(),
                "interpretation": "posterior_under_declared_source_Gaussian_law_not_human_probability",
            }
        )
    )
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {"frame_key": "test", "events": [{"event_id": "one", "vector": [0.2, 0.7]}]}
        )
        + "\n"
    )
    output = tmp_path / "output.jsonl"
    command = [
        sys.executable,
        str(ROOT / "scripts/score_grain_vector_likelihood_cold.py"),
        "--bundle",
        str(bundle),
        "--expected-sha256",
        digest(bundle),
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert np.allclose(
        rows[0]["posterior"],
        M.read(z, y, variance, np.array([[0.2, 0.7]]))["posterior"][0],
    )
    command[command.index("--expected-sha256") + 1] = "bad"
    assert subprocess.run(command, capture_output=True, check=False).returncode != 0
