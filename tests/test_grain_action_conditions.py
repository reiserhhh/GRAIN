#!/usr/bin/env python3
"""Finite identities for the post-result condition diagnostic."""

import numpy as np
import pytest

from scripts import diagnose_grain_action_geometry as diagnostic
from scripts.diagnose_grain_action_geometry import grouped_covariance, split_functional


def test_functional_conditions_sum_to_symmetric_integral():
    """Origin/path/body terms reconstruct the actual declared two-path matrix."""
    a, b, c = np.random.default_rng(1).normal(size=(3, 4, 5))
    parts = split_functional(a, b, c)
    np.testing.assert_allclose(sum(parts.values()), (b + c) / 2, atol=1e-12)


def test_fold_covariance_does_not_invent_within_fold_signal():
    """A constant-per-fold control can correlate only through fold means."""
    groups = np.repeat(np.arange(4), 6)
    x = np.arange(24, dtype=float)
    y = -groups.astype(float)
    result = grouped_covariance(x, y, groups)
    assert abs(result["within_fold_covariance"]) < 1e-12
    assert result["closure_error"] < 1e-12
    assert abs(result["total_r"] - result["between_contribution_to_total_r"]) < 1e-12


def test_diagnostic_destination_survives_input_hash_loop(tmp_path, monkeypatch):
    """An output name must not be overwritten by names of verified source files."""
    out = tmp_path / "logic_action_geometry_v1"
    out.mkdir()
    monkeypatch.setattr(diagnostic, "BASE", tmp_path)
    monkeypatch.setattr(diagnostic, "verify", lambda _: None)
    monkeypatch.setattr(diagnostic, "read", lambda _: {"files": {"source.json": "h"}})
    monkeypatch.setattr(diagnostic, "check", lambda *_: None)

    def stop_before_data():
        raise RuntimeError("Synthetic check stops before reading author data")

    monkeypatch.setattr(diagnostic, "donors_and_design", stop_before_data)
    with pytest.raises(RuntimeError, match="Synthetic check"):
        diagnostic.run("named_output")
    assert (out / "named_output").is_dir()
    assert not (out / "source.json").exists()


def test_centered_displacements_can_have_nonzero_mean_path_action():
    """Quadratic curvature survives zero-mean inputs that defeat one linear slope."""
    d = np.array([[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]])
    j = np.array([0.2, -0.3])
    hessian = np.array([[2.0, 0.4], [0.4, 3.0]])
    integrated = j + 0.5 * d @ hessian
    exact = d @ j + 0.5 * np.einsum("ni,ij,nj->n", d, hessian, d)
    np.testing.assert_allclose(np.sum(integrated * d, axis=1), exact)
    np.testing.assert_allclose(d.mean(0), 0)
    assert exact.mean() > 0
    assert abs((d @ integrated.mean(0)).mean()) < 1e-12
