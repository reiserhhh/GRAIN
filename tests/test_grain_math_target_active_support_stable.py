#!/usr/bin/env python3
"""Companion checks include near-degenerate columns and independent tall SVD."""

import numpy as np
import pytest

from grain_research.construction.math_target_active_support_stable import (
    alternate_basis,
    source_basis,
)


def test_nearly_dependent_columns_stable_projector():
    rng = np.random.default_rng(36)
    q = np.linalg.qr(rng.standard_normal((300, 10)))[0]
    x = q @ np.diag(np.geomspace(1, 1e-8, 10)) @ rng.standard_normal((10, 15))
    fitted = source_basis(x, ["source"] * 15)
    metadata = {"rank": fitted["rank"], "rank_tolerance": fitted["tolerance"]}
    direct = alternate_basis(x, metadata)
    a = fitted["basis"]
    np.testing.assert_allclose(a @ a.T, direct @ direct.T, atol=1e-8)


def test_stable_source_isolation_and_zero():
    with pytest.raises(ValueError, match="application"):
        source_basis(np.eye(3), ["source", "application", "source"])
    fitted = source_basis(np.zeros((4, 2)), ["source"] * 2)
    assert fitted["rank"] == 0 and fitted["basis"].shape == (4, 0)
