#!/usr/bin/env python3
"""Space sharing and response sharing are distinct mathematical quantities."""

import numpy as np

from scripts.analyze_grain_cross_query_geometry import geometry


def test_same_space_can_have_orthogonal_readouts():
    u = np.eye(8)[:, :2]
    result = geometry(u, u, u[:, 0], u[:, 1])
    assert result["normalized_projector_overlap"] == 1
    assert result["functional_cosine"] == 0
    assert result["normalized_operator_rank1_energy"] == 0.5


def test_rotating_basis_does_not_change_overlap():
    u = np.eye(8)[:, :2]
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    result = geometry(u, u @ rotation, u[:, 0], -u[:, 0])
    assert result["normalized_projector_overlap"] == 1
    assert result["functional_cosine"] == -1
    assert np.isclose(result["normalized_operator_rank1_energy"], 1)
