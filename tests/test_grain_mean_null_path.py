#!/usr/bin/env python3
"""Mean-null geometry and line-integral witnesses, without a neural checkpoint."""

import numpy as np
import pytest
import torch

from scripts.run_grain_mean_null_path import mean_null_direction, simpson


def test_projection_preserves_means_and_unselected_states():
    h = torch.randn(12, 7, dtype=torch.float64)
    groups = [[0, 1, 2], [4, 5, 6], [8, 9]]
    d = mean_null_direction(h, groups, [0, 2])
    for group in (groups[0], groups[2]):
        assert torch.allclose(
            d[group].sum(0), torch.zeros(7, dtype=h.dtype), atol=1e-14
        )
        assert torch.allclose((h + d)[group], h[group].mean(0).expand(len(group), -1))
    assert torch.equal(d[[3, 4, 5, 6, 7, 10, 11]], torch.zeros(7, 7, dtype=h.dtype))


def test_mean_readout_has_zero_effect_but_token_contrast_does_not():
    h = torch.tensor([[1.0, 0.0], [-1.0, 2.0]], dtype=torch.float64)
    d = mean_null_direction(h, [[0, 1]], [0])
    assert torch.allclose((h + d).mean(0), h.mean(0))
    f = lambda x: x[0, 0] * x[1, 1]
    assert abs(float(f(h + d) - f(h))) > 1


def test_simpson_completeness_for_quadratic_path():
    h = torch.tensor([[1.0, 0.0], [-1.0, 2.0]], dtype=torch.float64)
    d = mean_null_direction(h, [[0, 1]], [0])
    score = lambda v: v[0, 0] * v[1, 1]
    slopes = {}
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        leaf = (h + a * d).requires_grad_(True)
        (g,) = torch.autograd.grad(score(leaf), leaf)
        slopes[a] = float((g * d).sum())
    delta = float(score(h + d) - score(h))
    assert abs(simpson(slopes) - delta) < 1e-12
    assert abs(simpson({a: slopes[a] for a in (0.0, 0.5, 1.0)}) - delta) < 1e-12
    assert not np.isclose(slopes[0.0], delta)


def test_reject_incomplete_quadrature():
    with pytest.raises(ValueError):
        simpson({0.0: 1, 0.5: 2})


def test_mean_projection_is_idempotent_and_residual_orthogonal():
    generator = torch.Generator().manual_seed(4)
    h = torch.randn(10, 4, generator=generator, dtype=torch.float64)
    groups = [[0, 1, 2], [5, 6, 7, 8]]
    d = mean_null_direction(h, groups, [0, 1])
    projected = h + d
    assert torch.allclose(
        mean_null_direction(projected, groups, [0, 1]), torch.zeros_like(h), atol=1e-14
    )
    assert abs(float((projected * d).sum())) < 1e-13


def test_better_teacher_fit_can_lose_outcome_covariance():
    u, v = np.array([-3.0, -3.0, 3.0, 3.0]), np.array([-1.0, 1.0, -1.0, 1.0])
    teacher, old, new = u + v, 0.1 * u + v, u
    assert np.mean((teacher - new) ** 2) < np.mean((teacher - old) ** 2)
    assert np.corrcoef(teacher, new)[0, 1] > np.corrcoef(teacher, old)[0, 1]
    assert np.corrcoef(v, new)[0, 1] == 0
    assert np.corrcoef(v, old)[0, 1] > 0.9
    fitted = 0.2 * u
    assert np.isclose(
        np.mean((teacher - fitted) ** 2), np.mean(v**2) + np.mean((u - fitted) ** 2)
    )
