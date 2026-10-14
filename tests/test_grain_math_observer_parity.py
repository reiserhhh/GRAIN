#!/usr/bin/env python3
"""Synthetic tests for observation support, parity and cycle retention."""

import numpy as np
import pytest

from grain_research.construction import math_author_current as C
from grain_research.construction import math_observer_parity as M


def test_lossless_cuts_and_missing():
    """Missing quartet support cannot become an all-zero feature."""
    text = " ".join(f"word{k}" for k in range(50))
    assert "".join(M.four_slices(text)) == text
    assert M.four_slices("word " * 47) == []
    event = "word " * 12
    assert (
        "".join(M.event_slices(event + "\n" + event + "\n" + event))
        == event + "\n" + event + "\n" + event
    )
    with pytest.raises(ValueError):
        M.event_slices("one\ntwo\nthree")


def test_area_is_translation_invariant_and_swap_odd():
    """A triangular area retains order while its mean does not."""
    x = np.random.default_rng(3).normal(size=(3, 4))
    mean, step, area = M.event_observations(x)
    rev = M.event_observations(x[[1, 0, 2]])
    shifted = M.event_observations(x + 71)
    np.testing.assert_allclose(mean, rev[0])
    np.testing.assert_allclose(step, -rev[1])
    np.testing.assert_allclose(area, -rev[2])
    np.testing.assert_allclose(area, shifted[2], atol=1e-12)


def test_odd_step_can_have_zero_cycle():
    """An exchange-odd exact difference does not retain circulation."""
    x = np.random.default_rng(5).normal(size=(3, 4))
    step = x[None, :, :] - x[:, None, :]
    _, odd = M.parity(step)
    assert np.linalg.norm(odd) > 0
    np.testing.assert_allclose(M.cycle_action(step), 0, atol=1e-15)
    for direction in (-1, 1):
        path = C.euler_path(C.cycle_counts(direction), 10)[:-1]
        np.testing.assert_allclose(M.integrate(path, step), 0, atol=1e-15)


def test_noncollinear_area_retains_cycle():
    """Second-order area can distinguish matched loops with equal occupancy."""
    x = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], float)
    f = np.array(
        [[M.event_observations(np.array([a, b, np.zeros(4)]))[2] for b in x] for a in x]
    )
    plus = C.euler_path(C.cycle_counts(1), 3)[:-1]
    minus = C.euler_path(C.cycle_counts(-1), 3)[:-1]
    difference = M.integrate(plus, f) - M.integrate(minus, f)
    np.testing.assert_allclose(difference, M.cycle_action(f), atol=1e-15)
    assert difference[0] > 0
    even, odd = M.parity(f)
    np.testing.assert_allclose(f, even + odd)
    np.testing.assert_allclose(M.cycle_action(even), 0, atol=1e-15)


def test_bad_kernel_refused():
    """Incomplete or nonfinite cells cannot be silently integrated."""
    with pytest.raises(ValueError):
        M.parity(np.ones((2, 3, 4)))
    f = np.ones((3, 3, 4))
    f[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        M.cycle_action(f)
