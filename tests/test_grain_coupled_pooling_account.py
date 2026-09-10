#!/usr/bin/env python3
"""The pooled linear-moment quadratic has an affine-position B response range."""

import numpy as np
import pytest

from scripts.report_grain_coupled_pooling import positional_account


def test_range_projection_orthogonality_and_zero_control():
    rng = np.random.default_rng(68)
    p = np.linspace(0, 1, 30)
    x = np.column_stack((np.ones(len(p)), p))
    target = rng.normal(size=(len(p), 8))
    projection = x @ np.linalg.lstsq(x, target, rcond=None)[0]
    a = positional_account(target, projection, p)
    assert a["orthogonality_error"] < 1e-14
    assert a["prediction_error"] == pytest.approx(a["residual_energy"])
    assert a["prediction_off_subspace_rms"] < 1e-14
    zero = positional_account(target, np.zeros_like(target), p)
    assert zero["prediction_error"] == pytest.approx(zero["target_energy"])


def test_actual_quadratic_cannot_leave_position_span_on_unchanged_B():
    from test_grain_coupled_pooling import toy

    from grain_research.vector_mechanisms import coupled_pooling as f

    rng, records, frozen = toy()
    originals = records[::4]
    m = f.reference(
        [r["body"] for r in originals], [r["last"] for r in originals], frozen
    )
    n = len(m["pool_center"])
    m |= {
        "arm": np.array(f.ARMS[1]),
        "coef": np.r_[frozen["coef"], rng.normal(size=n * (n + 1) // 2)],
    }
    plus, minus = records[2], records[3]
    delta = (
        f.predict(plus["body"], plus["last"], m)[1]
        - f.predict(minus["body"], minus["last"], m)[1]
    ) / 2
    cut = len(delta) // 2
    a = positional_account(
        delta[cut:], delta[cut:], np.linspace(-1, 1, len(delta))[cut:]
    )
    assert a["residual_energy"] < 1e-25
    assert a["prediction_error"] == 0


def test_bad_or_empty_response_rejected():
    with pytest.raises(ValueError):
        positional_account(np.zeros((4, 2)), np.zeros((4, 2)), np.arange(4))
    with pytest.raises(ValueError):
        positional_account(np.ones((4, 2)), np.ones((4, 2)), np.ones(4))
