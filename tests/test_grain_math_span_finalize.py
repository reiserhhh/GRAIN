#!/usr/bin/env python3
"""Finite material-family resampling and fail-closed finalization tests."""

import numpy as np
import pytest

from scripts.finalize_grain_math_span_packet import empirical_interval, finalize


def test_family_interval_keeps_constant_and_signed_units():
    assert empirical_interval([0.2] * 4) == pytest.approx([0.2, 0.2])
    lo, hi = empirical_interval([-0.5, -0.3, -0.1, -0.2])
    assert -0.5 <= lo < hi < 0
    np.testing.assert_allclose(
        empirical_interval([1, 2, 3, 4]), empirical_interval([1, 2, 3, 4])
    )


def test_family_interval_does_not_accept_one_family_or_invalid_values():
    for values in ([1], [1, np.nan], [[1, 2], [3, 4]]):
        with pytest.raises(ValueError, match="families"):
            empirical_interval(values)


def test_finalization_never_overwrites_prior_results(tmp_path):
    with pytest.raises(FileExistsError, match="preserve"):
        finalize(tmp_path, tmp_path, tmp_path, tmp_path)
