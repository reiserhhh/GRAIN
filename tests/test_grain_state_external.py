#!/usr/bin/env python3
"""Constant state controls remain reportable without fabricating correlation."""

import numpy as np
import pytest

from scripts.analyze_grain_state_transport import external_quality


def test_constant_control_has_undefined_r_and_chance_auc():
    result = external_quality(
        np.ones(6) * 0.4, np.arange(6) / 5, [0, 0, 0, 1, 1, 1], binary=True
    )
    assert result["function_external_r"] is None
    assert result["delta_r"] is None
    assert result["function_auc"] == 0.5
    assert result["bootstrap_valid"] == 0


def test_constant_control_rejects_invalid_linkage():
    with pytest.raises(ValueError):
        external_quality([0.4, 0.4], [1, 2, 3], [0, 1])
