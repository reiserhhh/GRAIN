#!/usr/bin/env python3
"""Check fixed-prediction paired, stratified comparison arithmetic."""

import numpy as np
import pytest

from scripts.compare_grain_gptpro_f16_targets import paired_interval


def test_zero_pairwise_loss_difference_stays_zero():
    result = paired_interval([np.zeros(200), np.zeros(200), np.zeros(200)])
    assert result["authors"] == 600
    assert result["mean_loss_difference"] == 0
    assert result["paired_author_95_ci"] == [0, 0]


def test_each_seed_retains_its_author_weight():
    result = paired_interval([np.full(2, 3.0), np.full(5, -1.0)])
    assert result["mean_loss_difference"] == pytest.approx(1 / 7)
    assert result["paired_author_95_ci"] == pytest.approx([1 / 7, 1 / 7])
