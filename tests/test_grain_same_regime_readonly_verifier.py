#!/usr/bin/env python3
"""Test the mechanical channel-name adaptation, not a new scientific method."""

import numpy as np
import pytest

from scripts.verify_grain_same_regime_reader_hm274 import reader_channel


def test_literal_channels():
    """The archive's order channel is the reader's odd projection."""
    x = np.random.default_rng(17).normal(size=(48, 4, 5))
    np.testing.assert_allclose(reader_channel(x, "order"), (x[:, 0] - x[:, 1]) / 2)
    np.testing.assert_allclose(
        reader_channel(x, "join"), (x[:, 0] + x[:, 1] - x[:, 2] - x[:, 3]) / 2
    )


def test_reject_unknown_or_incomplete():
    """Do not substitute an unregistered channel or incomplete observation."""
    with pytest.raises(ValueError):
        reader_channel(np.zeros((48, 4, 5)), "other")
    with pytest.raises(ValueError):
        reader_channel(np.zeros((47, 4, 5)), "order")
