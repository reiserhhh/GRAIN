#!/usr/bin/env python3
"""F12 transport and F13 uncertainty checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f12 import affine, inverse_affine
from grain_research.gptpro_foundation_v3.f13 import mdc, paired_change_variance


def test_affine_roundtrip():
    values = np.asarray([-.5, .1, .8])
    np.testing.assert_allclose(inverse_affine(affine(values, 1., 1.5), 1., 1.5), values)


def test_paired_change_and_mdc_are_nonnegative():
    assert paired_change_variance(.2, .2, .2) == 0.0
    assert mdc(.2) > 0
