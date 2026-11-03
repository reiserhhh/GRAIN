#!/usr/bin/env python3
"""Distinguish a uniform personal offset from an address-specific pattern."""

import numpy as np

from scripts.describe_grain_publication_response_shape_hm292 import decompose


def test_uniform_offset_has_no_shape_even_with_different_occupancy():
    """A reliable global habit is not automatically a condition response shape."""
    r = np.full(8, 1 / 8)
    p = np.array([0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0])[None, None]
    offset = np.array([1.0, -2.0, 3.0])
    response = (p[..., None] * offset / np.sqrt(r)[None, None, :, None]).reshape(
        1, 1, 24
    )
    result = decompose(
        {"address_mass": r}, {"response": response, "address_probability": p}
    )
    np.testing.assert_allclose(result["shape"], 0, atol=1e-14)
    np.testing.assert_allclose(result["level"], response)


def test_shape_reconstruction_and_zero_sum():
    """Keep nonorthogonal level/shape and missing-address weighted moments."""
    rng = np.random.default_rng(292)
    r = np.full(8, 1 / 8)
    p = np.full((3, 2, 8), 1 / 8)
    raw = rng.normal(size=(3, 2, 8, 3))
    response = (raw / np.sqrt(r)[None, None, :, None]).reshape(3, 2, 24)
    result = decompose(
        {"address_mass": r}, {"response": response, "address_probability": p}
    )
    np.testing.assert_allclose(result["level"] + result["shape"], response)
    np.testing.assert_allclose(
        (result["shape"].reshape(3, 2, 8, 3) * np.sqrt(r)[None, None, :, None]).sum(2),
        0,
        atol=1e-14,
    )
