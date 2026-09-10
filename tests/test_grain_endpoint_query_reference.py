#!/usr/bin/env python3
"""Post-account algebra checks for the reference scope; not new data analyses."""

import numpy as np

from grain_research.vector_mechanisms.endpoint_query_contrast import decompose


def test_common_origin_shift_needs_query_specific_intercept_compensation():
    rng = np.random.default_rng(789)
    p, r, c = rng.normal(size=(2, 7)), rng.normal(size=(2, 7)), rng.normal(size=7)
    before, after = decompose(p, r), decompose(p + c, r)
    np.testing.assert_allclose(after[0], before[0])
    np.testing.assert_allclose(after[1] - before[1], (r[0] - r[1]) @ c)
    intercept = -r @ c
    np.testing.assert_allclose(after[2] + intercept[0] - intercept[1], before[2])


def test_context_conditioned_reader_has_zero_body_direction_but_nonzero_context_direction():
    rng = np.random.default_rng(790)
    a, r0, p, d, dc = (
        rng.normal(size=(7, 3)),
        rng.normal(size=7),
        rng.normal(size=7),
        rng.normal(size=7),
        rng.normal(size=3),
    )
    c = rng.normal(size=3)

    def score(v, z):
        return (r0 + a @ z) @ v

    eps = 1e-5
    body = (score(p + eps * d, c) - score(p - eps * d, c)) / (2 * eps)
    context = (score(p, c + eps * dc) - score(p, c - eps * dc)) / (2 * eps)
    np.testing.assert_allclose(body, (r0 + a @ c) @ d, atol=1e-8)
    np.testing.assert_allclose(context, (a @ dc) @ p, atol=1e-8)
