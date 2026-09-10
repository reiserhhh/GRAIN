#!/usr/bin/env python3
"""Post-account mathematical check: content and selection are jointly integrable."""

import numpy as np
import pytest

from grain_research.vector_mechanisms import evidence_components as f
from grain_research.vector_mechanisms.token_potential import hidden_gradient
from tests.test_grain_nonlinear_pooling import toy


@pytest.mark.parametrize("arm", f.reader.ARMS)
def test_content_selection_curls_cancel_but_parts_are_not_separate_scores(arm):
    rng, records, local = toy()
    model = f.reader.reference(records, local, arm)
    model["coef"] = np.r_[local["coef"], rng.normal(size=48)]
    body, last = records[0]["body"], records[0]["last"]
    a, b = rng.normal(size=body.shape), rng.normal(size=body.shape)

    def gradients(x):
        return np.stack(
            [hidden_gradient(v, model) for v in f.densities(x, last, model)]
        )

    da = (gradients(body + 1e-5 * a) - gradients(body - 1e-5 * a)) / 2e-5
    db = (gradients(body + 1e-5 * b) - gradients(body - 1e-5 * b)) / 2e-5
    curl = np.einsum("cth,th->c", da, b) - np.einsum("cth,th->c", db, a)
    assert abs(curl[1]) > 1e-8 and abs(curl[2]) > 1e-8
    np.testing.assert_allclose([curl[0], curl[3], curl[1] + curl[2]], 0, atol=1e-8)
