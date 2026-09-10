#!/usr/bin/env python3
"""Source offset, composed derivatives and non-additive endpoint witnesses."""

import numpy as np
import pytest
import torch

from grain_research.vector_mechanisms.context_carrier import cached_suffix_kwargs
from grain_research.vector_mechanisms.selection_endpoint import endpoint
from grain_research.vector_mechanisms.source_offset import (
    NEW_CANDIDATES,
    effect_summary,
    family,
    ingredient_effects,
    offset_geometry,
)
from grain_research.vector_mechanisms.whole_path_assembly import reverse_response
from tests.test_grain_complete_transport import tiny_model


def fixture():
    rng = np.random.default_rng(56)
    x, d = rng.normal(size=(9, 16)), rng.normal(size=(4, 9, 16))
    m = {
        "mode": np.asarray("pointwise"),
        "mean": np.zeros(16),
        "basis": np.eye(16, 3),
        "center": np.zeros(6),
        "scale": np.ones(6),
        "coef": rng.normal(size=(6, 16)) * 0.05,
    }
    return x, d, m


def test_source_decomposition_and_reference_invariant():
    x, d, m = fixture()
    f = family(x, d, m)
    for index in (0, 1):
        np.testing.assert_allclose(
            f["joint"][index],
            f["offset_only"][index]
            + f["nonintercept_only"][index]
            - f["identity"][index],
            atol=1e-12,
        )
    geo = offset_geometry(x, m, f["joint"][0])
    assert geo["energy_identity_error"] < 1e-12
    # Changing current material does not refit or redefine the common coefficient.
    other = family(x + 10, d, m)
    np.testing.assert_allclose(
        other["offset_only"][0] - (x + 10), f["offset_only"][0] - x, atol=1e-12
    )
    m["center"][3] = 1
    with pytest.raises(ValueError, match="intercept"):
        family(x, d, m)


@pytest.mark.parametrize("candidate", NEW_CANDIDATES)
def test_new_functions_match_actual_suffix_composed_derivatives(candidate):
    x, d, m = fixture()
    model = tiny_model()
    mx, md = family(x, d, m)[candidate]
    t, td = (
        torch.tensor(mx[None], dtype=torch.float32),
        torch.tensor(md, dtype=torch.float32),
    )
    kw = cached_suffix_kwargs(model, t)
    ids, c = torch.tensor([1, 2, 3]), torch.tensor([-0.5, 0.0, 0.5])
    _, _, tau = reverse_response(model, t, td, kw, ids, c, block=1)
    _, jvp = torch.func.jvp(
        lambda z: endpoint(model, z, kw, ids, c, block=1), (t,), (td[:1],)
    )
    np.testing.assert_allclose(float(jvp), tau[0], atol=1e-7, rtol=1e-4)
    eps, scores = 0.001, []
    for sign in (1, -1):
        px = family(x + sign * eps * d[0], d, m)[candidate][0]
        with torch.no_grad():
            scores.append(
                float(
                    endpoint(
                        model,
                        torch.tensor(px[None], dtype=torch.float32),
                        kw,
                        ids,
                        c,
                        block=1,
                    )
                )
            )
    np.testing.assert_allclose(
        (scores[0] - scores[1]) / (2 * eps), tau[0], atol=1e-4, rtol=0.004
    )


def test_nonlinear_endpoint_ingredients_and_signed_error_account():
    x = np.asarray([0.1, 0.4, 0.8])
    vals = np.stack([np.tanh(x), np.tanh(x + 2), np.tanh(x + 0.2), np.tanh(x + 2.2)])
    a = ingredient_effects(vals, x)
    np.testing.assert_allclose(a["offset"] + a["remainder"], a["change"])
    assert np.linalg.norm(a["interaction"]) > 0.01
    s = effect_summary(a)
    assert s["mse_accounting_error"] < 1e-12 and s["allocation_max_error"] < 1e-12
    # A pure offset preserves the input tangent but changes the nonlinear reader.
    assert (1 / np.cosh(x + 2) ** 2 < 1 / np.cosh(x) ** 2).all()
