#!/usr/bin/env python3
"""Independent dense/filter identities for dynamic transport, no private data."""

import numpy as np
import pytest
from scipy.special import logsumexp

from grain_research.gptpro_foundation_v3 import f16_dynamic_transport as m
from grain_research.gptpro_foundation_v3.f12_transport import (
    fit_transport,
    transport_score,
)
from grain_research.gptpro_foundation_v3.f16_dynamic_gaussian_reference import (
    conditional_filter,
    joint_future,
)

MODEL = {
    "prior": {
        "mu": [0.1, 0.4, 0.6],
        "Sigma": [[1.0, 0.1, 0.03], [0.1, 0.3, 0.02], [0.03, 0.02, 0.2]],
    },
    "log_sigma": np.log([0.5, 0.5]).tolist(),
}


def test_compiler_matches_independent_augmented_filter():
    """Same full target moments via recursive versus dense Gaussian conditioning."""
    observed, _ = m.simulate(42, "pilot", 4)
    z, y = observed["conditions"], observed["old"]
    kernel = m.compile_source(MODEL, z, 32)
    mean, state, personal, ess = m.component(kernel, y[..., None], np.ones((1, 1)))
    a = (np.broadcast_to(z, y.shape), y, np.ones_like(y), np.ones_like(y, bool))
    fm, fc, ll = conditional_filter(MODEL, kernel["eta"], a)
    tm, tv, tp = joint_future(fm, fc, kernel["eta"], MODEL["log_sigma"])
    target = np.array([[1, 0, 0, 0, 0], [0, 0.25, 0.25, 0.25, 0.25]])
    tm = tm @ target.T
    tv = np.einsum("li,kaij,mj->kalm", target, tv, target)
    tp = np.einsum("li,kaij,mj->kalm", target, tp, target)
    lw = ll + kernel["log_weights"][:, None]
    w = np.exp(lw - logsumexp(lw, axis=0))
    center = np.einsum("ka,kal->al", w, tm)
    delta = tm - center[None]
    expected_p = np.einsum("ka,kalm->alm", w, tp) + np.einsum(
        "ka,kal,kam->alm", w, delta, delta
    )
    np.testing.assert_allclose(mean[0], center, atol=1e-10)
    np.testing.assert_allclose(state[0], np.einsum("ka,kalm->alm", w, tv), atol=1e-10)
    np.testing.assert_allclose(personal[0], expected_p, atol=1e-10)
    np.testing.assert_allclose(ess[0], 1 / (w * w).sum(0), atol=1e-10)


@pytest.mark.parametrize("mode", ["global", "conditional"])
@pytest.mark.parametrize("negative", [False, True])
def test_basis_exactly_replays_inverse(mode, negative):
    """Four basis coordinates retain signed global and condition-specific maps."""
    observed, _ = m.simulate(61, "pilot", 5)
    z, old = observed["conditions"], observed["old"]
    scale = -1 if negative else 1
    new = (
        (1 + 1.5 * old) * scale
        if mode == "global"
        else (z + (1 + 0.25 * z) * old) * scale
    )
    cal = fit_transport(old.ravel(), new.ravel(), np.tile(z, 5), mode)
    basis, coeffs, support = m.inverse_basis(new, z, [cal, cal])
    values = np.einsum("atd,jd->jat", basis, coeffs)
    direct, mask = transport_score(cal, new, z)
    assert support.all() and mask.all()
    np.testing.assert_allclose(values, np.stack([direct, direct]), atol=1e-12)
    np.testing.assert_allclose(values[0], old, atol=1e-12)


def test_failed_calibration_not_dropped():
    """One invalid draw rejects a mixture rather than renormalizing survivors."""
    observed, _ = m.simulate(12, "pilot", 4)
    z, y = observed["conditions"], observed["old"]
    valid = fit_transport(y.ravel(), y.ravel(), np.tile(z, 4), "global")
    basis, coefs, support = m.inverse_basis(y, z, [valid, {"status": "MISSING_ANCHOR"}])
    assert basis is None and coefs is None and not support.any()


def test_mixture_total_covariance_and_pilot_effect():
    """All source-pilot combinations reproduce the complete second moment."""
    obs, _ = m.simulate(3, "application", 3)
    y, z = obs["old"], obs["conditions"]
    other = {**MODEL, "prior": {**MODEL["prior"], "mu": [0.7, 0.2, 0.5]}}
    kernels = [m.compile_source(model, z, 16) for model in (MODEL, other)]
    basis = np.stack([y, np.ones_like(y)], -1)
    coefs = np.array([[1, 0], [1, 0.2], [0.8, -0.1]])
    result = m.integrate(kernels, basis, coefs, chunk=2)
    collapsed = m.collapse(result, slice(None))
    cells = [m.component(k, basis, coefs) for k in kernels]
    raw_m = np.stack([c[0] for c in cells])
    raw_v = np.stack([c[1] + c[2] for c in cells])
    center = raw_m.mean((0, 1))
    expected = (raw_v + raw_m[..., :, None] * raw_m[..., None, :]).mean(
        (0, 1)
    ) - center[..., :, None] * center[..., None, :]
    np.testing.assert_allclose(collapsed["mean"], center, atol=1e-10)
    np.testing.assert_allclose(collapsed["covariance"], expected, atol=1e-10)
    assert np.max(collapsed["pilot"]) > 0
    assert np.linalg.eigvalsh(collapsed["covariance"]).min() > 0


def test_roles_and_future_plan():
    """Independent roles, requested real future and no truth in observed output."""
    p, truth = m.simulate(22, "pilot", 10)
    a, _ = m.simulate(22, "application", 10)
    assert set(p) == {"old", "conditions"} and not np.array_equal(p["old"], a["old"])
    np.testing.assert_array_equal(truth["future_conditions"], [1, -1, 1, -1])
    np.testing.assert_allclose(truth["T3"], truth["future"].mean(1))
    with pytest.raises(ValueError):
        m.simulate(22, "source", 10)
    with pytest.raises(ValueError):
        m.compile_source(MODEL, [-1] * 10)
