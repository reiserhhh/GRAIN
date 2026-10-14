#!/usr/bin/env python3
"""Synthetic gauge/reference/route checks, with no neural observations."""

import numpy as np
import pytest

from grain_research.construction import math_continuation_routes as C
from grain_research.construction import math_source_route_readout as F


def test_references_and_duplicate_covector_guard():
    """Frequency and distinct-word conventions stay explicit and fixed."""
    ids, w, b = [2, 2, 5], [[1, 0], [1, 0], [0, 3]], [0, 0, 0]
    refs = F.reference_covectors(ids, w, b)
    np.testing.assert_allclose(refs["occurrence"][0], [2 / 3, 1])
    np.testing.assert_allclose(refs["unique"][0], [0.5, 1.5])
    with pytest.raises(ValueError, match="one frozen"):
        F.reference_covectors(ids, [[1, 0], [2, 0], [0, 3]], b)


def test_gauge_and_total_recovery():
    """A separate arbitrary common logit shift at every cell changes neither part."""
    rng = np.random.default_rng(260)
    z, a, m, g = rng.normal(size=(4, 2, 11, 7))
    old, new = F.relative_parts(z, a, m), F.relative_parts(z + g, a + g, m + g)
    for key in old:
        np.testing.assert_allclose(old[key], new[key], atol=3e-15)
    np.testing.assert_allclose(old["shape"] + old["calibration"], old["total"])


def test_reference_reallocation_not_removal():
    """Changing the declared reference moves equal and opposite signed amounts."""
    rng = np.random.default_rng(261)
    z, a, m, delta = rng.normal(size=(4, 2, 11, 3))
    old, new = F.relative_parts(z, a, m), F.relative_parts(z, a, m + delta)
    np.testing.assert_allclose(old["total"], new["total"], atol=2e-15)
    np.testing.assert_allclose(
        new["shape"] - old["shape"],
        -(new["calibration"] - old["calibration"]),
        atol=2e-15,
    )


def test_common_field_has_zero_relative_range():
    """A nonzero common output shift is not an alternative-word distinction."""
    d = np.ones((5, 3, 2))
    w = np.asarray([[1, 0], [0, 1], [0.2, 0.8]])
    np.testing.assert_allclose(F.support_oscillation(d, w), 0, atol=1e-15)
    d[..., 1] = 0
    assert np.all(F.support_oscillation(d, w) > 0)


def test_route_and_source_linearity_retains_signed_interaction():
    """All four routes, including negative interaction, reconstruct each account."""
    rng = np.random.default_rng(262)
    shape = {
        k: rng.normal(size=(24, 5)) for k in ("native", "relay", "direct", "closed")
    }
    cal = {k: rng.normal(size=(24, 5)) for k in shape}
    total = {k: shape[k] + cal[k] for k in shape}
    source = rng.normal(size=(24, 5))
    ss, aa, tt = [C.decompose(v) for v in (shape, cal, total)]
    for key in tt:
        np.testing.assert_allclose(ss[key] + aa[key], tt[key], atol=2e-15)
        np.testing.assert_allclose(
            C.contacts(source, ss[key]) + C.contacts(source, aa[key]),
            C.contacts(source, tt[key]),
            atol=1e-15,
        )


def test_reject_partial_or_nonfinite_states():
    """Missing voices and invalid coordinates are not imputed."""
    with pytest.raises(ValueError, match="Complete"):
        F.state_difference(np.zeros((2, 10, 3, 4)))
    with pytest.raises(ValueError, match="Reference"):
        F.relative_parts(
            np.zeros((2, 11, 3)), np.zeros((2, 11, 3)), np.zeros((2, 11, 2))
        )
