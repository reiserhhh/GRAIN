#!/usr/bin/env python3
"""Recipient targets, source exclusion and signed bilinear algebra stay explicit."""

import numpy as np
import pytest

from grain_research.vector_mechanisms.compiled_query_transfer import (
    borrow,
    contrast_accounting,
    contrasts,
    fidelity,
    validate_sources,
)


def test_borrowing_is_asymmetric_and_recipient_direction_stays_fixed():
    rng = np.random.default_rng(44)
    w, d = rng.normal(size=(2, 2, 7)), rng.normal(size=(2, 4, 7))
    a = borrow(w, d)
    assert a.shape == (2, 2, 2, 4)
    for recipient in range(2):
        for donor in range(2):
            for mode in range(2):
                np.testing.assert_allclose(
                    a[recipient, donor, mode], d[recipient] @ w[donor, mode]
                )
    assert not np.allclose(a[0, 1], a[1, 0])
    c = contrasts(w, d)
    np.testing.assert_allclose(c["contrast"], c["reader"] + c["transport"], atol=1e-14)
    swapped = contrasts(w[::-1], d[::-1])
    for key in c:
        np.testing.assert_allclose(swapped[key], -c[key], atol=1e-14)


def test_signed_cancellation_and_zero_contrast_are_not_dropped():
    r = np.array([1.0, 2.0, -3.0, 0.5])
    result = contrast_accounting(r, -r)
    assert result["contrast_rms"] == 0
    assert result["reader_rms"] > 0
    assert np.isnan(result["reader_projected_share"])
    assert result["variance_closure_error"] < 1e-14
    t = -0.7 * r + np.array([0.2, -0.1, 0.3, 0.1])
    result = contrast_accounting(r, t)
    assert result["signed_cross_energy"] < 0
    assert result["reader_projected_share"] > 1
    assert result["reader_projected_share"] + result[
        "transport_projected_share"
    ] == pytest.approx(1)
    assert result["variance_closure_error"] < 1e-14
    assert result["energy_closure_error"] < 1e-14


def test_equal_readers_or_equal_directions_isolates_the_other_component():
    rng = np.random.default_rng(19)
    w, d = rng.normal(size=(2, 2, 8)), rng.normal(size=(2, 4, 8))
    w[1] = w[0]
    np.testing.assert_array_equal(contrasts(w, d)["reader"], 0)
    w[1] *= 2
    d[1] = d[0]
    np.testing.assert_array_equal(contrasts(w, d)["transport"], 0)


@pytest.mark.parametrize("index", range(4))
def test_source_exclusion_and_donor_membership(index):
    members = sorted(set(range(4)) - {index})
    validate_sources("source4", index, members)
    validate_sources("new8", index, list(range(4)))
    with pytest.raises(ValueError, match="exclusion"):
        validate_sources("source4", index, list(range(4)))
    with pytest.raises(ValueError, match="exclusion"):
        validate_sources("source4", index, members[:-1] + members[:1])


def test_fidelity_uses_recipient_target_not_donor_target():
    recipient = np.array([1.0, -2.0, 3.0, 4.0])
    donor_target = -recipient
    m = fidelity(recipient, donor_target)
    assert m["action_r"] == pytest.approx(-1)
    assert m["action_mse_over_zero"] == pytest.approx(4)
    assert np.isnan(fidelity(np.zeros(4), donor_target)["action_mse_over_zero"])
    with pytest.raises(ValueError):
        borrow(np.ones((2, 2, 8)), np.ones((2, 3, 8)))
    with pytest.raises(ValueError):
        fidelity([1, 2, np.nan], [1, 2, 3])
