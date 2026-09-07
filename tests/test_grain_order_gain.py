"""Finite symmetric accounting and prediction-boundary witnesses."""

import pytest

from scripts.analyze_grain_order_gain import account


def test_constant_gain_prediction_and_identity():
    """When only amplitude changes, the original gain predicts the new response."""
    r = account(2, 4, 6, 12)
    assert r["gain_term"] == 0
    assert r["amplitude_term"] == r["observed_change"] == 6
    assert r["original_gain_prediction"] == 12


def test_opposing_terms_can_cancel():
    """Equal observed outputs can conceal changing amplitude and finite gain."""
    r = account(1, 2, 2, 2)
    assert r["amplitude_term"] == 1.5 and r["gain_term"] == -1.5
    assert r["observed_change"] == 0
    assert r["original_gain_prediction"] == 4
    assert (
        r["observed_change"] ** 2
        == r["amplitude_term"] ** 2
        + r["gain_term"] ** 2
        + 2 * r["amplitude_term"] * r["gain_term"]
    )


def test_degenerate_gain_rejected():
    """No division by a near-zero projection or hidden row exclusion."""
    with pytest.raises(ValueError, match="Near-zero"):
        account(1e-9, 1, 1, 1)
