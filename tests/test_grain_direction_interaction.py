"""Readout additivity, multiplication and misleading cancellation witnesses."""

from scripts.analyze_grain_direction_interaction import decompose


def test_additive_readout_has_no_interaction():
    """For f(x,y)=x+y, separated displacements sum functionally."""
    r = decompose(0, 3, 1, 2)
    assert r["interaction"] == 0
    assert r["total"] == r["direction"] + r["complement"]


def test_multiplicative_readout_requires_both_components():
    """For f(x,y)=x*y, two zero isolated responses can produce a joint response."""
    r = decompose(0, 2, 0, 0)
    assert r["direction"] == r["complement"] == 0
    assert r["interaction"] == 2


def test_good_projection_can_hide_cancellation():
    """Zero projected error does not imply a functionally irrelevant complement."""
    r = decompose(0, 1, 1, 5)
    assert r["projection_error"] == 0
    assert r["complement"] == 5 and r["interaction"] == -5
    assert (
        r["projection_error"] ** 2
        == r["complement"] ** 2
        + r["interaction"] ** 2
        + 2 * r["complement"] * r["interaction"]
    )
