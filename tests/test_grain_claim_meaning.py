#!/usr/bin/env python3
"""Finite semantic/common worlds and full claim-rule safety before observation."""

import numpy as np
import pytest
from test_grain_person_bound_response import cold_fixture

from grain_research.construction import math_claim_meaning as M
from grain_research.construction import math_person_bound_tree as T


def fixture_logits(direction, common=0.0, availability_odd=0.0):
    """Build full synthetic laws,not evidence about an actual author."""
    shape = (48, 2, 2, 2, 50)
    d, c, a = (np.full(shape, v) for v in (direction, common, availability_odd))
    return np.stack(
        (
            T.tree_log_probabilities(d + c, 2 + a),
            T.tree_log_probabilities(-d + c, 2 - a),
        )
    )


def bank():
    """Use the registered complete synthetic bank from existing addressed tests."""
    return cold_fixture()["item_bank"]


def test_semantic_and_common_worlds_are_separated_without_deletion():
    fields, sides, _ = M.readings(fixture_logits(3, 5, 7), bank())
    assert fields["semantic_direction"]["items"][:, 0] == pytest.approx(3)
    assert fields["affirmation_direction"]["items"][:, 0] == pytest.approx(5)
    assert fields["availability_even"]["items"][:, 0] == pytest.approx(2)
    assert fields["availability_odd"]["items"][:, 0] == pytest.approx(7)
    for key in sides[0]["direction"]:
        odd, even = (
            fields["semantic_direction"][key],
            fields["affirmation_direction"][key],
        )
        assert even + odd == pytest.approx(sides[0]["direction"][key])
        assert even - odd == pytest.approx(sides[1]["direction"][key])


def test_exact_category_swap_has_no_common_direction_or_odd_availability():
    z = fixture_logits(4)
    assert np.exp(z[1]) == pytest.approx(np.exp(z[0])[..., [1, 0, 2]])
    fields, _, _ = M.readings(z, bank())
    assert np.max(abs(fields["affirmation_direction"]["items"])) < 1e-12
    assert np.max(abs(fields["availability_odd"]["items"])) < 1e-12


def test_involuting_claim_changes_only_odd_sectors():
    z = fixture_logits(3, 5, 7)
    a, _, _ = M.readings(z, bank())
    b, _, _ = M.readings(z[::-1], bank())
    for channel in M.CHANNELS:
        sign = -1 if channel in ("semantic_direction", "availability_odd") else 1
        assert b[channel]["items"] == pytest.approx(sign * a[channel]["items"])


@pytest.mark.parametrize("actor", [-1, 1])
def test_whole_fit_claim_not_behavior_is_logically_denied(actor):
    plus, minus = (M.question("Worry about things.", actor, p) for p in (1, -1))
    assert plus.replace("It is true that", "It is false that") == minus
    assert '"I worry about things."' in plus and '"I worry about things."' in minus
    assert f"writer {'R' if actor == 1 else 'S'}" in plus
    assert "THE CLAIM" in plus


@pytest.mark.parametrize("bad", [-2, 0, 2])
def test_invalid_claim_polarity_is_rejected(bad):
    with pytest.raises(ValueError):
        M.question("Worry about things.", 1, bad)


def test_absent_head_bias_keeps_full_new_alphabet():
    assert M.exported_bias(3).shape == (3,)
    assert np.array_equal(M.exported_bias(3), np.zeros(3))
    with pytest.raises(ValueError):
        M.exported_bias(3, [0, 0])


def test_cold_runs_both_actual_laws_and_rejects_application_labels():
    record = cold_fixture()
    record["version"] = "HM289-v1"
    record["item_logits"] = fixture_logits(3, 5).tolist()
    result = M.cold(record)
    assert set(result["channels"]) == set(M.CHANNELS)
    record["application_labels"] = [1] * 24
    with pytest.raises(ValueError, match="registered"):
        M.cold(record)


def test_missing_denial_law_is_not_imputed():
    with pytest.raises(ValueError, match="Both complete"):
        M.readings(fixture_logits(3)[0], bank())
