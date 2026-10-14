#!/usr/bin/env python3
"""Synthetic mechanism tests, not tests of human psychological validity."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_choice_evidence as M
from scripts import run_grain_choice_set_hm135 as R


def test_unchosen_option_changes_likelihood_without_changing_chosen_act():
    """The registered simple choice law yields opposite conditional evidence."""
    e = R.example()
    assert e["low_option_open"] == pytest.approx(1)
    assert e["high_option_open"] == pytest.approx(-1)
    assert e["low_option_closed"] == e["high_option_closed"] == 0


def test_unavailable_utility_has_no_effect():
    """An absent alternative is not in the partition function."""
    assert M.choice_log_likelihood([[0], [100]], [5], 0, [True, False]) == 0


def test_uniform_shift_and_option_permutation():
    """Choice likelihood uses relative utility, not arbitrary coordinates."""
    f, t = np.array([[0, 1], [1, 1], [-1, 1]]), [2, 3]
    value = M.choice_log_likelihood(f, t, 0, [True, True, True])
    assert value == pytest.approx(
        M.choice_log_likelihood(f, [2, -90], 0, [True, True, True])
    )
    assert value == pytest.approx(
        M.choice_log_likelihood(f[[2, 0, 1]], t, 1, [True, True, True])
    )


def test_finite_log_likelihood_for_large_utilities():
    """Stable log partition handles large, finite utility differences."""
    assert M.choice_log_likelihood(
        [[1000], [-1000]], [1], 1, [True, True]
    ) == pytest.approx(-2000)


@pytest.mark.parametrize(
    "mask,chosen", [([False, True], 0), ([1, 1], 0), ([True], 0), ([True, True], 3)]
)
def test_invalid_choice_support(mask, chosen):
    """Do not impute an unavailable or malformed chosen event."""
    with pytest.raises(ValueError):
        M.choice_log_likelihood([[0], [1]], [1], chosen, mask)


def test_complete_character_and_nonspecific_controls():
    """Availability interaction is distinct from a generic option cue or level."""
    x = np.array([[3 - 2 - 5 + 7, 3 - 2 + 5 - 7], [3 + 2 - 5 - 7, 3 + 2 + 5 + 7]])
    e = M.opportunity_effects(x)
    assert e.tolist() == [-10, 18, 10, 28]
    assert M.opportunity_effects([[0, 10], [3, 13]])[-1] == 0
    alternate = sum(
        b * c * x[(b + 1) // 2, (c + 1) // 2]
        for b, c in itertools.product((-1, 1), repeat=2)
    )
    assert alternate == e[-1]


def test_registered_performed_action_is_identical():
    """Materials change options, not the act performed by the person."""
    spec = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/grain_choice_set_hm135.json"
        ).read_text()
    )
    M.validate_material_spec(spec)
    spec["banks"][0]["behaviours"][0][1] = (
        "Changed act. Unused option: a different option"
    )
    with pytest.raises(ValueError):
        M.validate_material_spec(spec)


def test_engine_binding_is_restored():
    """The scoped adapter does not change old experiment globals permanently."""
    original = R.R.SPEC, R.R.CODE
    with R.engine():
        assert R.R.SPEC == R.SPEC
        assert all(p in R.R.CODE for p in R.CODE)
    assert (R.R.SPEC, R.R.CODE) == original


def test_cube_field_reconstruction_and_no_constant_success():
    """Every corner/mode is required; a constant field has no mechanism effect."""
    cube = np.zeros((2, 6, 2, 2, 4, 2, 7, 4))
    effects, parts, fields = R.derive(cube)
    assert np.array_equal(parts.sum(axis=4), cube[:, :, :, :, 3] - cube[:, :, :, :, 0])
    assert not effects.any()
    assert all(f["K_V_joint_signed_projections"] == [None, None, None] for f in fields)
    cube[0, 0, 1, 1, 0, 0, 0, 0] = 1
    assert R.derive(cube)[0][0, 0, 3, 0, 0, 0] == 1
