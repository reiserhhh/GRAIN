#!/usr/bin/env python3
"""Measurement construction and its distinct identification/transport limits."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_proxy_measurement as M

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hm252_test_generator", ROOT / "scripts/run_grain_proxy_measurement_hm252.py"
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
CONFIG = json.loads((ROOT / "configs/grain_proxy_measurement_hm252.json").read_text())
ALPHABET = np.eye(4)


def bundle(bank=0, coarse=False):
    """Supply only observed simulator laws to the actual constructor."""
    return M.compile_bundle(
        R.simulator_joint(CONFIG, bank, coarse), ALPHABET, CONFIG["targets"]
    )


def profile_inputs(index=0, reverse=False):
    """Generate evaluator-only author material, not a constructor parameter."""
    p = CONFIG["application_profiles"][index]
    w = np.asarray(CONFIG["w_channels"])
    if reverse:
        w = w[:, [2, 1, 0]]
    law = np.einsum("cu,cuw->cw", p["regimes"], w)
    return (
        np.repeat([0, 1], 4),
        np.tile(ALPHABET, (2, 1)),
        (np.asarray(p["choice"])[:, None] * law).flatten(),
    )


def test_registered_alphabet_only():
    assert np.array_equal(M.decode(ALPHABET, ALPHABET), range(4))
    with pytest.raises(ValueError, match="Out-of-alphabet"):
        M.decode(np.ones((1, 4)), ALPHABET)


def test_distinct_pair_ledger_and_census():
    j = R.simulator_joint(CONFIG)
    records = R.census_records(j, ALPHABET, 0)
    assert np.allclose(M.observed_joint(**records, alphabet=ALPHABET), j)
    records["units"][:, 0] = records["units"][:, 1]
    with pytest.raises(ValueError, match="distinct events"):
        M.observed_joint(**records, alphabet=ALPHABET)


def test_cold_scores_without_hidden_regimes_or_application_outcomes():
    b = bundle()
    reading = M.cold_profile(b, *profile_inputs())
    p = CONFIG["application_profiles"][0]
    expected = np.einsum(
        "c,cu,cuq->q", p["choice"], p["regimes"], CONFIG["behavior_channels"]
    )
    assert np.allclose([t["absolute"] for t in reading["targets"]], expected)
    assert set(b) == {
        "version",
        "target_ids",
        "alphabet",
        "conditions",
        "reference",
        "contract",
    }
    bad = copy.deepcopy(b)
    bad["latent_regimes"] = []
    with pytest.raises(ValueError, match="schema"):
        M.cold_profile(bad, *profile_inputs())


def test_bridge_not_unique_but_author_functional_is():
    program = bundle()["conditions"][0]
    assert program["rank"] == 3
    assert np.asarray(program["null_basis"]).shape == (4, 1)
    p = np.asarray(CONFIG["application_profiles"][0]["regimes"][0]) @ np.asarray(
        CONFIG["w_channels"][0]
    )
    original = M.functional_reading(program, p)
    varied = copy.deepcopy(program)
    varied["representative"] = (
        np.asarray(program["representative"])
        + 10 * np.asarray(program["null_basis"])[:, :1]
    ).tolist()
    alternate = M.functional_reading(varied, p)
    for a, b in zip(original, alternate, strict=True):
        assert a["value"] == pytest.approx(b["value"])
        assert a["value"] == pytest.approx(a["adjoint_value"])


def test_foreign_profile_keeps_ambiguity_not_minimum_norm_score():
    reading = M.functional_reading(bundle()["conditions"][0], [1, 0, 0, 0])
    assert all(
        x["status"] == "UNIDENTIFIED_FUNCTIONAL" and x["value"] is None for x in reading
    )
    assert all(
        abs(x["feasible_witness_values"][1] - x["feasible_witness_values"][0]) > 0.01
        for x in reading
    )


def test_unrepresentable_behavior_is_no_bridge_not_zero():
    b = bundle(coarse=True)
    assert all(
        status == "NO_MOMENT_BRIDGE" for p in b["conditions"] for status in p["status"]
    )
    reading = M.cold_profile(b, *profile_inputs())
    assert all(t["absolute"] is None for t in reading["targets"])


def test_source_mixture_shift_does_not_change_bridge_mean():
    for index in range(6):
        readings = [
            M.cold_profile(bundle(bank), *profile_inputs(index)) for bank in range(2)
        ]
        assert np.allclose(
            [t["absolute"] for t in readings[0]["targets"]],
            [t["absolute"] for t in readings[1]["targets"]],
        )
    ordinary = [
        M.cold_profile(bundle(bank), *profile_inputs())["targets"][0]["ordinary"]
        for bank in range(2)
    ]
    assert abs(ordinary[0] - ordinary[1]) > 0.05


def test_condition_choice_is_kept_not_centered_away():
    a = M.cold_profile(bundle(), *profile_inputs(0))
    b = M.cold_profile(bundle(), *profile_inputs(2))
    assert a["choice"] == pytest.approx([0.8, 0.2])
    assert b["choice"] == pytest.approx([0.2, 0.8])
    assert abs(a["targets"][0]["absolute"] - b["targets"][0]["absolute"]) > 0.1


def test_changed_channel_can_pass_moments_but_break_behavior_meaning():
    stable = M.cold_profile(bundle(), *profile_inputs())
    changed = M.cold_profile(bundle(), *profile_inputs(reverse=True))
    assert all(
        t["status"] == "IDENTIFIED_MOMENT_FUNCTIONAL" for t in changed["targets"]
    )
    assert (
        abs(changed["targets"][0]["absolute"] - stable["targets"][0]["absolute"]) > 0.2
    )


def test_missing_and_invalid_inputs_not_filled_with_fake_support():
    joint = R.simulator_joint(CONFIG)
    joint[:, 0] = 0
    joint /= joint.sum()
    with pytest.raises(ValueError, match="Missing Z"):
        M.compile_bundle(joint, ALPHABET, CONFIG["targets"])
    with pytest.raises(ValueError, match="Positive within"):
        M.functional_reading(bundle()["conditions"][0], [-0.1, 0.3, 0.3, 0.5])


def test_renaming_vector_coordinates_preserves_registered_laws():
    b = bundle()
    transformed = copy.deepcopy(b)
    matrix = np.array([[1, 2, 0, 0], [0, 2, 1, 0], [0, 0, 3, 1], [0, 0, 0, 4]])
    transformed["alphabet"] = (ALPHABET @ matrix).tolist()
    c, v, weights = profile_inputs()
    old = M.cold_profile(b, c, v, weights)
    new = M.cold_profile(transformed, c, v @ matrix, weights)
    assert old == new
