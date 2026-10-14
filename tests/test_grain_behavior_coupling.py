#!/usr/bin/env python3
"""Joint operational meaning beyond identical marginal author scores."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_behavior_coupling as M
from scripts import run_grain_behavior_coupling_hm276 as R

CFG = json.loads((Path(__file__).resolve().parents[1] / R.CONFIG).read_text())
ALPHABET = np.eye(4)


def fixture(criterion=0, bank=0):
    """Give the constructor only a complete observed joint law."""
    item = CFG["criterion_laws"][criterion]
    z, w, y = R.channels(CFG, item["conditional_joint_adjustment"])
    joint = np.einsum("u,uz,uw,uy->zwy", CFG["source_priors"][bank], z, w, y)
    return M.compile_bundle(joint, ALPHABET, item["id"], np.ones(4) @ w / 4), w, y


def test_joint_source_census_and_ledger():
    _, w, y = fixture()
    z, _, _ = R.channels(CFG, 0.04)
    j = np.einsum("u,uz,uw,uy->zwy", CFG["source_priors"][0], z, w, y)
    records = R.census_records(j, ALPHABET, "test")
    assert np.allclose(M.observed_joint(**records, alphabet=ALPHABET), j)
    records["units"][:, 0] = records["units"][:, 1]
    with pytest.raises(ValueError, match="Distinct"):
        M.observed_joint(**records, alphabet=ALPHABET)


@pytest.mark.parametrize("criterion", [0, 1])
@pytest.mark.parametrize("bank", [0, 1])
def test_identical_means_different_joint_behavior(criterion, bank):
    b, w, y = fixture(criterion, bank)
    for k in CFG["author_joint_parameters"]:
        truth = R.author_prior(k) @ y
        read = M.cold_profile(b, R.record(R.author_prior(k) @ w, ALPHABET, "a"))[
            "reading"
        ]
        assert read["marginal_means"] == pytest.approx([0.5, 0.5])
        assert read["probabilities_00_01_10_11"] == pytest.approx(truth)
        assert read["coupling"] == pytest.approx(
            (0.04 + CFG["criterion_laws"][criterion]["conditional_joint_adjustment"])
            * k
        )
        assert read["paired_single_weight_product_coupling"] == pytest.approx(0.04 * k)


def test_same_geometry_can_have_opposite_behavioral_meaning():
    b0, w0, _ = fixture(0)
    b1, w1, _ = fixture(1)
    assert np.array_equal(w0, w1)
    record = R.record(R.author_prior(0.6) @ w0, ALPHABET, "a")
    a, b = [M.cold_profile(x, record)["reading"] for x in (b0, b1)]
    assert a["coupling"] == pytest.approx(0.048)
    assert b["coupling"] == pytest.approx(-0.024)


def test_relink_loses_coupling_without_changing_marginals():
    b, w, _ = fixture()
    paired = (R.author_prior(0.6) @ w).reshape(2, 2)
    relink = np.outer(paired.sum(1), paired.sum(0))
    assert np.allclose(paired.sum(0), relink.sum(0))
    assert np.allclose(paired.sum(1), relink.sum(1))
    original = M.cold_profile(b, R.record(paired.flatten(), ALPHABET, "o"))["reading"]
    changed = M.cold_profile(b, R.record(relink.flatten(), ALPHABET, "r"))["reading"]
    assert original["coupling"] == pytest.approx(0.048)
    assert changed["coupling"] == pytest.approx(0, abs=1e-12)


def test_cold_excludes_latent_outcomes_and_role_swap():
    b, w, _ = fixture()
    r = R.record(R.author_prior(0.6) @ w, ALPHABET, "a")
    bad = copy.deepcopy(r)
    bad["outcomes"] = []
    with pytest.raises(ValueError, match="excludes"):
        M.cold_profile(b, bad)
    bad = copy.deepcopy(r)
    bad["vectors"] = np.asarray(bad["vectors"])[:, ::-1].tolist()
    with pytest.raises(ValueError, match="role-bound"):
        M.cold_profile(b, bad)
    bad = copy.deepcopy(r)
    bad["frame"] = ["other", "frame"]
    with pytest.raises(ValueError, match="frame"):
        M.cold_profile(b, bad)


def test_event_weights_not_probabilities_and_no_clipping():
    b, _, _ = fixture(criterion=1)
    h = np.asarray(b["bridge"]["representative"])
    assert np.any(h < 0)
    result = M.read_law(b["bridge"], [1, 0, 0, 0])
    assert result["status"] == "INADMISSIBLE_JOINT_READING"
    assert min(result["unclipped_cells"]) < 0


def test_cold_copied_event_and_foreign_alphabet_rejected():
    b, w, _ = fixture()
    r = R.record(R.author_prior(0.6) @ w, ALPHABET, "a")
    bad = copy.deepcopy(r)
    bad["units"][0][1] = bad["units"][0][0]
    with pytest.raises(ValueError, match="Distinct"):
        M.cold_profile(b, bad)
    bad = copy.deepcopy(r)
    bad["vectors"][0][0] = [2, 0, 0, 0]
    with pytest.raises(ValueError, match="Out-of-alphabet"):
        M.cold_profile(b, bad)
