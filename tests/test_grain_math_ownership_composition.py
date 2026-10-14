#!/usr/bin/env python3
"""Mechanism checks: group parity is not a prediction-accuracy objective."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from grain_research.construction import math_ownership_composition as M


def test_complete_material_has_every_directed_pair_and_balanced_owners():
    spec = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/grain_math_passive_event_composition_v1.json"
        ).read_text()
    )
    docs = M.documents(spec)
    assert len(docs) == len({d["body"] for d in docs}) == 576
    groups = {}
    for d in docs:
        key = (d["family"], d["focal"], d["companion"], *d["coordinates"][:2])
        groups.setdefault(key, []).append(d)
    for pair in groups.values():
        assert len(pair) == 2
        assert Counter(pair[0]["body"].split()) == Counter(pair[1]["body"].split())
        assert pair[0]["body"] != pair[1]["body"]


def test_mixed_ownership_action_equals_eight_character_coefficient():
    values = np.random.default_rng(1).normal(size=(8, 6))
    effects = M.group_effects(values)
    assert effects["identity_error"] < 1e-14
    assert np.allclose(effects["gate"], 8 * effects["coefficients"][7])


def test_even_odd_projection_reconstructs_and_swap_changes_only_odd():
    values = np.random.default_rng(2).normal(size=(8, 4))
    a = M.tensor(values)
    e = M.group_effects(values)
    swapped = M.group_effects(a[:, :, ::-1].reshape(8, 4))
    assert np.allclose(a[:, :, 1], e["even"] + e["odd"])
    assert np.allclose(a[:, :, 0], e["even"] - e["odd"])
    assert np.allclose(swapped["even"], e["even"])
    assert np.allclose(swapped["odd"], -e["odd"])
    pure_even = np.stack((e["even"], e["even"]), axis=2).reshape(8, 4)
    assert np.allclose(M.group_effects(pure_even)["odd"], 0)


def test_parseval_and_mixed_difference_compatibility():
    v = np.random.default_rng(3).normal(size=(8, 6))
    e = M.group_effects(v)
    assert np.square(v).sum() / 8 == pytest.approx(np.square(e["coefficients"]).sum())
    a = M.tensor(v)
    dx = a[1] - a[0]
    dy = a[:, 1] - a[:, 0]
    assert np.allclose(dx[1] - dx[0], dy[1] - dy[0])


def test_qualified_and_generic_mechanisms_are_distinguished():
    qualified = M.group_effects(M.toy_world("qualified", 4)["main_logits"])
    generic = M.group_effects(M.toy_world("generic", 4)["main_logits"])
    assert np.linalg.norm(qualified["gate"]) > 0.1
    assert np.linalg.norm(qualified["other"]) < 1e-14
    assert np.linalg.norm(generic["own"]) > 0.1
    assert np.linalg.norm(generic["gate"]) < 1e-14


def test_neutral_control_cancels_matched_role_marker_not_qualified_action():
    for kind in ("role_marker", "qualified"):
        w = M.toy_world(kind, 5)
        a = M.group_effects(w["main_logits"])["gate"]
        b = M.group_effects(w["neutral_logits"])["gate"]
        assert (np.linalg.norm(a - b) < 1e-14) == (kind == "role_marker")


def test_link_only_can_produce_native_ownership_interaction():
    z = M.toy_world("link_only", 6)["main_logits"]
    assert np.linalg.norm(M.group_effects(z)["gate"]) < 1e-14
    assert np.linalg.norm(M.group_effects(M.decode(z)["score"])["gate"]) > 1e-4


def test_link_control_keeps_role_specific_slopes_but_removes_both_cross_terms():
    z = np.random.default_rng(7).normal(size=(8, 6, 10))
    clean, decoded = M.link_controls(z)
    original, corrected = M.group_effects(z), M.group_effects(clean)
    assert np.linalg.norm(corrected["own"]) < 1e-14
    assert np.linalg.norm(corrected["other"]) < 1e-14
    assert np.allclose(original["coefficients"][:6], corrected["coefficients"][:6])
    assert set(decoded) == {"score", "log_odds"}


@pytest.mark.parametrize("values", [np.zeros((7, 6)), np.full((8, 6), np.nan)])
def test_incomplete_nonfinite_inputs_fail(values):
    with pytest.raises(ValueError):
        M.group_effects(values)
