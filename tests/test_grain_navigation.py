#!/usr/bin/env python3
"""Controlled-world units, factorial pairing and immutable reader replay tests."""

import copy

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.navigation import (
    build_worlds,
    load_functions,
    quality,
    reference_world_ids,
    render_world,
    validate_worlds,
    write_json,
)
from scripts.run_grain_navigation_baseline import (
    checked_arrays,
    paired_effects,
    save_unit,
)


def test_worlds_have_scene_level_split_and_no_human_targets():
    worlds = build_worlds()
    assert len(worlds) == 96
    assert len({tuple(w["texts"]) for w in worlds}) == 96
    assert {
        s: sum(w["split"] == s for w in worlds)
        for s in ("construction", "development", "held_scene")
    } == {"construction": 48, "development": 24, "held_scene": 24}
    assert all(
        "personality_score" not in w and not w["world_facts_are_personality_labels"]
        for w in worlds
    )


def test_references_are_fixed_construction_only():
    worlds = {w["world_id"]: w for w in build_worlds()}
    assert len(reference_world_ids()) == 3
    for refs in reference_world_ids().values():
        assert len(refs) == 2
        assert all(worlds[w]["split"] == "construction" for w in refs)


@pytest.mark.parametrize("change", ["split", "duplicate", "truth"])
def test_reject_invalid_worlds(change):
    worlds = copy.deepcopy(build_worlds())
    if change == "split":
        worlds[0]["split"] = "held_scene"
    elif change == "duplicate":
        worlds[0] = worlds[1]
    else:
        worlds[0]["world_facts_are_personality_labels"] = True
    with pytest.raises(ValueError):
        validate_worlds(worlds)


def test_role_rendering_keeps_action_but_changes_ownership():
    own = render_world(0, "self", "voluntary", 0)
    other = render_world(0, "other", "voluntary", 0)
    assert (
        own[0]
        .replace("I introduced myself", "My colleague introduced themselves")
        .replace(" I had", " they had")
        == other[0]
    )
    assert "said they felt" in render_world(4, "other", "voluntary", 0)[0]
    assert own[1] == other[1]
    with pytest.raises(ValueError):
        render_world(0, "invalid", "voluntary", 0)


def test_factorial_response_has_exact_complete_partners():
    rows = []
    for w in build_worlds():
        value = (
            2 * (w["actor"] == "self")
            + 3 * (w["opportunity"] == "voluntary")
            + w["realization"]
        )
        rows.append(
            {
                k: w[k]
                for k in ("scene_id", "split", "actor", "opportunity", "realization")
            }
            | {
                "query": "q",
                "reference": "r",
                "native_score": value,
                "score": value,
                "mean": value,
                "configuration": 0,
                "final": 0,
            }
        )
    frame = pd.DataFrame(rows)
    pairs = paired_effects(frame)
    assert len(pairs) == 144
    for axis, delta in (("actor", 2), ("opportunity", 3), ("realization", 1)):
        assert (pairs.loc[pairs.axis == axis, "score_delta"] == delta).all()
    with pytest.raises(ValueError, match="Incomplete"):
        paired_effects(frame.iloc[1:])


def test_unit_hash_rejects_changed_arrays(tmp_path):
    path = tmp_path / "unit.json"
    save_unit(path, {"index": 0}, P=np.zeros((2, 3)), Q=np.ones(3))
    meta, arrays = checked_arrays(path)
    assert meta["index"] == 0 and arrays["P"].shape == (2, 3)
    np.savez_compressed(path.with_suffix(".npz"), P=np.ones((2, 3)))
    with pytest.raises(ValueError, match="changed"):
        checked_arrays(path)


def test_fixed_two_projection_formula_replays(tmp_path):
    path = tmp_path / "functions.npz"
    np.savez_compressed(
        path,
        q__Z=np.zeros((2, 3)),
        q__h=np.zeros(3),
        q__G=np.ones((2, 3)),
        q__g=np.ones(3) * 2,
        q__b=0.5,
    )
    function = load_functions(path, queries=("q",))["q"]
    result = function.parts(np.ones((2, 3)), np.ones(3))
    assert result["score"] == pytest.approx(12.5)
    assert result["mean"] + result["configuration"] == pytest.approx(6)


def test_quality_constant_is_unavailable_not_zero_correlation():
    assert quality([1, 1, 1], [1, 2, 3])["r"] is None
    assert quality([1, 2, 3], [1, 2, 3])["r"] == pytest.approx(1)
    with pytest.raises(ValueError):
        quality([1, 2, 3], [1, 2])


def test_json_rejects_nan_and_preserves_old_commit(tmp_path):
    path = tmp_path / "value.json"
    write_json(path, {"a": 1})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"a": np.nan})
    assert path.read_bytes() == original
