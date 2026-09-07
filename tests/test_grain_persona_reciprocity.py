"""No-download tests for reciprocal intervention material and aggregation."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grain_research.vector_mechanisms.reciprocity import (
    analysis_tables,
    arm_schedule,
    build_material,
    fit_directions,
    material_identity,
    random_direction,
    validate_material,
    weighted_effect_summary,
)
from scripts.run_grain_persona_reciprocity import (
    EXPECTED_METADATA,
    ROOT,
    run,
    validate_config,
)


def test_complete_material_and_disjoint_scenarios():
    rows = build_material()
    assert len(rows) == 120
    dev = [r for r in rows if r["split"] == "development"]
    test = [r for r in rows if r["split"] == "evaluation"]
    assert len(dev) == 24 and len(test) == 96
    assert len({r["group_id"] for r in dev}) == 6
    assert not {r["group_id"] for r in dev} & {r["group_id"] for r in test}
    assert not any("extravert" in r["evidence"] for r in test if r["task"] == "R")


def test_ownership_swap_keeps_action_word_bag():
    pair = [
        r
        for r in build_material()
        if r["group_id"] == "garden" and r["variant"] == "R_owner"
    ]
    assert Counter(pair[0]["evidence"].split()) == Counter(pair[1]["evidence"].split())
    assert pair[0]["evidence"] != pair[1]["evidence"]


def test_negative_question_inverts_answer_mapping_not_semantic_direction():
    rows = build_material()
    row = next(r for r in rows if r["variant"] == "R_negative" and r["polarity"] == 1)
    assert row["high_answer"] == "No" and row["low_answer"] == "Yes"
    assert row["expected_sign"] == 1
    assert all(r["expected_sign"] == 1 for r in rows if r["variant"] == "R_fact")


def test_material_validation_and_identity_cover_truth_and_duplicates():
    rows = build_material(smoke=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_material(rows + [rows[0]])
    with pytest.raises(ValueError, match="Incomplete"):
        validate_material(rows[1:])
    modified = copy.deepcopy(rows)
    modified[0]["high_answer"] += " "
    assert material_identity(rows) != material_identity(modified)


def test_direction_fit_never_reads_evaluation_states():
    rows = build_material()
    states = {
        r["case_id"]: {"block_16": np.array([[r["polarity"], 2 * r["polarity"], 3.0]])}
        for r in rows
        if r["split"] == "development"
    }
    fitted = fit_directions(rows, states, ["block_16"])
    assert set(fitted) == {"G__block_16", "R__block_16"}
    np.testing.assert_array_equal(fitted["G__block_16"], [2, 4, 0])
    for value in states.values():
        value["block_16"][:] = 0
    with pytest.raises(ValueError, match="Degenerate"):
        fit_directions(rows, states, ["block_16"])


def test_random_controls_have_fixed_seed_and_equal_norm():
    v = np.arange(8.0)
    random = random_direction(v, "source:site:1")
    np.testing.assert_array_equal(random, random_direction(v, "source:site:1"))
    assert np.linalg.norm(random) == pytest.approx(np.linalg.norm(v))
    assert not np.allclose(random, random_direction(v, "source:site:2"))
    arms = arm_schedule([-1, 0.5, 1], 2)
    assert len(arms) == 7
    assert len(set(arms)) == len(arms)


def test_group_aggregation_preserves_heterogeneity_and_effective_authors():
    result = weighted_effect_summary([[1, 2], [-1, -2]], [1, 1])
    np.testing.assert_array_equal(result["mean"], [0, 0])
    np.testing.assert_array_equal(result["population_covariance"], [[1, 2], [2, 4]])
    assert result["effective_authors"] == 2
    np.testing.assert_array_equal(result["positive_weight_fraction"], [0.5, 0.5])
    assert weighted_effect_summary([[1], [2]], [1, 0])["effective_authors"] == 1


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [1], [np.nan, 1]])
def test_group_summary_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        weighted_effect_summary([[1], [2]], weights)


def test_summary_uses_semantic_signed_pairs_and_scenario_ci():
    baseline, effects = [], []
    for group in ("g1", "g2"):
        for polarity in (-1, 1):
            case = f"{group}_{polarity}"
            baseline.append(
                {
                    "case_id": case,
                    "group_id": group,
                    "variant": "R_negative",
                    "split": "evaluation",
                    "polarity": polarity,
                    "expected_sign": polarity,
                    "margin": polarity * 3,
                }
            )
            for kind, replica, slope in (
                ("signal", -1, 2),
                ("random", 0, 0.25),
                ("random", 1, -0.25),
            ):
                for dose in (-1.0, 1.0):
                    effects.append(
                        {
                            "case_id": case,
                            "group_id": group,
                            "variant": "R_negative",
                            "source": "G",
                            "site": "block_16",
                            "kind": kind,
                            "replica": replica,
                            "dose": dose,
                            "margin": polarity * 3 + slope * dose,
                        }
                    )
    cap, matrix = analysis_tables(pd.DataFrame(baseline), pd.DataFrame(effects))
    assert cap.iloc[0].accuracy == 1
    assert cap.iloc[0].high_minus_low_gap == 6
    assert matrix.iloc[0].signal_slope == 2
    assert matrix.iloc[0].random_slope == 0
    assert matrix.iloc[0].paired_ci_low == 2
    assert matrix.iloc[0].n_scenarios == 2


def test_cached_file_fingerprints_are_sha256_and_config_is_bounded():
    assert all(len(v) == 64 and int(v, 16) > 0 for v in EXPECTED_METADATA.values())
    config = json.loads(
        (ROOT / "configs/grain_persona_reciprocity_v1.json").read_text()
    )
    validate_config(config)
    bad = copy.deepcopy(config)
    bad["doses"] = [-1, 1, float("nan")]
    with pytest.raises(ValueError):
        validate_config(bad)


def test_prepare_is_offline_replayable_and_will_not_overwrite(tmp_path):
    args = SimpleNamespace(
        config=ROOT / "configs/grain_persona_reciprocity_v1.json",
        model_dir=tmp_path / "nonexistent_model",
        output_dir=tmp_path / "prepared",
        smoke=True,
        prepare_only=True,
    )
    run(args)
    assert (
        json.loads((args.output_dir / "run_status.json").read_text())["status"]
        == "PREPARED_ONLY"
    )
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    assert not manifest["private_data_read"]
    assert (args.output_dir / "source_snapshot.zip").exists()
    with pytest.raises(ValueError, match="Fresh"):
        run(args)


def test_read_write_pairing_survives_coordinate_change_and_cancellation():
    r, v = np.array([3.0, 5.0]), np.array([2.0, -1.0])
    transform = np.array([[2.0, 1.0], [0.0, 3.0]])
    assert np.linalg.solve(transform.T, r) @ (transform @ v) == pytest.approx(r @ v)
    a = np.array([1.0, -1.0])
    assert np.mean(a * a) == 1 and np.mean(a) * np.mean(a) == 0
