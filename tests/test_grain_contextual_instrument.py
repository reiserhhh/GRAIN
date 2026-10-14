#!/usr/bin/env python3
"""Common instrument math, event identity, sampling clock and source-only scores."""

import copy

import numpy as np
import pytest

from grain_research.contextual_instrument import (
    evidence_rows,
    gaussian_readout,
    process_model,
    score_records,
    text_key,
)


def fixture(kind="conditional_profile"):
    """Known numeric readout fixture, not a mock human personality questionnaire."""
    spec = {
        "id": "test",
        "kind": kind,
        "query": "q",
        "clock_unit": "round",
        "allowed_scope": "self_event_report",
        "target_description": "test expression",
    }
    if kind == "conditional_profile":
        main = {
            "mean": [0.1, -0.1],
            "covariance": [[0.2, 0.03], [0.03, 0.15]],
            "noise": [0.3, 0.4],
        }
        spec["target_names"] = ["level", "condition", "low_condition", "high_condition"]
    else:
        main = {
            "mean": [0.1, 0.2, -0.1],
            "covariance": np.diag([0.1, 0.08, 0.03]).tolist(),
            "noise": 0.05,
            "rate": 0.3,
            "stationary": 0.06,
        }
        spec["target_names"] = [
            "level",
            "condition",
            "partner",
            "current_state",
            "current_response",
            "window_state",
        ]
    bundle = {
        "spec": spec,
        "main": main,
        "bootstrap": [copy.deepcopy(main) for _ in range(3)],
        "pilot_reports_per_occasion": 2,
        "calibration": {"conditional": True, "coef": [0.0, 1.0, 0.0, 0.0]},
    }
    rows, margins = [], {}
    for t in range(8):
        for rep in range(2):
            text = f"event {t} report {rep}"
            rows.append(
                {
                    "author": "a",
                    "event": f"{t}:{rep}",
                    "text": text,
                    "scope": "self_event_report",
                    "condition": (-1, 1)[t % 2],
                    "time": float(t),
                    "partner": (-1.0, 1.0)[(t // 2) % 2],
                    "replicate": rep,
                }
            )
            margins[text_key("q", text)] = 0.07 * t + 0.03 * rep
    return bundle, rows, margins


def test_gaussian_kernel_matches_precision_solution():
    rng = np.random.default_rng(33)
    a = rng.normal(size=(5, 5))
    s, mean = a @ a.T + np.eye(5), rng.normal(size=5)
    h, r, y = rng.normal(size=(8, 5)), rng.uniform(0.2, 1, 8), rng.normal(size=8)
    result = gaussian_readout(mean, s, h, r)
    post = np.linalg.inv(np.linalg.inv(s) + h.T @ np.diag(1 / r) @ h)
    expected = post @ (np.linalg.solve(s, mean) + h.T @ (y / r))
    np.testing.assert_allclose(
        result["offset"] + result["kernel"] @ y, expected, atol=1e-12
    )
    np.testing.assert_allclose(result["covariance"], post, atol=1e-12)


@pytest.mark.parametrize("kind", ["conditional_profile", "occasion_process"])
def test_score_accounting_repeat_and_batch_invariance(kind):
    bundle, rows, margins = fixture(kind)
    single = score_records(rows, margins, bundle, clock_unit="round")[0]["a"]
    duplicate = score_records(
        rows + [rows[0].copy()] * 4, margins, bundle, clock_unit="round"
    )[0]["a"]
    batch = score_records(
        rows + [dict(r, author="b") for r in rows], margins, bundle, clock_unit="round"
    )[0]["a"]
    for name, target in single["targets"].items():
        assert target == duplicate["targets"][name] == batch["targets"][name]
        np.testing.assert_allclose(
            target["source_term"] + sum(r["contribution"] for r in target["evidence"]),
            target["score"],
            atol=1e-12,
        )
        assert target["construction_variance"] < 1e-25
        assert target["perfect_repeat_gain"] == 0


def test_clock_relabeling_is_not_input_order():
    bundle, rows, margins = fixture("occasion_process")
    original = score_records(rows, margins, bundle, clock_unit="round")[0]["a"]
    reordered = score_records(
        list(reversed(rows)), margins, bundle, clock_unit="round"
    )[0]["a"]
    relabeled = score_records(
        [dict(r, time=7 - r["time"]) for r in rows], margins, bundle, clock_unit="round"
    )[0]["a"]
    for name in original["targets"]:
        np.testing.assert_allclose(
            original["targets"][name]["score"],
            reordered["targets"][name]["score"],
            atol=1e-12,
        )
    assert (
        abs(
            original["targets"]["current_state"]["score"]
            - relabeled["targets"]["current_state"]["score"]
        )
        > 0.01
    )
    with pytest.raises(ValueError):
        score_records(rows, margins, bundle, clock_unit="unix_seconds")


def test_missing_cell_and_unqualified_are_not_zero():
    bundle, rows, margins = fixture()
    onecell = score_records(
        [r for r in rows if r["condition"] == -1], margins, bundle, clock_unit="round"
    )[0]["a"]
    assert onecell["empirically_absent_condition"] and onecell["counts"] == [8, 0]
    none = score_records(
        [dict(r, scope="unknown") for r in rows], {}, bundle, clock_unit="round"
    )[0]["a"]
    assert none["targets"] == {} and none["status"] == "insufficient_qualified_evidence"


def test_conflicting_identity_or_latent_fields_rejected():
    bundle, rows, margins = fixture()
    with pytest.raises(ValueError):
        evidence_rows(rows + [dict(rows[0], time=19)], margins, bundle)
    with pytest.raises(ValueError):
        evidence_rows([dict(rows[0], truth=0.9)], margins, bundle)


def test_fewer_process_reports_increase_conditional_variance():
    bundle, rows, margins = fixture("occasion_process")
    evidence = evidence_rows(rows, margins, bundle)
    many = process_model(evidence, bundle["main"], 2)[0]
    few = process_model(
        [r for r in evidence if r["replicate"] == 0], bundle["main"], 2
    )[0]
    assert few["covariance"][-1, -1] > many["covariance"][-1, -1]


def test_affine_unit_change_moves_levels_not_dimensionless_positions():
    bundle, rows, margins = fixture()
    old = score_records(rows, margins, bundle, clock_unit="round")[0]["a"]
    moved = copy.deepcopy(bundle)
    c, d = 0.4, 1.7
    moved["calibration"]["coef"] = [-c / d, 1 / d, 0.0, 0.0]
    for fitted in [moved["main"], *moved["bootstrap"]]:
        fitted["mean"] = (c + d * np.array(fitted["mean"])).tolist()
        fitted["covariance"] = (d * d * np.array(fitted["covariance"])).tolist()
        fitted["noise"] = (d * d * np.array(fitted["noise"])).tolist()
    new = score_records(rows, margins, moved, clock_unit="round")[0]["a"]
    for name in old["targets"]:
        o, n = old["targets"][name], new["targets"][name]
        np.testing.assert_allclose(
            n["score"], d * o["score"] + (0 if name == "condition" else c), atol=1e-12
        )
        np.testing.assert_allclose(
            n["source_relative_position"], o["source_relative_position"], atol=1e-12
        )
