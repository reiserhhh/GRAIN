#!/usr/bin/env python3
"""Numeric eligibility/history tests without private data or source refitting."""

import copy

import numpy as np
import pytest

from grain_research.gptpro_foundation_v3 import f16_evidence_ledger as ledger
from grain_research.gptpro_foundation_v3 import f16_t1_source as t1


@pytest.fixture
def example():
    """A declared four-cell unit fixture, not an empirical source fit."""
    model = {"mu": [-0.4, -0.1, 0.1, 0.4], "Sigma": np.eye(4).tolist(), "sigma2": 0.2}
    base = {
        "target_id": "T1",
        "world_contract": "F04_L_four_condition",
        "reader_hash": t1.READER_HASH,
        "formula_version": "T1_four_cell_source_mixture_v1",
        "scorer_code_hash": t1.code_hash(),
        "seed": 20260915,
        "source_models": [model],
        "source_bootstraps": 1,
        "personal_draws": 12,
        "reference_mu": 0.0,
        "reference_sigma": 1.0,
    }
    bundle = ledger.build_bundle(base)
    history = [
        {
            "record_id": f"r{i}",
            "op": "new",
            "recorded_at": i,
            "material_ref": f"fixture:{i}",
            "qualification": "actual",
            "observation": {
                "author_id": "a",
                "role": "application",
                "world": "L",
                "event_id": f"e{i}",
                "t": i,
                "z": z,
                "y": float(i) / 2,
            },
        }
        for i, z in enumerate(t1.CONDITIONS)
    ]
    raw = {
        "author_id": "a",
        "world": "L",
        "reader_hash": t1.READER_HASH,
        "target_time": 3,
        "history": history,
    }
    return bundle, raw


def operation(raw, kind, **kwargs):
    """Append a report at the fixed target cutoff."""
    identity = f"op{len(raw['history'])}"
    raw["history"].append(
        {
            "record_id": identity,
            "op": kind,
            "recorded_at": raw["target_time"],
            "material_ref": "edit:" + identity,
            **kwargs,
        }
    )
    return identity


def test_alias_summary_do_not_add_likelihood(example):
    bundle, raw = example
    original = ledger.score_evidence(raw, bundle)
    operation(raw, "alias", parent_id="r0")
    operation(raw, "summary", parents=["r0", "r3"])
    result = ledger.score_evidence(raw, bundle)
    assert result["result"] == original["result"]
    assert result["ledger"]["eligible_events"] == 4
    assert len(result["ledger"]["annotation_ids"]) == 2


def test_revision_replaces_and_restoring_value_recovers_score(example):
    bundle, raw = example
    original = ledger.score_evidence(raw, bundle)
    summary = operation(raw, "summary", parents=["r0", "r3"])
    alias = operation(raw, "alias", parent_id="r3")
    changed = {**raw["history"][3]["observation"], "y": 2.5}
    revision = operation(
        raw, "revise", parent_id="r3", observation=changed, qualification="negated"
    )
    after = ledger.score_evidence(raw, bundle)
    assert after["ledger"]["eligible_events"] == 4
    assert after["ledger"]["stale_derived"] == [summary, alias]
    assert after["result"]["score"] > original["result"]["score"]
    operation(
        raw,
        "revise",
        parent_id=revision,
        observation=raw["history"][3]["observation"],
        qualification="actual",
    )
    assert ledger.score_evidence(raw, bundle)["result"] == original["result"]


def test_retraction_equals_omitted_event_not_zero(example):
    bundle, raw = example
    operation(raw, "retract", parent_id="r3")
    result = ledger.score_evidence(raw, bundle)
    assert result["ledger"]["inactive_report_ids"] == ["r3"]
    assert result["result"]["support"] == "model_borrowed"
    direct = t1.cold_score(
        {
            "world_contract": bundle["scorer"]["world_contract"],
            "reader_hash": bundle["reader_hash"],
            "records": [r["observation"] for r in raw["history"][:3]],
        },
        bundle["scorer"],
    )
    assert result["result"] == direct


@pytest.mark.parametrize("qualification", ["quoted", "hypothetical", "unknown"])
def test_ineligible_is_unobserved_prior_not_measured_zero(example, qualification):
    bundle, raw = example
    for r in raw["history"]:
        r["qualification"] = qualification
    result = ledger.score_evidence(raw, bundle)
    assert result["result"]["prior_only"]
    assert result["result"]["score"] == pytest.approx(0.8)
    assert not result["measured_self_evidence_present"]
    assert len(result["ledger"]["excluded_active"]) == 4


@pytest.mark.parametrize(
    "field,value",
    [("t", 1), ("z", -1.0), ("event_id", "other"), ("author_id", "other")],
)
def test_revision_cannot_change_event_context(example, field, value):
    bundle, raw = example
    changed = {**raw["history"][3]["observation"], field: value}
    operation(
        raw, "revise", parent_id="r3", observation=changed, qualification="actual"
    )
    with pytest.raises(ValueError):
        ledger.score_evidence(raw, bundle)


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_id",
        "duplicate_event",
        "future",
        "unordered",
        "bad_parent",
        "invalid_value",
        "bad_qualification",
        "nonobject",
        "repeat",
        "truth",
        "mixed_role",
        "bad_summary",
    ],
)
def test_bad_records_fail_closed(example, damage):
    bundle, raw = example
    if damage in ("duplicate_id", "duplicate_event"):
        row = copy.deepcopy(raw["history"][-1])
        if damage == "duplicate_event":
            row["record_id"] = "another"
        raw["history"].append(row)
    elif damage == "future":
        raw["history"][-1]["recorded_at"] = 4
    elif damage == "unordered":
        raw["history"] = list(reversed(raw["history"]))
    elif damage == "bad_parent":
        operation(raw, "retract", parent_id="absent")
    elif damage == "invalid_value":
        raw["history"][-1]["observation"]["y"] = "1.2"
    elif damage == "bad_qualification":
        raw["history"][-1]["qualification"] = []
    elif damage == "nonobject":
        raw["history"].append(0)
    elif damage == "repeat":
        operation(raw, "repeat", parent_id="r0")
    elif damage == "truth":
        raw["history"][-1]["observation"]["truth"] = 0
    elif damage == "mixed_role":
        raw["history"][-1]["observation"]["role"] = "source"
        raw["history"][-1]["qualification"] = "quoted"
    else:
        operation(raw, "summary", parents=[["r0"]])
    with pytest.raises(ValueError):
        ledger.score_evidence(raw, bundle)


def test_authentication_and_no_implicit_revival(example):
    bundle, raw = example
    changed = copy.deepcopy(bundle)
    changed["scorer"]["reference_mu"] += 1
    with pytest.raises(ValueError):
        ledger.score_evidence(raw, changed)
    operation(raw, "retract", parent_id="r0")
    operation(raw, "alias", parent_id="r0")
    with pytest.raises(ValueError):
        ledger.score_evidence(raw, bundle)


@pytest.mark.parametrize(
    "name",
    [
        "full_plain",
        "full_history",
        "missing_plus",
        "missing_minus",
        "selection_high",
        "late_offset",
        "quoted_tail",
        "hypothetical_all",
    ],
)
def test_case_plan_matches_independent_observation_selection(example, name):
    """All case operations match a separate direct-observation derivation."""
    from scripts.audit_grain_gptpro_f16_evidence import expected_rows
    from scripts.run_grain_gptpro_f16_evidence import case_input

    bundle, raw = example
    frozen = copy.deepcopy(raw)
    config = {
        "tail_count": 2,
        "correction_delta": 0.25,
        "late_offset_delta": 1.0,
        "selection_threshold": 0.75,
    }
    changed = case_input(raw, {"name": name}, config, "case.json")
    result = ledger.score_evidence(changed, bundle)
    assert result["ledger"]["records"] == expected_rows(frozen, name, config)
    assert raw == frozen
    assert changed["target_time"] == raw["target_time"]


def test_mixture_audit_detects_false_score_even_when_count_matches(example):
    """Checking that a file/count exists must never replace score verification."""
    from scripts.audit_grain_gptpro_f16_evidence import check_mixture

    bundle, raw = example
    result = ledger.score_evidence(raw, bundle)
    config = {"source_bootstraps": 1, "atol": 1e-10, "rtol": 1e-8}
    assert check_mixture(result, bundle["scorer"], config) < 1e-12
    result["result"]["score"] += 0.2
    with pytest.raises(AssertionError):
        check_mixture(result, bundle["scorer"], config)


def test_delayed_report_does_not_change_event_time_and_future_is_rejected(example):
    bundle, raw = example
    raw["history"][0]["recorded_at"] = 3
    for r in raw["history"][1:]:
        r["recorded_at"] = 3
    result = ledger.score_evidence(raw, bundle)
    assert [r["t"] for r in result["ledger"]["records"]] == [0, 1, 2, 3]
    raw["history"][-1]["recorded_at"] = 4
    with pytest.raises(ValueError):
        ledger.score_evidence(raw, bundle)


def test_exact_fixed_support_kernel_and_balanced_shift(example):
    """Symmetric fixture contrast cancels a common offset; single-cell edit acts."""
    from scripts.analyze_grain_f16_evidence_actions import response_weights

    bundle, raw = example
    rows = [r["observation"] for r in raw["history"]]
    intercepts, weights = response_weights(rows, bundle["scorer"])
    y = np.array([r["y"] for r in rows])
    result = ledger.score_evidence(raw, bundle)["result"]
    assert float(np.mean(intercepts + weights @ y)) == pytest.approx(result["score"])
    assert float(weights.sum()) == pytest.approx(0, abs=1e-12)
    assert weights[0, -1] > 0 and weights[0, 0] < 0
