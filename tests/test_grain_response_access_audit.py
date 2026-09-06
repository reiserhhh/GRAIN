"""Scalar ledger counterexamples for the independent access audit."""

import copy

import pandas as pd
import pytest

from grain_research.vector_mechanisms.response_access import summarize
from scripts.audit_grain_response_access import CONDITIONS, SITES, frame_audit
from scripts.audit_grain_response_depth import compare


@pytest.fixture
def ledger():
    rows, checks = [], []
    for code, gap in (("A", 0), ("B", -0.3), ("C", 0.8), ("D", 0.2)):
        for c, (cut, sham, role) in CONDITIONS.items():
            effective = 27 if sham else cut
            tail = -4 + (
                gap * (effective + 1) / 28
                if role == "actual" or c.startswith("patch_")
                else 0
            )
            first = -3.0 if role == "actual" else -4.0
            rows.append(
                {
                    "participant_code": code,
                    "condition": c,
                    "cut": cut,
                    "sham": sham,
                    "layers": 27 - cut,
                    "prefix_tokens": 16,
                    "response_tokens": 20,
                    "edges_per_layer": 320 if cut != 27 else 0,
                    "tail_logp": tail,
                    "first_logp": first,
                    "mean_logp": (first + 19 * tail) / 20,
                }
            )
        checks.append(
            {
                "participant_code": code,
                "cut0_tail_error": 0,
                "mask_sham_error": 0,
                "self_sham_error": 0,
                "first_token_error": 0,
                "patch_cut_errors": {str(k): 0 for k in SITES},
            }
        )
    return pd.DataFrame(rows), checks


def test_full_ledger_and_independent_bootstrap(ledger):
    frame, checks = ledger
    value = frame_audit(frame, list("ABCD"), checks)
    compare(summarize(frame), value)


@pytest.mark.parametrize(
    "attack",
    [
        "drop",
        "duplicate",
        "edge",
        "cut",
        "first",
        "endpoint",
        "patch",
        "sham",
        "check",
        "nonfinite",
    ],
)
def test_bad_access_evidence_is_rejected(ledger, attack):
    f, checks = ledger
    checks = copy.deepcopy(checks)
    index = f.index[f.condition.eq("actual_10")][0]
    if attack == "drop":
        f = f.iloc[1:]
    elif attack == "duplicate":
        f = pd.concat([f.iloc[1:], f.iloc[1:2]])
    elif attack == "edge":
        f.loc[index, "edges_per_layer"] = 0
    elif attack == "cut":
        f.loc[index, "cut"] = 16
    elif attack == "first":
        f.loc[index, "first_logp"] += 0.1
    elif attack == "endpoint":
        i = f.index[f.condition.eq("actual_-1")][0]
        f.loc[i, "tail_logp"] += 0.1
        f.loc[i, "mean_logp"] += 0.095
    elif attack == "patch":
        i = f.index[f.condition.eq("patch_10_cut")][0]
        f.loc[i, "tail_logp"] += 0.1
        f.loc[i, "mean_logp"] += 0.095
    elif attack == "sham":
        i = f.index[f.condition.eq("mask_sham")][0]
        f.loc[i, "tail_logp"] += 0.1
        f.loc[i, "mean_logp"] += 0.095
    elif attack == "check":
        checks[0]["patch_cut_errors"]["16"] = 0.01
    else:
        f.loc[index, "tail_logp"] = float("nan")
    with pytest.raises(ValueError):
        frame_audit(f, list("ABCD"), checks)
