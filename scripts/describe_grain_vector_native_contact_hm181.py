#!/usr/bin/env python3
"""Post-outcome same-event descriptor;no fits,selection or new neural calls."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_context_relation as C
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_vector_lift_hm181 as S

PLAN = "docs/GRAIN_VECTOR_LIFT_NATIVE_CONTACT_PLAN_HM181.md"
SCRIPT = "scripts/describe_grain_vector_native_contact_hm181.py"
OLD = ROOT / "results/grain_math_response_v1/context_relation_action_r1"


def main():
    """Join fixed source vector readings to all cached native N context endpoints."""
    out = S.DEFAULT / "post_outcome_native_contact_r1"
    R.check(
        S.DEFAULT, ("compiled_seal.json", "scored_seal.json", "verification_seal.json")
    )
    R.check(OLD, ("scored_seal.json", "analysis_seal.json"))
    z, old = R.prepared(S.DEFAULT), R.prepared(OLD)
    if not np.array_equal(z["panel"], old["application_indices"]):
        raise ValueError("Different author panels")
    legacy = {
        r["author_key"]: r
        for r in map(json.loads, (OLD / "cold_input.jsonl").read_text().splitlines())
    }
    count = 0
    for line in (S.DEFAULT / "cold_input.jsonl").read_text().splitlines():
        r = json.loads(line)
        old_events = legacy["1/" + r["frame_key"]]["events"]
        current_ids = [(e["event_id"], e["slot"]) for e in r["events"]]
        prior_ids = [(e["event_id"], e["slot"]) for e in old_events]
        if current_ids != prior_ids:
            raise ValueError("Same-author measurements have different text evidence")
        count += 1
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        str((p / name).relative_to(ROOT))
        for p, names in (
            (
                S.DEFAULT,
                (
                    "manifest.json",
                    "compiled_seal.json",
                    "scored_seal.json",
                    "cold_input.jsonl",
                    "cold_output.jsonl",
                ),
            ),
            (
                OLD,
                (
                    "manifest.json",
                    "analysis_seal.json",
                    "mechanism_accounts.npz",
                    "cold_input.jsonl",
                ),
            ),
        )
        for name in names
    ]
    R.write(
        out / "manifest.json",
        {
            "classification": "POST_OUTCOME_DESCRIPTION_NOT_NEW_CONFIRMATORY_HYPOTHESIS",
            "code": {
                p: R.digest(ROOT / p)
                for p in (
                    PLAN,
                    SCRIPT,
                    "grain_research/construction/math_context_relation.py",
                )
            },
            "inputs": {p: R.digest(ROOT / p) for p in paths},
            "original_scope_complete": False,
        },
    )
    with np.load(OLD / "mechanism_accounts.npz", allow_pickle=False) as file:
        endpoints = np.concatenate(
            (
                file["native_ordinal_actions"][..., None],
                file["native_log_odds_actions"],
            ),
            axis=2,
        )
    readings = np.zeros((24, 2, 5, 2, 5))
    for row in map(
        json.loads, (S.DEFAULT / "cold_output.jsonl").read_text().splitlines()
    ):
        key = tuple(map(int, row["frame_key"].split("/")))
        readings[key] = row["retained_readings"]
    actions = C.context_action(np.moveaxis(readings, 2, 0))
    direct = (
        readings[:, :, 2] - readings[:, :, 1] - readings[:, :, 3] + readings[:, :, 4]
    )
    action_error = float(np.max(np.abs(actions - direct)))
    rows = []
    for sv, q, endpoint in np.ndindex(2, 5, 10):
        for group in ("author_mean", "view0", "view1"):
            x, y = actions[:, :, sv, q], endpoints[:, :, endpoint]
            if group == "author_mean":
                x, y = x.mean(1), y.mean(1)
            else:
                view = int(group[-1])
                x, y = x[:, view], y[:, view]
            rows.append(
                {
                    "source_view": sv,
                    "source_target": S.V.TARGETS[q],
                    "native_query": "N",
                    "endpoint": "ordinal" if endpoint == 0 else f"digit{endpoint}_vs0",
                    "group": group,
                    **C.signed_agreement(x, y),
                }
            )
    R.table(out / "all_native_links.csv", rows)
    R.archive(
        out / "actions.npz",
        readings=readings,
        actions=actions,
        native_endpoints=endpoints,
    )
    result = {
        "matched_raw_frames": count,
        "native_query": "N_only",
        "links": len(rows),
        "all_five_targets_both_source_views_kept": True,
        "contact_action_error": action_error,
        "source_N_links": [
            r for r in rows if r["source_target"] == "N" and r["group"] == "author_mean"
        ],
        "no_fit_new_neural_API_or_application_labels": True,
        "accuracy_or_causal_equivalence_claim": False,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", result)
    R.seal(
        out,
        "description_seal.json",
        ("all_native_links.csv", "actions.npz", "summary.json"),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
