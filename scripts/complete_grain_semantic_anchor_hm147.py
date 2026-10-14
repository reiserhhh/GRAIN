#!/usr/bin/env python3
"""Complete HM147 metadata tables without changing frozen source/scoring code."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_semantic_anchor_hm147 as H


def control_indices(rows):
    """Recognize historical 'none' control axes without reclassifying targets."""
    return {
        (row["condition"], row["situation"], row["seed"]): i
        for i, row in enumerate(rows)
        if row["axis"] not in H.TARGETS and row["pole"] == 0
    }


def finish(out):
    """Complete only failed tables from the already frozen numeric scores."""
    H.check(out)
    for name, value in H.load(out / "source_seal.json").items():
        if H.digest(out / name) != value:
            raise ValueError("Source program changed")
    destination = out / "03_completion"
    destination.mkdir(exist_ok=False)
    rows = H.load(out / "application_metadata.json")
    scores = H.archive(out / "cold_scores.npz", ("scores", "sectors"))["scores"]
    controls = []
    lookup = control_indices(rows)
    if len(lookup) != 27:
        raise ValueError("Expected all27 historical application control records")
    for (condition, task, seed), i in sorted(lookup.items()):
        if not condition.startswith("nonpersona"):
            continue
        j = lookup["neutral", task, seed]
        for a, arm in enumerate(H.ARMS):
            for k, target in enumerate(H.TARGETS):
                controls.append(
                    {
                        "condition": condition,
                        "task": task,
                        "seed": seed,
                        "target": target,
                        "arm": arm,
                        "minus_neutral": float(scores[i, a, k] - scores[j, a, k]),
                    }
                )
    if len(controls) != 108:
        raise ValueError("Incomplete nonpersona contrast support")
    with (out / "paired_sector_effects.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    if len(pairs) != 324:
        raise ValueError("Existing target contrast table incomplete")
    totals = []
    for role in ("development", "evaluation", "all_application"):
        for manipulated in H.TARGETS:
            for target in H.TARGETS:
                for arm in H.ARMS:
                    selected = [
                        row
                        for row in pairs
                        if (role == "all_application" or row["role"] == role)
                        and row["manipulated_axis"] == manipulated
                        and row["target"] == target
                        and row["arm"] == arm
                    ]
                    totals.append(
                        {
                            "role": role,
                            "manipulated_axis": manipulated,
                            "target": target,
                            "arm": arm,
                            "pairs": len(selected),
                            "mean_contrast": float(
                                np.mean([float(row["contrast"]) for row in selected])
                            ),
                            "positive_pairs": sum(
                                float(row["contrast"]) > 0 for row in selected
                            ),
                        }
                    )
    H.write_csv(destination / "control_effects.csv", controls)
    H.write_csv(destination / "effects_by_role.csv", totals)
    H.write_json(
        destination / "summary.json",
        {
            "status": "EXECUTED_SOURCE_ANCHORED_JOINT_VALUATION_NOT_HUMAN_VALIDITY",
            "source": H.load(out / "source_summary.json"),
            "cold_rows": 135,
            "paired_effect_rows": len(pairs),
            "control_effect_rows": len(controls),
            "effects_by_role": totals,
            "new_model_API_training_private_calls": 0,
            "failure": "Original metadata analysis expected an empty axis for controls; historical metadata uses 'none'. Source construction and cold scores unchanged. Empty control_effects.csv and two existing full target tables retained.",
            "scientific_scope": "Prompt allocation anchors on existing generated text, not human personality; no native answer fit or performance-selected sector",
        },
    )
    names = (
        "cold_scores.npz",
        "paired_sector_effects.csv",
        "effects_by_task_family.csv",
        "03_completion/control_effects.csv",
        "03_completion/effects_by_role.csv",
        "03_completion/summary.json",
    )
    H.write_json(
        out / "output_seal.json", {name: H.digest(out / name) for name in names}
    )
    H.write_json(
        destination / "companion_code_seal.json",
        {"scripts/complete_grain_semantic_anchor_hm147.py": H.digest(Path(__file__))},
    )
    print(
        json.dumps(
            {
                "completed": str(destination),
                "original_source_and_scores_unchanged": True,
                "controls": len(controls),
            }
        )
    )


def verify(out):
    """Preserve the original independent kernel/gauge verifier and its receipt."""
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        H.verify(out)
    result = json.loads(capture.getvalue())
    H.write_json(out / "03_completion/verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    action = sys.argv[1]
    {"finish": finish, "verify": verify}[action](H.DEFAULT)
