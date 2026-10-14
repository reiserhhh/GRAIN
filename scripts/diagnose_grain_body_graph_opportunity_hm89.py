#!/usr/bin/env python3
"""Audit the HM89 event-local support identity without new model calls."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_hierarchical_query_hm88 import load, sha, verify, write

OUT = ROOT / "results/grain_math_response_v1/body_relation_graph_r1"


def records(path):
    """Read sealed primary numerical records without changing them."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def run():
    """Separate causal root opportunity from weight-dependent role allocation."""
    verify(OUT, load(OUT / "02/output_seal.json"))
    verify(OUT, load(OUT / "00/manifest.json")["inputs"])
    rows = load(OUT / "00/rows.json")
    counts = {
        frame: {"behavior_tokens": 0, "actor_only": 0, "both_people": 0}
        for frame in ("actor_first", "reporter_first")
    }
    for row in rows:
        labels = np.array(row["root_labels"])
        event_ids = np.array(
            [
                row["event_order"][
                    next(j for j, group in enumerate(row["groups"]) if pos in group)
                ]
                for pos in row["body_indices"]
            ]
        )
        for event, span in enumerate(row["action_groups"]):
            actor = 0 if row["coordinates"][event + 2] > 0 else 1
            for i in span:
                roots = set(labels[:i][event_ids[:i] == event]) & {0, 1}
                counts[row["frame"]]["behavior_tokens"] += 1
                counts[row["frame"]]["actor_only"] += int(roots == {actor})
                counts[row["frame"]]["both_people"] += int(roots == {0, 1})
    for frame, expected in (
        ("actor_first", "actor_only"),
        ("reporter_first", "both_people"),
    ):
        if counts[frame][expected] != counts[frame]["behavior_tokens"]:
            raise ValueError("Causal root opportunity does not match frame: " + frame)
    selected = [
        row
        for row in records(OUT / "02/token_role_transport.csv")
        if row["method"] == "event_harmonic" and row["frame"] == "actor_first"
    ]
    error = max(
        abs(float(row["actor_alignment"]) - (1 - float(row["unresolved"])))
        for row in selected
    )
    if len(selected) != 384 or error > 1e-12:
        raise ValueError("Actor-only support identity failed")
    contrasts = {}
    for row in records(OUT / "02/author_law_contrasts.csv"):
        if row["is_target"] == "True":
            contrasts.setdefault(row["arm"], []).append(
                float(row["author_law_contrast"])
            )
    if len(contrasts) != 7 or any(len(values) != 24 for values in contrasts.values()):
        raise ValueError("Incomplete fixed-arm contrasts")
    result = {
        "status": "RESULT_AFTER_SUPPORT_DIAGNOSTIC",
        "new_model_calls": 0,
        "not_new_generic_theorem": True,
        "code_sha256": sha(Path(__file__)),
        "input_hashes": {
            name: sha(OUT / name)
            for name in (
                "00/rows.json",
                "02/output_seal.json",
                "verification.json",
            )
        },
        "causal_root_opportunity": counts,
        "actor_only_support_identity": {"records": len(selected), "max_error": error},
        "target_contrasts": {
            arm: {
                "conditions": len(values),
                "positive": sum(value > 0 for value in values),
                "mean": float(np.mean(values)),
            }
            for arm, values in contrasts.items()
        },
        "interpretation": (
            "Actor-first local alignment is forced by only one causally available "
            "named root; this is not independent evidence of semantic role intelligence."
        ),
        "original_scope_complete": False,
    }
    write(OUT / "opportunity_diagnostic.json", result)
    print(result["actor_only_support_identity"])


if __name__ == "__main__":
    run()
