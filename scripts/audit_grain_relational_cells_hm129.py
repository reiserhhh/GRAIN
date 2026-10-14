#!/usr/bin/env python3
"""Post-result diagnosis of frozen HM129 observations, with no model or refit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def diagnose(output):
    """Recount all assignments and test fixed-index invariance on declared pairs."""
    output = Path(output)
    truth = {
        t["id"]: t for t in json.loads((output / "evaluation_truth.json").read_text())
    }
    summary = json.loads((output / "summary.json").read_text())
    primary = json.loads((output / "result_seal.json").read_text())
    for filename, expected in primary.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen result changed: {filename}")
    totals, pairs, envelopes = {}, [], {}
    with np.load(output / "source_scored.npz", allow_pickle=False) as source:
        states = source["real_coordinates"]
        envelopes["source"] = {
            "mean": states.mean((0, 1)).tolist(),
            "minimum": states.min((0, 1)).tolist(),
            "maximum": states.max((0, 1)).tolist(),
        }
    for stage in ("source", "cold"):
        with np.load(output / f"{stage}_scored.npz", allow_pickle=False) as scored:
            for index, world_id in enumerate(scored["ids"]):
                wid = int(world_id)
                domain = (
                    "source"
                    if stage == "source"
                    else ("cold_behaviour" if "family" in truth[wid] else "cold_fact")
                )
                for arm in ("real", "permuted", "position"):
                    total = totals.setdefault(
                        (domain, arm),
                        {
                            "records": 0,
                            "entity_correct": 0,
                            "relation_correct": 0,
                            "joint_correct": 0,
                            "entity_confusion": np.zeros((3, 3), int),
                            "relation_confusion": np.zeros((2, 2), int),
                        },
                    )
                    for gold, routed in zip(
                        truth[wid]["roles"], scored[f"{arm}_roles"][index], strict=True
                    ):
                        total["records"] += 1
                        total["entity_correct"] += int(gold[0] == routed[0])
                        total["relation_correct"] += int(gold[1] == routed[1])
                        total["joint_correct"] += int(tuple(gold) == tuple(routed))
                        total["entity_confusion"][gold[0], routed[0]] += 1
                        total["relation_confusion"][gold[1], routed[1]] += 1
    for (domain, arm), total in totals.items():
        for key in ("records", "entity_correct", "relation_correct", "joint_correct"):
            if total[key] != summary["domains"][domain][arm][key]:
                raise RuntimeError("Independent summary recount differs")
    with np.load(output / "cold_scored.npz", allow_pickle=False) as scored:
        lookup = {int(wid): i for i, wid in enumerate(scored["ids"])}
        conditions = {
            (t["family"], t["wording"], t["header_reverse"], t["assignment"]): wid
            for wid, t in truth.items()
            if "family" in t
        }
        behaviour_indices = [lookup[wid] for wid in conditions.values()]
        coords = scored["real_coordinates"][behaviour_indices]
        envelopes["cold_behaviour"] = {
            "mean": coords.mean((0, 1)).tolist(),
            "minimum": coords.min((0, 1)).tolist(),
            "maximum": coords.max((0, 1)).tolist(),
        }
        for key, a in conditions.items():
            if key[3] != 0:
                continue
            b = conditions[key[:3] + (1,)]
            if truth[a]["roles"] != truth[b]["roles"]:
                raise RuntimeError("Matched material roles changed")
            ia, ib = lookup[a], lookup[b]
            for e, gold in enumerate(truth[a]["roles"]):
                delta = (
                    scored["real_coordinates"][ib, e]
                    - scored["real_coordinates"][ia, e]
                )
                changed = bool(
                    np.any(scored["real_roles"][ib, e] != scored["real_roles"][ia, e])
                )
                pairs.append(
                    {
                        "first": a,
                        "second": b,
                        "record": e,
                        "entity": gold[0],
                        "relation": gold[1],
                        "entity_delta": float(delta[0]),
                        "relation_delta": float(delta[1]),
                        "routed_cell_changed": changed,
                    }
                )
    grouped = {}
    for name, selected in (
        ("all", pairs),
        ("unchanged_third_person", [r for r in pairs if r["entity"] == 2]),
    ):
        delta = np.array([[r["entity_delta"], r["relation_delta"]] for r in selected])
        grouped[name] = {
            "records": len(selected),
            "routed_cell_changed": sum(r["routed_cell_changed"] for r in selected),
            "coordinate_rms": np.sqrt((delta**2).mean(0)).tolist(),
        }
    recount = {
        f"{domain}/{arm}": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in total.items()
        }
        for (domain, arm), total in totals.items()
    }
    return {
        "status": "POST_RESULT_NO_REFIT",
        "primary_hash_entries_checked": len(primary),
        "routing_records_recounted": sum(v["records"] for v in totals.values()),
        "coordinate_envelopes": envelopes,
        "counts_and_confusions": recount,
        "fixed_index_pairs": grouped,
        "new_model_calls_or_fits": 0,
        "primary_gate_unchanged": summary["limited_modular_gate"],
    }, pairs


def main():
    """Save an independent result-after diagnostic without editing primary files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report, rows = diagnose(args.output)
    path = args.output / "post_result_diagnostic.json"
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output / "fixed_index_pair_drift.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
