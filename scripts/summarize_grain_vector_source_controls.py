#!/usr/bin/env python3
"""Independently audit VM05 source-control membership and paired OOF effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_control_audit import (
    audit_count_match,
    audit_discussion_controls,
    paired_seed_average,
    restrict_oof,
)
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_heads import metrics
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof


def legacy_path(root, task, arm):
    """Resolve the frozen pre-control source comparisons without picking winners."""
    suffix = {"raw": "raw", "cue_mask": "cue", "placebo_mask": "placebo"}[arm]
    if task == "big5":
        return root / f"vm05_big5_word_char_{suffix}_r1"
    return root / f"vm05_{task[:2].lower()}_grain_intermediate_{suffix}_r1"


def main():
    """Require actual completed results; optionally publish an explicitly partial audit."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--legacy-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    batch = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    if (
        material_hash({k: v for k, v in batch.items() if k != "identity"})
        != batch["identity"]
        or hashlib.sha256(a.views.read_bytes()).hexdigest() != batch["views_sha256"]
        or hashlib.sha256((a.preflight_dir / "manifest.json").read_bytes()).hexdigest()
        != batch["preflight_manifest_sha256"]
    ):
        raise ValueError("Changed batch material identity")
    source = json.loads(a.views.read_text())
    budget = pd.read_csv(
        a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
    )
    bridge = [b for b in source if b["source_group"] == "bridge"]
    records, selections, objects, audit_rows, pending = [], {}, {}, [], []
    for case in batch["cases"]:
        folder = a.batch_dir / case["id"]
        if not (folder / "summary.json").exists():
            pending.append(case["id"])
            continue
        result = load_oof(folder)
        m = result["manifest"]
        if any(
            m[k] != case[c]
            for k, c in (
                ("task", "task"),
                ("feature", "feature"),
                ("arm", "arm"),
                ("thread_control", "mode"),
                ("drop_seed", "seed"),
                ("exclude_reference", "exclude_reference"),
            )
        ):
            raise ValueError("Case differs from frozen batch")
        bundles = [b for b in source if b["source_group"] == m["source_group"]]
        if m["exclude_reference"]:
            bundles = exclude_reference(bundles, budget)
        lookup = {b["author_id"]: b["official_folds"][m["task"]] for b in bundles}
        if any(lookup[u] != f for u, f in zip(result["ids"], result["folds"])):
            raise ValueError("OOF prediction assigned to wrong official fold")
        row, log = audit_discussion_controls(folder, result, bundles, bridge)
        key = (m["task"], m["arm"], m["thread_control"], m["drop_seed"])
        objects[key], selections[key] = result, log
        audit_rows.append(row)
        records.append(
            {
                "case": case["id"],
                "task": m["task"],
                "arm": m["arm"],
                "mode": m["thread_control"],
                "seed": m["drop_seed"],
                "exclude_reference": m["exclude_reference"],
                "authors": len(result["ids"]),
                "pooled_score": result["summary"]["score"],
                "mean_fold_score": result["mean_fold_score"],
                "bootstrap_lower": result["summary"]["bootstrap"]["lower"],
                "bootstrap_upper": result["summary"]["bootstrap"]["upper"],
                "source_identity": m["identity"],
                **(
                    metrics(result["y"], result["p"], m["task"])[0]
                    if m["task"] != "big5"
                    else {}
                ),
            }
        )
    if pending and not a.allow_partial:
        raise ValueError(f"{len(pending)} source cases remain; not a complete audit")
    # The older Big5 baseline has the same evaluation authors; it is not rerun or selected.
    for arm in ("raw", "cue_mask", "placebo_mask"):
        objects[("big5", arm, "retain", 731)] = load_oof(
            legacy_path(a.legacy_dir, "big5", arm)
        )
    contrasts, count_rows = [], []

    def compare(name, key, ref_keys):
        if key not in objects or any(k not in objects for k in ref_keys):
            return
        contrasts.append(
            {
                "contrast": name,
                "task": key[0],
                "arm": key[1],
                "candidate_mode": key[2],
                "reference_modes": ",".join(sorted({k[2] for k in ref_keys})),
                **paired_seed_average(objects[key], [objects[k] for k in ref_keys]),
            }
        )

    tasks = ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont")
    arms = ("raw", "cue_mask", "placebo_mask")
    for task in tasks:
        for arm in arms:
            retained, purged = (task, arm, "retain", 731), (task, arm, "purge", 731)
            compare("purge_minus_retain", purged, [retained])
            random = [(task, arm, "random_count", seed) for seed in range(731, 736)]
            if purged in selections:
                for key in random:
                    if key in selections:
                        count_rows.append(
                            {
                                "task": task,
                                "arm": arm,
                                "seed": key[3],
                                "matched_selections": audit_count_match(
                                    selections[purged], selections[key]
                                ),
                            }
                        )
            compare("purge_minus_mean_of_five_count_controls", purged, random)
            if task != "big5" and retained in objects:
                current = objects[retained]
                old = restrict_oof(
                    load_oof(legacy_path(a.legacy_dir, task, arm)), current["ids"]
                )
                contrasts.append(
                    {
                        "contrast": "exclude_reference_refit_minus_original_on_same_authors",
                        "task": task,
                        "arm": arm,
                        "candidate_mode": "retain",
                        "reference_modes": "legacy_retain_restricted",
                        **paired_seed_average(current, [old]),
                    }
                )
        for mode in ("retain", "purge"):
            for candidate, baseline in (
                ("cue_mask", "raw"),
                ("placebo_mask", "raw"),
                ("cue_mask", "placebo_mask"),
            ):
                compare(
                    f"{candidate}_minus_{baseline}",
                    (task, candidate, mode, 731),
                    [(task, baseline, mode, 731)],
                )

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new audit output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    pd.DataFrame(records).to_csv(out / "source_control_scores.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "membership_audit.csv", index=False)
    pd.DataFrame(count_rows).to_csv(out / "nested_count_match.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_control_contrasts.csv", index=False)
    summary = {
        "status": "PARTIAL" if pending else "COMPLETE",
        "verified_cases": len(records),
        "planned_cases": len(batch["cases"]),
        "membership_records_reconstructed": sum(
            r["selection_records"] for r in audit_rows
        ),
        "random_count_pairs_verified": len(count_rows),
        "paired_contrasts": len(contrasts),
        "pending_cases": pending,
        "batch_identity": batch["identity"],
        "bridge_labels_read": False,
        "new_label_values_read": False,
        "inference_scope": "descriptive sensitivity; conditional OOF bootstrap, not causal effects",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
