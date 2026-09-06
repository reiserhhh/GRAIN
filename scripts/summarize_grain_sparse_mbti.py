#!/usr/bin/env python3
"""Recompute complete sparse MBTI arms and paired differences, without fitting."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof, paired_delta


def main():
    """Snapshot verified terminal arms; never treat unfinished jobs as estimates."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Use a fresh sparse comparison directory")
    batch = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    if batch["identity"] != material_hash(
        {k: v for k, v in batch.items() if k != "identity"}
    ):
        raise ValueError("Changed batch manifest")
    results, rows, pending, hashes = {}, [], [], {}
    for case in batch["cases"]:
        folder = a.batch_dir / case["id"]
        if not (folder / "summary.json").is_file():
            pending.append(case["id"])
            continue
        r = load_oof(folder)
        m = r["manifest"]
        if (
            m["task"] != case["task"]
            or m["arm"] != case["arm"]
            or m["feature"] != "word_char"
            or m["pca_applied"]
            or m["permutation_seed"] is not None
            or m["thread_control"] != "retain"
            or m["exclude_reference"] != batch["exclude_reference"]
        ):
            raise ValueError("Unexpected sparse condition")
        y, predicted = r["y"], r["p"] >= 0.5
        rows.append(
            {
                "case": case["id"],
                "task": m["task"],
                "arm": m["arm"],
                "authors": len(y),
                "auc": r["summary"]["score"],
                "mean_fold_auc": r["mean_fold_score"],
                "balanced_accuracy_at_0_5": balanced_accuracy_score(y, predicted),
                "macro_f1_at_0_5": f1_score(y, predicted, average="macro"),
                "predicted_positive_rate": float(predicted.mean()),
                "true_positive_rate": float(y.mean()),
            }
        )
        results[(m["task"], m["arm"])] = r
        hashes[case["id"]] = {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "summary.json", "private_oof_predictions.csv")
        }
    contrasts = []
    for task in ("EI_cont", "SN_cont", "TF_cont", "JP_cont"):
        for candidate, baseline in (
            ("cue_mask", "raw"),
            ("placebo_mask", "raw"),
            ("cue_mask", "placebo_mask"),
        ):
            if (task, candidate) in results and (task, baseline) in results:
                contrasts.append(
                    {
                        "task": task,
                        "candidate": candidate,
                        "baseline": baseline,
                        **paired_delta(
                            results[task, candidate], results[task, baseline]
                        ),
                    }
                )
    if not rows:
        raise ValueError("No completed source arm")
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "paired_contrasts.csv", index=False)
    result = {
        "status": "COMPLETE" if not pending else "PARTIAL_VERIFIED_ARMS_ONLY",
        "verified_arms": len(rows),
        "planned_arms": len(batch["cases"]),
        "pending": pending,
        "paired_contrasts": len(contrasts),
        "paired_bootstrap_draws": 1000,
        "bootstrap_seed": 731,
        "scope": "conditional on fitted OOF heads; exploratory contrasts, not refit inference",
        "batch_identity": batch["identity"],
        "source_sha256": hashes,
        "model_refits": 0,
        "new_label_values_read": False,
        "bridge_labels_read": False,
    }
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in [
            Path(__file__),
            ROOT / "scripts/summarize_grain_vector_external_links.py",
            ROOT / "grain_research/vector_mechanisms/external_heads.py",
            ROOT / "grain_research/vector_mechanisms/controls.py",
            ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ]:
            z.write(path, path.relative_to(ROOT))
    write_json(out / "summary.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "source_sha256"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
