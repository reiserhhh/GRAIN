#!/usr/bin/env python3
"""Recompute completed source OOF metrics and fixed paired contrasts, no new labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import TRAITS, task_score
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json


def load_oof(folder):
    """Verify complete repeated-trait rows and recompute, rather than trust summary."""
    summary = json.loads((folder / "summary.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        summary["status"] != "COMPLETE"
        or summary["parent_identity"] != manifest["identity"]
    ):
        raise ValueError("Incomplete source result")
    if (
        material_hash({k: v for k, v in manifest.items() if k != "identity"})
        != manifest["identity"]
    ):
        raise ValueError("Source execution manifest mismatch")
    frame = pd.read_csv(
        folder / "private_oof_predictions.csv", dtype={"author_id": str}
    )
    if (
        frame.duplicated(["author_id", "trait"]).any()
        or frame.groupby("author_id").fold.nunique().max() != 1
    ):
        raise ValueError("Repeated or inconsistent author predictions")
    names = list(TRAITS) if manifest["task"] == "big5" else [manifest["task"]]
    truth = frame.pivot(index="author_id", columns="trait", values="y_true").reindex(
        columns=names
    )
    predicted = frame.pivot(
        index="author_id", columns="trait", values="prediction"
    ).reindex(index=truth.index, columns=names)
    if (
        len(truth) != manifest["authors"]
        or not np.isfinite(truth).all().all()
        or not np.isfinite(predicted).all().all()
    ):
        raise ValueError("Missing author/trait outcome rows")
    y, p = truth.to_numpy(), predicted.to_numpy()
    if manifest["task"] != "big5":
        y, p = y[:, 0], p[:, 0]
    folds = frame.groupby("author_id").fold.first().reindex(truth.index).to_numpy()
    score = task_score(y, p, manifest["task"])
    if not np.isclose(score, summary["score"], atol=1e-12):
        raise ValueError("Summary does not reproduce from OOF rows")
    fold_scores = [
        task_score(y[folds == f], p[folds == f], manifest["task"]) for f in range(5)
    ]
    intercept = np.empty_like(y)
    for f in range(5):
        intercept[folds == f] = y[folds != f].mean(axis=0)
    return {
        "summary": summary,
        "manifest": manifest,
        "ids": truth.index.to_numpy(),
        "y": y,
        "p": p,
        "folds": folds,
        "mean_fold_score": float(np.mean(fold_scores)),
        "intercept_only_pooled_score": task_score(y, intercept, manifest["task"]),
    }


def paired_delta(candidate, baseline, draws=1000):
    """Compare the same authors/targets; never pair permuted and original outcomes."""
    m, b = candidate["manifest"], baseline["manifest"]
    if m["task"] != b["task"] or m["permutation_seed"] != b["permutation_seed"]:
        raise ValueError("Different target or label permutation")
    for field in ("ids", "y", "folds"):
        if not np.array_equal(candidate[field], baseline[field]):
            raise ValueError("Unpaired source evidence or targets")
    rng = np.random.default_rng(731)
    y, x, z = candidate["y"], candidate["p"], baseline["p"]
    delta = task_score(y, x, m["task"]) - task_score(y, z, m["task"])
    values = []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        values.append(
            task_score(y[ix], x[ix], m["task"]) - task_score(y[ix], z[ix], m["task"])
        )
    finite = np.asarray(values)[np.isfinite(values)]
    return {
        "delta": delta,
        "lower": float(np.quantile(finite, 0.025)),
        "upper": float(np.quantile(finite, 0.975)),
        "authors": len(y),
        "draws": draws,
        "undefined_draws": draws - len(finite),
        "scope": "paired authors conditional on fitted OOF heads, exploratory not refit inference",
    }


def main():
    """Write descriptive table and only the predeclared same-material contrasts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    objects, rows = {}, []
    for folder in a.runs:
        result = load_oof(folder)
        m, s = result["manifest"], result["summary"]
        key = (m["task"], m["feature"], m["arm"], m["pca"], m["permutation_seed"])
        if key in objects:
            raise ValueError("Ambiguous multiple runs of a condition")
        objects[key] = result
        rows.append(
            {
                "run": folder.name,
                "task": m["task"],
                "feature": m["feature"],
                "arm": m["arm"],
                "pca_applied": m["pca"] and m["feature"] != "word_char",
                "permutation_seed": m["permutation_seed"],
                "authors": m["authors"],
                "pooled_score": s["score"],
                "mean_fold_score": result["mean_fold_score"],
                "bootstrap_lower": s["bootstrap"]["lower"],
                "bootstrap_upper": s["bootstrap"]["upper"],
                "intercept_only_pooled_score": result["intercept_only_pooled_score"],
                "source_manifest_identity": m["identity"],
                "not_full_GRAIN_comparison": True,
            }
        )
    keys = {
        "sparse_raw": ("big5", "word_char", "raw", True, None),
        "sparse_cue": ("big5", "word_char", "cue_mask", True, None),
        "sparse_placebo": ("big5", "word_char", "placebo_mask", True, None),
        "intermediate_pca": ("big5", "grain_intermediate", "raw", True, None),
        "intermediate_full": ("big5", "grain_intermediate", "raw", False, None),
    }
    comparisons = []
    for candidate, baseline in [
        ("sparse_cue", "sparse_raw"),
        ("sparse_placebo", "sparse_raw"),
        ("sparse_cue", "sparse_placebo"),
        ("intermediate_pca", "sparse_raw"),
        ("intermediate_full", "sparse_raw"),
        ("intermediate_full", "intermediate_pca"),
    ]:
        if keys[candidate] in objects and keys[baseline] in objects:
            comparisons.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    **paired_delta(objects[keys[candidate]], objects[keys[baseline]]),
                }
            )
    for task in ("EI_cont", "SN_cont", "TF_cont", "JP_cont"):
        for candidate, baseline in (
            ("cue_mask", "raw"),
            ("placebo_mask", "raw"),
            ("cue_mask", "placebo_mask"),
        ):
            ck = (task, "grain_intermediate", candidate, True, None)
            bk = (task, "grain_intermediate", baseline, True, None)
            if ck in objects and bk in objects:
                comparisons.append(
                    {
                        "candidate": f"{task}_{candidate}",
                        "baseline": f"{task}_{baseline}",
                        **paired_delta(objects[ck], objects[bk]),
                    }
                )
    pd.DataFrame(rows).to_csv(out / "source_comparison.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_comparison.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "verified_runs": len(rows),
        "paired_contrasts": len(comparisons),
        "bridge_labels_read": False,
        "new_label_values_read": False,
        "scope": "Intermediate source readouts, not whole GRAIN/LLM mechanism closure",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
