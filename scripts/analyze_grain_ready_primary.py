#!/usr/bin/env python3
"""Audit all35 raw same-ready heads and paired increments without selecting a winner."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import metrics
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import (
    digest,
    load_registry_inputs,
)
from grain_research.vector_mechanisms.ready_registry import TASKS
from scripts.run_grain_ready_primary_batch import (
    COHORT,
    FEATURES,
    cases,
    reuse_big5_profile,
    validate_ready,
)
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_neural_source_primary import score
from scripts.summarize_grain_vector_external_links import load_oof

OBSERVERS = ("sorted_distances", *FEATURES)
PAIRS = ((4, 2), (3, 2), (1, 2), (5, 0), (5, 4), (6, 0), (6, 2), (4, 0), (2, 0))


def require_complete_batch(m, s):
    """Require every frozen case, not whichever favorable task finished first."""
    expected = {c["id"] for c in cases()}
    if (
        material_hash({k: v for k, v in m.items() if k != "identity"}) != m["identity"]
        or m["protocol"] != "VM05_RAW_READY_PRIMARY_BATCH_V1"
        or m["cases"] != cases()
        or m["cohort"] != COHORT
        or m["arm"] != "raw"
        or not m["pca"]
        or m["readout"] != "body_token"
        or m["bridge_labels_read"]
        or s["status"] != "COMPLETE"
        or s.get("child_pid")
        or s.get("reused_profiles") != 1
        or len(s["completed"]) != 34
        or {r["case"] for r in s["completed"]} != expected
    ):
        raise ValueError("All34 new cases plus the original Big5 profile are required")


def require_pairing(packets):
    """Reject even one different author, target or official fold among seven heads."""
    if len(packets) != 7 or any(
        not np.array_equal(packets[0][key], row[key])
        for row in packets[1:]
        for key in ("ids", "y", "folds")
    ):
        raise ValueError("Seven identical source cohorts, targets and folds required")


def paired_increments(y, predictions, folds, task, draws=1000):
    """Keep all fixed contrasts and both estimands on shared within-fold author draws."""
    y, predictions, folds = map(np.asarray, (y, predictions, folds))
    if (
        task not in TASKS
        or predictions.shape != (7, *y.shape)
        or folds.shape != (len(y),)
        or set(folds) != set(range(5))
        or not np.isfinite(y).all()
        or not np.isfinite(predictions).all()
        or (task == "big5" and (y.ndim != 2 or y.shape[1] != 5))
        or (task != "big5" and (y.ndim != 1 or not np.isin(y, [0, 1]).all()))
        or draws < 20
    ):
        raise ValueError("Require seven finite paired heads and five supported folds")
    parts = [np.flatnonzero(folds == f) for f in range(5)]

    def estimates(indices):
        joined = np.concatenate(indices)
        by_fold = np.array(
            [[score(y[ix], p[ix], task) for p in predictions] for ix in indices]
        )
        return np.array(
            [
                [score(y[joined], p[joined], task) for p in predictions],
                by_fold.mean(axis=0),
            ]
        )

    observed = estimates(parts)
    if not np.isfinite(observed).all():
        raise ValueError("Undefined observed source metric")
    rng = np.random.default_rng(731)
    samples = np.asarray(
        [
            estimates([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(draws)
        ]
    )
    finite = samples[np.isfinite(samples).all(axis=(1, 2))]
    if len(finite) < 0.95 * draws:
        raise ValueError("Insufficient bootstrap target support")
    fold_scores = np.array(
        [[score(y[ix], p[ix], task) for p in predictions] for ix in parts]
    )
    rows = []
    for candidate, reference in PAIRS:
        for k, estimand in enumerate(("pooled", "mean_official_fold")):
            low, high = np.quantile(
                finite[:, k, candidate] - finite[:, k, reference], [0.025, 0.975]
            )
            rows.append(
                {
                    "candidate": OBSERVERS[candidate],
                    "baseline": OBSERVERS[reference],
                    "estimand": estimand,
                    "candidate_score": observed[k, candidate],
                    "baseline_score": observed[k, reference],
                    "delta": observed[k, candidate] - observed[k, reference],
                    "lower": low,
                    "upper": high,
                    "positive_fold_count": int(
                        np.sum(fold_scores[:, candidate] > fold_scores[:, reference])
                    ),
                    "draws": draws,
                    "undefined_draws": draws - len(finite),
                    "tests_grain_increment_over_neural": (candidate, reference)
                    in ((5, 4), (6, 2)),
                    "tests_neural_increment_over_grain": (candidate, reference)
                    in ((5, 0), (6, 0)),
                }
            )
    return rows


def main():
    """Reconstruct registry/reuse evidence, then summarize all raw source tasks."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "batch-dir",
        "views",
        "registry-dir",
        "registry-audit-dir",
        "profile-dir",
        "preflight-dir",
        "parent-dir",
        "vector-dir",
        "primary-dir",
        "resource-dir",
        "predecessor-dir",
        "big5-profile-parent",
        "output-dir",
    ):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    m = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    status = json.loads((a.batch_dir / "batch_status.json").read_text())
    require_complete_batch(m, status)
    paths = {
        "views": a.views,
        "preflight": a.preflight_dir / "manifest.json",
        "source_parent": a.parent_dir / "manifest.json",
        "vector_parent": a.vector_dir / "manifest.json",
        "registry_audit": a.registry_audit_dir / "summary.json",
        "primary_parent": a.primary_dir / "batch_manifest.json",
        "strong_parent": a.resource_dir / "batch_manifest.json",
        "sensitivity_parent": a.predecessor_dir / "batch_manifest.json",
    }
    for name in m["input_sha256"]:
        if name.startswith("big5_profile/"):
            suffix = name.split("/", 1)[1]
            path = (a.big5_profile_parent / suffix).resolve()
            if not path.is_relative_to(a.big5_profile_parent.resolve()):
                raise ValueError("Invalid historical profile path")
            paths[name] = path
    if (
        set(paths) != set(m["input_sha256"])
        or any(digest(path) != m["input_sha256"][name] for name, path in paths.items())
        or any(digest(ROOT / name) != sha for name, sha in m["scoring_code"].items())
    ):
        raise ValueError("Frozen ready source/code parent changed")
    source, common, reg, profiles, registry = load_registry_inputs(
        a.views, a.registry_dir, a.registry_audit_dir, a.profile_dir, a.preflight_dir
    )
    if registry["identity"] != m["registry_identity"]:
        raise ValueError("Different ready registry")
    reuse = reuse_big5_profile(a.big5_profile_parent, source, reg, profiles)
    previous_reuse = json.loads((a.batch_dir / "reused_big5_profile.json").read_text())
    if reuse != previous_reuse:
        raise ValueError("Historical Big5 reuse no longer reproduces")
    packets, commitments = {}, {}
    recorded = {r["case"]: r for r in status["completed"]}
    for c in cases():
        folder = a.batch_dir / c["id"] / "fit"
        accepted = validate_ready(folder, c, source, common, reg, registry)
        if accepted != recorded[c["id"]]:
            raise ValueError("Completed case differs from execution record")
        packets[c["task"], c["feature"]] = load_oof(folder)
        commits = (
            "manifest.json",
            "summary.json",
            "private_oof_predictions.csv",
            "saved_head_replay.csv",
            "private_full_source_head.joblib",
        )
        names = [*commits, *(f"private_fold_{f}.joblib" for f in range(5))]
        commitments[c["id"]] = {n: digest(folder / n) for n in names}
    packets["big5", "sorted_distances"] = load_oof(a.big5_profile_parent)
    rows, scores, traits = [], [], []
    for task in TASKS:
        group = [packets[task, feature] for feature in OBSERVERS]
        require_pairing(group)
        rows.extend(
            dict(task=task, authors=len(group[0]["ids"]), **r)
            for r in paired_increments(
                group[0]["y"], [r["p"] for r in group], group[0]["folds"], task
            )
        )
        for feature, r in zip(OBSERVERS, group, strict=True):
            scores.append(
                {
                    "task": task,
                    "feature": feature,
                    "authors": len(r["ids"]),
                    "pooled_score": r["summary"]["score"],
                    "mean_fold_score": r["mean_fold_score"],
                }
            )
            traits.extend(
                dict(task=task, feature=feature, **r0)
                for r0 in metrics(r["y"], r["p"], task)
            )
        print(json.dumps({"task": task, "heads": 7}), flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    code = (
        "scripts/analyze_grain_ready_primary.py",
        "scripts/summarize_grain_neural_source_primary.py",
        "scripts/analyze_grain_ei_source_gap.py",
    )
    manifest = {
        "protocol": "VM05_RAW_READY_PAIRED_COMPARISON_V1",
        "batch_identity": m["identity"],
        "registry_identity": registry["identity"],
        "cohort": COHORT,
        "inputs": commitments,
        "reused_big5": reuse,
        "source_code_sha256": {n: digest(ROOT / n) for n in code},
        "observers": list(OBSERVERS),
        "contrasts": [[OBSERVERS[i], OBSERVERS[j]] for i, j in PAIRS],
        "bootstrap": "1000 shared author-within-fold draws seed731; pointwise conditional on fitted OOF",
        "analysis_implementation_after_some_point_estimates": True,
        "profile_scope": "16 sorted frozen landmark distances; not the complete GRAIN4W framework",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
    ) as z:
        for n in code:
            if n not in z.namelist():
                z.write(ROOT / n, n)
    tables = {
        "observer_scores.csv": scores,
        "metrics_by_trait.csv": traits,
        "paired_increments.csv": rows,
    }
    for name, values in tables.items():
        pd.DataFrame(values).to_csv(a.output_dir / name, index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "new_source_heads_accepted": 34,
        "historical_source_heads_reused": 1,
        "saved_folds_accepted": 175,
        "paired_rows": len(rows),
        "model_refits": 0,
        "source_label_tables_read": False,
        "existing_source_oof_outcomes_reused": True,
        "bridge_labels_read": False,
        "program_complete": False,
        "outputs_sha256": {n: digest(a.output_dir / n) for n in tables},
        "scope": "Complete raw reference-free stage only; no full-family closure, coordinate identification or winner selection",
    }
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
