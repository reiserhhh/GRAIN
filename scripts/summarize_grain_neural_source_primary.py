#!/usr/bin/env python3
"""Compare complete frozen observer triads without new fits or new label tables."""

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
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_heads import metrics
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.analyze_grain_ei_source_gap import rank_auc
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_neural_source_queue import (
    cases,
    validate_result,
)
from scripts.summarize_grain_vector_external_links import load_oof

OBSERVERS = ("bge_large", "e5_large", "qwen_block_10")
PAIRS = ((2, 0), (1, 0), (2, 1))


def save_tables(out, scores, traits, paired):
    """Hash only declared CSVs, never macOS AppleDouble filesystem sidecars."""
    tables = {
        "observer_scores.csv": scores,
        "metrics_by_trait.csv": traits,
        "paired_observer_comparison.csv": paired,
    }
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / filename, index=False)
    return {filename: digest(out / filename) for filename in tables}


def score(y, p, task):
    """Compute independent centered-dot Pearson or tied-rank binary AUC."""
    if task != "big5":
        return rank_auc(y, p)
    a, b = y - y.mean(axis=0), p - p.mean(axis=0)
    denominator = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    if np.any(denominator <= 1e-20):
        return float("nan")
    return float(np.mean((a * b).sum(axis=0) / denominator))


def paired_triads(y, predictions, folds, task, draws=1000):
    """Use the same within-fold author draws for all observers and both estimands."""
    y, predictions, folds = np.asarray(y), np.asarray(predictions), np.asarray(folds)
    if (
        predictions.shape != (3, *y.shape)
        or folds.shape != (len(y),)
        or set(folds) != set(range(5))
        or draws < 20
        or not np.isfinite(y).all()
        or not np.isfinite(predictions).all()
        or (task == "big5" and (y.ndim != 2 or y.shape[1] != 5))
    ):
        raise ValueError(
            "Require three paired observers and five complete official folds"
        )
    parts = [np.flatnonzero(folds == f) for f in range(5)]

    def estimates(indices):
        joined = np.concatenate(indices)
        pooled = [score(y[joined], p[joined], task) for p in predictions]
        by_fold = [[score(y[ix], p[ix], task) for p in predictions] for ix in indices]
        return np.asarray([pooled, np.mean(by_fold, axis=0)])

    observed = estimates(parts)
    if not np.isfinite(observed).all():
        raise ValueError("Undefined observed source score")
    rng = np.random.default_rng(731)
    sampled = np.asarray(
        [
            estimates([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(draws)
        ]
    )
    finite = sampled[np.isfinite(sampled).all(axis=(1, 2))]
    if len(finite) < 0.95 * draws:
        raise ValueError("Insufficient bootstrap class/trait support")
    rows = []
    for a, b in PAIRS:
        for j, estimand in enumerate(("pooled", "mean_official_fold")):
            low, high = np.quantile(finite[:, j, a] - finite[:, j, b], [0.025, 0.975])
            rows.append(
                {
                    "candidate": OBSERVERS[a],
                    "baseline": OBSERVERS[b],
                    "estimand": estimand,
                    "candidate_score": observed[j, a],
                    "baseline_score": observed[j, b],
                    "delta": observed[j, a] - observed[j, b],
                    "lower": low,
                    "upper": high,
                    "draws": draws,
                    "undefined_draws": draws - len(finite),
                    "planned_primary_contrast": (a, b) == (2, 0),
                }
            )
    return rows


def main():
    """Freeze a completed-packet snapshot; incomplete triads are never ranked."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("batch-dir", "views", "preflight-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    batch = json.loads(
        (a.batch_dir / "batch_manifest.json").read_text(encoding="utf-8")
    )
    if (
        batch["protocol"] != "VM05_FULL_NEURAL_SOURCE_QUEUE_V1"
        or material_hash({k: v for k, v in batch.items() if k != "identity"})
        != batch["identity"]
        or batch["cases"] != cases()
        or digest(a.views) != batch["views_sha256"]
        or digest(a.preflight_dir / "manifest.json")
        != batch["preflight_manifest_sha256"]
    ):
        raise ValueError("Changed full-source execution design")
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    source = json.loads(a.views.read_text(encoding="utf-8"))
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Common evidence changed")
    source = exclude_reference(
        source,
        pd.read_csv(a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}),
    )
    groups = [
        (task, arm)
        for arm in ("raw", "cue_mask", "placebo_mask")
        for task in ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont")
    ]
    accepted, pending, inputs, scores, traits, paired = [], [], {}, [], [], []
    for task, arm in groups:
        names = [f"{task}_{arm}_{feature}" for feature in OBSERVERS]
        if not all(
            (a.batch_dir / (name + "_replay") / "summary.json").exists()
            for name in names
        ):
            pending.append({"task": task, "arm": arm})
            continue
        packets = []
        for name, feature in zip(names, OBSERVERS, strict=True):
            folder, replay = a.batch_dir / name, a.batch_dir / (name + "_replay")
            r = load_oof(folder)
            validate_result(
                r,
                {"task": task, "arm": arm, "feature": feature},
                source,
                common["material_sha256"],
            )
            rs = json.loads((replay / "summary.json").read_text())
            rf = pd.read_csv(replay / "replay_by_fold.csv")
            if (
                rs["status"] != "COMPLETE"
                or rs["fold_models"] != 5
                or rs["new_label_values_read"]
                or rs["bridge_labels_read"]
                or rs["model_refits"]
                or not np.isfinite(rs["max_absolute_error"])
                or rs["max_absolute_error"] > 1e-10
                or len(rf) != 5
                or set(rf["fold"]) != set(range(5))
                or set(rf["run"]) != {name}
                or not np.isfinite(rf["max_absolute_error"]).all()
                or (rf["max_absolute_error"] > 1e-10).any()
                or digest(folder / "private_full_source_head.joblib")
                != r["summary"]["full_head_sha256"]
            ):
                raise ValueError("Saved source head/replay mismatch")
            if packets and any(
                not np.array_equal(r[k], packets[0][k]) for k in ("ids", "y", "folds")
            ):
                raise ValueError(
                    "Observer comparison is not author/target/fold matched"
                )
            files = [
                folder / n
                for n in (
                    "manifest.json",
                    "summary.json",
                    "private_oof_predictions.csv",
                    "private_full_source_head.joblib",
                )
            ]
            files += [folder / f"private_fold_{f}.joblib" for f in range(5)]
            files += [replay / "summary.json", replay / "replay_by_fold.csv"]
            inputs[name] = {
                str(path.relative_to(a.batch_dir)): digest(path) for path in files
            }
            packets.append(r)
            scores.append(
                {
                    "task": task,
                    "arm": arm,
                    "observer": feature,
                    "authors": len(r["ids"]),
                    "pooled_score": r["summary"]["score"],
                    "mean_fold_score": r["mean_fold_score"],
                }
            )
            traits.extend(
                dict(task=task, arm=arm, observer=feature, **row)
                for row in metrics(r["y"], r["p"], task)
            )
        paired.extend(
            dict(task=task, arm=arm, authors=len(packets[0]["ids"]), **row)
            for row in paired_triads(
                packets[0]["y"], [r["p"] for r in packets], packets[0]["folds"], task
            )
        )
        accepted.append({"task": task, "arm": arm})
        print(
            json.dumps(
                {"compared_task": task, "arm": arm, "authors": len(packets[0]["ids"])}
            ),
            flush=True,
        )
    if not accepted or (pending and not a.allow_partial):
        raise ValueError("Complete triads required; full report needs all45 heads")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    code = [
        "scripts/summarize_grain_neural_source_primary.py",
        "scripts/analyze_grain_ei_source_gap.py",
    ]
    manifest = {
        "protocol": "VM05_PRIMARY_NEURAL_PAIRED_READOUT_V1",
        "batch_identity": batch["identity"],
        "inputs": inputs,
        "code_sha256": {n: digest(ROOT / n) for n in code},
        "bootstrap": "1000 author-within-fold draws, seed731; pointwise, fixed fitted heads",
        "implementation_after_first_primary_results": True,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
    ) as z:
        for name in code:
            if name not in z.namelist():
                z.write(ROOT / name, name)
    output_hashes = save_tables(a.output_dir, scores, traits, paired)
    summary = {
        "status": "PARTIAL" if pending else "COMPLETE",
        "parent_identity": manifest["identity"],
        "accepted_groups": accepted,
        "pending_groups": pending,
        "heads": len(scores),
        "planned_heads": 45,
        "paired_rows": len(paired),
        "source_label_tables_read": False,
        "previously_saved_oof_outcomes_reused": True,
        "bridge_labels_read": False,
        "model_refits": 0,
        "program_complete": False,
        "scope": "Source readability only; pointwise exploratory CI, not simultaneous/refit or internal-use evidence",
        "outputs_sha256": output_hashes,
    }
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
