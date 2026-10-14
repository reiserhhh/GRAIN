#!/usr/bin/env python3
"""Replay legacy Big5/MCD metrics and distinguish H2 protocol snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.navigation import digest, read_json, write_json

PERSONA = ROOT.parent / "project persona"
TRAITS = [
    "Extraversion",
    "Neuroticism",
    "Agreeableness",
    "Conscientiousness",
    "Openness",
]
DEFAULT = (
    ROOT
    / "results/grain_vector_mechanisms/all_plan_closure_v1/historical_acceptance_r2"
)


def correlation(y, p):
    """Finite paired Pearson with explicit support, never a fabricated zero."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    keep = np.isfinite(y) & np.isfinite(p)
    n = int(keep.sum())
    return (
        float(pearsonr(y[keep], p[keep]).statistic)
        if n >= 3 and np.std(y[keep]) > 0 and np.std(p[keep]) > 0
        else None
    ), n


def result_tables(directory):
    """Exclude macOS AppleDouble sidecars from scientific output inventory."""
    return sorted(p for p in directory.glob("*.csv") if not p.name.startswith("."))


def score_big5(frame, rule):
    """Keep raw, clipped and per-trait valid-range estimates distinct."""
    rows = []
    for trait in TRAITS:
        y, p = (
            frame[f"gt_{trait}"].to_numpy(float),
            frame[f"pred_{trait}"].to_numpy(float),
        )
        finite = np.isfinite(y) & np.isfinite(p)
        invalid = finite & ((p < 0) | (p > 1))
        if rule == "drop_oob_per_trait":
            p = np.where(invalid, np.nan, p)
        elif rule == "clip_0_1":
            p = np.clip(p, 0, 1)
        elif rule != "raw":
            raise ValueError("Unknown score rule")
        r, n = correlation(y, p)
        rows.append({"trait": trait, "pearson": r, "n": n, "n_oob": int(invalid.sum())})
    return rows


def run(out, persona=PERSONA):
    """Read cached outputs only; no legacy script execution, training or inference."""
    if out.exists():
        raise ValueError("Use a fresh reconciliation output")
    out.mkdir(parents=True)
    inputs = {}

    def record(path):
        inputs[str(path.relative_to(persona))] = digest(path)
        return path

    def csv(path, **kwargs):
        return pd.read_csv(record(path), **kwargs)

    def js(path):
        return read_json(record(path))

    root = persona / "analysis/output/pandora_official_big5_cv_full"
    final = js(root / "BIG5_CV_FINAL_REPORT.json")
    record(root / "BIG5_RECONCILIATION_REPORT.md")
    folds, rows, matches = [], [], []
    for fold in range(5):
        data = csv(root / f"fold_{fold}/fold_{fold}_predictions.csv")
        summary = js(root / f"fold_{fold}/PANDORA_OFFICIAL_BIG5_CV_SUMMARY.json")
        if data.user_id.duplicated().any() or set(data.official_fold.astype(int)) != {
            fold
        }:
            raise ValueError("Big5 user/fold identity error")
        folds.append(data)
        raw = score_big5(data, "raw")
        raw_mean = float(np.mean([r["pearson"] for r in raw]))
        if not np.isclose(
            raw_mean,
            summary["fold_reports"][str(fold)]["metrics"]["mean_pearson"],
            atol=1e-10,
        ):
            raise ValueError("Raw fold summary mismatch")
    pooled = pd.concat(folds, ignore_index=True)
    if pooled.user_id.duplicated().any():
        raise ValueError("Authors cross Big5 test folds")
    for fold, data in [(str(i), d) for i, d in enumerate(folds)] + [("pooled", pooled)]:
        target = final["pooled"] if fold == "pooled" else final["fold_reports"][fold]
        for rule in ("raw", "clip_0_1", "drop_oob_per_trait"):
            metrics = score_big5(data, rule)
            rows.extend(
                {"model": "lora", "fold": fold, "rule": rule, **r} for r in metrics
            )
            if rule == "drop_oob_per_trait":
                mean = float(np.mean([r["pearson"] for r in metrics]))
                ok = (
                    abs(mean - target["mean_pearson"]) <= 5.01e-5
                    and len(data) == target["n"]
                )
                ok &= all(
                    r["n"] == target["traits"][r["trait"]]["n_clean"]
                    and abs(r["pearson"] - target["traits"][r["trait"]]["pearson"])
                    <= 5.01e-5
                    for r in metrics
                )
                matches.append(
                    {
                        "fold": fold,
                        "mean_pearson": mean,
                        "final_rounding_match": bool(ok),
                    }
                )
    control_summary = js(root / "base_control/PANDORA_OFFICIAL_BIG5_CV_SUMMARY.json")
    base_files = sorted((root / "base_control").glob("fold_*_predictions.csv"))
    controls = []
    for path in base_files:
        data = csv(path)
        controls.extend(data.official_fold.astype(int).unique().tolist())
        for rule in ("raw", "clip_0_1", "drop_oob_per_trait"):
            rows.extend(
                {"model": "base", "fold": path.stem, "rule": rule, **r}
                for r in score_big5(data, rule)
            )
    pd.DataFrame(rows).to_csv(out / "big5_metric_rules.csv", index=False)
    pd.DataFrame(matches).to_csv(out / "big5_final_replay.csv", index=False)

    mcd = persona / "results/pandora_baseline_mcd_integration"
    pred = csv(mcd / "predictions.csv")
    stored_folds = csv(mcd / "metrics_by_fold.csv").set_index(
        ["condition", "trait", "fold"]
    )
    stored_board = csv(mcd / "leaderboard.csv").set_index("condition")
    mcd_rows, fold_error = [], 0.0
    for (condition, trait, fold), data in pred.groupby(
        ["condition", "trait", "fold"], sort=True
    ):
        if data.author.duplicated().any():
            raise ValueError("Duplicated MCD author within trait/fold")
        r, n = correlation(data.true, data.pred)
        saved = stored_folds.loc[(condition, trait, fold)]
        fold_error = max(fold_error, abs(r - saved.pearson))
        if n != saved.n_test:
            raise ValueError("MCD count mismatch")
        mcd_rows.append(
            {
                "condition": condition,
                "trait": trait,
                "fold": int(fold),
                "pearson": r,
                "n": n,
            }
        )
    for _, data in pred.groupby(["condition", "trait"]):
        if data.author.duplicated().any() or set(data.fold) != set(range(5)):
            raise ValueError("MCD cross-fold duplicate or missing fold")
    mf = pd.DataFrame(mcd_rows)
    board = (
        mf.groupby("condition", as_index=False)
        .pearson.mean()
        .rename(columns={"pearson": "mean_fold_trait_r"})
    )
    for _, r in board.iterrows():
        fold_error = max(
            fold_error,
            abs(r.mean_fold_trait_r - stored_board.loc[r.condition, "mean_pearson"]),
        )
    reference_rows = []
    for variant in ("LR-N", "LR-NP"):
        for trait in sorted(pred.trait.unique()):
            path = (
                persona
                / f"data_sets/PANDORA_official/pandora_baseline/res/{trait}-{variant}/preds.csv"
            )
            data = csv(path)
            vals = [
                correlation(d.true, d.pred)[0]
                for _, d in data.groupby("fold", sort=True)
            ]
            if len(vals) != 5:
                raise ValueError("Official reference fold count")
            reference_rows.append(
                {
                    "condition": variant,
                    "trait": trait,
                    "mean_fold_r": float(np.mean(vals)),
                }
            )
    reference = pd.DataFrame(reference_rows)
    mf.to_csv(out / "mcd_fold_replay.csv", index=False)
    board.to_csv(out / "mcd_leaderboard_replay.csv", index=False)
    reference.to_csv(out / "official_reference_replay.csv", index=False)
    boot = csv(mcd / "paired_bootstrap_samples.csv")
    boot_rows = []
    for comparison, data in boot.groupby("comparison", sort=True):
        values = data.bootstrap_delta.to_numpy(float)
        tail = int(min((values <= 0).sum(), (values >= 0).sum()))
        boot_rows.append(
            {
                "comparison": comparison,
                "samples": len(values),
                "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "opposite_tail_count": tail,
                "legacy_empirical_two_sided_tail": 2 * tail / len(values),
                "scope": "stored bootstrap distribution replay, not a new resampling run",
            }
        )
    pd.DataFrame(boot_rows).to_csv(out / "mcd_stored_bootstrap_replay.csv", index=False)

    h2 = persona / "analysis/output/h2_corrected"
    partitions = {}
    for name in ("big5_train", "mbti_train", "bridge_eval"):
        data = csv(h2 / f"datasets/{name}.csv", usecols=["user_id"])
        if data.user_id.isna().any() or data.user_id.duplicated().any():
            raise ValueError("H2 partition identity error")
        partitions[name] = set(data.user_id)
    overlaps = {
        f"{a}__{b}": len(partitions[a] & partitions[b])
        for a, b in (
            ("big5_train", "mbti_train"),
            ("big5_train", "bridge_eval"),
            ("mbti_train", "bridge_eval"),
        )
    }
    cross = js(h2 / "cross_framework/cross_framework_results.json")
    original = js(h2 / "shared_rep/H2_CORRECTED_RESULTS.json")
    same = js(h2 / "same_user_consistency/SAME_USER_CONSISTENCY.json")
    for name in (
        "run_h2_shared_rep_corrected.py",
        "run_cross_framework_path_comparison.py",
        "run_hybrid_v1_confirmation.py",
    ):
        record(persona / "_archive/scripts" / name)
    h2_rows = []
    for path in cross["paths"]:
        predicted = np.asarray(path["mat_pred_5d"], float)
        actual = np.asarray(cross["mat_true_5d"], float)
        r, n = correlation(predicted.ravel(), actual.ravel())
        h2_rows.append(
            {
                "path": path["name"],
                "matrix_cells": n,
                "replayed_element_r": r,
                "stored_element_r": path["element_r_5d"],
                "rounding_match": abs(r - path["element_r_5d"]) < 5.01e-5,
                "stored_auc": path["mean_auc"],
                "stored_bal": path["mean_bal"],
                "scope": "matrix arithmetic only; no prediction refit",
            }
        )
    pd.DataFrame(h2_rows).to_csv(out / "h2_matrix_replay.csv", index=False)
    result = {
        "status": "RECONCILIATION_COMPLETE_WITH_PROTOCOL_CAVEATS",
        "source_repo": str(persona),
        "big5": {
            "authors": len(pooled),
            "final_clean_replay": all(x["final_rounding_match"] for x in matches),
            "pooled_clean_r": matches[-1]["mean_pearson"],
            "base_folds_in_files": sorted(set(controls)),
            "base_folds_in_summary": control_summary["folds"],
            "old_artifact_acceptance": "PASS under drop_oob_per_trait",
            "full_paired_base_control_closed": sorted(set(controls)) == list(range(5)),
        },
        "mcd": {
            "prediction_rows": len(pred),
            "fold_cells": len(mcd_rows),
            "max_replay_error": fold_error,
            "leaderboard": board.to_dict("records"),
            "official_baselines": reference.groupby("condition")
            .mean_fold_r.mean()
            .to_dict(),
            "bootstrap": boot_rows,
            "training_repeated": False,
        },
        "h2": {
            "partition_counts": {k: len(v) for k, v in partitions.items()},
            "overlaps": overlaps,
            "original_threshold": "prediction > 0",
            "cross_framework_threshold": "prediction > mean of evaluation predictions",
            "historical_4x4_r": original["structural_consistency"]["corrected"][
                "element_r"
            ],
            "legacy_same_run_4x4_r": original["structural_consistency"]["legacy"][
                "element_r"
            ],
            "legacy_external_snapshot_r": original["research_questions"][
                "Q3_structural_consistency_legacy"
            ],
            "matrix_arithmetic_pass": all(x["rounding_match"] for x in h2_rows),
            "same_user_auc": same["discrimination"]["auc_cosine"],
            "same_user_p": same["discrimination"]["p_value_permutation"],
            "same_user_scope": "historical result, not rerun; cross-author canonical-pair r is not this test",
        },
        "scope": "metric/partition replay, not cold training replication or closure of every historical experiment",
        "new_model_calls": 0,
        "training": False,
    }
    write_json(out / "summary.json", result)
    write_json(
        out / "manifest.json",
        {
            "inputs": inputs,
            "script_sha256": digest(Path(__file__)),
            "outputs": {p.name: digest(p) for p in result_tables(out)},
        },
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    run(parser.parse_args().output)
