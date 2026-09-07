#!/usr/bin/env python3
"""D2-D4: aggregate real evidence effects, group paths and held-author readouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_real_author_path import read, sha, write

SEED = 20260908


def correlation(x, y):
    """Return a finite association or explicit None for an undefined axis."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(pearsonr(x, y).statistic)


def cluster_mean(frame, column, draws=1000):
    """Bootstrap authors, averaging their paired views before resampling."""
    values = frame.groupby("author_id")[column].mean().to_numpy()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Require finite author-level contrasts")
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return {
        "authors": len(values),
        "mean": float(values.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def fit_low_rank(x, y, rank, ridge=1.0):
    """Fit an origin-preserving response operator using training vectors only."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Invalid training response matrix")
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(rank, int(np.sum(s > max(s[0], 1.0) * 1e-10)))
    basis = vt[:k].T
    z = x @ basis
    weights = np.linalg.solve(z.T @ z + ridge * np.eye(k), z.T @ y)
    return basis, weights


def prediction_quality(y, p):
    """Compare a response prediction with the no-change zero predictor."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    base = float(np.mean(y**2))
    mse = float(np.mean((y - p) ** 2))
    return {
        "r": correlation(y, p),
        "rmse": float(np.sqrt(mse)),
        "zero_rmse": float(np.sqrt(base)),
        "mse_ratio_to_zero": mse / base if base else None,
        "sign_agreement": float(np.mean(np.sign(y) == np.sign(p))),
    }


def finite_curvature(full, half_patch, whole_patch):
    """Measure deviation from a linear response at the two observed doses."""
    return np.asarray(whole_patch) - 2 * np.asarray(half_patch) + np.asarray(full)


def load_packet(folder):
    """Collect the complete fixed panel without accessing psychological outcomes."""
    status, manifest = read(folder / "run_status.json"), read(folder / "manifest.json")
    material = read(folder / "private_material.json")
    if (
        status["status"] != "COMPLETE"
        or len(material) != status["completed_author_views"]
        or sha(folder / "private_material.json") != manifest["material_sha256"]
        or sha(folder / "private_profiles.npz") != manifest["profiles_sha256"]
    ):
        raise ValueError("Use the complete frozen real-author packet")
    natural, effects, deltas, response_y = [], [], [], []
    for row in material:
        name = f"{row['author_id']}_{row['view']}"
        base = folder / "private_packets" / name
        packet = read(base.with_suffix(".json"))
        if packet["state_sha256"] != sha(base.with_suffix(".npz")):
            raise ValueError("Saved vector changed")
        scores = packet["readouts"]
        if len(scores) != 9 or any(
            packet[k] != row[k] for k in ("author_id", "view", "task", "split")
        ):
            raise ValueError("Author/condition mismatch")
        prefix = {k: row[k] for k in ("author_id", "view", "task", "split")}
        full = scores["full"]["conditional_score"]
        record = {**prefix, **{k: v["conditional_score"] for k, v in scores.items()}}
        record["selected_minus_control_length"] = sum(
            row["comment_lengths"][i] for i in row["selected"]
        ) - sum(row["comment_lengths"][i] for i in row["control"])
        record["abs_removal_advantage"] = abs(record["remove_selected"] - full) - abs(
            record["remove_control"] - full
        )
        record["preservation_advantage"] = abs(record["control_only"] - full) - abs(
            record["selected_only"] - full
        )
        record["selection_contrast"] = np.mean(
            np.abs(np.array(row["selection_scores"])[row["selected"]])
        ) - np.mean(np.abs(np.array(row["selection_scores"])[row["control"]]))
        natural.append(record)
        with np.load(base.with_suffix(".npz"), allow_pickle=False) as z:
            for donor in ("remove_selected", "remove_control"):
                d = z[f"{donor}__block_16"] - z["full__block_16"]
                natural_shift = scores[donor]["conditional_score"] - full
                for alpha in (0.5, 1.0):
                    patched = (
                        scores[f"patch_{donor}_{alpha}"]["conditional_score"] - full
                    )
                    effects.append(
                        {
                            **prefix,
                            "donor": donor,
                            "alpha": alpha,
                            "natural_shift": natural_shift,
                            "patched_shift": patched,
                            "displacement_norm": float(np.linalg.norm(alpha * d)),
                            "full_score": full,
                        }
                    )
                    deltas.append(alpha * d)
                    response_y.append(patched)
    return (
        pd.DataFrame(natural),
        pd.DataFrame(effects),
        np.asarray(deltas),
        np.asarray(response_y),
    )


def d2_summaries(natural, effects):
    """Quantify evidence selection and query-end vector transfer, retaining both tasks."""
    rows = []
    for task, group in natural.groupby("task"):
        for column in (
            "abs_removal_advantage",
            "preservation_advantage",
            "selection_contrast",
            "selected_minus_control_length",
        ):
            rows.append(
                {"task": task, "measure": column, **cluster_mean(group, column)}
            )
        means = group.groupby("author_id").mean(numeric_only=True)
        for condition in (
            "selected_only",
            "control_only",
            "remove_selected",
            "remove_control",
        ):
            rows.append(
                {
                    "task": task,
                    "measure": condition + "_full_score_r",
                    "authors": len(means),
                    "mean": correlation(means[condition], means["full"]),
                }
            )
        for donor in ("remove_selected", "remove_control"):
            whole = means[f"patch_{donor}_1.0"] - means["full"]
            curvature = finite_curvature(
                means["full"], means[f"patch_{donor}_0.5"], means[f"patch_{donor}_1.0"]
            )
            whole_rms = float(np.sqrt(np.mean(whole**2)))
            curve_rms = float(np.sqrt(np.mean(curvature**2)))
            rows.append(
                {
                    "task": task,
                    "measure": f"curvature_{donor}",
                    "authors": len(means),
                    "curvature_rms": curve_rms,
                    "patched_rms": whole_rms,
                    "curvature_to_effect_ratio": curve_rms / whole_rms
                    if whole_rms
                    else None,
                }
            )
    for (task, donor, alpha), group in effects.groupby(["task", "donor", "alpha"]):
        means = group.groupby("author_id").mean(numeric_only=True)
        result = prediction_quality(means.natural_shift, means.patched_shift)
        rows.append(
            {
                "task": task,
                "measure": f"vector_transfer_{donor}_{alpha}",
                "authors": len(means),
                **result,
                "natural_rms": float(np.sqrt(np.mean(means.natural_shift**2))),
                "patched_rms": float(np.sqrt(np.mean(means.patched_shift**2))),
            }
        )
    return rows


def group_analysis(natural, profiles, membership):
    """Predict held authors' response profiles from independent GRAIN/E5 neighbors."""
    rows, private = [], []
    lookup = {uid: i for i, uid in enumerate(profiles["author_ids"])}
    role = {r["author_id"]: r["split"] for r in membership}
    for task, frame in natural.groupby("task"):
        means = frame.groupby("author_id").mean(numeric_only=True)
        ids = means.index.tolist()
        columns = [
            "full",
            "remove_selected",
            "remove_control",
            "selected_only",
            "control_only",
            "patch_remove_selected_1.0",
            "patch_remove_control_1.0",
        ]
        y = means[columns].to_numpy(copy=True)
        y[:, 1:] -= y[:, [0]]
        train = np.array([role[uid] == "discovery" for uid in ids])
        test = ~train
        center, scale = y[train].mean(0), np.maximum(y[train].std(0), 1e-8)
        yy = (y - center) / scale
        coords = [
            "full_score",
            "remove_selected_delta",
            "remove_control_delta",
            "selected_only_delta",
            "control_only_delta",
            "patch_selected_delta",
            "patch_control_delta",
        ]
        geometry = {
            "grain_whitened": "euclidean",
            "canonical_distances": "euclidean",
            "e5_mean": "cosine",
        }
        for name, metric in geometry.items():
            x = profiles[name][[lookup[uid] for uid in ids]]
            distances = pairwise_distances(x[test], x[train], metric=metric)
            neighbors = np.argsort(distances, axis=1, kind="stable")[:, :3]
            p = yy[train][neighbors].mean(1)
            for j, coord in enumerate(coords):
                measured = prediction_quality(yy[test, j], p[:, j])
                rows.append(
                    {
                        "task": task,
                        "geometry": name,
                        "coordinate": coord,
                        "train_authors": int(train.sum()),
                        "held_authors": int(test.sum()),
                        **measured,
                    }
                )
            rows.append(
                {
                    "task": task,
                    "geometry": name,
                    "coordinate": "all_response_coordinates",
                    "train_authors": int(train.sum()),
                    "held_authors": int(test.sum()),
                    **prediction_quality(yy[test].ravel(), p.ravel()),
                }
            )
            for i, uid in enumerate(np.array(ids)[test]):
                private.append(
                    {
                        "author_id": uid,
                        "task": task,
                        "geometry": name,
                        "standardized_error": float(np.mean((p[i] - yy[test][i]) ** 2)),
                    }
                )
    return rows, private


def response_readouts(effects, deltas, y, output):
    """Fit compact linear operators on discovery authors; score held-author interventions."""
    rows, records = [], []
    for task in sorted(effects.task.unique()):
        tr = (effects.task == task) & (effects.split == "discovery")
        te = (effects.task == task) & (effects.split == "held_author")
        if set(effects.loc[tr, "author_id"]) & set(effects.loc[te, "author_id"]):
            raise ValueError("Author leakage in response readout")
        for rank in (2, 8):
            basis, weights = fit_low_rank(deltas[tr], y[tr], rank)
            p = (deltas[te] @ basis) @ weights
            np.savez_compressed(
                output / f"operator_{task}_rank{rank}.npz", basis=basis, weights=weights
            )
            label = f"rank{rank}"
            rows.append(
                {
                    "task": task,
                    "model": label,
                    "held_authors": effects.loc[te, "author_id"].nunique(),
                    "held_interventions": int(te.sum()),
                    "actual_rank": len(weights),
                    **prediction_quality(y[te], p),
                }
            )
            for i, index in enumerate(np.flatnonzero(te)):
                records.append(
                    {
                        **effects.iloc[index].to_dict(),
                        "model": label,
                        "observed_response": float(y[index]),
                        "predicted_response": float(p[i]),
                    }
                )
        # A scalar baseline tests whether displacement size alone explains the response.
        size_train = np.linalg.norm(deltas[tr], axis=1)
        coefficient = float(size_train @ y[tr] / (size_train @ size_train + 1.0))
        pred = np.linalg.norm(deltas[te], axis=1) * coefficient
        rows.append(
            {
                "task": task,
                "model": "norm_only",
                "held_authors": effects.loc[te, "author_id"].nunique(),
                "held_interventions": int(te.sum()),
                "actual_rank": 1,
                **prediction_quality(y[te], pred),
            }
        )
    return rows, records


def main():
    """Generate the complete predeclared mechanism summaries and operator artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or not args.output_dir.resolve().is_relative_to(
        ROOT / "results"
    ):
        raise ValueError("Use a fresh ignored results directory")
    natural, effects, deltas, y = load_packet(args.input_dir)
    args.output_dir.mkdir(parents=True)
    with np.load(args.input_dir / "private_profiles.npz", allow_pickle=False) as z:
        profiles = dict(z)
    d2 = d2_summaries(natural, effects)
    d3, neighbor_private = group_analysis(
        natural, profiles, read(args.input_dir / "membership.json")
    )
    d4, operator_private = response_readouts(effects, deltas, y, args.output_dir)
    for name, rows in (
        ("d2_evidence_vector_summary", d2),
        ("d3_neighborhood_summary", d3),
        ("d4_operator_summary", d4),
        ("private_natural", natural),
        ("private_effects", effects),
        ("private_neighbor_errors", neighbor_private),
        ("private_operator_predictions", operator_private),
    ):
        pd.DataFrame(rows).to_csv(args.output_dir / (name + ".csv"), index=False)
    write(
        args.output_dir / "summary.json",
        {
            "status": "COMPLETE_DISCOVERY_ANALYSIS",
            "author_views": len(natural),
            "authors": natural.author_id.nunique(),
            "vector_interventions": len(effects),
            "source_manifest_sha256": sha(args.input_dir / "manifest.json"),
            "analysis_code_sha256": sha(__file__),
            "labels_read": False,
            "bridge_outcomes_read": False,
            "bootstrap_unit": "author mean over two views",
            "group_knn": 3,
            "operator_ranks": [2, 8],
            "ridge": 1.0,
            "new_forwards": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_DISCOVERY_ANALYSIS",
                "authors": natural.author_id.nunique(),
            }
        )
    )


if __name__ == "__main__":
    main()
