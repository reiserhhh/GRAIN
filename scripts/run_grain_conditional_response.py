#!/usr/bin/env python3
"""D3B: freeze shared/conditional vector operators, then evaluate fresh authors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_real_author_path import load_packet, prediction_quality
from scripts.run_grain_real_author_path import key, read, sha, write

RIDGES = (1.0, 10.0, 100.0)
MODELS = ("shared", "grain_conditioned")


def grouped_folds(groups, nfolds=3):
    """Assign all views/donors/doses of an author to the same fixed CV fold."""
    unique = sorted(set(groups), key=lambda x: key(x, "d3b_inner_cv"))
    if len(unique) < nfolds:
        raise ValueError("Too few development authors")
    assignment = {uid: j % nfolds for j, uid in enumerate(unique)}
    return np.array([assignment[uid] for uid in groups])


def design(model, x, z):
    """Build origin-preserving response features with optional profile interactions."""
    reduced = (np.asarray(x) @ model["basis"]) / model["response_scale"]
    if not bool(model["conditional"]):
        return reduced
    phi = (np.asarray(z) - model["profile_mean"]) @ model["profile_basis"]
    phi /= model["profile_scale"]
    interactions = (reduced[:, :, None] * phi[:, None, :]).reshape(len(x), -1)
    return np.column_stack([reduced, interactions])


def fit_operator(x, z, y, groups, conditional, ridge, rank=8, profile_rank=4):
    """Learn all coordinates and gain coefficients exclusively from training authors."""
    x, z, y = np.asarray(x, float), np.asarray(z, float), np.asarray(y, float)
    if x.ndim != 2 or z.ndim != 2 or len(z) != len(x) or y.shape != (len(x),):
        raise ValueError("Invalid aligned vector/profile/response arrays")
    if not all(np.isfinite(a).all() for a in (x, z, y)) or ridge <= 0:
        raise ValueError("Finite data and positive ridge required")
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    count = min(rank, int(np.sum(s > max(s[0], 1.0) * 1e-10)))
    if count == 0:
        raise ValueError("No varying response directions")
    basis = vt[:count].T
    model = {
        "basis": basis,
        "response_scale": np.maximum(np.sqrt(np.mean((x @ basis) ** 2, axis=0)), 1e-10),
        "conditional": np.asarray(conditional),
        "ridge": np.asarray(ridge),
    }
    if conditional:
        _, indices = np.unique(groups, return_index=True)
        author_z = z[indices]
        center = author_z.mean(0)
        _, sz, vz = np.linalg.svd(author_z - center, full_matrices=False)
        dim = min(profile_rank, int(np.sum(sz > max(sz[0], 1.0) * 1e-10)))
        profile_basis = vz[:dim].T
        model.update(
            {
                "profile_mean": center,
                "profile_basis": profile_basis,
                "profile_scale": np.maximum(
                    np.std((author_z - center) @ profile_basis, axis=0), 1e-10
                ),
            }
        )
    features = design(model, x, z)
    model["weights"] = np.linalg.solve(
        features.T @ features + ridge * np.eye(features.shape[1]), features.T @ y
    )
    return model


def predict(model, x, z):
    """Evaluate the frozen mathematical operator, with no model forward call."""
    return design(model, x, z) @ model["weights"]


def dataset(folder):
    """Align paired intervention rows and independent frozen GRAIN author profiles."""
    _, rows, x, y = load_packet(folder)
    with np.load(folder / "private_profiles.npz", allow_pickle=False) as p:
        lookup = {uid: j for j, uid in enumerate(p["author_ids"])}
        z = p["grain_whitened"][[lookup[uid] for uid in rows.author_id]]
    return rows, x, z, y


def train(args):
    """Select ridge in grouped inner CV and freeze both models before new outcomes."""
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    rows, x, z, y = dataset(args.development)
    cv, selected = [], {}
    for task in sorted(rows.task.unique()):
        ix = np.flatnonzero((rows.task == task).to_numpy())
        xx, zz, yy = x[ix], z[ix], y[ix]
        groups = rows.iloc[ix].author_id.to_numpy()
        folds = grouped_folds(groups)
        selected[task] = {}
        for name in MODELS:
            conditional = name != "shared"
            choices = []
            for ridge in RIDGES:
                predictions = np.empty(len(ix))
                for fold in range(3):
                    fit = folds != fold
                    model = fit_operator(
                        xx[fit], zz[fit], yy[fit], groups[fit], conditional, ridge
                    )
                    predictions[~fit] = predict(model, xx[~fit], zz[~fit])
                measured = prediction_quality(yy, predictions)
                cv.append({"task": task, "model": name, "ridge": ridge, **measured})
                choices.append((float(np.mean((predictions - yy) ** 2)), ridge))
            _, ridge = min(choices)
            model = fit_operator(xx, zz, yy, groups, conditional, ridge)
            path = out / f"{task}_{name}.npz"
            np.savez_compressed(path, **model)
            selected[task][name] = {
                "ridge": ridge,
                "model_file": path.name,
                "sha256": sha(path),
            }
    pd.DataFrame(cv).to_csv(out / "development_cv.csv", index=False)
    write(
        out / "frozen_selection.json",
        {
            "status": "FROZEN_BEFORE_FRESH_AUTHOR_EVALUATION",
            "development": str(args.development.resolve()),
            "development_manifest_sha256": sha(args.development / "manifest.json"),
            "development_authors": rows.author_id.nunique(),
            "private_development_ids": sorted(rows.author_id.unique().tolist()),
            "selection": "minimum grouped inner-CV MSE, ridge 1/10/100, three author folds",
            "ranks": {"response": 8, "profile": 4},
            "script_sha256": sha(__file__),
            "models": selected,
            "labels_read": False,
        },
    )
    print(pd.DataFrame(cv).to_string(index=False), flush=True)


def paired_mse_interval(rows, y, shared, conditional, draws=2000):
    """Bootstrap paired MSE changes by author, not by repeated intervention row."""
    frame = pd.DataFrame(
        {
            "author_id": rows.author_id.to_numpy(),
            "change": (y - conditional) ** 2 - (y - shared) ** 2,
        }
    )
    a = frame.groupby("author_id").change.mean().to_numpy()
    rng = np.random.default_rng(20260908)
    b = a[rng.integers(len(a), size=(draws, len(a)))].mean(1)
    return {
        "conditional_minus_shared_mse": float(a.mean()),
        "ci_lower": float(np.quantile(b, 0.025)),
        "ci_upper": float(np.quantile(b, 0.975)),
        "authors": len(a),
    }


def evaluate(args):
    """Read fresh complete authors once, retaining aggregate and per-condition results."""
    out = args.output_dir
    frozen = read(out / "frozen_selection.json")
    if frozen["script_sha256"] != sha(__file__):
        raise ValueError("Frozen analysis changed")
    if (out / "summary.json").exists():
        raise ValueError("Evaluation already recorded; do not overwrite")
    rows, x, z, y = dataset(args.holdout)
    if set(rows.author_id) & set(frozen["private_development_ids"]):
        raise ValueError("Development/holdout author overlap")
    if not (rows.split == "held_author").all():
        raise ValueError("Require a wholly new held-author panel")
    metrics, comparisons, private = [], [], []
    for task in sorted(rows.task.unique()):
        ix = np.flatnonzero((rows.task == task).to_numpy())
        frame = rows.iloc[ix].copy()
        predictions = {}
        for name in MODELS:
            spec = frozen["models"][task][name]
            path = out / spec["model_file"]
            if sha(path) != spec["sha256"]:
                raise ValueError("Frozen fitted operator changed")
            with np.load(path, allow_pickle=False) as saved:
                model = dict(saved)
            p = predict(model, x[ix], z[ix])
            predictions[name] = p
            metrics.append(
                {
                    "task": task,
                    "model": name,
                    "condition": "all",
                    "held_authors": frame.author_id.nunique(),
                    "observations": len(ix),
                    "ridge": spec["ridge"],
                    **prediction_quality(y[ix], p),
                }
            )
            for donor in sorted(frame.donor.unique()):
                for alpha in (0.5, 1.0):
                    mask = ((frame.donor == donor) & (frame.alpha == alpha)).to_numpy()
                    metrics.append(
                        {
                            "task": task,
                            "model": name,
                            "condition": f"{donor}_{alpha}",
                            "held_authors": frame.author_id.nunique(),
                            "observations": int(mask.sum()),
                            "ridge": spec["ridge"],
                            **prediction_quality(y[ix][mask], p[mask]),
                        }
                    )
            part = frame[["author_id", "view", "task", "donor", "alpha"]].copy()
            part["model"], part["response"], part["prediction"] = name, y[ix], p
            private.append(part)
        comparisons.append(
            {
                "task": task,
                **paired_mse_interval(
                    frame,
                    y[ix],
                    predictions["shared"],
                    predictions["grain_conditioned"],
                ),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "held_author_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_model_comparison.csv", index=False)
    pd.concat(private, ignore_index=True).to_csv(
        out / "private_held_predictions.csv", index=False
    )
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "fresh_authors": rows.author_id.nunique(),
            "holdout_manifest_sha256": sha(args.holdout / "manifest.json"),
            "frozen_selection_sha256": sha(out / "frozen_selection.json"),
            "metrics": metrics,
            "comparisons": comparisons,
            "labels_read": False,
        },
    )
    print(
        pd.DataFrame(metrics).query("condition == 'all'").to_string(index=False),
        flush=True,
    )
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


def main():
    """Run the two finite stages without calling a neural model or reading labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("fit", "evaluate"))
    parser.add_argument("--development", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / "results"):
        raise ValueError("Private artifacts must stay in ignored results/")
    (train if args.phase == "fit" else evaluate)(args)


if __name__ == "__main__":
    main()
