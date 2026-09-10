#!/usr/bin/env python3
"""Construct conditional action moments on fixed author-disjoint source folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_energy import (
    fit_distribution,
    mixture_moments,
    mixture_weights,
)
from grain_research.vector_mechanisms.joint_readout import predict_joint
from grain_research.vector_mechanisms.kernel_action import contrast_features
from scripts.run_grain_joint_carrier_readout import (
    ARMS,
    OLD,
    PAIRS,
    PROJECTION,
    check,
    verify,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write

PARENT = BASE / "logic_joint_carrier_readout_v1"
PLAN = ROOT / "docs/GRAIN_CONDITIONAL_ENERGY_PLAN_V1.md"


def source_inputs():
    """Load only source metadata and numeric projections, not questionnaire fields."""
    data = pd.read_csv(
        PROJECTION / "private_targets.csv",
        dtype={"author_id": str},
        usecols=["index", "author_id", "query", "view", "fold"],
    )
    data = data[data["query"] == "neuroticism"].sort_values(["author_id", "view"])
    if (
        len(data) != 72
        or data.groupby("author_id").fold.nunique().ne(1).any()
        or data.duplicated(["author_id", "view"]).any()
    ):
        raise ValueError("Old24 source authors/three views and fixed folds required")
    designs = {r["index"]: r for r in read(OLD / "private_design.json")}
    packets = {}
    for index in data["index"].unique():
        with np.load(PROJECTION / "private_projected" / f"author_{index:03d}.npz") as z:
            packets[index] = dict(z)
    with np.load(PROJECTION / "private_directions.npz") as z:
        eta = z["neuroticism__eta"].copy()
    pairs = []
    for r in data[data["view"] == 0].to_dict("records"):
        design = designs[r["index"]]
        if design["query"] != r["query"] or str(design["author_id"]) != r["author_id"]:
            raise ValueError("Metadata/projection identity mismatch")
        sizes = [len(g) for g in design["views"][0]["groups"]]
        for name, (i, j) in PAIRS.items():
            pairs.append({**r, "pair": name, "i": i, "j": j, "sizes": sizes})
    return pairs, packets, eta


def features(pairs, packets, eta, model):
    """Reconstruct precisely the old input family with its frozen source geometry."""
    left, right = [], []
    for r in pairs:
        z, size = packets[r["index"]], r["sizes"]
        offsets = np.cumsum([0] + size)
        if offsets[-1] != len(z["c_0"]):
            raise ValueError("Comment sizes do not match projected body")
        vectors = []
        for idx in (r["i"], r["j"]):
            sl = slice(offsets[idx], offsets[idx + 1])
            phi = contrast_features(
                z["c_0"][sl],
                z["d_0"][sl],
                z["position_0"][sl],
                eta,
                model["center"],
                model["geometry_scale"],
                model["arm"],
            )
            vectors.append(np.concatenate(phi) * size[idx] / offsets[-1])
        left.append(vectors[0])
        right.append(vectors[1])
    return np.array(left), np.array(right)


def construct(name):
    """Build one candidate family; all means/moments use training authors only."""
    verify(PARENT)
    freeze = read(PARENT / "joint_freeze.json")
    for path, digest in freeze["files"].items():
        check(PARENT / path, digest)
    out = BASE / name
    out.mkdir(exist_ok=False)
    files = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/conditional_energy.py",
        ROOT / "grain_research/vector_mechanisms/joint_readout.py",
        ROOT / "grain_research/vector_mechanisms/kernel_action.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONDITIONAL_ENERGY_V1",
            "parent_manifest_sha256": sha(PARENT / "manifest.json"),
            "joint_freeze_sha256": sha(PARENT / "joint_freeze.json"),
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "source_only": True,
            "human_labels_read": False,
            "new_neural_calls": 0,
            "optimizations": 0,
            "penalty": 0.01,
        },
    )
    pairs, packets, eta = source_inputs()
    old = read(PARENT / "private_joint_predictions.json")
    lookup = {(r["author_id"], r["pair"], r["arm"]): r for r in old}
    if len(lookup) != 192:
        raise ValueError("Unique old pair/arm records required")
    targets = np.array(
        [lookup[r["author_id"], r["pair"], "zero"]["actual_mixed"] for r in pairs]
    )
    output, models, arrays = (
        [],
        [],
        {k: [] for k in ("mean", "covariance", "weights", "axis")},
    )
    replay_feature, replay_prediction, replay_mean = 0.0, 0.0, 0.0
    for old_model in read(PARENT / "private_joint_models.json"):
        fold, arm = old_model["fold"], old_model["arm"]
        if arm not in ARMS:
            raise ValueError("Only inherited input families permitted")
        left, right = features(pairs, packets, eta, old_model)
        train = np.array([r["fold"] != fold for r in pairs])
        if train.sum() != 36 or (~train).sum() != 12:
            raise ValueError("Exactly18/6 train/evaluation authors required")
        replay_feature = max(
            replay_feature,
            float(np.max(abs(left[train] - old_model["left"]))),
            float(np.max(abs(right[train] - old_model["right"]))),
        )
        predicted_old = predict_joint(left[~train], right[~train], old_model)
        expected_old = np.array(
            [
                lookup[r["author_id"], r["pair"], arm]["predicted_mixed"]
                for r, held in zip(pairs, ~train, strict=True)
                if held
            ]
        )
        replay_prediction = max(
            replay_prediction, float(np.max(abs(predicted_old - expected_old)))
        )
        for pair in PAIRS:
            subset = np.array([r["pair"] == pair for r in pairs])
            fit, held = train & subset, (~train) & subset
            if fit.sum() != 18 or held.sum() != 6:
                raise ValueError("Pair-specific18/6 author support required")
            model = fit_distribution(
                left[fit], right[fit], targets[fit], old_model["scale"]
            )
            weights = mixture_weights(left[held], right[held], model)
            moments = mixture_moments(weights, targets[fit])
            mean = targets[fit].mean(0)
            axis = mean / np.linalg.norm(mean)
            model_index = len(models)
            models.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "pair": pair,
                    "training_authors": [
                        r["author_id"] for r, yes in zip(pairs, fit, strict=True) if yes
                    ],
                    **{
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in model.items()
                    },
                }
            )
            for j, ix in enumerate(np.flatnonzero(held)):
                r = pairs[ix]
                old_row = lookup[r["author_id"], pair, arm]
                old_mean = np.asarray(
                    lookup[r["author_id"], pair, "pair_train_mean"]["predicted_mixed"]
                )
                replay_mean = max(replay_mean, float(np.max(abs(mean - old_mean))))
                arrays["mean"].append(moments["mean"][j])
                arrays["covariance"].append(moments["covariance"][j])
                arrays["weights"].append(weights[j])
                arrays["axis"].append(axis)
                parallel = float(weights[j, 1:] @ (targets[fit] @ axis) ** 2)
                output.append(
                    {
                        "author_id": r["author_id"],
                        "fold": fold,
                        "pair": pair,
                        "arm": arm,
                        "model_index": model_index,
                        "row_index": len(output),
                        "energy": moments["energy"][j],
                        "mean_energy": moments["mean_energy"][j],
                        "variance_trace": moments["variance_trace"][j],
                        "parallel_energy": parallel,
                        "perpendicular_energy": moments["energy"][j] - parallel,
                        "reference_mass": moments["reference_mass"][j],
                        "effective_atoms": moments["effective_atoms"][j],
                        "old_vector_energy": float(
                            np.sum(np.asarray(old_row["predicted_mixed"]) ** 2)
                        ),
                        "pair_mean_energy": float(mean @ mean),
                        "pair_energy_mean": float(
                            np.sum(targets[fit] ** 2, axis=1).mean()
                        ),
                    }
                )
    if (
        len(output) != 96
        or len(models) != 16
        or max(replay_feature, replay_prediction, replay_mean) > 1e-12
    ):
        raise ValueError("Frozen dimensions or exact old replay failed")
    pd.DataFrame(output).to_csv(out / "private_moment_predictions.csv", index=False)
    np.savez_compressed(
        out / "private_moments.npz", **{k: np.array(v) for k, v in arrays.items()}
    )
    write(out / "private_distribution_models.json", models)
    summary = {
        "status": "PREDICTIONS_FROZEN",
        "authors": 24,
        "pairs": 48,
        "input_families": 2,
        "rows": 96,
        "smoother_constructions": 16,
        "optimizations": 0,
        "human_labels_read": False,
        "new_neural_calls": 0,
        "feature_replay_max": replay_feature,
        "old_prediction_replay_max": replay_prediction,
        "pair_mean_replay_max": replay_mean,
    }
    write(out / "construction_summary.json", summary)
    names = (
        "private_moment_predictions.csv",
        "private_moments.npz",
        "private_distribution_models.json",
        "construction_summary.json",
    )
    write(
        out / "prediction_freeze.json",
        {
            "manifest_sha256": sha(out / "manifest.json"),
            "files": {n: sha(out / n) for n in names},
            "held_native_targets_used_for_selection": False,
            "human_labels_read": False,
        },
    )
    print(summary)


def verified(out):
    """Verify the fixed program and its source/OOF numerical packet."""
    verify(PARENT)
    m = read(out / "manifest.json")
    for path, digest in m["code_hashes"].items():
        check(ROOT / path, digest)
    check(PARENT / "manifest.json", m["parent_manifest_sha256"])
    check(PARENT / "joint_freeze.json", m["joint_freeze_sha256"])
    for path, digest in read(PARENT / "joint_freeze.json")["files"].items():
        check(PARENT / path, digest)
    frozen = read(out / "prediction_freeze.json")
    check(out / "manifest.json", frozen["manifest_sha256"])
    for path, digest in frozen["files"].items():
        check(out / path, digest)
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="logic_conditional_energy_v1")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Simple output name required")
    construct(args.name)
