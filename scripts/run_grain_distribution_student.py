#!/usr/bin/env python3
"""S5B: nonlinear readout versus empirical-distribution teacher regression."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.external_material import unpack_features
from scripts.analyze_grain_functional_basis import author_folds
from scripts.run_grain_real_author_path import BASE, key, read, sha, write
from scripts.run_grain_student_learning_curve import feature_cache
from scripts.run_grain_supervised_student import freeze_and_report, teacher_arrays

METHODS = ("linear_mean", "mean_rff", "bag_rff", "mean_plus_bag")
LAMBDAS = (0.001, 0.01, 0.1, 1.0, 10.0)


def load_bags(connection, ids):
    """Read cue-mask comment vectors from verified private caches in author order."""
    hashes = read(connection / "private_cache_hashes.json")
    mapping = {
        Path(name).stem: (ROOT / name, digest) for name, digest in hashes.items()
    }
    result = []
    for uid in ids:
        path, digest = mapping[uid]
        if sha(path) != digest:
            raise ValueError("Comment embedding cache changed")
        metadata = read(path.with_suffix(".json"))
        with np.load(path, allow_pickle=False) as z:
            values = unpack_features(dict(z), metadata["feature_aliases"])[
                "cue_mask_comment_vectors"
            ]
        if values.shape != (3, 12, 1024) or not np.isfinite(values).all():
            raise ValueError("Expected complete three-view cue-mask bags")
        result.append(values.reshape(36, 1024))
    return np.asarray(result, np.float32)


def fixed_projection(bags, width=256, seed=731):
    """Map into fixed Gaussian Fourier frequencies; this step fits no sample data."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(bags.shape[-1], width)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=width).astype(np.float32)
    return bags @ omega, phase


def bandwidth(x, seed=731):
    """Estimate a median pair distance from training data only, excluding self pairs."""
    flat = np.asarray(x).reshape(-1, x.shape[-1])
    if len(flat) < 2:
        raise ValueError("At least two training points needed")
    rng = np.random.default_rng(seed)
    a = rng.integers(len(flat), size=1024)
    b = (a + rng.integers(1, len(flat), size=1024)) % len(flat)
    distance = np.linalg.norm(flat[a] - flat[b], axis=1)
    positive = distance[distance > 1e-8]
    return float(np.median(positive)) if len(positive) else 1.0


def fourier_features(projection, phase, sigma, aggregate):
    """Return phi(mean h) or mean phi(h), approximating a Gaussian kernel map."""
    if sigma <= 0 or aggregate not in ("mean", "bag"):
        raise ValueError("Positive bandwidth and known aggregation required")
    if aggregate == "mean":
        projection = projection.mean(1)
    mapped = np.sqrt(2 / len(phase)) * np.cos(projection / sigma + phase)
    return mapped.mean(1) if aggregate == "bag" else mapped


def raw_features(bags, projected, phase, train):
    """Build train-bandwidth feature maps for all authors without fitting on tests."""
    means = bags.mean(1)
    mean_sigma, bag_sigma = bandwidth(means[train]), bandwidth(bags[train])
    blocks = {
        "linear_mean": means,
        "mean_rff": fourier_features(projected, phase, mean_sigma, "mean"),
        "bag_rff": fourier_features(projected, phase, bag_sigma, "bag"),
    }
    return blocks, {"mean_sigma": mean_sigma, "bag_sigma": bag_sigma}


def block_normalize(raw, train):
    """Center and give each kernel block unit training trace, preserving its geometry."""
    blocks, parameters = {}, {}
    for name, x in raw.items():
        center = x[train].mean(0)
        scale = max(
            float(np.sqrt(np.mean(np.sum((x[train] - center) ** 2, axis=1)))), 1e-8
        )
        blocks[name] = (x - center) / scale
        parameters[name + "_center"] = center
        parameters[name + "_scale"] = scale
    blocks["mean_plus_bag"] = np.column_stack(
        (blocks["linear_mean"], blocks["bag_rff"])
    ) / np.sqrt(2)
    return blocks, parameters


def select_models(bags, projected, phase, teacher, names, outer_train, salt):
    """Choose risk-normalized penalties on inner-training transformations and teachers."""
    inner = author_folds(names[outer_train], 3, salt)
    losses = {(m, l): [] for m in METHODS for l in LAMBDAS}
    for fold in range(3):
        train = outer_train[inner != fold]
        valid = outer_train[inner == fold]
        raw, _ = raw_features(bags, projected, phase, train)
        blocks, _ = block_normalize(raw, train)
        for method, x in blocks.items():
            for penalty in LAMBDAS:
                model = Ridge(alpha=len(train) * penalty).fit(x[train], teacher[train])
                losses[method, penalty].append(
                    float(np.mean((model.predict(x[valid]) - teacher[valid]) ** 2))
                )
    chosen = {}
    for method in METHODS:
        penalty = min(LAMBDAS, key=lambda l: (np.mean(losses[method, l]), l))
        chosen[method] = penalty, float(np.mean(losses[method, penalty]))
    return chosen


def main():
    """Execute a finite kernel-mean comparison and save frozen scoring parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private artifacts belong in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_models").mkdir()
    source = args.connection / "private_predictions.csv"
    start = time.monotonic()
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S5B_KERNEL_MEAN_STUDENT_V1",
            "script_sha256": sha(__file__),
            "source_sha256": sha(source),
            "methods": METHODS,
            "lambdas": LAMBDAS,
            "rff_width": 256,
            "random_seed": 731,
            "bandwidth": "training median distance; 1024 nonself pairs",
            "scaling": "training scalar unit-trace per block, not per-frequency whitening",
            "selection": "inner3 teacher MSE only; outer5 s4_outer",
            "time_cap_seconds": 1200,
            "new_neural_forwards": 0,
            "scope": "exposed D1 source authors, finite random-feature approximation",
        },
    )
    ids, feature = feature_cache(args.connection, args.panel)
    bags = load_bags(args.connection, ids)
    if not np.allclose(bags.mean(1), feature["TF_cont"]["mean"], atol=1e-7):
        raise ValueError("Comment bags no longer reproduce S4 mean")
    targets = pd.read_csv(
        source, dtype={"author_id": str}, usecols=["author_id", "query", "native_query"]
    )
    rows, selections = [], []
    with threadpool_limits(limits=2):
        projected, phase = fixed_projection(bags)
        for query in ("TF_cont", "neuroticism"):
            names, h, teacher = teacher_arrays(ids, bags, targets, query)
            lookup = {uid: j for j, uid in enumerate(ids)}
            p = projected[[lookup[uid] for uid in names]]
            outer = author_folds(names, 5, "s4_outer")
            for fold in range(5):
                train = np.array(
                    sorted(
                        np.flatnonzero(outer != fold),
                        key=lambda j: key(names[j], f"s4_{fold}"),
                    )
                )
                test = np.flatnonzero(outer == fold)
                specs = select_models(
                    h, p, phase, teacher, names, train, f"s4_inner_{fold}"
                )
                raw, sigmas = raw_features(h, p, phase, train)
                blocks, params = block_normalize(raw, train)
                predictions = {}
                for method, x in blocks.items():
                    penalty, loss = specs[method]
                    model = Ridge(alpha=len(train) * penalty).fit(
                        x[train], teacher[train]
                    )
                    predictions[method] = model.predict(x[test])
                    np.savez_compressed(
                        out / "private_models" / f"{query}_{method}_{fold}.npz",
                        coef=model.coef_,
                        intercept=model.intercept_,
                        **params,
                        **sigmas,
                    )
                    selections.append(
                        {
                            "query": query,
                            "fold": fold,
                            "method": method,
                            "lambda": penalty,
                            "inner_mse": loss,
                            **sigmas,
                        }
                    )
                chosen = min(specs, key=lambda m: (specs[m][1], m))
                predictions["inner_chosen"] = predictions[chosen]
                selections.append(
                    {
                        "query": query,
                        "fold": fold,
                        "method": "inner_chosen",
                        "chosen": chosen,
                    }
                )
                for method, values in predictions.items():
                    rows.extend(
                        {
                            "author_id": names[j],
                            "query": query,
                            "method": method,
                            "fold": fold,
                            "teacher": float(teacher[j]),
                            "prediction": float(v),
                        }
                        for j, v in zip(test, values, strict=True)
                    )
                print(f"S5B {query} fold {fold + 1}/5: {chosen}", flush=True)
                if time.monotonic() - start > 1200:
                    pd.DataFrame(rows).to_csv(out / "private_partial.csv", index=False)
                    raise TimeoutError("S5B exceeded twenty-minute fit cap")
    freeze_and_report(out, rows, selections, source, start)


if __name__ == "__main__":
    main()
