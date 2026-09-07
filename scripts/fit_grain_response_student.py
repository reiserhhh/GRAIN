#!/usr/bin/env python3
"""S2: fixed-E5, query-conditioned students fitted to values and natural-edit effects."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_functional_basis import author_folds
from scripts.analyze_grain_multiquery_sensitivity import corr, load_panel
from scripts.run_grain_multiquery_sensitivity import QUERIES, edit_indices, unit_vectors
from scripts.run_grain_real_author_path import BASE, sha, write

CONDITIONS = ("full", "e1", "e2", "joint")
RIDGES, GAMMAS = (1.0, 10.0, 100.0), (0.0, 0.5, 1.0)


def panel_data(folder):
    """Join teacher scores to the fixed per-comment encoder without psychological labels."""
    manifest, records = load_panel(folder)
    path = folder / "private_encoder.npz"
    if sha(path) != manifest["encoder_sha256"]:
        raise ValueError("Frozen encoder changed")
    with np.load(path, allow_pickle=False) as z:
        encoded, poles = z["comment_vectors"], z["poles"]
    meta, values, comments = [], [], []
    for material, packet, _ in records:
        masks = edit_indices(12, material["e1"], material["e2"])
        vectors = encoded[packet["material_row"]]
        for q, query in enumerate(QUERIES):
            meta.append(
                {
                    "author_id": material["author_id"],
                    "view": material["view"],
                    "query": query,
                    "q": q,
                    "source_group": material["source_group"],
                }
            )
            scores = packet["queries"][query]["readouts"]
            values.append(
                [
                    scores[name]["conditional_score"] if name in scores else np.nan
                    for name in CONDITIONS
                ]
            )
            comments.append([vectors[masks[name]] for name in CONDITIONS])
    return pd.DataFrame(meta), np.asarray(values), comments, poles


def representative_landmarks(bank, count=16):
    """Select training-only farthest-point landmarks, without outcomes or test text."""
    bank = np.unique(np.asarray(bank, float), axis=0)
    count = min(count, len(bank))
    selected = [int(np.argmin(np.sum((bank - bank.mean(0)) ** 2, axis=1)))]
    distance = np.sum((bank - bank[selected[0]]) ** 2, axis=1)
    for _ in range(1, count):
        nxt = int(np.argmax(distance))
        if distance[nxt] < 1e-14:
            break
        selected.append(nxt)
        distance = np.minimum(distance, np.sum((bank - bank[nxt]) ** 2, axis=1))
    result = bank[selected]
    pairwise = np.sum((result[:, None] - result[None, :]) ** 2, axis=-1)
    positive = pairwise[pairwise > 1e-14]
    bandwidth_squared = float(np.median(positive)) if len(positive) else 1.0
    return result, bandwidth_squared


def raw_blocks(comments, queries, spec):
    """Preserve mean, query-relevant aggregation, and optional distribution information."""
    output = {"mean": [], "query": [], "std": [], "kernel": []}
    poles = unit_vectors(spec["poles"])
    for vectors, q in zip(comments, queries, strict=True):
        v = np.asarray(vectors, float)
        output["mean"].append(v.mean(0))
        if spec["family"] == "mean":
            continue
        relevance = np.abs(unit_vectors(v) @ (poles[q, 1] - poles[q, 0]))
        weights = np.exp(20.0 * (relevance - relevance.max()))
        output["query"].append((weights / weights.sum()) @ v)
        if spec["family"] in ("distribution", "quadratic"):
            output["std"].append(v.std(0))
            landmarks = spec["landmarks"]
            squared = np.maximum(
                np.sum(v * v, axis=1)[:, None]
                + np.sum(landmarks**2, axis=1)
                - 2 * v @ landmarks.T,
                0,
            )
            output["kernel"].append(
                np.exp(-squared / (2 * spec["bandwidth_squared"])).mean(0)
            )
    return {name: np.asarray(value) for name, value in output.items() if value}


def transform_features(comments, queries, spec):
    """Apply fixed block scales/landmarks and optional low-rank quadratic expansion."""
    blocks = raw_blocks(comments, queries, spec)
    pieces = [
        (blocks[name] - info["center"]) / info["scale"] / np.sqrt(blocks[name].shape[1])
        for name, info in spec["blocks"].items()
    ]
    z = np.column_stack(pieces)
    if spec["family"] == "quadratic":
        coordinates = (z @ spec["quadratic_basis"]) / spec["quadratic_coordinate_scale"]
        products = np.column_stack(
            [coordinates[:, i] * coordinates[:, j] for i, j in spec["quadratic_pairs"]]
        )
        products = (products - spec["quadratic_center"]) / spec["quadratic_scale"]
        z = np.column_stack([z, products / np.sqrt(products.shape[1])])
    return z


def fit_features(comments, queries, family, poles):
    """Fit all reference geometry on training full texts; no edit outcomes are used."""
    spec = {"family": family, "poles": poles}
    if family in ("distribution", "quadratic"):
        spec["landmarks"], spec["bandwidth_squared"] = representative_landmarks(
            np.vstack(comments)
        )
    blocks = raw_blocks(comments, queries, spec)
    spec["blocks"] = {
        name: {"center": block.mean(0), "scale": np.maximum(block.std(0), 1e-6)}
        for name, block in blocks.items()
    }
    if family == "quadratic":
        parent = {**spec, "family": "distribution"}
        z = transform_features(comments, queries, parent)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        basis = vt[: min(4, len(vt))].T
        scale = np.maximum(np.sqrt(np.mean((z @ basis) ** 2, axis=0)), 1e-8)
        coordinates = (z @ basis) / scale
        pairs = list(itertools.combinations_with_replacement(range(basis.shape[1]), 2))
        products = np.column_stack(
            [coordinates[:, i] * coordinates[:, j] for i, j in pairs]
        )
        spec.update(
            quadratic_basis=basis,
            quadratic_coordinate_scale=scale,
            quadratic_pairs=pairs,
            quadratic_center=products.mean(0),
            quadratic_scale=np.maximum(products.std(0), 1e-6),
        )
    return spec


def value_difference_kernel(kbb, kbe, kee):
    """Construct the exact Gram matrix for value and finite-difference observations."""
    return np.block([[kbb, kbe - kbb], [kbe.T - kbb, kee - kbe.T - kbe + kbb]])


def fit_kernel_student(x0, x1, q, y0, y1, eta, gamma, ridge):
    """Fit shared-plus-query-specific linear features with optional response matching."""
    center = np.array([y0[q == j].mean() for j in range(len(QUERIES))])
    scale = np.array([max(y0[q == j].std(), 1e-4) for j in range(len(QUERIES))])
    delta_scale = np.array(
        [
            max(np.sqrt(np.mean((y1[q == j] - y0[q == j]) ** 2)), 1e-4)
            for j in range(len(QUERIES))
        ]
    )
    task_kernel = (1 - gamma) + gamma * (q[:, None] == q[None, :])
    kbb, kbe, kee = (
        (x0 @ x0.T) * task_kernel,
        (x0 @ x1.T) * task_kernel,
        (x1 @ x1.T) * task_kernel,
    )
    target = (y0 - center[q]) / scale[q]
    if eta:
        kernel = value_difference_kernel(kbb, kbe, kee)
        weights = np.concatenate([np.ones(len(q)), np.full(len(q), np.sqrt(eta))])
        target = np.concatenate([target, (y1 - y0) / scale[q]])
    else:
        kernel, weights = kbb, np.ones(len(q))
    matrix = weights[:, None] * kernel * weights[None, :] + ridge * np.eye(len(weights))
    coefficient = weights * np.linalg.solve(matrix, weights * target)
    return {
        "x0": x0,
        "x1": x1,
        "q": q,
        "coef": coefficient,
        "eta": eta,
        "gamma": gamma,
        "ridge": ridge,
        "center": center,
        "scale": scale,
        "delta_scale": delta_scale,
    }


def predict_kernel_student(x, q, model):
    """Evaluate the frozen value/difference kernel; no LLM or target labels needed."""
    task_kernel = (1 - model["gamma"]) + model["gamma"] * (
        q[:, None] == model["q"][None, :]
    )
    base = (x @ model["x0"].T) * task_kernel
    kernel = (
        np.column_stack([base, (x @ model["x1"].T) * task_kernel - base])
        if model["eta"]
        else base
    )
    return (kernel @ model["coef"]) * model["scale"][q] + model["center"][q]


def select_specs(comments, values, q, groups, family, poles, salt):
    """Select finite student specifications using inner-author CV and E1 only."""
    folds = author_folds(groups, 3, salt)
    candidates = list(itertools.product((0.0, 1.0, 10.0), GAMMAS, RIDGES))
    losses = {spec: [] for spec in candidates}
    for fold in range(3):
        train, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        feature = fit_features([comments[j][0] for j in train], q[train], family, poles)
        x0, x1 = [
            transform_features([comments[j][c] for j in train], q[train], feature)
            for c in (0, 1)
        ]
        v0, v1 = [
            transform_features([comments[j][c] for j in val], q[val], feature)
            for c in (0, 1)
        ]
        for eta, gamma, ridge in candidates:
            model = fit_kernel_student(
                x0, x1, q[train], values[train, 0], values[train, 1], eta, gamma, ridge
            )
            p0, p1 = (
                predict_kernel_student(v0, q[val], model),
                predict_kernel_student(v1, q[val], model),
            )
            value_error = (p0 - values[val, 0]) / model["scale"][q[val]]
            difference_error = (p1 - p0 - values[val, 1] + values[val, 0]) / model[
                "delta_scale"
            ][q[val]]
            losses[eta, gamma, ridge].append(
                float(np.mean(value_error**2) + np.mean(difference_error**2))
            )
    selected = {}
    for training in ("value_only", "response_match"):
        allowed = [
            spec for spec in candidates if (spec[0] == 0) == (training == "value_only")
        ]
        spec = min(allowed, key=lambda s: (np.mean(losses[s]), s))
        selected[training] = {
            "eta": spec[0],
            "gamma": spec[1],
            "ridge": spec[2],
            "inner_loss": float(np.mean(losses[spec])),
        }
    return selected


def fit_from_spec(comments, values, q, family, poles, spec):
    """Fit one selected feature map and student on the supplied training authors."""
    feature = fit_features([row[0] for row in comments], q, family, poles)
    x0, x1 = [
        transform_features([row[c] for row in comments], q, feature) for c in (0, 1)
    ]
    model = fit_kernel_student(
        x0, x1, q, values[:, 0], values[:, 1], spec["eta"], spec["gamma"], spec["ridge"]
    )
    return feature, model


def student_metrics(frame):
    """Report author-average values and each edit's responses separately, not questionnaire accuracy."""
    metrics = []
    for (family, training, query), g in frame.groupby(["family", "training", "query"]):
        for condition in CONDITIONS:
            part = g[g.condition == condition].dropna(subset=["target"])
            means = part.groupby("author_id")[
                ["target", "prediction", "baseline"]
            ].mean()
            if len(means) < 3:
                continue
            error = np.mean((means.prediction - means.target) ** 2)
            base = np.mean((means.baseline - means.target) ** 2)
            metrics.append(
                {
                    "family": family,
                    "training": training,
                    "query": query,
                    "condition": condition,
                    "authors": len(means),
                    "r": corr(means.target, means.prediction),
                    "rmse": float(np.sqrt(error)),
                    "mse_ratio_to_baseline": float(error / base)
                    if base > 1e-14
                    else None,
                }
            )
    return pd.DataFrame(metrics)


def main():
    """Run bounded nested CV, then freeze one student without inspecting external labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-quadratic", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private students must remain in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    meta, values, comments, poles = panel_data(args.panel)
    q, groups = meta.q.to_numpy(), meta.author_id.to_numpy()
    outer = author_folds(groups, 5, "s2_outer")
    families = ["mean", "query", "distribution"] + (
        ["quadratic"] if args.include_quadratic else []
    )
    selections, predictions = [], []
    with threadpool_limits(limits=2):
        for family in families:
            for fold in range(5):
                train, val = (
                    np.flatnonzero(outer != fold),
                    np.flatnonzero(outer == fold),
                )
                material = [comments[j] for j in train]
                selected = select_specs(
                    material,
                    values[train],
                    q[train],
                    groups[train],
                    family,
                    poles,
                    f"s2_inner_{fold}",
                )
                for training, spec in selected.items():
                    selections.append(
                        {"family": family, "training": training, "fold": fold, **spec}
                    )
                    feature, model = fit_from_spec(
                        material, values[train], q[train], family, poles, spec
                    )
                    p = np.column_stack(
                        [
                            predict_kernel_student(
                                transform_features(
                                    [comments[j][c] for j in val], q[val], feature
                                ),
                                q[val],
                                model,
                            )
                            for c in range(4)
                        ]
                    )
                    for c, name in enumerate(CONDITIONS):
                        part = meta.iloc[val].drop(columns="q").copy()
                        (
                            part["family"],
                            part["training"],
                            part["fold"],
                            part["condition"],
                        ) = family, training, fold, name
                        part["target"] = (
                            values[val, c]
                            if c == 0
                            else values[val, c] - values[val, 0]
                        )
                        part["prediction"] = p[:, c] if c == 0 else p[:, c] - p[:, 0]
                        part["baseline"] = model["center"][q[val]] if c == 0 else 0.0
                        predictions.append(part)
                print(f"S2 {family} outer fold {fold + 1}/5 complete", flush=True)
            pd.concat(predictions, ignore_index=True).to_csv(
                out / "private_oof_predictions.csv", index=False
            )
        choices = []
        for family in families:
            selected = select_specs(
                comments, values, q, groups, family, poles, "s2_final_development"
            )
            choices.extend(
                {"family": family, "training": training, **spec}
                for training, spec in selected.items()
            )
        chosen = min(
            choices, key=lambda s: (s["inner_loss"], s["family"], s["training"])
        )
        feature, model = fit_from_spec(
            comments, values, q, chosen["family"], poles, chosen
        )
    artifact = out / "frozen_student.joblib"
    joblib.dump({"feature": feature, "model": model, "queries": QUERIES}, artifact)
    frame = pd.concat(predictions, ignore_index=True)
    metrics = student_metrics(frame)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selected_models.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "authors": len(set(groups)),
        "cases": len(meta),
        "seconds": time.monotonic() - start,
        "neural_forwards": 0,
        "labels_read": False,
        "panel_manifest_sha256": sha(args.panel / "manifest.json"),
        "script_sha256": sha(__file__),
        "selection": "nested author CV; normalized full-value + E1-difference MSE; E2/joint never fitted",
        "kernel": "linear shared-plus-query-specific; gamma0/.5/1; per-query training target scales",
        "feature_scaling": "training center/std per coordinate, each block divided by sqrt(width)",
        "frozen_spec": chosen,
        "student_sha256": sha(artifact),
        "metrics": metrics.astype(object)
        .where(pd.notna(metrics), None)
        .to_dict("records"),
    }
    write(out / "summary.json", summary)
    report = "# S2 fixed-encoder response students\n\n"
    report += (
        "No LLM forward at student inference. E5 per-comment vectors are required.\n"
    )
    report += (
        "Targets are teacher scores/effects, not psychological questionnaire labels.\n"
    )
    report += "All transforms and selection use training authors; only E1 response is fitted.\n\n"
    report += metrics.to_markdown(index=False, floatfmt=".4f")
    report += "\n\nScore baseline is outer-training query mean; difference baseline is zero.\n"
    report += "Raw students are unbounded regressions, not calibrated personality probabilities.\n"
    report += (
        "Frozen full-development choice (before external labels): " + str(chosen) + "\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    print(
        metrics[metrics.condition.isin(["full", "e2"])].to_string(index=False),
        flush=True,
    )
    print("Frozen student: " + str(chosen), flush=True)


if __name__ == "__main__":
    main()
