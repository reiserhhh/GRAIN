#!/usr/bin/env python3
"""S12: frozen-fold teachers for source body/query/KV representations, then Y linkage."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.analyze_grain_functional_basis import author_folds
from scripts.analyze_grain_teacher_student_bridge import (
    covariance_split,
    paired_intervals,
)
from scripts.run_grain_contrast_student import METHODS, TARGETS, select, specification
from scripts.run_grain_query_representation import (
    REPRESENTATIONS,
    validate_packet,
    verify,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write
from scripts.run_grain_source_remainder import CONNECTION, CURVE
from scripts.run_grain_supervised_student import fit_student


def assemble(out, material):
    """Join complete three-view packets by exact author, returning no labels."""
    names, arrays, diagnostics, hashes = [], {k: [] for k in REPRESENTATIONS}, [], {}
    paths = sorted((out / "private_packets").glob("author_*.json"))
    if len(paths) != len(material):
        raise ValueError("Complete cohort required before model fitting")
    for i, author in enumerate(material):
        path = out / "private_packets" / f"author_{i:03d}.json"
        packet = validate_packet(path, author)
        if {v["view"] for v in packet["views"]} != {0, 1, 2}:
            raise ValueError("Duplicate view or missing view")
        names.append(author["author_id"])
        hashes[path.name] = sha(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            for name in REPRESENTATIONS:
                arrays[name].append(saved[name].mean(0))
        diagnostics.extend(packet["views"])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate author")
    return (
        np.asarray(names),
        {k: np.asarray(v) for k, v in arrays.items()},
        diagnostics,
        hashes,
    )


def target_quality(target, prediction, null):
    """Measure finite readout fidelity against an out-of-fold train-mean control."""
    target, prediction, null = [
        np.asarray(x, float) for x in (target, prediction, null)
    ]
    if (
        target.ndim != 1
        or target.shape != prediction.shape
        or target.shape != null.shape
    ):
        raise ValueError("Require paired one-dimensional target/readout/null arrays")
    if not np.isfinite([target, prediction, null]).all():
        raise ValueError("Finite quality arrays required")
    mse, null_mse = np.mean((target - prediction) ** 2), np.mean((target - null) ** 2)
    fidelity = scalar_metrics(target, prediction)
    return {
        "teacher_r": fidelity["pearson_r"],
        "teacher_r_lower": fidelity["r_lower"],
        "teacher_r_upper": fidelity["r_upper"],
        "teacher_mse": float(mse),
        "teacher_mse_over_null": float(mse / null_mse) if null_mse else None,
    }


def fit_oof(names, features, targets, out):
    """Fit all declared arms without outcome access and persist affine replay maps."""
    rows, selections = [], []
    folds = author_folds(names, 5, "s4_outer")
    old = pd.read_csv(
        CURVE / "private_oof_before_outcomes.csv", dtype={"author_id": str, "size": str}
    )
    old = old[
        (old["query"] == "neuroticism")
        & (old.family == "mean")
        & (old["size"] == "all")
    ].set_index("author_id")
    if not np.array_equal(folds, old.loc[names, "fold"].to_numpy()):
        raise ValueError("S4 fold membership changed")
    start = time.monotonic()
    with threadpool_limits(limits=2):
        for fold in range(5):
            train, test = folds != fold, folds == fold
            for rep, x in features.items():
                for method in METHODS:
                    family, rank = specification(method)
                    for target in TARGETS:
                        y = targets[target].to_numpy()
                        alpha = select(
                            x[train],
                            y[train],
                            names[train],
                            method,
                            f"s11_inner_{fold}",
                        )
                        coef, intercept = fit_student(
                            x[train], y[train], family, rank, alpha
                        )
                        prediction = x[test] @ coef + intercept
                        path = (
                            out
                            / "private_models"
                            / f"{rep}_{method}_{target}_fold{fold}.npz"
                        )
                        np.savez_compressed(path, coef=coef, intercept=intercept)
                        with np.load(path, allow_pickle=False) as saved:
                            replay = x[test] @ saved["coef"] + saved["intercept"]
                        if not np.allclose(replay, prediction, atol=1e-12, rtol=1e-12):
                            raise ValueError("Affine model replay changed")
                        selections.append(
                            {
                                "fold": fold,
                                "representation": rep,
                                "method": method,
                                "target": target,
                                "alpha": alpha,
                            }
                        )
                        rows.extend(
                            {
                                "author_id": uid,
                                "fold": fold,
                                "representation": rep,
                                "method": method,
                                "target": target,
                                "teacher": float(t),
                                "prediction": float(p),
                                "null_prediction": float(y[train].mean()),
                            }
                            for uid, t, p in zip(
                                names[test], y[test], prediction, strict=True
                            )
                        )
                        if time.monotonic() - start > 600:
                            raise TimeoutError("S12 finite fit budget reached")
            print({"status": "FIT", "complete_folds": fold + 1}, flush=True)
    frame = pd.DataFrame(rows)
    if (
        len(frame) != len(names) * 18
        or frame.duplicated(["author_id", "representation", "method", "target"]).any()
    ):
        raise ValueError("Incomplete or duplicated OOF predictions")
    pd.DataFrame(selections).to_csv(out / "selected_models.csv", index=False)
    return frame


def report(frame, targets, out):
    """Connect only frozen predictions to exposed source outcomes; report every arm."""
    labels = pd.read_csv(
        CONNECTION / "private_predictions.csv",
        dtype={"author_id": str},
        usecols=["author_id", "query", "y_true"],
    )
    labels = labels[labels["query"] == "neuroticism"][["author_id", "y_true"]]
    frame = frame.merge(labels, on="author_id", validate="many_to_one")
    if not np.isfinite(frame.y_true).all():
        raise ValueError("Missing exposed-source label")
    s11 = BASE / "contrast_student_s11_r1"
    if (
        sha(s11 / "private_oof_before_outcomes.csv")
        != read(s11 / "summary.json")["oof_sha256"]
    ):
        raise ValueError("Frozen S11 comparison changed")
    previous = pd.read_csv(
        s11 / "private_oof_before_outcomes.csv", dtype={"author_id": str}
    )
    metrics, intervals, moments = [], [], []
    for (rep, method, target), g in frame.groupby(
        ["representation", "method", "target"]
    ):
        g = g.set_index("author_id")
        scope = {"representation": rep, "method": method, "target": target}
        ext = scalar_metrics(g.y_true, g.prediction)
        metrics.append(
            {
                **scope,
                "authors": len(g),
                **target_quality(g.teacher, g.prediction, g.null_prediction),
                "external_r": ext["pearson_r"],
                "external_lower": ext["r_lower"],
                "external_upper": ext["r_upper"],
                "external_p": ext["pearson_p"],
            }
        )
        moments.append({**scope, **covariance_split(g.y_true, g.teacher, g.prediction)})
        body = (
            frame[
                (frame.representation == "body_mean")
                & (frame.method == method)
                & (frame.target == target)
            ]
            .set_index("author_id")
            .loc[g.index, "prediction"]
        )
        e5 = (
            previous[(previous.method == method) & (previous.target == target)]
            .set_index("author_id")
            .loc[g.index, "prediction"]
        )
        for control, values in (
            ("body_mean_matched", body),
            ("s11_e5_matched", e5),
            ("s9_new_teacher", targets.loc[g.index, "new_teacher"]),
            ("s4_student", targets.loc[g.index, "student"]),
        ):
            intervals.append(
                {
                    **scope,
                    "control": control,
                    **paired_intervals(g.y_true, g.prediction, values, False),
                }
            )
    for name, records in (
        ("metrics", metrics),
        ("paired_intervals", intervals),
        ("covariance_split", moments),
    ):
        pd.DataFrame(records).to_csv(out / f"{name}.csv", index=False)
    return metrics, intervals


def main():
    """Run the finite paired representation experiment after complete source collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private analysis artifacts belong in ignored results")
    manifest, material = verify(args.collection, "mps")
    state = read(args.collection / "run_status.json")
    if state["status"] != "COMPLETE" or state["calls"] != 639:
        raise ValueError("Full frozen collection required")
    names, features, diagnostics, hashes = assemble(args.collection, material)
    source = Path(manifest["source"])
    targets = pd.read_csv(
        source / "private_components_before_outcomes.csv", dtype={"author_id": str}
    )
    targets = (
        targets[targets["query"] == "neuroticism"].set_index("author_id").loc[names]
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_models").mkdir()
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_S12_REPRESENTATION_STUDENT_V1",
            "script_sha256": sha(__file__),
            "collection_manifest_sha256": sha(args.collection / "manifest.json"),
            "source_components_sha256": sha(
                source / "private_components_before_outcomes.csv"
            ),
            "private_packet_hashes": hashes,
            "methods": METHODS,
            "targets": TARGETS,
            "representations": REPRESENTATIONS,
            "authors": len(names),
            "dimensions": 1536,
            "selection": "teacher-only inner3 MSE; all18 fixed arms reported",
            "scope": "same exposed D1 source; no questionnaire fitting; pointwise inference",
        },
    )
    start = time.monotonic()
    frame = fit_oof(names, features, targets, out)
    oof = out / "private_oof_before_outcomes.csv"
    frame.to_csv(oof, index=False)
    write(out / "oof_frozen.json", {"sha256": sha(oof), "outcomes_used": False})
    metrics, intervals = report(frame, targets, out)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": sha(out / "manifest.json"),
            "oof_sha256": sha(oof),
            "wall_seconds": time.monotonic() - start,
            "new_forwards": 0,
            "questionnaire_fit": False,
            "collector_calls": state["calls"],
            "collector_seconds": state["seconds"],
            "max_origin_replay_error": max(d["s9_replay_error"] for d in diagnostics),
            "max_head_reconstruction_error": max(
                d["head_reconstruction_error"] for d in diagnostics
            ),
            "metrics": metrics,
            "paired_intervals": intervals,
            "interval_scope": "paired authors conditional on frozen OOF fits, not repeated training",
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
