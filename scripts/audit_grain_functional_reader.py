#!/usr/bin/env python3
"""Reconstruct the frozen functional reader fits and independently score outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from grain_research.functional_reader import (
    ARMS,
    block_loss,
    build_blocks,
    stacked_system,
)
from grain_research.response_accounting import direct_difference
from scripts.run_grain_functional_reader import BASE, load_inputs
from scripts.run_grain_response_accounting import write


def panel_mean(values, ix, n):
    """Direct address-based averaging, independent of the fit aggregation helper."""
    out = np.full((n * 36, values.shape[-1]), np.nan)
    for j in np.unique(ix):
        out[j] = values[ix == j].mean(0)
    return out.reshape(n, 36, -1)


def audit(out):
    """Verify source selection, all cold predictions, least squares and all metrics."""
    summary = json.loads((out / "summary.json").read_text())
    for name, sha in summary["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed result artifact: " + name)
    contract = json.loads((out / "contract.json").read_text())
    for key, path in (
        ("plan_sha256", ROOT / "docs/GRAIN_FUNCTIONAL_READER_PLAN_V1.md"),
        ("runner_sha256", ROOT / "scripts/run_grain_functional_reader.py"),
        ("module_sha256", ROOT / "grain_research/functional_reader.py"),
        ("semantic_module_sha256", ROOT / "grain_research/semantic_link.py"),
        ("accounting_module_sha256", ROOT / "grain_research/response_accounting.py"),
    ):
        if digest(path) != contract[key]:
            raise ValueError("Changed source code/plan: " + key)
    inputs, panel = load_inputs()
    fold, ix = panel["folds"], inputs["comment_ix"]
    source, held = fold != 0, fold == 0
    metrics = pd.read_csv(out / "metrics_by_facet.csv")
    selection = json.loads((out / "source_selection.json").read_text())
    with np.load(out / "private_predictions.npz", allow_pickle=False) as saved:
        saved = {k: saved[k] for k in saved.files}
    with np.load(out / "private_normal_equations.npz", allow_pickle=False) as normal:
        normal = {k: normal[k] for k in normal.files}
    cold_error = metric_error = solver_error = target_error = 0.0
    metric_count, replayed = 0, 0
    tradeoffs, geometry = [], []
    for mode in contract["maps"]:
        chosen = min(
            (r for r in selection["tuning"] if r["map"] == mode),
            key=lambda r: r["validation_point_mse"],
        )["alpha"]
        if chosen != selection["selected_alpha"][mode]:
            raise ValueError("Alpha selection not determined by source point loss")
        models = {
            arm: joblib.load(out / f"{mode}_{arm}.joblib")
            for arm in (*ARMS, "permuted_balanced")
        }
        reference = models["point"]
        scaler = reference["target_scaler"]
        x = panel_mean(
            reference["coordinates"].design(
                inputs["g"], inputs["z"], inputs["subreddits"]
            ),
            ix,
            len(fold),
        )
        y = panel_mean((inputs["target"] - scaler.mean_) / scaler.scale_, ix, len(fold))
        if not np.array_equal(np.isfinite(y), np.isfinite(saved["teacher"])):
            raise ValueError("Teacher missingness differs")
        target_error = max(target_error, float(np.nanmax(abs(y - saved["teacher"]))))
        blocks = build_blocks(
            x[source],
            y[source],
            panel["conditions"][source],
            panel["assignments"][0, source],
            panel["strict"][source],
        )
        base_beta = models["point"]["beta"]
        point_singular = np.linalg.svd(blocks["point"].x, compute_uv=False)
        for arm, model in models.items():
            if (
                not np.array_equal(model["source_authors"], np.flatnonzero(source))
                or model["alpha"] != chosen
            ):
                raise ValueError("Wrong fitted cohort or alpha")
            name = mode + "_" + arm
            pred = x @ model["beta"]
            if not np.array_equal(np.isfinite(pred), np.isfinite(saved[name])):
                raise ValueError("Predicted missingness differs")
            cold_error = max(cold_error, float(np.nanmax(abs(pred - saved[name]))))
            replayed += int(np.isfinite(pred).all(2).sum())
            fit_blocks, fit_arm = blocks, arm
            if arm == "permuted_balanced":
                order = np.load(
                    out / f"private_{mode}_permutation.npy", allow_pickle=False
                )
                if not np.array_equal(np.sort(order), np.arange(source.sum())):
                    raise ValueError("Invalid source permutation")
                fit_blocks = build_blocks(
                    x[source],
                    y[source][order],
                    panel["conditions"][source],
                    panel["assignments"][0, source],
                    panel["strict"][source],
                )
                fit_arm = "balanced"
            sx, sy = stacked_system(fit_blocks, fit_arm)
            np.testing.assert_allclose(sx.T @ sx, normal[name + "_gram"], atol=1e-10)
            np.testing.assert_allclose(sx.T @ sy, normal[name + "_rhs"], atol=1e-10)
            if arm == "balanced":
                penalty = np.eye(sx.shape[1]) * np.sqrt(chosen)
                penalty[0, 0] = 0
                independent = np.linalg.lstsq(
                    np.vstack([sx, penalty]),
                    np.vstack([sy, np.zeros((len(penalty), sy.shape[-1]))]),
                    rcond=None,
                )[0]
                solver_error = max(
                    solver_error, float(np.max(abs(independent - model["beta"])))
                )
                singular = np.linalg.svd(sx, compute_uv=False)
                geometry.append(
                    {
                        "map": mode,
                        "point_design_rank": int(
                            np.sum(point_singular > point_singular[0] * 1e-10)
                        ),
                        "balanced_design_rank": int(
                            np.sum(singular > singular[0] * 1e-10)
                        ),
                        "columns": sx.shape[1],
                        "point_rows": len(blocks["point"].x),
                        "balanced_pseudo_rows": len(sx),
                    }
                )
            if arm in ARMS:
                for part, block in blocks.items():
                    tradeoffs.append(
                        {
                            "map": mode,
                            "arm": arm,
                            "part": part,
                            "source_risk": block_loss(block, model["beta"]),
                            "point_source_risk": block_loss(block, base_beta),
                        }
                    )
            # Metric arithmetic uses cell means, not stored accounting operators.
            good = held[:, None] & np.isfinite(y).all(2)
            yy, pp = y[good], pred[good]
            aids = np.broadcast_to(np.arange(len(y))[:, None], good.shape)[good]
            weights = np.asarray([1 / (np.sum(aids == a) * held.sum()) for a in aids])
            batches = {"comment": (yy, pp, weights)}
            by_part = {
                k: ([], [])
                for k in ("level", "total", "within", "composition", "turnover")
            }
            for u in np.flatnonzero(held):
                valid = np.isfinite(y[u]).all(1)
                by_part["level"][0].append(y[u, valid].mean(0))
                by_part["level"][1].append(pred[u, valid].mean(0))
                if not panel["strict"][u] or any(
                    not np.any(valid & (panel["assignments"][0, u] == t))
                    for t in (0, 1)
                ):
                    continue
                td, pdiff = [
                    direct_difference(
                        panel["conditions"][u], panel["assignments"][0, u], a[u]
                    )
                    for a in (y, pred)
                ]
                for j, part in enumerate(
                    ("total", "within", "composition", "turnover")
                ):
                    if part in ("within", "composition") and not panel["common"][0, u]:
                        continue
                    by_part[part][0].append(td[j])
                    by_part[part][1].append(pdiff[j])
            for part, (truth, prediction) in by_part.items():
                batches[part] = (
                    np.asarray(truth),
                    np.asarray(prediction),
                    np.ones(len(truth)) / len(truth),
                )
            for part, (yy, pp, weights) in batches.items():
                a, b = yy - weights @ yy, pp - weights @ pp
                r = (weights @ (a * b)) / np.sqrt(
                    (weights @ (a * a)) * (weights @ (b * b))
                )
                mse = weights @ ((yy - pp) ** 2)
                for q, target in enumerate(contract["target_names"]):
                    kind, facet = target.split(":")
                    row = metrics[
                        (metrics.model == mode + ":" + arm)
                        & (metrics.cohort == "held")
                        & (metrics.part == part)
                        & (metrics.function == kind)
                        & (metrics.facet == facet)
                    ]
                    if len(row) != 1:
                        raise ValueError("Missing or duplicate metric")
                    error = max(
                        abs(float(row.iloc[0].reader_teacher_r) - r[q]),
                        abs(float(row.iloc[0].mse) - mse[q]),
                    )
                    metric_error = max(metric_error, float(error))
                    metric_count += 1
    if max(cold_error, target_error, solver_error, metric_error) > 1e-8:
        raise ArithmeticError("Independent computation disagrees")
    pd.DataFrame(tradeoffs).to_csv(out / "source_functional_tradeoffs.csv", index=False)
    pd.DataFrame(geometry).to_csv(out / "source_design_geometry.csv", index=False)
    result = {
        "status": "PASS",
        "cold_comment_predictions": replayed,
        "held_metric_rows_recomputed": metric_count,
        "cold_prediction_max_error": cold_error,
        "teacher_max_error": target_error,
        "augmented_lstsq_max_error": solver_error,
        "held_metric_max_error": metric_error,
        "scope": "independent numerical method, not independent investigator or psychometric validation",
        "summary_sha256": digest(out / "summary.json"),
        "audit_script_sha256": digest(Path(__file__)),
        "audit_artifacts": {
            n: digest(out / n)
            for n in ("source_functional_tradeoffs.csv", "source_design_geometry.csv")
        },
    }
    write(out / "audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "functional_reader_r1")
    with threadpool_limits(limits=2):
        audit(parser.parse_args().output)
