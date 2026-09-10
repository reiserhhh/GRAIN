#!/usr/bin/env python3
"""Fit one reference-only logit potential and evaluate unchanged cached targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.comment_carrier import partition, reference_state
from grain_research.vector_mechanisms.level_response import correlation
from grain_research.vector_mechanisms.query_functional import response_quality
from grain_research.vector_mechanisms.retained_logit import (
    fit_curvature,
    logit,
    path_account,
    quadratic_delta,
    radial_parts,
    sigmoid,
)
from scripts.diagnose_grain_retained_logit import ACTION, PLAN, reference_paths
from scripts.run_grain_action_geometry import PARENT, verify
from scripts.run_grain_joint_carrier_readout import (
    OLD,
    STATES,
    check,
    donors_and_design,
)
from scripts.run_grain_real_author_path import BASE, read, sha, write

AMENDMENT = ROOT / "docs/GRAIN_RETAINED_LOGIT_SURROGATE_AMENDMENT_V1.md"


def freeze(out):
    """Bind source functions and predictions before evaluation endpoint reads."""
    verify(ACTION)
    donors, design = donors_and_design()
    inputs = [
        ACTION / "manifest.json",
        ACTION / "prediction_freeze.json",
        PARENT / "carrier_nodes_freeze.json",
    ]
    code = [
        Path(__file__),
        PLAN,
        AMENDMENT,
        ROOT / "scripts/diagnose_grain_retained_logit.py",
        ROOT / "grain_research/vector_mechanisms/retained_logit.py",
    ]
    out.mkdir(exist_ok=False)
    write(
        out / "manifest.json",
        {
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
            "code": {str(p.relative_to(ROOT)): sha(p) for p in code},
            "reference_authors": 4,
            "evaluation_authors": 20,
            "fits": 2,
            "penalty": 0.01,
            "human_labels_read": False,
            "new_neural_calls": 0,
        },
    )
    with np.load(ACTION / "private_displacements.npz") as z:
        displacements = z["displacements"].reshape(24, 136, 1536)
    frames = []
    for j, row in enumerate(design):
        v = row["views"][0]
        g, _, _, rest = partition(v["groups"], v["p_mask"])
        with np.load(STATES / f"author_{row['index']:03d}.npz") as z:
            h = z["view_0"]
            ref, interpolation = reference_state(h, g, v["p_mask"], donors)
            if (
                interpolation
                or len(rest) != 136
                or not np.array_equal(
                    h[rest].astype(float) - ref[rest].astype(float), displacements[j]
                )
            ):
                raise ValueError("Reference geometry or inherited displacement changed")
            frame = ref[rest].astype(float)
        norms = np.linalg.norm(frame, axis=1)
        if not (norms > 0).all():
            raise ValueError("Nonzero reference role vectors required")
        frames.append(frame / norms[:, None])
    frames = np.array(frames)
    if not np.array_equal(frames, np.broadcast_to(frames[0], frames.shape)):
        raise ValueError("Role reference frames vary across layouts")
    paths = reference_paths()
    rows, model, diagnostics = [], {"frame": frames[0]}, []
    for body in (0, 1):
        selected = [p for p in paths if p["body_case"] == body]
        gradients = [
            path_account(
                p["fractions"], p["probabilities"], p["gradients"], p["displacement"]
            )[1]
            for p in selected
        ]
        d = np.array([p["displacement"] for p in selected])
        mean, coef, support = fit_curvature(
            d,
            np.broadcast_to(frames[0], d.shape),
            [p["fractions"] for p in selected],
            gradients,
        )
        origin = float(np.mean([logit(p["probabilities"][0]) for p in selected]))
        model.update(
            {
                f"body{body}_origin_gradient": mean,
                f"body{body}_coefficients": coef,
                f"body{body}_support": support,
                f"body{body}_origin_logit": np.array(origin),
            }
        )
        error, energy = [], []
        for p, v in zip(selected, gradients, strict=True):
            x = p["fractions"][1:, None, None] * p["displacement"]
            par, perp = radial_parts(x, frames[0])
            pred = coef[0, :, None] * par + coef[1, :, None] * perp
            target = v[1:] - v[0]
            error.append(float(np.mean(np.sum((pred - target) ** 2, axis=(1, 2)))))
            energy.append(float(np.mean(np.sum(target**2, axis=(1, 2)))))
        diagnostics.append(
            {
                "body_case": body,
                "reference_paths": 4,
                "curvature_coefficients": coef.size,
                "zero_support_components": int((support == 0).sum()),
                "gradient_increment_error_over_zero": float(
                    np.mean(error) / np.mean(energy)
                ),
                "origin_logit_mean": origin,
            }
        )
        for j, row in enumerate(design):
            linear, quadratic = quadratic_delta(displacements[j], frames[j], mean, coef)
            rows.append(
                {
                    "index": row["index"],
                    "role": row["role"],
                    "body_case": body,
                    "origin_linear": linear,
                    "radial_quadratic": quadratic,
                    "source_origin_logit": origin,
                }
            )
    pd.DataFrame(rows).to_csv(out / "private_logit_predictions.csv", index=False)
    np.savez_compressed(out / "private_models.npz", **model)
    pd.DataFrame(diagnostics).to_csv(out / "reference_fit_summary.csv", index=False)
    write(
        out / "prediction_freeze.json",
        {
            "manifest_sha256": sha(out / "manifest.json"),
            "files": {
                name: sha(out / name)
                for name in (
                    "private_models.npz",
                    "private_logit_predictions.csv",
                    "reference_fit_summary.csv",
                )
            },
            "evaluation_endpoints_read": False,
            "human_labels_read": False,
            "new_neural_calls": 0,
        },
    )
    print(
        {
            "status": "PREDICTIONS_FROZEN",
            "rows": len(rows),
            "fits": 2,
            "new_neural_calls": 0,
            "evaluation_endpoints_read": False,
        }
    )


def verified_predictions(out):
    """Refuse changed programs, provenance or predictions at evaluation time."""
    verify(ACTION)
    manifest, frozen = read(out / "manifest.json"), read(out / "prediction_freeze.json")
    check(out / "manifest.json", frozen["manifest_sha256"])
    for category in ("inputs", "code"):
        for name, digest in manifest[category].items():
            check(ROOT / name, digest)
    for name, digest in frozen["files"].items():
        check(out / name, digest)
    return pd.read_csv(out / "private_logit_predictions.csv")


def quality(y, prediction):
    """Add Pearson to original-unit action errors, not questionnaire accuracy."""
    return {
        **response_quality(np.asarray(y), np.asarray(prediction)),
        "r": correlation(y, prediction),
    }


def evaluate(out):
    """After freezing, score old model targets and explicit origin dependencies."""
    predicted = verified_predictions(out)
    _, design = donors_and_design()
    rows, endpoint_hashes = [], {}
    for row in design:
        path = OLD / "private_packets" / f"author_{row['index']:03d}.json"
        endpoint_hashes[str(path.relative_to(ROOT))] = sha(path)
        score = read(path)["scores"]
        for body, keys in enumerate((("b00", "b01"), ("b10", "zero"))):
            p0, p1 = [score[key]["conditional_score"] for key in keys]
            selected = predicted[
                (predicted["index"] == row["index"]) & (predicted.body_case == body)
            ]
            if len(selected) != 1 or selected.iloc[0].role != row["role"]:
                raise ValueError("Exact author/body/role prediction join required")
            origin, actual_delta = float(logit(p0)), float(logit(p1) - logit(p0))
            r = selected.iloc[0]
            for arm in ("origin_linear", "radial_quadratic"):
                delta = float(r[arm])
                rows.append(
                    {
                        "index": row["index"],
                        "role": row["role"],
                        "body_case": body,
                        "arm": arm,
                        "actual_logit_delta": actual_delta,
                        "predicted_logit_delta": delta,
                        "actual_probability_delta": p1 - p0,
                        "source_origin_prediction": float(
                            sigmoid(r.source_origin_logit + delta)
                            - sigmoid(r.source_origin_logit)
                        ),
                        "oracle_origin_prediction": float(
                            sigmoid(origin + delta) - sigmoid(origin)
                        ),
                    }
                )
    data = pd.DataFrame(rows)
    data.to_csv(out / "private_endpoint_scores.csv", index=False)
    logit_rows = []
    for (role, body, arm), g in data.groupby(["role", "body_case", "arm"]):
        logit_rows.append(
            {
                "role": role,
                "body_case": body,
                "arm": arm,
                **quality(g.actual_logit_delta, g.predicted_logit_delta),
            }
        )
    pd.DataFrame(logit_rows).to_csv(out / "logit_comparison.csv", index=False)
    allocated = (
        data.groupby(["index", "role", "arm"])[
            [
                "actual_probability_delta",
                "source_origin_prediction",
                "oracle_origin_prediction",
            ]
        ]
        .mean()
        .reset_index()
    )
    retained = []
    for row in allocated.itertuples():
        for origin in ("source_origin", "oracle_origin"):
            retained.append(
                {
                    "index": row.index,
                    "role": row.role,
                    "arm": row.arm + "__" + origin,
                    "actual": row.actual_probability_delta,
                    "predicted": getattr(row, origin + "_prediction"),
                }
            )
    check(
        ACTION / "private_carrier_predictions.csv",
        read(ACTION / "prediction_freeze.json")["files"][
            "private_carrier_predictions.csv"
        ],
    )
    old = pd.read_csv(
        ACTION / "private_carrier_predictions.csv", dtype={"author_id": str}
    )
    for row in design:
        actual = allocated[
            allocated["index"] == row["index"]
        ].actual_probability_delta.to_numpy()
        for arm in ("frozen_full", "support_calibrated", "reference_mean"):
            selected = old[(old.author_id == str(row["author_id"])) & (old.arm == arm)]
            if (
                len(selected) != 1
                or selected.iloc[0].role != row["role"]
                or not np.allclose(
                    actual, selected.iloc[0].actual_retained, atol=1e-12, rtol=0
                )
            ):
                raise ValueError("Old and new retained targets or membership differ")
            retained.append(
                {
                    "index": row["index"],
                    "role": row["role"],
                    "arm": arm,
                    "actual": float(actual[0]),
                    "predicted": float(selected.iloc[0].predicted_retained),
                }
            )
    all_retained = pd.DataFrame(retained)
    all_retained.to_csv(out / "private_retained_scores.csv", index=False)
    scores = []
    for (role, arm), g in all_retained.groupby(["role", "arm"]):
        scores.append({"role": role, "arm": arm, **quality(g.actual, g.predicted)})
    scores = pd.DataFrame(scores)
    scores.to_csv(out / "retained_comparison.csv", index=False)
    ev = all_retained[all_retained.role == "evaluation"].pivot(
        index="index", columns="arm", values="predicted"
    )
    truth = (
        all_retained[all_retained.role == "evaluation"]
        .groupby("index")
        .actual.first()
        .loc[ev.index]
        .to_numpy()
    )
    boot = np.random.default_rng(2401).integers(0, len(truth), size=(2000, len(truth)))
    contrasts = []
    for origin in ("source_origin", "oracle_origin"):
        arm = "radial_quadratic__" + origin
        for baseline in (
            "origin_linear__" + origin,
            "frozen_full",
            "support_calibrated",
        ):
            improvement = (ev[baseline].to_numpy() - truth) ** 2 - (
                ev[arm].to_numpy() - truth
            ) ** 2
            low, high = np.quantile(improvement[boot].mean(1), [0.025, 0.975])
            contrasts.append(
                {
                    "arm": arm,
                    "baseline": baseline,
                    "authors": 20,
                    "mse_improvement": float(improvement.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_repetitions": 2000,
                }
            )
    pd.DataFrame(contrasts).to_csv(out / "paired_mse_comparison.csv", index=False)
    names = (
        "private_endpoint_scores.csv",
        "logit_comparison.csv",
        "private_retained_scores.csv",
        "retained_comparison.csv",
        "paired_mse_comparison.csv",
    )
    write(
        out / "evaluation_summary.json",
        {
            "status": "COMPLETE",
            "fits": 2,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "prediction_freeze_sha256": sha(out / "prediction_freeze.json"),
            "endpoint_hashes": endpoint_hashes,
            "files": {name: sha(out / name) for name in names},
            "oracle_origins_not_runtime": True,
        },
    )
    print(scores[scores.role == "evaluation"].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "evaluate"])
    parser.add_argument("--name", default="logic_retained_logit_surrogate_v1")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Simple output name required")
    (freeze if args.mode == "freeze" else evaluate)(BASE / args.name)
