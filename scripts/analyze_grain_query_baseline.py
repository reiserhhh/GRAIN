#!/usr/bin/env python3
"""Evaluate source-directed baseline reading without fitting on held endpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.comment_carrier import score
from grain_research.vector_mechanisms.complete_pair import error_account
from grain_research.vector_mechanisms.joint_readout import centered_logits
from scripts.analyze_grain_conditional_energy import author_draws, ci
from scripts.analyze_grain_conditional_energy import corr as _corr
from scripts.run_grain_joint_carrier_readout import OLD, check
from scripts.run_grain_query_baseline import verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def corr(x, y):
    """Return NaN for raw constants before floating centering creates roundoff."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite([x, y]).all():
        raise ValueError("Matched finite scalar arrays required")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return _corr(x, y)


def field_geometry(jacobians):
    """Describe empirical source-gradient geometry, not calibrated population axes."""
    a = np.asarray(jacobians, float)
    if a.ndim < 2 or not np.isfinite(a).all():
        raise ValueError("Finite source derivatives required")
    a = a.reshape(len(a), -1)
    gram = a @ a.T
    norms = np.sqrt(np.diag(gram))
    if np.any(norms == 0):
        raise ValueError("Nonzero source derivative fields required")
    cosine = gram / np.outer(norms, norms)
    pairs = cosine[np.triu_indices(len(a), 1)]
    e = np.maximum(np.linalg.eigvalsh(gram)[::-1], 0)
    p = e / e.sum()
    nz = p > 0
    return {
        "source_fields": len(a),
        "coordinates": a.shape[1],
        "cosine_mean": pairs.mean(),
        "cosine_min": pairs.min(),
        "cosine_max": pairs.max(),
        "common_field_energy_ratio": float(
            np.mean(a, axis=0) @ np.mean(a, axis=0) / (np.trace(gram) / len(a))
        ),
        "top1_energy_ratio": p[0],
        "effective_rank": float(np.exp(-np.sum(p[nz] * np.log(p[nz])))),
        "rank95": int(np.searchsorted(np.cumsum(p), 0.95) + 1),
    }


def analyze(out, construction, name):
    """Read cached20 evaluation endpoints only after construction freeze is verified."""
    verified(out)
    built = out / construction
    freeze = read(built / "prediction_freeze.json")
    check(out / "source_freeze.json", freeze["source_freeze_sha256"])
    for p, d in freeze["files"].items():
        check(built / p, d)
    dest = built / name
    dest.mkdir(exist_ok=False)
    write(
        dest / "manifest.json",
        {
            "analyzer_sha256": sha(Path(__file__)),
            "prediction_freeze_sha256": sha(built / "prediction_freeze.json"),
            "human_labels_read": False,
            "new_neural_calls": 0,
            "bootstrap": "2000 author draws, fixed predictions, seed2501",
        },
    )
    authors = [
        r for r in read(out / "private_design.json") if r["role"] == "evaluation"
    ]
    with np.load(built / "private_predictions.npz") as z:
        pred = dict(z)
    actual, values = [], None
    for r in authors:
        packet = read(OLD / "private_packets" / f"author_{r['index']:03d}.json")
        if packet["author_id"] != r["author_id"] or packet["query"] != "neuroticism":
            raise ValueError("Held endpoint identity changed")
        if values is not None and values != packet["alphabet"]["values"]:
            raise ValueError("Alphabet changed across authors")
        values = packet["alphabet"]["values"]
        actual.append(centered_logits(packet["scores"]["zero"]["candidate_logits"]))
    actual = np.array(actual)
    y = np.array([score(a, values) for a in actual])
    assemblies = {
        "source_affine": pred["logits"],
        "source_kernel": pred["kernel_logits"],
        "source_mean": pred["mean_logits"],
        "mean_action_only": centered_logits(pred["mean_logits"] + pred["mean_action"]),
        "retained_action_only": centered_logits(
            pred["mean_logits"] + pred["retained_action"]
        ),
    }
    draws = author_draws([r["author_id"] for r in authors])
    metrics = []
    score_arrays = {}
    native_variation = np.mean((actual - actual.mean(0)) ** 2)
    for arm, z in assemblies.items():
        p = np.array([score(a, values) for a in z])
        score_arrays[arm] = p
        lo, hi, _ = ci([corr(y[ix], p[ix]) for ix in draws])
        predvar = z - z.mean(0)
        actvar = actual - actual.mean(0)
        metrics.append(
            {
                "arm": arm,
                "authors": 20,
                "score_r": corr(y, p),
                "r_low": lo,
                "r_high": hi,
                "score_rmse": np.sqrt(np.mean((y - p) ** 2)),
                "score_mae": np.mean(abs(y - p)),
                "actual_score_sd": y.std(),
                "predicted_score_sd": p.std(),
                "score_mse_over_global_variance": np.mean((y - p) ** 2) / y.var(),
                "logit_rmse": np.sqrt(np.mean((actual - z) ** 2)),
                "author_centered_logit_mse_over_variation": np.mean(
                    (actvar - predvar) ** 2
                )
                / native_variation,
                "author_centered_logit_r": corr(actvar.ravel(), predvar.ravel()),
            }
        )
    comparisons = []
    for other in ("source_kernel", "source_mean"):
        delta = (y - score_arrays["source_affine"]) ** 2 - (
            y - score_arrays[other]
        ) ** 2
        lo, hi, _ = ci([delta[ix].mean() for ix in draws])
        comparisons.append(
            {
                "comparison": "source_affine_minus_" + other,
                "score_mse_delta": delta.mean(),
                "low": lo,
                "high": hi,
                "authors": 20,
            }
        )
    identity = error_account(
        {
            "constant_error": pred["mean_logits"] - actual,
            "mean_action": centered_logits(pred["mean_action"]),
            "retained_action": centered_logits(pred["retained_action"]),
        }
    )
    direct = np.mean((pred["logits"] - actual) ** 2, 1)
    assembly_error = float(np.max(abs(identity["full_mse"] - direct)))
    if assembly_error > 1e-12:
        raise ValueError(
            "Directional error accounting does not reconstruct predictions"
        )
    pd.DataFrame(metrics).to_csv(dest / "baseline_comparison.csv", index=False)
    pd.DataFrame(comparisons).to_csv(dest / "paired_score_mse.csv", index=False)
    pd.DataFrame([{k: float(np.mean(v)) for k, v in identity.items()}]).to_csv(
        dest / "directional_error_account.csv", index=False
    )
    pd.DataFrame(
        {"author_id": [r["author_id"] for r in authors], "actual": y, **score_arrays}
    ).to_csv(dest / "private_score_predictions.csv", index=False)
    fd = []
    replay = []
    gradients = {k: [] for k in ("M", "R", "M_R", "score_M_R")}
    for slot in range(4):
        meta = read(out / "private_packets" / f"source_{slot:02d}.json")
        with np.load(out / "private_packets" / f"source_{slot:02d}.npz") as z:
            for eps, estimate in zip((0.001, 0.002), z["fd"], strict=True):
                true = z["analytic"]
                fd.append(
                    {
                        "region": meta["region"],
                        "epsilon": eps,
                        "vector_relative_error": np.linalg.norm(estimate - true)
                        / np.linalg.norm(true),
                        "absolute_rms_error": np.sqrt(np.mean((estimate - true) ** 2)),
                        "derivative_rms": np.sqrt(np.mean(true**2)),
                        "same_sign": int(np.sum(estimate * true > 0)),
                        "candidates": len(true),
                    }
                )
            jm, jr = z["jm"], z["jr"]
            g = np.concatenate([jm.reshape(10, -1), jr.reshape(10, -1)], 1)
            gradients["M"].append(jm)
            gradients["R"].append(jr)
            gradients["M_R"].append(g)
            logits = z["logits"].astype(float)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            values_array = np.asarray(values)
            v = p * (values_array - p @ values_array)
            gradients["score_M_R"].append(v @ g)
        replay.append(meta["logit_replay_max"])
    fdd = pd.DataFrame(fd)
    fdd.groupby(["region", "epsilon"]).agg(
        checks=("same_sign", "size"),
        relative_error_mean=("vector_relative_error", "mean"),
        relative_error_max=("vector_relative_error", "max"),
        absolute_rms_error_mean=("absolute_rms_error", "mean"),
        derivative_rms_mean=("derivative_rms", "mean"),
        same_sign=("same_sign", "sum"),
        candidates=("candidates", "sum"),
    ).reset_index().to_csv(dest / "derivative_checks.csv", index=False)
    pd.DataFrame(
        [{"field": key, **field_geometry(g)} for key, g in gradients.items()]
    ).to_csv(dest / "source_gradient_geometry.csv", index=False)
    source_scores = np.array(
        [[score(z, values) for z in a] for a in pred["per_source"]]
    )
    disagreement = {
        "per_source_score_sd_mean": float(np.mean(source_scores.std(1))),
        "per_source_score_sd_max": float(np.max(source_scores.std(1))),
        "aggregate_prediction_score_sd": float(score_arrays["source_affine"].std()),
        "is_calibrated_uncertainty": False,
    }
    files = [
        "baseline_comparison.csv",
        "paired_score_mse.csv",
        "directional_error_account.csv",
        "private_score_predictions.csv",
        "derivative_checks.csv",
        "source_gradient_geometry.csv",
    ]
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "source_authors": 4,
            "evaluation_authors": 20,
            "native_logit_replay_max": max(replay),
            "error_assembly_max": assembly_error,
            "error_identity_max": float(np.max(abs(identity["identity_error"]))),
            "source_disagreement": disagreement,
            "derivative_relative_error_max": float(fdd.vector_relative_error.max()),
            "derivative_signs": int(fdd.same_sign.sum()),
            "derivative_candidate_checks": int(fdd.candidates.sum()),
            "human_labels_read": False,
            "new_neural_calls": 0,
            "files": {p: sha(dest / p) for p in files},
        },
    )
    print(
        {"status": "ANALYSIS_COMPLETE", "comparisons": len(metrics), "held_authors": 20}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="logic_query_baseline_v1")
    p.add_argument("--construction", default="numerical")
    p.add_argument("--name", default="analysis")
    a = p.parse_args()
    analyze(BASE / a.source, a.construction, a.name)
