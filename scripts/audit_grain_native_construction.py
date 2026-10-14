#!/usr/bin/env python3
"""Replay native cache arithmetic and frozen same-budget score/metric identities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.analyze_grain_author_discovery import read_native
from scripts.prepare_grain_predecessor_reference import BASE, digest, write


def correlation(x, y):
    """Vectorized Pearson over the author axis, keeping every trait."""
    x, y = x - x.mean(axis=-2, keepdims=True), y - y.mean(axis=-2, keepdims=True)
    return np.sum(x * y, axis=-2) / np.sqrt(
        np.sum(x * x, axis=-2) * np.sum(y * y, axis=-2)
    )


def audit(out):
    """Audit current inputs, logits, source-only transforms, raw maps and intervals."""
    summary = json.loads((out / "summary.json").read_text())
    contract = json.loads((out / "contract.json").read_text())
    for name, sha in summary["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed study output")
    for name, sha in contract["inputs"].items():
        if digest(Path(name)) != sha:
            raise ValueError("Changed study source")
    for name, sha in contract["code"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Changed producing code/protocol")
    with np.load(out / "private_material.npz", allow_pickle=False) as z:
        material = {k: z[k] for k in z.files}
    with np.load(out / "private_predictions.npz", allow_pickle=False) as z:
        predictions = {k: z[k] for k in summary["arms"]}
        np.testing.assert_array_equal(z["held_indices"], material["held"])
    ids = json.loads((out / "private_authors.json").read_text())["ids"]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated author")
    folds, y, held, source = (material[k] for k in ("folds", "y", "held", "source"))
    np.testing.assert_array_equal(held, np.where(folds == 0)[0])
    np.testing.assert_array_equal(source, np.where(folds != 0)[0])
    np.testing.assert_array_equal(material["train"], np.where(folds >= 2)[0])
    np.testing.assert_array_equal(material["valid"], np.where(folds == 1)[0])
    if set(source) & set(held):
        raise ValueError("Author overlap")
    records, _, _, _ = read_native(BASE / "d1_direct_snapshot_47")
    replay = np.zeros_like(material["views"])
    count = 0
    for u, uid in enumerate(ids):
        if records[uid]["source_group"] != "big5_only":
            raise ValueError("Wrong author role")
        for j, trait in enumerate(TRAITS):
            rows = [
                r
                for r in records[uid]["readouts"]
                if r["arm"] == "raw" and r["task"] == trait.lower()
            ]
            if sorted(r["view"] for r in rows) != [0, 1, 2]:
                raise ValueError("Incomplete raw views")
            for row in rows:
                logits = np.array(row["candidate_logits"])
                probability = np.exp(logits - max(logits))
                probability /= probability.sum()
                replay[u, row["view"], j] = sum(
                    k * probability[k] / 9 for k in range(10)
                )
                count += 1
    native_error = float(abs(replay - material["views"]).max())
    np.testing.assert_allclose(replay.mean(1), material["native"], atol=1e-12)
    np.testing.assert_allclose(
        replay[held].mean(1), predictions["native_raw"], atol=1e-12
    )
    max_score_error = 0.0
    raw_parts = {"native_raw": {"native": material["native"][held]}}
    for arm in summary["arms"][1:]:
        obj = joblib.load(out / f"private_model_{arm}.joblib")
        feature, head = obj["features"], obj["head"]
        np.testing.assert_array_equal(feature.fit_indices_, source)
        parts = []
        for key in feature.keys_:
            part = material[key]
            if key == "semantic":
                np.testing.assert_allclose(
                    feature.pca_.mean_, material[key][source].mean(0), atol=1e-13
                )
                part = (part - feature.pca_.mean_) @ feature.pca_.components_.T
            parts.append(part)
        unscaled = np.column_stack(parts)
        np.testing.assert_allclose(
            feature.scaler_.mean_, unscaled[source].mean(0), atol=1e-13
        )
        np.testing.assert_allclose(
            feature.scaler_.var_, unscaled[source].var(0), atol=1e-13
        )
        x = (unscaled - feature.scaler_.mean_) / feature.scaler_.scale_
        direct = x[held] @ head.coef_.T + head.intercept_
        np.testing.assert_allclose(direct, predictions[arm], atol=1e-11)
        max_score_error = max(
            max_score_error, float(abs(direct - predictions[arm]).max())
        )
        # Cold refit checks exact source use; it is verification, not a new candidate.
        cold = Ridge(alpha=obj["chosen"]["alpha"], solver="lsqr").fit(
            x[source], y[source]
        )
        np.testing.assert_allclose(cold.predict(x[held]), predictions[arm], atol=1e-10)
        with np.load(out / f"private_affine_{arm}.npz", allow_pickle=False) as z:
            raw_parts[arm] = {
                k: material[k][held] @ z[k] for k in z.files if k != "intercept"
            }
            raw = z["intercept"] + sum(raw_parts[arm].values())
        np.testing.assert_allclose(raw, predictions[arm], atol=1e-11)
    metrics = pd.read_csv(out / "metrics_by_trait.csv")
    norms = pd.read_csv(out / "source_norms.csv")
    signed = pd.read_csv(out / "signed_block_contributions.csv")
    for row in metrics.itertuples():
        j = list(TRAITS).index(row.trait)
        a, b = y[held, j], predictions[row.arm][:, j]
        np.testing.assert_allclose(pearsonr(a, b).statistic, row.r, atol=1e-12)
        np.testing.assert_allclose(spearmanr(a, b).statistic, row.rho, atol=1e-12)
        np.testing.assert_allclose(
            [abs(a - b).mean(), np.sqrt(np.mean((a - b) ** 2))],
            [row.mae, row.rmse],
            atol=1e-12,
        )
        norms_row = norms[(norms.arm == row.arm) & (norms.trait == row.trait)].iloc[0]
        if row.arm == "native_raw":
            src = material["native"][source, j]
        else:
            with np.load(
                out / f"private_affine_{row.arm}.npz", allow_pickle=False
            ) as z:
                src = (
                    z["intercept"]
                    + sum(
                        material[k][source] @ z[k] for k in z.files if k != "intercept"
                    )
                )[:, j]
        np.testing.assert_allclose(
            [src.mean(), src.std(ddof=1)],
            [norms_row.score_mean, norms_row.score_sd],
            atol=1e-12,
        )
        pieces = signed[(signed.arm == row.arm) & (signed.trait == row.trait)]
        for component in pieces.itertuples():
            part = raw_parts[row.arm][component.block][:, j]
            cov = np.cov(a, part, ddof=1)[0, 1]
            np.testing.assert_allclose(
                [cov, cov / (a.std(ddof=1) * b.std(ddof=1))],
                [component.cov_y_component, component.signed_r_contribution],
                atol=1e-12,
            )
        np.testing.assert_allclose(
            pieces.signed_r_contribution.sum(), row.r, atol=1e-12
        )
    for delta in summary["deltas"]:
        rng = np.random.default_rng(delta["seed"])
        ix = rng.integers(0, len(held), (delta["draws"], len(held)))
        a, b = predictions[delta["arm"]], predictions[delta["reference"]]
        values = (
            correlation(y[held][ix], a[ix]) - correlation(y[held][ix], b[ix])
        ).mean(1)
        if not np.isfinite(values).all() or delta["invalid_draws"]:
            raise ValueError("Unexpected degenerate bootstrap")
        np.testing.assert_allclose(
            np.quantile(values, [0.025, 0.975]),
            [delta["lower"], delta["upper"]],
            atol=1e-12,
        )
    result = {
        "status": "PASS",
        "native_logits_replayed": count,
        "native_score_max_error": native_error,
        "score_max_error": max_score_error,
        "criterion_rows_checked": len(metrics),
        "norm_rows_checked": len(norms),
        "signed_block_rows_checked": len(signed),
        "paired_intervals_checked": len(summary["deltas"]),
        "final_source_refits_checked": 5,
        "source_authors": len(source),
        "held_authors": len(held),
        "scope": "same-agent independent arithmetic/cold replay, not independent-executor replication",
        "summary_sha256": digest(out / "summary.json"),
        "audit_code_sha256": digest(Path(__file__)),
    }
    write(out / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "all_plan_closure_v1/native_construction_r1",
    )
    with threadpool_limits(limits=2):
        audit(parser.parse_args().output)
