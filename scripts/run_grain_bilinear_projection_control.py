#!/usr/bin/env python3
"""Full-embedding reference and fixed-projection accounting for the paired study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.local_bilinear import keyed_directions
from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_local_construction import ESSAYS, correlation_row, load_texts
from scripts.run_grain_predecessor_reference import mean_pearson, paired_delta

PLAN = ROOT / "docs/GRAIN_LOCAL_BILINEAR_PROJECTION_ADDENDUM.md"


def run(args):
    """Execute the source-only companion, preserving the original comparison."""
    out = args.study / "projection_control"
    out.mkdir(exist_ok=False)
    write(
        out / "contract.json",
        {
            "plan_sha256": digest(PLAN),
            "script_sha256": digest(Path(__file__)),
            "study_sha256": digest(args.study / "contract.json"),
            "alphas": [1.0, 10.0, 100.0, 1000.0],
        },
    )
    datasets = load_texts()
    with np.load(DEFAULT / "private_arrays.npz", allow_pickle=False) as z:
        ys = {"pandora": z["y"]}
    ytable = pd.read_csv(ESSAYS, usecols=["user_id", *TRAITS]).set_index("user_id")
    ys["essays"] = ytable.loc[datasets["essays"]["ids"], list(TRAITS)].to_numpy()
    chunks, stop = [], 0
    for path in sorted((DEFAULT / "sentence_batches").glob("batch_*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["start"] != stop or meta["sha256"] != digest(path):
            raise ValueError("Invalid item cache")
        with np.load(path, allow_pickle=False) as z:
            chunks.append(z["vectors"])
        stop = meta["stop"]
        if stop >= 300:
            break
    q = np.concatenate(chunks)[:300]
    items = json.loads((DEFAULT / "ipip_neo300.json").read_text())["items"]
    directions, facets = keyed_directions(q, items)
    rows, comparisons, energy, checks = [], [], [], []
    for domain in ("pandora", "essays"):
        root = args.study / domain
        summary = json.loads((root / "summary.json").read_text())
        for name, sha in summary["files"].items():
            if digest(root / name) != sha:
                raise ValueError("Changed comparison inputs")
        with np.load(root / "private_blocks.npz", allow_pickle=False) as z:
            h, s = z["semantic"], z["affinity"]
            train, val, held = (z[k] for k in ("train", "val", "held"))
        source = np.sort(np.r_[train, val])
        y = ys[domain]
        projection_error = float(np.max(abs(h @ directions - s)))
        if projection_error > 1e-7:
            raise ValueError("Unexpected normalized-mean projection mismatch")
        pca = joblib.load(root / "private_model_semantic.joblib")["features"].pca_
        retained = np.sum((pca.components_ @ directions) ** 2, axis=0) / np.sum(
            directions**2, axis=0
        )
        energy.extend(
            {"domain": domain, "facet": f, "retained_directional_energy": float(v)}
            for f, v in zip(facets, retained, strict=True)
        )
        scaler = StandardScaler().fit(h[train])
        x, xv = scaler.transform(h[train]), scaler.transform(h[val])
        tuning = []
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            model = Ridge(alpha=alpha, solver="lsqr").fit(x, y[train])
            tuning.append(
                {
                    "alpha": alpha,
                    "validation_mean_r": float(mean_pearson(y[val], model.predict(xv))),
                }
            )
        chosen = max(tuning, key=lambda r: r["validation_mean_r"])
        scaler = StandardScaler().fit(h[source])
        head = Ridge(alpha=chosen["alpha"], solver="lsqr").fit(
            scaler.transform(h[source]), y[source]
        )
        pred = head.predict(scaler.transform(h[held]))
        beta = head.coef_.T / scaler.scale_[:, None]
        intercept = head.intercept_ - scaler.mean_ @ beta
        cold = h[held] @ beta + intercept
        np.testing.assert_allclose(cold, pred, atol=1e-10)
        joblib.dump(
            {"scaler": scaler, "head": head, "source": source, "chosen": chosen},
            out / f"private_{domain}_model.joblib",
        )
        np.savez_compressed(
            out / f"private_{domain}_scores.npz",
            prediction=pred,
            held=held,
            beta=beta,
            intercept=intercept,
        )
        for j, trait in enumerate(TRAITS):
            rows.append(
                {
                    "domain": domain,
                    "arm": "full_semantic_ridge",
                    "trait": trait,
                    **correlation_row(y[held, j], pred[:, j]),
                }
            )
        with np.load(root / "private_predictions.npz", allow_pickle=False) as z:
            for arm in ("semantic", "additive", "outer", "paired"):
                comparisons.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "reference": "full_semantic_ridge",
                        **paired_delta(y[held], z[arm], pred, draws=1000),
                    }
                )
        checks.append(
            {
                "domain": domain,
                "source": len(source),
                "held": len(held),
                "projection_error": projection_error,
                "cold_error": float(np.max(abs(cold - pred))),
                "tuning": tuning,
                "chosen": chosen,
                "mean_r": float(mean_pearson(y[held], pred)),
            }
        )
        print(json.dumps(checks[-1]), flush=True)
    pd.DataFrame(rows).to_csv(out / "metrics_by_trait.csv", index=False)
    pd.DataFrame(energy).to_csv(out / "directional_energy.csv", index=False)
    write(
        out / "summary.json",
        {
            "checks": checks,
            "deltas": comparisons,
            "new_neural_calls": 0,
            "files": {
                p.name: digest(p)
                for p in out.iterdir()
                if p.is_file() and not p.name.startswith("._")
            },
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--study", type=Path, default=BASE / "all_plan_closure_v1/local_bilinear_r2"
    )
    with threadpool_limits(limits=2):
        run(p.parse_args())
