#!/usr/bin/env python3
"""M29/M31: complete source query matrices and paired function/native accounting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import AXES, BIG5
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def paired_quality(function, native, target, binary=False, draws=2000):
    """Resample participants, keeping native/function/criterion tuples together."""
    f, n, y = (np.asarray(x, dtype=float) for x in (function, native, target))
    if f.shape != n.shape or f.shape != y.shape or f.ndim != 1:
        raise ValueError("Aligned participant vectors required")
    if not all(np.isfinite(x).all() for x in (f, n, y)):
        raise ValueError("Finite inputs required")
    result = {
        "n": len(y),
        "native_fidelity_r": association(f, n)["pearson_r"],
        "function_external_r": association(f, y)["pearson_r"],
        "native_external_r": association(n, y)["pearson_r"],
        "function_mae": float(np.mean(np.abs(f - y))),
        "function_rmse": float(np.sqrt(np.mean((f - y) ** 2))),
        "native_mae": float(np.mean(np.abs(n - y))),
        "native_rmse": float(np.sqrt(np.mean((n - y) ** 2))),
        "function_outside_0_1_fraction": float(np.mean((f < 0) | (f > 1))),
        "mae_units": "unclipped affine score against normalized criterion; not calibrated questionnaire/probability error",
    }
    result["delta_r"] = result["function_external_r"] - result["native_external_r"]
    rng, samples = np.random.default_rng(20260909), []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        if any(np.ptp(x[ix]) < 1e-14 for x in (f, n, y)):
            continue
        fr, nr = np.corrcoef(f[ix], y[ix])[0, 1], np.corrcoef(n[ix], y[ix])[0, 1]
        row = {"function_r": fr, "native_r": nr, "delta_r": fr - nr}
        if binary:
            fa, na = roc_auc_score(y[ix], f[ix]), roc_auc_score(y[ix], n[ix])
            row.update(function_auc=fa, native_auc=na, delta_auc=fa - na)
        samples.append(row)
    table = pd.DataFrame(samples)
    for k in table:
        result[f"{k}_lower"], result[f"{k}_upper"] = (
            float(v) for v in table[k].quantile([0.025, 0.975])
        )
    if binary:
        result["function_auc"] = float(roc_auc_score(y, f))
        result["native_auc"] = float(roc_auc_score(y, n))
        result["delta_auc"] = result["function_auc"] - result["native_auc"]
    result["bootstrap_valid"] = len(samples)
    return result


def run(source, out):
    """Build all source links without fitting coefficients, norms or thresholds."""
    if read(source / "external_summary.json")["status"] != "COMPLETE_M29":
        raise ValueError("M29 source analysis must complete first")
    if (
        sha(source / "private_predictions.csv")
        != read(source / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Frozen functions changed")
    data = pd.read_csv(source / "private_external_links.csv", dtype={"author_id": str})
    if data.duplicated(["author_id", "query"]).any() or set(data["query"]) != {
        *BIG5,
        *AXES,
    }:
        raise ValueError(
            "Exactly one observation for each planned author-query required"
        )
    out.mkdir(parents=True, exist_ok=False)
    paired, terms, matrix = [], [], []
    for q, g in data.groupby("query"):
        paired.append(
            {"query": q, **paired_quality(g.score, g.prediction, g.y_true, q in AXES)}
        )
        denominator = np.std(g.score, ddof=1) * np.std(g.y_true, ddof=1)
        contributions = {
            k: float(np.cov(g[k], g.y_true)[0, 1] / denominator)
            for k in ("mean", "configuration", "final")
        }
        total = association(g.score, g.y_true)["pearson_r"]
        terms.append(
            {
                "query": q,
                "total_r": total,
                **contributions,
                "accounting_error": abs(total - sum(contributions.values())),
            }
        )
    for group, subset in data.groupby("source_group"):
        ys = subset.pivot(index="author_id", columns="query", values="y_true")
        for arm in ("score", "mean", "configuration", "final", "prediction"):
            xs = subset.pivot(index="author_id", columns="query", values=arm).reindex(
                ys.index
            )
            if xs.isna().any().any() or ys.isna().any().any():
                raise ValueError("Incomplete within-source trait coverage")
            for q in xs:
                for k in ys:
                    matrix.append(
                        {
                            "source_group": group,
                            "arm": arm,
                            "query": q,
                            "criterion": k,
                            "intended": q == k,
                            **association(xs[q], ys[k], binary=k in AXES),
                        }
                    )
    pairs, accounting, links = (
        pd.DataFrame(paired),
        pd.DataFrame(terms),
        pd.DataFrame(matrix),
    )
    links["q_bh_source_arm"] = links.groupby(["source_group", "arm"]).p.transform(
        bh_adjust
    )
    pairs.to_csv(out / "paired_native_function_metrics.csv", index=False)
    accounting.to_csv(out / "signed_external_r_contributions.csv", index=False)
    links.to_csv(out / "full_query_criterion_matrix.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE_SOURCE_FUNCTION_ATLAS",
            "queries": 9,
            "links": len(links),
            "authors_by_source": data.groupby("source_group")
            .author_id.nunique()
            .to_dict(),
            "max_covariance_error": float(accounting.accounting_error.max()),
            "new_fits": 0,
            "source_sha256": sha(source / "private_external_links.csv"),
            "function_sha256": read(source / "manifest.json")["functions_sha256"],
        },
    )
    print(pairs.drop(columns="mae_units").round(4).to_string(index=False), flush=True)
    print(accounting.round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Use local ignored mechanism outputs")
    run(a.source_dir.resolve(), a.output_dir.resolve())
