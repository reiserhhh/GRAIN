#!/usr/bin/env python3
"""M30/M31: paired contexts and descriptive common/specific query accounting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def paired_delta(x, z, y, seed=20260909, draws=2000):
    """Bootstrap an exact same-participant difference of two correlations."""
    x, z, y = (np.asarray(v, dtype=float) for v in (x, z, y))
    if not (x.shape == z.shape == y.shape) or x.ndim != 1:
        raise ValueError("Matched one-dimensional participant arrays required")
    finite = np.isfinite(x) & np.isfinite(z) & np.isfinite(y)
    x, z, y = x[finite], z[finite], y[finite]
    if len(y) < 4 or any(np.ptp(v) < 1e-14 for v in (x, z, y)):
        raise ValueError("Nonconstant matched participant scores required")
    rng, delta = np.random.default_rng(seed), []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        if any(np.ptp(v[ix]) < 1e-14 for v in (x, z, y)):
            continue
        delta.append(np.corrcoef(x[ix], y[ix])[0, 1] - np.corrcoef(z[ix], y[ix])[0, 1])
    return {
        "n": len(y),
        "delta_r_free_minus_direct": float(
            np.corrcoef(x, y)[0, 1] - np.corrcoef(z, y)[0, 1]
        ),
        "delta_lower": float(np.quantile(delta, 0.025)),
        "delta_upper": float(np.quantile(delta, 0.975)),
        "bootstrap_valid": len(delta),
    }


def common_specific(scores, definitions, labels):
    """Account for covariance without discarding shared signal or fitting outcomes.

    Standardization and projection describe this whole observed sample; they are
    not a held-out predictor. Other phases of the same semantic query family are
    excluded from the common score to avoid a duplicate wording self-proxy.
    """
    if not scores.index.equals(labels.index):
        raise ValueError("Exact participant alignment required")
    if (scores.std() < 1e-14).any():
        raise ValueError("Nonconstant query scores required")
    z = (scores - scores.mean()) / scores.std()
    corr = z.corr().to_numpy()
    structures = {
        "pc1_fraction_standardized": float(np.linalg.eigvalsh(corr)[-1] / len(corr)),
        "median_off_diagonal_r": float(np.median(corr[np.triu_indices(len(corr), 1)])),
        "queries": len(corr),
        "participants": len(scores),
        "fit_scope": "descriptive whole-sample query geometry, no human-outcome fit",
    }
    rows = []
    for q in scores:
        others = [
            k for k in scores if definitions[k]["family"] != definitions[q]["family"]
        ]
        if not others:
            raise ValueError("At least one independent semantic family required")
        c = z[others].mean(axis=1).to_numpy()
        x, y = z[q].to_numpy(), labels[q].to_numpy()
        c = c - c.mean()
        loading = float(c @ x / (c @ c))
        shared = loading * c
        residual = x - shared
        denominator = float(np.std(x, ddof=1) * np.std(y, ddof=1))
        total = float(np.cov(x, y)[0, 1] / denominator)
        shared_link = float(np.cov(shared, y)[0, 1] / denominator)
        specific_link = float(np.cov(residual, y)[0, 1] / denominator)
        rows.append(
            {
                "query": q,
                "phase": definitions[q]["phase"],
                "other_query_count": len(others),
                "common_loading": loading,
                "query_variance_common_fraction": float(np.var(shared) / np.var(x)),
                "total_external_r": total,
                "shared_external_r_contribution": shared_link,
                "specific_external_r_contribution": specific_link,
                "accounting_error": abs(total - shared_link - specific_link),
                "specific_external_r": association(residual, y)["pearson_r"],
            }
        )
    return structures, pd.DataFrame(rows)


def load_view(folder):
    """Verify the frozen native predictions before opening criterion files."""
    if (
        read(folder / "external_summary.json")["status"]
        != "COMPLETE_EXPLORATORY_EXTERNAL_LINK"
    ):
        raise ValueError("Completed external analysis required")
    path = folder / "private_predictions.csv"
    if sha(path) != read(folder / "prediction_freeze.json")["sha256"]:
        raise ValueError("Frozen prediction bytes changed")
    data = pd.read_csv(path)
    if data.duplicated(["participant_code", "query"]).any():
        raise ValueError("Duplicate query-participant scores")
    return data.pivot(
        index="participant_code", columns="query", values="native_score"
    ).sort_index()


def run(free, direct, inventory, out):
    """Build paired view contrasts and complete signed association decompositions."""
    x, z = load_view(free), load_view(direct)
    defs = read(free / "queries.json")
    if (
        defs != read(direct / "queries.json")
        or not x.index.equals(z.index)
        or not x.columns.equals(z.columns)
    ):
        raise ValueError("Same participants and frozen queries required in both views")
    criteria_path = inventory / "private_meps_scores.csv"
    for path in (free, direct):
        if sha(criteria_path) != read(path / "manifest.json")["criterion_cache_sha256"]:
            raise ValueError("Criterion cache changed")
    y = pd.read_csv(criteria_path).pivot(
        index="participant_code", columns="scale", values="score"
    )
    y = y.reindex(index=x.index, columns=x.columns)
    if x.isna().any().any() or z.isna().any().any() or y.isna().any().any():
        raise ValueError("Incomplete view or criterion matrix")
    out.mkdir(parents=True, exist_ok=False)
    paired = []
    for q in x:
        a, b = association(x[q], y[q]), association(z[q], y[q])
        paired.append(
            {
                "query": q,
                "phase": defs[q]["phase"],
                "free_r": a["pearson_r"],
                "direct_r": b["pearson_r"],
                "free_p": a["p"],
                "direct_p": b["p"],
                "cross_view_r": association(x[q], z[q])["pearson_r"],
                **paired_delta(x[q], z[q], y[q]),
            }
        )
    pairs = pd.DataFrame(paired)
    for view in ("free", "direct"):
        pairs[f"{view}_q_bh_phase"] = pairs.groupby("phase")[f"{view}_p"].transform(
            bh_adjust
        )
    pairs.to_csv(out / "paired_view_associations.csv", index=False)
    summaries = {}
    for view, scores, folder in (("free_chat", x, free), ("direct_answers", z, direct)):
        geometry, account = common_specific(scores, defs, y)
        account.to_csv(out / f"{view}_signed_common_specific.csv", index=False)
        scores.corr().to_csv(out / f"{view}_query_correlation.csv")
        cover = pd.DataFrame(read(folder / "private_coverage.json")).set_index(
            "participant_code"
        )
        lengths = np.log1p(cover.reindex(scores.index).selected_chars.to_numpy())
        opportunity = pd.DataFrame(
            [
                {
                    "query": q,
                    "score_log_chars_r": association(scores[q], lengths)["pearson_r"],
                    "criterion_log_chars_r": association(y[q], lengths)["pearson_r"],
                }
                for q in scores
            ]
        )
        opportunity.to_csv(out / f"{view}_text_amount_links.csv", index=False)
        summaries[view] = geometry | {
            "max_accounting_error": float(account.accounting_error.max())
        }
    write(
        out / "summary.json",
        {
            "status": "COMPLETE_PAIRED_CONTEXT_DESCRIPTION",
            "participants": len(x),
            "queries": len(x.columns),
            "training": False,
            "common_specific_is_held_out_prediction": False,
            "views": summaries,
            "inputs": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    free / "private_predictions.csv",
                    direct / "private_predictions.csv",
                    criteria_path,
                )
            },
        },
    )
    print(pairs.round(4).to_string(index=False), flush=True)
    print(summaries, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("free-dir", "direct-dir", "inventory-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Local ignored mechanism output required")
    run(
        a.free_dir.resolve(),
        a.direct_dir.resolve(),
        a.inventory_dir.resolve(),
        a.output_dir.resolve(),
    )
