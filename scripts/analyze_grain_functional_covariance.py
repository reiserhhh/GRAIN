#!/usr/bin/env python3
"""M25B: locate matched-query relations in mean/configuration/final/remainder terms."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    corr,
    read,
    sha,
    write,
)

QUERIES = ("TF_cont", "neuroticism", "confirmation_seeking")
PARTS = ("mean", "configuration", "final", "native_remainder")


def covariance_ledger(left, right):
    """Signed component cross-covariances exactly add to total cross-covariance."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (
        left.ndim != 2
        or right.ndim != 2
        or len(left) != len(right)
        or len(left) < 3
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Matched finite observation-by-component matrices required")
    a, b = left - left.mean(0), right - right.mean(0)
    matrix = a.T @ b / len(a)
    total = float(np.mean(a.sum(1) * b.sum(1)))
    return matrix, total, abs(float(matrix.sum()) - total)


def main():
    """Keep all query pairs and strata; no learned shared factor or new neural call."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    source, out = args.analysis_dir.resolve(), args.output_dir.resolve()
    if not source.is_relative_to(BASE) or not out.is_relative_to(BASE):
        raise ValueError("Use ignored result paths")
    if read(source / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("M25 analysis incomplete")
    events = pd.read_csv(source / "private_responses.csv")
    groups, arrays = {}, {}
    for query in QUERIES:
        rows = events[events["query"] == query].copy()
        rows["native_remainder"] = rows.native_score - rows[
            ["mean", "configuration", "final"]
        ].sum(axis=1)
        ref = (
            rows[rows.role == "reference"]
            .groupby(["view", "condition"])[list(PARTS)]
            .mean()
        )
        held = (
            rows[rows.role == "evaluation"]
            .set_index(["author_id", "view", "condition"])
            .sort_index()
        )
        centers = np.array([ref.loc[(v, c)].to_numpy() for _, v, c in held.index])
        arrays[query] = held[list(PARTS)].to_numpy() - centers
        groups[query] = held.index
    if any(not groups[q].equals(groups[QUERIES[0]]) for q in QUERIES):
        raise ValueError("Cross-query material mismatch")
    index = groups[QUERIES[0]].to_frame(index=False)
    rows, summary = [], []
    for left, right in itertools.combinations(QUERIES, 2):
        for stratum in ("pooled", "0/e1", "0/e2", "1/e1", "1/e2"):
            keep = (
                np.ones(len(index), bool)
                if stratum == "pooled"
                else (index.view.astype(str) + "/" + index.condition).to_numpy()
                == stratum
            )
            a, b = arrays[left][keep], arrays[right][keep]
            matrix, total, error = covariance_ledger(a, b)
            scale = float(np.std(a.sum(1)) * np.std(b.sum(1)))
            summary.append(
                {
                    "left": left,
                    "right": right,
                    "stratum": stratum,
                    "events": int(keep.sum()),
                    "native_r": corr(a.sum(1), b.sum(1)),
                    "functional_r": corr(a[:, :3].sum(1), b[:, :3].sum(1)),
                    "configuration_r": corr(a[:, 1], b[:, 1]),
                    "native_covariance": total,
                    "functional_covariance": float(matrix[:3, :3].sum()),
                    "accounting_error": error,
                    "functional_r_units": float(matrix[:3, :3].sum() / scale)
                    if scale > 1e-14
                    else None,
                }
            )
            for i, lp in enumerate(PARTS):
                for j, rp in enumerate(PARTS):
                    rows.append(
                        {
                            "left": left,
                            "right": right,
                            "stratum": stratum,
                            "left_part": lp,
                            "right_part": rp,
                            "covariance": float(matrix[i, j]),
                            "native_r_contribution": float(matrix[i, j] / scale)
                            if scale > 1e-14
                            else None,
                        }
                    )
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out / "signed_cross_path_covariance.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "matched_query_summary.csv", index=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_M25B_FUNCTIONAL_COVARIANCE_V1",
            "status": "COMPLETE",
            "script_sha256": sha(__file__),
            "response_sha256": sha(source / "private_responses.csv"),
            "parent_manifest_sha256": sha(source / "manifest.json"),
            "fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "max_accounting_error": max(s["accounting_error"] for s in summary),
        },
    )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
