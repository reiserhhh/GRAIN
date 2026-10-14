#!/usr/bin/env python3
"""Result-after HM64 readback: control contrast, null units and numeric overlap."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_author_coupling as M
from scripts import run_grain_math_author_coupling as R


def main():
    """Read frozen outcomes without changing primary inputs, units or results."""
    out = R.OUT / "04_finite_readback"
    R.sealed(R.OUT)
    out.mkdir(exist_ok=False)
    inputs = (
        R.OUT / "manifest.json",
        R.OUT / "output_seal.json",
        R.OUT / "private_author_summaries.csv",
        R.OUT / "identity_permutation_null.csv",
        R.OUT / "summary_by_cohort.csv",
        R.N.SLICES.parent / "manifest.json",
        ROOT / "scripts/run_grain_dynamic_path_channels.py",
        Path(__file__).resolve(),
    )
    R.N.write(
        out / "manifest.json", {str(p.relative_to(ROOT)): R.N.sha(p) for p in inputs}
    )
    supports, _, _ = R.load_numeric()
    overlap = []
    for author, item in supports.items():
        sets = [{h.tobytes() for h in item["h"][item["panel"] == p]} for p in range(3)]
        overlap.append(
            {
                "author": author,
                "group": R.cohort(item),
                "cross_panel_exact_vector_matches": sum(
                    len(sets[a] & sets[b])
                    for a, b in itertools.combinations(range(3), 2)
                ),
            }
        )
    pd.DataFrame(overlap).to_csv(out / "private_exact_overlap.csv", index=False)
    upstream = R.N.read(R.N.SLICES.parent / "manifest.json")
    code_key = "scripts/run_grain_dynamic_path_channels.py"
    bound_disjoint_builder = upstream["hashes"][code_key] == R.N.sha(ROOT / code_key)
    author = pd.read_csv(R.OUT / "private_author_summaries.csv")
    rows = []
    for group, metric in itertools.product(R.GROUPS, M.METRICS):
        frame = author[(author.group == group) & (author.metric == metric)]
        wide = frame.pivot(index="author", columns="arm", values="mean")
        delta = (wide.joint_with_coupling - wide.joint_with_shifted).to_numpy()[:, None]
        mean, ci, _ = M.bootstrap_summary(delta, R.SEED, R.BOOTSTRAPS)
        rows.append(
            {
                "group": group,
                "metric": metric,
                "delta": float(mean[0]),
                "lower": float(ci[0, 0]),
                "upper": float(ci[1, 0]),
            }
        )
    pd.DataFrame(rows).to_csv(out / "secondary_joint_pairing_contrast.csv", index=False)
    null = pd.read_csv(R.OUT / "identity_permutation_null.csv")
    nulls = []
    for (group, arm, metric), values in null.groupby(["group", "arm", "metric"]):
        score = values.mean_win_fraction.to_numpy()
        bounds = np.quantile(score, [0.025, 0.975])
        nulls.append(
            {
                "group": group,
                "arm": arm,
                "metric": metric,
                "null_mean": float(score.mean()),
                "null_lower": float(bounds[0]),
                "null_upper": float(bounds[1]),
            }
        )
    pd.DataFrame(nulls).to_csv(out / "null_summary.csv", index=False)
    R.N.write(
        out / "summary.json",
        {
            "status": "RESULT_AFTER_HM64_READBACK_NOT_NEW_VALIDATION",
            "cross_panel_exact_vector_matches": int(
                sum(o["cross_panel_exact_vector_matches"] for o in overlap)
            ),
            "authors_with_exact_matches": int(
                sum(o["cross_panel_exact_vector_matches"] > 0 for o in overlap)
            ),
            "upstream_disjoint_comment_thread_builder_hash_verified": bound_disjoint_builder,
            "raw_source_records_not_reopened": True,
            "historical_ordering": "Upstream code sorts timestamps; HM64 uses only inherited index order, not dates or equal time intervals",
            "frozen_plan_wording_erratum": "not chronological time was too broad; no chronological/state inference is made from this cyclic pairing control",
            "primary_science_unchanged": True,
            "new_calls_fits_labels_raw": [0, 0, 0, 0],
        },
    )
    R.N.write(
        out / "output_seal.json",
        {p.name: R.N.sha(p) for p in out.iterdir() if p.name != "manifest.json"},
    )
    print(R.N.read(out / "summary.json"), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
