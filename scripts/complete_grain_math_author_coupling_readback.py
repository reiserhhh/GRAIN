#!/usr/bin/env python3
"""Finish result-after metadata readback without altering frozen HM64 science."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_author_coupling as M
from scripts import run_grain_math_author_coupling as R


def main():
    """Reuse the completed overlap table; correct one historical manifest key."""
    old = R.OUT / "04_finite_readback"
    out = R.OUT / "05_finite_readback_completion"
    R.sealed(R.OUT)
    manifest = R.N.read(old / "manifest.json")
    for name, digest in manifest.items():
        if R.N.sha(ROOT / name) != digest:
            raise ValueError("Preserve the original failed readback code/inputs")
    out.mkdir(exist_ok=False)
    inputs = (
        old / "manifest.json",
        old / "private_exact_overlap.csv",
        Path(__file__).resolve(),
    )
    R.N.write(
        out / "manifest.json", {str(p.relative_to(ROOT)): R.N.sha(p) for p in inputs}
    )
    overlap = pd.read_csv(old / "private_exact_overlap.csv")
    upstream = R.N.read(R.N.SLICES.parent / "manifest.json")
    code_key = "scripts/run_grain_dynamic_path_channels.py"
    builder_ok = upstream["input_sha256"][code_key] == R.N.sha(ROOT / code_key)
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
    summary = {
        "status": "RESULT_AFTER_HM64_READBACK_COMPLETED_NOT_NEW_VALIDATION",
        "retained_failure": "04_finite_readback used hashes; upstream schema is input_sha256; no primary changes or rerun",
        "cross_panel_exact_vector_matches": int(
            overlap.cross_panel_exact_vector_matches.sum()
        ),
        "authors_with_exact_matches": int(
            overlap.cross_panel_exact_vector_matches.gt(0).sum()
        ),
        "upstream_disjoint_comment_thread_builder_hash_verified": builder_ok,
        "raw_source_records_not_reopened": True,
        "historical_ordering": "Upstream sorts timestamps; HM64 uses inherited indices, not dates/equal intervals",
        "frozen_plan_wording_erratum": "not chronological time was too broad; cyclic pairing is not a time/state intervention",
        "primary_science_unchanged": True,
        "new_calls_fits_labels_raw": [0, 0, 0, 0],
    }
    R.N.write(out / "summary.json", summary)
    R.N.write(
        out / "output_seal.json",
        {p.name: R.N.sha(p) for p in out.iterdir() if p.name != "manifest.json"},
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
