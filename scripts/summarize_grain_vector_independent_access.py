#!/usr/bin/env python3
"""Analyze frozen cross-current access without treating control as trait validity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_natural_patch import aggregate


def sign_flip_p(values, seed=913, draws=20000):
    """Two-sided author-level random-sign test for the frozen mean contrast."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Need finite author contrasts")
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1, 1], size=(draws, len(x))) * x).mean(axis=1))
    return float((1 + (null >= abs(x.mean())).sum()) / (draws + 1))


def main():
    """Report both confirmation endpoints regardless of sign or code competence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda name: json.loads((a.parent_dir / name).read_text(encoding="utf-8"))
    manifest, status = read("manifest.json"), read("run_status.json")
    if status["status"] != "COMPLETE" or not status["all_instrument_controls_exact"]:
        raise ValueError("Only complete, exact-instrument confirmation may be analyzed")
    frame = pd.read_csv(a.parent_dir / "interventions.csv")
    n = manifest["eligible_authors"]
    if status["authors"] != n or frame.author_id.nunique() != n or len(frame) != n * 24:
        raise ValueError("Missing independent-author conditions")
    if (
        frame.duplicated(["author_id", "condition", "method", "sign"]).any()
        or frame.norm_match_error.max() > 0.01
    ):
        raise ValueError("Duplicate or unmatched intervention")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new analysis directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    codes = pd.read_csv(a.parent_dir / "code_controls.csv")
    code_counts = codes.groupby("swapped").correct.sum()
    paired = codes.pivot(
        index=["scenario", "positive_evidence"], columns="swapped", values="margin"
    )
    opposite = int((paired[False] * paired[True] < 0).sum())
    qualified = bool(len(codes) == 16 and code_counts.min() >= 6 and opposite >= 6)
    code_summary = {
        "normal_correct": int(code_counts.loc[False]),
        "swapped_correct": int(code_counts.loc[True]),
        "per_code_total": 8,
        "opposite_pairs": opposite,
        "total_pairs": 8,
        "limited_code_control_qualified": qualified,
    }
    write_json(out / "code_control_summary.json", code_summary)
    frame["abs_linear_error"] = abs(frame.delta - frame.predicted_delta)
    frame["resolved"] = (abs(frame.delta) >= 0.125) & (
        abs(frame.predicted_delta) >= 0.125
    )
    frame["sign_agreement"] = np.where(
        frame.resolved, np.sign(frame.delta) == np.sign(frame.predicted_delta), np.nan
    )
    aggregate(
        frame,
        ["condition", "method", "sign"],
        [
            "delta",
            "predicted_delta",
            "abs_linear_error",
            "resolved",
            "sign_agreement",
            "gradient_cosine",
        ],
    ).to_csv(out / "effects.csv", index=False)
    ids = ["author_id", "condition", "method"]
    plus = frame[frame.sign == 1]
    minus = frame[frame.sign == -1]
    both = plus.merge(
        minus, on=ids, suffixes=("_plus", "_minus"), validate="one_to_one"
    )
    both["signed_response"] = (both.delta_plus - both.delta_minus) / 2
    both["oriented_response"] = both.signed_response * both.polarity_plus
    both["oriented_prediction"] = (
        (both.predicted_delta_plus - both.predicted_delta_minus)
        / 2
        * both.polarity_plus
    )
    both["gradient_cosine"] = both.gradient_cosine_plus
    both.to_csv(out / "author_effects.csv", index=False)
    aggregate(
        both,
        ["condition", "method"],
        [
            "signed_response",
            "oriented_response",
            "oriented_prediction",
            "gradient_cosine",
        ],
    ).to_csv(out / "signed_effects.csv", index=False)
    candidate = both[both.method == "candidate"]
    random = (
        both[both.method.str.startswith("random_")]
        .groupby(["author_id", "condition"], as_index=False)
        .oriented_response.mean()
        .rename(columns={"oriented_response": "random_mean"})
    )
    contrasts = candidate.merge(
        random, on=["author_id", "condition"], validate="one_to_one"
    )
    contrasts["advantage"] = contrasts.oriented_response - contrasts.random_mean
    contrasts.to_csv(out / "random_contrasts.csv", index=False)
    aggregate(contrasts, ["condition"], ["advantage"]).to_csv(
        out / "random_comparisons.csv", index=False
    )
    endpoints = []
    for condition in ("transfer", "code_swapped"):
        sub = contrasts[contrasts.condition == condition]
        summary = grouped_bootstrap(sub.advantage, sub.author_id)
        endpoints.append(
            {
                "condition": condition,
                **summary,
                "p_two_sided": sign_flip_p(sub.advantage),
                "instrument_qualified": condition != "code_swapped" or qualified,
            }
        )
    order = np.argsort([r["p_two_sided"] for r in endpoints])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, endpoints[i]["p_two_sided"] * (2 - rank)))
        endpoints[i]["p_holm"] = running
    pd.DataFrame(endpoints).to_csv(out / "confirmation_endpoints.csv", index=False)
    cross = candidate.pivot(
        index="author_id", columns="condition", values="oriented_response"
    )
    cross["transfer_minus_source"] = cross.transfer - cross.source
    cross["code_minus_transfer"] = cross.code_swapped - cross.transfer
    cross.to_csv(out / "within_author_transfer.csv")
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "parent_identity": manifest["identity"],
            "code_sha256": code_fingerprint(ROOT),
            "authors": n,
            "opened_authors": 48,
            "runtime": status,
            "code_control": code_summary,
            "confirmation_endpoints": endpoints,
            "scope": "Cross-current procedure transfer in one model, not shared personality direction or truth accuracy",
            "next": "VM05 matched-material external links and second observer; no reopening local direction search",
        },
    )
    print(
        json.dumps({"authors": n, "codes": code_summary, "endpoints": endpoints}),
        flush=True,
    )


if __name__ == "__main__":
    main()
