#!/usr/bin/env python3
"""M27: complete source-factor atlas and explicit local MEPS score inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import (
    association,
    audit_scored_meps,
    bh_adjust,
    catalog,
)
from grain_research.vector_mechanisms.readout_cache_cpu import read, sha, write

D1 = ROOT / "results/grain_vector_mechanisms/d1_author_discovery_r1"


def linked_matrix(frame, block, observer, arm):
    """Link different query/criterion columns only through the same source author."""
    if frame.duplicated(["author_id", "trait"]).any():
        raise ValueError("Duplicate author-query scores")
    pred = frame.pivot(index="author_id", columns="trait", values="prediction")
    truth = frame.pivot(index="author_id", columns="trait", values="y_true")
    external, native = [], []
    for q in pred.columns:
        for k in truth.columns:
            common = pd.concat([pred[q], truth[k]], axis=1).dropna()
            external.append(
                {
                    "block": block,
                    "observer": observer,
                    "arm": arm,
                    "query": q,
                    "criterion": k,
                    "intended": q == k,
                    **association(
                        common.iloc[:, 0], common.iloc[:, 1], binary=block == "mbti"
                    ),
                }
            )
            pair = (
                pred[[q, k]].dropna()
                if q != k
                else pd.concat([pred[q], pred[q]], axis=1).dropna()
            )
            native.append(
                {
                    "block": block,
                    "observer": observer,
                    "arm": arm,
                    "query": q,
                    "other_query": k,
                    **association(pair.iloc[:, 0], pair.iloc[:, 1]),
                }
            )
    return external, native


def delta_interval(frame, observer_a, arm_a, observer_b, arm_b, *, draws):
    """Paired author bootstrap; never compare predictions on unmatched samples."""
    a = frame[(frame.observer == observer_a) & (frame.arm == arm_a)].set_index(
        "author_id"
    )
    b = frame[(frame.observer == observer_b) & (frame.arm == arm_b)].set_index(
        "author_id"
    )
    if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
        raise ValueError("Paired contrast author membership mismatch")
    b = b.loc[a.index]
    if not np.allclose(a.y_true, b.y_true, rtol=0, atol=1e-12):
        raise ValueError("Source outcomes differ by observer or arm")
    y, pa, pb = a.y_true.to_numpy(), a.prediction.to_numpy(), b.prediction.to_numpy()
    field = "auc" if frame.block.iloc[0] == "mbti" else "pearson_r"
    metric = lambda x, z: association(x, z, binary=field == "auc")[field]
    full_a, full_b = metric(pa, y), metric(pb, y)
    rng, boot = np.random.default_rng(20260909), []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        x, z = metric(pa[ix], y[ix]), metric(pb[ix], y[ix])
        if x is not None and z is not None:
            boot.append(x - z)
    return {
        "trait": frame.trait.iloc[0],
        "metric": field,
        "n": len(y),
        "observer_a": observer_a,
        "arm_a": arm_a,
        "observer_b": observer_b,
        "arm_b": arm_b,
        "value_a": full_a,
        "value_b": full_b,
        "delta": full_a - full_b if full_a is not None and full_b is not None else None,
        "delta_lower": float(np.quantile(boot, 0.025)) if boot else None,
        "delta_upper": float(np.quantile(boot, 0.975)) if boot else None,
        "bootstrap_valid": len(boot),
        "bootstrap_draws": draws,
    }


def analyze(out, meps_root, authorized, draws):
    """Build only aggregate tracked artifacts; private linked outcomes stay ignored."""
    out.mkdir(parents=True, exist_ok=False)
    source = D1 / "private_predictions.csv"
    d1 = read(D1 / "summary.json")
    if sha(source) != d1["outputs_sha256"][source.name]:
        raise ValueError("Frozen D1 predictions changed")
    write(out / "factor_catalog.json", catalog())
    frame = pd.read_csv(source, dtype={"author_id": str})
    frame["block"] = np.where(frame.task == "big5", "big5", "mbti")
    if (
        frame.trait.nunique() != 9
        or frame.duplicated(["author_id", "trait", "observer", "arm"]).any()
    ):
        raise ValueError("Expected unique, complete nine-target source cache")
    label_counts = frame.groupby(["author_id", "trait"]).y_true.nunique(dropna=False)
    if (label_counts != 1).any():
        raise ValueError("Labels vary across frozen methods")
    if not np.isfinite(frame[["prediction", "y_true"]].to_numpy()).all():
        raise ValueError("D1 source has nonfinite records")
    external, native = [], []
    for (block, observer, arm), group in frame.groupby(["block", "observer", "arm"]):
        e, n = linked_matrix(group, block, observer, arm)
        external.extend(e)
        native.extend(n)
    external, native = pd.DataFrame(external), pd.DataFrame(native)
    external["q_bh_within_block_observer_arm"] = external.groupby(
        ["block", "observer", "arm"]
    ).p.transform(bh_adjust)
    external.to_csv(out / "external_query_criterion_matrix.csv", index=False)
    native.to_csv(out / "predicted_query_matrix.csv", index=False)
    external[external.intended].to_csv(out / "diagonal_metrics.csv", index=False)
    criteria = []
    labels = frame.drop_duplicates(["author_id", "trait"])
    for block, group in labels.groupby("block"):
        truth = group.pivot(index="author_id", columns="trait", values="y_true")
        for q in truth.columns:
            for k in truth.columns:
                criteria.append(
                    {
                        "block": block,
                        "query": q,
                        "criterion": k,
                        **association(truth[q], truth[k]),
                    }
                )
    pd.DataFrame(criteria).to_csv(out / "true_criterion_matrix.csv", index=False)
    contrasts = []
    designs = [
        ("native_query", "raw", "e5_large", "raw"),
        ("native_query", "cue_mask", "native_query", "raw"),
        ("native_query", "cue_mask", "native_query", "placebo_mask"),
    ]
    for trait, group in frame.groupby("trait"):
        for design in designs:
            contrasts.append(delta_interval(group, *design, draws=draws))
        print(f"D1 contrasts complete: {trait}", flush=True)
    pd.DataFrame(contrasts).to_csv(out / "paired_author_contrasts.csv", index=False)
    hashes = {
        str(source.relative_to(ROOT)): sha(source),
        "factor_catalog.json": sha(out / "factor_catalog.json"),
        "plan": sha(ROOT / "docs/GRAIN_ALL_FACTOR_RESEARCH_GOAL_V5.md"),
    }
    meps = {"status": "not_requested", "training": False}
    if authorized:
        scores, audit, paths = audit_scored_meps(meps_root, authorized=True)
        scores.to_csv(out / "private_meps_scores.csv", index=False)
        audit.to_csv(out / "meps_score_audit.csv", index=False)
        wide = scores.pivot(index="participant_code", columns="scale", values="score")
        pairs = []
        for i, a in enumerate(wide.columns):
            for b in wide.columns[i + 1 :]:
                pairs.append(
                    {"scale_a": a, "scale_b": b, **association(wide[a], wide[b])}
                )
        pairs = pd.DataFrame(pairs)
        pairs["q_bh_all_scale_pairs"] = bh_adjust(pairs.p)
        pairs.to_csv(out / "meps_criterion_matrix.csv", index=False)
        meps = {
            "status": "SCORING_AND_MEMBERSHIP_AUDITED",
            "participants": len(wide),
            "score_columns": len(wide.columns),
            "unique_query_targets": int(audit["query"].nunique()),
            "max_stored_error": float(audit.stored_max_error.max()),
            "training": False,
            "external_text_predictions": "next_M30",
            "authorization": "2026-09-09 explicit user local exploratory linkage; V5 scope",
        }
        hashes.update({str(p.relative_to(meps_root)): sha(p) for p in paths})
    write(
        out / "summary.json",
        {
            "status": "COMPLETE_M27",
            "source_factor_count": 9,
            "external_matrix_cells": len(external),
            "new_fits": 0,
            "new_forwards": 0,
            "source_authors_by_block": labels.groupby("block")
            .author_id.nunique()
            .to_dict(),
            "meps": meps,
            "inputs_sha256": hashes,
            "code_sha256": {
                name: sha(ROOT / name)
                for name in (
                    "grain_research/vector_mechanisms/factor_catalog.py",
                    "scripts/analyze_grain_all_factor_inventory.py",
                )
            },
            "interpretation": "exploratory source atlas, not new confirmation or mechanism proof",
        },
    )
    print(
        {"status": "COMPLETE_M27", "matrix_cells": len(external), "meps": meps},
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--meps-root", type=Path)
    parser.add_argument("--allow-meps-scored-external", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.allow_meps_scored_external and args.meps_root is None:
        parser.error("Explicit MEPS root required")
    if args.bootstrap < 1:
        parser.error("Positive bootstrap count required")
    with threadpool_limits(2):
        analyze(
            args.output_dir,
            args.meps_root,
            args.allow_meps_scored_external,
            args.bootstrap,
        )
