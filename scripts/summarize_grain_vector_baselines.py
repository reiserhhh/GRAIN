#!/usr/bin/env python3
"""Source/length/semantic-neighbor sensitivity for the fixed comment comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.comparison import (
    conditional_rank,
    node_bootstrap_weights,
    pair_concordance,
)
from grain_research.vector_mechanisms.controls import material_hash
from scripts.run_grain_vector_mechanism_controls import write_json


def condition_masks(rows, authors, vectors):
    """Keep all channels; only change impostor eligibility, without target labels."""
    n = len(authors)
    by_key = {(r["author_id"], r["history_index"]): r for r in rows}
    masks = {
        k: []
        for k in (
            "all",
            "cross_source_positive",
            "same_target_source",
            "length_ratio_1.25",
            "bge_target_neighbors_5",
            "e5_target_neighbors_5",
            "no_clue_author",
        )
    }
    clean = np.array(
        [
            not any(any(by_key[u, i]["leakage_flags"].values()) for i in range(3))
            for u in authors
        ]
    )
    for a, b in ((0, 1), (0, 2), (1, 2)):
        target = [by_key[u, b] for u in authors]
        source_a = np.array([by_key[u, a]["subreddit"] for u in authors])
        source_b = np.array([r["subreddit"] for r in target])
        off = ~np.eye(n, dtype=bool)
        masks["all"].append(off)
        masks["cross_source_positive"].append(off & (source_a != source_b)[:, None])
        masks["same_target_source"].append(
            off & (source_b[:, None] == source_b[None, :])
        )
        lengths = np.array([r["word_count"] for r in target])
        masks["length_ratio_1.25"].append(
            off & (abs(np.log(lengths[:, None] / lengths)) <= np.log(1.25))
        )
        masks["no_clue_author"].append(off & clean[:, None] & clean[None, :])
        for encoder in ("bge", "e5"):
            h = vectors[f"{encoder}_large"][:, b].astype(float)
            h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
            sim = h @ h.T
            np.fill_diagonal(sim, -np.inf)
            neighbors = np.argsort(-sim, axis=1, kind="stable")[:, :5]
            mask = np.zeros((n, n), dtype=bool)
            np.put_along_axis(mask, neighbors, True, axis=1)
            masks[f"{encoder}_target_neighbors_5"].append(mask)
    return {k: np.stack(v) for k, v in masks.items()}


def main():
    """Report coverage and conditional paired changes, not causal nuisance removal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_status.json").exists():
        raise ValueError("Use a new output directory")
    rows = json.loads((args.parent_dir / "input_ledger.json").read_text())
    rows = [r for r in rows if r["split"] == "pilot"]
    authors = sorted({r["author_id"] for r in rows})
    with np.load(args.parent_dir / "comparison_vectors.npz") as data:
        vectors = {k: data[k] for k in data.files}
    masks = condition_masks(rows, authors, vectors)
    weights = node_bootstrap_weights(len(authors))
    results, coverage, deltas = [], [], []
    for name, mask in masks.items():
        active = mask.sum(axis=(0, 2)) > 0
        coverage.append(
            {
                "condition": name,
                "edges": int(mask.sum()),
                "authors_with_comparison": int(active.sum()),
                "author_views": int((mask.sum(axis=2) > 0).sum()),
                "scope": "semantic neighbor arms condition on embeddings, not pure topic",
            }
        )
        bootstrap = {}
        points = {}
        for key, values in vectors.items():
            c = pair_concordance(values)
            point = conditional_rank(c, mask, np.ones(len(authors)))[0]
            boot = conditional_rank(c, mask, weights)
            valid = boot[np.isfinite(boot)]
            interval_ok = len(valid) > 0 and active.sum() >= 8
            results.append(
                {
                    "condition": name,
                    "method": key,
                    "author_rank_auc": point,
                    "ci_low": np.quantile(valid, 0.025) if interval_ok else None,
                    "ci_high": np.quantile(valid, 0.975) if interval_ok else None,
                    "valid_bootstraps": len(valid),
                    "status": "DESCRIPTIVE_SPARSE"
                    if active.sum() < 8
                    else "EXPLORATORY",
                }
            )
            bootstrap[key], points[key] = boot, point
        for baseline in ("bge_large", "e5_large", "tfidf_word_reference_only"):
            key = "qwen_block_10__history_mean"
            delta = bootstrap[key] - bootstrap[baseline]
            valid = delta[np.isfinite(delta)]
            enough = len(valid) > 0 and active.sum() >= 8
            deltas.append(
                {
                    "condition": name,
                    "candidate": key,
                    "baseline": baseline,
                    "delta": points[key] - points[baseline],
                    "ci_low": np.quantile(valid, 0.025) if enough else None,
                    "ci_high": np.quantile(valid, 0.975) if enough else None,
                }
            )
    pd.DataFrame(results).to_csv(output / "conditional_comparison.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(deltas).to_csv(output / "paired_differences.csv", index=False)
    write_json(
        output / "run_status.json",
        {
            "status": "COMPLETE",
            "authors": len(authors),
            "labels_loaded": False,
            "parent_manifest_sha256": material_hash(
                json.loads((args.parent_dir / "manifest.json").read_text())
            ),
        },
    )
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__ == "__main__":
    main()
