#!/usr/bin/env python3
"""Known evidence availability, comment-resampling intervals and blind calibration queue.

Intervals are conditional on fixed detected matches, not confidence intervals for
true personality. This adapter makes that missing uncertainty layer explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.predecessors import TRAITS, keyed_proportions
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write


def sampling_interval(comment_counts, *, draws=200, seed=731):
    """Resample whole comments, preserving sentence dependencies within comments.

    Report unavailable when fewer than four observed comments carry evidence.
    A degenerate interval is labeled explicitly, not treated as perfect accuracy.
    """
    c = np.asarray(comment_counts, float)
    if c.ndim != 3 or c.shape[2] != 2 or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("Expected nonnegative comment x trait x polarity counts")
    if len(c) < 1:
        raise ValueError("At least one comment is required")
    rng = np.random.default_rng(seed)
    totals = c[rng.integers(0, len(c), (draws, len(c)))].sum(1)
    scores = keyed_proportions(totals)
    estimates, rows = keyed_proportions(c.sum(0)), []
    for j in range(c.shape[1]):
        evidence_comments = int(np.sum(c[:, j].sum(1) > 0))
        observed = scores[:, j][np.isfinite(scores[:, j])]
        lower, upper, status = None, None, "insufficient_evidence_comments"
        if evidence_comments >= 4 and len(observed) >= int(0.95 * draws):
            lower, upper = np.quantile(observed, [0.025, 0.975]).tolist()
            status = (
                "conditional_empirical_interval"
                if upper > lower
                else "degenerate_empirical_interval"
            )
        rows.append(
            {
                "trait_index": j,
                "evidence_comments": evidence_comments,
                "point": float(estimates[j]) if np.isfinite(estimates[j]) else None,
                "resampling_lower": lower,
                "resampling_upper": upper,
                "available_draw_fraction": len(observed) / draws,
                "status": status,
            }
        )
    return rows


def run(args):
    """Use only an existing source cache and fixed matches, without loading Big5 Y."""
    out = args.output
    if out.exists():
        raise ValueError("Use a fresh output directory")
    out.mkdir(parents=True)
    data = json.loads((args.material / "private_material.json").read_text())
    contract = json.loads((args.paraphrase / "run_contract.json").read_text())
    if contract["source_manifest"] != digest(args.material / "material_manifest.json"):
        raise ValueError("Material mismatch")
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    if (
        digest(args.material / "private_material.json")
        != manifest["files"]["private_material.json"]
    ):
        raise ValueError("Source data checksum mismatch")
    items = json.loads((args.material / "ipip_neo300.json").read_text())["items"]
    evidence = pd.read_csv(args.paraphrase / "private_sentence_evidence.csv")
    n = len(data["bundles"])
    counts = np.zeros((n, 36, len(TRAITS), 2))
    for row in evidence.itertuples():
        if row.accepted:
            counts[
                row.author_index,
                row.comment_index,
                TRAITS.index(row.trait),
                0 if row.key == 1 else 1,
            ] += 1
    records = []
    for author, c in enumerate(counts):
        records.extend(
            {"author_index": author, "trait": TRAITS[r["trait_index"]], **r}
            for r in sampling_interval(c, seed=731 + author)
        )
    frame = pd.DataFrame(records)
    frame.to_csv(out / "private_comment_resampling.csv", index=False)
    group = frame.groupby("trait", sort=False)
    group.agg(
        authors=("author_index", "size"),
        available_points=("point", "count"),
        median_evidence_comments=("evidence_comments", "median"),
    ).to_csv(out / "availability_by_trait.csv")
    # The earlier uniform packet mostly sampled rejected pairs. Retain it; add a
    # score-stratified discovery-only queue with inclusion probabilities recorded.
    folds = [b["official_folds"]["big5"] for b in data["bundles"]]
    evidence["source_side"] = [folds[int(i)] != 0 for i in evidence.author_index]
    strata = pd.cut(
        evidence.similarity,
        [-1.001, 0.35, 0.5, 0.6, 1.001],
        labels=["low", "middle", "near", "accepted"],
        right=False,
    )
    rng, selected, mapping = (
        np.random.default_rng(901),
        [],
        {item["id"]: item for item in items},
    )
    allocation = []
    for category in ("low", "middle", "near", "accepted"):
        pool = np.flatnonzero(evidence.source_side & (strata == category))
        chosen = rng.choice(pool, size=min(20, len(pool)), replace=False)
        allocation.append(
            {"stratum": category, "available": len(pool), "sampled": len(chosen)}
        )
        for i in chosen:
            record, s = evidence.iloc[i], data["sentences"][int(i)]
            selected.append(
                {
                    "audit_id": f"PAIR_{int(i):06d}",
                    "sentence": s["text"],
                    "item_id": record.item_id,
                    "item_text": mapping[record.item_id]["text"],
                    "self_attribution": "",
                    "simpa_category_1_to_7": "",
                    "rater": "",
                    "rater_kind": "",
                }
            )
    rng.shuffle(selected)
    pd.DataFrame(selected).to_csv(
        out / "private_stratified_annotation_packet.csv", index=False
    )
    # Sampler metadata stays separate so annotators do not see retrieval confidence.
    evidence.to_csv(out / "private_annotation_sampling_frame.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "authors": n,
        "total_matched_sentences": int(counts.sum()),
        "authors_with_any_match": int(np.sum(counts.sum((1, 2, 3)) > 0)),
        "score_status_counts": frame.status.value_counts().to_dict(),
        "annotation_allocation": allocation,
        "personality_labels_loaded": False,
        "uncertainty_scope": "Comment resampling of observed assignments only; no detector validity, author-reference uncertainty or calendar-time dependence propagated.",
        "annotation_status": "NOT_ANNOTATED; no claim of human or model-rated precision",
        "code_sha256": digest(Path(__file__)),
        "material_sha256": digest(args.material / "material_manifest.json"),
        "evidence_sha256": digest(args.paraphrase / "private_sentence_evidence.csv"),
    }
    write(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Inspect existing matches, never restart any training queue."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--paraphrase",
        type=Path,
        default=BASE / "predecessor_integration_v1/paraphrase_r1",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=BASE / "predecessor_integration_v1/evidence_sampling_r1",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
