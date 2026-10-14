#!/usr/bin/env python3
"""WP3: observable opportunity share per author from thread structure (label-free).

For each author, the share of comments that reply to another author's question
(parent comment contains a question mark) is an observed opportunity mix. The
script streams author totals from the licensed CSV, joins the antecedent pairs,
splits each author's comments by within-author ordinal parity and reports the
split-half agreement of the share. Only anonymised aggregates are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT.parent / "project persona" / "data_sets" / "PANDORA_official" / "all_comments_since_2015.csv"


def spearman_brown(r):
    return 2 * r / (1 + r) if r is not None and r > -1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-comments", type=int, default=50)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    # Pass: per-author totals and per-author parity halves by within-author ordinal (file order).
    totals, half_totals = {}, {}
    seen = {}
    for chunk in pd.read_csv(CSV, usecols=["author", "id", "parent_id"], chunksize=args.chunksize, dtype=str):
        chunk = chunk.drop_duplicates("id")
        for author, group in chunk.groupby("author", sort=False):
            n0 = seen.get(author, 0)
            k = len(group)
            totals[author] = totals.get(author, 0) + k
            idx = np.arange(n0, n0 + k)
            half_totals.setdefault(author, [0, 0])
            half_totals[author][0] += int((idx % 2 == 0).sum())
            half_totals[author][1] += int((idx % 2 == 1).sum())
            seen[author] = n0 + k
    pairs = pd.read_csv(args.pairs, usecols=["id", "author", "parent_author", "parent_has_question_mark", "other_author"])
    pairs = pairs.drop_duplicates("id")
    asked = pairs[pairs.other_author & pairs.parent_has_question_mark]
    # Simpler and exact: recompute ordinals in one pass with cumulative counters.
    counters, ordinal = {}, {}
    asked_ids = set(asked.id)
    for chunk in pd.read_csv(CSV, usecols=["author", "id"], chunksize=args.chunksize, dtype=str):
        chunk = chunk.drop_duplicates("id")
        for author, group in chunk.groupby("author", sort=False):
            base = counters.get(author, 0)
            for i, cid in enumerate(group.id):
                if cid in asked_ids:
                    ordinal[cid] = base + i
            counters[author] = base + len(group)
    asked = asked.assign(ordinal=asked.id.map(ordinal))
    asked = asked.dropna(subset=["ordinal"])
    asked["half"] = (asked.ordinal.astype(int) % 2)
    per = asked.groupby(["author", "half"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    rows = []
    for author, (h0, h1) in per.iterrows():
        t0, t1 = half_totals.get(author, [0, 0])
        if t0 + t1 < args.min_comments:
            continue
        rows.append({
            "author_code": hashlib.sha256(("wp3:" + author).encode()).hexdigest()[:12],
            "comments": t0 + t1, "share_half0": h0 / t0 if t0 else np.nan, "share_half1": h1 / t1 if t1 else np.nan,
            "share_all": (h0 + h1) / (t0 + t1),
        })
    table = pd.DataFrame(rows).dropna()
    r_pearson = float(np.corrcoef(table.share_half0, table.share_half1)[0, 1]) if len(table) > 2 else None
    r_spearman = float(stats.spearmanr(table.share_half0, table.share_half1).statistic) if len(table) > 2 else None
    rho_volume = float(stats.spearmanr(table.share_all, table.comments).statistic) if len(table) > 2 else None
    rng = np.random.default_rng(20260914)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, len(table), len(table))
        boot.append(np.corrcoef(table.share_half0.to_numpy()[idx], table.share_half1.to_numpy()[idx])[0, 1])
    summary = {
        "authors_with_min_comments": len(table),
        "min_comments": args.min_comments,
        "asked_replies_total": len(asked),
        "share_all_mean": float(table.share_all.mean()), "share_all_sd": float(table.share_all.sd() if hasattr(table.share_all, 'sd') else table.share_all.std()),
        "split_half_pearson": r_pearson, "split_half_pearson_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "split_half_spearman": r_spearman,
        "spearman_brown_full_length": spearman_brown(r_pearson),
        "share_vs_log_volume_spearman": rho_volume,
        "note": "share = replies to another PANDORA author's question-mark comment / all comments; observed opportunity mix, not a psychological trait; parity split is by within-author file order, not calendar time",
    }
    table.to_csv(args.output / "author_share_anonymised.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
