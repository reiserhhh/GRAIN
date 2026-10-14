#!/usr/bin/env python3
"""WP2c: source-fitted metadata condition addresses and held-author readout.

No topic names or raw texts are exported. Predict a declared textual response
indicator (polar-prefix usage), not personality or exposure to an opportunity.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def group_partition(values):
    """Deterministic author split independent of response labels."""
    return values.map(
        lambda x: int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16) % 10 < 7
    )


def fit_cells(source, keys, global_mean, minimum=100):
    """Learn supported condition cells only on source authors; Beta shrinkage."""
    table = source.groupby(keys).y.agg(["sum", "count"])
    table = table[table["count"] >= minimum]
    table["probability"] = (table["sum"] + 20 * global_mean) / (table["count"] + 20)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cols = [
        "id",
        "author",
        "parent_author",
        "parent_id",
        "subreddit",
        "parent_subreddit",
        "created_utc",
        "body",
        "parent_has_question_mark",
        "other_author",
        "lang",
    ]
    df = pd.read_csv(args.pairs, usecols=cols)
    same = df[~df.other_author & df.lang.eq("en")]
    markers = same.body.fillna("").str.contains(
        r"\b(?:correction|actually|i mean|take that back|edit)\b",
        case=False,
        regex=True,
    )
    revision_candidates = {
        "same_author_parent_reply_pairs": len(same),
        "explicit_marker_candidates": int(markers.sum()),
        "scope": "Discourse markers only; not verified revisions, aliases or event identity.",
    }
    df = df[df.other_author & df.lang.eq("en")].drop_duplicates("id").copy()
    df["y"] = (
        df.body.fillna("")
        .str.match(r"^\s*(yes|no|yeah|yep|nope|nah)\b", case=False)
        .astype(float)
    )
    df["community"] = df.parent_subreddit.fillna("__missing__")
    df["question"] = df.parent_has_question_mark.astype(int)
    source_mask = group_partition(df.author)
    source, held = df[source_mask].copy(), df[~source_mask].copy()
    # Remove shared parents and actors from source: no identical antecedent or
    # held-author text supplies a fitted response cell.
    held_authors = set(held.author)
    source = source[
        ~source.parent_author.isin(held_authors)
        & ~source.parent_id.isin(set(held.parent_id))
    ]
    p0 = float(source.y.mean())
    variants = {}
    joint_prediction = None
    for name, keys in (
        ("community", ["community"]),
        ("community_question", ["community", "question"]),
    ):
        cells = fit_cells(source, keys, p0)
        joined = held.join(cells[["probability"]], on=keys)
        known = joined.probability.notna()
        prediction = joined.probability.fillna(p0).to_numpy()
        variants[name] = {
            "source_cells": len(cells),
            "held_supported_share": float(known.mean()),
            "brier": float(np.mean((prediction - held.y.to_numpy()) ** 2)),
        }
        if name == "community_question":
            joint_prediction = prediction
    held["residual"] = held.y.to_numpy() - joint_prediction
    held["time"] = pd.to_numeric(held.created_utc, errors="coerce")
    ordered = held.dropna(subset=["time"]).sort_values(["author", "time", "id"]).copy()
    ordered["half"] = ordered.groupby("author").cumcount() % 2
    means = ordered.groupby(["author", "half"]).agg(
        n=("y", "size"), raw=("y", "mean"), residual=("residual", "mean")
    )
    valid = means[means.n >= 20].reset_index()
    raw = valid.pivot(index="author", columns="half", values="raw").dropna()
    residual = valid.pivot(index="author", columns="half", values="residual").dropna()

    def reliability(table):
        return float(table[0].corr(table[1])) if len(table) > 2 else None

    summary = {
        "source_rows": len(source),
        "held_rows": len(held),
        "source_authors": source.author.nunique(),
        "held_authors": held.author.nunique(),
        "author_overlap": len(set(source.author) & set(held.author)),
        "parent_overlap": len(set(source.parent_id) & set(held.parent_id)),
        "community_metadata_agreement": float(
            df.subreddit.eq(df.parent_subreddit).mean()
        ),
        "source_polar_rate": p0,
        "held_polar_rate": float(held.y.mean()),
        "global_brier": float(np.mean((held.y - p0) ** 2)),
        "conditions": variants,
        "author_reliability": {
            "n": len(raw),
            "raw": reliability(raw),
            "condition_residual": reliability(residual),
        },
        "ordering": "timestamp then id; alternating halves, NOT cross-day retest",
        "scope": "Observed reply contexts, not full exposure sets. Polar-prefix usage is a textual indicator, not personality.",
        "source_rules": "author hash 70/30; >=100 source rows/cell; prior strength 20 fixed before evaluation",
        "new_neural_calls": 0,
    }
    summary["revision_candidate_inventory"] = revision_candidates
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
