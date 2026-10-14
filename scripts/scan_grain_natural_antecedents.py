#!/usr/bin/env python3
"""Scan the licensed local PANDORA comment file for in-corpus antecedent turns.

Three streaming passes over the raw CSV: (1) collect all comment ids; (2) find
comments whose parent comment is itself in the corpus; (3) fetch those parent
rows. Writes aggregate counts (public) and a local private pair file that stays
inside the gitignored results tree. No user text or identifier is printed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT.parent / "project persona" / "data_sets" / "PANDORA_official" / "all_comments_since_2015.csv"
QUESTION = re.compile(r"\?")


def encode(series):
    """Base-36 comment ids fit in uint64 (Reddit ids are <= 7 characters)."""
    out = np.zeros(len(series), dtype=np.uint64)
    for i, value in enumerate(series):
        try:
            out[i] = int(str(value), 36)
        except (TypeError, ValueError):
            out[i] = 0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.time()
    ids = []
    total = 0
    for chunk in pd.read_csv(CSV, usecols=["id"], chunksize=args.chunksize, nrows=args.max_rows, dtype=str):
        ids.append(encode(chunk["id"].to_numpy()))
        total += len(chunk)
    ids = np.sort(np.concatenate(ids))
    pass1 = time.time() - start
    children, parent_wanted = [], []
    stats = {"rows": total, "parent_is_comment": 0, "parent_in_corpus": 0}
    cols = ["id", "author", "parent_id", "subreddit", "created_utc", "body", "word_count", "lang"]
    for chunk in pd.read_csv(CSV, usecols=cols, chunksize=args.chunksize, nrows=args.max_rows, dtype={"id": str, "author": str, "parent_id": str, "subreddit": str, "body": str, "lang": str}):
        is_comment = chunk["parent_id"].astype(str).str.startswith("t1_")
        stats["parent_is_comment"] += int(is_comment.sum())
        sub = chunk[is_comment]
        codes = encode(sub["parent_id"].astype(str).str[3:].to_numpy())
        pos = np.searchsorted(ids, codes)
        pos[pos >= len(ids)] = 0
        hit = ids[pos] == codes
        stats["parent_in_corpus"] += int(hit.sum())
        matched = sub[hit].copy()
        matched["parent_code"] = codes[hit]
        children.append(matched)
        parent_wanted.append(codes[hit])
    children = pd.concat(children, ignore_index=True) if children else pd.DataFrame(columns=cols)
    wanted = np.unique(np.concatenate(parent_wanted)) if parent_wanted else np.array([], dtype=np.uint64)
    pass2 = time.time() - start - pass1
    parents = []
    for chunk in pd.read_csv(CSV, usecols=["id", "author", "body", "subreddit", "word_count"], chunksize=args.chunksize, nrows=args.max_rows, dtype=str):
        codes = encode(chunk["id"].to_numpy())
        pos = np.searchsorted(wanted, codes)
        pos[pos >= len(wanted)] = 0
        hit = wanted[pos] == codes if len(wanted) else np.zeros(len(codes), bool)
        part = chunk[hit].copy()
        part["parent_code"] = codes[hit]
        parents.append(part)
    parents = pd.concat(parents, ignore_index=True) if parents else pd.DataFrame()
    parents = parents.rename(columns={"id": "parent_id_raw", "author": "parent_author", "body": "parent_body", "subreddit": "parent_subreddit", "word_count": "parent_word_count"})
    duplicates = {"child_rows_dropped": int(children.duplicated("id").sum()), "parent_rows_dropped": int(parents.duplicated("parent_code").sum())}
    children = children.drop_duplicates("id")
    parents = parents.drop_duplicates("parent_code")
    pairs = children.merge(parents, on="parent_code", how="inner", validate="many_to_one")
    pairs["other_author"] = pairs["author"] != pairs["parent_author"]
    pairs["parent_has_question_mark"] = pairs["parent_body"].astype(str).str.contains(QUESTION)
    pairs["parent_word_count"] = pd.to_numeric(pairs["parent_word_count"], errors="coerce")
    pairs["word_count"] = pd.to_numeric(pairs["word_count"], errors="coerce")
    private = args.output / "private_antecedent_pairs.csv"
    pairs.to_csv(private, index=False)
    other = pairs[pairs.other_author]
    summary = {
        **stats,
        "duplicate_rows_in_corpus": duplicates,
        "unique_comment_ids": len(ids),
        "pairs_with_parent_row": len(pairs),
        "pairs_other_author": len(other),
        "pairs_same_author": int((~pairs.other_author).sum()),
        "other_author_parent_has_question_mark": int(other.parent_has_question_mark.sum()),
        "other_author_parent_question_short": int((other.parent_has_question_mark & (other.parent_word_count <= 60)).sum()),
        "other_author_reply_short": int((other.word_count <= 60).sum()),
        "other_author_question_and_short_reply": int((other.parent_has_question_mark & (other.parent_word_count <= 60) & (other.word_count <= 60)).sum()),
        "distinct_child_authors_other": int(other.author.nunique()),
        "distinct_parent_authors_other": int(other.parent_author.nunique()),
        "english_share_other": float((other.lang == "en").mean()) if len(other) else None,
        "top_subreddits_other_counts_only": other.subreddit.value_counts().head(15).tolist() if len(other) else [],
        "seconds": {"pass1_ids": pass1, "pass2_children": pass2, "total": time.time() - start},
        "private_file": str(private.resolve().relative_to(ROOT)),
        "scope": "structural counts over the licensed local corpus; no text or user identifiers exported",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
