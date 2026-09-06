#!/usr/bin/env python3
"""Inventory current Essays/X text for domain mechanisms without outcome labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_meps_preflight import tokenizer_setup


def digest(value):
    """Stable text fingerprint for local duplicate auditing, not a public ID."""
    return hashlib.sha256(str(value).encode()).hexdigest()


def file_sha(path):
    """Hash a source container; hashing bytes is not selective column parsing."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def text_summary(frame, *, author, text):
    """Profile legitimate rows and cross-author exact-copy overlap without prints."""
    valid = (
        frame[author].notna()
        & frame[author].astype(str).str.strip().ne("")
        & frame[text].notna()
        & frame[text].astype(str).str.strip().ne("")
    )
    f = frame.loc[valid, [author, text]].copy()
    f["text_hash"] = f[text].map(digest)
    counts = f.groupby(author).size()
    chars = f[text].str.len()
    copies = f.groupby("text_hash")[author].nunique()
    return {
        "rows": len(frame),
        "valid_text_rows": len(f),
        "authors": int(f[author].nunique()),
        "missing_author_rows": int(frame[author].isna().sum()),
        "invalid_author_or_text_rows": int((~valid).sum()),
        "rows_per_author_min_median_max": [
            int(counts.min()),
            float(counts.median()),
            int(counts.max()),
        ],
        "text_chars_min_median_max": [
            int(chars.min()),
            float(chars.median()),
            int(chars.max()),
        ],
        "total_chars": int(chars.sum()),
        "cross_author_shared_exact_texts": int((copies > 1).sum()),
    }


def main():
    """Save a bounded tokenizer preflight, not a neural domain-sensitivity result."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Keep row-level domain evidence in the ignored results tree")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Use a fresh domain inventory directory")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    essays = a.data_root / "prepared/big5/essays_original_prepared.csv"
    x_path = (
        a.data_root
        / "x_fullmarkettext/memory_semis_3m_20260318_20260617_merged/x_posts.csv"
    )
    ecols = ["user_id", "text"]
    xcols = [
        "post_id",
        "account_id",
        "timestamp",
        "symbol",
        "text",
        "lang",
        "query_group",
        "query_language",
    ]
    manifest = {
        "status": "FROZEN_INVENTORY_BEFORE_PARSING",
        "source_files": {"essays": file_sha(essays), "x_market": file_sha(x_path)},
        "parsed_columns": {"essays": ecols, "x_market": xcols},
        "psychological_or_market_outcomes_parsed": False,
        "source_containers_fully_hashed": True,
        "sample_rule": "first 200 distinct record IDs by sha256(domain-preflight-v1:ID), separately per reported language; exclude conflicted IDs",
        "author_cross_corpus_pairing": False,
        "neural_model_forwards": 0,
        "code_sha256": file_sha(Path(__file__)),
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    e = pd.read_csv(essays, usecols=ecols, dtype={"user_id": str})
    x = pd.read_csv(x_path, usecols=xcols, dtype={"account_id": str, "post_id": str})
    er = text_summary(e, author="user_id", text="text")
    er.update(
        single_document_is_not_longitudinal_retest=True,
        independently_observed_sessions=False,
        author_id_duplicates=int(e.user_id.duplicated().sum()),
    )
    xr = text_summary(x, author="account_id", text="text")
    x["text_sha256"] = x.text.fillna("").map(digest)
    identity_groups = x.groupby("post_id").agg(
        authors=("account_id", "nunique"), texts=("text_sha256", "nunique")
    )
    conflicted = set(
        identity_groups.index[
            (identity_groups.authors > 1) | (identity_groups.texts > 1)
        ]
    )
    canonical = (
        x[
            ~x.post_id.isin(conflicted)
            & x.post_id.notna()
            & x.account_id.notna()
            & x.text.notna()
        ]
        .drop_duplicates("post_id")
        .copy()
    )
    times = pd.to_datetime(
        canonical.timestamp, errors="coerce", utc=True, format="mixed"
    )
    lang = canonical.lang.fillna("<missing>").astype(str)
    canonical["reported_language"] = lang
    language_rows = []
    for language, g in canonical.groupby("reported_language", sort=True):
        counts = g.groupby("account_id").size()
        language_rows.append(
            {
                "reported_language": language,
                "distinct_posts": len(g),
                "authors": int(g.account_id.nunique()),
                "authors_at_least_2_posts": int((counts >= 2).sum()),
                "authors_at_least_6_posts": int((counts >= 6).sum()),
                "authors_at_least_12_posts": int((counts >= 12).sum()),
                "median_characters": float(g.text.str.len().median()),
            }
        )
    xr.update(
        unique_post_ids=int(x.post_id.nunique()),
        duplicate_post_id_rows=int(x.post_id.duplicated().sum()),
        conflicted_post_ids=len(conflicted),
        canonical_posts=len(canonical),
        canonical_authors=int(canonical.account_id.nunique()),
        invalid_timestamps=int(times.isna().sum()),
        timestamps_with_explicit_zone=int(
            canonical.timestamp.fillna("")
            .str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
            .sum()
        ),
        timestamp_min=times.min().isoformat(),
        timestamp_max=times.max().isoformat(),
        query_groups=int(x.query_group.nunique()),
        symbols=int(x.symbol.nunique()),
        collector_strata_not_causal_author_choices=True,
        language_is_source_metadata_not_validated=True,
    )
    pd.DataFrame(language_rows).to_csv(out / "language_inventory.csv", index=False)
    tokenizers, specs = tokenizer_setup(Path.home() / ".cache/huggingface/hub")
    selections = []
    evalid = e[e.text.notna() & e.user_id.notna()].copy()
    if evalid.user_id.duplicated().any():
        raise ValueError("Essays requires explicit duplicate author handling")
    evalid["rank"] = evalid.user_id.map(lambda s: digest("domain-preflight-v1:" + s))
    for r in evalid.sort_values("rank").head(200).itertuples():
        selections.append(("essays", "en_dataset_description", r.user_id, r.text))
    canonical["rank"] = canonical.post_id.map(
        lambda s: digest("domain-preflight-v1:" + s)
    )
    for language, g in canonical.groupby("reported_language", sort=True):
        for r in g.sort_values("rank").head(200).itertuples():
            if r.text.strip():
                selections.append(("x_market", language, r.post_id, r.text))
    rows = []
    for corpus, language, record, text in selections:
        for model, (tokenizer, prefix) in tokenizers.items():
            ids = tokenizer(prefix + text, truncation=False, verbose=False)["input_ids"]
            unknown = sum(v == tokenizer.unk_token_id for v in ids)
            content = sum(
                v not in set(tokenizer.all_special_ids) or v == tokenizer.unk_token_id
                for v in ids
            )
            rows.append(
                {
                    "corpus": corpus,
                    "reported_language": language,
                    "record_sha256": digest(record),
                    "model": model,
                    "chars": len(text),
                    "tokens": len(ids),
                    "unknown_tokens": unknown,
                    "content_tokens": content,
                    "within_512": len(ids) <= 512,
                }
            )
    tokens = pd.DataFrame(rows)
    tokens.to_csv(out / "private_token_audit.csv", index=False)
    aggregate = (
        tokens.groupby(["corpus", "reported_language", "model"])
        .agg(
            records=("tokens", "size"),
            median_tokens=("tokens", "median"),
            max_tokens=("tokens", "max"),
            within_512=("within_512", "sum"),
            unknown_tokens=("unknown_tokens", "sum"),
            content_tokens=("content_tokens", "sum"),
        )
        .reset_index()
    )
    aggregate["unknown_fraction"] = aggregate.unknown_tokens / aggregate.content_tokens
    aggregate.to_csv(out / "domain_tokenizer_coverage.csv", index=False)
    result = {
        "status": "COMPLETE_DOMAIN_INVENTORY_ONLY",
        "identity": manifest["identity"],
        "essays": er,
        "x_market": xr,
        "tokenizer_inputs": len(rows),
        "sampled_records": len(selections),
        "tokenizers": specs,
        "model_forwards": 0,
        "psychological_or_market_outcomes_parsed": False,
        "new_neural_domain_comparison_complete": False,
    }
    write_json(out / "summary.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "tokenizers"}), flush=True
    )


if __name__ == "__main__":
    main()
