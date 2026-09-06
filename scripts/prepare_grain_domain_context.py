#!/usr/bin/env python3
"""Freeze Essays/X context pairs from outcome-free columns and explicit budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.domain_context import (
    qualify_case,
    salted,
    sextile_spans,
    spread_indices,
    user_parts,
)
from grain_research.vector_mechanisms.response_context import context_derangements
from scripts.prepare_grain_vector_domains import digest, file_sha
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_response_context import EXPECTED_FILES


def main():
    """Prepare a fixed local material manifest, never fit to outcome/embedding values."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--inventory-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private material must stay in ignored local results")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Use a fresh material directory")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoTokenizer

    inventory = json.loads((a.inventory_dir / "manifest.json").read_text())
    if inventory["identity"] != material_hash(
        {k: v for k, v in inventory.items() if k != "identity"}
    ):
        raise ValueError("Unsealed domain inventory")
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        if file_sha(a.model_dir / name) != EXPECTED_FILES[name]:
            raise ValueError("Changed Qwen tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.model_dir), local_files_only=True, trust_remote_code=False
    )
    ep = a.data_root / "prepared/big5/essays_original_prepared.csv"
    xp = (
        a.data_root
        / "x_fullmarkettext/memory_semis_3m_20260318_20260617_merged/x_posts.csv"
    )
    if (
        file_sha(ep) != inventory["source_files"]["essays"]
        or file_sha(xp) != inventory["source_files"]["x_market"]
    ):
        raise ValueError("Source container changed since domain inventory")
    e = pd.read_csv(ep, usecols=["user_id", "text"], dtype={"user_id": str})
    xcols = ["post_id", "account_id", "timestamp", "symbol", "text", "lang"]
    x = pd.read_csv(
        xp, usecols=xcols, dtype={"post_id": str, "account_id": str, "symbol": str}
    )
    exclusions = Counter()
    candidates = []
    for row in e.itertuples(index=False):
        try:
            spans = sextile_spans(row.text)
            case = {
                "corpus": "essays",
                "language": "en",
                "stratum": "essays|en|single_prompt",
                "author_id": row.user_id,
                "context": row.text[: spans[2][1]],
                "current": row.text[spans[5][0] :],
                "context_source_spans": spans[:3],
                "current_source_span": spans[5],
                "gap_source_spans": spans[3:5],
                "source_document_sha256": digest(row.text),
                "source_document_chars": len(row.text),
            }
            candidates.append(qualify_case(case, tokenizer))
        except ValueError:
            exclusions["essays_complete_span_or_token_envelope_rejected"] += 1
    essay_eligible = len(candidates)
    cases = sorted(candidates, key=lambda c: salted(c["author_id"]))[:96]
    exclusions["essays_hash_cap_not_selected"] = max(0, essay_eligible - 96)
    if (
        x.post_id.isna().any()
        or x.account_id.isna().any()
        or x.post_id.duplicated().any()
    ):
        raise ValueError("Unexpected missing or duplicate X identity")
    x["text_hash"] = x.text.fillna("").map(digest)
    copied = set(x.groupby("text_hash").account_id.nunique().loc[lambda v: v > 1].index)
    x["when"] = pd.to_datetime(x.timestamp, utc=True, errors="coerce", format="mixed")
    allowed = (
        x.lang.isin(["en", "ja"])
        & x.text.notna()
        & x.text.str.len().ge(40)
        & x.symbol.notna()
        & x.when.notna()
    )
    allowed &= ~x.text.fillna("").str.match(r"^\s*RT\s+@", case=False)
    allowed &= ~x.text_hash.isin(copied)
    exclusions["x_rows_outside_language_length_time_or_copy_rules"] = int(
        (~allowed).sum()
    )
    x = x.loc[allowed].sort_values(["when", "post_id"], kind="stable")
    before = len(x)
    x = x.drop_duplicates(["account_id", "text_hash"], keep="first")
    exclusions["x_within_account_exact_duplicate_rows"] = before - len(x)
    counts = (
        x.groupby(["account_id", "lang", "symbol"]).size().reset_index(name="count")
    )
    counts = counts[counts["count"] >= 6].copy()
    counts["tie"] = [
        salted(f"{r.account_id}|{r.lang}|{r.symbol}") for r in counts.itertuples()
    ]
    choice = counts.sort_values(
        ["count", "tie"], ascending=[False, True], kind="stable"
    ).drop_duplicates("account_id")
    x_candidates = []
    for row in choice.itertuples(index=False):
        g = x[
            (x.account_id == row.account_id)
            & (x.lang == row.lang)
            & (x.symbol == row.symbol)
        ].sort_values(["when", "post_id"], kind="stable")
        posts = g.iloc[spread_indices(len(g))]
        try:
            case = {
                "corpus": "x_market",
                "language": row.lang,
                "stratum": f"x_market|{row.lang}|{row.symbol}",
                "author_id": row.account_id,
                "context": "\n".join(posts.text.iloc[:3]),
                "current": posts.text.iloc[5],
                "context_post_ids": posts.post_id.iloc[:3].tolist(),
                "current_post_id": posts.post_id.iloc[5],
                "gap_post_ids": posts.post_id.iloc[3:5].tolist(),
                "context_timestamps": posts.timestamp.iloc[:3].tolist(),
                "current_timestamp": posts.timestamp.iloc[5],
                "available_posts": len(g),
            }
            x_candidates.append(qualify_case(case, tokenizer))
        except ValueError:
            exclusions["x_complete_post_or_token_envelope_rejected"] += 1
    cell_counts = Counter(c["stratum"] for c in x_candidates)
    for language in ("en", "ja"):
        selected = 0
        cells = sorted(
            {c["stratum"] for c in x_candidates if c["language"] == language},
            key=salted,
        )
        for cell in cells:
            members = sorted(
                [c for c in x_candidates if c["stratum"] == cell],
                key=lambda c: salted(c["author_id"]),
            )
            if len(members) < 4:
                exclusions["x_candidates_in_cells_below_four"] += len(members)
                continue
            exclusions["x_candidates_above_cell_cap"] += max(0, len(members) - 12)
            members = members[:12]
            if selected + len(members) > 96:
                exclusions["x_candidates_above_language_cap"] += len(members)
                continue
            cases.extend(members)
            selected += len(members)
    cases.sort(key=lambda c: (c["stratum"], salted(c["author_id"])))
    for i, c in enumerate(cases):
        c["case_code"] = f"D{i:04}"
    if len({(c["corpus"], c["author_id"]) for c in cases}) != len(cases):
        raise ValueError("One author appears in multiple frozen cells")
    for cell in sorted({c["stratum"] for c in cases}):
        indices = [i for i, c in enumerate(cases) if c["stratum"] == cell]
        matched, random = context_derangements(
            [cases[i]["context_tokens"] for i in indices]
        )
        for k, index in enumerate(indices):
            case = cases[index]
            case["matched_case_code"] = cases[indices[matched[k]]]["case_code"]
            case["random_case_code"] = cases[indices[random[k]]]["case_code"]
            for donor in (matched[k], random[k]):
                prefix, body, _, _ = user_parts(
                    tokenizer, case["current"], cases[indices[donor]]["context"]
                )
                _, alone, _, _ = user_parts(tokenizer, case["current"])
                if len(prefix) > 768 or body != alone:
                    raise ValueError(
                        "Frozen donor violates full context budget or token identity"
                    )
    write_json(out / "private_cases.json", cases)
    summary = {
        "status": "COMPLETE_DOMAIN_CONTEXT_MATERIAL",
        "inventory_identity": inventory["identity"],
        "source_files": inventory["source_files"],
        "parsed_columns": {"essays": ["user_id", "text"], "x_market": xcols},
        "essay_eligible_before_cap": essay_eligible,
        "x_candidates_before_cell_gates": len(x_candidates),
        "x_eligible_cell_counts": dict(cell_counts),
        "exclusions": dict(exclusions),
        "cases": len(cases),
        "by_corpus_language": dict(
            Counter(c["corpus"] + "|" + c["language"] for c in cases)
        ),
        "by_cell": dict(Counter(c["stratum"] for c in cases)),
        "case_material_sha256": material_hash(cases),
        "code": {
            str(p.relative_to(ROOT)): file_sha(p)
            for p in [
                Path(__file__),
                ROOT / "grain_research/vector_mechanisms/domain_context.py",
                ROOT / "docs/GRAIN_VM06_DOMAIN_CONTEXT_V1.md",
            ]
        },
        "labels_read": False,
        "neural_model_forwards": 0,
    }
    summary["identity"] = material_hash(summary)
    write_json(out / "manifest.json", summary)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"code", "by_cell", "x_eligible_cell_counts"}
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
