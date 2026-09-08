#!/usr/bin/env python3
"""Cache-only fixed-token and multi-query interpretation of the full-score readout."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.level_response import correlation
from scripts.run_grain_level_response import verify
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write


def main():
    """Attribute to fixed template coordinates, not author words or causal personality."""
    out = BASE / "complete_level_response_v1"
    manifest = verify(out)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    with np.load(out / "reference.npz", allow_pickle=False) as a:
        ref = dict(a)
    templates = {}
    for q, ids in manifest["question_tokens"].items():
        tokens = [tokenizer.decode([i]) for i in ids]
        text = "".join(tokens)
        instruction = text.index("Return only" if q == "TF_cont" else "Use 0")
        scaffold = text.index("<|im_end|>")
        ends = np.cumsum([len(t) for t in tokens])
        templates[q] = np.where(
            ends <= instruction,
            "semantic_query",
            np.where(ends <= scaffold, "answer_instruction", "chat_scaffold"),
        )
    rows = []
    for d in read(out / "private_design.json"):
        q, i = d["query"], d["index"]
        path = out / "private_packets" / f"unit_{i:02d}_{q}.npz"
        with np.load(path, allow_pickle=False) as a:
            full, edited = a["full_P"].astype(float), a["edited_P"].astype(float)
        for target, delta in (
            ("level", full - ref[f"{q}__Z"]),
            ("natural", edited - full),
        ):
            contributions = np.sum(ref[f"{q}__G"] * delta, axis=1)
            absolute = np.abs(contributions).sum()
            for part in ("semantic_query", "answer_instruction", "chat_scaffold"):
                use = templates[q] == part
                rows.append(
                    {
                        "index": i,
                        "query": q,
                        "target": target,
                        "region": part,
                        "signed_contribution": float(contributions[use].sum()),
                        "absolute_share": float(
                            np.abs(contributions[use]).sum() / absolute
                        )
                        if absolute
                        else 0.0,
                    }
                )
    segments = pd.DataFrame(rows)
    segments.to_csv(out / "private_full_template_contributions.csv", index=False)
    summary = (
        segments.groupby(["query", "target", "region"])
        .agg(
            mean_absolute_share=("absolute_share", "mean"),
            mean_signed_contribution=("signed_contribution", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out / "full_template_region_summary.csv", index=False)
    predicted = pd.read_csv(out / "private_predictions.csv").query("method == 'PQ'")
    shared = []
    for target in ("level", "natural"):
        group = predicted[predicted.target == target]
        for method, col in (("native", "actual"), ("constructed", "predicted")):
            values = group.pivot(index="index", columns="query", values=col)
            for left, right in itertools.combinations(values.columns, 2):
                shared.append(
                    {
                        "target": target,
                        "method": method,
                        "query1": left,
                        "query2": right,
                        "authors": len(values),
                        "r": correlation(
                            values[left].to_numpy(), values[right].to_numpy()
                        ),
                    }
                )
    pd.DataFrame(shared).to_csv(out / "cross_query_readout.csv", index=False)
    write(
        out / "interpretation_summary.json",
        {
            "status": "COMPLETE",
            "neural_calls": 0,
            "scope": "coordinate attribution and same-author query covariance; not semantic causal fractions",
            "script_sha256": sha(Path(__file__)),
            "regions": summary.to_dict("records"),
            "cross_query": shared,
        },
    )
    print(summary.to_string(index=False))
    print(pd.DataFrame(shared).to_string(index=False))


if __name__ == "__main__":
    main()
