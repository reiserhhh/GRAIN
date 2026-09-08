#!/usr/bin/env python3
"""Cache-only interpretation of frozen question-region matrices and fixed prompt tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_query_region_functional import FULL, PARENT, verify_inputs
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write


def interpret(out):
    """Verify token contraction and summarize known fixed query text, not author text."""
    verify_inputs(out)
    summary = read(out / "summary.json")
    if summary["status"] != "COMPLETE":
        raise ValueError("Complete functional packet required")
    design, original = (
        read(PARENT / "private_design.json"),
        read(FULL / "private_inputs.json"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    tokens = pd.read_csv(out / "private_token_contributions.csv")
    predictions = pd.read_csv(out / "private_predictions.csv")
    contributions, concentration, segments = [], [], []
    for q, group in tokens.groupby("query"):
        ids_by_author = []
        for d in (d for d in design if d["query"] == q):
            entry = next(a for a in original if a["index"] == d["index"])
            ids_by_author.append(
                np.array(entry["queries"][q]["edited_ids"])[d["question_mask"]]
            )
        if any(not np.array_equal(ids_by_author[0], row) for row in ids_by_author):
            raise ValueError("Only exactly shared question tokens may be decoded")
        words = [tokenizer.decode([int(i)]) for i in ids_by_author[0]]
        fixed_text = "".join(words)
        instruction = (
            fixed_text.index("Return only")
            if q == "TF_cont"
            else fixed_text.index("Use 0")
        )
        scaffold = fixed_text.index("<|im_end|>")
        ends = np.cumsum([len(word) for word in words])
        parts = np.where(
            ends <= instruction,
            "semantic_query",
            np.where(ends <= scaffold, "answer_instruction", "chat_scaffold"),
        )
        for position, u in group.groupby("question_token_position"):
            contributions.append(
                {
                    "query": q,
                    "token_position": int(position),
                    "fixed_query_token": words[position],
                    "region": parts[position],
                    "mean_absolute_local_contribution": float(
                        u.local_contribution.abs().mean()
                    ),
                }
            )
        for index, u in group.groupby("index"):
            values = u.sort_values(
                "question_token_position"
            ).local_contribution.to_numpy()
            expected = predictions.query(
                "query == @q and index == @index and method == 'local_simpson'"
            ).predicted
            if len(expected) != 1 or not np.isclose(
                values.sum(), expected.iloc[0], rtol=0, atol=1e-12
            ):
                raise ValueError(
                    "Token contributions do not sum to local matrix response"
                )
            total = np.abs(values).sum()
            concentration.append(
                {
                    "index": int(index),
                    "query": q,
                    "top5_absolute_fraction": float(
                        np.sort(np.abs(values))[-5:].sum() / total
                    ),
                    "net_over_absolute_sum": float(abs(values.sum()) / total),
                }
            )
            for name in ("semantic_query", "answer_instruction", "chat_scaffold"):
                use = parts == name
                segments.append(
                    {
                        "index": int(index),
                        "query": q,
                        "region": name,
                        "signed_sum": float(values[use].sum()),
                        "absolute_share": float(np.abs(values[use]).sum() / total),
                    }
                )
    details = out / "interpretation"
    details.mkdir(exist_ok=True)
    fixed = pd.DataFrame(contributions)
    fixed.to_csv(details / "fixed_query_token_contributions.csv", index=False)
    pd.DataFrame(concentration).to_csv(
        details / "private_concentration.csv", index=False
    )
    pd.DataFrame(segments).to_csv(details / "private_template_regions.csv", index=False)
    region_summary = (
        pd.DataFrame(segments)
        .groupby(["query", "region"])
        .agg(
            mean_absolute_share=("absolute_share", "mean"),
            signed_rms=("signed_sum", lambda x: float(np.sqrt(np.mean(x**2)))),
        )
    )
    region_summary.to_csv(details / "template_region_summary.csv")
    concentration_summary = (
        pd.DataFrame(concentration)
        .groupby("query")[["top5_absolute_fraction", "net_over_absolute_sum"]]
        .mean()
    )
    concentration_summary.to_csv(details / "token_concentration_summary.csv")
    preflight = read(
        BASE / "query_region_functional_v1_endpoint_preflight/run_status.json"
    )
    evaluation = predictions.query("role == 'evaluation' and method == 'frozen_full'")
    result = {
        "status": "COMPLETE_INTERPRETED",
        "neural_calls": 0,
        "interpretation_scope": "Post-result decomposition of measured path contributions; not token ablation or questionnaire importance",
        "total_pairs_including_preflight": preflight["calls"]
        + summary["cost"]["calls"],
        "total_seconds_including_preflight": preflight["seconds"]
        + summary["cost"]["seconds"],
        "eval_signs_correct": int((evaluation.actual * evaluation.predicted > 0).sum()),
        "eval_units": len(evaluation),
        "eval_authors": 2,
        "next_decision": "Freeze full query matrices and test transport on unused source authors, without new evaluation gradients",
        "source_summary_sha256": sha(out / "summary.json"),
        "script_sha256": sha(Path(__file__)),
        "template_regions": region_summary.reset_index().to_dict("records"),
        "concentration": concentration_summary.reset_index().to_dict("records"),
    }
    write(details / "mechanism_decision.json", result)
    print(region_summary.to_string())
    print(
        {
            k: result[k]
            for k in (
                "status",
                "eval_signs_correct",
                "eval_units",
                "total_pairs_including_preflight",
                "total_seconds_including_preflight",
            )
        }
    )


def main():
    """Interpret cached effects only; never call the observer or access labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=BASE / "query_region_functional_v1"
    )
    out = parser.parse_args().source.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError("Keep private interpretation inside ignored results")
    interpret(out)


if __name__ == "__main__":
    main()
