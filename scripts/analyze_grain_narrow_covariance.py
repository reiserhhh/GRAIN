#!/usr/bin/env python3
"""M31: all narrower query/criterion links and exact matched-author path sharing."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.mood_catalog import COMPOSITES
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_functional_covariance import PARTS, covariance_ledger


def run(source, inventory, mood, out):
    """Keep mean/configuration/final/remainder links and all eligible human targets."""
    if read(source / "summary.json")["status"] != "COMPLETE_NARROW_FUNCTION_TRANSPORT":
        raise ValueError("Complete narrow-function transport first")
    if (
        sha(source / "private_function_values.csv")
        != read(source / "function_freeze.json")["sha256"]
    ):
        raise ValueError("Function values changed")
    data = pd.read_csv(source / "private_function_values.csv")
    data["native_remainder"] = data.native_score - data.score
    ext = pd.read_csv(inventory / "private_meps_scores.csv").pivot(
        index="participant_code", columns="scale", values="score"
    )
    mood_targets = pd.read_csv(mood / "private_mood_scores.csv").pivot(
        index="participant_code", columns="scale", values="score"
    )
    criteria = ext.join(mood_targets, how="inner", validate="one_to_one")
    if (
        len(criteria) != 46
        or len(criteria.columns) != 34
        or criteria.isna().any().any()
    ):
        raise ValueError("Exact46-person/34-criterion numeric join required")
    out.mkdir(parents=True, exist_ok=False)
    all_links, relations, detail = [], [], []
    for view, frame in data.groupby("view"):
        score = frame.pivot(
            index="participant_code", columns="query", values="score"
        ).sort_index()
        native = frame.pivot(
            index="participant_code", columns="query", values="native_score"
        ).reindex(score.index)
        human = criteria.reindex(score.index)
        target = {}
        for q in score:
            k = (
                f"{'post' if view == 'free_chat' else 'mid'}_{q}"
                if q in COMPOSITES
                else q
            )
            target[q] = human[k]
            for criterion in human:
                all_links.append(
                    {
                        "view": view,
                        "query": q,
                        "criterion": criterion,
                        "intended": k == criterion,
                        **association(score[q], human[criterion]),
                    }
                )
        target = pd.DataFrame(target, index=score.index)
        score.corr().to_csv(out / f"{view}_functional_query_matrix.csv")
        native.corr().to_csv(out / f"{view}_native_query_matrix.csv")
        target.corr().to_csv(out / f"{view}_intended_human_matrix.csv")
        for left, right in itertools.combinations(score.columns, 2):
            a = (
                frame[frame["query"] == left]
                .set_index("participant_code")
                .reindex(score.index)
            )
            b = (
                frame[frame["query"] == right]
                .set_index("participant_code")
                .reindex(score.index)
            )
            matrix, total, error = covariance_ledger(a[list(PARTS)], b[list(PARTS)])
            scale = float(np.std(a.native_score) * np.std(b.native_score))
            info = {"view": view, "left": left, "right": right, "n": len(a)}
            relations.append(
                {
                    **info,
                    "native_r": association(a.native_score, b.native_score)[
                        "pearson_r"
                    ],
                    "function_r": association(a.score, b.score)["pearson_r"],
                    "human_r": association(target[left], target[right])["pearson_r"],
                    "native_covariance": total,
                    "accounting_error": error,
                    "configuration_configuration_contribution": float(
                        matrix[1, 1] / scale
                    ),
                    "final_final_contribution": float(matrix[2, 2] / scale),
                }
            )
            for i, j in itertools.product(range(4), repeat=2):
                detail.append(
                    {
                        **info,
                        "left_part": PARTS[i],
                        "right_part": PARTS[j],
                        "covariance": float(matrix[i, j]),
                        "native_r_contribution": float(matrix[i, j] / scale),
                    }
                )
    links, pairs, terms = (
        pd.DataFrame(all_links),
        pd.DataFrame(relations),
        pd.DataFrame(detail),
    )
    links["q_bh_view_all"] = links.groupby("view").p.transform(bh_adjust)
    links.to_csv(out / "all_query_all_criterion_links.csv", index=False)
    pairs.to_csv(out / "cross_query_external_relations.csv", index=False)
    terms.to_csv(out / "signed_path_covariance.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE_NARROW_COVARIANCE_ATLAS",
            "queries": 25,
            "criteria": 34,
            "views": 2,
            "external_links": len(links),
            "query_pairs": len(pairs),
            "signed_terms": len(terms),
            "max_accounting_error": float(pairs.accounting_error.max()),
            "neural_calls": 0,
            "fits": 0,
            "interpretation": "matched-author score covariance, not uniquely identified causal neural paths",
            "overlapping_item_composites": "retained explicitly, not independent replications",
            "inputs": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    source / "private_function_values.csv",
                    inventory / "private_meps_scores.csv",
                    mood / "private_mood_scores.csv",
                )
            },
        },
    )
    print(
        {
            "status": "COMPLETE",
            "external_links": len(links),
            "matched_pairs": len(pairs),
            "signed_terms": len(terms),
            "max_error": float(pairs.accounting_error.max()),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "inventory-dir", "mood-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Use ignored local outputs")
    run(
        a.source_dir.resolve(),
        a.inventory_dir.resolve(),
        a.mood_dir.resolve(),
        a.output_dir.resolve(),
    )
