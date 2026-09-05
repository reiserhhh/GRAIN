#!/usr/bin/env python3
"""Recompute context effects without questionnaires or hidden-state model fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import (
    ARMS,
    QUESTIONS,
    anchored_decomposition,
    paired_author_rank,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def summarize(parent, output, views_for_leakage_filter=None):
    """Separate shared-history similarity from anchor-centered interaction geometry."""
    status = json.loads((parent / "run_status.json").read_text())
    if status["status"] != "COMPLETE":
        raise ValueError("Cannot summarize an incomplete run as a full pilot")
    frame = pd.read_csv(parent / "context_predictions.csv")
    manifest = json.loads((parent / "manifest.json").read_text())
    excluded_authors = set()
    if views_for_leakage_filter is not None:
        bundles = json.loads(views_for_leakage_filter.read_text(encoding="utf-8"))
        if material_hash(bundles) != manifest["source_view_sha256"]:
            raise ValueError("Leakage sensitivity views do not match parent material")
        flagged = {
            b["author_id"]
            for b in bundles
            if any(
                any(c["leakage_flags"].values()) for c in b["history"] + b["current"]
            )
        }
        excluded_authors = set(
            frame.loc[
                frame.author_id.isin(flagged) | frame.donor_id.isin(flagged),
                "author_id",
            ]
        )
        frame = frame[~frame.author_id.isin(excluded_authors)]
    anchor_only = manifest.get("current_design") == "two_shared_synthetic_anchors"
    independent_history = (
        manifest.get("history_design") == "three_nonoverlapping_single_comments"
    )
    arms = manifest.get("arms", ARMS)
    if frame.case_id.duplicated().any():
        raise ValueError("Duplicate observations")
    authors = sorted(frame.author_id.unique())
    current_indices = (0, 1) if anchor_only else (0, 1, 2)
    expected = len(authors) * len(current_indices) * len(arms) * len(QUESTIONS)
    if len(frame) != expected:
        raise ValueError("Incomplete author-by-current-by-arm-by-task grid")
    data, meta = {}, {}
    for row in frame.to_dict("records"):
        key = (row["author_id"], row["current_index"], row["arm"], row["task"])
        with np.load(parent / "private_cache" / f"{row['case_id']}.npz") as archive:
            data[key] = {k: archive[k].astype(np.float64) for k in archive.files}
        meta[key] = row
    features = sorted(next(iter(data.values())))
    checks = {
        "pre_question_task_max_abs": 0.0,
        "input_content_across_history_max_abs": 0.0,
    }
    for uid in authors:
        for i in current_indices:
            for arm in arms:
                a, b = (
                    data[(uid, i, arm, "behavior")],
                    data[(uid, i, arm, "disposition")],
                )
                for feature in features:
                    if not feature.endswith("__decision"):
                        checks["pre_question_task_max_abs"] = max(
                            checks["pre_question_task_max_abs"],
                            float(np.max(np.abs(a[feature] - b[feature]))),
                        )
                    if feature.startswith("input_embedding") and not feature.endswith(
                        "__history_mean"
                    ):
                        raw = data[(uid, i, "none", "behavior")][feature]
                        checks["input_content_across_history_max_abs"] = max(
                            checks["input_content_across_history_max_abs"],
                            float(np.max(np.abs(a[feature] - raw))),
                        )
    if max(checks.values()) > 5e-4:
        raise ValueError(f"Causal/input invariance check failed: {checks}")
    if anchor_only:
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        none_delta = 0.0
        for feature in features:
            for task in QUESTIONS:
                for arm in arms:
                    first, second = [
                        np.stack(
                            [data[(uid, i, arm, task)][feature] for uid in authors]
                        )
                        for i in (0, 1)
                    ]
                    if arm == "none":
                        none_delta = max(
                            none_delta,
                            float(np.max(abs(first - first[0]))),
                            float(np.max(abs(second - second[0]))),
                        )
                    rank = paired_author_rank(first, second)
                    rank.pop("per_author_rank")
                    rows.append({"feature": feature, "task": task, "arm": arm, **rank})
        if none_delta > 5e-4:
            raise ValueError(
                "Identical no-history anchors differ across nominal authors"
            )
        pd.DataFrame(rows).to_csv(
            output / "shared_history_anchor_correspondence.csv", index=False
        )
        if independent_history:
            independent_rows = []
            for feature in features:
                for task in QUESTIONS:
                    for left, right in (
                        ("history_0", "history_1"),
                        ("history_0", "history_2"),
                        ("history_1", "history_2"),
                    ):
                        for first_index, second_index in (
                            (0, 0),
                            (1, 1),
                            (0, 1),
                            (1, 0),
                        ):
                            first = np.stack(
                                [
                                    data[(uid, first_index, left, task)][feature]
                                    for uid in authors
                                ]
                            )
                            second = np.stack(
                                [
                                    data[(uid, second_index, right, task)][feature]
                                    for uid in authors
                                ]
                            )
                            rank = paired_author_rank(first, second)
                            rank.pop("per_author_rank")
                            independent_rows.append(
                                {
                                    "feature": feature,
                                    "task": task,
                                    "history_pair": f"{left}__{right}",
                                    "first_anchor": first_index,
                                    "second_anchor": second_index,
                                    "shared_anchor": first_index == second_index,
                                    **rank,
                                }
                            )
            independent_frame = pd.DataFrame(independent_rows)
            independent_frame.to_csv(
                output / "independent_history_correspondence.csv", index=False
            )
            independent_frame.groupby(["feature", "task", "shared_anchor"])[
                "author_rank_auc"
            ].agg(["mean", "min", "max"]).reset_index().to_csv(
                output / "independent_history_summary.csv", index=False
            )
        write_json(
            output / "validation.json",
            {
                **checks,
                "no_history_across_authors_max_abs": none_delta,
                "authors": len(authors),
                "cases": len(frame),
                "labels_loaded": False,
                "current_is_author_text": False,
                "independent_history_test": independent_history,
                "leakage_source_exclusions": len(excluded_authors),
                "parent_identity": manifest["identity"],
                "analysis_code_sha256": code_fingerprint(ROOT),
                "interpretation": "The shared_history table reuses input. Only the separate independent_history table uses disjoint author comments; neither table is personality validity.",
            },
        )
        print(
            json.dumps(
                {
                    "authors": len(authors),
                    "cases": len(frame),
                    "anchor_only": True,
                    "controls": checks,
                    "identical_input_delta": none_delta,
                }
            ),
            flush=True,
        )
        return
    correspondence, displacement, margins = [], [], []
    for task in QUESTIONS:
        for uid in authors:
            for i in (0, 1):
                for arm in ARMS:
                    current = meta[(uid, i, arm, task)]
                    none = meta[(uid, i, "none", task)]
                    anchor = meta[(uid, 2, arm, task)]
                    none_anchor = meta[(uid, 2, "none", task)]
                    margins.append(
                        {
                            "author_id": uid,
                            "task": task,
                            "current_index": i,
                            "arm": arm,
                            "margin": current["margin"],
                            "delta_vs_none": current["margin"] - none["margin"],
                            "anchor_centered_delta": current["margin"]
                            - none["margin"]
                            - anchor["margin"]
                            + none_anchor["margin"],
                            "prompt_tokens": current["prompt_tokens"],
                            "position_delta": current["content_position"]
                            - none["content_position"],
                        }
                    )
        for feature in features:
            vectors = {
                name: [[], []]
                for name in (
                    "raw_none",
                    "raw_same",
                    "raw_other_condition",
                    "same_minus_other",
                    "anchor_centered_same_minus_other",
                )
            }
            for uid in authors:
                for i in (0, 1):
                    get = lambda arm, index=i, uid=uid, task=task, feature=feature: (
                        data[(uid, index, arm, task)][feature]
                    )
                    none, same, other = get("none"), get("same"), get("other_condition")
                    raw = same - other
                    anchor = get("same", 2) - get("other_condition", 2)
                    values = {
                        "raw_none": none,
                        "raw_same": same,
                        "raw_other_condition": other,
                        "same_minus_other": raw,
                        "anchor_centered_same_minus_other": anchored_decomposition(
                            same, get("same", 2), other, get("other_condition", 2)
                        )["interaction"],
                    }
                    for name, value in values.items():
                        vectors[name][i].append(value)
                    for left, right in (
                        ("same", "none"),
                        ("other_length", "none"),
                        ("other_condition", "none"),
                        ("same", "other_length"),
                        ("same", "other_condition"),
                        ("same_reversed", "same"),
                    ):
                        delta = get(left) - get(right)
                        d_anchor = get(left, 2) - get(right, 2)
                        base_norm = max(np.linalg.norm(get(right)), 1e-12)
                        displacement.append(
                            {
                                "author_id": uid,
                                "task": task,
                                "current_index": i,
                                "feature": feature,
                                "contrast": f"{left}_minus_{right}",
                                "norm": float(np.linalg.norm(delta)),
                                "relative_norm": float(
                                    np.linalg.norm(delta) / base_norm
                                ),
                                "anchor_centered_norm": float(
                                    np.linalg.norm(delta - d_anchor)
                                ),
                                "anchor_norm": float(np.linalg.norm(d_anchor)),
                                "position_delta": meta[(uid, i, left, task)][
                                    "content_position"
                                ]
                                - meta[(uid, i, right, task)]["content_position"],
                            }
                        )
            for name, pair in vectors.items():
                rank = paired_author_rank(np.stack(pair[0]), np.stack(pair[1]))
                rank.pop("per_author_rank")
                correspondence.append(
                    {"feature": feature, "task": task, "representation": name, **rank}
                )
    output.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(displacement)
    d.to_csv(output / "context_displacements.csv", index=False)
    numeric = ["norm", "relative_norm", "anchor_centered_norm", "anchor_norm"]
    grouped = (
        d.groupby(["feature", "task", "contrast", "author_id"])[numeric]
        .mean()
        .reset_index()
    )
    summary = (
        grouped.groupby(["feature", "task", "contrast"])[numeric]
        .agg(["mean", "median"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, c)) for c in summary.columns]
    summary.to_csv(output / "displacement_summary.csv", index=False)
    pd.DataFrame(correspondence).to_csv(
        output / "author_correspondence.csv", index=False
    )
    m = pd.DataFrame(margins)
    m.to_csv(output / "context_margin_effects.csv", index=False)
    margin_summary = (
        m.groupby(["task", "arm"])[["margin", "delta_vs_none", "anchor_centered_delta"]]
        .agg(["mean", "median"])
        .reset_index()
    )
    margin_summary.columns = ["_".join(filter(None, c)) for c in margin_summary.columns]
    margin_summary.to_csv(output / "margin_summary.csv", index=False)
    relations = []
    for setting, sub in d.groupby(["feature", "task", "contrast"]):
        r = (
            spearmanr(sub.position_delta.abs(), sub.norm).statistic
            if sub.position_delta.nunique() > 1 and sub.norm.nunique() > 1
            else None
        )
        relations.append(
            {
                "feature": setting[0],
                "task": setting[1],
                "contrast": setting[2],
                "position_abs_vs_effect_spearman": float(r)
                if r is not None and np.isfinite(r)
                else None,
            }
        )
    pd.DataFrame(relations).to_csv(output / "position_sensitivity.csv", index=False)
    matching = pd.read_csv(parent / "donor_matching.csv")
    matching = matching[matching.author_id.isin(authors)]
    matched = matching[matching.status == "MATCHED"]
    validation = {
        **checks,
        "authors": len(authors),
        "cases": len(frame),
        "labels_loaded": False,
        "leakage_source_exclusions": len(excluded_authors),
        "parent_identity": json.loads((parent / "manifest.json").read_text())[
            "identity"
        ],
        "analysis_code_sha256": code_fingerprint(ROOT),
        "median_donor_relative_token_error": float(
            np.median(
                abs(matched.donor_tokens - matched.own_tokens) / matched.own_tokens
            )
        ),
        "max_donor_relative_token_error": float(
            np.max(abs(matched.donor_tokens - matched.own_tokens) / matched.own_tokens)
        ),
        "unique_donors": int(matched.donor_id.nunique()),
        "scope": "Exploratory model context effects, not personality accuracy or human test-retest",
        "caveats": [
            "No-history comparisons change semantic content AND current absolute position",
            "Matching reduces but does not eliminate length/topic/time differences",
            "Shared history and donor reuse induce dependence; no inferential p-value claimed",
            "Anchor-centered interaction depends on the chosen anchor, not a pure person component",
            "Tasks differ in evidence-to-generalization demand, not a pure psychological axis",
            "Only one fixed model and question wording; no questionnaire validation",
        ],
    }
    write_json(output / "validation.json", validation)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--views-for-leakage-filter",
        type=Path,
        help="Exclude whole targets touched by flagged source authors; labels never loaded",
    )
    args = parser.parse_args()
    summarize(args.parent_dir, args.output_dir, args.views_for_leakage_filter)
