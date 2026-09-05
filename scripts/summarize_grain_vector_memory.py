#!/usr/bin/env python3
"""Author-clustered natural memory effects, scoped synthetic accuracy and provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def clustered(values, authors, prefix):
    """Label uncertainty by its author sampling unit and descriptive estimand."""
    return {prefix + "_" + k: v for k, v in grouped_bootstrap(values, authors).items()}


def cosine(a, b):
    """Return missing for zero effects, rather than fabricate stability."""
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm > 1e-12 else np.nan


def main():
    """Never tune readout thresholds or rank layers from these results."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    status, manifest = (
        read(a.parent_dir / "run_status.json"),
        read(a.parent_dir / "manifest.json"),
    )
    if status["status"] != "COMPLETE":
        raise ValueError("Incomplete model execution")
    review = read(a.review)
    if manifest["review_sha256"] != material_hash(review):
        raise ValueError("Semantic review identity mismatch")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new analysis directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    df = pd.read_csv(a.parent_dir / "readouts.csv")
    df["feature_row"] = np.arange(len(df))
    features = np.load(a.parent_dir / "representations.npz")
    if len(df) != status["cases"] or df.job_id.duplicated().any():
        raise ValueError("Readout count/identity mismatch")
    if df.job_id.tolist() != [r["job_id"] for r in manifest["jobs"]]:
        raise ValueError("Feature/readout order mismatch")
    eligible = {r["author_id"] for r in review["reviews"] if r["audit"]["facts"]}
    source_rows, facts = [], []
    for i, r in enumerate(review["reviews"]):
        audit = r["audit"]
        source_rows.append(
            {
                "author_index": i,
                "strict_status": audit["strict_status"],
                "status": audit["status"],
                "repair": audit["repair"],
                "facts": len(audit["facts"]),
                "citation_valid": sum(
                    f["status"] == "EXACT_QUOTE_SUPPORTED" for f in audit["facts"]
                ),
                "semantic_accepted": sum(
                    f["semantic_review"] == "ACCEPT" for f in audit["facts"]
                ),
            }
        )
        facts.extend(
            {
                "author_index": i,
                "fact_id": f["fact_id"],
                "citation_status": f["status"],
                "semantic_review": f["semantic_review"],
                "reason": f["review_reason"],
            }
            for f in audit["facts"]
        )
    pd.DataFrame(source_rows).to_csv(a.output_dir / "summary_audit.csv", index=False)
    pd.DataFrame(facts).to_csv(
        a.output_dir / "semantic_audit_redacted.csv", index=False
    )
    natural = df[df.kind == "natural"]
    rows, vectors, alignment = [], [], []
    for scope in ("all_authors", "schema_usable_authors"):
        subset = (
            natural
            if scope == "all_authors"
            else natural[natural.author_id.isin(eligible)]
        )
        for task, taskrows in subset.groupby("task"):
            base = taskrows[taskrows.arm == "raw_full"].set_index("case_id")
            for arm, sub in taskrows.groupby("arm"):
                paired = base.loc[sub.case_id]
                delta = sub.margin.to_numpy() - paired.margin.to_numpy()
                rows.append(
                    {
                        "scope": scope,
                        "task": task,
                        "arm": arm,
                        "currents": len(sub),
                        "authors": sub.author_id.nunique(),
                        "positive_count": int(sub.predicted_yes.sum()),
                        "mean_margin": float(sub.margin.mean()),
                        "mean_prompt_tokens": float(sub.prompt_tokens.mean()),
                        "token_ratio_to_raw": float(
                            (
                                sub.prompt_tokens.to_numpy()
                                / paired.prompt_tokens.to_numpy()
                            ).mean()
                        ),
                        "flips_from_raw": int(
                            (
                                sub.predicted_yes.to_numpy()
                                != paired.predicted_yes.to_numpy()
                            ).sum()
                        ),
                        "yes_no_mass_mean": float(sub.yes_no_probability_mass.mean()),
                        **clustered(delta, sub.author_id, "delta_margin"),
                    }
                )
                for key in ("block_10_current_mean", "block_16_decision"):
                    x, b = (
                        features[key][sub.feature_row],
                        features[key][paired.feature_row],
                    )
                    delta_vec = x - b
                    relative = np.linalg.norm(delta_vec, axis=1) / np.maximum(
                        np.linalg.norm(b, axis=1), 1e-12
                    )
                    vectors.append(
                        {
                            "scope": scope,
                            "task": task,
                            "arm": arm,
                            "feature": key,
                            **clustered(
                                relative, sub.author_id, "relative_displacement"
                            ),
                        }
                    )
                    if scope == "all_authors" and arm != "raw_full":
                        for uid, own in sub.groupby("author_id"):
                            if len(own) == 2:
                                original = base.loc[own.case_id]
                                d = (
                                    features[key][own.feature_row]
                                    - features[key][original.feature_row]
                                )
                                alignment.append(
                                    {
                                        "task": task,
                                        "arm": arm,
                                        "feature": key,
                                        "author_id": uid,
                                        "current_pair_cosine": cosine(d[0], d[1]),
                                    }
                                )
    pd.DataFrame(rows).to_csv(a.output_dir / "memory_ablation.csv", index=False)
    pd.DataFrame(vectors).to_csv(a.output_dir / "context_displacement.csv", index=False)
    align = pd.DataFrame(alignment)
    if len(align):
        align.to_csv(a.output_dir / "private_current_pair_alignment.csv", index=False)
        summary = []
        for keys, sub in align.groupby(["task", "arm", "feature"]):
            summary.append(
                dict(zip(("task", "arm", "feature"), keys, strict=True))
                | clustered(sub.current_pair_cosine, sub.author_id, "cosine")
            )
        pd.DataFrame(summary).to_csv(
            a.output_dir / "current_pair_alignment.csv", index=False
        )
    interaction = []
    for (cid, task), sub in natural.groupby(["case_id", "task"]):
        arms = sub.set_index("arm")
        for score in ("margin", "conditional_yes"):
            x = arms[score]
            interaction.append(
                {
                    "case_id": cid,
                    "task": task,
                    "author_id": sub.iloc[0].author_id,
                    "scale": score,
                    "interaction": float(x.raw_full - x.half_a - x.half_b + x.none),
                }
            )
    interaction = pd.DataFrame(interaction)
    interaction.to_csv(a.output_dir / "private_evidence_interactions.csv", index=False)
    chain_rows = []
    for (cid, task), sub in natural[natural.author_id.isin(eligible)].groupby(
        ["case_id", "task"]
    ):
        scores = sub.set_index("arm").margin
        steps = [
            ("source_span_selection", "summary_quotes", "raw_full"),
            ("cited_claim_rewording", "summary_cited", "summary_quotes"),
            ("semantic_rejection_filter", "summary_reviewed", "summary_cited"),
        ]
        total = 0.0
        for name, upper, lower in steps:
            effect = float(scores[upper] - scores[lower])
            total += effect
            chain_rows.append(
                {
                    "task": task,
                    "step": name,
                    "author_id": sub.iloc[0].author_id,
                    "effect": effect,
                }
            )
        if not np.isclose(total, scores.summary_reviewed - scores.raw_full, atol=1e-10):
            raise ValueError("Memory path telescoping identity failed")
    chain = pd.DataFrame(chain_rows)
    pd.DataFrame(
        [
            {
                "task": task,
                "step": step,
                **clustered(sub.effect, sub.author_id, "effect"),
            }
            for (task, step), sub in chain.groupby(["task", "step"])
        ]
    ).to_csv(a.output_dir / "memory_path_decomposition.csv", index=False)
    pd.DataFrame(
        [
            {
                "task": task,
                "scale": scale,
                **clustered(sub.interaction, sub.author_id, "interaction"),
            }
            for (task, scale), sub in interaction.groupby(["task", "scale"])
        ]
    ).to_csv(a.output_dir / "evidence_interactions.csv", index=False)
    controlled = df[df.kind == "synthetic_on_natural"]
    controls = []
    grouped = [("all", controlled)]
    grouped += [
        (f"{key}={value}", sub)
        for key in ("arm", "placement", "negative_question")
        for value, sub in controlled.groupby(key)
    ]
    grouped += [
        (f"placement={place};negative_question={neg}", sub)
        for (place, neg), sub in controlled.groupby(["placement", "negative_question"])
    ]
    for label, sub in grouped:
        truth = sub.expected_yes.astype(int)
        controls.append(
            {
                "group": label,
                "prompts": len(sub),
                "authors": sub.author_id.nunique(),
                "positive_rate": float(sub.predicted_yes.mean()),
                "true_positive_rate": float(truth.mean()),
                "balanced_accuracy": float(
                    balanced_accuracy_score(truth, sub.predicted_yes)
                )
                if truth.nunique() == 2
                else None,
                "auc": float(roc_auc_score(truth, sub.margin))
                if truth.nunique() == 2
                else None,
                **clustered(sub.correct, sub.author_id, "accuracy"),
            }
        )
    pd.DataFrame(controls).to_csv(a.output_dir / "controlled_updates.csv", index=False)
    flip_rows = []
    for keys, sub in controlled.groupby(["case_id", "arm", "placement"]):
        by_polarity = sub.set_index("negative_question")
        flip_rows.append(
            {
                "case_id": keys[0],
                "arm": keys[1],
                "placement": keys[2],
                "author_id": sub.iloc[0].author_id,
                "complementary_answers": int(
                    by_polarity.loc[False, "predicted_yes"]
                    != by_polarity.loc[True, "predicted_yes"]
                ),
                "both_correct": int(sub.correct.sum() == 2),
            }
        )
    flip = pd.DataFrame(flip_rows)
    polarity = [
        {
            "placement": place,
            **clustered(sub.complementary_answers, sub.author_id, "complementarity"),
            **clustered(sub.both_correct, sub.author_id, "both_correct"),
        }
        for place, sub in flip.groupby("placement")
    ]
    pd.DataFrame(polarity).to_csv(
        a.output_dir / "polarity_consistency.csv", index=False
    )
    output = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "analysis_code_sha256": code_fingerprint(ROOT),
        "authors": int(df.author_id.nunique()),
        "natural_prompts": len(natural),
        "controlled_prompts": len(controlled),
        "strict_summary_usable": sum(
            r["strict_status"] == "PARSED" for r in source_rows
        ),
        "repaired_summary_usable": len(eligible),
        "summary_facts": len(facts),
        "citation_valid": sum(r["citation_valid"] for r in source_rows),
        "semantic_accepted": sum(r["semantic_accepted"] for r in source_rows),
        "controlled_overall": controls[0],
        "runtime": status,
        "limitations": [
            "Six discovery authors; two currents each are not twelve people.",
            "Synthetic truth is one fixed event template on natural backgrounds, not personality labels.",
            "Only two generated summaries meet schema after bounded repair; missing summaries remain missing.",
            "One source-only analyst; not independent inter-rater validation.",
            "Natural memory arms differ in content, token length and absolute position.",
            "Author bootstrap is conditional on these fixed prompt/control families.",
            "Layer 10/16 were fixed before this experiment; no accuracy or label-based layer selection.",
            "Half-history finite differences are descriptive nonadditivity, not isolated causal synergy.",
        ],
    }
    write_json(a.output_dir / "summary.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
