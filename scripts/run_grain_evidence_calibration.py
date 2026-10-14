#!/usr/bin/env python3
"""Blinded-model evidence audit and bounded assignment-error sensitivity analysis.

No psychological outcomes are loaded. Independent model judgments are not human
gold labels. Default retrieval/NLI rules are evaluated, not retuned on 80 pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import (
    literal_confusion,
    load_model_annotations,
    verify_frozen_top1,
)
from grain_research.predecessors import TRAITS, keyed_proportions
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_entailment_reference import MODEL, REVISION

BASELINE = BASE / "predecessor_integration_v1"
ANNOTATION_SHA = "d3043f3424e6ce776541dc4b2fa774c8be27e3d6c6fbc2bddb9945e851fb2ac9"


def directed_code(frame, scope):
    """Return same/reverse/uninformative as 0/1/2, never infer a person's trait."""
    if scope not in ("general", "episode_inclusive"):
        raise ValueError("Unknown evidence scope")
    same = [1] if scope == "general" else [1, 3]
    reverse = [2] if scope == "general" else [2, 4]
    self = frame.self_attribution.eq("self")
    return np.select(
        [
            self & frame.simpa_category.isin(same),
            self & frame.simpa_category.isin(reverse),
        ],
        [0, 1],
        default=2,
    )


def stratified_rate(frame, values):
    """Estimate a finite-pool proportion under the recorded stratified sample."""
    v = np.asarray(values, float)
    if len(v) != len(frame) or not np.isfinite(v).all():
        raise ValueError("One finite value per sampled pair required")
    return float(np.average(v, weights=frame.sampling_weight))


def precision_summary(frame, prediction, correct):
    """Separate sample counts from sampling-weighted precision and acceptance."""
    p, c = np.asarray(prediction, bool), np.asarray(correct, bool)
    weight = frame.sampling_weight.to_numpy(float)
    denominator = float(weight[p].sum())
    return {
        "sample_predicted_positive": int(p.sum()),
        "sample_correct_positive": int(np.sum(p & c)),
        "weighted_acceptance": stratified_rate(frame, p),
        "weighted_precision": float(weight[p & c].sum() / denominator)
        if denominator
        else None,
    }


def infer_pairs(frame):
    """Use the pinned standard NLI model on premise/sentence and hypothesis/item."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval()
    labels = {int(i): str(v).lower() for i, v in model.config.id2label.items()}
    if set(labels.values()) != {"contradiction", "neutral", "entailment"}:
        raise ValueError("Unknown NLI label encoding")
    result = []
    for start in range(0, len(frame), 8):
        f = frame.iloc[start : start + 8]
        batch = tokenizer(
            f.sentence.tolist(),
            f.item_text.tolist(),
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if batch["input_ids"].shape[1] > 512:
            raise ValueError("Pair exceeds NLI context; no silent truncation")
        with torch.inference_mode():
            probabilities = torch.softmax(model(**batch).logits, -1).numpy()
        for row, p in zip(f.itertuples(), probabilities, strict=True):
            result.append(
                {
                    "audit_id": row.audit_id,
                    "nli": labels[int(p.argmax())],
                    **{"p_" + labels[i]: float(p[i]) for i in labels},
                }
            )
    return pd.DataFrame(result)


def assignment_sensitivity(evidence, bundles, accepted_labels, *, scope, draws=400):
    """Propagate an explicitly assumed exchangeable three-way detector model.

    A single Dirichlet draw per replicate is shared across authors. The draw
    governs same-key, reverse-key and unsupported assignments. Each source
    sentence gets one assignment before resampling its whole comment, so a
    repeated bootstrap copy cannot acquire a different label. This is a model-
    coded sensitivity distribution, not a confidence interval for personality.
    """
    counts = np.bincount(directed_code(accepted_labels, scope), minlength=3)
    rng = np.random.default_rng(731)
    q = rng.dirichlet(counts + 1.0, size=draws)
    source = [i for i, b in enumerate(bundles) if b["official_folds"]["big5"] != 0]
    selected = evidence[evidence.author_index.isin(source) & evidence.accepted]
    rows, totals, any_author = [], np.zeros(draws), np.zeros(draws)
    for author in source:
        matches = selected[selected.author_index == author]
        m = len(matches)
        values = rng.random((draws, m))
        assignments = np.where(
            values < q[:, :1], 0, np.where(values < q[:, :2].sum(1)[:, None], 1, 2)
        )
        c = np.zeros((draws, 36, len(TRAITS), 2))
        for j, row in enumerate(matches.itertuples()):
            retained = assignments[:, j] < 2
            ix = np.flatnonzero(retained)
            key = (0 if row.key == 1 else 1) ^ assignments[ix, j]
            c[ix, row.comment_index, TRAITS.index(row.trait), key] += 1
        sampled = c[np.arange(draws)[:, None], rng.integers(0, 36, (draws, 36))].sum(1)
        totals += sampled.sum((1, 2))
        any_author += sampled.sum((1, 2)) > 0
        scores = keyed_proportions(sampled)
        for t, trait in enumerate(TRAITS):
            finite = scores[:, t][np.isfinite(scores[:, t])]
            enough = len(finite) >= int(0.95 * draws)
            rows.append(
                {
                    "author_index": author,
                    "trait": trait,
                    "scope": scope,
                    "score_available_draw_fraction": len(finite) / draws,
                    "conditional_draw_median": float(np.median(finite))
                    if len(finite)
                    else None,
                    "sensitivity_lower": float(np.quantile(finite, 0.025))
                    if enough
                    else None,
                    "sensitivity_upper": float(np.quantile(finite, 0.975))
                    if enough
                    else None,
                    "status": "model_assumption_sensitivity"
                    if enough
                    else "insufficient_draw_availability",
                }
            )
    return pd.DataFrame(rows), {
        "scope": scope,
        "source_authors": len(source),
        "annotated_accepted_pairs": len(accepted_labels),
        "observed_direction_counts_same_reverse_none": counts.tolist(),
        "dirichlet_prior": [1, 1, 1],
        "posterior_mean_same_reverse_none": (
            (counts + 1) / (counts.sum() + 3)
        ).tolist(),
        "retained_sentence_count_draw_interval": np.quantile(
            totals, [0.025, 0.5, 0.975]
        ).tolist(),
        "authors_with_any_evidence_draw_interval": np.quantile(
            any_author, [0.025, 0.5, 0.975]
        ).tolist(),
        "assumptions": "Same error probabilities across source authors/items within accepted stratum; conditional independent assignments; exchangeable whole comments. Model annotator correctness is NOT established or propagated.",
    }


def run(args):
    """Check identity and annotation provenance before looking at scorer outputs."""
    out = args.output
    if out.exists():
        raise ValueError("Use a fresh output directory")
    out.mkdir(parents=True)
    packet_path = args.audit / "private_stratified_annotation_packet.csv"
    label_path = args.audit / "private_blind_sol_annotations.json"
    packet = pd.read_csv(packet_path)[["audit_id", "sentence", "item_id", "item_text"]]
    annotation, labels = load_model_annotations(
        label_path, packet.audit_id, args.annotation_sha
    )
    data = json.loads((args.material / "private_material.json").read_text())
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    if (
        digest(args.material / "private_material.json")
        != manifest["files"]["private_material.json"]
    ):
        raise ValueError("Changed source texts")
    evidence = pd.read_csv(args.audit / "private_annotation_sampling_frame.csv")
    verified = verify_frozen_top1(args.material, args.producer, evidence, data)
    sampling = json.loads((args.audit / "summary.json").read_text())
    allocation = {r["stratum"]: r for r in sampling["annotation_allocation"]}
    frame = packet.merge(labels, on="audit_id", validate="one_to_one")
    frame["sentence_index"] = frame.audit_id.str.replace(
        "PAIR_", "", regex=False
    ).astype(int)
    similarities, authors = [], []
    for row in frame.itertuples():
        e, s = evidence.iloc[row.sentence_index], data["sentences"][row.sentence_index]
        if (
            row.sentence != s["text"]
            or row.item_id != e.item_id
            or int(e.author_index) != s["author_index"]
            or data["bundles"][s["author_index"]]["official_folds"]["big5"] == 0
        ):
            raise ValueError("Annotation linkage or source-only permission violation")
        similarities.append(e.similarity)
        authors.append(int(e.author_index))
    frame["similarity"] = similarities
    frame["author_index"] = authors
    frame["stratum"] = pd.cut(
        frame.similarity,
        [-1.001, 0.35, 0.5, 0.6, 1.001],
        labels=["low", "middle", "near", "accepted"],
        right=False,
    ).astype(str)
    for stratum, group in frame.groupby("stratum"):
        if len(group) != allocation[stratum]["sampled"]:
            raise ValueError("Sampling stratum count differs")
    frame["sampling_weight"] = [
        allocation[s]["available"] / allocation[s]["sampled"] for s in frame.stratum
    ]
    frame = frame.merge(infer_pairs(frame), on="audit_id", validate="one_to_one")
    frame["literal_expected"] = frame.relation.map(
        {
            "entails": "entailment",
            "contradicts": "contradiction",
            "neutral": "neutral",
        }
    )
    resolved = frame.literal_expected.notna()
    frame.to_csv(out / "private_pair_assessment.csv", index=False)
    errors = frame[resolved & frame.nli.ne(frame.literal_expected)]
    errors.to_csv(out / "private_nli_disagreements.csv", index=False)
    confusion = pd.crosstab(frame.literal_expected, frame.nli)
    confusion.to_csv(out / "literal_confusion.csv")
    pd.crosstab(frame.stratum, frame.simpa_category).to_csv(
        out / "annotation_categories_by_stratum.csv"
    )
    comparisons = []
    rules = {
        "cosine_0.6": frame.similarity >= 0.6,
        "nli_argmax_entailment": frame.nli.eq("entailment"),
        "cosine_and_nli": (frame.similarity >= 0.6) & frame.nli.eq("entailment"),
    }
    for scope in ("general", "episode_inclusive"):
        correct = directed_code(frame, scope) == 0
        for rule, predicted in rules.items():
            comparisons.append(
                {
                    "scope": scope,
                    "rule": rule,
                    **precision_summary(frame, predicted, correct),
                }
            )
    pd.DataFrame(comparisons).to_csv(out / "weighted_rule_comparison.csv", index=False)
    sensitivities = []
    for scope in ("general", "episode_inclusive"):
        scores, result = assignment_sensitivity(
            evidence,
            data["bundles"],
            frame[frame.stratum.eq("accepted")],
            scope=scope,
        )
        scores.to_csv(out / f"private_{scope}_error_sensitivity.csv", index=False)
        sensitivities.append(result)
    summary = {
        "status": "COMPLETE",
        "pairs": len(frame),
        "unique_authors": len(set(authors)),
        "rater_kind": annotation["rater_kind"],
        "rater_model": annotation["rater_model"],
        "independent_annotation_scope": annotation["scope"],
        "resolved_literal_pairs": int(resolved.sum()),
        "neutral_baseline_agreement_resolved": float(
            frame.loc[resolved, "literal_expected"].eq("neutral").mean()
        ),
        "literal_confusion": literal_confusion(frame),
        "evidence_contract_verification": verified,
        "literal_agreement_resolved": float(
            frame.loc[resolved, "nli"]
            .eq(frame.loc[resolved, "literal_expected"])
            .mean()
        ),
        "model_general_same_direction_pairs": int(
            np.sum(directed_code(frame, "general") == 0)
        ),
        "model_episode_inclusive_same_direction_pairs": int(
            np.sum(directed_code(frame, "episode_inclusive") == 0)
        ),
        "rule_comparisons": comparisons,
        "sensitivity": sensitivities,
        "threshold_fitted": False,
        "decision": "Do not retune with this small model-coded packet. Retain literal entailment and episode/general evidence as separate channels.",
        "personality_outcomes_loaded": False,
        "scope": "Model-assisted source-only evidence pilot, not expert calibration or personality validity. Weighted estimates target the first-I/top-1 candidate pool, not all text or all possible items.",
        "sources": {
            "packet": digest(packet_path),
            "annotation": digest(label_path),
            "sampling_summary": digest(args.audit / "summary.json"),
            "sampling_frame": digest(
                args.audit / "private_annotation_sampling_frame.csv"
            ),
            "material_manifest": digest(args.material / "material_manifest.json"),
            "code": digest(Path(__file__)),
            "evidence_contract_code": digest(
                ROOT / "grain_research/evidence_contracts.py"
            ),
            "nli_model": MODEL,
            "nli_revision": REVISION,
        },
    }
    # The accepted stratum is a simple random sample; other NLI-selected rows
    # have unequal sampling weights and must not receive this binomial interval.
    accepted = frame[frame.stratum.eq("accepted")]
    summary["accepted_stratum_model_judgment_intervals"] = {}
    for scope in ("general", "episode_inclusive"):
        k = int(np.sum(directed_code(accepted, scope) == 0))
        ci = binomtest(k, len(accepted)).proportion_ci(method="exact")
        summary["accepted_stratum_model_judgment_intervals"][scope] = {
            "same_direction": k,
            "sample": len(accepted),
            "clopper_pearson_95": [ci.low, ci.high],
            "scope": "IID binomial approximation ignoring small finite-pool correction and annotator error; model-coded label agreement only",
        }
    write(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    """Run only the existing 80-pair source queue with cached model weights."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument("--audit", type=Path, default=BASELINE / "evidence_sampling_r1")
    p.add_argument("--producer", type=Path, default=BASELINE / "paraphrase_r1")
    p.add_argument("--annotation-sha", default=ANNOTATION_SHA)
    p.add_argument("--output", type=Path, default=BASELINE / "evidence_audit_r1")
    run(p.parse_args())


if __name__ == "__main__":
    main()
