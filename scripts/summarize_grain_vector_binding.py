#!/usr/bin/env python3
"""Separate additive lexical/record features, contextual binding, and output bias."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.binding import factorial_interaction
from grain_research.vector_mechanisms.controls import material_hash
from scripts.run_grain_vector_mechanism_baselines import MODELS, encode_local
from scripts.run_grain_vector_mechanism_controls import write_json


def fit_probe(x, labels, train, test):
    """Fixed low-capacity probe; collapse sub-tolerance margins to numerical ties."""
    scaler = StandardScaler().fit(x[train])
    keep = scaler.var_ > 1e-10
    if not keep.any():
        return np.zeros(int(test.sum()))
    model = LogisticRegression(
        C=0.01, solver="liblinear", max_iter=2000, random_state=731
    )
    model.fit(scaler.transform(x[train])[:, keep], labels[train])
    scores = model.decision_function(scaler.transform(x[test])[:, keep])
    if np.ptp(scores) < 1e-7:
        scores[:] = 0
    return scores


def main():
    """Inspect all fixed layers without selecting a personality prediction direction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_status.json").exists():
        raise ValueError("Use a new analysis directory")
    manifest = json.loads((args.parent_dir / "manifest.json").read_text())
    status = json.loads((args.parent_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["cases"] != 288:
        raise ValueError("Expected complete binding factorial")
    cases = manifest["cases"]
    frame = pd.read_csv(args.parent_dir / "predictions.csv")
    if frame.case_id.tolist() != [r["case_id"] for r in cases]:
        raise ValueError("Feature/result row ordering mismatch")
    features = {}
    for i, row in enumerate(cases):
        with np.load(args.parent_dir / "cache" / f"{row['case_id']}.npz") as trace:
            for site in trace.files:
                features.setdefault(
                    "qwen_" + site, np.empty((len(cases), len(trace[site])))
                )[i] = trace[site]
    train = frame.split.eq("discovery").to_numpy()
    passages = np.array([r["passage"] for r in cases])
    for ngram in ((1, 1), (1, 2)):
        vectorizer = CountVectorizer(ngram_range=ngram)
        vectorizer.fit(passages[train])
        features[f"lexical_counts_{ngram[1]}gram"] = (
            vectorizer.transform(passages).toarray().astype(float)
        )
    provenance = {}
    texts = list(
        dict.fromkeys(
            [
                r[key]
                for r in cases
                for key in ("passage", "identity_record", "event_record")
            ]
        )
    )
    indices = {t: i for i, t in enumerate(texts)}
    for name, spec in MODELS.items():
        vectors, provenance[name] = encode_local(texts, spec, args.device)
        vectors = vectors.astype(float)
        features[name + "_joint"] = np.stack(
            [vectors[indices[r["passage"]]] for r in cases]
        )
        features[name + "_additive_records"] = np.stack(
            [
                vectors[indices[r["identity_record"]]]
                + vectors[indices[r["event_record"]]]
                for r in cases
            ]
        )
        print(f"ENCODED {name} unique_texts={len(texts)}", flush=True)
    np.savez_compressed(out / "comparison_vectors.npz", **features)
    write_json(out / "encoder_provenance.json", provenance)
    probes, detail = [], []
    for task in frame.task.unique():
        mask = frame.task.eq(task).to_numpy()
        tr, te = train & mask, (~train) & mask
        y = frame.expected_yes.to_numpy()
        for key, x in features.items():
            scores = fit_probe(x, y, tr, te)
            probes.append(
                {
                    "task": task,
                    "method": key,
                    "heldout_auc": roc_auc_score(y[te], scores),
                    "accuracy": np.mean((scores >= 0) == y[te]),
                    "score_range": np.ptp(scores),
                    "train_scenarios": 8,
                    "eval_scenarios": 4,
                    "scope": "fixed auxiliary probe; not the LLM's output or personality validity",
                }
            )
            detail.extend(
                {
                    "case_id": frame.case_id.iloc[i],
                    "method": key,
                    "task": task,
                    "score": score,
                    "expected_yes": y[i],
                }
                for i, score in zip(np.flatnonzero(te), scores, strict=True)
            )
    pd.DataFrame(probes).to_csv(out / "layer_readability.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "probe_predictions.csv", index=False)
    interactions, outputs = [], []
    for (group, order, task), subset in frame.groupby(["group_id", "order", "task"]):
        ix = {(int(r.a), int(r.b)): i for i, r in subset.iterrows()}
        margins = {ab: frame.margin.iloc[i] for ab, i in ix.items()}
        outputs.append(
            {
                "group_id": group,
                "order": order,
                "task": task,
                "finite_difference_margin": factorial_interaction(margins),
                "auc": roc_auc_score(subset.expected_yes, subset.margin),
                "accuracy": subset.correct_default.mean(),
            }
        )
        for key, x in features.items():
            h = {ab: x[i] for ab, i in ix.items()}
            delta = factorial_interaction(h)
            scale = np.mean([np.linalg.norm(v) for v in h.values()])
            d0, d1 = h[1, 0] - h[0, 0], h[1, 1] - h[0, 1]
            denom = np.linalg.norm(d0) * np.linalg.norm(d1)
            interactions.append(
                {
                    "group_id": group,
                    "order": order,
                    "task": task,
                    "method": key,
                    "interaction_norm": np.linalg.norm(delta),
                    "relative_interaction_norm": np.linalg.norm(delta)
                    / max(scale, 1e-12),
                    "identity_effect_cosine_across_event": np.dot(d0, d1) / denom
                    if denom > 1e-10
                    else None,
                }
            )
    pd.DataFrame(interactions).to_csv(
        out / "representation_interactions.csv", index=False
    )
    output_frame = pd.DataFrame(outputs)
    output_frame.to_csv(out / "output_interactions.csv", index=False)
    summary = []
    for (order, task), sub in output_frame.groupby(["order", "task"]):
        for metric in ("finite_difference_margin", "auc", "accuracy"):
            summary.append(
                {
                    "order": order,
                    "task": task,
                    "metric": metric,
                    **grouped_bootstrap(sub[metric], sub.group_id),
                }
            )
    pd.DataFrame(summary).to_csv(out / "output_summary.csv", index=False)
    multisets = json.loads((args.parent_dir / "token_multisets.json").read_text())
    mismatch = 0
    for _, sub in frame.groupby(["group_id", "order", "task"]):
        bags = [multisets[k] for k in sub.case_id]
        mismatch += int(any(b != bags[0] for b in bags))
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "parent_identity": manifest["identity"],
            "literal_word_bag_control": "tested in unit tests",
            "token_bag_mismatched_quartets": mismatch,
            "scope": "synthetic relational binding",
            "cases_sha256": material_hash(cases),
        },
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
