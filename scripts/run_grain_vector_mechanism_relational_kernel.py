#!/usr/bin/env python3
"""Test an explicit tensor-product mechanism without training a language model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.binding import relational_grams
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_baselines import MODELS, encode_local
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Fixed alpha and official cached embeddings; all reference fits are discovery-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_status.json").exists():
        raise ValueError("Use a new output directory")
    parent = json.loads((args.parent_dir / "manifest.json").read_text())
    # Record addition/product is order invariant. Use one order so duplicate rows do not reweight fitting.
    cases = [
        r
        for r in parent["cases"]
        if r["task"] == "narrator_positive" and r["order"] == "identity_first"
    ]
    if len(cases) != 48:
        raise ValueError("Expected 12 complete relational quartets")
    train = np.array([r["split"] == "discovery" for r in cases])
    texts = list(
        dict.fromkeys(
            [r[k] for r in cases for k in ("identity_record", "event_record")]
        )
    )
    lookup = {t: i for i, t in enumerate(texts)}
    ix_a = [lookup[r["identity_record"]] for r in cases]
    ix_b = [lookup[r["event_record"]] for r in cases]
    train_text = list(
        dict.fromkeys(
            [
                r[k]
                for r in cases
                if r["split"] == "discovery"
                for k in ("identity_record", "event_record")
            ]
        )
    )
    manifest = {
        "parent_identity": parent["identity"],
        "code_sha256": code_fingerprint(ROOT),
        "alpha": 0.01,
        "cases": cases,
        "scope": "synthetic relational recovery; not personality prediction",
        "encoders": MODELS,
        "centering": "training-side only",
        "kernel_weights": "fixed equal weights",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    archive_source(ROOT, out / "source_snapshot.zip")
    features, provenance = {}, {}
    lexical = CountVectorizer(ngram_range=(1, 2)).fit(train_text)
    features["word12_counts"] = lexical.transform(texts).toarray().astype(float)
    for name, spec in MODELS.items():
        features[name], provenance[name] = encode_local(texts, spec, args.device)
    np.savez_compressed(out / "record_vectors.npz", **features)
    write_json(out / "encoder_provenance.json", provenance)
    write_json(out / "record_order.json", [{"sha256": material_hash(t)} for t in texts])
    rows, predictions = [], []
    for name, x in features.items():
        kernels = relational_grams(x[ix_a], x[ix_b], train)
        for kind, kernel in kernels.items():
            for task in ("narrator_binding", "alias_action"):
                truth = np.array(
                    [
                        int(r["a"] == r["b"])
                        if task == "narrator_binding"
                        else int(r["b"] == 0)
                        for r in cases
                    ]
                )
                coefficients = np.linalg.solve(
                    kernel[np.ix_(train, train)] + 0.01 * np.eye(train.sum()),
                    2 * truth[train] - 1,
                )
                score = kernel[np.ix_(~train, train)] @ coefficients
                if np.ptp(score) < 1e-7:
                    score[:] = 0
                rows.append(
                    {
                        "encoder": name,
                        "kernel": kind,
                        "task": task,
                        "heldout_auc": roc_auc_score(truth[~train], score),
                        "accuracy": np.mean((score >= 0) == truth[~train]),
                        "min_eigenvalue": np.linalg.eigvalsh(kernel).min(),
                        "train_scenarios": 8,
                        "heldout_scenarios": 4,
                        "heldout_cases": 16,
                        "scope": "explicit fixed kernel ridge comparator, not evidence that the LLM implements it",
                    }
                )
                predictions.extend(
                    {
                        "case_id": cases[i]["case_id"],
                        "encoder": name,
                        "kernel": kind,
                        "task": task,
                        "score": s,
                        "truth": truth[i],
                    }
                    for i, s in zip(np.flatnonzero(~train), score, strict=True)
                )
    pd.DataFrame(rows).to_csv(out / "kernel_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "predictions.csv", index=False)
    write_json(
        out / "run_status.json",
        {
            "status": "COMPLETE",
            "cases": 48,
            "real_personality_labels": False,
            "model_weights_updated": False,
            "alpha": 0.01,
        },
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
