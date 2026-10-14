#!/usr/bin/env python3
"""Label-free structure, evidence availability, and resampling diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.predecessors import MeaningExtraction
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write


def loading_alignment(reference, alternative):
    """Match axes using training vocabulary loadings, never held-out scores/labels."""
    vocab = reference.vectorizer_.get_feature_names_out()
    other = alternative.vectorizer_.get_feature_names_out()
    union = sorted(set(vocab) | set(other))
    ix = {word: j for j, word in enumerate(union)}
    one = np.zeros((len(union), reference.loadings_.shape[1]))
    two = np.zeros((len(union), alternative.loadings_.shape[1]))
    one[[ix[w] for w in vocab]] = reference.loadings_
    two[[ix[w] for w in other]] = alternative.loadings_
    congruence = (
        one.T
        @ two
        / np.maximum(
            np.linalg.norm(one, axis=0)[:, None] * np.linalg.norm(two, axis=0)[None, :],
            1e-12,
        )
    )
    rows, cols = linear_sum_assignment(-abs(congruence))
    signs = np.where(congruence[rows, cols] >= 0, 1, -1)
    return rows, cols, signs, abs(congruence[rows, cols])


def run(args):
    """Inspect a fixed structure reference on 120 held-out authors; no trait access."""
    out = args.output
    if out.exists():
        raise ValueError("Use a new output directory")
    out.mkdir(parents=True)
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if digest(args.material / name) != checksum:
            raise ValueError("Material changed")
    data = json.loads((args.material / "private_material.json").read_text())
    bundles = data["bundles"]
    folds = np.asarray([b["official_folds"]["big5"] for b in bundles])
    documents = np.asarray(data["documents"])
    train, held = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
    records, reliability = [], []
    with threadpool_limits(limits=4):
        reference = MeaningExtraction().fit(documents[train])
        baseline = reference.transform(documents[held])
        halves = [
            reference.transform(np.asarray(data[key])[held])
            for key in ("half_a", "half_b")
        ]
        for j in range(baseline.shape[1]):
            reliability.append(
                {
                    "factor_code": f"MEM_{j + 1:02d}",
                    "authors": len(held),
                    "half_r": pearsonr(halves[0][:, j], halves[1][:, j]).statistic,
                }
            )
        rng = np.random.default_rng(731)
        for repeat in range(8):
            keep = rng.choice(train, int(0.8 * len(train)), replace=False)
            alternative = MeaningExtraction().fit(documents[keep])
            row, col, sign, phi = loading_alignment(reference, alternative)
            inferred = alternative.transform(documents[held])
            for j, k, s, congruence in zip(row, col, sign, phi, strict=True):
                records.append(
                    {
                        "repeat": repeat,
                        "factor_code": f"MEM_{j + 1:02d}",
                        "loading_congruence": congruence,
                        "held_score_r": pearsonr(
                            baseline[:, j], inferred[:, k] * s
                        ).statistic,
                    }
                )
    pd.DataFrame(records).to_csv(out / "dropout_structure.csv", index=False)
    pd.DataFrame(reliability).to_csv(out / "split_half_structure.csv", index=False)
    rows = [c for b in bundles for view in b["views"] for c in view]
    audit = {
        "authors": len(bundles),
        "comments": len(rows),
        "known_subreddit": sum(bool(c.get("subreddit")) for c in rows),
        "known_timestamp": sum(c.get("timestamp") is not None for c in rows),
        "known_thread": sum(bool(c.get("thread_id")) for c in rows),
        "leakage_flag_records_present": sum(bool(c.get("leakage_flags")) for c in rows),
        "true_leakage_flags": {
            key: sum(bool(c.get("leakage_flags", {}).get(key)) for c in rows)
            for key in ("mbti_type_term", "big5_term", "report_clue")
        },
        "opportunity_set_observed": False,
        "nonselection_interpreted_as_zero_preference": False,
        "calendar_time_model_fitted": False,
        "random_comment_views_are_time_replicates": False,
        "note": "Subreddit is observed selected context, not the set of topics offered to the author. Timestamps are available metadata, not an automatically regular observation schedule.",
    }
    write(out / "data_contract_report.json", audit)
    report = {
        "status": "COMPLETE",
        "source_manifest": digest(args.material / "material_manifest.json"),
        "code_sha256": digest(Path(__file__)),
        "labels_loaded": False,
        "discovery_authors": len(train),
        "held_authors": len(held),
        "dropout_repeats": 8,
        "retained_training_fraction": 0.8,
        "alignment": "training_loading_union_vocabulary_hungarian_sign_corrected",
        "median_loading_congruence": float(
            pd.DataFrame(records).loading_congruence.median()
        ),
        "median_held_score_r": float(pd.DataFrame(records).held_score_r.median()),
        "median_half_r": float(pd.DataFrame(reliability).half_r.median()),
        "scope": "MEM_adaptation; component reproducibility is not factor meaning or personality validity",
    }
    write(out / "structure_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    """Run a bounded vocabulary-structure check without any external labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument(
        "--output", type=Path, default=BASE / "predecessor_integration_v1/structure_r1"
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
