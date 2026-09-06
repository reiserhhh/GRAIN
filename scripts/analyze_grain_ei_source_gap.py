#!/usr/bin/env python3
"""Post-hoc pooled-versus-within-fold audit of the complete fixed EI/raw controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof


def rank_auc(y, p):
    """Independent tied-rank AUC; undefined class support is not replaced by chance."""
    y, p = np.asarray(y), np.asarray(p)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Finite aligned binary outcomes and scores required")
    positive = y == 1
    n, m = int(positive.sum()), int((~positive).sum())
    if not n or not m:
        return float("nan")
    ranks = rankdata(p, method="average")
    return float((ranks[positive].sum() - n * (n + 1) / 2) / (n * m))


def gap_values(y, predictions):
    """Partition the score contrast algebraically, without claiming causal mediation."""
    scores = np.asarray([rank_auc(y, p) for p in predictions])
    retained, purged, counts = scores[0], scores[1], scores[2:]
    if len(counts) != 5:
        raise ValueError("Exactly five frozen count seeds required")
    middle = float(np.mean(counts))
    return np.asarray(
        [
            retained,
            purged,
            middle,
            middle - retained,
            purged - middle,
            purged - retained,
        ]
    )


NAMES = (
    "retain_auc",
    "purge_auc",
    "mean_count_auc",
    "count_minus_retain",
    "purge_minus_count",
    "purge_minus_retain",
)


def analyze(y, predictions, folds, draws=1000):
    """Resample authors within fixed folds; retain both pooled and mean-fold estimands."""
    y, predictions, folds = np.asarray(y), np.asarray(predictions), np.asarray(folds)
    if (
        predictions.shape != (7, len(y))
        or len(folds) != len(y)
        or set(folds) != set(range(5))
        or draws < 20
    ):
        raise ValueError("Seven complete paired controls and five folds required")
    parts = [np.flatnonzero(folds == f) for f in range(5)]
    per_fold = np.stack([gap_values(y[ix], predictions[:, ix]) for ix in parts])
    observed = np.stack([gap_values(y, predictions), per_fold.mean(0)])
    rng = np.random.default_rng(731)
    samples = []
    for _ in range(draws):
        sampled = [rng.choice(ix, size=len(ix), replace=True) for ix in parts]
        joined = np.concatenate(sampled)
        samples.append(
            np.stack(
                [
                    gap_values(y[joined], predictions[:, joined]),
                    np.mean(
                        [gap_values(y[ix], predictions[:, ix]) for ix in sampled],
                        axis=0,
                    ),
                ]
            )
        )
    samples = np.asarray(samples)
    valid = samples[np.isfinite(samples).all(axis=(1, 2))]
    if len(valid) < 0.95 * draws:
        raise ValueError("Insufficient bootstrap class support")
    result = []
    for a, estimand in enumerate(("pooled", "mean_official_fold")):
        for b, name in enumerate(NAMES):
            lo, hi = np.quantile(valid[:, a, b], [0.025, 0.975])
            result.append(
                {
                    "estimand": estimand,
                    "quantity": name,
                    "estimate": observed[a, b],
                    "lower": lo,
                    "upper": hi,
                    "draws": draws,
                    "undefined_draws": draws - len(valid),
                }
            )
    return result, per_fold


def main():
    """Read only previously audited OOF artifacts, never source label tables or text."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    batch = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    audit = json.loads((a.audit_dir / "summary.json").read_text())
    if (
        audit["batch_identity"] != batch["identity"]
        or audit["verified_cases"] < 7
        or audit["random_count_pairs_verified"] < 5
    ):
        raise ValueError("Require independently audited complete first seven controls")
    names = [
        "EI_cont_raw_retain_731",
        "EI_cont_raw_purge_731",
        *[f"EI_cont_raw_random_count_{seed}" for seed in range(731, 736)],
    ]
    packets = [load_oof(a.batch_dir / name) for name in names]
    parent = packets[0]
    for i, obj in enumerate(packets):
        m = obj["manifest"]
        mode = "retain" if i == 0 else "purge" if i == 1 else "random_count"
        seed = 731 if i < 2 else 731 + i - 2
        if (
            m["task"] != "EI_cont"
            or m["arm"] != "raw"
            or m["feature"] != "word_char"
            or m["thread_control"] != mode
            or m["drop_seed"] != seed
            or not m["exclude_reference"]
            or m["permutation_seed"] is not None
            or any(not np.array_equal(obj[k], parent[k]) for k in ("ids", "y", "folds"))
        ):
            raise ValueError("Changed or unpaired frozen EI source controls")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    fields = {
        "protocol": "VM05_EI_SOURCE_GAP_POSTHOC_V1",
        "posthoc": True,
        "batch_identity": batch["identity"],
        "audit_sha256": digest(a.audit_dir / "summary.json"),
        "inputs": {
            name: {
                "identity": obj["manifest"]["identity"],
                "oof_sha256": digest(
                    a.batch_dir / name / "private_oof_predictions.csv"
                ),
            }
            for name, obj in zip(names, packets)
        },
        "bootstrap": "1000 author draws within each fixed official fold, seed731",
        "authors": len(parent["ids"]),
        "bridge_labels_read": False,
        "model_refits": 0,
        "script_sha256": digest(Path(__file__)),
    }
    fields["identity"] = material_hash(fields)
    write_json(a.output_dir / "manifest.json", fields)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    rows, folds = analyze(
        parent["y"], np.stack([p["p"] for p in packets]), parent["folds"]
    )
    pd.DataFrame(rows).to_csv(a.output_dir / "paired_gap_estimands.csv", index=False)
    pd.DataFrame(folds, columns=NAMES).rename_axis("fold").to_csv(
        a.output_dir / "gap_by_fold.csv"
    )
    for obj in packets:
        if not np.isclose(
            rank_auc(obj["y"], obj["p"]), obj["summary"]["score"], atol=1e-12
        ):
            raise ValueError("Independent rank AUC does not reproduce stored result")
    summary = {
        "status": "COMPLETE",
        "parent_identity": fields["identity"],
        "posthoc": True,
        "authors": len(parent["ids"]),
        "controls": 7,
        "draws": 1000,
        "model_refits": 0,
        "bridge_labels_read": False,
        "new_label_values_read": False,
        "scope": "Conditional on seven fitted heads and fixed folds/seeds; not refit uncertainty or causal decomposition",
        "max_additive_identity_error": float(
            abs(folds[:, 3] + folds[:, 4] - folds[:, 5]).max()
        ),
        "outputs_sha256": {
            name: digest(a.output_dir / name)
            for name in ("paired_gap_estimands.csv", "gap_by_fold.csv")
        },
    }
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
