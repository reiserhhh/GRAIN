#!/usr/bin/env python3
"""Paired post-hoc three-mask comparison using complete audited EI OOF families."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.analyze_grain_ei_source_gap import NAMES, control_names, gap_values
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof

ARMS = ("raw", "cue_mask", "placebo_mask")
PAIRS = ((1, 0), (2, 0), (1, 2))
QUANTITIES = (0, 4, 5)


def contrasts(values):
    """Contrast masks on retained AUC and two source-removal gaps, not CI endpoints."""
    return np.stack(
        [values[a, :, QUANTITIES] - values[b, :, QUANTITIES] for a, b in PAIRS]
    )


def analyze(y, predictions, folds, draws=1000):
    """Reuse the identical author draw for all21 models to retain cross-arm covariance."""
    y, predictions, folds = np.asarray(y), np.asarray(predictions), np.asarray(folds)
    if (
        predictions.shape != (3, 7, len(y))
        or folds.shape != y.shape
        or set(folds) != set(range(5))
        or draws < 20
    ):
        raise ValueError("Require3 complete paired7-control families and5 folds")
    parts = [np.flatnonzero(folds == f) for f in range(5)]

    def estimates(indices):
        joined = np.concatenate(indices)
        return np.asarray(
            [
                [
                    gap_values(y[joined], arm[:, joined]),
                    np.mean([gap_values(y[ix], arm[:, ix]) for ix in indices], axis=0),
                ]
                for arm in predictions
            ]
        )

    observed = contrasts(estimates(parts))
    rng, samples = np.random.default_rng(731), []
    for _ in range(draws):
        indices = [rng.choice(ix, len(ix), replace=True) for ix in parts]
        samples.append(contrasts(estimates(indices)))
    samples = np.asarray(samples)
    valid = samples[np.isfinite(samples).all(axis=(1, 2, 3))]
    if len(valid) < 0.95 * draws:
        raise ValueError("Insufficient bootstrap class support")
    rows = []
    for i, (a, b) in enumerate(PAIRS):
        for j, quantity in enumerate(QUANTITIES):
            for k, estimand in enumerate(("pooled", "mean_official_fold")):
                lo, hi = np.quantile(valid[:, i, j, k], [0.025, 0.975])
                rows.append(
                    {
                        "contrast": f"{ARMS[a]}_minus_{ARMS[b]}",
                        "quantity": NAMES[quantity],
                        "estimand": estimand,
                        "estimate": float(observed[i, j, k]),
                        "lower": float(lo),
                        "upper": float(hi),
                        "draws": draws,
                        "undefined_draws": draws - len(valid),
                    }
                )
    return rows


def main():
    """Read only audited prior OOF artifacts; never tune prompts, heads or source membership."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("batch-dir", "audit-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((a.audit_dir / "summary.json").read_text(encoding="utf-8"))
    batch = json.loads(
        (a.batch_dir / "batch_manifest.json").read_text(encoding="utf-8")
    )
    scores = pd.read_csv(a.audit_dir / "source_control_scores.csv")
    if (
        audit["batch_identity"] != batch["identity"]
        or audit["verified_cases"] < 21
        or audit["random_count_pairs_verified"] < 15
        or scores["case"].duplicated().any()
    ):
        raise ValueError("Full21-case independent audit required")
    families = []
    inputs = {}
    parent = None
    for arm in ARMS:
        family = []
        for i, name in enumerate(control_names(arm)):
            if name not in set(scores["case"]) or name in audit["pending_cases"]:
                raise ValueError("Unaudited family member")
            packet = load_oof(a.batch_dir / name)
            m = packet["manifest"]
            if parent is None:
                parent = packet
            mode = "retain" if i == 0 else "purge" if i == 1 else "random_count"
            seed = 731 if i < 2 else 731 + i - 2
            if (
                m["task"] != "EI_cont"
                or m["arm"] != arm
                or m["feature"] != "word_char"
                or m["thread_control"] != mode
                or m["drop_seed"] != seed
                or not m["exclude_reference"]
                or m["permutation_seed"] is not None
                or scores.loc[scores["case"] == name, "source_identity"].item()
                != m["identity"]
                or any(
                    not np.array_equal(packet[k], parent[k])
                    for k in ("ids", "y", "folds")
                )
            ):
                raise ValueError("Changed or unpaired source family")
            inputs[name] = {
                "identity": m["identity"],
                "oof_sha256": digest(
                    a.batch_dir / name / "private_oof_predictions.csv"
                ),
            }
            family.append(packet["p"])
        families.append(family)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    code = (
        "scripts/analyze_grain_ei_mask_specificity.py",
        "scripts/analyze_grain_ei_source_gap.py",
        "scripts/summarize_grain_vector_external_links.py",
    )
    m = {
        "protocol": "VM05_EI_MASK_SPECIFICITY_POSTHOC_V1",
        "posthoc": True,
        "inputs": inputs,
        "audit_sha256": digest(a.audit_dir / "summary.json"),
        "code_sha256": {name: digest(ROOT / name) for name in code},
        "bootstrap": "1000 paired author-within-fold draws, seed731, conditional on fitted heads",
        "authors": len(parent["ids"]),
        "bridge_labels_read": False,
        "model_refits": 0,
    }
    m["identity"] = material_hash(m)
    write_json(a.output_dir / "manifest.json", m)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
    ) as z:
        for name in code:
            if name not in z.namelist():
                z.write(ROOT / name, name)
    rows = analyze(parent["y"], families, parent["folds"])
    path = a.output_dir / "paired_mask_specificity.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    s = {
        "status": "COMPLETE",
        "parent_identity": m["identity"],
        "posthoc": True,
        "authors": len(parent["ids"]),
        "models": 21,
        "contrasts": len(rows),
        "draws": 1000,
        "source_label_tables_read": False,
        "new_label_values_read": False,
        "previously_audited_oof_outcomes_reused": True,
        "bridge_labels_read": False,
        "model_refits": 0,
        "outputs_sha256": {path.name: digest(path)},
        "scope": "Unadjusted exploratory paired sensitivity, not causal or refit uncertainty",
    }
    write_json(a.output_dir / "summary.json", s)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
