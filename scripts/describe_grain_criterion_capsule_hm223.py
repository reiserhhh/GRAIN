#!/usr/bin/env python3
"""Predeclared Gram/natural-routing descriptors without changing sealed science."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_criterion_capsule as C
from scripts import run_grain_observed_family_hm170 as R


def geometry(values):
    """Describe author distances, not psychological dimensions or information shares."""
    x = np.asarray(values, float).reshape(24, -1)
    uncentered = x @ x.T / x.shape[1]
    centered = x - x.mean(0, keepdims=True)
    gram = centered @ centered.T / x.shape[1]
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues.min() < -1e-10:
        raise ValueError("Invalid measured Gram; no PSD rescue")
    positive = eigenvalues[eigenvalues > 1e-12]
    p = positive / positive.sum() if len(positive) else positive
    effective = float(np.exp(-np.sum(p * np.log(p)))) if len(p) else 0.0
    return (
        uncentered,
        gram,
        {
            "authors": 24,
            "energy": float(np.trace(uncentered) / 24),
            "centered_energy": float(np.trace(gram) / 24),
            "centered_rank_at1e-12": len(positive),
            "effective_rank": effective,
            "minimum_eigenvalue_unclipped": float(eigenvalues.min()),
        },
    )


def main():
    """Save separately sealed requested descriptors; never refit or inject again."""
    primary = ROOT / "results/grain_math_response_v1/criterion_capsule_r1"
    out = primary / "predeclared_descriptors_r1"
    R.check(
        primary,
        (
            "captured_seal.json",
            "cold_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        ),
    )
    out.mkdir(exist_ok=False)
    R.write(
        out / "descriptor_manifest.json",
        {
            "code": R.digest(Path(__file__)),
            "purpose": "HM223predeclaredGram/naturalroutingmetadata,NOTnewscientificarm",
            "primary_summary": R.digest(primary / "summary.json"),
            "capsules": R.digest(primary / "capsules.npz"),
            "no_model_fit_or_calls": True,
        },
    )
    with np.load(primary / "capsules.npz", allow_pickle=False) as z:
        bank = z["values"]
    common = np.stack([C.condition_values(v, "common") for v in bank])
    specific = bank - common
    arrays, summary = {}, {}
    for name, values in (("full", bank), ("common", common), ("specific", specific)):
        raw, centered, account = geometry(values)
        arrays[name + "_uncentered"] = raw
        arrays[name + "_centered"] = centered
        summary[name] = account
    baseline = R.read(primary / "cold/144.json")["receipts"]
    changed = []
    for i in range(144):
        receipts = R.read(primary / "cold" / f"{i:03d}.json")["receipts"]
        changed.append(
            sum(
                r["final_p_hash"] != b["final_p_hash"]
                for r, b in zip(receipts, baseline, strict=True)
            )
        )
    summary["natural_final_probability_changed_layer_counts"] = {
        "min": min(changed),
        "max": max(changed),
        "total": sum(changed),
        "batches": 144,
    }
    summary["orthogonal_common_specific_cross"] = float(np.mean(common * specific))
    summary["original_scope_complete"] = False
    R.archive(out / "grams.npz", **arrays)
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "descriptor_seal.json",
        ("descriptor_manifest.json", "grams.npz", "summary.json"),
    )
    print(summary)


if __name__ == "__main__":
    main()
