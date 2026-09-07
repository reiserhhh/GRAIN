#!/usr/bin/env python3
"""Decompose frozen operator error and query-bin non-invariance without new forwards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def decompose(observed, oracle, predicted):
    """Partition squared error exactly into amplitude, shape and their signed overlap."""
    observed, oracle, predicted = map(np.asarray, (observed, oracle, predicted))
    denom = float(np.dot(observed, observed))
    if denom <= 1e-12:
        raise ValueError("No observed effect energy")
    amplitude, shape = predicted - oracle, oracle - observed
    result = {
        "amplitude_energy": float(np.dot(amplitude, amplitude) / denom),
        "shape_energy": float(np.dot(shape, shape) / denom),
        "cross_energy": float(2 * np.dot(amplitude, shape) / denom),
        "total_error_energy": float(np.sum((predicted - observed) ** 2) / denom),
    }
    result["identity_error"] = abs(
        result["total_error_energy"]
        - result["amplitude_energy"]
        - result["shape_energy"]
        - result["cross_energy"]
    )
    return result


def analyze(parent):
    """Use only immutable synthetic predictions, preserving both positive and null cases."""
    state = json.loads((parent / "run_status.json").read_text())
    if state["status"] != "COMPLETE":
        raise ValueError("Complete P1F required")
    for name, expected in json.loads(
        (parent / "output_hashes.json").read_text()
    ).items():
        if hashlib.sha256((parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Parent changed")
    pred = pd.read_csv(parent / "frozen_operator_predictions.csv")
    oracle = (
        pred[pred.gap_view == "oracle_gap"]
        .set_index(["case_id", "patch"])
        .predicted_shift
    )
    terms = []
    for (variant, form, view), sub in pred[pred.gap_view != "oracle_gap"].groupby(
        ["variant", "form", "gap_view"]
    ):
        truth_oracle = oracle.reindex(
            pd.MultiIndex.from_frame(sub[["case_id", "patch"]])
        ).to_numpy()
        terms.append(
            {
                "variant": variant,
                "form": form,
                "view": view,
                "settings": sub.group_id.nunique(),
                **decompose(
                    sub["shift"].to_numpy(),
                    truth_oracle,
                    sub.predicted_shift.to_numpy(),
                ),
            }
        )
    effects = pd.read_csv(parent / "transport_rows.csv")
    profiles = []
    for case, sub in effects.groupby("case_id"):
        sub = sub[sub.site == "block_16"].set_index("patch")
        bins = sub.loc[["bin1", "bin2", "bin3", "bin4"], "shift"].to_numpy()
        full = float(sub.loc["Q", "shift"])
        meta = sub.iloc[0]
        gap = float(meta.donor_gap)
        profiles.append(
            {
                "case_id": case,
                "variant": meta.variant,
                "form": meta.form,
                "group_id": meta.group_id,
                "level": int(meta.level),
                "donor_gap": gap,
                **{f"bin{i + 1}": float(v) for i, v in enumerate(bins)},
                "full_query_shift": full,
                "additivity_defect": full - float(bins.sum()),
                "relative_additivity_defect": (full - float(bins.sum())) / abs(gap)
                if abs(gap) > 1e-8
                else None,
                "small_gap": abs(gap) < 0.1,
            }
        )
    profiles = pd.DataFrame(profiles)
    comparisons = []
    for (variant, group, level), sub in profiles.groupby(
        ["variant", "group_id", "level"]
    ):
        sub = sub.set_index("form")
        a, b = [
            sub.loc[form, [f"bin{i}" for i in range(1, 5)]].to_numpy(dtype=float)
            for form in ("canonical", "rewrite")
        ]
        scale = np.linalg.norm(a) * np.linalg.norm(b)
        comparisons.append(
            {
                "variant": variant,
                "group_id": group,
                "level": level,
                "bin_profile_cosine": float(np.dot(a, b) / scale)
                if scale > 1e-12
                else None,
                "canonical_norm": float(np.linalg.norm(a)),
                "rewrite_norm": float(np.linalg.norm(b)),
            }
        )
    return {
        "loss_decomposition": pd.DataFrame(terms),
        "bin_profiles": profiles,
        "query_form_profile_comparison": pd.DataFrame(comparisons),
    }, state["identity"]


def main():
    """Write fresh analysis and source; no fitting, neural calls or overwritten results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tables, identity = analyze(args.parent_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / (name + ".csv"), index=False)
    (args.output_dir / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    receipt = {
        "parent_identity": identity,
        "status": "COMPLETE",
        "new_neural_calls": 0,
        "refitted": False,
        "identity_error_max": float(tables["loss_decomposition"].identity_error.max()),
        "outputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output_dir.iterdir()
            if p.is_file() and not p.name.startswith("._")
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
