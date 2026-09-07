#!/usr/bin/env python3
"""Independently replay P1F stored margins and frozen vector readouts without forwards."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Hash a local receipt or array without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(parent: Path, operator: Path, shape: Path):
    """Verify complete cells, candidates, state restoration and frozen operator math."""
    manifest = json.loads((parent / "manifest.json").read_text())
    status = json.loads((parent / "run_status.json").read_text())
    if status["status"] != "COMPLETE" or status["calls"] != 482:
        raise ValueError("P1F is incomplete or call accounting changed")
    for name, expected in json.loads(
        (parent / "output_hashes.json").read_text()
    ).items():
        if sha(parent / name) != expected:
            raise ValueError("P1F output changed")
    with zipfile.ZipFile(parent / "source_snapshot.zip") as z:
        for name, expected in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError("P1F source changed")
    previous = json.loads((operator / "summary.json").read_text())
    if previous["identity"] != manifest["parent_identity"]:
        raise ValueError("Different operator parent")
    for name, expected in previous["outputs"].items():
        if sha(operator / name) != expected:
            raise ValueError("Frozen operator changed")
    for name, expected in json.loads(
        (shape / "output_hashes.json").read_text()
    ).items():
        if sha(shape / name) != expected:
            raise ValueError("Frozen shape changed")
    native = pd.read_csv(parent / "baselines.csv")
    effects = pd.read_csv(parent / "effects.csv")
    predicted = pd.read_csv(parent / "frozen_operator_predictions.csv")
    summaries = pd.read_csv(parent / "frozen_operator_scores.csv")
    if len(native) != 48 or native.case_id.nunique() != 48 or len(effects) != 432:
        raise ValueError("Wrong panel size")
    expected_cells = {("block_04", "E")} | {
        ("block_16", k) for k in ("E", "Q", "D", "EQD", "bin1", "bin2", "bin3", "bin4")
    }
    for _, sub in effects.groupby("case_id"):
        if len(sub) != 9 or set(zip(sub.site, sub.patch)) != expected_cells:
            raise ValueError("Patch panel incomplete")
    arithmetic = max(
        float(abs(df.margin - df.high_logit + df.low_logit).max())
        for df in (native, effects)
    )
    lookup = native.set_index("case_id").margin
    full = effects[effects.patch == "EQD"]
    full_error = float(abs(full.margin - full.donor_case.map(lookup)).max())
    layouts = json.loads((parent / "token_layouts.json").read_text())
    for layout in layouts.values():
        if [i for part in layout["bins"] for i in part] != layout["partitions"]["Q"]:
            raise ValueError("Query bin coverage changed")
    old = pd.read_csv(shape / "operator_scores.csv")
    betas = {
        r.variant: np.array(json.loads(r.coefficients))
        for r in old.itertuples()
        if r.split == "fit" and r.degree == 1
    }
    with np.load(operator / "linear_gap_models.npz", allow_pickle=False) as loaded:
        models = {k: loaded[k] for k in loaded.files}
    states = {
        case: np.load(parent / "activations" / (case + ".npz"), allow_pickle=False)[
            "block_16"
        ].astype(float)
        for case in native.case_id
    }
    gap_error, effect_error, metric_error = 0.0, 0.0, 0.0
    if (
        len(predicted) != 640
        or predicted.duplicated(["case_id", "patch", "gap_view"]).any()
    ):
        raise ValueError("Frozen prediction grid changed")
    for row in predicted.itertuples():
        observed_gap = lookup[row.donor_case] - lookup[row.case_id]
        if row.gap_view == "oracle_gap":
            gap = observed_gap
        else:
            indices = layouts[row.case_id]["partitions"][row.gap_view]
            contrast = states[row.donor_case][indices].mean(axis=0) - states[
                row.case_id
            ][indices].mean(axis=0)
            key = row.variant + "__" + row.gap_view
            gap = float(np.dot(models[key], contrast) / models[key + "__scale"][0])
        factor = (
            float(betas[row.variant].sum())
            if row.patch == "Q"
            else betas[row.variant][int(row.patch[-1]) - 1]
        )
        gap_error = max(gap_error, abs(gap - row.predicted_gap))
        effect_error = max(effect_error, abs(gap * factor - row.predicted_shift))
    for r in summaries.itertuples():
        sub = predicted[
            (predicted.variant == r.variant)
            & (predicted.form == r.form)
            & (predicted.gap_view == r.gap_view)
        ]
        observed = sub.margin.to_numpy() - sub.case_id.map(lookup).to_numpy()
        err = sub.predicted_shift.to_numpy() - observed
        value = np.dot(err, err) / np.dot(observed, observed)
        metric_error = max(metric_error, abs(value - r.relative_error_energy))
    if (
        max(arithmetic, full_error, metric_error) > 1e-8
        or max(gap_error, effect_error) > 1e-5
    ):
        raise ValueError("Independent numerical replay failed")
    return {
        "status": "VERIFIED",
        "parent_identity": manifest["identity"],
        "arithmetic_max": arithmetic,
        "full_restore_max": full_error,
        "float64_gap_replay_max": gap_error,
        "float64_effect_replay_max": effect_error,
        "relative_error_metric_max": metric_error,
        "new_neural_calls": 0,
        "interpretation": "numerical validity only, not psychological validation",
    }


def main():
    """Write a separate immutable receipt with audit source and hashes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--operator-dir", type=Path, required=True)
    p.add_argument("--shape-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    receipt = audit(a.parent_dir, a.operator_dir, a.shape_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    receipt["source_sha256"] = sha(Path(__file__))
    (a.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
