#!/usr/bin/env python3
"""Independent raw-logit matrix reconstruction of P1Q's finite interaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    """Hash a frozen artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result, parent):
    """Apply an independent contrast matrix to raw logits and verify every summary."""
    for folder in (result, parent):
        if json.loads((folder / "run_status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("Incomplete source")
        for name, digest in json.loads(
            (folder / "output_hashes.json").read_text()
        ).items():
            if sha(folder / name) != digest:
                raise ValueError("Artifact digest")
        m = json.loads((folder / "manifest.json").read_text())
        with zipfile.ZipFile(folder / "source_snapshot.zip") as z:
            for name, digest in m["code"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != digest:
                    raise ValueError("Source snapshot")
    manifest = json.loads((result / "manifest.json").read_text())
    if (
        manifest["parent_identity"]
        != json.loads((parent / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Parent identity")
    base = pd.read_csv(parent / "baselines.csv").set_index("case_id")
    effects = pd.read_csv(parent / "effects.csv").set_index(
        ["case_id", "axis", "patch"]
    )
    saved = pd.read_csv(result / "components.csv")
    if len(saved) != 160 or saved.duplicated(["case_id", "axis"]).any():
        raise ValueError("Cell population")
    source_geometry = pd.read_csv(parent / "geometry.csv").set_index(
        ["case_id", "axis"]
    )
    layouts = json.loads((parent / "token_layouts.json").read_text())
    if set(zip(saved.case_id, saved.axis)) != set(source_geometry.index):
        raise ValueError("Cell identities")
    names = ["direction", "complement", "interaction", "total", "projection_error"]
    matrix = np.array(
        [[-1, 0, 1, 0], [-1, 0, 0, 1], [1, 1, -1, -1], [-1, 1, 0, 0], [0, 1, -1, 0]],
        dtype=np.longdouble,
    )
    fresh, errors = [], []
    rounding_max = 0.0
    for r in saved.itertuples():
        b = base.loc[r.case_id]
        if any(getattr(r, k) != b[k] for k in ("group_id", "entity", "criterion")):
            raise ValueError("Component metadata")
        patches = np.load(
            parent / "patch_states" / (r.case_id + "__" + r.axis + ".npz")
        )
        x = np.load(parent / "activations" / (r.case_id + ".npz"))["block_16"][
            layouts[r.case_id]["anchor"][0]
        ].astype(float)
        rounding = float(
            abs(
                patches["natural"].astype(float)
                - patches["projected"].astype(float)
                - patches["complement"].astype(float)
                + x
            ).max()
        )
        rounding_max = max(rounding_max, rounding)
        errors.extend(
            (
                abs(rounding - r.state_addition_fp32_error),
                abs(
                    r.projection_energy_fraction
                    - source_geometry.loc[
                        (r.case_id, r.axis), "projection_energy_fraction"
                    ]
                ),
            )
        )
        values = [np.longdouble(b.high_logit) - np.longdouble(b.low_logit)]
        for patch in ("natural", "projected", "complement"):
            v = effects.loc[(r.case_id, r.axis, patch)]
            values.append(np.longdouble(v.high_logit) - np.longdouble(v.low_logit))
        coefficient = matrix @ values
        errors.extend(abs(float(v) - getattr(r, k)) for k, v in zip(names, coefficient))
        v = effects.loc[(r.case_id, r.axis, "matched_random")]
        random = float(
            np.longdouble(v.high_logit) - np.longdouble(v.low_logit) - values[0]
        )
        errors.append(abs(random - r.random_shift))
        fresh.append(
            {
                "case_id": r.case_id,
                "axis": r.axis,
                "criterion": b.criterion,
                "group_id": b.group_id,
                **dict(zip(names, coefficient.astype(float))),
                "random_shift": random,
                "projection_energy_fraction": r.projection_energy_fraction,
            }
        )
    frame = pd.DataFrame(fresh)
    summary = pd.read_csv(result / "energy_summary.csv", keep_default_na=False)
    if len(summary) != 30:
        raise ValueError("Summary population")
    for r in summary.itertuples():
        s = frame[(frame.criterion == r.criterion) & (frame.axis == r.axis)]
        if r.scope == "setting":
            s = s[s.group_id == r.group_id]
        elif r.scope != "pooled":
            raise ValueError("Unknown scope")
        if len(s) != r.cases:
            raise ValueError("Summary denominator")
        for feature in (*names, "random_shift"):
            label = "random_sse" if feature == "random_shift" else feature + "_sse"
            errors.append(abs(np.dot(s[feature], s[feature]) - getattr(r, label)))
        errors.extend(
            (
                abs(2 * np.dot(s.complement, s.interaction) - r.signed_cross_term),
                abs(s.projection_energy_fraction.mean() - r.mean_projection_energy),
            )
        )
    maximum = float(max(errors))
    if maximum > 1e-9 or rounding_max > 1e-5:
        raise ValueError("Independent raw-logit reconstruction failed")
    return {
        "status": "VERIFIED",
        "cells": 160,
        "summaries": 30,
        "new_neural_calls": 0,
        "raw_logit_and_energy_max": maximum,
        "state_addition_fp32_max": rounding_max,
        "parent_identity": manifest["parent_identity"],
        "coverage": "Independent longdouble contrast matrix, dot-product energies and saved-array rounding; source vector provenance also covered by P1P audit.",
    }


def main():
    """Write the separate audit receipt and exact source used."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result-dir", "parent-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    receipt = audit(a.result_dir, a.parent_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    receipt["source_sha256"] = sha(Path(__file__))
    (a.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
