#!/usr/bin/env python3
"""P1Q: exact finite response accounting for direction/complement interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    """Hash a source file or immutable result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decompose(base, natural, projected, complement):
    """Separate finite components with signed readout interaction retained."""
    d, c = projected - base, complement - base
    interaction = natural - projected - complement + base
    return {
        "direction": d,
        "complement": c,
        "interaction": interaction,
        "total": natural - base,
        "projection_error": natural - projected,
    }


def run(args):
    """Use the full declared P1P cache; never fit or select another direction."""
    start = time.monotonic()
    parent = args.parent_dir
    if json.loads((parent / "run_status.json").read_text())["status"] != "COMPLETE":
        raise ValueError("Complete P1P required")
    for name, digest in json.loads((parent / "output_hashes.json").read_text()).items():
        if sha(parent / name) != digest:
            raise ValueError("Changed parent")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        ROOT / "docs/GRAIN_DIRECTION_INTERACTION_P1Q_PLAN.md",
        ROOT / "tests/test_grain_direction_interaction.py",
    ]
    manifest = {
        "parent_identity": json.loads((parent / "manifest.json").read_text())[
            "identity"
        ],
        "new_neural_calls": 0,
        "max_seconds": 60,
        "code": {str(p.relative_to(ROOT)): sha(p) for p in paths},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT))
    try:
        base = pd.read_csv(parent / "baselines.csv").set_index("case_id")
        effects = pd.read_csv(parent / "effects.csv")
        geometry = pd.read_csv(parent / "geometry.csv").set_index(["case_id", "axis"])
        layouts = json.loads((parent / "token_layouts.json").read_text())
        rows = []
        for (case, axis), sub in effects.groupby(["case_id", "axis"]):
            index = sub.set_index("patch").margin
            if (
                set(index.index)
                != {"natural", "projected", "complement", "matched_random"}
                or len(index) != 4
            ):
                raise ValueError("Incomplete intervention cell")
            b = base.loc[case]
            values = decompose(
                b.margin, index.natural, index.projected, index.complement
            )
            states = np.load(parent / "patch_states" / (case + "__" + axis + ".npz"))
            x = np.load(parent / "activations" / (case + ".npz"))["block_16"][
                layouts[case]["anchor"][0]
            ].astype(float)
            rounding = float(
                abs(
                    states["natural"].astype(float)
                    - states["projected"].astype(float)
                    - states["complement"].astype(float)
                    + x
                ).max()
            )
            rows.append(
                {
                    "case_id": case,
                    "axis": axis,
                    "group_id": b.group_id,
                    "entity": b.entity,
                    "criterion": b.criterion,
                    **values,
                    "random_shift": index.matched_random - b.margin,
                    "projection_energy_fraction": geometry.loc[
                        (case, axis), "projection_energy_fraction"
                    ],
                    "state_addition_fp32_error": rounding,
                }
            )
        if len(rows) != 160:
            raise ValueError("Incomplete160-cell panel")
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "components.csv", index=False)
        aggregate = []
        for scope, keys in (
            ("pooled", ["criterion", "axis"]),
            ("setting", ["group_id", "criterion", "axis"]),
        ):
            for key, s in frame.groupby(keys):
                aggregate.append(
                    {
                        **dict(zip(keys, key)),
                        "scope": scope,
                        "cases": len(s),
                        "total_sse": float(np.sum(s.total**2)),
                        "direction_sse": float(np.sum(s.direction**2)),
                        "projection_error_sse": float(np.sum(s.projection_error**2)),
                        "complement_sse": float(np.sum(s.complement**2)),
                        "interaction_sse": float(np.sum(s.interaction**2)),
                        "signed_cross_term": float(
                            2 * np.sum(s.complement * s.interaction)
                        ),
                        "random_sse": float(np.sum(s.random_shift**2)),
                        "mean_projection_energy": float(
                            s.projection_energy_fraction.mean()
                        ),
                    }
                )
        summary = pd.DataFrame(aggregate)
        summary.to_csv(out / "energy_summary.csv", index=False)
        scalar = float(
            abs(
                frame.total - frame.direction - frame.complement - frame.interaction
            ).max()
        )
        energy = float(
            abs(
                summary.projection_error_sse
                - summary.complement_sse
                - summary.interaction_sse
                - summary.signed_cross_term
            ).max()
        )
        rounding = float(frame.state_addition_fp32_error.max())
        if (
            max(scalar, energy) > 1e-9
            or rounding > 1e-5
            or time.monotonic() - start > 60
        ):
            raise ValueError("Finite identity/resource gate")
        status = {
            "status": "COMPLETE",
            "new_neural_calls": 0,
            "cells": len(frame),
            "summaries": len(summary),
            "scalar_identity_max": scalar,
            "energy_identity_max": energy,
            "state_addition_fp32_max": rounding,
            "seconds": time.monotonic() - start,
        }
        (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        (out / "output_hashes.json").write_text(
            json.dumps(
                {
                    str(p.relative_to(out)): sha(p)
                    for p in out.rglob("*")
                    if p.is_file() and not p.name.startswith("._")
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(status))
    except Exception as exc:
        (out / "run_status.json").write_text(
            json.dumps({"status": "FAILED", "error": str(exc)}) + "\n"
        )
        raise


def main():
    """Read one frozen source and write a fresh analysis directory."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
