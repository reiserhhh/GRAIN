#!/usr/bin/env python3
"""P1S: finite amplitude/gain accounting and original-gain transfer from cache."""

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
    """Hash the input or output bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def account(a0, a1, y0, y1):
    """Symmetric finite decomposition; original-gain prediction uses no y1."""
    if min(abs(a0), abs(a1)) < 1e-6:
        raise ValueError("Near-zero amplitude; gain undefined")
    g0, g1 = y0 / a0, y1 / a1
    return {
        "original_gain": g0,
        "reversed_gain": g1,
        "amplitude_term": (g1 + g0) * (a1 - a0) / 2,
        "gain_term": (a1 + a0) * (g1 - g0) / 2,
        "observed_change": y1 - y0,
        "original_gain_prediction": g0 * a1,
    }


def run(args):
    """Use all paired states and logits, without a new observer or readout fit."""
    start = time.monotonic()
    for root in (args.original_dir, args.reversed_dir):
        if json.loads((root / "run_status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("Complete source required")
        for name, digest in json.loads(
            (root / "output_hashes.json").read_text()
        ).items():
            if sha(root / name) != digest:
                raise ValueError("Source changed")
    old_manifest = json.loads((args.original_dir / "manifest.json").read_text())
    new_manifest = json.loads((args.reversed_dir / "manifest.json").read_text())
    if new_manifest["parent_identity"] != old_manifest["identity"] or sha(
        args.original_dir / "directions.npz"
    ) != sha(args.reversed_dir / "directions.npz"):
        raise ValueError("Different parent/directions")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__),
        ROOT / "docs/GRAIN_ORDER_GAIN_P1S_PLAN.md",
        ROOT / "tests/test_grain_order_gain.py",
    ]
    manifest = {
        "original_identity": old_manifest["identity"],
        "reversed_identity": new_manifest["identity"],
        "new_neural_calls": 0,
        "max_seconds": 60,
        "code": {str(p.relative_to(ROOT)): sha(p) for p in paths},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT))
    try:
        newrows = json.loads((args.reversed_dir / "materials.json").read_text())["rows"]
        direction = np.load(args.original_dir / "directions.npz")
        stores = []
        for root in (args.original_dir, args.reversed_dir):
            base = pd.read_csv(root / "baselines.csv").set_index("case_id")
            effects = pd.read_csv(root / "effects.csv")
            effects = effects[effects.axis == "B"].set_index(["case_id", "patch"])
            layouts = json.loads((root / "token_layouts.json").read_text())
            donors = json.loads((root / "donors.json").read_text())
            stores.append((root, base, effects, layouts, donors))
        rows, remainder = [], []
        for r in newrows:
            values = []
            for order, case, store in zip(
                ("original", "reversed"), (r["parent_case"], r["case_id"]), stores
            ):
                root, base, effects, layouts, donors = store
                source = (
                    donors[case]["B"]["anchor"]
                    if order == "original"
                    else donors[case]["anchor"]
                )
                x = np.load(root / "activations" / (case + ".npz"))["block_16"][
                    layouts[case]["anchor"][0]
                ].astype(float)
                y = np.load(root / "activations" / (source + ".npz"))["block_16"][
                    layouts[source]["anchor"][0]
                ].astype(float)
                a = float(np.dot(direction[r["entity"] + "_direction"], y - x))
                zero = base.loc[case, "margin"]
                full, projected, complement = [
                    float(effects.loc[(case, p), "margin"] - zero)
                    for p in ("natural", "projected", "complement")
                ]
                values.append((a, projected))
                remainder.append(
                    {
                        "case_id": case,
                        "order": order,
                        "group_id": r["group_id"],
                        "criterion": r["criterion"],
                        "complement": complement,
                        "interaction": full - projected - complement,
                        "projection_error": full - projected,
                    }
                )
            (a0, y0), (a1, y1) = values
            if min(abs(a0), abs(a1)) < 1e-6:
                rows.append(
                    {
                        "case_id": r["case_id"],
                        "parent_case": r["parent_case"],
                        "gain_defined": False,
                        "original_amplitude": a0,
                        "reversed_amplitude": a1,
                    }
                )
                continue
            rows.append(
                {
                    "case_id": r["case_id"],
                    "parent_case": r["parent_case"],
                    "group_id": r["group_id"],
                    "entity": r["entity"],
                    "criterion": r["criterion"],
                    "gain_defined": True,
                    "original_amplitude": a0,
                    "reversed_amplitude": a1,
                    "original_shift": y0,
                    "reversed_shift": y1,
                    **account(a0, a1, y0, y1),
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "paired_components.csv", index=False)
        if len(frame) != 64 or not frame.gain_defined.all():
            raise ValueError("Incomplete/undefined full gain panel; rows retained")
        summaries = []
        for scope, keys in (
            ("pooled", ["criterion"]),
            ("setting", ["group_id", "criterion"]),
        ):
            for key, s in frame.groupby(keys):
                key = (key,) if not isinstance(key, tuple) else key
                unchanged = float(np.sum((s.reversed_shift - s.original_shift) ** 2))
                prediction = float(
                    np.sum((s.reversed_shift - s.original_gain_prediction) ** 2)
                )
                summaries.append(
                    {
                        **dict(zip(keys, key)),
                        "scope": scope,
                        "cases": len(s),
                        "change_sse": unchanged,
                        "amplitude_sse": float(np.sum(s.amplitude_term**2)),
                        "gain_sse": float(np.sum(s.gain_term**2)),
                        "signed_cross_term": float(
                            2 * np.sum(s.amplitude_term * s.gain_term)
                        ),
                        "original_gain_prediction_sse": prediction,
                        "zero_prediction_sse": float(np.sum(s.reversed_shift**2)),
                        "prediction_sse_over_unchanged": prediction / unchanged
                        if unchanged > 1e-12
                        else np.nan,
                    }
                )
        summary = pd.DataFrame(summaries)
        summary.to_csv(out / "gain_summary.csv", index=False)
        rest = pd.DataFrame(remainder)
        rest.to_csv(out / "remainder_components.csv", index=False)
        energy = []
        for scope, keys in (
            ("pooled", ["order", "criterion"]),
            ("setting", ["order", "group_id", "criterion"]),
        ):
            for key, s in rest.groupby(keys):
                energy.append(
                    {
                        **dict(zip(keys, key)),
                        "scope": scope,
                        "cases": len(s),
                        "projection_error_sse": float(np.sum(s.projection_error**2)),
                        "complement_sse": float(np.sum(s.complement**2)),
                        "interaction_sse": float(np.sum(s.interaction**2)),
                        "signed_cross_term": float(
                            2 * np.sum(s.complement * s.interaction)
                        ),
                    }
                )
        energy = pd.DataFrame(energy)
        energy.to_csv(out / "remainder_energy.csv", index=False)
        scalar = float(
            abs(frame.observed_change - frame.amplitude_term - frame.gain_term).max()
        )
        squared = float(
            abs(
                summary.change_sse
                - summary.amplitude_sse
                - summary.gain_sse
                - summary.signed_cross_term
            ).max()
        )
        residual = float(
            abs(
                energy.projection_error_sse
                - energy.complement_sse
                - energy.interaction_sse
                - energy.signed_cross_term
            ).max()
        )
        if (
            max(scalar, squared, residual) > 1e-9
            or len(summary) != 10
            or len(energy) != 20
            or time.monotonic() - start > 60
        ):
            raise ValueError("Full accounting/resource gate")
        status = {
            "status": "COMPLETE",
            "new_neural_calls": 0,
            "pairs": 64,
            "gain_summaries": 10,
            "remainder_summaries": 20,
            "scalar_max": scalar,
            "squared_max": squared,
            "remainder_max": residual,
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
    """Write a new immutable cache analysis, never overwrite the source runs."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original-dir", "reversed-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
