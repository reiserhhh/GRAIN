#!/usr/bin/env python3
"""Read-only native-access progress inspection; incomplete fields remain absent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_native_access_hm131 as R


def inspect(out):
    """Describe only complete predeclared fields; never select/fill missing cells."""
    m = R.check(out)
    status = R.H.load(out / "status.json")
    rows = {r["id"]: r for r in R.H.load(out / "00/rows.json")}
    design = R.H.load(out / "00/design.json")
    native = np.full((2, 6, 8, 2, 2, 2, 6, 4), np.nan)
    for d in design[: status["completed"]]:
        if d["noop"]:
            continue
        r, raw = rows[d["row"]], R.raw_unit(out, d["batch"])
        bank = m["spec"]["banks"].index(r["bank"])
        b, o = (r["polarity"] + 1) // 2, (r["acting_person"] + 1) // 2
        native[bank, r["family"], d["route"], b, o] = raw["readings"].reshape(2, 6, 4)[
            ::-1
        ]
    output = {
        "status": status,
        "complete_fields": [],
        "incomplete_fields": [],
        "route_components": [],
    }
    for bank, bank_id in enumerate(m["spec"]["banks"]):
        coefficients = {}
        for route in range(8):
            field = native[bank, :, route]
            if not np.isfinite(field).all():
                output["incomplete_fields"].append([bank_id, route])
                continue
            coefficients[route] = np.stack([R.M.characters(f) for f in field])
        full = coefficients.get(7)
        for route, beta in coefficients.items():
            for unit in range(4):
                value = beta[:, 7, :, unit]
                reference = full[:, 7, :, unit]
                norm, refnorm = np.linalg.norm(value), np.linalg.norm(reference)
                output["complete_fields"].append(
                    {
                        "bank": bank_id,
                        "route": route,
                        "unit": R.UNITS[unit],
                        "qualified_RMS": float(np.sqrt(np.mean(value**2))),
                        "common_RMS": float(np.sqrt(np.mean(beta[:, 3, :, unit] ** 2))),
                        "norm_ratio_to_full": float(norm / max(refnorm, 1e-30)),
                        "cosine_to_full": float(
                            np.sum(value * reference) / max(norm * refnorm, 1e-30)
                        ),
                        "diagonal": np.diag(value).tolist(),
                    }
                )
        if len(coefficients) == 8:
            fields = np.stack([coefficients[r][:, 7] for r in range(8)])
            parts = R.M.mobius(fields)
            for component, unit in np.ndindex(8, 4):
                value, reference = parts[component, :, :, unit], fields[7, :, :, unit]
                output["route_components"].append(
                    {
                        "bank": bank_id,
                        "component": component,
                        "unit": R.UNITS[unit],
                        "qualified_RMS": float(np.sqrt(np.mean(value**2))),
                        "signed_alignment_to_full": float(
                            np.sum(value * reference) / max(np.sum(reference**2), 1e-30)
                        ),
                    }
                )
    return output


def main():
    """Print complete field observations without creating or changing artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/native_access_r2",
    )
    args = parser.parse_args()
    print(json.dumps(inspect(args.out), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
