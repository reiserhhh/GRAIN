#!/usr/bin/env python3
"""Check finite response signatures using polynomial differences, not subtraction."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    """Audit every signature and summary including deliberately absent cubic scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.input / "protocol_before_run.json").read_text())
    source = Path(protocol["input"])
    edits = dict(zip(protocol["edit_names"], protocol["edits"], strict=True))
    theta = {}
    for seed in protocol["seeds"]:
        with np.load(source / f"truth_{seed}_application.npz") as z:
            theta[str(seed)] = z["theta"]
    models = {}
    groups = {}
    maximum = 0.0
    count = 0
    missing = 0
    with (args.input / "signatures.csv").open() as stream:
        for row in csv.DictReader(stream):
            a, b = theta[row["seed"]][int(row["author_index"])]
            z = float(row["condition"])
            da, db = edits[row["edit"]]
            r = a + b * z
            delta = da + db * z
            maximum = max(maximum, abs(delta - float(row["target_signature"])))
            world, method = row["world"], row["method"]
            changed = {
                "identity": delta,
                "global": 1.5 * delta,
                "conditional": (1 + 0.25 * z) * delta,
                "nonlinear": delta
                + 0.1 * (3 * r * r * delta + 3 * r * delta * delta + delta**3),
            }[world]
            key = (row["seed"], world, method)
            if method != "uncalibrated":
                if key not in models:
                    models[key] = json.loads(
                        (source / f"calibration_{'_'.join(key)}.json").read_text()
                    )
                model = models[key]
                if model["status"] != "READY":
                    assert row["supported"] == "False" and row["signature"] == ""
                    missing += 1
                    count += 1
                    groups.setdefault((world, method), [])
                    continue
                c = model["coefficients"]
                if method == "global":
                    changed /= c[1]
                elif method == "conditional":
                    changed /= c[2] + c[3] * z
                else:
                    raise ValueError(
                        "unexpected supported cubic; audit requires explicit polynomial composition"
                    )
            assert row["supported"] == "True"
            maximum = max(maximum, abs(changed - float(row["signature"])))
            groups.setdefault((world, method), []).append((changed - delta) ** 2)
            count += 1
    with (args.input / "metrics.csv").open() as stream:
        for row in csv.DictReader(stream):
            values = groups[row["world"], row["method"]]
            assert int(row["supported"]) == len(values)
            if values:
                maximum = max(
                    maximum, abs(np.mean(values) - float(row["signature_mse"]))
                )
            else:
                assert row["signature_mse"] == ""
    if maximum > 1e-10:
        raise ValueError("signature replay failed")
    summary = {
        "status": "PASS_ALL_SIGNATURES_POLYNOMIAL_REPLAY",
        "rows": count,
        "missing": missing,
        "max_difference": maximum,
        "independent_agent": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
