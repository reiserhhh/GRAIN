#!/usr/bin/env python3
"""Finite action signatures under coordinate and reader-channel transport."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grain_research.gptpro_foundation_v3.f12_transport import transport_score


def channel(value, z, name):
    """Apply the unchanged four-world F12 observation channel."""
    if name == "identity":
        return value
    if name == "global":
        return 1 + 1.5 * value
    if name == "conditional":
        return z + (1 + 0.25 * z) * value
    return value + 0.1 * value**3


def main():
    """Freeze operational details, then compare four finite interventions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol_before_run.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    root = Path(config["input"])
    z = np.array(config["conditions"])
    edits = np.array(config["edits"])
    x = np.column_stack([np.ones(4), z])
    angle = config["rotation_radians"]
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    s = rotation @ np.diag(config["coordinate_scales"])
    rows = []
    max_coordinate = 0.0
    max_grid = 0.0
    off_grid = []
    for seed in config["seeds"]:
        with np.load(root / f"truth_{seed}_application.npz") as stored:
            theta = stored["theta"]
        h = theta @ s.T
        new_x = x @ np.linalg.inv(s)
        baseline = theta @ x.T
        max_coordinate = max(
            max_coordinate, float(np.max(np.abs(h @ new_x.T - baseline)))
        )
        for edit_index, edit in enumerate(edits):
            changed = (theta + edit) @ x.T
            signature = changed - baseline
            rotated = (h + edit @ s.T) @ new_x.T - h @ new_x.T
            max_coordinate = max(
                max_coordinate, float(np.max(np.abs(signature - rotated)))
            )
            polynomial = np.prod(z[:, None] - z[None, :], axis=1)
            altered = changed + 0.2 * (theta[:, 0] + edit[0])[:, None] * polynomial
            altered_base = baseline + 0.2 * theta[:, 0, None] * polynomial
            max_grid = max(
                max_grid, float(np.max(np.abs((altered - altered_base) - signature)))
            )
            # At an unobserved condition the same four actions can disagree.
            off_grid.append(float(0.2 * edit[0] * np.prod(-z)))
            for world in ("identity", "global", "conditional", "nonlinear"):
                before = channel(baseline, z, world)
                after = channel(changed, z, world)
                methods = ["uncalibrated", "global", "conditional"] + (
                    ["cubic"] if world == "nonlinear" else []
                )
                for method in methods:
                    if method == "uncalibrated":
                        observed = after - before
                        valid = np.ones_like(observed, dtype=bool)
                    else:
                        model = json.loads(
                            (
                                root / f"calibration_{seed}_{world}_{method}.json"
                            ).read_text()
                        )
                        for field in ("coefficients", "conditions"):
                            if field in model:
                                model[field] = np.array(model[field])
                        b, b_ok = transport_score(
                            model, before, np.broadcast_to(z, before.shape)
                        )
                        a, a_ok = transport_score(
                            model, after, np.broadcast_to(z, after.shape)
                        )
                        observed = a - b
                        valid = a_ok & b_ok
                    for i in range(len(theta)):
                        for j in range(4):
                            rows.append(
                                {
                                    "seed": seed,
                                    "author_index": i,
                                    "edit": config["edit_names"][edit_index],
                                    "condition": z[j],
                                    "world": world,
                                    "method": method,
                                    "target_signature": signature[i, j],
                                    "supported": bool(valid[i, j]),
                                    "signature": float(observed[i, j])
                                    if valid[i, j]
                                    else None,
                                }
                            )
    metrics = []
    for world, method in dict.fromkeys((r["world"], r["method"]) for r in rows):
        group = [r for r in rows if (r["world"], r["method"]) == (world, method)]
        valid = [r for r in group if r["supported"]]
        metrics.append(
            {
                "world": world,
                "method": method,
                "rows": len(group),
                "supported": len(valid),
                "signature_mse": float(
                    np.mean(
                        [(r["signature"] - r["target_signature"]) ** 2 for r in valid]
                    )
                )
                if valid
                else None,
            }
        )
    for name, values in (("signatures.csv", rows), ("metrics.csv", metrics)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    if max_coordinate > 1e-10 or max_grid > 1e-12:
        raise ValueError("coordinate certificate failed")
    result = {
        "status": "FINITE_SIGNATURES_AND_NONIDENTITY_COUNTEREXAMPLE_COMPUTED",
        "rows": len(rows),
        "max_coordinate_error": max_coordinate,
        "same_grid_counterexample_error": max_grid,
        "off_grid_signature_difference_max": max(abs(v) for v in off_grid),
        "signature_design_rank": int(np.linalg.matrix_rank(edits @ x.T)),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "scope": "finite model interventions; four signed actions span only two independent directions",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
