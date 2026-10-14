#!/usr/bin/env python3
"""F12 conditional DIF table from stored transport scores (no refits).

For each seed x channel world x calibration method (application authors):
per-condition bias of the transported score against the true old-unit
expected response a + b z, and the residual condition dependence left after the
best possible global affine renorm (least squares against truth, an oracle
upper bound on what any global renorm could achieve). If DIF survives the
oracle renorm, no new-domain norm can erase it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/grain_gptpro_foundation_v3/F12/transport_full_r1"
SEEDS = (20260915, 20260916, 20260917)
WORLDS = {"identity": ("uncalibrated", "global", "conditional"), "global": ("uncalibrated", "global", "conditional"),
          "conditional": ("uncalibrated", "global", "conditional"), "nonlinear": ("uncalibrated", "global", "conditional", "cubic")}


def main():
    """Write dif_table.csv and summary.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {}
    for seed in SEEDS:
        obs = np.load(RUN / f"observations_{seed}_application.npz")
        z = obs["conditions"]
        theta = np.load(RUN / f"truth_{seed}_application.npz")["theta"]
        expected = theta[:, [0]] + theta[:, [1]] * z[None, :]
        for world, methods in WORLDS.items():
            for method in methods:
                path = RUN / f"scores_{seed}_{world}_application_{method}.npz"
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                d = np.load(path)
                score, support = d["score"], d["support"]
                row = {"seed": seed, "world": world, "method": method, "supported_share": float(support.mean())}
                if support.sum() < 10:
                    row.update({"status": "NO_SUPPORTED_SCORES"})
                    rows.append(row)
                    continue
                s, e, zz = score[support], expected[support], np.broadcast_to(z, score.shape)[support]
                alpha, beta = np.linalg.lstsq(np.column_stack([np.ones_like(s), s]), e, rcond=None)[0]
                resid = alpha + beta * s - e
                biases, after = [], []
                for level in np.unique(z):
                    m = zz == level
                    biases.append(float(np.mean(s[m] - e[m])))
                    after.append(float(np.mean(resid[m])))
                    row[f"bias_z{level:+.3f}"] = biases[-1]
                    row[f"after_renorm_z{level:+.3f}"] = after[-1]
                row.update({"status": "OK", "dif_range": max(biases) - min(biases), "dif_range_after_oracle_renorm": max(after) - min(after),
                            "oracle_renorm_scale": float(beta), "oracle_renorm_offset": float(alpha)})
                rows.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("seed", "world", "method", "status"), k))
    with (args.output / "dif_table.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for world, methods in WORLDS.items():
        for method in methods:
            sel = [r for r in rows if r["world"] == world and r["method"] == method and r["status"] == "OK"]
            summary[f"{world}:{method}"] = {
                "seeds_ok": len(sel),
                "mean_dif_range": float(np.mean([r["dif_range"] for r in sel])) if sel else None,
                "mean_dif_range_after_oracle_renorm": float(np.mean([r["dif_range_after_oracle_renorm"] for r in sel])) if sel else None,
            }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (args.output / "input_manifest.json").write_text(json.dumps(hashes | {"code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
