#!/usr/bin/env python3
"""M24B: test the old frozen functional matrix on new mean/configuration effects."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.coordinate_cube import (
    anchored_effects,
    query_cube,
)
from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    corr,
    read,
    sha,
    write,
)
from scripts.analyze_grain_conditional_edits import LEVEL, paired_metrics


def main():
    """Use reference coefficients fixed before these authors; no fitting or selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cube_dir, out = args.cube_dir.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(BASE) or not cube_dir.is_relative_to(BASE):
        raise ValueError("Use private result paths")
    if read(cube_dir / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("M24 cube incomplete")
    lm = read(LEVEL / "manifest.json")
    if sha(LEVEL / "reference.npz") != lm["reference_sha256"]:
        raise ValueError("Frozen reference changed")
    out.mkdir(parents=True, exist_ok=False)
    rows, metrics, links, hashes = [], [], [], {}
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as ref:
        for path in sorted((cube_dir / "private_packets").glob("unit_*.json")):
            d = read(path)
            old = LEVEL / "private_packets" / path.with_suffix(".npz").name
            if sha(old) != d["input_array_sha256"]:
                raise ValueError("M24 source states changed")
            hashes[str(path.relative_to(ROOT))] = sha(path)
            q = d["query"]
            with np.load(old, allow_pickle=False) as z:
                cube = query_cube(
                    z["full_P"], z["full_Q"], z["edited_P"], z["edited_Q"]
                )
            values = {
                k: float(np.sum(ref[f"{q}__G"] * p) + ref[f"{q}__g"] @ h)
                for k, (p, h) in cube.items()
            }
            pred = anchored_effects(values)
            actual = anchored_effects(d["values"])
            for effect in (
                "mean",
                "configuration",
                "final",
                "mean_configuration",
                "mean_final",
                "configuration_final",
                "triple",
            ):
                rows.append(
                    {
                        "author_id": d["author_id"],
                        "query": q,
                        "effect": effect,
                        "actual": actual[effect],
                        "prediction": pred[effect],
                    }
                )
            rows.append(
                {
                    "author_id": d["author_id"],
                    "query": q,
                    "effect": "conditional_delta",
                    "actual": d["values"]["111"] - d["values"]["000"],
                    "prediction": values["111"] - values["000"],
                }
            )
            rows.append(
                {
                    "author_id": d["author_id"],
                    "query": q,
                    "effect": "natural_delta",
                    "actual": d["natural_delta"],
                    "prediction": values["111"] - values["000"],
                }
            )
    frame = pd.DataFrame(rows)
    for (query, effect), g in frame.groupby(["query", "effect"]):
        metrics.append(
            {
                "query": query,
                "effect": effect,
                **paired_metrics(g.actual.to_numpy(), g.prediction.to_numpy()),
            }
        )
    for effect in (
        "mean",
        "configuration",
        "final",
        "conditional_delta",
        "natural_delta",
    ):
        g = frame[frame.effect == effect]
        for target in ("actual", "prediction"):
            p = g.pivot(index="author_id", columns="query", values=target)
            links.append(
                {
                    "effect": effect,
                    "target": target,
                    "authors": len(p),
                    "cross_query_r": corr(
                        p.TF_cont.to_numpy(), p.neuroticism.to_numpy()
                    ),
                    "cross_query_covariance": float(
                        np.cov(p.TF_cont, p.neuroticism, ddof=0)[0, 1]
                    ),
                }
            )
    frame.to_csv(out / "private_effects.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "reference_effect_metrics.csv", index=False)
    pd.DataFrame(links).to_csv(out / "functional_links.csv", index=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_M24B_FROZEN_REFERENCE_CONNECTION",
            "script_sha256": sha(__file__),
            "reference_sha256": lm["reference_sha256"],
            "cube_manifest_sha256": sha(cube_dir / "manifest.json"),
            "private_input_hashes": hashes,
            "fits": 0,
            "new_neural_calls": 0,
            "human_labels_read": False,
            "status": "COMPLETE",
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
