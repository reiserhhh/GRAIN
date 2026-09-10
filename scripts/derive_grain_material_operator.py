#!/usr/bin/env python3
"""Replay all frozen local covariances through one operator per source author/view."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.material_operator import (
    build_operator,
    read_operator,
)
from scripts.run_grain_material_coupling import FACTORS, FC, SLICES, SOURCE
from scripts.run_grain_material_geometry import PREVIOUS, source_feature_index, verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def derive(out, name):
    """Construct shared operators and numerical replay without reading human labels."""
    verify(out)
    freeze = read(out / "prediction_freeze.json")["files"]
    for n, expected in freeze.items():
        if sha(out / n) != expected:
            raise ValueError("Geometry statistics changed")
    dest = out / name
    dest.mkdir(exist_ok=False)
    plan = ROOT / "docs/GRAIN_MATERIAL_OPERATOR_EXTENSION_V1.md"
    helper = ROOT / "grain_research/vector_mechanisms/material_operator.py"
    write(
        dest / "manifest.json",
        {
            "plan_sha256": sha(plan),
            "code_sha256": sha(Path(__file__)),
            "helper_sha256": sha(helper),
            "geometry_freeze_sha256": sha(out / "prediction_freeze.json"),
            "label_reads": 0,
            "fits": 0,
            "new_neural_calls": 0,
        },
    )
    start = time.monotonic()
    old = pd.read_csv(out / "private_geometry_views.csv", dtype={"author_id": str})
    meta = pd.read_csv(SOURCE / "private_feature_rows.csv", dtype={"author_id": str})
    meta = source_feature_index(
        meta, old[["author_id", "view"]].itertuples(index=False, name=None)
    )
    slices = pd.read_csv(SLICES / "private_factor_slices.csv", dtype={"author_id": str})
    means = slices.groupby(["author_id", "view", "comment"], sort=True)[FC].mean()
    operators, rows, mapping = [], [], {}
    with np.load(SOURCE / "private_features.npz", allow_pickle=False) as z:
        flat, edges = z["bags"], z["offsets"]
        for (uid, view), g in means.groupby(level=[0, 1], sort=True):
            g = g.sort_index()
            comment = g.index.get_level_values("comment").to_numpy(int)
            index = int(meta.loc[(uid, view), "feature_index"])
            h = flat[edges[index] : edges[index + 1]][comment]
            c = build_operator(g.to_numpy(float), h)
            if c.shape != (4, 1024):
                raise ValueError("Unexpected local operator shape")
            mapping[(uid, view)] = len(rows)
            operators.append(c)
            rows.append(
                {"author_id": uid, "view": view, "eligible_comments": len(comment)}
            )
    count, error = 0, 0.0
    with np.load(PREVIOUS / "private_local_maps.npz", allow_pickle=False) as weights:
        for (uid, view, q, fold), group in old.groupby(
            ["author_id", "view", "query", "fold"], sort=True
        ):
            value = read_operator(
                operators[mapping[(uid, view)]], weights[f"{q}__{fold}__coef"]
            )
            expected = group.set_index("factor").loc[FACTORS, "S"].to_numpy()
            error = max(error, float(np.max(abs(value - expected))))
            count += len(value)
    if len(rows) != 828 or count != 15804 or error > 1e-12:
        raise ValueError("Full source operator replay failed")
    np.savez_compressed(dest / "private_operators.npz", operators=np.stack(operators))
    pd.DataFrame(rows).to_csv(dest / "private_operator_rows.csv", index=False)
    write(
        dest / "summary.json",
        {
            "status": "COMPLETE",
            "operators": len(rows),
            "operator_shape": [4, 1024],
            "query_views": count // 4,
            "factor_values_replayed": count,
            "replay_max": error,
            "query_independent_construction": True,
            "fits": 0,
            "label_reads": 0,
            "new_neural_calls": 0,
            "seconds": time.monotonic() - start,
            "files": {
                n: sha(dest / n)
                for n in ["private_operators.npz", "private_operator_rows.csv"]
            },
        },
    )
    print(read(dest / "summary.json"), flush=True)


def main():
    """Keep each no-fit construction in its own immutable private subdirectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_material_geometry_v1_r2"
    )
    parser.add_argument("--name", default="operator_r1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (
        not out.is_relative_to(BASE.resolve())
        or not args.name.replace("_", "").isalnum()
    ):
        raise ValueError("Use ignored results and simple output name")
    derive(out, args.name)


if __name__ == "__main__":
    main()
