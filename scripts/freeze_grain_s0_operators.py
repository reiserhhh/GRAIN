#!/usr/bin/env python3
"""Freeze S0 operators before scoring the fresh S1 common-edit panel."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.analyze_grain_functional_basis import (
    BASE,
    METHODS,
    candidate_predictions,
    fit_estimator,
    response_basis,
)
from scripts.analyze_grain_real_author_path import load_packet
from scripts.run_grain_real_author_path import sha, write


def main():
    """Use existing development response targets only; never read an S1 packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Keep private fitted arrays in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        BASE / name for name in ("d2_real_author_path_r1", "d3b_fresh_author_path_r1")
    ]
    loaded = [load_packet(source) for source in sources]
    rows = pd.concat([item[1] for item in loaded], ignore_index=True)
    x, y = (
        np.vstack([item[2] for item in loaded]),
        np.concatenate([item[3] for item in loaded]),
    )
    selected = []
    with threadpool_limits(limits=2):
        for task in sorted(rows.task.unique()):
            ix = (rows.task == task).to_numpy()
            for method in METHODS:
                predictions = candidate_predictions(
                    x[ix],
                    y[ix],
                    rows.author_id.to_numpy()[ix],
                    method,
                    "s0_final_development_cv",
                )
                rank, ridge = min(
                    predictions,
                    key=lambda spec: (np.mean((predictions[spec] - y[ix]) ** 2), spec),
                )
                basis = response_basis(x[ix], y[ix], method, max(rank, 1))
                model = fit_estimator(x[ix], y[ix], basis, ridge)
                file = out / f"{task}_{method}.npz"
                np.savez_compressed(
                    file,
                    basis=np.empty((x.shape[1], 0)) if basis is None else basis,
                    full=np.asarray(basis is None),
                    scale=model["scale"],
                    coef=model["coef"],
                )
                selected.append(
                    {
                        "task": task,
                        "method": method,
                        "rank": rank,
                        "ridge": ridge,
                        "inner_mse": float(
                            np.mean((predictions[rank, ridge] - y[ix]) ** 2)
                        ),
                        "file": file.name,
                        "sha256": sha(file),
                    }
                )
    write(
        out / "manifest.json",
        {
            "status": "FROZEN_BEFORE_S1_RESPONSES",
            "models": selected,
            "script_sha256": sha(__file__),
            "authors": rows.author_id.nunique(),
            "basis_code_sha256": sha(
                Path(__file__).with_name("analyze_grain_functional_basis.py")
            ),
            "source_manifest_hashes": {
                str(p): sha(p / "manifest.json") for p in sources
            },
            "psychological_labels_read": False,
        },
    )
    print(
        pd.DataFrame(selected).drop(columns=["sha256"]).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
