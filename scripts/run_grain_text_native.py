#!/usr/bin/env python3
"""Native text-package bounded author-held-out reference with cached E5 inputs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.predecessors import TRAITS
from scripts.prepare_grain_predecessor_reference import DEFAULT, digest, write


def aligned_oof(path, author_ids, folds, y):
    """Reject cohort/order drift before comparing previous predictions."""
    with np.load(path, allow_pickle=False) as z:
        for key, expected in (("author_ids", author_ids), ("folds", folds), ("y", y)):
            if not np.array_equal(z[key], expected):
                raise ValueError("Prior OOF cohort/order differs: " + key)
        prediction = z["semantic_pca_ridge"].copy()
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prior OOF predictions")
    return prediction


def run(args):
    """Train only within four development folds and evaluate the fifth once."""
    if args.audit_existing:
        with np.load(args.material / "private_arrays.npz", allow_pickle=False) as a:
            aligned_oof(args.local_oof, a["author_ids"], a["folds"], a["y"])
        write(
            args.output / "alignment_audit.json",
            {
                "status": "PASS",
                "source_arrays_sha256": digest(args.material / "private_arrays.npz"),
                "prior_oof_sha256": digest(args.local_oof),
                "identities_folds_labels_equal": True,
            },
        )
        print("Verified prior OOF identities, folds and labels")
        return
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    write(
        args.output / "contract.json",
        {
            "input": "same frozen mean E5 embeddings as predecessor reference",
            "outer_held_fold": 0,
            "native_source_cv": "3 outer x 2 inner, per-trait RMSE selection",
            "preprocess": "native center/scale then PCA64 within each fit",
            "penalty": [0.001, 0.01, 0.1, 1],
            "mixture": 0,
            "seed": 731,
            "status_scope": "single-fold native capability reference, not full CV or new lockbox",
            "comparability": "same held authors/input, different preprocessing/grid/selection from local Python adapter",
            "material_sha256": digest(args.material / "material_manifest.json"),
            "code_sha256": digest(Path(__file__)),
            "R_sha256": digest(ROOT / "scripts/fit_grain_text_native.R"),
            "prior_oof_sha256": digest(args.local_oof),
        },
    )
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(args.material / name) != sha:
            raise ValueError("Changed material")
    with np.load(args.material / "private_arrays.npz", allow_pickle=False) as a:
        x, y, folds = a["comments"].mean(1), a["y"], a["folds"]
        prediction = aligned_oof(args.local_oof, a["author_ids"], folds, y)
    pd.DataFrame(x).to_csv(args.output / "private_semantic.csv", index=False)
    pd.DataFrame(y, columns=TRAITS).to_csv(
        args.output / "private_labels.csv", index=False
    )
    pd.DataFrame({"fold": folds}).to_csv(args.output / "private_folds.csv", index=False)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        VECLIB_MAXIMUM_THREADS="2",
    )
    with (args.output / "native.log").open("w") as log:
        try:
            completed = subprocess.run(
                [
                    "/usr/local/bin/Rscript",
                    str(ROOT / "scripts/fit_grain_text_native.R"),
                    str(args.output.resolve()),
                    str(args.rlib.resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            write(
                args.output / "summary.json",
                {
                    "status": "STOPPED_BUDGET",
                    "seconds": 600,
                    "partial_outputs_retained": True,
                },
            )
            raise
    if completed.returncode:
        raise RuntimeError("Native text failed; inspect native.log")
    rows = pd.read_csv(args.output / "metrics.csv")
    # Prior OOF results on precisely these already-defined held-out authors.
    local = [
        float(np.corrcoef(y[folds == 0, j], prediction[folds == 0, j])[0, 1])
        for j in range(5)
    ]
    rows["local_adapter_r_same_fold"] = local
    rows.to_csv(args.output / "comparison.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "held_authors": int((folds == 0).sum()),
        "native_mean_r": float(rows.r.mean()),
        "local_adapter_same_fold_mean_r": float(np.mean(local)),
        "interpretation": "implementation reference, not an optimized or protocol-identical superiority test",
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rlib", type=Path, required=True)
    p.add_argument("--audit-existing", action="store_true")
    p.add_argument(
        "--local-oof",
        type=Path,
        default=DEFAULT.parent / "comparison_r2/private_oof.npz",
    )
    run(p.parse_args())
