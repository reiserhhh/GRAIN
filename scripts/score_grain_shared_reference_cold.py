#!/usr/bin/env python3
"""Cold numeric pair law: no native answers, application labels or fitting."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_shared_reference as M
from grain_research.construction import math_source_target_law as S


def compute(inputs, output):
    """Compute complete pair marginals of one shared-reference family."""
    start = time.monotonic()
    if output.exists():
        raise FileExistsError("Cold output already exists; no overwrite")
    with np.load(inputs, allow_pickle=False) as z:
        if set(z.files) != {"weights", "anchors", "authors", "edges"}:
            raise ValueError(
                "Only SOURCE laws, anchors and opaque graph addresses allowed"
            )
        w, y, authors, edges = (
            z[k] for k in ("weights", "anchors", "authors", "edges")
        )
    if (
        w.shape != (24, 3, 386)
        or y.shape != (386, 5)
        or not np.array_equal(authors, np.arange(24))
        or edges.shape != (32, 2)
        or edges.dtype.kind not in "iu"
        or np.any((edges < 0) | (edges >= 24))
    ):
        raise ValueError("Complete frozen SOURCE/author/target scope required")
    for arm in range(3):
        S.validate_laws(w[:, arm], y)
    k = M.kernels(y)
    shape = (3, 24, 24)
    pi = np.full(shape + (5,), 0.5)
    conditional = np.full_like(pi, np.nan)
    same = np.ones(shape)
    changed = np.zeros(shape)
    covariance = np.empty(shape + (5, 5))
    audit_rows = []
    for arm in range(3):
        for i in range(24):
            for j in range(i, 24):
                a = M.accounts(w[i, arm], w[j, arm], y, k)
                pi[arm, i, j] = a["probability"]
                conditional[arm, i, j] = a[
                    "conditional_probability_given_changed_source"
                ]
                same[arm, i, j] = same[arm, j, i] = a["same_source_probability"]
                changed[arm, i, j] = changed[arm, j, i] = a[
                    "changed_source_probability"
                ]
                covariance[arm, i, j] = a["cross_covariance"]
                if i != j:
                    pi[arm, j, i] = 1 - a["probability"]
                    conditional[arm, j, i] = 1 - conditional[arm, i, j]
                    covariance[arm, j, i] = a["cross_covariance"].T
                audit_rows.append(
                    [
                        arm,
                        i,
                        j,
                        a["row_mass_error"],
                        a["column_mass_error"],
                        a["mean_difference_error"],
                    ]
                )
        print(f"COLD shared-reference arm{arm}:300 joint pair laws", flush=True)
    independent = np.stack([S.superiority(w[:, a], y) for a in range(3)])
    with output.open("xb") as f:
        np.savez_compressed(
            f,
            shared_probability=pi,
            independent_probability=independent,
            same_source_probability=same,
            changed_source_probability=changed,
            conditional_probability_given_changed_source=conditional,
            cross_covariance=covariance,
            marginal_audit=np.asarray(audit_rows),
        )
    print(json.dumps({"seconds": time.monotonic() - start, "pairs": len(audit_rows)}))


def main():
    """Parse two explicit local numeric paths; never load a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    compute(args.inputs, args.output)


if __name__ == "__main__":
    main()
