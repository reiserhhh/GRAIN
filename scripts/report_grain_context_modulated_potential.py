#!/usr/bin/env python3
"""Publish NEXT64 aggregates and a labeled post-outcome source-geometry account."""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_context_modulated_potential import ARMS, OUT, manifest, old, path


def geometry_account(raw, standardized):
    """Account for the regular simplex induced by full sample-rank PCA whitening."""
    x, c = np.asarray(raw, float), np.asarray(standardized, float)
    n = len(x)
    if (
        x.ndim != 2
        or c.shape != (n, n - 1)
        or np.linalg.matrix_rank(x - x.mean(0)) != n - 1
    ):
        raise ValueError("Full centered sample rank required for this identity")
    dr, dc = pdist(x), pdist(c)
    return {
        "source_authors": n,
        "raw_pair_distance_min": float(dr.min()),
        "raw_pair_distance_max": float(dr.max()),
        "raw_pair_distance_cv": float(dr.std() / dr.mean()),
        "standardized_pair_distance_min": float(dc.min()),
        "standardized_pair_distance_max": float(dc.max()),
        "standardized_pair_distance_cv": float(dc.std() / dc.mean()),
        "simplex_gram_error": float(
            np.max(abs(c @ c.T - (n * np.eye(n) - np.ones((n, n)))))
        ),
        "expected_pair_distance": float(np.sqrt(2 * n)),
    }


def publish():
    """Keep private identities/individual effects out of every public artifact."""
    start = time.process_time()
    frozen_manifest = manifest(OUT)
    audit = old.read(OUT / "audit.json")
    summary = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Primary numerical audit required")
    dst = ROOT / "reports/grain_vector_mechanisms/context_modulated_potential_v1"
    dst.mkdir(exist_ok=True)
    files = {}
    for name, digest in summary["files"].items():
        if name.startswith("private_"):
            continue
        source = OUT / "analysis" / name
        if old.sha(source) != digest:
            raise ValueError("Aggregate changed")
        if set(pd.read_csv(source, nrows=0).columns) & {
            "author_key",
            "text",
            "user_id",
            "tau",
        }:
            raise ValueError("Private columns in public table")
        shutil.copyfile(source, dst / name)
        files[name] = old.sha(dst / name)
    old.frozen(OUT, "models")
    jobs = sorted(
        [j for j in old.read(OUT / "private_jobs.json") if j["cohort"] == "source4"],
        key=lambda j: j["index"],
    )
    rows = []
    for qi, q in enumerate(old.QUERIES):
        old.check_paths(
            frozen_manifest,
            [old.input_path(j["query_keys"][qi]) for j in jobs],
        )
        model = old.arrays(path(OUT, q, ARMS[2]))
        raw = np.array([old.pure_input(j["query_keys"][qi])[2] for j in jobs])
        c = ((raw - model["context_mean"]) @ model["context_basis"]) / model[
            "context_scale"
        ]
        rows.append({"query_name": q, **geometry_account(raw, c)})
    pd.DataFrame(rows).to_csv(dst / "context_geometry.csv", index=False)
    files["context_geometry.csv"] = old.sha(dst / "context_geometry.csv")
    old.write(
        dst / "publication.json",
        {
            "primary_commit": old.read(OUT / "manifest.json")["commit"],
            "audit": audit,
            "promotion": summary["promotion"],
            "criteria": summary["criteria"],
            "files": files,
            "post_outcome_analysis": "context_geometry only; no fit or changed predictions",
            "post_outcome_code": {
                str(p.relative_to(ROOT)): old.sha(p)
                for p in (
                    Path(__file__),
                    ROOT / "tests/test_grain_context_geometry_account.py",
                )
            },
            "publication_cpu_seconds": time.process_time() - start,
            "scope": "Model logit contrast and GRAIN directional responses; source4 in-fit, exposed new8 source-disjoint application; no human-Y results",
        },
    )
    print({"public_tables": len(files), "directory": str(dst)})


if __name__ == "__main__":
    publish()
