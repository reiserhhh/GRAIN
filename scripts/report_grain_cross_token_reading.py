#!/usr/bin/env python3
"""Publish aggregate NEXT67 results and a post-outcome paired-query account."""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_cross_token_reading import OUT, old, verify


def paired_account(left, right):
    """Equal-author average energies; query half-difference is not unique personality variance."""
    if not left or len(left) != len(right):
        raise ValueError("Paired query vectors required")
    rows = []
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Identical finite vector layout required")
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0:
            raise ValueError("Nonzero vectors required for orientation")
        common, difference = (a + b) / 2, (a - b) / 2
        rows.append(
            [
                np.mean(a * a),
                np.mean(b * b),
                np.mean(common * common),
                np.mean(difference * difference),
                np.sum(a * b) / denom,
            ]
        )
    x = np.array(rows)
    means = x.mean(0)
    total = (means[0] + means[1]) / 2
    return {
        "authors": len(left),
        "left_energy": float(means[0]),
        "right_energy": float(means[1]),
        "common_energy": float(means[2]),
        "difference_energy": float(means[3]),
        "difference_fraction": float(means[3] / total),
        "author_cosine_mean": float(means[4]),
        "author_cosine_min": float(x[:, 4].min()),
        "author_cosine_max": float(x[:, 4].max()),
        "energy_identity_error": float(abs(total - means[2] - means[3])),
    }


def publish():
    """Copy only aggregates, then compare paired queries on byte-identical BODY inputs."""
    start = time.process_time()
    m = verify(OUT)
    audit = old.read(OUT / "audit.json")
    summary = old.read(OUT / "analysis/summary.json")
    if audit["status"] != "PASS":
        raise ValueError("Primary audit required")
    old.verify_hashes(OUT / "analysis", summary["files"])
    dst = ROOT / "reports/grain_vector_mechanisms/cross_token_reading_v1"
    dst.mkdir(exist_ok=True)
    hashes = {}
    for name, digest in summary["files"].items():
        if name.startswith("private_"):
            continue
        src = OUT / "analysis" / name
        if set(pd.read_csv(src, nrows=0).columns) & {"author_key", "text", "user_id"}:
            raise ValueError("Private field in public table")
        shutil.copyfile(src, dst / name)
        hashes[name] = digest
    jobs = old.read(OUT / "private_jobs.json")
    banks = {"B_gradient": [[], []], "B_GRAIN_action": [[], []]}
    for j in jobs:
        materials, directions = [], []
        for qi, k in enumerate(j["query_keys"]):
            data, body, _ = old.pure_input(k)
            ds = old.body_directions(data)[:, data["positions"]]
            cut = len(body) // 2
            materials.append(body)
            directions.append(ds)
            pair = [
                old.arrays(OUT / "capture" / (k + "_" + s + ".npz"))[
                    "body_gradient"
                ].astype(float)
                for s in ("plus", "minus")
            ]
            delta = (pair[0][cut:] - pair[1][cut:]) / 2
            banks["B_gradient"][qi].append(delta)
            banks["B_GRAIN_action"][qi].append(
                np.einsum("th,fth->f", delta, ds[:, cut:])
            )
        np.testing.assert_array_equal(materials[0], materials[1])
        np.testing.assert_array_equal(directions[0], directions[1])
    records = []
    for channel, args in banks.items():
        record = paired_account(*args)
        if record != paired_account(*args):
            raise ValueError("Paired arithmetic replay changed")
        records.append(
            {
                "channel": channel,
                "left_query": old.QUERIES[0],
                "right_query": old.QUERIES[1],
            }
            | record
        )
    p = dst / "paired_query_account.csv"
    pd.DataFrame(records).to_csv(p, index=False)
    hashes[p.name] = old.sha(p)
    old.write(
        dst / "publication.json",
        {
            "primary_commit": m["commit"],
            "audit": audit,
            "files": hashes,
            "scope": "source4, two fixed model queries, no human labels or new8 use",
            "post_outcome_scope": "paired-query source account only; byte-identical BODY and GRAIN directions, equal-author energy",
            "post_outcome_code_sha256": old.sha(Path(__file__)),
            "post_outcome_test_sha256": old.sha(
                ROOT / "tests/test_grain_cross_token_account.py"
            ),
            "post_outcome_cpu_seconds": time.process_time() - start,
            "new_neural_calls_or_fits": 0,
        },
    )
    print(
        {"status": "PUBLISHED", "tables": len(hashes), "paired_query_account": records},
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        publish()
