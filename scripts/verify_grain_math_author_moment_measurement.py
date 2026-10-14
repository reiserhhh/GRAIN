#!/usr/bin/env python3
"""HM62 alternate direct integration and primal/dual proof checks; no new LPs."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_author_moment_measurement import OUT, check
from scripts.run_grain_math_construct_routing import write


def verify(out):
    """Certify optimum using saved feasible primal/dual objects, not solver flags."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete finite HM62 required")
    for name, sha in read_json(out / "03/manifest.json")["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen HM62 result drift")
    barrier = read_json(out / "01/source_barrier.json")
    if (
        barrier["linear_programs"] != 96
        or digest(out / "01/source_witnesses.npz") != barrier["file_sha256"]
    ):
        raise ValueError("Source witness barrier mismatch")
    bits = np.asarray(list(itertools.product((0, 1), repeat=6)))
    x, degree = 2 * bits - 1, bits.sum(1)
    w = np.asarray([np.prod(x[:, s.astype(bool)], axis=1) for s in bits]).T
    targets, tables = m["targets"], {}
    with (ROOT / m["spec"]["source"] / "03/native.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            key = int(r["family"]), r["query"]
            for unit in ("score", "log_odds"):
                tables.setdefault((key[0], unit), np.zeros((64, 6)))[
                    int(r["corner"]), targets.index(key[1])
                ] = float(r[unit])
    with np.load(out / "00/native_tables.npz", allow_pickle=False) as arrays:
        for (f, u), v in tables.items():
            if not np.array_equal(v, arrays[f"family{f}_{u}"]):
                raise ValueError("Native-to-analysis binding failed")
    maximum, count = 0.0, 0

    def compare(a, b):
        """Record actual finite alternate differences with shape binding."""
        nonlocal maximum, count
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Alternate shape/value failure")
        maximum = max(maximum, float(np.max(np.abs(a - b), initial=0)))
        count += a.size

    strength = m["spec"]["parity_strength"]
    for name in ("profiles", "contrasts"):
        with (out / f"03/{name}.csv").open(newline="") as stream:
            for r in csv.DictReader(stream):
                f, u, j, q = (
                    int(r["family"]),
                    r["unit"],
                    int(r["subset"]),
                    targets.index(r["query"]),
                )
                if (
                    r["mask"] != "".join(map(str, bits[j]))
                    or int(r["order"]) != degree[j]
                ):
                    raise ValueError("Wrong subset binding")
                v = tables[f, u][:, q]
                pp, pm = (1 + strength * w[:, j]) / 64, (1 - strength * w[:, j]) / 64
                a = w[:, degree < degree[j]].T
                compare(a @ pp, a @ pm)
                if name == "profiles":
                    p = pp if int(r["side"]) == 1 else pm
                    compare(p @ v, float(r["mean_reading"]))
                    c = np.asarray([np.mean(v * w[:, k]) for k in range(64)])
                    compare(
                        (w.T @ p)[degree <= 1] @ c[degree <= 1],
                        float(r["first_moment_only"]),
                    )
                else:
                    compare((pp - pm) @ v, float(r["contrast"]))
                    compare(
                        2 * strength * np.mean(v * w[:, j]),
                        float(r["coefficient_formula"]),
                    )
    with np.load(out / "03/witnesses.npz", allow_pickle=False) as arrays:
        saved = {key: arrays[key].copy() for key in arrays.files}
    with np.load(out / "01/source_witnesses.npz", allow_pickle=False) as source:
        for key in source.files:
            compare(source[key], saved[key])
    bound_rows = list(csv.DictReader((out / "03/bounds.csv").open(newline="")))
    for r in bound_rows:
        f, u, q, order = (
            int(r["family"]),
            r["unit"],
            targets.index(r["query"]),
            int(r["order"]),
        )
        a, v = w[:, degree <= order].T, tables[f, u][:, q]
        mu = a @ np.full(64, 1 / 64)
        for side in ("lower", "upper"):
            key = f"family{f}_{u}_q{q}_order{order}_{side}"
            p, lam = saved[key + "_p"], saved[key + "_dual"]
            compare(a @ p, mu)
            if p.min() < -1e-8:
                raise ValueError("Negative witness mass")
            slack = v - a.T @ lam if side == "lower" else a.T @ lam - v
            if slack.min() < -1e-8:
                raise ValueError("Invalid dual moment bound")
            compare(v @ p, mu @ lam)
            compare(v @ p, float(r[side]))
        compare(float(r["upper"]) - float(r["lower"]), float(r["width"]))
        compare(v.mean(), float(r["uniform_mean"]))
    with (out / "03/source_witness_transfer.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            u, q, order = r["unit"], targets.index(r["query"]), int(r["order"])
            key = f"family0_{u}_q{q}_order{order}"
            dp = saved[key + "_upper_p"] - saved[key + "_lower_p"]
            compare(dp @ tables[0, u][:, q], float(r["source_width"]))
            compare(
                dp @ tables[1, u][:, q],
                float(r["application_contrast_on_same_distributions"]),
            )
    with (
        np.load(
            ROOT / m["spec"]["source"] / "03/coefficients.npz", allow_pickle=False
        ) as old,
        (out / "03/vector_contrasts.csv").open(newline="") as stream,
    ):
        for r in csv.DictReader(stream):
            c = old[f"family{r['family']}_{r['stage']}_coefficients"][int(r["subset"])]
            compare(
                2 * strength * np.sqrt(np.sum(c**2)),
                float(r["paired_mean_vector_displacement_norm"]),
            )
    if maximum > m["spec"]["tolerance"] or len(bound_rows) != 96:
        raise ArithmeticError("HM62 incomplete or inaccurate numerical certificate")
    result = {
        "status": "VERIFIED_FINITE_HM62_NOT_GLOBAL",
        "maximum_alternate_error": maximum,
        "compared_values": count,
        "certified_intervals": 96,
        "source_then_application_analysis": True,
        "new_model_calls": 0,
        "same_agent_alternate_arithmetic_not_independent_replication": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result)


if __name__ == "__main__":
    verify(OUT)
