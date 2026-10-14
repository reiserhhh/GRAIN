#!/usr/bin/env python3
"""Recompute HM61 coefficients, links, edges and source transfer independently."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_heterogeneous_evidence_cube import (
    OUT,
    check,
    read_tables,
    seal_source,
)


def verify(out):
    """Use direct summation and SciPy links; a verifier is not human replication."""
    m = check(out)
    files = read_json(out / "03/manifest.json")["files"]
    for name, sha in files.items():
        if digest(out / name) != sha:
            raise ValueError("Frozen result drift " + name)
    rows = read_json(out / "00/rows.json")
    seal_source(out, rows)
    barrier = read_json(out / "01/source_barrier.json")
    if (
        max(
            read_json(out / f"01/units/unit_{r['unit']:04d}.json")["finished_epoch"]
            for r in rows
            if r["family"] == 0
        )
        > barrier["created_epoch"]
    ):
        raise ValueError("Source barrier precedes its captures")
    if (
        min(
            read_json(out / f"01/units/unit_{r['unit']:04d}.json")["finished_epoch"]
            for r in rows
            if r["family"] == 1
        )
        < barrier["created_epoch"]
    ):
        raise ValueError("Application preceded source coefficient freeze")
    tables, _, prefix_error = read_tables(out, rows)
    bits = np.array(list(itertools.product((0, 1), repeat=6)))
    x = 2 * bits - 1
    w = np.prod(x[:, None, :] ** bits[None, :, :], axis=2)
    order = bits.sum(1)
    with np.load(out / "03/coefficients.npz", allow_pickle=False) as f:
        coef = {k: f[k].copy() for k in f.files}
    error, count = 0.0, 0

    def compare(a, b):
        """Accumulate actual differences, including shape agreement and finiteness."""
        nonlocal error, count
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Bad alternate comparison")
        error = max(error, float(np.max(np.abs(a - b), initial=0)))
        count += a.size

    raw, altcoef, edges = {}, {}, {}
    for family, table in tables.items():
        z = table["logits"]
        zz = z - z.mean(-1, keepdims=True)
        raw[family] = {
            "score": softmax(z, axis=-1) @ (np.arange(10) / 9),
            "log_odds": logsumexp(z[:, :, 5:], axis=-1)
            - logsumexp(z[:, :, :5], axis=-1),
        }
        stages = {
            "centered_logits": zz,
            "body_endpoint": table["body"],
            "query_endpoint": table["endpoint"],
            **raw[family],
        }
        for stage, v in stages.items():
            c = np.asarray(
                [
                    np.mean(v * row.reshape((64,) + (1,) * (v.ndim - 1)), axis=0)
                    for row in w.T
                ]
            )
            altcoef[family, stage] = c
            compare(c, coef[f"family{family}_{stage}_coefficients"])
            reconstructed = np.tensordot(w, c, axes=(1, 0))
            compare(reconstructed, v)
            compare(np.sum(c**2), np.sum(v**2) / 64)
        za = np.tensordot(
            w[:, order <= 1],
            altcoef[family, "centered_logits"][order <= 1],
            axes=(1, 0),
        )
        link = {
            "score": softmax(za, axis=-1) @ (np.arange(10) / 9),
            "log_odds": logsumexp(za[:, :, 5:], axis=-1)
            - logsumexp(za[:, :, :5], axis=-1),
        }
        for unit in ("score", "log_odds"):
            control_coef = np.tensordot(w.T / 64, link[unit], axes=(1, 0))
            compare(
                control_coef, coef[f"family{family}_{unit}_link_control_coefficients"]
            )
            edges[family, unit] = np.asarray(
                [
                    [
                        raw[family][unit][k + 2 ** (5 - j)] - raw[family][unit][k]
                        for k in range(64)
                        if x[k, j] < 0
                    ]
                    for j in range(6)
                ]
            )
    with (out / "03/edges.csv").open(newline="") as stream:
        edge_rows = list(csv.DictReader(stream))
    targets = m["spec"]["targets"]
    for r in edge_rows:
        expected = edges[int(r["family"]), r["unit"]][
            targets.index(r["content"]),
            int(r["condition_index"]),
            targets.index(r["query"]),
        ]
        compare(expected, float(r["contrast"]))
    with (out / "03/source_transfer.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            unit, degree, q = r["unit"], int(r["order"]), targets.index(r["query"])
            c = altcoef[0, unit]
            p = np.tensordot(w[:, order <= degree], c[order <= degree], axes=(1, 0))
            pred = np.asarray(
                [
                    [p[k + 2 ** (5 - j), q] - p[k, q] for k in range(64) if x[k, j] < 0]
                    for j in range(6)
                ]
            ).ravel()
            actual = edges[1, unit][:, :, q].ravel()
            compare(np.sqrt(np.mean((actual - pred) ** 2)), float(r["edge_rms_error"]))
    with (out / "03/geometry.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            family, unit = int(r["family"]), r["unit"]
            c = altcoef[family, unit]
            j = np.zeros((64, 6, 6))
            for corner in range(64):
                for variable in range(6):
                    for term in range(64):
                        if bits[term, variable]:
                            others = [
                                i for i in range(6) if bits[term, i] and i != variable
                            ]
                            multiplier = np.prod(x[corner, others])
                            j[corner, :, variable] += multiplier * c[term]
            minimum = min(np.linalg.eigvalsh((v + v.T) / 2).min() for v in j)
            compare(minimum, float(r["symmetric_corner_min_eigenvalue"]))
            atoms = read_json(out / "00/isolated_operators.json")
            a0 = np.asarray(atoms[unit]["family0"]) / 2
            if r["source_frame_kappa"]:
                maximum = max(
                    np.sqrt(
                        np.linalg.eigvalsh(
                            (np.linalg.inv(a0) @ v - np.eye(6)).T
                            @ (np.linalg.inv(a0) @ v - np.eye(6))
                        ).max()
                    )
                    for v in j
                )
                compare(maximum, float(r["source_frame_kappa"]))
    status = read_json(out / "run_status.json")
    if any(
        status[k] != 768
        for k in ("forward_calls", "attempted_forward_calls", "completed_units")
    ):
        raise ValueError("Actual acquisition count mismatch")
    if (
        error > m["spec"]["arithmetic_tolerance"]
        or prefix_error > m["spec"]["body_prefix_tolerance"]
    ):
        raise ValueError("Alternate arithmetic or prefix discrepancy")
    result = {
        "status": "VERIFIED_FINITE_HM61_NOT_GLOBAL",
        "actual_readouts": 768,
        "maximum_alternate_error": error,
        "compared_values": count,
        "prefix_hidden_max_error": prefix_error,
        "source_before_application": True,
        "same_agent_alternate_arithmetic_not_independent_replication": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    verify(p.parse_args().out)
