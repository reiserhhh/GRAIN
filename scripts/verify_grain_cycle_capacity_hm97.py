#!/usr/bin/env python3
"""Independently recompute HM97 capacities using incidence matrices and QR."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import qr, svdvals
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_cycle_capacity_hm97 import check, settings
from scripts.run_grain_hierarchical_query_hm88 import unit
from scripts.run_grain_observer_parity_hm96 import load, seal, table, verify, write


def verify_capacity():
    """Check each support, all channels and every positive-law witness."""
    spec, out = settings()
    check(out)
    verify(out, load(out / "02_seal.json"))
    verify(out, load(out / "03_seal.json"))
    input_rows = load(out / "00/rows.json")
    for row in input_rows:
        unit(out, row["unit"])
    dest = out / "04"
    dest.mkdir()
    data = np.load(out / "03/measured_capacity.npz")
    n = spec["states"]
    edges = list(itertools.combinations(range(n), 2))
    incidence = np.zeros((n, len(edges)))
    for e, (i, j) in enumerate(edges):
        incidence[i, e], incidence[j, e] = 1, -1
    laplacian = incidence @ incidence.T
    cycle_projector = (
        np.eye(len(edges)) - incidence.T @ np.linalg.pinv(laplacian) @ incidence
    )
    records, errors, checks = [], [], 0

    def compare(actual, expected):
        nonlocal checks
        x, y = np.asarray(actual), np.asarray(expected)
        if x.shape != y.shape:
            raise ValueError("Verification shape mismatch")
        errors.append(float(np.max(np.abs(x - y))) if x.size else 0.0)
        checks += x.size

    compare(cycle_projector @ cycle_projector, cycle_projector)
    compare(incidence @ cycle_projector, np.zeros((n, len(edges))))
    for context in range(4):
        kernel = data["target_kernel"][context]
        f = np.array([(kernel[i, j] - kernel[j, i]) / 2 for i, j in edges])
        cycle_f = cycle_projector @ f
        compare(
            cycle_f, np.array([data[f"target_cycle_{context}"][i, j] for i, j in edges])
        )
        for bi, bank in enumerate(spec["banks"]):
            x, z = data[f"points_{context}_{bi}"], data[f"anchor_{context}_{bi}"]
            coordinates = list(itertools.combinations(range(x.shape[1]), 2))
            a = np.array(
                [
                    [
                        (x[i, p] - z[p]) * (x[j, q] - z[q]) / 2
                        - (x[i, q] - z[q]) * (x[j, p] - z[p]) / 2
                        for p, q in coordinates
                    ]
                    for i, j in edges
                ]
            )
            ac = cycle_projector @ a
            compare(
                ac,
                np.array([data[f"area_cycle_{context}_{bi}"][i, j] for i, j in edges]),
            )
            singular = svdvals(ac)
            cutoff = max(spec["rank_atol"], spec["rank_rtol"] * singular[0])
            rank = int(np.count_nonzero(singular > cutoff))
            q, _, _ = qr(ac, mode="economic", pivoting=True)
            projection = q[:, :rank] @ q[:, :rank].T
            alt_shared, alt_residual = (
                projection @ cycle_f,
                (np.eye(len(edges)) - projection) @ cycle_f,
            )
            compare(
                alt_shared,
                np.array([data[f"shared_{context}_{bi}"][i, j] for i, j in edges]),
            )
            compare(
                alt_residual,
                np.array([data[f"residual_{context}_{bi}"][i, j] for i, j in edges]),
            )
            for target in range(18):
                plus, minus = (
                    data["witness_plus"][context, bi, target],
                    data["witness_minus"][context, bi, target],
                )
                delta = plus - minus
                compare(plus.sum(1), np.full(n, 1 / n))
                compare(minus.sum(0), np.full(n, 1 / n))
                compare(plus + plus.T, minus + minus.T)
                compare(delta.sum(1) @ x, np.zeros(x.shape[1]))
                area_diff = np.array(
                    [
                        sum(
                            delta[i, j] * a[e, channel] - delta[j, i] * a[e, channel]
                            for e, (i, j) in enumerate(edges)
                        )
                        for channel in range(len(coordinates))
                    ]
                )
                compare(area_diff, np.zeros(len(coordinates)))
                if plus.min() <= 0 or minus.min() <= 0:
                    raise ValueError("Illegal pair-mass witness")
                residual = data[f"residual_{context}_{bi}"][..., target]
                size = np.max(np.abs(residual))
                scale = 1 / (2 * n**2 * size) if size > spec["arithmetic_atol"] else 0
                direct = float(
                    sum(
                        delta[i, j] * kernel[i, j, target]
                        for i in range(n)
                        for j in range(n)
                    )
                )
                expected = float(2 * scale * np.sum(residual**2))
                compare(direct, expected)
                records.append(
                    {
                        "context": context,
                        "bank": bank,
                        "channel": target,
                        "area_rank": rank,
                        "target_contrast": direct,
                        "expected_residual_contrast": expected,
                        "area_difference_max": float(np.max(np.abs(area_diff))),
                    }
                )
    maximum = max(errors)
    if maximum > spec["arithmetic_atol"]:
        raise ValueError(f"Finite mechanism mismatch {maximum}")
    table(dest / "alternate_witnesses.csv", records)
    summary = {
        "status": "VERIFIED_FINITE_HM97_NOT_GLOBAL",
        "scalar_checks": int(checks),
        "alternate_max_error": maximum,
        "actual_trace_checks": len(input_rows),
        "incidence_cycle_rank": round(np.trace(cycle_projector)),
        "same_agent_alternate_implementation_not_third_party": True,
        "psychological_validity_claim": False,
        "original_scope_complete": False,
    }
    write(dest / "summary.json", summary)
    write(out / "04_seal.json", seal(out, dest))
    write(out / "verification.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        verify_capacity()
