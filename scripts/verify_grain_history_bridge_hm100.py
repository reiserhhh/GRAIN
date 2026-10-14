#!/usr/bin/env python3
"""Alternate subset and derivative arithmetic on every declared HM100 capture."""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_history_bridge import primitives
from scripts.run_grain_history_bridge_hm100 import check, settings, stage
from scripts.run_grain_observer_parity_hm96 import seal, write
from scripts.verify_grain_typed_cycle_hm99 import alternate


def alternate_scores(low, high):
    """Enumerate primitive subset values and Shapley factorial weights directly."""
    corners = []
    for mask in range(8):
        value, logits, gate = [
            high[k] if mask & (1 << bit) else low[k]
            for k, bit in ((1, 0), (0, 1), (2, 2))
        ]
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        corners.append((p * gate * value).sum(-1).sum(-1))
    corners = np.stack(corners, -1)
    result = np.zeros(corners.shape[:-1] + (3,))
    for factor in range(3):
        for mask in range(8):
            if mask & (1 << factor):
                continue
            count = mask.bit_count()
            weight = (
                math.factorial(count) * math.factorial(2 - count) / math.factorial(3)
            )
            result[..., factor] += weight * (
                corners[..., mask | (1 << factor)] - corners[..., mask]
            )
    return corners, result


def alternate_integrals(low, high, nodes):
    """Use the explicit softmax derivative, rather than production covariance."""
    points, weights = np.polynomial.legendre.leggauss(nodes)
    l0, c0, g0 = low
    dl, dc, dg = [y - x for x, y in zip(low, high, strict=True)]
    answer = np.zeros(l0.shape[:-2] + (3,))
    for point, weight in zip(points, weights, strict=True):
        s = (point + 1) / 2
        l, c, g = l0 + s * dl, c0 + s * dc, g0 + s * dg
        p = softmax(l, axis=-1)
        dp = p * dl - p * (p * dl).sum(-1, keepdims=True)
        for k, integrand in enumerate((p * g * dc, dp * g * c, p * dg * c)):
            answer[..., k] += weight / 2 * integrand.sum(-1).sum(-1)
    return answer


def run():
    """Verify native replay, all corners, paths, sectors and author comparisons."""
    spec, out = settings()
    check(out)
    for n in (1, 2, 3, 4):
        stage(out, n)
    datasets = []
    for name in (
        "01/compiled.npz",
        "02/primitives.npz",
        "03/accounts.npz",
        "04/cycles.npz",
    ):
        with np.load(out / name, allow_pickle=False) as z:
            datasets.append({k: z[k].copy() for k in z.files})
    source, exported, accounts, stored = datasets
    errors, count = [], 0

    def same(x, y):
        nonlocal count
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape or not np.isfinite([x, y]).all():
            raise ValueError("Alternate finite shapes required")
        errors.append(float(np.max(np.abs(x - y))))
        count += x.size

    refs = {
        (c, f, j): n for n, (c, f, i, j) in enumerate(exported["addresses"]) if i == j
    }
    if len(refs) != 72:
        raise ValueError("Missing current-state references")
    for n, (c, f, i, j) in enumerate(exported["addresses"]):
        low = primitives(source, exported, refs[(c, f, j)])
        high = primitives(source, exported, n)
        corners, shapley = alternate_scores(low, high)
        integral = alternate_integrals(low, high, spec["alternate_quadrature_nodes"])
        address = (c, f, i, j)
        same(corners, accounts["corners"][address])
        same(shapley, accounts["shapley"][address])
        same(integral, accounts["integral"][address])
        same(integral.sum(-1), corners[..., 7] - corners[..., 0])
        same(
            accounts["orders"][address].sum(-1),
            np.repeat((corners[..., 7] - corners[..., 0])[..., None], 6, -1),
        )
        for mask in range(8):
            reconstructed = sum(
                accounts["interactions"][address][..., sub]
                for sub in range(8)
                if sub & mask == sub
            )
            same(reconstructed, corners[..., mask])
    with np.load(ROOT / spec["typed"] / "01/compiled.npz", allow_pickle=False) as z:
        difference = z["plus"] - z["minus"]
    if np.any(difference.sum(1) > 1e-12) or np.any(np.abs(difference.sum(2)) > 1e-12):
        raise ValueError("Author currents not balanced")
    for c, f, account, factor in itertools.product(
        range(2), range(2), ("integral", "shapley"), range(3)
    ):
        name = spec["factors"][factor]
        kernel = accounts[account][c, f, ..., factor].reshape(18, 18, 18)
        parts = alternate(kernel, 6, 3)
        component = stored[f"{account}_{name}_{c}_{f}"]
        same(sum(parts.values()), component)
        for sector, part in parts.items():
            same(part, stored[f"{account}_{name}_{c}_{f}_{sector}"])
        first = np.array(
            [[np.sum(d * component[..., q]) for q in range(18)] for d in difference]
        )
        second = np.einsum("aij,ijq->aq", difference, kernel)
        same(first, second)
    if max(errors) > spec["arithmetic_atol"]:
        raise ValueError("Alternate arithmetic mismatch: " + str(max(errors)))
    summary = {
        "status": "VERIFIED_FINITE_HM100_NOT_GLOBAL",
        "scalar_checks": count,
        "max_error": max(errors),
        "captures": len(exported["addresses"]),
        "quadrature_nodes": spec["alternate_quadrature_nodes"],
        "same_agent_not_third_party": True,
        "new_neural_calls": 0,
        "predictor_fit": False,
        "original_scope_complete": False,
    }
    dest = out / "05"
    dest.mkdir()
    write(dest / "summary.json", summary)
    write(out / "05_seal.json", seal(out, dest))
    write(out / "verification.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
