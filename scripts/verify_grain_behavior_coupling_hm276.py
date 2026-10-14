#!/usr/bin/env python3
"""Independent rational census/equations, not production bridge helper replay."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R

BITS = tuple(product((0, 1), repeat=2))
DEFAULT = ROOT / "results/grain_math_response_v1/behavior_coupling_r1"


def rational_channels(config, gamma):
    """Enumerate exact pair emissions and all behavioral outcomes separately."""
    emissions = []
    for name in ("z_accuracy", "w_accuracy"):
        a = F(str(config[name]))
        emissions.append(
            [
                [
                    np.prod([a if x == y else 1 - a for x, y in zip(u, o, strict=True)])
                    for o in BITS
                ]
                for u in BITS
            ]
        )
    low, slope = F(str(config["behavior_low"])), F(str(config["behavior_slope"]))
    behavior = []
    for u0, u1 in BITS:
        a, b = low + slope * u0, low + slope * u1
        both = a * b + gamma * (2 * u0 - 1) * (2 * u1 - 1)
        behavior.append([1 - a - b + both, b - both, a - both, both])
    return *emissions, behavior


def direct_solve(k, m):
    """Solve the four rational equations by explicit elimination, without SVD."""
    aug = [list(a) + list(b) for a, b in zip(k, m, strict=True)]
    for i in range(4):
        pivot = next(j for j in range(i, 4) if aug[j][i])
        aug[i], aug[pivot] = aug[pivot], aug[i]
        divisor = aug[i][i]
        aug[i] = [x / divisor for x in aug[i]]
        for j in range(4):
            if j != i:
                factor = aug[j][i]
                aug[j] = [x - factor * y for x, y in zip(aug[j], aug[i], strict=True)]
    return [row[4:] for row in aug]


def vector(values):
    """Evaluate exact numbers only at the comparison boundary."""
    return np.asarray(values, dtype=float)


def verify(out):
    """Check all observations, joint laws, comparison operations and cold output."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    summary = R.read(out / "summary.json")
    errors, checks = [], 0
    with np.load(out / "observed_accounts.npz", allow_pickle=False) as archive:
        for ci, criterion in enumerate(cfg["criterion_laws"]):
            gamma = F(str(criterion["conditional_joint_adjustment"]))
            z, w, y = rational_channels(cfg, gamma)
            for bi, raw_prior in enumerate(cfg["source_priors"]):
                prior = [F(str(p)) for p in raw_prior]
                joint = [
                    [
                        [
                            sum(
                                prior[u] * z[u][zz] * w[u][ww] * y[u][yy]
                                for u in range(4)
                            )
                            for yy in range(4)
                        ]
                        for ww in range(4)
                    ]
                    for zz in range(4)
                ]
                errors.append(
                    float(np.max(abs(vector(joint) - archive[f"c{ci}b{bi}/joint"])))
                )
                counts = [[sum(cell) for cell in row] for row in joint]
                mass = [sum(row) for row in counts]
                kernel = [[p / mass[i] for p in row] for i, row in enumerate(counts)]
                means = [
                    [
                        sum(joint[i][ww][yy] for ww in range(4)) / mass[i]
                        for yy in range(4)
                    ]
                    for i in range(4)
                ]
                h = direct_solve(kernel, means)
                bundle = R.read(out / f"bundle_c{ci}_b{bi}.json")
                errors.append(
                    float(
                        np.max(
                            abs(vector(h) - vector(bundle["bridge"]["representative"]))
                        )
                    )
                )
                checks += 64 + 16
                for ai, raw_kappa in enumerate(cfg["author_joint_parameters"]):
                    kappa = F(str(raw_kappa))
                    up = [(1 + kappa * (2 * a - 1) * (2 * b - 1)) / 4 for a, b in BITS]
                    wp = [sum(up[u] * w[u][ww] for u in range(4)) for ww in range(4)]
                    qp = [sum(wp[ww] * h[ww][yy] for ww in range(4)) for yy in range(4)]
                    truth = [sum(up[u] * y[u][yy] for u in range(4)) for yy in range(4)]
                    assert qp == truth
                    a, b = qp[2] + qp[3], qp[1] + qp[3]
                    covariance = qp[3] - a * b
                    assert a == b == F(1, 2)
                    assert covariance == (F(1, 25) + gamma) * kappa
                    row = next(
                        r
                        for r in summary["rows"]
                        if r["source_bank"] == bi
                        and r["criterion"] == criterion["id"]
                        and r["author_law"] == ai
                    )
                    errors.append(abs(row["coupling"] - float(covariance)))
                    assert (
                        abs(
                            row["paired_single_weight_product_coupling"]
                            - float(kappa / 25)
                        )
                        < 1e-11
                    )
                    assert abs(row["relinked_coupling"]) < 1e-11
                    errors.append(
                        float(
                            np.max(
                                abs(vector(truth) - archive[f"c{ci}b{bi}a{ai}/truth"])
                            )
                        )
                    )
                    errors.append(
                        float(
                            np.max(
                                abs(
                                    vector(wp)
                                    - vector(
                                        R.read(out / f"input_a{ai}.json")["weights"]
                                    )
                                )
                            )
                        )
                    )
                    checks += 4 + 4 + 3
    cold = R.read(out / "cold_readback.json")
    errors.append(abs(cold["reading"]["coupling"] - float(F(6, 125))))
    result = {
        "PASS": max(errors) < 1e-11,
        "rational_cell_equation_and_functional_checks": checks,
        "maximum_alternate_error": max(errors),
        "exact_joint_equals_held_behavior_in_all12_cases": True,
        "all_marginal_means_equal_half": True,
        "same_agent_not_blind_replication": True,
        "new_neural_API_private_training": 0,
        "original_scope_complete": False,
    }
    if not result["PASS"]:
        raise AssertionError(result)
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    verify(parser.parse_args().out)
