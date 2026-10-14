#!/usr/bin/env python3
"""One alternate direct-product audit of frozen joint-evidence accounts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_joint_evidence as J
from scripts import run_grain_joint_evidence_hm174 as H
from scripts import run_grain_observed_family_hm170 as R


def direct(b, cells):
    """Product in probability space, independently of the log-sum-exp scorer."""
    p, y = np.asarray(b["source_mass"]), np.asarray(b["source_binary_targets"])
    pi = np.asarray(b["prior"])
    weight = np.prod(p[:, cells], axis=1)
    rho0, rho1 = np.asarray(b["rho0"]), np.asarray(b["rho1"])
    positive = pi * np.prod(rho1[cells], axis=0)
    negative = (1 - pi) * np.prod(rho0[cells], axis=0)
    sj1, sj0 = weight @ y, weight @ (1 - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        probability = np.stack(
            [
                np.asarray(b["event_probability"])[cells].mean(0),
                positive / (positive + negative),
                sj1 / (sj1 + sj0),
            ]
        )
        odds = np.stack([np.log(positive / negative), np.log(sj1 / sj0)])
    return probability, odds


def verify(out):
    """Audit every family; no scientific rescoring/overwriting of old results."""
    H.check(out, ("scored_seal.json", "analysis_seal.json"))
    z, b = H.load(out), R.read(out / "source_bundle.json")
    J.validate_bundle(b)
    with np.load(out / "composition_accounts.npz", allow_pickle=False) as a:
        stored = {k: a[k] for k in a.files}
    max_error, checked = 0.0, 0
    for arm in range(2):
        for u in range(24):
            for v in range(2):
                groups = J.evidence_groups(
                    z["cells"][arm, u, v], z["selected"][u, v], z["control"][u, v]
                )
                for k, cells in enumerate(groups):
                    p, o = direct(b, cells)
                    for actual, expected in (
                        (p, stored["probabilities"][arm, u, v, k]),
                        (o, stored["binary_log_odds"][arm, u, v, k]),
                    ):
                        if not np.array_equal(
                            np.isfinite(actual), np.isfinite(expected)
                        ):
                            raise ValueError("Different joint support")
                        use = np.isfinite(actual)
                        max_error = max(
                            max_error,
                            float(np.max(np.abs(actual[use] - expected[use])))
                            if use.any()
                            else 0,
                        )
                        checked += int(use.sum())
    zlog = stored["native_logits"]
    alternate = np.empty((24, 2, 9))
    for u in range(24):
        for v in range(2):
            for k in range(1, 10):
                row = zlog[u, v]
                alternate[u, v, k - 1] = (
                    (row[2, k] - row[2, 0])
                    - (row[1, k] - row[1, 0])
                    - (row[3, k] - row[3, 0])
                    + (row[4, k] - row[4, 0])
                )
    native_error = float(np.max(np.abs(alternate - stored["native_log_odds_J_sum"])))
    if max_error > 2e-12 or native_error > 2e-12:
        raise ValueError("Alternate mechanism accounts disagree")
    result = {
        "status": "PASS",
        "direct_product_scalars": checked,
        "native_axes": alternate.size,
        "direct_product_max_error": max_error,
        "native_contrast_max_error": native_error,
        "auditor": "same_agent_alternate_formula_not_blind",
        "scientific_rerun": False,
    }
    R.write(out / "audit_receipt.json", result)
    print(json.dumps(result))


def main():
    """Audit once; do not start a new precision campaign or experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=H.DEFAULT)
    verify(p.parse_args().out)


if __name__ == "__main__":
    main()
