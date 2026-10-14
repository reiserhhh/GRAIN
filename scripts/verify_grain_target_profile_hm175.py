#!/usr/bin/env python3
"""One direct probability-space audit of the profile-law experiment."""

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
from scripts import run_grain_target_profile_hm175 as G


def direct(bundle, cells):
    """Compute posterior ratios without the scorer's log-domain implementation."""
    parent = bundle["parent_bundle"]
    p, y, pi = (
        np.asarray(parent[k]) for k in ("source_mass", "source_binary_targets", "prior")
    )
    mass, weight, labels = (
        np.asarray(bundle[k])
        for k in ("profile_mass", "profile_weights", "profile_labels")
    )
    rho0, rho1 = np.asarray(parent["rho0"]), np.asarray(parent["rho1"])
    iw0, iw1 = (
        (1 - pi) * np.prod(rho0[cells], axis=0),
        pi * np.prod(rho1[cells], axis=0),
    )
    cw = weight * np.prod(mass[:, cells], axis=1)
    aw = np.prod(p[:, cells], axis=1) / len(p)
    j0, j1 = (
        np.stack([iw0, cw @ (1 - labels), aw @ (1 - y)]),
        np.stack([iw1, cw @ labels, aw @ y]),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return j1 / (j0 + j1), np.log(j1 / j0)


def verify(out):
    """Audit all cold families and signed matrix entries once, without refitting."""
    m = R.check(out, ("scored_seal.json", "analysis_seal.json"))
    parent = ROOT / m["config"]["parent"]
    z, bundle = H.load(parent), R.read(out / "source_bundle.json")
    with np.load(out / "cold_accounts.npz", allow_pickle=False) as a:
        prob, odds = a["probabilities"], a["binary_log_odds"]
    with np.load(out / "mechanism_accounts.npz", allow_pickle=False) as a:
        joint = {k: a[k] for k in a.files}
    err, scalars = 0.0, 0
    for arm in range(2):
        for u in range(24):
            for v in range(2):
                for k, cells in enumerate(
                    J.evidence_groups(
                        z["cells"][arm, u, v], z["selected"][u, v], z["control"][u, v]
                    )
                ):
                    for actual, expected in zip(
                        direct(bundle, cells),
                        (prob[arm, u, v, k], odds[arm, u, v, k]),
                        strict=True,
                    ):
                        if not np.array_equal(
                            np.isfinite(actual), np.isfinite(expected)
                        ):
                            raise ValueError("Changed support")
                        use = np.isfinite(actual)
                        err = max(
                            err,
                            float(np.max(np.abs(actual[use] - expected[use])))
                            if use.any()
                            else 0,
                        )
                        scalars += int(use.sum())
    p, y = (
        np.asarray(bundle["parent_bundle"][k])
        for k in ("source_mass", "source_binary_targets")
    )
    labels, weight, mu = (
        np.asarray(bundle[k])
        for k in ("profile_labels", "profile_weights", "profile_mass")
    )
    alt_within, alt_between = np.zeros((5, 2, 8, 8)), np.zeros((5, 2, 8, 8))
    for q in range(5):
        for g in (0, 1):
            rho = p[y[:, q] == g].mean(0)
            for c in np.flatnonzero(labels[:, q] == g):
                use = np.all(y == labels[c], axis=1)
                centered = p[use] - mu[c]
                alt_within[q, g] += centered.T @ centered / len(p)
                deviation = mu[c] - rho
                alt_between[q, g] += weight[c] * np.outer(deviation, deviation)
    matrix_error = float(
        max(
            np.max(np.abs(alt_within - joint["within_profile"])),
            np.max(np.abs(alt_between - joint["between_profiles"])),
        )
    )
    if max(err, matrix_error) > 2e-12:
        raise ValueError("Alternate mechanism account failed")
    receipt = {
        "status": "PASS",
        "direct_product_scalars": scalars,
        "direct_product_max_error": err,
        "centered_matrix_entries": int(alt_within.size + alt_between.size),
        "centered_covariance_max_error": matrix_error,
        "scientific_rerun": False,
        "auditor": "same_agent_alternate_formula_not_blind",
    }
    R.write(out / "audit_receipt.json", receipt)
    print(json.dumps(receipt))


def main():
    """Perform a single finite alternate audit, not a precision search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=G.DEFAULT)
    verify(parser.parse_args().out)


if __name__ == "__main__":
    main()
