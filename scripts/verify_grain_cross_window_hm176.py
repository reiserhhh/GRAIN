#!/usr/bin/env python3
"""Alternate observed-pair and finite-population audit without rerunning science."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_cross_window_hm176 as H
from scripts import run_grain_observed_family_hm170 as R


def verify(out):
    """Directly re-sum observed pairs, cross-author laws and every cold readout."""
    manifest = R.check(out, ("scored_seal.json", "analysis_seal.json"))
    z = R.prepared(out)
    p, y, cells = z["source_mass"], z["source_targets"], z["source_cells"]
    n, t = len(p), cells.shape[2]
    direct = np.zeros((3, 5, 2, 8, 8))
    cross_pairs = 0
    for u in range(n):
        joint = np.zeros((8, 8))
        for a in cells[u, 0]:
            for b in cells[u, 1]:
                joint[a, b] += 1 / (t * t * n)
                cross_pairs += 1
        same_group = np.flatnonzero(np.all(y == y[u], axis=1))
        pool, other = np.zeros((8, 8)), np.zeros((8, 8))
        for v in same_group:
            mat = np.outer(
                np.bincount(cells[u, 0], minlength=8) / t,
                np.bincount(cells[v, 1], minlength=8) / t,
            )
            pool += mat / (n * len(same_group))
            if v != u:
                other += mat / (n * (len(same_group) - 1))
        for q in range(5):
            direct[0, q, int(y[u, q])] += pool
            direct[1, q, int(y[u, q])] += other
            direct[2, q, int(y[u, q])] += joint
    with np.load(out / "mechanism_accounts.npz", allow_pickle=False) as a:
        account = {k: a[k] for k in a.files}
    errors = {
        "direct_observation_laws": float(np.max(np.abs(direct - account["joint_laws"])))
    }
    reconstruction = account["choice"] + account["response"] + account["unresolved"]
    errors["condition_reconstruction"] = float(
        np.max(np.abs(reconstruction - (direct[2] - direct[1])))
    )
    # A profile-product reference includes self; the distinct-author law does not.
    corrected = np.zeros((5, 2, 8, 8))
    for profile in np.unique(y, axis=0):
        idx = np.flatnonzero(np.all(y == profile, axis=1))
        x0 = p[idx, 0] - p[idx, 0].mean(0)
        x1 = p[idx, 1] - p[idx, 1].mean(0)
        val = (len(idx) / (len(idx) - 1)) * x0.T @ x1 / n
        for q in range(5):
            corrected[q, int(profile[q])] += val
    errors["distinct_author_covariance"] = float(
        np.max(np.abs(corrected - (direct[2] - direct[1])))
    )
    den = direct.sum(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = direct[:, :, 1] / den
    actual = [
        json.loads(line)
        for line in (out / "cold_output.jsonl").read_text().splitlines()
    ]
    max_cold, values, support_error = 0.0, 0, 0
    for i, row in enumerate(actual):
        expected = np.full((3, 5), np.nan)
        early, late = z["held_cells"][i]
        for k in range(3):
            for q in range(5):
                table = [psi[k, q, a, b] for a in early for b in late]
                if np.isfinite(table).all():
                    expected[k, q] = sum(table) / len(table)
        scored = np.asarray(row["pair_target_readout"], float)
        support_error += int(np.sum(np.isfinite(expected) != np.isfinite(scored)))
        ok = np.isfinite(expected) & np.isfinite(scored)
        max_cold = max(
            max_cold,
            float(np.max(np.abs(expected[ok] - scored[ok]))) if ok.any() else 0,
        )
        values += int(ok.sum())
    errors["cold_pair_sum"] = max_cold
    nr = account["matched_null"]
    tail = (1 + (nr >= account["matched_observed"]).sum(0)) / (len(nr) + 1)
    errors["matched_tail"] = float(np.max(np.abs(tail - account["matched_tail"])))
    if (
        max(errors.values()) > 2e-12
        or support_error
        or len(actual) != manifest["config"]["application_authors"]
    ):
        raise ValueError("Alternate observed law/readout audit failed")
    receipt = {
        "status": "PASS",
        "source_cross_pairs_enumerated": cross_pairs,
        "joint_law_entries": int(direct.size),
        "cold_scalar_readouts": values,
        "support_errors": support_error,
        "errors": errors,
        "scientific_rerun": False,
        "auditor": "same_agent_alternate_formula_not_blind",
    }
    R.write(out / "audit_receipt.json", receipt)
    print(json.dumps(receipt))


def main():
    """Check the single frozen experiment, never choose a more favorable law."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=H.DEFAULT)
    verify(parser.parse_args().out)


if __name__ == "__main__":
    main()
