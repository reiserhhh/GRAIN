#!/usr/bin/env python3
"""Alternate actual-pair,marginal,log-interaction and cold-readout audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_joint_meaning_hm177 as H
from scripts import run_grain_observed_family_hm170 as R


def alternate_ipf(t):
    """Use reversed marginal-update order to audit the same entropy solution."""
    targets = [t.sum(axis) for axis in (0, 1, 2)]
    allowed = np.ones_like(t, bool)
    for axis, target in enumerate(targets):
        allowed &= np.expand_dims(target > 0, axis)
    x = allowed.astype(float)
    if t.sum() == 0:
        return x
    x *= t.sum() / x.sum()
    for _ in range(25000):
        for axis, target in enumerate(targets):
            current = x.sum(axis)
            ratio = np.zeros_like(target)
            idx = current > 0
            ratio[idx] = target[idx] / current[idx]
            x *= np.expand_dims(ratio, axis)
        if (
            max(
                np.max(np.abs(x.sum(axis) - target))
                for axis, target in enumerate(targets)
            )
            <= 1e-11
        ):
            return x
    raise ValueError("Alternate fixed audit did not converge")


def verify(out):
    """Re-sum actual observations and every cold scalar without rerunning science."""
    manifest = R.check(out, ("scored_seal.json", "analysis_seal.json"))
    z, b = R.prepared(out), R.read(out / "source_bundle.json")
    y, cells = z["source_targets"], z["source_cells"]
    profiles = np.asarray(b["profiles"])
    tables, laws = np.asarray(b["profile_tables"]), np.asarray(b["joint_laws"])
    direct = np.zeros_like(tables[2])
    pair_count = 0
    for u in range(len(y)):
        c = np.flatnonzero(np.all(profiles == y[u], axis=1))[0]
        for a in cells[u, 0]:
            for v in cells[u, 1]:
                direct[c, a, v] += 1 / (len(y) * 17 * 17)
                pair_count += 1
    errors = {"direct_observed_law": float(np.max(np.abs(direct - tables[2])))}
    alternate, independence = np.zeros_like(direct), np.zeros_like(direct)
    maximum_margin, rectangles, cycle_error = 0.0, 0, 0.0
    for z0 in range(2):
        for z1 in range(2):
            s0, s1 = slice(4 * z0, 4 * z0 + 4), slice(4 * z1, 4 * z1 + 4)
            t = direct[:, s0, s1]
            alternate[:, s0, s1] = alternate_ipf(t)
            h = tables[1, :, s0, s1]
            for axis in range(3):
                maximum_margin = max(
                    maximum_margin, float(np.max(np.abs(h.sum(axis) - t.sum(axis))))
                )
            for c in range(len(profiles)):
                row, col, mass = t[c].sum(1), t[c].sum(0), t[c].sum()
                if mass > 0:
                    independence[c, s0, s1] = np.outer(row, col) / mass
            for a in range(4):
                for aa in range(a + 1, 4):
                    for v in range(4):
                        for vv in range(v + 1, 4):
                            values = []
                            for c in range(len(profiles)):
                                rect = h[c][np.ix_([a, aa], [v, vv])]
                                if np.all(rect > 0):
                                    log = np.log(rect)
                                    values.append(
                                        log[0, 0] - log[0, 1] - log[1, 0] + log[1, 1]
                                    )
                            if values:
                                rectangles += len(values)
                                cycle_error = max(cycle_error, float(np.ptp(values)))
    errors.update(
        {
            "alternate_common_law": float(np.max(np.abs(alternate - tables[1]))),
            "independence_law": float(np.max(np.abs(independence - tables[0]))),
            "all_two_way_margins": maximum_margin,
            "shared_rectangle_log_contrast": cycle_error,
        }
    )
    recomposed = np.zeros_like(laws)
    for k in range(3):
        for c in range(len(profiles)):
            for q in range(5):
                recomposed[k, q, int(profiles[c, q])] += tables[k, c]
    errors["all_target_laws"] = float(np.max(np.abs(recomposed - laws)))
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = recomposed[:, :, 1] / recomposed.sum(2)
    output = [
        json.loads(line)
        for line in (out / "cold_output.jsonl").read_text().splitlines()
    ]
    cold_error, support_error, values = 0.0, 0, 0
    for i, row in enumerate(output):
        for k in range(3):
            for q in range(5):
                parts = [
                    psi[k, q, a, v]
                    for a in z["held_cells"][i, 0]
                    for v in z["held_cells"][i, 1]
                ]
                valid = np.isfinite(parts).all()
                actual = row["pair_target_readout"][k][q]
                support_error += int(valid != row["supported"][k][q])
                if valid and actual is not None:
                    cold_error = max(cold_error, abs(sum(parts) / len(parts) - actual))
                    values += 1
                elif valid or actual is not None:
                    support_error += 1
    errors["cold_pair_sum"] = cold_error
    with np.load(out / "mechanism_accounts.npz", allow_pickle=False) as a:
        tail = (1 + (a["matched_null"] >= a["matched_observed"]).sum(0)) / (
            len(a["matched_null"]) + 1
        )
        errors["matched_tail"] = float(np.max(np.abs(tail - a["matched_tail"])))
    if (
        max(errors.values()) > 2e-8
        or support_error
        or len(output) != manifest["config"]["application_authors"]
    ):
        raise ValueError("Alternate joint-meaning audit failed")
    receipt = {
        "status": "PASS",
        "source_pairs_enumerated": pair_count,
        "common_rectangles": rectangles,
        "profile_law_entries": int(tables.size),
        "cold_values": values,
        "support_errors": support_error,
        "errors": errors,
        "same_agent_not_blind": True,
        "scientific_rerun": False,
    }
    R.write(out / "audit_receipt.json", receipt)
    print(json.dumps(receipt))


def main():
    """Audit one already frozen finite experiment, not choose new parameters."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=H.DEFAULT)
    verify(p.parse_args().out)


if __name__ == "__main__":
    main()
