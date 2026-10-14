#!/usr/bin/env python3
"""Fraction census and independent literal source-rule verification; no model calls."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_relational_diagnostic_update import BITS
from scripts import run_grain_observed_family_hm170 as R


def census(accuracy, rho, prior):
    """Enumerate rational probabilities and risks without production probability helpers."""
    a, r, p = Q(str(accuracy)), Q(str(rho)), Q(str(prior))
    v, k = a * (1 - a), r * a * (1 - a)
    ch = {(0, 0): a * a + k, (0, 1): v - k, (1, 0): v - k, (1, 1): (1 - a) ** 2 + k}
    joint = {
        (y, x, z): (p if y else 1 - p) * ch[(x, z) if y else (1 - x, 1 - z)]
        for y in (0, 1)
        for x in (0, 1)
        for z in (0, 1)
    }
    mass = {(x, z): sum(joint[y, x, z] for y in (0, 1)) for x in (0, 1) for z in (0, 1)}
    full = {(x, z): joint[1, x, z] / mass[x, z] for x in (0, 1) for z in (0, 1)}
    mx = {
        x: sum(joint[1, x, z] for z in (0, 1)) / sum(mass[x, z] for z in (0, 1))
        for x in (0, 1)
    }
    mz = {
        z: sum(joint[1, x, z] for x in (0, 1)) / sum(mass[x, z] for x in (0, 1))
        for z in (0, 1)
    }
    left, right = {(x, z): mx[x] for x, z in mass}, {(x, z): mz[z] for x, z in mass}
    product = {}
    for x, z in mass:
        high = p * (a if x == 0 else 1 - a) * (a if z == 0 else 1 - a)
        low = (1 - p) * (a if x == 1 else 1 - a) * (a if z == 1 else 1 - a)
        product[x, z] = high / (high + low)
    average = {key: (left[key] + right[key]) / 2 for key in mass}

    def risk(reading):
        return sum(joint[y, x, z] * (y - reading[x, z]) ** 2 for y, x, z in joint)

    rows = []
    for key, reading, single in (
        ("perfect_same_target", full, average),
        (
            "complement_wrong_key",
            {xz: 1 - value for xz, value in full.items()},
            {xz: 1 - value for xz, value in average.items()},
        ),
    ):
        rows.append(
            {
                "reader": key,
                "contact": sum(
                    mass[xz] * (full[xz] - product[xz]) * (reading[xz] - single[xz])
                    for xz in mass
                ),
                "target_squared_risk": risk(reading),
                "regret_against_declared_Y": risk(reading) - risk(full),
                "joint_over_own_singleton_mean_risk_gain": risk(single) - risk(reading),
            }
        )
    energy = sum(mass[xz] * (full[xz] - left[xz]) ** 2 for xz in mass)
    if energy != risk(left) - risk(full):
        raise ValueError("Exact rational refinement identity failed")
    for x in (0, 1):
        if sum(mass[x, z] * (full[x, z] - mx[x]) for z in (0, 1)) != 0:
            raise ValueError("Exact conditional innovation mean is nonzero")
    return rows, energy


def verify(out):
    """Check the complete registered finite experiment and separate cold execution."""
    cfg = R.check(out, ("computed_seal.json",))["config"]
    summary = R.read(out / "summary.json")
    exact_error, checks = 0.0, 0
    for i, rho in enumerate(cfg["world_rhos"]):
        rows, energy = census(cfg["world_accuracy"], rho, cfg["world_prior"])
        for row in rows:
            actual = next(
                x
                for x in summary["world_rows"]
                if x["rho"] == rho and x["reader"] == row["reader"]
            )
            for name, value in row.items():
                if name != "reader":
                    exact_error = max(exact_error, abs(float(value) - actual[name]))
                    checks += 1
        exact_error = max(
            exact_error,
            abs(float(energy) - summary["identities"][i]["innovation_energy"]),
        )
        checks += 1
    parent = ROOT / cfg["source_parent"]
    source_error, old_error, identities, count = 0.0, 0.0, 0.0, 0
    with (
        np.load(parent / "source_pair_bank.npz", allow_pickle=False) as bank,
        np.load(parent / "log_kernel_accounts.npz", allow_pickle=False) as logs,
        np.load(parent / "conditional_profiles.npz", allow_pickle=False) as old,
        np.load(out / "source_rule_accounts.npz", allow_pickle=False) as result,
    ):
        codes, folds = bank["codes"], bank["folds"]
        for space in ("E5", "GRAIN15"):
            for name, mask in (
                ("bank0", folds == 0),
                ("bank1", folds == 1),
                ("pooled", np.ones(386, bool)),
            ):
                key = f"{space}/{name}"
                profile, target = result[key + "/profiles"], result[key + "/targets"]
                k = logs[space][:528, mask]
                # One scale per query/role keeps literal sums away from underflow.
                anchor = k.max(axis=(1, 3), keepdims=True)
                density = np.exp(k - anchor)
                single = density.mean(axis=-1)
                f = (density[:, :, 0] * density[:, :, 1]).mean(axis=-1)
                j = single[:, :, 0] * single[:, :, 1]
                raw = np.zeros((528, 4, 32))
                for g in range(32):
                    ix = codes[mask] == g
                    raw[:, 0, g] = single[:, ix, 0].sum(axis=-1)
                    raw[:, 1, g] = single[:, ix, 1].sum(axis=-1)
                    raw[:, 2, g] = f[:, ix].sum(axis=-1)
                    raw[:, 3, g] = j[:, ix].sum(axis=-1)
                literal = raw / raw.sum(axis=-1, keepdims=True)
                source_error = max(
                    source_error,
                    float(np.max(abs(literal - profile))),
                    float(np.max(abs(literal @ BITS - target))),
                )
                historical = old[f"actual/{key}/targets"][:528, :2]
                old_error = max(
                    old_error, float(np.max(abs(target[:, 2:] - historical)))
                )
                for field, base in (
                    ("evidence_after_left", target[:, 0]),
                    ("evidence_after_right", target[:, 1]),
                    ("model_replacement", target[:, 3]),
                    ("equal_singleton_departure", (target[:, 0] + target[:, 1]) / 2),
                ):
                    identities = max(
                        identities,
                        float(
                            np.max(abs(base + result[key + "/" + field] - target[:, 2]))
                        ),
                    )
                count += profile.size + target.size + historical.size
        cold = R.read(out / "cold_readback.json")
        cold_error = max(
            float(
                np.max(
                    abs(
                        np.asarray(cold["spaces"][s]["accounts"]["targets"])
                        - result[f"{s}/pooled/targets"][0]
                    )
                )
            )
            for s in ("E5", "GRAIN15")
        )
    passed = max(exact_error, source_error, old_error, identities, cold_error) < 1e-12
    receipt = {
        "PASS": passed,
        "Fraction_scalar_checks": checks,
        "Fraction_max_error": exact_error,
        "literal_source_values": count,
        "literal_source_error": source_error,
        "old_HM273_F_J_error": old_error,
        "telescoping_error": identities,
        "separate_cold_CLI_error": cold_error,
        "same_agent_not_blind_replication": True,
        "new_neural_API_private_training_or_application_label_use": 0,
    }
    R.write(out / "verification.json", receipt)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not passed:
        raise ValueError("Verification failed; preserved without scientific retry")
    print(receipt)


def main():
    """Verify a new packet once without modifying frozen scientific artifacts."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    verify(p.parse_args().out)


if __name__ == "__main__":
    main()
