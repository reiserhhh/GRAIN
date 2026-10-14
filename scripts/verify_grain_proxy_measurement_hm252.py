#!/usr/bin/env python3
"""Alternate census sums and functional solves; not independent blind audit."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grain_proxy_measurement_hm252 as R


def exact_joint(config, bank):
    """Independent rational enumeration of every latent-generated census cell."""
    shape = (2, 4, 4, 2, 2)
    result = np.zeros(shape)
    for c, z, w, y0, y1 in np.ndindex(shape):
        value = F(0)
        for u in range(3):
            means = [F(str(x)) for x in config["behavior_channels"][c][u]]
            y = (means[0] if y0 else 1 - means[0]) * (means[1] if y1 else 1 - means[1])
            value += (
                F(str(config["source_condition_mass"][c]))
                * F(str(config["source_priors"][bank][c][u]))
                * F(str(config["z_channels"][c][u][z]))
                * F(str(config["w_channels"][c][u][w]))
                * y
            )
        result[c, z, w, y0, y1] = float(value)
    return result


def verify(out):
    """Verify exact source accounts, nonunique functional and all cold readouts."""
    R.check(out)
    config = R.read(R.SPEC)
    w_channels = np.asarray(config["w_channels"], float)
    behavior = np.asarray(config["behavior_channels"], float)
    census_error, moment_error, latent_error, alternative_error, cold_error = (
        [],
        [],
        [],
        [],
        [],
    )
    count = 0
    for bank in range(2):
        bundle = R.read(out / f"bundle_{bank}.json")
        exact = exact_joint(config, bank)
        with np.load(out / "evaluation_only.npz", allow_pickle=False) as data:
            census_error.append(
                float(np.max(np.abs(exact - data["source_joint"][bank])))
            )
        for c, program in enumerate(bundle["conditions"]):
            k, m = np.zeros((4, 4)), np.zeros((4, 2))
            for z in range(4):
                mass = float(
                    sum(
                        exact[c, z, w, a, b]
                        for w in range(4)
                        for a in range(2)
                        for b in range(2)
                    )
                )
                for w in range(4):
                    k[z, w] = (
                        sum(exact[c, z, w, a, b] for a in range(2) for b in range(2))
                        / mass
                    )
                for q in range(2):
                    m[z, q] = (
                        sum(
                            exact[c, z, w, a, b] * (a if q == 0 else b)
                            for w in range(4)
                            for a in range(2)
                            for b in range(2)
                        )
                        / mass
                    )
            moment_error.append(
                float(
                    max(
                        np.max(np.abs(k - program["kernel"])),
                        np.max(np.abs(m - program["means"])),
                    )
                )
            )
            alternate = np.zeros((4, 2))
            alternate[:3] = np.linalg.solve(k[:3, :3], m[:3])
            if np.max(np.abs(k @ alternate - m)) > 1e-9:
                raise AssertionError("Alternate free-coordinate bridge is not feasible")
            h = np.asarray(program["representative"])
            latent_error.append(float(np.max(np.abs(w_channels[c] @ h - behavior[c]))))
            for profile in config["application_profiles"]:
                p = np.asarray(profile["regimes"][c]) @ w_channels[c]
                alternative_error.append(float(np.max(np.abs(p @ alternate - p @ h))))
                count += 2
        cold = R.read(out / f"cold_{bank}.json")
        for i, profile in enumerate(config["application_profiles"]):
            for q in range(2):
                expected = sum(
                    F(str(profile["choice"][c]))
                    * sum(
                        F(str(profile["regimes"][c][u]))
                        * F(str(config["behavior_channels"][c][u][q]))
                        for u in range(3)
                    )
                    for c in range(2)
                )
                cold_error.append(
                    abs(
                        cold["profiles"][i]["reading"]["targets"][q]["absolute"]
                        - float(expected)
                    )
                )
                count += 1
    negative = R.read(out / "negative_cases.json")
    alien_gap = [
        abs(x["feasible_witness_values"][1] - x["feasible_witness_values"][0])
        for c in negative["alien_law"]
        for x in c
    ]
    if not all(gap > 1e-9 for gap in alien_gap):
        raise AssertionError("Alien-law ambiguity witness disappeared")
    if not all(
        status == "NO_MOMENT_BRIDGE"
        for p in negative["coarsened_W"]["conditions"]
        for status in p["status"]
    ):
        raise AssertionError("Unrepresentable criterion was falsely calibrated")
    summary = R.read(out / "summary.json")
    if (
        not summary["primary_exact_construction_PASS"]
        or not summary["copied_event_rejected"]
    ):
        raise AssertionError("Positive construction/event qualification failed")
    errors = {
        "rational_census": max(census_error),
        "literal_moments": max(moment_error),
        "latent_equation_evaluation_only": max(latent_error),
        "alternative_bridge_functionals": max(alternative_error),
        "rational_cold_readouts": max(cold_error),
    }
    if max(errors.values()) > 1e-9:
        raise AssertionError("Alternate scientific readback failed")
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "same_agent_not_blind": True,
            "source_census_cells": 256,
            "functional_accounts": count,
            "errors": errors,
            "alien_witness_gaps": alien_gap,
            "not_estimated_human_accuracy": True,
        },
    )
    R.seal(out, "verification_seal.json", ["verification.json"])
    print(json.dumps({"PASS": True, "errors": errors}))


def main():
    """Run one read-only science audit with an exclusive result file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / R.read(R.SPEC)["output"])
    verify(parser.parse_args().out.resolve())


if __name__ == "__main__":
    main()
