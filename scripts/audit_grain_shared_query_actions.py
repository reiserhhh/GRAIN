#!/usr/bin/env python3
"""Independently replay saved logits, patch bytes and finite carrier allocations."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.navigation import digest, read_json, write_json
from scripts.run_grain_navigation_baseline import checked_arrays
from scripts.run_grain_shared_query_actions import DEFAULT, REFERENCE, verify


def logit_score(native, alphabet):
    """Reconstruct conditional expectation and full-vocabulary candidate mass."""
    z = np.array(native["candidate_logits"], float)
    p = np.exp(z - z.max())
    p /= p.sum()
    score = float(p @ np.asarray(alphabet["values"]))
    np.testing.assert_allclose(score, native["conditional_score"], atol=1e-12)
    np.testing.assert_allclose(p, native["conditional_probabilities"], atol=1e-12)
    log_mass = float(
        z.max()
        + np.log(np.exp(z - z.max()).sum())
        - native["full_vocab_log_normalizer"]
    )
    np.testing.assert_allclose(np.exp(log_mass), native["candidate_mass"], atol=1e-12)
    return score


def audit(out):
    """Use six permutation orders rather than the runner's subset decomposition."""
    m = verify(out)
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Unfinished packet")
    for name, expected in read_json(out / "evaluation_manifest.json")["files"].items():
        if digest(out / name) != expected:
            raise ValueError("Changed evaluation table")
    design = {d["index"]: d for d in read_json(out / "design.json")}
    states = {i: checked_arrays(out / "states" / f"unit_{i:03d}.json") for i in design}
    table = pd.read_csv(out / "cube_cells.csv").set_index(["index", "mask"])
    terms = pd.read_csv(out / "component_actions.csv").set_index(["index", "reader"])
    levels = pd.read_csv(out / "levels.csv").set_index("index")
    identity_error, direct_error, allocation_error = 0.0, 0.0, 0.0
    checks = 0
    with np.load(REFERENCE / "reference.npz", allow_pickle=False) as file:
        coef = dict(file)
    for i, (meta, z) in states.items():
        score = logit_score(meta["native"], m["alphabet"])
        replay = logit_score(meta["identity_replay"], m["alphabet"])
        identity_error = max(identity_error, abs(score - replay))
        np.testing.assert_allclose(score, levels.loc[i, "native"], atol=1e-12)
        assert len(z["state"]) == len(design[i]["ids"])
        checks += 2
    patch_checks = 0
    for edge in read_json(out / "edges.json"):
        b, d = states[edge["base"]][1]["state"], states[edge["donor"]][1]["state"]
        question = np.array(design[edge["base"]]["question_mask"], bool)
        groups = np.where(question, 1, 0)
        groups[-1] = 2
        all_values = {"native": [], "affine": []}
        q = edge["query"]
        for mask in range(8):
            mixed = np.stack(
                [
                    d[i] if mask & (1 << int(group)) else b[i]
                    for i, group in enumerate(groups)
                ]
            )
            if mask in (0, 7):
                meta = states[edge["base"] if mask == 0 else edge["donor"]][0]["native"]
            else:
                row = read_json(
                    out / "actions" / f"edge_{edge['index']:03d}_mask_{mask}.json"
                )
                if (
                    hashlib.sha256(np.ascontiguousarray(mixed).tobytes()).hexdigest()
                    != row["state_sha256"]
                ):
                    raise ValueError(
                        "Patch reconstruction does not match executed bytes"
                    )
                meta = row["native"]
                patch_checks += 1
                checks += 1
            native = logit_score(meta, m["alphabet"])
            affine = float(
                coef[q + "__b"]
                + np.sum(
                    coef[q + "__G"] * (mixed[question].astype(float) - coef[q + "__Z"])
                )
                + coef[q + "__g"] @ (mixed[-1].astype(float) - coef[q + "__h"])
            )
            direct_error = max(
                direct_error, abs(affine - table.loc[(edge["index"], mask), "affine"])
            )
            np.testing.assert_allclose(
                [native, affine],
                table.loc[(edge["index"], mask), ["native", "affine"]].to_numpy(float),
                atol=1e-10,
            )
            all_values["native"].append(native)
            all_values["affine"].append(affine)
        for reader, values in all_values.items():
            average = np.zeros(3)
            for order in itertools.permutations(range(3)):
                mask = 0
                for bit in order:
                    new = mask | (1 << bit)
                    average[bit] += (values[new] - values[mask]) / math.factorial(3)
                    mask = new
            target = terms.loc[(edge["index"], reader)]
            actual = target[["phi_C", "phi_P", "phi_Q"]].to_numpy(float)
            allocation_error = max(allocation_error, float(abs(actual - average).max()))
            np.testing.assert_allclose(actual, average, atol=1e-12)
            np.testing.assert_allclose(sum(average), values[7] - values[0], atol=1e-12)
            for mask in range(1, 8):
                value = sum(
                    (-1) ** (mask.bit_count() - part.bit_count()) * values[part]
                    for part in range(8)
                    if part & mask == part
                )
                name = "I_" + "".join(c for i, c in enumerate("CPQ") if mask & (1 << i))
                np.testing.assert_allclose(value, target[name], atol=1e-12)
    cross_count = 0
    for path in sorted((out / "cross_query").glob("edge_*.json")):
        row = read_json(path)
        edge = read_json(out / "edges.json")[row["edge"]]
        h = states[row["recipient"]][1]["state"].copy()
        h[-1] += (
            states[edge["donor"]][1]["state"][-1] - states[edge["base"]][1]["state"][-1]
        )
        assert (
            hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest()
            == row["state_sha256"]
        )
        logit_score(row["native"], m["alphabet"])
        cross_count += 1
        checks += 1
    if (checks, patch_checks, cross_count) != (560, 384, 48) or identity_error > 1e-5:
        raise ValueError("Incomplete logical call/identity audit")
    result = {
        "status": "PASS",
        "native_logit_checks": checks,
        "cube_patch_byte_checks": patch_checks,
        "cross_query_byte_checks": cross_count,
        "affine_cell_checks": 512,
        "carrier_allocation_checks": 128,
        "max_identity_error": identity_error,
        "max_direct_affine_error": direct_error,
        "max_allocation_error": allocation_error,
        "independent_model_rerun": False,
    }
    write_json(out / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT)
    audit(p.parse_args().out.resolve())
