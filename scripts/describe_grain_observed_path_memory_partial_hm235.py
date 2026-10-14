#!/usr/bin/env python3
"""Execute registered Markov secondary accounts after preserving an IPF failure."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observed_path_memory as M
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_source_meaning_bridge_hm233 as S

PARENT = ROOT / "results/grain_math_response_v1/observed_path_memory_r1"
MODELS = ("profile_markov", "author_markov_mixture", "observed_path")


def laws(mass, targets):
    """Keep only the three already declared nonsolver reference laws."""
    profiles, ix, full = M.aggregate(mass, targets)
    mixture = np.zeros_like(full)
    np.add.at(mixture, ix, M.markov_table(mass) / len(mass))
    return profiles, np.stack([M.markov_table(full), mixture, full])


def run(out):
    """Produce secondary data,explicitly leaving the original main question open."""
    parent = R.check(PARENT)
    failure = R.read(PARENT / "compute_failure.json")
    if (
        failure["type"] != "ValueError"
        or "Fixed IPF budget exhausted" not in failure["message"]
    ):
        raise ValueError("Expected preserved fixed-budget solver failure")
    out.mkdir(parents=True, exist_ok=False)
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    R.write(
        out / "manifest.json",
        {
            "code": {relative: R.digest(ROOT / relative)},
            "parent_science_code": parent["code"],
            "inputs": {
                str((PARENT / name).relative_to(ROOT)): R.digest(PARENT / name)
                for name in (
                    "manifest.json",
                    "prepared_seal.json",
                    "observations.npz",
                    "cold_inputs.jsonl",
                    "compute_failure.json",
                )
            },
            "purpose": "registered secondary only,not an IPF repair or primary substitute",
            "primary_target_assignment_recurrence": "NOT_COMPUTED",
            "original_scope_complete": False,
        },
    )
    z = R.prepared(PARENT)
    start = time.perf_counter()
    mass = np.array(
        [[M.triple_mass(path) for path in row] for row in z["source_paths"]]
    )
    reference = mass.mean((0, 1))
    fields = np.empty((2, 2, 2, 3, 5, 8, 8, 8))
    full_profiles = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(int)
    tables = np.zeros((2, 2, 2, 3, 32, 8, 8, 8))
    readout, accounts = [], []
    adjacent_error = 0.0
    for label_law in range(2):
        for bank in range(2):
            used = z["bank"] == bank
            for window in range(2):
                profiles, t = laws(mass[used, window], z["targets"][label_law, used])
                fields[label_law, bank, window] = M.fields(profiles, t)
                codes = profiles @ (1 << np.arange(5))
                for g, code in enumerate(codes):
                    tables[label_law, bank, window, :, code] = t[:, g]
                for model in range(2):
                    error = max(
                        float(np.max(abs(t[model].sum(axis) - t[2].sum(axis))))
                        for axis in (1, 3)
                    )
                    adjacent_error = max(adjacent_error, error)
                    accounts.append(
                        {
                            "label_law": ("actual", "old_permutation")[label_law],
                            "bank": bank,
                            "window": window,
                            "reference": MODELS[model],
                            "KL_observed_reference_nats": M.divergence(t[2], t[model]),
                            "adjacent_full_profile_max_error": error,
                            "profiles_present": len(profiles),
                        }
                    )
        profiles, pooled = laws(mass.mean(1), z["targets"][label_law])
        readout.append(M.fields(profiles, pooled))
    readout = np.array(readout)
    comparisons, crossed = [], np.empty((2, 2, 5))
    for label_law in range(2):
        for ref in range(2):
            delta = fields[label_law, :, :, 2] - fields[label_law, :, :, ref]
            pairs = (
                ((0, 0), (1, 1), "cross_author_cross_window"),
                ((0, 1), (1, 0), "cross_author_cross_window"),
                ((0, 0), (1, 0), "cross_author_same_window"),
                ((0, 1), (1, 1), "cross_author_same_window"),
                ((0, 0), (0, 1), "same_author_cross_window"),
                ((1, 0), (1, 1), "same_author_cross_window"),
            )
            primary = []
            for left, right, kind in pairs:
                inner, cosine, coverage, cells = M.recurrence(
                    delta[left], delta[right], reference
                )
                if kind == "cross_author_cross_window":
                    primary.append(inner)
                for q, target in enumerate(("E", "N", "A", "C", "O")):
                    comparisons.append(
                        {
                            "label_law": ("actual", "old_permutation")[label_law],
                            "reference": MODELS[ref],
                            "comparison": kind,
                            "left_bank": left[0],
                            "left_window": left[1],
                            "right_bank": right[0],
                            "right_window": right[1],
                            "target": target,
                            "signed_field_inner_probability_squared": float(inner[q]),
                            "field_cosine": None
                            if not np.isfinite(cosine[q])
                            else float(cosine[q]),
                            "reference_mass_coverage": coverage,
                            "joint_supported_cells": cells,
                        }
                    )
            crossed[label_law, ref] = np.mean(primary, axis=0)
    R.archive(
        out / "accounts.npz",
        author_mass=mass,
        reference=reference,
        tables=tables,
        full_profiles=full_profiles,
        fields=fields,
        secondary_crossed=crossed,
        readout=readout,
    )
    R.table(out / "secondary_law_accounts.csv", accounts)
    R.table(out / "secondary_recurrence.csv", comparisons)
    # Cold scoring is a separate process,without loading SOURCE author/label arrays.
    R.archive(out / "cold_readout.npz", readout=readout)
    R.write(
        out / "summary.json",
        {
            "status": "SECONDARY_COMPLETE_PRIMARY_PENDING_SOLVER_FAILURE",
            "preserved_primary_failure": failure,
            "primary_target_assignment_recurrence": None,
            "secondary_cross_author_cross_window_law_reference_ENACO": crossed.tolist(),
            "secondary_units": "probability-field products,notpredictionr or personality fractions",
            "secondary_KL_rows": accounts,
            "adjacent_full_profile_max_error": adjacent_error,
            "compute_seconds": time.perf_counter() - start,
            "source_authors": 386,
            "source_triples": 12352,
            "new_neural_API_private_training_application_label_calls": 0,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "computed_seal.json",
        (
            "accounts.npz",
            "cold_readout.npz",
            "secondary_law_accounts.csv",
            "secondary_recurrence.csv",
            "summary.json",
        ),
    )
    print(json.dumps(R.read(out / "summary.json")), flush=True)


def cold(out):
    """Cold apply only the three source readings to observed held histories."""
    readout = S.selected(out / "cold_readout.npz", ("readout",))["readout"]
    support = []
    with (
        (PARENT / "cold_inputs.jsonl").open() as f,
        (out / "cold_outputs.jsonl").open("x") as g,
    ):
        for line in f:
            value = M.cold_record(readout, json.loads(line))
            support.append(value["supported"])
            g.write(json.dumps(value, allow_nan=False) + "\n")
    R.write(
        out / "cold_receipt.json",
        {
            "windows": len(support),
            "authors": len(support) // 2,
            "supported_counts": np.array(support).sum(0).tolist(),
            "new_API_neural_train_private_calls": 0,
            "application_labels_read": False,
            "SOURCE_author_arrays_read": False,
        },
    )
    R.seal(out, "cold_seal.json", ("cold_outputs.jsonl", "cold_receipt.json"))
    print(json.dumps(R.read(out / "cold_receipt.json")), flush=True)


def verify(out):
    """Directly replay the secondary laws without invoking the failed solver."""
    a = S.selected(
        out / "accounts.npz", ("tables", "fields", "full_profiles", "author_mass")
    )
    z = R.prepared(PARENT)
    count_error = direct_error = cold_error = 0.0
    for u in range(386):
        for w in range(2):
            counts = np.zeros((8, 8, 8))
            for t in range(16):
                counts[tuple(z["source_paths"][u, w, t : t + 3])] += 1
            count_error = max(
                count_error, float(np.max(abs(counts / 16 - a["author_mass"][u, w])))
            )
    for i in np.ndindex(2, 2, 2):
        table = a["tables"][i]
        direct = np.zeros_like(table[0])
        for g in range(32):
            for b in range(8):
                f = table[2, g, :, b, :]
                if f.sum() > 0:
                    direct[g, :, b, :] = np.outer(f.sum(1), f.sum(0)) / f.sum()
        direct_error = max(direct_error, float(np.max(abs(direct - table[0]))))
    readout = S.selected(out / "cold_readout.npz", ("readout",))["readout"]
    with (
        (PARENT / "cold_inputs.jsonl").open() as f,
        (out / "cold_outputs.jsonl").open() as g,
    ):
        for line, response in zip(f, g, strict=True):
            r, v = json.loads(line), json.loads(response)
            cells = [e["cell"] for e in r["events"]]
            expected = (
                sum(
                    (
                        readout[..., cells[t], cells[t + 1], cells[t + 2]]
                        for t in range(16)
                    )
                )
                / 16
            )
            actual = np.asarray(v["scores"], float)
            if not np.array_equal(np.isfinite(expected), np.isfinite(actual)):
                raise ValueError("Cold support status mismatch")
            known = np.isfinite(expected)
            cold_error = max(
                cold_error,
                float(np.max(abs(expected[known] - actual[known])))
                if known.any()
                else 0,
            )
    summary = R.read(out / "summary.json")
    passed = (
        max(
            count_error,
            direct_error,
            cold_error,
            summary["adjacent_full_profile_max_error"],
        )
        < 1e-12
    )
    value = {
        "PASS": passed,
        "count_error": count_error,
        "profile_markov_direct_error": direct_error,
        "cold_error": cold_error,
        "same_agent_not_blind": True,
        "primary_remains_NOT_COMPUTED": True,
    }
    R.write(out / "verification.json", value)
    R.seal(out, "verification_seal.json", ("verification.json",))
    print(json.dumps(value), flush=True)
    if not passed:
        raise ValueError("Secondary verification failed")


def check(out):
    """Verify original failure and companion provenance without rerunning science."""
    R.check(PARENT)
    manifest = R.read(out / "manifest.json")
    for group in ("code", "parent_science_code", "inputs"):
        for name, value in manifest[group].items():
            if R.digest(ROOT / name) != value:
                raise ValueError("Changed input or code:" + name)
    for stage in ("computed_seal.json", "cold_seal.json", "verification_seal.json"):
        for name, value in R.read(out / stage).items():
            if R.digest(out / name) != value:
                raise ValueError("Changed output:" + name)
    print("HM235 secondary/failure seals PASS;primary remains pending")


def main():
    """Execute only a requested bounded stage;never overwrite original r1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "cold", "verify", "check"))
    parser.add_argument(
        "--output", type=Path, default=PARENT / "partial_markov_accounts_r1"
    )
    args = parser.parse_args()
    globals()[args.stage](args.output)


if __name__ == "__main__":
    main()
