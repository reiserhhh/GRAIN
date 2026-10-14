#!/usr/bin/env python3
"""Independent zero-forward arithmetic replay of HM12 token-mean interventions."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = ROOT / "results/grain_math_response_v1/token_mean_action_r1"
OUTPUT = ROOT / "results/grain_math_response_v1/token_mean_verification_r1"


def categorical_score(record, values):
    """Recompute a conditional categorical expectation without the original reader helper."""
    logits = np.asarray(record["candidate_logits"], float)
    mass = np.exp(logits - logits.max())
    return float(mass @ values / mass.sum())


def read_table(path):
    """Load numeric result tables for an independent address-by-address replay."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify(packet, out):
    """Check every actual readout, decomposition, binding rectangle and frozen artifact."""
    if out.exists():
        raise FileExistsError("preserve prior verification")
    manifest = frozen_files(packet)
    for name, expected in read_json(packet / "02/artifact_hashes.json").items():
        if digest(packet / name) != expected:
            raise ValueError(f"frozen HM12 artifact changed: {name}")
    source = ROOT / manifest["spec"]["source_packet"]
    source_ledger = read_json(packet / "00/source_hashes.json")
    for name, expected in source_ledger.items():
        if digest(source / name) != expected:
            raise ValueError(f"frozen old artifact changed: {name}")
    design = read_json(packet / "00/design.json")
    captures = read_json(packet / "00/capture_addresses.json")
    inputs = {(r["doc_id"], r["query"]): r for r in captures}
    canonical = {
        r["doc_id"]: r for r in captures if r["query"] == manifest["spec"]["queries"][0]
    }
    references = {
        (r["pair_id"], r["query"]): r for r in read_json(packet / "00/references.json")
    }
    cache = {}

    def state(doc_id, span):
        address = canonical[doc_id]
        if doc_id not in cache:
            with np.load(source / "01/units" / f"unit_{address['unit']:04d}.npz") as a:
                cache[doc_id] = {
                    s: a[f"L16_{s}"].astype(float) for s in address["spans"]
                }
        return cache[doc_id][span]

    max_geometry, max_score, max_action, max_centered_ideal = 0.0, 0.0, 0.0, 0.0
    max_noop, max_live, rounded_max = 0.0, 0.0, 0.0
    noops, grouped = 0, defaultdict(dict)
    values = np.asarray(manifest["alphabet"]["values"], float)
    records = {}
    for row in design:
        unit = read_json(packet / "01/units" / f"unit_{row['unit']:04d}.json")
        if unit["address"] != row:
            raise ValueError("intervention address changed")
        current, donor = (
            state(row["receiver_doc"], row["span"]),
            state(row["donor_doc"], row["span"]),
        )
        delta = donor - current
        t = len(delta)
        # Check the orthogonal token-space projection separately from mean()/broadcast.
        j = np.ones((t, t)) / t
        projected, residual = j @ delta, (np.eye(t) - j) @ delta
        max_centered_ideal = max(
            max_centered_ideal, float(np.max(np.abs(residual.sum(axis=0) / t)))
        )
        exact_mean = np.tile(delta.sum(axis=0) / t, (t, 1))
        exact_residual = delta - exact_mean
        ideal = exact_mean if row["arm"] == "mean_only" else exact_residual
        actual = (current + ideal).astype(np.float32).astype(float)
        shift = actual - current
        calculated = {
            "full_displacement_norm": np.linalg.norm(delta),
            "mean_displacement_norm": np.linalg.norm(projected),
            "centered_displacement_norm": np.linalg.norm(residual),
            "intervention_norm": np.linalg.norm(shift),
            "rounded_mean_shift_norm": np.linalg.norm(shift.sum(axis=0) / t),
            "rounded_mean_shift_max": np.max(np.abs(shift.sum(axis=0) / t)),
            "rounding_max": np.max(np.abs(shift - ideal)),
        }
        for name, value in calculated.items():
            max_geometry = max(max_geometry, abs(float(value) - unit[name]))
        expected_noop = bool(np.array_equal(actual, current))
        if expected_noop != unit["is_exact_input_noop"] or t != row["tokens"]:
            raise ValueError("token opportunity or exact input identity changed")
        if (
            row["positions"]
            != inputs[row["receiver_doc"], row["query"]]["spans"][row["span"]][
                "positions"
            ]
        ):
            raise ValueError("receiver token positions changed")
        score = categorical_score(unit["native"], values)
        max_score = max(max_score, abs(score - unit["native"]["conditional_score"]))
        ref = references[row["pair_id"], row["query"]]
        action = score - ref["baseline"]
        max_action = max(max_action, abs(action - unit["action"]))
        max_live = max(max_live, unit["live_body_error"], unit["live_question_error"])
        if row["arm"] == "centered_only":
            rounded_max = max(rounded_max, calculated["rounded_mean_shift_norm"])
            if t == 1:
                noops += 1
                max_noop = max(max_noop, abs(action))
        grouped[row["pair_id"], row["query"]][row["arm"]] = score
        records[row["unit"]] = unit
    split, contrast, table_error = {}, defaultdict(list), 0.0
    for key, arms in grouped.items():
        ref = references[key]
        b, f, m, c = (
            ref["baseline"],
            ref["full"],
            arms["mean_only"],
            arms["centered_only"],
        )
        split[key] = {
            "full_action": f - b,
            "mean_action": m - b,
            "centered_action": c - b,
            "mixed_action": (f - b) - ((m - b) + (c - b)),
            "full_minus_mean": f - m,
            "centered_minus_baseline": c - b,
        }
        contrast[ref["family"], ref["axis"], ref["order"], ref["query"]].append(ref)
    decomposition = read_table(packet / "02/response_decomposition.csv")
    for row in decomposition:
        key = int(row["pair_id"]), row["query"]
        for name, value in split[key].items():
            table_error = max(table_error, abs(float(row[name]) - value))
    binding = read_table(packet / "02/binding_by_family.csv")
    for row in binding:
        key = int(row["family"]), row["axis"], int(row["order"]), row["query"]
        refs = sorted(contrast[key], key=lambda r: r["polarity"])
        if len(refs) != 2 or [r["polarity"] for r in refs] != [0, 1]:
            raise ValueError("missing binding polarity")
        low, high = refs
        for name in (
            "full_action",
            "mean_action",
            "centered_action",
            "mixed_action",
            "full_minus_mean",
        ):
            value = (
                split[high["pair_id"], high["query"]][name]
                - split[low["pair_id"], low["query"]][name]
            )
            table_error = max(table_error, abs(float(row[name]) - value))
    cells = defaultdict(list)
    for row in binding:
        cells[row["role"], row["axis"], row["order"], row["query"]].append(row)
    summary_rows = read_table(packet / "02/binding_summary.csv")
    for row in summary_rows:
        selected = cells[row["role"], row["axis"], row["order"], row["query"]]
        value = np.mean([float(r[row["metric"]]) for r in selected])
        table_error = max(table_error, abs(value - float(row["mean"])))
    expected_counts = (768, 384, 192, 180)
    if (
        len(records),
        len(decomposition),
        len(binding),
        len(summary_rows),
    ) != expected_counts:
        raise ValueError("result table coverage differs from frozen design")
    if (
        max_geometry > 1e-10
        or max_score > 1e-12
        or max_action > 1e-12
        or table_error > 1e-12
        or max_live != 0
        or max_noop > manifest["spec"]["noop_readout_tolerance"]
    ):
        raise ValueError("independent arithmetic or actual replay checks failed")
    result = {
        "status": "FINITE_PACKET_VERIFIED_NOT_GLOBAL_SCOPE_COMPLETE",
        "actual_calls_checked": len(records),
        "source_artifact_hashes_checked": len(source_ledger),
        "table_counts": list(expected_counts),
        "single_token_actual_noops": noops,
        "noop_max_readout_error": max_noop,
        "geometry_max_error": max_geometry,
        "categorical_score_max_error": max_score,
        "action_max_error": max_action,
        "table_max_error": table_error,
        "centered_ideal_mean_max": max_centered_ideal,
        "centered_actual_rounded_mean_norm_max": float(rounded_max),
        "live_state_max_error": max_live,
        "new_neural_calls": 0,
        "verifier_is_same_agent_alternate_formula_not_blind_rebuilder": True,
        "verifier_sha256": digest(Path(__file__)),
        "primary_manifest_sha256": digest(packet / "00/manifest.json"),
    }
    write_json(out / "verification.json", result)
    write_json(
        out / "frozen_primary_artifact_hashes.json",
        {
            str(p.relative_to(packet)): digest(p)
            for p in sorted(packet.rglob("*"))
            if p.is_file()
        },
    )
    print(result, flush=True)


def main():
    """Read a completed packet without a concurrent collector or mutation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with collector_guard(args.packet):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
