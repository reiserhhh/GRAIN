#!/usr/bin/env python3
"""Freeze a completed span packet and report conditional family action intervals."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_native_joint import frozen_files, write_json
from scripts.run_grain_math_response import write_csv

DEFAULT = ROOT / "results/grain_math_response_v1/span_binding_r1"
OUTPUT = ROOT / "results/grain_math_response_v1/span_binding_final_verification_r1"


def empirical_interval(values, seed=1021, draws=2000):
    """Resample fixed material families, not neurons, patches or human authors."""
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("at least two finite material families required")
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def finalize(source, diagnostic, readout, out):
    """Require full numeric verification and preserve all roles, orders and arms."""
    if out.exists():
        raise FileExistsError("preserve the existing final verification")
    frozen_files(source)
    primary = read_json(source / "04/summary.json")
    status = read_json(source / "run_status.json")
    if (
        primary["status"] != "COMPLETE"
        or status["status"] != "COMPLETE_PATCH"
        or primary["completed_units"] != primary["expected_units"]
        or primary["completed_units"] != 9696
        or primary["capture_calls"] != 672
        or primary["complete_polarity_contrasts"] != 4800
        or primary["noop_units"] != 96
    ):
        raise ValueError("full declared packet has not completed")
    for name in (diagnostic, readout):
        check = read_json(name / "verification.json")
        if check["status"] != "COMPLETE_CHECKED" or check["new_model_calls"] != 0:
            raise ValueError("complete zero-forward numeric checks required")
        for path, expected in read_json(name / "checked_source_hashes.json").items():
            if digest(source / path) != expected:
                raise ValueError(f"verified source artifact changed: {path}")
    with (source / "04/binding_action_by_family.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(dict)
    for row in rows:
        key = (
            row["role"],
            row["axis"],
            int(row["layer"]),
            row["query"],
            int(row["order"]),
        )
        group = grouped[key]
        address = int(row["family"]), row["arm"]
        if address in group:
            raise ValueError("duplicate family/arm contrast")
        group[address] = float(row["binding_action"])
    summary = []
    counts = {"source": 8, "validation": 4, "application": 4}
    arms = ("full_property", "projection", "complement", "random_matched", "neutral")
    for key, group in sorted(grouped.items()):
        families = sorted({address[0] for address in group})
        if len(families) != counts[key[0]] or len(group) != len(families) * len(arms):
            raise ValueError("incomplete family-level arm table")
        random = np.array([group[f, "random_matched"] for f in families])
        for arm in arms:
            actions = np.array([group[f, arm] for f in families])
            low, high = empirical_interval(actions)
            diff_low, diff_high = empirical_interval(actions - random)
            summary.append(
                {
                    "role": key[0],
                    "axis": key[1],
                    "layer": key[2],
                    "query": key[3],
                    "order": key[4],
                    "arm": arm,
                    "families": len(families),
                    "mean_binding_action": float(actions.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "mean_action_minus_random": float((actions - random).mean()),
                    "difference_ci_low": diff_low,
                    "difference_ci_high": diff_high,
                    "positive_family_count": int((actions > 0).sum()),
                    "negative_family_count": int((actions < 0).sum()),
                }
            )
    if len(summary) != 900:
        raise ValueError("all role/axis/layer/query/order/arm cells required")
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "conditional_action_intervals.csv", summary)
    snapshot = out / "primary_as_checked"
    snapshot.mkdir()
    for path in (source / "04").iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot / path.name)
    hashes = {
        str(path.relative_to(source)): digest(path)
        for path in source.rglob("*")
        if path.is_file() and path.suffix in {".json", ".npz", ".csv"}
    }
    write_json(out / "frozen_primary_artifact_hashes.json", hashes)
    codes = {
        name: digest(ROOT / name)
        for name in (
            "scripts/finalize_grain_math_span_packet.py",
            "scripts/diagnose_grain_math_span_binding.py",
            "scripts/diagnose_grain_math_span_readout.py",
        )
    }
    write_json(
        out / "verification.json",
        {
            "status": "FINITE_PACKET_COMPLETE_CHECKED_NOT_GLOBAL_SCOPE_COMPLETE",
            "new_model_calls": 0,
            "capture_calls": primary["capture_calls"],
            "intervention_calls": primary["patch_calls"],
            "complete_polarity_contrasts": len(rows),
            "interval_cells": len(summary),
            "primary_artifacts_frozen": len(hashes),
            "companion_code_hashes": codes,
            "source": str(source),
            "diagnostic": str(diagnostic),
            "readout": str(readout),
            "interval_scope": "empirical family bootstrap conditional on frozen source frame; not source-refit uncertainty or human population CI",
            "result_after_interval_diagnostic": True,
            "original_18_100_44_scope_complete": False,
            "human_personality_validity_claim": False,
        },
    )
    print(read_json(out / "verification.json"), flush=True)


def main():
    """No collector, frame fitting or layer selection is performed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/span_binding_diagnostic_r2",
    )
    parser.add_argument(
        "--readout",
        type=Path,
        default=ROOT
        / "results/grain_math_response_v1/span_binding_readout_diagnostic_r1",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    finalize(args.source, args.diagnostic, args.readout, args.out)


if __name__ == "__main__":
    main()
