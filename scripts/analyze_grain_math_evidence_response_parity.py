#!/usr/bin/env python3
"""HM54 sign-even/odd evidence-response decomposition; zero model calls or fits."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_evidence_response_operation import OUT, check
from scripts.run_grain_math_response import write_csv
from scripts.verify_grain_math_relational_cut import csv_rows


def parity(native, plus, minus):
    """Separate evidence response preserved under reversing a fixed perturbation."""
    even_action = (plus + minus) / 2 - native
    odd_action = (plus - minus) / 2
    reconstructed = (
        native + even_action + odd_action,
        native + even_action - odd_action,
    )
    return {
        "native": native,
        "plus": plus,
        "minus": minus,
        "even_action": even_action,
        "odd_action": odd_action,
        "reconstruction_error": max(
            abs(reconstructed[0] - plus), abs(reconstructed[1] - minus)
        ),
        "both_signs_smaller": abs(plus) < abs(native) and abs(minus) < abs(native),
    }


def summarize(rows):
    """Report RMS by component without fitting or orienting to receiver labels."""
    result = {
        "pairs": len(rows),
        "both_signs_smaller": sum(r["both_signs_smaller"] for r in rows),
    }
    for key in ("native", "plus", "minus", "even_action", "odd_action"):
        result[key + "_rms"] = math.sqrt(
            math.fsum(r[key] ** 2 for r in rows) / len(rows)
        )
    return result


def analyze(out):
    """Use verified frozen contrasts only and seal this result-after companion."""
    check(out)
    if (
        read_json(out / "verification.json")["status"]
        != "VERIFIED_FINITE_HM54_NOT_GLOBAL"
    ):
        raise ValueError("Verified complete HM54 required")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / name) != sha:
            raise ValueError("Changed result " + name)
    inputs = csv_rows(out / "03/evidence_pair_actions.csv")
    selected = {}
    for row in inputs:
        if row["arm"] in ("source_plus", "source_minus"):
            key = (row["content"], row["class"], row["person"])
            selected.setdefault(key, {})[row["arm"]] = row
    if len(selected) != 48 or any(
        set(v) != {"source_plus", "source_minus"} for v in selected.values()
    ):
        raise ValueError("Missing 48 paired sign observations")
    rows = []
    for key, arms in selected.items():
        plus, minus = arms["source_plus"], arms["source_minus"]
        if (
            plus["pair_id"] != minus["pair_id"]
            or abs(
                float(plus["native_log_odds_contrast"])
                - float(minus["native_log_odds_contrast"])
            )
            > 1e-10
        ):
            raise ValueError("Matched native contrast changed")
        rows.append(
            dict(zip(("content", "class", "person"), key, strict=True))
            | parity(
                float(plus["native_log_odds_contrast"]),
                float(plus["patched_log_odds_contrast"]),
                float(minus["patched_log_odds_contrast"]),
            )
        )
    groups = defaultdict(list)
    for row in rows:
        groups[row["class"]].append(row)
    summaries = [{"class": k} | summarize(v) for k, v in groups.items()]
    target = out / "04_response_parity"
    if target.exists():
        raise FileExistsError("Preserve result-after companion")
    target.mkdir()
    write_csv(target / "pairs.csv", rows)
    write_csv(target / "class_summary.csv", summaries)
    write(
        target / "summary.json",
        {
            "scope": "result_after_finite_C2_arithmetic_not_new_group_theorem",
            "pairs": 48,
            "classes": summaries,
            "maximum_reconstruction_error": max(
                r["reconstruction_error"] for r in rows
            ),
            "new_model_calls": 0,
            "fits": 0,
            "nonzero_even_component_compatible_with_nonlinear_payload": True,
            "input_sha256": digest(out / "03/evidence_pair_actions.csv"),
            "code_sha256": digest(Path(__file__)),
        },
    )
    print(
        f"PARITY:48 pairs,0 calls,maximum reconstruction {max(r['reconstruction_error'] for r in rows):.3e}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    analyze(parser.parse_args().output)
