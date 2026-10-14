#!/usr/bin/env python3
"""Preserve HM39's failed verifier; audit only dimensionless ratios by exact sums."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_curvature_ratio_audit import exact_ratio_audit
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_response import write_csv, write_json

OUT = ROOT / "results/grain_math_response_v1/readout_curvature_r1"
DEST = ROOT / "results/grain_math_response_v1/readout_curvature_verification_r2"
PRIMARY = ROOT / "scripts/verify_grain_math_readout_curvature.py"
SHA = "e22153828e930237fdaf5ec0992561471db945ff18b92da07b2b071d288fa1e1"


def main():
    """Hash-seal companion code and run the original physical verifier unchanged."""
    started = time.monotonic()
    if DEST.exists() or digest(PRIMARY) != SHA:
        raise ValueError("preserve output and exact frozen original verifier")
    failure = read_json(OUT.parent / "readout_curvature_verification_r1/failure.json")
    reserve = failure["elapsed_seconds"]
    if (
        reserve != 8.442760915961117
        or failure["status"] != "VERIFICATION_FAILED_NOT_ACCEPTED"
    ):
        raise ValueError("exact retained failed verification required")
    code = (
        "scripts/verify_grain_math_readout_curvature_r2.py",
        "grain_research/construction/math_curvature_ratio_audit.py",
        "tests/test_grain_math_curvature_ratio_audit.py",
        "docs/GRAIN_READOUT_CURVATURE_NUMERICAL_AUDIT_R2.md",
    )
    seal = {name: digest(ROOT / name) for name in code}
    seal[str(PRIMARY.relative_to(ROOT))] = SHA
    source = PRIMARY.read_text()
    old = """                else:
                    close(float(row[field]), value)
        elapsed = time.monotonic() - started"""
    new = """                elif field.endswith("_over_total_ms"):
                    ratio_callback(float(row[field]), value, values, field, row)
                else:
                    close(float(row[field]), value)
        elapsed = time.monotonic() - started"""
    if source.count(old) != 1:
        raise ValueError("only the original metric-field check may change")
    source = source.replace(old, new)
    source = source.replace(
        'remaining = spec["maximum_analysis_seconds"] - status["analysis_seconds"]',
        'remaining = spec["maximum_analysis_seconds"] - status["analysis_seconds"] - PRIOR_FAILED_SECONDS',
    )
    source = source.replace(
        'elapsed + status["analysis_seconds"]',
        'elapsed + status["analysis_seconds"] + PRIOR_FAILED_SECONDS',
    )
    rows = []

    def ratio_callback(actual, alternate, values, field, row):
        a = exact_ratio_audit(values, field, actual)
        b = exact_ratio_audit(values, field, alternate)
        rows.append(
            {
                **{k: row[k] for k in ("query", "role", "arm", "measure")},
                "field": field,
                **a,
                "alternate_error": b["absolute_error"],
            }
        )

    module = types.ModuleType("_hm39_ratio_companion")
    module.__file__ = str(PRIMARY)
    module.__dict__.update(ratio_callback=ratio_callback, PRIOR_FAILED_SECONDS=reserve)
    # Only hash-verified local frozen code, with one checked numerical adapter.
    exec(compile(source, str(PRIMARY), "exec"), module.__dict__)  # noqa: S102
    with threadpool_limits(limits=2):
        module.verify(OUT, DEST)
    write_json(
        DEST / "companion_manifest.json",
        {
            "files": seal,
            "prior_failed_seconds": reserve,
            "primary_unchanged": True,
            "ratio_checks_only": True,
            "physical_absolute_tolerance": 1e-8,
            "new_model_calls": 0,
        },
    )
    write_csv(DEST / "ratio_audit.csv", rows)
    summary = read_json(DEST / "summary.json")
    summary.update(
        ratio_fields=len(rows),
        unresolved_denominator_ratio_fields=sum(
            not r["denominator_resolved_at_physical_1e8"] for r in rows
        ),
        prior_failed_verification_seconds=reserve,
        primary_verification_failure_preserved=True,
    )
    write_json(DEST / "summary.json", summary)
    for name, sha in seal.items():
        if digest(ROOT / name) != sha:
            raise ValueError("companion code changed during numerical verification")
    elapsed = time.monotonic() - started
    summary["verification_seconds_including_companion"] = elapsed
    summary["combined_analysis_seconds"] = (
        read_json(OUT / "run_status.json")["analysis_seconds"] + reserve + elapsed
    )
    if summary["combined_analysis_seconds"] > 120:
        raise TimeoutError("combined budget includes all companion work")
    write_json(DEST / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
