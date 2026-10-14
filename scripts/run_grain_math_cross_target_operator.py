#!/usr/bin/env python3
"""HM60: immutable CPU reanalysis of six-target passive evidence responses."""

from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_cross_target_operator as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_event_readout_composition import check as source_check
from scripts.run_grain_math_event_readout_composition import observation
from scripts.run_grain_math_response import write_csv

SPEC = ROOT / "configs/grain_math_cross_target_operator_v1.json"
OUT = ROOT / "results/grain_math_response_v1/cross_target_operator_r1"
CODE = (
    "configs/grain_math_cross_target_operator_v1.json",
    "docs/GRAIN_CROSS_TARGET_EVIDENCE_OPERATOR_PLAN_V1.md",
    "grain_research/construction/math_cross_target_operator.py",
    "scripts/run_grain_math_cross_target_operator.py",
    "scripts/verify_grain_math_cross_target_operator.py",
    "tests/test_grain_math_cross_target_operator.py",
)
IDENTITY = (
    "unit",
    "parent_unit",
    "role",
    "content",
    "query",
    "family",
    "polarity",
    "order",
    "regime",
)


def prepare(out):
    """Freeze the pre-analysis design and exact raw source identities before statistics."""
    if out.exists():
        raise FileExistsError("Preserve previous preparations/results")
    spec = read_json(SPEC)
    if (
        spec["targets"] != list(M.TARGETS)
        or spec["regimes"] != list(M.REGIMES)
        or spec["units"] != list(M.UNITS)
        or spec["new_forwards"] != 0
        or spec["numerical_zero_norm"] != M.ZERO_NORM
    ):
        raise ValueError("Fixed CPU-only design changed")
    source = ROOT / spec["source"]
    m = source_check(source)
    if (
        read_json(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM56_NOT_GLOBAL"
        or read_json(source / "binding_verification.json")["status"]
        != "VERIFIED_HM56_RAW_COMPONENT_BINDING"
        or read_json(source / "run_status.json")["status"] != "COLLECTED"
        or m["targets"] != spec["targets"]
    ):
        raise ValueError("Complete verified HM56 required")
    files = {
        str((source / n).relative_to(ROOT)): digest(source / n)
        for n in (
            "00/manifest.json",
            "00/rows.json",
            "verification.json",
            "binding_verification.json",
            "run_status.json",
            "03/native.csv",
        )
    }
    with (source / "03/native.csv").open(newline="") as stream:
        native = {int(r["unit"]): r for r in csv.DictReader(stream)}
    rows = []
    error = 0.0
    for row in read_json(source / "00/rows.json"):
        raw, _ = observation(source, row, m)
        decoded = M.decode(raw["candidate_logits"])
        item = {k: row[k] for k in IDENTITY}
        item.update(
            candidate_logits=raw["candidate_logits"],
            candidate_mass=raw["candidate_mass"],
            **decoded,
        )
        saved = native[row["unit"]]
        if any(str(saved[k]) != str(row[k]) for k in IDENTITY):
            raise ValueError("Scientific CSV/source identity mismatch")
        error = max(
            error,
            *(abs(float(saved[k]) - decoded[k]) for k in M.UNITS),
            abs(raw["conditional_score"] - decoded["score"]),
        )
        if row["reused"]:
            path = (
                ROOT
                / m["spec"]["parent"]
                / m["reused"][str(row["parent_unit"])]["meta"]
            )
        else:
            path = source / f"01/units/unit_{row['unit']:04d}.json"
        for p in (path, path.with_suffix(".npz")):
            files[str(p.relative_to(ROOT))] = digest(p)
        item["source_meta"] = str(path.relative_to(ROOT))
        rows.append(item)
    M.validate_observations(rows)
    if (
        len(rows) != spec["expected_readouts"]
        or len(native) != len(rows)
        or error > spec["arithmetic_tolerance"]
    ):
        raise ValueError("Raw replay/coverage failure")
    (out / "00/code").mkdir(parents=True)
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        code[name] = digest(path)
    write(out / "00/observations.json", rows)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM60",
            "spec": spec,
            "code": code,
            "source_files": files,
            "observations_sha256": digest(out / "00/observations.json"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "readout_replay_max_error": error,
            "new_forwards": 0,
            "new_fit": False,
            "retrospective_reanalysis": True,
            "independent_human_units": 0,
            "source_model_weights_not_separately_hashed": True,
        },
    )
    write(
        out / "run_status.json", {"status": "PREPARED", "new_forwards": 0, "fit": False}
    )
    print(
        "HM60 PREPARED:1296 bound observations;0 neural calls;statistics not yet computed",
        flush=True,
    )


def check(out):
    """Reject source/design/snapshot drift; never load or invoke a neural observer."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen analysis code drift " + name)
    for name, sha in m["source_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen raw source drift " + name)
    if digest(out / "00/observations.json") != m["observations_sha256"]:
        raise ValueError("Frozen observation snapshot drift")
    source_check(ROOT / m["spec"]["source"])
    return m


def analyze(out):
    """Run all declared operators, spectra and label matchings without selection."""
    start = time.monotonic()
    m = check(out)
    if (out / "03").exists():
        raise FileExistsError("Do not overwrite previous scientific analysis")
    observations = read_json(out / "00/observations.json")
    entries, spectra, cycles, assignments, summary, families = [], [], [], [], [], []
    for unit in M.UNITS:
        operators = M.matrices(observations, unit)
        reference = operators["atom/pooled"]
        for name, d in operators.items():
            tags = {"unit": unit, "operator": name}
            g, c = M.graph_split(d)
            for q, content in itertools.product(range(6), repeat=2):
                entries.append(
                    tags
                    | {
                        "query": M.TARGETS[q],
                        "content": M.TARGETS[content],
                        "response": float(d[q, content]),
                        "gradient": float(g[q, content]),
                        "cycle": float(c[q, content]),
                    }
                )
            s, _ = M.spectrum(d)
            spectra.extend(
                tags
                | {"index": j + 1, "singular_value": float(x), "energy": float(x * x)}
                for j, x in enumerate(s)
            )
            cycles.extend(tags | r for r in M.rectangles(d))
            perms, traces, _ = M.assignment(d, m["spec"]["assignment_tie_tolerance"])
            assignments.extend(
                tags | {"permutation": "".join(map(str, p)), "trace": float(v)}
                for p, v in zip(perms, traces, strict=True)
            )
            summary.append(tags | M.describe(d, reference))
        families.extend({"unit": unit} | r for r in M.family_comparisons(operators))
    if (
        len(entries) != m["spec"]["expected_entries"]
        or len(cycles) != m["spec"]["expected_rectangles"]
        or len(assignments) != m["spec"]["expected_permutations"]
        or len(summary) != 2 * m["spec"]["expected_matrices_per_unit"]
    ):
        raise ValueError("Declared analysis count mismatch")
    for name, data in (
        ("entries", entries),
        ("spectra", spectra),
        ("rectangles", cycles),
        ("assignments", assignments),
        ("operator_summary", summary),
        ("family_comparisons", families),
    ):
        write_csv(out / "03" / (name + ".csv"), data)
    record = {
        "packet": "HM60",
        "source_readouts": len(observations),
        "new_forwards": 0,
        "fit_or_select_predictor": False,
        "CPU_seconds": time.monotonic() - start,
        "rows": {
            "entries": len(entries),
            "spectra": len(spectra),
            "rectangles": len(cycles),
            "assignments": len(assignments),
            "operator_summary": len(summary),
            "family_comparisons": len(families),
        },
        "pooled_atom": [r for r in summary if r["operator"] == "atom/pooled"],
        "pooled_accounts": [
            r
            for r in summary
            if r["operator"].startswith("account/")
            and r["operator"].endswith("/pooled")
        ],
        "families": families,
        "original_scope_complete": False,
        "assignment_ranks_are_not_population_pvalues": True,
    }
    write(out / "03/summary.json", record)
    write(
        out / "03/artifact_hashes.json",
        {
            p.name: digest(p)
            for p in sorted((out / "03").iterdir())
            if p.is_file() and not p.name.startswith("._")
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "ANALYZED",
            "new_forwards": 0,
            "fit": False,
            "CPU_seconds": record["CPU_seconds"],
        },
    )
    print(
        {
            "status": "ANALYZED",
            "rows": record["rows"],
            "CPU_seconds": record["CPU_seconds"],
        },
        flush=True,
    )


def main():
    """Separate design freezing from computation; retain failed stages as history."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "analyze": analyze}[args.stage](args.output)
    except Exception as error:
        if args.output.exists():
            write(
                args.output / f"{args.stage}_failure.json",
                {"error": repr(error), "new_forwards": 0},
            )
        raise


if __name__ == "__main__":
    main()
