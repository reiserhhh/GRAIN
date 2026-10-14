#!/usr/bin/env python3
"""HM62: exact moment-based construction diagnostics on frozen HM61 readings."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_author_moment_measurement as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_heterogeneous_evidence_cube import check as parent_check
from scripts.run_grain_math_response import write_csv

SPEC = ROOT / "configs/grain_math_author_moment_measurement_v1.json"
OUT = ROOT / "results/grain_math_response_v1/author_moment_measurement_r1"
CODE = (
    "configs/grain_math_author_moment_measurement_v1.json",
    "docs/GRAIN_AUTHOR_MOMENT_MEASUREMENT_PLAN_V1.md",
    "grain_research/construction/math_author_moment_measurement.py",
    "scripts/run_grain_math_author_moment_measurement.py",
    "scripts/verify_grain_math_author_moment_measurement.py",
    "tests/test_grain_math_author_moment_measurement.py",
)


def prepare(out):
    """Bind the existing numerical readings and freeze the new finite analysis."""
    if out.exists():
        raise FileExistsError("Keep previous finite results")
    spec = read_json(SPEC)
    src = ROOT / spec["source"]
    parent = parent_check(src)
    if (
        read_json(src / "verification.json")["status"]
        != "VERIFIED_FINITE_HM61_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM61 required")
    if (
        spec["families"] != [0, 1]
        or spec["orders"] != [1, 2, 3, 6]
        or spec["max_linear_programs"] != 192
    ):
        raise ValueError("Complete fixed finite design required")
    for name, sha in read_json(src / "03/manifest.json")["files"].items():
        if digest(src / name) != sha:
            raise ValueError("HM61 result drift")
    targets = parent["spec"]["targets"]
    tables = {(f, u): np.zeros((64, 6)) for f in (0, 1) for u in spec["units"]}
    seen = set()
    with (src / "03/native.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            f, k, q = int(r["family"]), int(r["corner"]), targets.index(r["query"])
            if (f, k, q) in seen:
                raise ValueError("Duplicate native configuration reading")
            seen.add((f, k, q))
            for u in spec["units"]:
                tables[f, u][k, q] = float(r[u])
    if len(seen) != 768:
        raise ValueError("All family/corner/query readings required")
    (out / "00/code").mkdir(parents=True)
    hashes = {}
    for name in CODE:
        dest = out / "00/code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        hashes[name] = digest(dest)
    np.savez_compressed(
        out / "00/native_tables.npz",
        **{f"family{f}_{u}": v for (f, u), v in tables.items()},
    )
    source_names = (
        "00/manifest.json",
        "03/manifest.json",
        "03/native.csv",
        "03/coefficients.npz",
        "verification.json",
    )
    write(
        out / "00/manifest.json",
        {
            "packet": "HM62",
            "spec": spec,
            "targets": targets,
            "code": hashes,
            "assets": {"00/native_tables.npz": digest(out / "00/native_tables.npz")},
            "source_files": {
                str((src / n).relative_to(ROOT)): digest(src / n) for n in source_names
            },
            "scope": spec["scope"],
            "new_model_calls": 0,
            "original_scope_complete": False,
            "result_after_analysis_not_new_blind_validation": True,
        },
    )
    write(out / "run_status.json", {"status": "PREPARED", "linear_programs": 0})


def check(out):
    """Validate the frozen analysis and input bindings without a model call."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen HM62 code drift:" + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen HM62 asset drift:" + name)
    for name, sha in m["source_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen HM61 source drift:" + name)
    return m


def run(out):
    """Describe all moments and attain bounded identification sets; do not tune."""
    started = time.monotonic()
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] != "PREPARED" or (out / "03").exists():
        raise RuntimeError("Do not overwrite or automatically retry finite analysis")
    spec, targets = m["spec"], m["targets"]
    _, masks, _ = M.cube()
    uniform = np.full(64, 1 / 64)
    outputs, contrasts, bounds, transfers, vectors = [], [], [], [], []
    witnesses, source_results = {}, {}
    summary = {"moment_intervals": {}, "parity_contrasts": {}}
    (out / "01").mkdir()
    try:
        for family in (0, 1):
            with np.load(out / "00/native_tables.npz", allow_pickle=False) as arrays:
                tables = {
                    u: arrays[f"family{family}_{u}"].copy() for u in spec["units"]
                }
            for unit, values in tables.items():
                coefficients = M.coefficients(values)
                for j in range(1, 64):
                    pp, pm = M.parity_profiles(masks[j], spec["parity_strength"])
                    pair_values = [M.integrate(values, p) for p in (pp, pm)]
                    pair_moments = [M.moments(p) for p in (pp, pm)]
                    for side, p, mu, val in zip(
                        (1, -1), (pp, pm), pair_moments, pair_values
                    ):
                        if np.max(np.abs(val - mu @ coefficients)) > spec["tolerance"]:
                            raise ArithmeticError(
                                "Distribution-to-target contraction failed"
                            )
                        for q, target in enumerate(targets):
                            outputs.append(
                                {
                                    "family": family,
                                    "unit": unit,
                                    "subset": j,
                                    "mask": "".join(map(str, masks[j])),
                                    "order": int(masks[j].sum()),
                                    "side": side,
                                    "query": target,
                                    "mean_reading": float(val[q]),
                                    "first_moment_only": float(
                                        mu[masks.sum(1) <= 1]
                                        @ coefficients[masks.sum(1) <= 1, q]
                                    ),
                                }
                            )
                    for q, target in enumerate(targets):
                        contrasts.append(
                            {
                                "family": family,
                                "unit": unit,
                                "subset": j,
                                "mask": "".join(map(str, masks[j])),
                                "order": int(masks[j].sum()),
                                "query": target,
                                "contrast": float(
                                    pair_values[0][q] - pair_values[1][q]
                                ),
                                "coefficient_formula": float(
                                    2 * spec["parity_strength"] * coefficients[j, q]
                                ),
                                "same_means": bool(masks[j].sum() > 1),
                                "same_means_and_covariance": bool(masks[j].sum() > 2),
                            }
                        )
                for order in spec["orders"]:
                    a, width_values = M.moment_design(order), []
                    mu = a @ uniform
                    for q, target in enumerate(targets):
                        if (
                            status["linear_programs"] + 2 > spec["max_linear_programs"]
                            or time.monotonic() - started > spec["max_seconds"]
                        ):
                            raise TimeoutError("192 LP /120 seconds bound")
                        interval = M.sharp_interval(values[:, q], mu, order)
                        status["linear_programs"] += 2
                        width_values.append(interval["width"])
                        key = f"family{family}_{unit}_q{q}_order{order}"
                        for side in ("lower", "upper"):
                            witnesses[f"{key}_{side}_p"] = interval[side][
                                "distribution"
                            ]
                            witnesses[f"{key}_{side}_dual"] = interval[side]["dual"]
                        bounds.append(
                            {
                                "family": family,
                                "unit": unit,
                                "query": target,
                                "order": order,
                                "lower": interval["lower"]["value"],
                                "upper": interval["upper"]["value"],
                                "width": interval["width"],
                                "uniform_mean": float(uniform @ values[:, q]),
                                "certificate_error": max(
                                    interval[side]["error"]
                                    for side in ("lower", "upper")
                                ),
                            }
                        )
                        if family == 0:
                            source_results[unit, q, order] = interval
                    summary["moment_intervals"][
                        f"family{family}_{unit}_order{order}"
                    ] = {
                        "mean_width": float(np.mean(width_values)),
                        "maximum_width": float(np.max(width_values)),
                        "minimum_width": float(np.min(width_values)),
                    }
                higher = masks.sum(1) > 2
                summary["parity_contrasts"][f"family{family}_{unit}"] = {
                    "all_higher_than_second_max_abs": float(
                        2
                        * spec["parity_strength"]
                        * np.max(np.abs(coefficients[higher]))
                    ),
                    "all_higher_than_first_max_abs": float(
                        2
                        * spec["parity_strength"]
                        * np.max(np.abs(coefficients[masks.sum(1) > 1]))
                    ),
                }
            if family == 0:
                source_arrays = {key: value for key, value in witnesses.items()}
                np.savez_compressed(out / "01/source_witnesses.npz", **source_arrays)
                write(
                    out / "01/source_barrier.json",
                    {
                        "file_sha256": digest(out / "01/source_witnesses.npz"),
                        "linear_programs": status["linear_programs"],
                        "created_epoch": time.time(),
                        "result_after_existing_application_data": True,
                    },
                )
            else:
                for (unit, q, order), interval in source_results.items():
                    p0, p1 = (
                        interval["lower"]["distribution"],
                        interval["upper"]["distribution"],
                    )
                    transfers.append(
                        {
                            "unit": unit,
                            "query": targets[q],
                            "order": order,
                            "source_width": interval["width"],
                            "application_contrast_on_same_distributions": float(
                                (p1 - p0) @ tables[unit][:, q]
                            ),
                            "first_family_witnesses_only": True,
                        }
                    )
            with np.load(
                ROOT / spec["source"] / "03/coefficients.npz", allow_pickle=False
            ) as old:
                for stage in ("body_endpoint", "query_endpoint"):
                    c = old[f"family{family}_{stage}_coefficients"]
                    for j in range(1, 64):
                        vectors.append(
                            {
                                "family": family,
                                "stage": stage,
                                "subset": j,
                                "order": int(masks[j].sum()),
                                "paired_mean_vector_displacement_norm": float(
                                    2 * spec["parity_strength"] * np.linalg.norm(c[j])
                                ),
                                "all_first_moments_identical": bool(masks[j].sum() > 1),
                            }
                        )
            status.update(status="ANALYZING", completed_families=family + 1)
            write(out / "run_status.json", status)
        if status["linear_programs"] != 192:
            raise ValueError("Full fixed LP battery missing")
        (out / "03").mkdir()
        np.savez_compressed(out / "03/witnesses.npz", **witnesses)
        for name, rows in (
            ("profiles", outputs),
            ("contrasts", contrasts),
            ("bounds", bounds),
            ("source_witness_transfer", transfers),
            ("vector_contrasts", vectors),
        ):
            write_csv(out / f"03/{name}.csv", rows)
        summary.update(
            packet="HM62",
            actual_linear_programs=192,
            actual_new_model_calls=0,
            CPU_seconds=time.monotonic() - started,
            table_counts={
                "profiles": len(outputs),
                "contrasts": len(contrasts),
                "bounds": len(bounds),
                "transfers": len(transfers),
                "vectors": len(vectors),
            },
            new_human_personality_accuracy=False,
            original_scope_complete=False,
            identification_intervals_not_confidence_intervals=True,
            result_after_analysis_not_new_blind_validation=True,
        )
        write(out / "03/summary.json", summary)
        write(
            out / "03/manifest.json",
            {
                "files": {
                    str(p.relative_to(out)): digest(p)
                    for p in (out / "03").iterdir()
                    if p.is_file()
                }
            },
        )
        status.update(
            status="COMPLETE",
            CPU_seconds=summary["CPU_seconds"],
            original_scope_complete=False,
        )
        write(out / "run_status.json", status)
        print(
            "HM62 ACTUAL:192 LPs,126 controlled distribution profiles/family,0 model calls",
            flush=True,
        )
    except BaseException as error:
        status.update(
            status="STOPPED", error=repr(error), CPU_seconds=time.monotonic() - started
        )
        write(out / "run_status.json", status)
        raise


def main():
    """Run only the declared finite preparation/analysis or integrity check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "check"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "check": check}[args.command](args.out)


if __name__ == "__main__":
    main()
