#!/usr/bin/env python3
"""Compare material organization at identical frozen kernel observation units."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_configuration_modes as M
from scripts import run_grain_configuration_kernel_hm144 as H

OUT = ROOT / "results/grain_math_response_v1/configuration_kernel_unit_replay_r1"
CODE = [
    "docs/GRAIN_CONFIGURATION_UNIT_REPLAY_PLAN_HM144.md",
    "scripts/replay_grain_configuration_unit_hm144.py",
    "grain_research/construction/math_configuration_modes.py",
]


def prepare(out):
    """Seal the pre-existing reference scales before any partition comparison."""
    H.verify_contract(H.OUTPUT)
    out.mkdir(parents=True, exist_ok=False)
    H.write(
        out / "contract.json",
        {
            "primary_seal_sha": H.sha(H.OUTPUT / "output_seal.json"),
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "draws": 1000,
            "seed": 144,
        },
    )


def check(out):
    """Check input identities without interpreting any private text or labels."""
    H.verify_contract(H.OUTPUT)
    c = H.read(out / "contract.json")
    if H.sha(H.OUTPUT / "output_seal.json") != c["primary_seal_sha"]:
        raise ValueError("primary seal changed")
    for p, digest in c["code"].items():
        if H.sha(ROOT / p) != digest:
            raise ValueError("replay code changed")
    for p, digest in H.read(H.OUTPUT / "output_seal.json").items():
        if H.sha(H.OUTPUT / p) != digest:
            raise ValueError("frozen primary output changed")
    return c


def run(out):
    """Execute all16 pairings without selecting an improved score."""
    c = check(out)
    H.write(out / "execution_started.json", {"status": "RUNNING"})
    began = time.monotonic()
    originals = H.read(H.OUTPUT / "result.json")["comparisons"]
    rows, results, saved, errors = [], [], {}, []
    with np.load(H.OUTPUT / "private_kernel_arrays.npz") as a:
        src = a["role_source_train"]
        for space in ("grain15", "e5_1024"):
            widths = next(
                r["bandwidth_squared"]
                for r in originals
                if r["space"] == space and r["partition"] == "chronological"
            )
            parts = {
                p: a[f"parcels_{p}_{space}"]
                for p in ("chronological", "venue_balanced")
            }
            errors.append(
                float(
                    np.max(
                        abs(
                            parts["chronological"].mean(1)
                            - parts["venue_balanced"].mean(1)
                        )
                    )
                )
            )
            for recipient in ("source_validation", "held"):
                us = a[f"role_{recipient}"]
                same_level = []
                for sp, source in parts.items():
                    for ap, application in parts.items():
                        label = f"{space}__{recipient}__{sp}__{ap}"
                        grams = M.substitution_kernels(
                            source[src], application[us], widths
                        )
                        cells = M.grouped_cells(*grams)
                        rng = np.random.default_rng(c["seed"])
                        boots = []
                        for _ in range(c["draws"]):
                            w = rng.multinomial(
                                len(src), np.full(len(src), 1 / len(src))
                            ) / len(src)
                            v = rng.multinomial(
                                len(us), np.full(len(us), 1 / len(us))
                            ) / len(us)
                            boots.append(M.grouped_cells(*grams, w, v))
                        boots = np.asarray(boots)
                        ci = np.quantile(boots, [0.025, 0.975], axis=0)
                        sn = M.grouped_cells(
                            *M.substitution_kernels(source[src], source[src], widths)
                        )
                        an = M.grouped_cells(
                            *M.substitution_kernels(
                                application[us], application[us], widths
                            )
                        )
                        alignment = cells[1, 1] / np.sqrt(sn[1, 1] * an[1, 1])
                        item = {
                            "space": space,
                            "recipient": recipient,
                            "source_partition": sp,
                            "application_partition": ap,
                            "bandwidth_squared": widths,
                            "cells": cells.tolist(),
                            "CI": ci.tolist(),
                            "departure_alignment": float(alignment),
                        }
                        results.append(item)
                        same_level.append(cells[0, 0])
                        saved[f"bootstrap_{label}"] = boots
                        for name, gs in zip(
                            ("ff", "mf", "fm", "mm"), grams, strict=True
                        ):
                            for j, mat in enumerate(gs):
                                saved[f"{name}_{j}_{label}"] = mat
                        rows.append(
                            {
                                "space": space,
                                "recipient": recipient,
                                "source_partition": sp,
                                "application_partition": ap,
                                "total_inner": float(cells.sum()),
                                "level_level": float(cells[0, 0]),
                                "level_departure": float(cells[0, 1]),
                                "departure_level": float(cells[1, 0]),
                                "departure_departure": float(cells[1, 1]),
                                "departure_alignment": float(alignment),
                                "DD_CI_low": float(ci[0, 1, 1]),
                                "DD_CI_high": float(ci[1, 1, 1]),
                            }
                        )
                        if sp == ap == "chronological":
                            original = next(
                                r
                                for r in originals
                                if r["space"] == space
                                and r["recipient"] == recipient
                                and r["partition"] == sp
                            )
                            errors.append(
                                abs(cells.sum() - original["signed_lancaster_inner"])
                            )
                errors.append(float(np.ptp(same_level)))
    if max(errors) > 1e-10:
        raise ValueError("pooled author location or primary replay changed")
    H.archive(out / "private_unit_arrays.npz", saved)
    H.table(out / "unit_metrics.csv", rows)
    H.write(
        out / "result.json",
        {
            "status": "FIXED_UNIT_PARTITION_RESPONSE_EXECUTED",
            "comparisons": results,
            "pooled_and_replay_maximum_error": max(errors),
            "CPU_wall_seconds": time.monotonic() - began,
            "new_model_API_private_raw_training_calls": 0,
            "original_scope_complete": False,
        },
    )
    H.write(
        out / "output_seal.json",
        {p.name: H.sha(p) for p in out.iterdir() if p.is_file()},
    )
    print(
        {
            "status": "EXECUTED",
            "comparisons": len(rows),
            "seconds": time.monotonic() - began,
        }
    )


def verify(out):
    """Recheck all signed identities using matrix centering, without re-running."""
    check(out)
    for p, digest in H.read(out / "output_seal.json").items():
        if H.sha(out / p) != digest:
            raise ValueError("unit replay output changed")
    errors = []
    with np.load(out / "private_unit_arrays.npz") as a:
        for r in H.read(out / "result.json")["comparisons"]:
            label = f"{r['space']}__{r['recipient']}__{r['source_partition']}__{r['application_partition']}"
            values = []
            for name in ("ff", "mf", "fm", "mm"):
                mats = [a[f"{name}_{j}_{label}"] for j in range(3)]
                n, m = mats[0].shape
                hs, ha = (
                    np.eye(n) - np.ones((n, n)) / n,
                    np.eye(m) - np.ones((m, m)) / m,
                )
                values.append(
                    float(np.prod([hs @ g @ ha for g in mats], axis=0).mean())
                )
            ff, mf, fm, mm = values
            alternative = np.array([[mm, mf - mm], [fm - mm, ff - mf - fm + mm]])
            errors.extend(abs(alternative - r["cells"]).ravel())
            errors.extend(
                abs(
                    np.quantile(a[f"bootstrap_{label}"], [0.025, 0.975], axis=0)
                    - r["CI"]
                ).ravel()
            )
    if max(errors) > 1e-10:
        raise ValueError("independent unit replay arithmetic failed")
    print(
        {
            "status": "READONLY_ALTERNATE_CHECK_PASS",
            "checks": len(errors),
            "maximum_error": max(errors),
        }
    )


def main():
    """Expose sealing, finite execution and repeatable verification separately."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "run", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    {"prepare": prepare, "run": run, "verify": verify}[a.command](a.out)


if __name__ == "__main__":
    main()
