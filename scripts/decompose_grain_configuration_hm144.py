#!/usr/bin/env python3
"""Keep frozen HM144 results and identify their level/departure contributions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_configuration_kernel as K
from grain_research.construction import math_configuration_modes as M
from scripts import run_grain_configuration_kernel_hm144 as H

PRIMARY = H.OUTPUT
OUT = ROOT / "results/grain_math_response_v1/configuration_kernel_decomposition_r1"
CODE = [
    "docs/GRAIN_CONFIGURATION_MODE_DECOMPOSITION_PLAN_HM144.md",
    "grain_research/construction/math_configuration_modes.py",
    "scripts/decompose_grain_configuration_hm144.py",
    "tests/test_grain_configuration_modes.py",
]


def prepare(out):
    """Seal this result-motivated follow-up without changing the primary packet."""
    H.verify_contract(PRIMARY)
    out.mkdir(parents=True, exist_ok=False)
    H.write(
        out / "contract.json",
        {
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "primary_seal_sha": H.sha(PRIMARY / "output_seal.json"),
            "bootstrap_draws": 1000,
            "seed": 144,
            "scope": "exact signed observation decomposition, not dynamic personality",
        },
    )


def contract(out):
    """Verify both frozen stages and all inputs before decomposition."""
    H.verify_contract(PRIMARY)
    c = H.read(out / "contract.json")
    for p, digest in c["code"].items():
        if H.sha(ROOT / p) != digest:
            raise ValueError("changed decomposition code")
    if H.sha(PRIMARY / "output_seal.json") != c["primary_seal_sha"]:
        raise ValueError("primary seal changed")
    for name, digest in H.read(PRIMARY / "output_seal.json").items():
        if H.sha(PRIMARY / name) != digest:
            raise ValueError("primary output changed")
    return c


def run(out):
    """Execute all signed contractions and fixed-feature paired intervals once."""
    c = contract(out)
    H.write(out / "execution_started.json", {"status": "RUNNING"})
    began = time.monotonic()
    saved, rows, results = {}, [], []
    with np.load(PRIMARY / "private_kernel_arrays.npz") as a:
        src = a["role_source_train"]
        for obs in H.read(PRIMARY / "result.json")["comparisons"]:
            label = f"{obs['partition']}__{obs['space']}__{obs['recipient']}"
            parcels = a[f"parcels_{obs['partition']}_{obs['space']}"]
            us = a[f"role_{obs['recipient']}"]
            grams = M.substitution_kernels(
                parcels[src], parcels[us], obs["bandwidth_squared"]
            )
            blocks = M.tensor_blocks(M.component_grams(*grams))
            cells = M.grouped_cells(*grams)
            error = abs(blocks.sum() - obs["signed_lancaster_inner"])
            if error > 1e-10 or abs(cells.sum() - blocks.sum()) > 1e-10:
                raise ValueError("original configuration not reconstructed")
            rng = np.random.default_rng(c["seed"])
            boots = []
            for _ in range(c["bootstrap_draws"]):
                w = rng.multinomial(len(src), np.full(len(src), 1 / len(src))) / len(
                    src
                )
                v = rng.multinomial(len(us), np.full(len(us), 1 / len(us))) / len(us)
                boots.append(M.grouped_cells(*grams, w, v))
            boots = np.asarray(boots)
            ci = np.quantile(boots, [0.025, 0.975], axis=0)
            source_modes = M.grouped_cells(
                *M.substitution_kernels(
                    parcels[src], parcels[src], obs["bandwidth_squared"]
                )
            )
            app_modes = M.grouped_cells(
                *M.substitution_kernels(
                    parcels[us], parcels[us], obs["bandwidth_squared"]
                )
            )
            modes = {"level": 0, "departure": 1}
            item = {
                "partition": obs["partition"],
                "space": obs["space"],
                "recipient": obs["recipient"],
                "cells": cells.tolist(),
                "bootstrap_CI": ci.tolist(),
                "64cell_closure_error": error,
                "source_norm_cells": source_modes.tolist(),
                "application_norm_cells": app_modes.tolist(),
                "level_alignment": float(
                    cells[0, 0] / np.sqrt(source_modes[0, 0] * app_modes[0, 0])
                ),
                "departure_alignment": float(
                    cells[1, 1] / np.sqrt(source_modes[1, 1] * app_modes[1, 1])
                ),
            }
            results.append(item)
            saved[f"blocks_{label}"] = blocks
            saved[f"bootstrap_{label}"] = boots
            for name, seq in zip(("ff", "mf", "fm", "mm"), grams, strict=True):
                for j, g in enumerate(seq):
                    saved[f"{name}_{j}_{label}"] = g
            for smode, si in modes.items():
                for amode, ai in modes.items():
                    rows.append(
                        {
                            "partition": obs["partition"],
                            "space": obs["space"],
                            "recipient": obs["recipient"],
                            "source_mode": smode,
                            "application_mode": amode,
                            "signed_inner": float(cells[si, ai]),
                            "CI_low": float(ci[0, si, ai]),
                            "CI_high": float(ci[1, si, ai]),
                        }
                    )
    H.archive(out / "private_mode_arrays.npz", saved)
    H.table(out / "mode_metrics.csv", rows)
    H.write(
        out / "result.json",
        {
            "status": "SIGNED_LEVEL_DEPARTURE_DECOMPOSITION_EXECUTED",
            "comparisons": results,
            "CPU_wall_seconds": time.monotonic() - began,
            "new_model_API_raw_training_calls": 0,
            "not_dynamic_personality_or_noise_deletion": True,
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
            "comparisons": len(results),
            "seconds": time.monotonic() - began,
        }
    )


def verify(out):
    """Check actual64-cell grouping and a four-substitution alternative identity."""
    contract(out)
    for name, digest in H.read(out / "output_seal.json").items():
        if H.sha(out / name) != digest:
            raise ValueError("changed frozen decomposition")
    errors = []
    with np.load(out / "private_mode_arrays.npz") as a:
        for obs in H.read(out / "result.json")["comparisons"]:
            label = f"{obs['partition']}__{obs['space']}__{obs['recipient']}"
            vals = []
            for name in ("ff", "mf", "fm", "mm"):
                mats = [a[f"{name}_{j}_{label}"] for j in range(3)]
                n, m = mats[0].shape
                hs, ha = (
                    np.eye(n) - np.ones((n, n)) / n,
                    np.eye(m) - np.ones((m, m)) / m,
                )
                vals.append(
                    float(np.prod([hs @ g @ ha for g in mats], axis=0).sum() / (n * m))
                )
            ff, mf, fm, mm = vals
            alternative = np.array([[mm, mf - mm], [fm - mm, ff - mf - fm + mm]])
            b = a[f"blocks_{label}"]
            grouped = np.array(
                [[b[0, 0], b[0, 1:].sum()], [b[1:, 0].sum(), b[1:, 1:].sum()]]
            )
            errors.extend(abs(alternative - obs["cells"]).ravel())
            errors.extend(abs(grouped - alternative).ravel())
            errors.extend(
                abs(
                    np.quantile(a[f"bootstrap_{label}"], [0.025, 0.975], axis=0)
                    - obs["bootstrap_CI"]
                ).ravel()
            )
    if max(errors) > 1e-10:
        raise ValueError("alternative decomposition check failed")
    print(
        {
            "status": "READONLY_ALTERNATE_CHECK_PASS",
            "checks": len(errors),
            "maximum_error": max(errors),
        }
    )


def main():
    """Expose sealed execution and repeatable no-write verification."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "run", "verify"])
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    {"prepare": prepare, "run": run, "verify": verify}[a.command](a.out)


if __name__ == "__main__":
    main()
