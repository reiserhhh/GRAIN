#!/usr/bin/env python3
"""Source compilation then complete exploratory typed-current transport."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_typed_cycle as M
from grain_research.construction.math_cycle_capacity import cycles
from scripts.run_grain_current_transport_hm98 import check as check_source
from scripts.run_grain_observer_parity_hm96 import load, seal, sha, table, verify, write

SPEC = ROOT / "configs/grain_typed_cycle_hm99.json"
CODE = (
    "configs/grain_typed_cycle_hm99.json",
    "docs/GRAIN_TYPED_CYCLE_PLAN_HM99.md",
    "grain_research/construction/math_typed_cycle.py",
    "scripts/run_grain_typed_cycle_hm99.py",
    "scripts/verify_grain_typed_cycle_hm99.py",
    "tests/test_grain_math_typed_cycle.py",
    "grain_research/construction/math_cycle_capacity.py",
    "scripts/run_grain_current_transport_hm98.py",
    "scripts/run_grain_observer_parity_hm96.py",
)


def settings():
    """Load a single fixed experiment, with no search space."""
    spec = load(SPEC)
    return spec, ROOT / spec["source"], ROOT / spec["output"]


def check(out, include_application=False):
    """Read back declared source assets and frozen code only."""
    m = load(out / "00/manifest.json")
    assets = (
        m["source"]
        if include_application
        else {
            k: v
            for k, v in m["source"].items()
            if "/05/" not in k and not k.endswith("/05_seal.json")
        }
    )
    verify(ROOT, m["code"] | assets)
    return m


def stage(out, number):
    """Verify explicit stage outputs, excluding AppleDouble metadata."""
    h = load(out / f"{number:02d}_seal.json")
    verify(out, {k: v for k, v in h.items() if not Path(k).name.startswith("._")})


def prepare(spec, source, out):
    """Register the follow-up before computing any typed application metric."""
    if out.exists():
        raise FileExistsError("Preserve all prior attempts")
    check_source(source)
    v = load(source / "verification.json")
    if v["status"] != "VERIFIED_FINITE_HM98_NOT_GLOBAL":
        raise ValueError("Verified actual predecessor required")
    assets = (
        "00/manifest.json",
        "03/compiled_bundle.npz",
        "03_seal.json",
        "05/measured_transport.npz",
        "05_seal.json",
        "verification.json",
    )
    (out / "00").mkdir(parents=True)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": {
                str((source / n).relative_to(ROOT)): sha(source / n) for n in assets
            },
            "source_aggregate_already_observed": True,
            "new_neural_calls": 0,
            "original_scope_complete": False,
        },
    )
    print("PREPARED HM99", flush=True)


def compile_source(spec, source, out):
    """Construct every source sector without reading the application kernel."""
    check(out)
    dest = out / "01"
    dest.mkdir()
    bases = M.sectors(spec["categories"], spec["levels"])
    d, plus, minus = M.positive_laws(bases)
    with np.load(source / "03/compiled_bundle.npz", allow_pickle=False) as data:
        f = data["kernel"].copy()
    if f.shape != (2, 18, 18, 18):
        raise ValueError("Complete inherited source support required")
    arrays = {"source_kernel": f, "currents": d, "plus": plus, "minus": minus}
    for name, a in bases.items():
        arrays[f"basis_{name}"] = a
    for c in range(2):
        coefficients, parts = M.project(f[c], bases)
        for name in bases:
            arrays[f"coefficient_{c}_{name}"] = coefficients[name]
            arrays[f"part_{c}_{name}"] = parts[name]
    np.savez_compressed(dest / "compiled.npz", **arrays)
    write(
        dest / "summary.json",
        {
            "dimensions": {k: v.shape[2] for k, v in bases.items()},
            "source_only": True,
            "application_kernel_read": False,
            "lookup_or_fit": False,
        },
    )
    write(out / "01_seal.json", seal(out, dest))
    print("COMPILED source-only", flush=True)


def cold(out):
    """Score new designed author laws with compiled source sectors only."""
    check(out)
    stage(out, 1)
    dest = out / "02"
    dest.mkdir()
    with np.load(out / "01/compiled.npz", allow_pickle=False) as data:
        difference = data["plus"] - data["minus"]
        arrays = {
            f"prediction_{c}_{name}": np.einsum(
                "aij,ijq->aq", difference, data[f"part_{c}_{name}"]
            )
            for c in range(2)
            for name in M.NAMES
        }
    np.savez_compressed(dest / "predictions.npz", **arrays)
    write(
        dest / "audit.json",
        {
            "data_input": "01/compiled.npz",
            "application_native_kernel_read": False,
            "supplied_correspondence": "designedcategoryxlevel identity;not inferred semantic equivalence",
            "not_OS_sandbox": True,
        },
    )
    write(out / "02_seal.json", seal(out, dest))
    print("COLD source author contrasts", flush=True)


def evaluate(spec, source, out):
    """Inspect every sector, frame and target, never select a winning subset."""
    start = time.monotonic()
    check(out, include_application=True)
    stage(out, 1)
    stage(out, 2)
    dest = out / "03"
    dest.mkdir()
    with np.load(out / "01/compiled.npz", allow_pickle=False) as data:
        compiled = {k: data[k].copy() for k in data.files}
    with np.load(source / "05/measured_transport.npz", allow_pickle=False) as data:
        app = data["native_kernel"].copy()
    bases = {k: compiled[f"basis_{k}"] for k in M.NAMES}
    arrays = {"application_kernel": app}
    metrics, rows, errors = [], [], []
    difference = compiled["plus"] - compiled["minus"]
    for a in range(2):
        coeff, part = M.project(app[a], bases)
        full_cycle = cycles(app[a])[2]
        errors.append(float(np.max(np.abs(sum(part.values()) - full_cycle))))
        for name in M.NAMES:
            arrays[f"application_coefficient_{a}_{name}"] = coeff[name]
            arrays[f"application_part_{a}_{name}"] = part[name]
        for s in range(2):
            for name in M.NAMES:
                comparison = M.compare(compiled[f"coefficient_{s}_{name}"], coeff[name])
                for q in range(18):
                    metrics.append(
                        {
                            "source_frame": spec["contexts"][s],
                            "application_frame": spec["contexts"][a],
                            "sector": name,
                            "dimension": bases[name].shape[2],
                            "field": spec["fields"][q // 6],
                            "target": spec["targets"][q % 6],
                            **{k: float(v[q]) for k, v in comparison.items()},
                        }
                    )
                actual = np.einsum("aij,ijq->aq", difference, part[name])
                predicted = np.einsum(
                    "aij,ijq->aq", difference, compiled[f"part_{s}_{name}"]
                )
                arrays[f"actual_{a}_{s}_{name}"] = actual
                arrays[f"predicted_{a}_{s}_{name}"] = predicted
                for q in range(18):
                    for i in range(len(difference)):
                        rows.append(
                            {
                                "source_context": s,
                                "application_context": a,
                                "sector": name,
                                "author_pair": i,
                                "channel": q,
                                "actual": float(actual[i, q]),
                                "predicted": float(predicted[i, q]),
                            }
                        )
    np.savez_compressed(dest / "accounts.npz", **arrays)
    table(dest / "sector_metrics.csv", metrics)
    table(dest / "author_contrasts.csv", rows)
    summary = {
        "status": "COMPUTED_AWAITING_ALTERNATE_VERIFICATION",
        "sector_rows": len(metrics),
        "author_contrast_rows": len(rows),
        "decomposition_max_error": max(errors),
        "analysis_seconds": time.monotonic() - start,
        "new_neural_calls": 0,
        "no_predictor_fit": True,
        "original_scope_complete": False,
        "psychological_validity_claim": False,
    }
    write(dest / "summary.json", summary)
    write(out / "03_seal.json", seal(out, dest))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compile", "cold", "evaluate"))
    args = parser.parse_args()
    spec, source, out = settings()
    with threadpool_limits(limits=1):
        if args.stage == "cold":
            cold(out)
        else:
            {"prepare": prepare, "compile": compile_source, "evaluate": evaluate}[
                args.stage
            ](spec, source, out)
