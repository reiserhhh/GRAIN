#!/usr/bin/env python3
"""Separate post-outcome lower bound for fixed SOURCE-zero behavior observations."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R

PARENT = ROOT / "results/grain_math_response_v1/event_behavior_correspondence_r1"
DEFAULT = PARENT / "source_fiber_bound_r1"
SCIENCE = (
    "docs/GRAIN_SOURCE_FIBER_BOUND_FOLLOWUP_HM262.md",
    "scripts/bound_grain_event_behavior_hm262.py",
    "tests/test_grain_source_fiber_bound.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def fiber_bound(source_delta, behavior_delta):
    """Bound any source-only decoder's paired squared error with exact-zero fibers."""
    x, y = (np.asarray(v, float) for v in (source_delta, behavior_delta))
    if x.ndim != 2 or y.shape != x.shape or x.shape[1] != 5 or not len(x):
        raise ValueError("Paired rows and five fields required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Missing observations cannot imply a zero fiber")
    mask = np.all(x == 0, axis=1)
    return y[mask].T @ y[mask] / (4 * len(x)), mask


def prepare(out):
    """Freeze a separate derivative; never alter or recalculate primary science."""
    if out.exists():
        raise FileExistsError("No replacement/retry of the post-outcome bound")
    pm = R.check(PARENT, ("analyzed_seal.json", "verification_seal.json"))
    data = R.read(PARENT / "source_actions.json")
    behavior = R.read(PARENT / "behavior.json")
    lifts = R.read(PARENT / "finite_lifts.json")
    x = np.array(data["changes"]["actual"]["full"])
    y = np.array(behavior["sum"]["balanced_key"])
    if x.shape != (48, 5) or y.shape != x.shape:
        raise ValueError("Registered48 uses cannot be expanded or filtered")
    mask = np.all(x == 0, axis=1)
    counts = np.array(lifts["actual"]["full"]["delta_matching_pairs"])
    if not np.all(counts[mask] == 8):
        raise ValueError("Literal zero equality is not an identified source-cell fiber")
    out.mkdir(parents=True, exist_ok=False)
    inputs = dict(pm["inputs"])
    inputs.update(
        {
            str((PARENT / n).relative_to(ROOT)): R.digest(PARENT / n)
            for n in (
                "manifest.json",
                "prepared_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
                "source_actions.json",
                "behavior.json",
                "finite_lifts.json",
            )
        }
    )
    R.write(
        out / "manifest.json",
        {
            "config": {"post_outcome": True, "max_cpu_seconds": 10},
            "code": {**pm["code"], **{p: R.digest(ROOT / p) for p in SCIENCE}},
            "inputs": inputs,
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "input.json", {"source_delta": x.tolist(), "behavior_delta": y.tolist()}
    )
    R.seal(out, "prepared_seal.json", ("input.json",))
    print("PREPARED_SOURCE_FIBER_BOUND", flush=True)


def run(out):
    """Calculate the operator floor and alternate literal checks with fixed weights."""
    R.check(out)
    if (out / "summary.json").exists():
        raise FileExistsError("No derivative rerun")
    began = time.monotonic()
    data = R.read(out / "input.json")
    x, y = (np.array(data[k]) for k in ("source_delta", "behavior_delta"))
    matrix, mask = fiber_bound(x, y)
    literal = np.array(
        [
            [
                sum(float(row[p]) * float(row[q]) for row in y[mask]) / (4 * len(x))
                for q in range(5)
            ]
            for p in range(5)
        ]
    )
    error = float(np.max(abs(matrix - literal)))
    if error > 1e-10:
        raise ArithmeticError("Finite operator-bound replay failed")
    result = {
        "post_outcome_same_observations": True,
        "uses": len(x),
        "source_zero_uses": int(mask.sum()),
        "operator_floor": matrix.tolist(),
        "trait_coordinate_squared_loss_floor": np.diag(matrix).tolist(),
        "literal_error": error,
        "literal_checks": 25,
        "scientific_CPU_seconds": time.monotonic() - began,
        "new_model_API_private_training_calls": 0,
        "not_psychological_Bayes_error_or_accuracy": True,
        "same_agent_not_blind": True,
        "original_scope_complete": False,
    }
    if result["scientific_CPU_seconds"] > 10:
        raise RuntimeError("Scientific cap exceeded; do not renew")
    R.write(out / "summary.json", result)
    R.seal(out, "analyzed_seal.json", ("summary.json",))
    print(json.dumps(result), flush=True)


def main():
    """Expose only finite numeric bound stages, never a model/representation search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, ("analyzed_seal.json",))
        print("ALL_SEALS_PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
