#!/usr/bin/env python3
"""Preserve the failed HM123 primary and run its identical bounded schedules."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_stage_shape_adapter as A
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_stage_mass_action_hm123 as W

CODE = (
    "grain_research/construction/math_stage_shape_adapter.py",
    "scripts/run_grain_stage_mass_action_shape_adapter_hm123.py",
    "tests/test_grain_stage_shape_adapter.py",
    "docs/GRAIN_STAGE_SHAPE_ADAPTER_HM123.md",
)


def prepare(out, original):
    """Freeze the adapter and debit time before reading any staged response."""
    if out.exists():
        raise FileExistsError("Preserve all started and failed runs")
    m = W.check(original)
    state = H.load(original / "01/status.json")
    remaining = A.remaining_budget(
        state,
        list((original / "01/units").glob("units_*")),
        m["spec"]["max_numeric_seconds"],
    )
    spec = {**m["spec"], "max_numeric_seconds": remaining}
    for folder in ("00", "01/units", "02"):
        (out / folder).mkdir(parents=True)
    for name in ("schedules.npz", "pairs.json"):
        shutil.copyfile(original / "00" / name, out / "00" / name)
    sources = m["source"] | {
        str((original / name).relative_to(ROOT)): H.sha(original / name)
        for name in ("00/manifest.json", "01/status.json", *m["inputs"])
    }
    H.write(
        out / "00/manifest.json",
        {
            **m,
            "spec": spec,
            "code": m["code"] | {n: H.sha(ROOT / n) for n in CODE},
            "source": sources,
            "inputs": H.seals(out, out / "00"),
            "shape_adapter": {
                "original": str(original.relative_to(ROOT)),
                "original_status": state,
                "original_limit_seconds": m["spec"]["max_numeric_seconds"],
                "remaining_seconds": remaining,
                "budget_reset": False,
                "original_recurrences": 0,
                "cached_history_layers": 17,
                "completed_blocks": 17,
                "final_projection_weight_group": 17,
                "unused_placeholder_is_not_a_history_capture": True,
                "same_schedule_input_bytes": True,
            },
        },
    )
    write_json(
        out / "01/status.json",
        {
            "state": "PREPARED",
            "completed_units": 0,
            "pending": None,
            "seconds": 0.0,
            "attempts": 0,
        },
    )
    print("HM123 SHAPE ADAPTER PREPARED", {"remaining_seconds": remaining}, flush=True)


def run(out):
    """Use the sealed scientific runner with only its admission guard adapted."""
    original = W.T.construct
    W.T.construct = A.construct
    try:
        W.run(out)
    finally:
        W.T.construct = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        if args.original is None:
            parser.error("--original is required for prepare")
        prepare(out, args.original.resolve())
    elif args.command == "run":
        run(out)
    else:
        W.analyze(out)
