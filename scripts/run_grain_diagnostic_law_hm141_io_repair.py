#!/usr/bin/env python3
"""Repair mutable HM141 status IO, without editing its sealed primary program."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_diagnostic_law_hm141 as P

H = P.H
REPAIR_FILES = (
    "scripts/run_grain_diagnostic_law_hm141_io_repair.py",
    "tests/test_grain_diagnostic_law_status_io.py",
)


class MutableStatusIO:
    """Replace only the operational status; keep scientific outputs exclusive."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def write(self, path, value):
        """Atomically save status without weakening immutable artifact writes."""
        path = Path(path)
        if path.name != "status.json":
            return self.base.write(path, value)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, prefix=".status-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def collect(out):
    """Recover only the verified zero-forward IO failure, retaining its budget."""
    manifest = P.check(out)
    status_path = out / "status.json"
    status = H.load(status_path)
    if (
        status["state"] != "PREPARED"
        or status["attempts"] != 0
        or status["completed"] != 0
        or status["pending"] is not None
        or list((out / "01/units").glob("b*"))
    ):
        raise ValueError("This repair cannot replay any attempted scientific forward")
    # The failed loader's duration was not persisted. Charge ALL wall time since
    # preparation as a conservative upper bound; never renew the original cap.
    charged = math.ceil(time.time() - status_path.stat().st_mtime)
    if not 0 <= charged < manifest["spec"]["max_model_seconds"]:
        raise ValueError("No original budget remains under the conservative charge")
    H.write(
        out / "status_io_repair.json",
        {
            "error": "FileExistsError in exclusive H.write(status.json) after model load; no forward entered",
            "original_status": status,
            "conservative_failed_load_wall_seconds": charged,
            "original_budget_seconds": manifest["spec"]["max_model_seconds"],
            "repair_code": {p: H.sha(ROOT / p) for p in REPAIR_FILES},
            "scope": "Operational status IO only; primary code, texts, hypotheses and intervention support unchanged",
        },
    )
    P.H = MutableStatusIO(H)
    status["model_seconds"] = float(charged)
    P.H.write(status_path, status)
    P.collect(out)


def main():
    """Keep repair execution explicit, bounded and independently authenticated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "analyze", "verify"))
    parser.add_argument("--out", type=Path, default=ROOT / H.load(P.SPEC)["out"])
    args = parser.parse_args()
    out = args.out.resolve()
    if args.phase == "collect":
        collect(out)
    else:
        H.verify_hashes(ROOT, H.load(out / "status_io_repair.json")["repair_code"])
        getattr(P, args.phase)(out)


if __name__ == "__main__":
    main()
