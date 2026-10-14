#!/usr/bin/env python3
"""Recover only HM127 mutable-status writes; preserve sealed scientific code."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import readback_grain_response_observability_hm127 as A
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_response_observability_hm127 as P

CODE = (
    "scripts/recover_grain_response_observability_hm127.py",
    "tests/test_grain_response_observability_recovery.py",
    "docs/GRAIN_HM127_STATUS_RECOVERY.md",
)


class StatusWriter:
    """Delegate all operations except the two explicitly mutable state files."""

    def __init__(self, delegate, out):
        self.delegate = delegate
        self.allowed = {
            (out / "status.json").resolve(),
            (out / "01/capture_status.json").resolve(),
        }

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def write(self, path, value):
        """Atomically update state; scientific artifacts retain exclusive create."""
        path = Path(path).resolve()
        if path not in self.allowed:
            return self.delegate.write(path, value)
        payload = json.dumps(value, indent=2, allow_nan=False)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".hm127_status_", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def mutable_status(out):
    """Scope the compatibility adapter to this packet's run and readback."""
    previous_p, previous_a = P.H, A.H
    P.H = A.H = StatusWriter(H, out)
    try:
        yield
    finally:
        P.H, A.H = previous_p, previous_a


def recover(out):
    """Authenticate completed preparation before one finite, budgeted capture."""
    P.check(out)
    state = H.load(out / "status.json")
    receipt = H.load(out / "01/capture_status.json")
    if (
        state != {"state": "PREPARING", "cpu_seconds": 0.0}
        or receipt["state"] != "PREPARED"
        or receipt["attempts"] != 0
        or receipt["completed"] != 0
        or receipt["pending"] is not None
        or receipt["model_seconds"] != 0
        or receipt["constructed_focus_states"] != 0
        or receipt["current_seconds"] != 0
        or any(p for p in (out / "01/units").iterdir() if not p.name.startswith("._"))
    ):
        raise ValueError(
            "Only the confirmed zero-call preparation defect is recoverable"
        )
    rows = H.load(out / "00/fresh_rows.json")
    if len(rows) != H.load(P.SPEC)["max_new_history_calls"]:
        raise ValueError("Preparation is incomplete")
    end = max(
        p.stat().st_mtime
        for p in (out / "00").iterdir()
        if p.is_file() and not p.name.startswith("._")
    )
    end = max(end, (out / "01/capture_status.json").stat().st_mtime)
    charged = max(0.0, end - (out / "status.json").stat().st_mtime) + 1.0
    if charged >= H.load(P.SPEC)["max_cpu_seconds"]:
        raise RuntimeError("Original budget cannot be renewed")
    recovery = out / "status_recovery_r1"
    recovery.mkdir()
    H.write(recovery / "initial_status.json", state)
    H.write(recovery / "initial_capture_status.json", receipt)
    H.write(
        recovery / "failure.json",
        {
            "failed_phase": "prepare.finally.status_write",
            "terminal_session_id": 26671,
            "terminal_exit_code": 1,
            "error": "FileExistsError: status.json opened with mode x in H.write",
            "prepared_inputs_authenticated": True,
            "model_attempts_before_recovery": 0,
            "charged_prepare_cpu_upper_seconds": charged,
            "accounting": "file-mtime preparation span plus one second; not exact CPU",
            "original_budget_not_reset": True,
            "scientific_code_and_sources_unchanged": True,
        },
    )
    H.write(
        recovery / "adapter_seal.json",
        {
            "code": {name: H.sha(ROOT / name) for name in CODE},
            "records": H.seals(out, recovery),
        },
    )
    StatusWriter(H, out).write(
        out / "status.json",
        {
            "state": "PREPARED",
            "cpu_seconds": charged,
            "last_phase": "prepare_status_recovery",
            "preparation_bookkeeping_failure_preserved": True,
            "budget_not_reset": True,
        },
    )
    print("HM127 zero-call status recovery; charged preparation upper", charged)


def run(out, phase):
    """Use sealed original science with independently sealed bookkeeping repair."""
    seal = H.load(out / "status_recovery_r1/adapter_seal.json")
    H.verify_hashes(ROOT, seal["code"])
    H.verify_hashes(out, seal["records"])
    P.check(out)
    with mutable_status(out):
        {"collect": P.collect, "analyze": P.analyze, "readback": A.main}[phase](out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("recover", "collect", "analyze", "readback"))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/response_observability_r1",
    )
    args = parser.parse_args()
    if args.phase == "recover":
        recover(args.out)
    else:
        run(args.out, args.phase)
