#!/usr/bin/env python3
"""One bounded compatibility companion; never restart or overwrite failed HM203."""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_body_summary_innovation as I
from grain_research.construction import math_body_summary_mps_compat as C
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_body_summary_innovation_hm203 as U

H, P, R = U.H, U.P, U.R
ORIGINAL = ROOT / "results/grain_math_response_v1/body_summary_innovation_r1"
DEFAULT = (
    ROOT / "results/grain_math_response_v1/body_summary_innovation_mps_recovery_r1"
)
CODE = tuple(
    dict.fromkeys(
        (
            "grain_research/construction/math_body_summary_mps_compat.py",
            "scripts/run_grain_body_summary_mps_recovery_hm203.py",
            "tests/test_grain_body_summary_mps_compat.py",
            "docs/GRAIN_HM203_MPS_COMPATIBILITY_RECOVERY_PLAN.md",
            *U.CODE,
        )
    )
)


def recovery_counts(original, companion_attempts, companion_completed):
    """Separate reused successes and failed attempts from new companion calls."""
    return {
        "original_attempts": original["attempts"],
        "original_completed": original["completed"],
        "companion_attempts": companion_attempts,
        "companion_completed": companion_completed,
        "combined_attempts": original["attempts"] + companion_attempts,
        "combined_completed": original["completed"] + companion_completed,
        "original_failed_attempts": original["attempts"] - original["completed"],
    }


def prepare(out):
    """Bind the failed receipt and reuse the sole completed capture byte-for-byte."""
    if out.exists():
        raise FileExistsError("No companion overwrite or automatic retry")
    old = R.check(ORIGINAL)
    status = H.read(ORIGINAL / "status.json")
    if (
        status["state"] != "FAILED"
        or status["completed"] != 1
        or status["attempts"] != 2
        or status["pending"] != [0, "self"]
        or "MPS Tensor to float64" not in status["error"]
        or (ORIGINAL / "collector.lock").exists()
    ):
        raise ValueError("Only the observed terminal dtype failure is recoverable")
    for p in ORIGINAL.rglob("*"):
        if (
            p.is_file()
            and not p.name.startswith("._")
            and p.relative_to(ORIGINAL).as_posix().startswith("01/b")
            and p.name not in ("b000_blind.npz", "b000_blind.json")
        ):
            raise ValueError("Unexpected original partial observation")
    inputs = dict(old["inputs"])
    inputs.update(
        {
            p.relative_to(ROOT).as_posix(): H.sha(p)
            for p in ORIGINAL.rglob("*")
            if p.is_file() and not p.name.startswith("._")
        }
    )
    blind = P.arrays(ORIGINAL / "01/b000_blind.npz")
    if blind["values"].dtype != np.float32:
        raise ValueError("Original capture dtype is not float32")
    receipt = H.read(ORIGINAL / "01/b000_blind.json")
    if receipt["sha256"] != H.sha(ORIGINAL / "01/b000_blind.npz"):
        raise ValueError("Original completed capture was modified")
    (out / "00").mkdir(parents=True)
    (out / "01").mkdir()
    for name in (
        "00/units.json",
        "00/observations.npz",
        "01/b000_blind.npz",
        "01/b000_blind.json",
        "01/head.npz",
    ):
        shutil.copyfile(ORIGINAL / name, out / name)
    H.write(
        out / "00/manifest.json",
        {
            **old,
            "code": {k: H.sha(ROOT / k) for k in CODE},
            "inputs": inputs,
            "recovery": {
                "original": ORIGINAL.relative_to(ROOT).as_posix(),
                "original_status": status,
                "reused_batches": 1,
                "new_companion_batches": 385,
                "new_companion_inquiries": 1925,
                "original_total_cap_seconds": old["spec"]["max_model_seconds"],
                "formula_change": False,
                "device_cast_only": True,
            },
        },
    )
    H.seal(
        out,
        "prepared_seal.json",
        (
            "00/units.json",
            "00/observations.npz",
            "00/manifest.json",
            "01/b000_blind.npz",
            "01/b000_blind.json",
            "01/head.npz",
        ),
    )
    H.status_write(
        out,
        {
            "state": "PREPARED",
            "completed": 1,
            "attempts": 0,
            "pending": None,
            "model_seconds": status["model_seconds"],
            **recovery_counts(status, 0, 0),
        },
    )
    print("PREPARED COMPANION:1reused+385new;original3000s cap", flush=True)


def collect(out):
    """Execute only absent registered calls once within the original time cap."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest, status = R.check(out), H.read(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No failed, uncertain or terminal companion restart")
    H.check_hashes(MODEL, manifest["model_files"])
    lock = out / "collector.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    spec = manifest["spec"]
    old_status = manifest["recovery"]["original_status"]
    units = H.read(out / "00/units.json")
    parent = ROOT / spec["parent"]
    try:
        torch.set_num_threads(2)
        start = time.monotonic()
        try:
            observer = DecoderObserver.load(
                str(MODEL),
                device=spec["device"],
                dtype=spec["dtype"],
                max_tokens=spec["max_tokens"],
            )
        finally:
            status["model_seconds"] += time.monotonic() - start
            H.status_write(out, status)
        if len(observer.model.model.layers) != spec["layers"]:
            raise ValueError("Frozen architecture changed")
        head = observer.model.lm_head
        alphabet = manifest["alphabet"]["token_ids"]
        saved = P.arrays(out / "01/head.npz")
        weights = head.weight.detach()[alphabet].float().cpu().numpy()
        bias = (
            np.zeros(10)
            if head.bias is None
            else head.bias.detach()[alphabet].float().cpu().numpy()
        )
        if not np.array_equal(weights, saved["weights"]) or not np.array_equal(
            bias, saved["bias"]
        ):
            raise ValueError("Frozen native head changed")
        H.write(
            out / "runtime.json",
            {
                "pid": os.getpid(),
                "torch": torch.__version__,
                "device": spec["device"],
                "dtype": spec["dtype"],
                "reused_blind_batches": 1,
                "formula_change": False,
            },
        )
        for b, unit in enumerate(units):
            native = P.arrays(parent / f"01/b{b:03d}_native.npz")
            blind_values = (
                P.arrays(out / "01/b000_blind.npz")["values"] if b == 0 else None
            )
            modes = (
                ([] if b == 0 else ["blind"])
                + (["self"] if b < 2 else [])
                + list(U.NEW_MODES)
            )
            for mode in modes:
                if status["model_seconds"] >= spec["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Original cap includes failure; no extension")
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=[b, mode]
                )
                status.update(
                    recovery_counts(
                        old_status, status["attempts"], status["completed"] - 1
                    )
                )
                H.status_write(out, status)
                start = time.monotonic()
                try:
                    if mode == "blind":
                        result = I.blind_batch(observer, unit)
                        blind_values = result[5]["values"]
                    else:
                        result = C.reader_batch(
                            observer,
                            unit,
                            native,
                            blind_values,
                            mode,
                            spec["donor_permutation"] if mode == "foreign" else None,
                        )
                    logits, state, masses, body_receipts = result[:4]
                    payload = {
                        "candidate_logits": logits[:, alphabet],
                        "normalizer": logsumexp(logits, axis=1),
                        "states": state,
                        "body_masses": masses,
                    }
                    if mode == "blind":
                        payload["values"] = blind_values
                    path = out / f"01/b{b:03d}_{mode}.npz"
                    H.save_npz(path, **payload)
                    H.write(
                        path.with_suffix(".json"),
                        {
                            "unit": b,
                            "mode": mode,
                            "address": [unit[k] for k in ("author", "view", "arm")],
                            "sha256": H.sha(path),
                            "blocks": result[6],
                            "body": body_receipts,
                            "blind": result[7] if mode == "blind" else [],
                            "compatibility": None if mode == "blind" else result[7],
                        },
                    )
                finally:
                    status["model_seconds"] += time.monotonic() - start
                    H.status_write(out, status)
                status.update(completed=status["completed"] + 1, pending=None)
                status.update(
                    recovery_counts(
                        old_status, status["attempts"], status["completed"] - 1
                    )
                )
                H.status_write(out, status)
            if (b + 1) % 8 == 0:
                print(
                    f"RECOVERED {b + 1}/96units,{status['completed']}/386registered;{status['model_seconds']:.3f}s total",
                    flush=True,
                )
        if status["companion_completed"] != 385 or status["completed"] != 386:
            raise ValueError("Full unchanged registered support required")
        H.seal(
            out,
            "collected_seal.json",
            [
                p.relative_to(out).as_posix()
                for p in sorted((out / "01").iterdir())
                if not p.name.startswith("._")
            ]
            + ["runtime.json"],
        )
        status.update(state="COLLECTED", pending=None)
        H.status_write(out, status)
    except Exception as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = repr(exc)
        H.status_write(out, status)
        raise
    finally:
        lock.unlink()


def analyze(out):
    """Use the sealed original analysis and separately disclose recovery counts."""
    U.analyze(out)
    manifest, status = R.check(out), H.read(out / "status.json")
    rows = [H.read(p) for p in sorted((out / "01").glob("b*.json"))]
    casts = [r["compatibility"] for r in rows if r["mode"] != "blind"]
    if len(casts) != 290 or any(r["installed_dtype"] != "float32" for r in casts):
        raise ValueError("Every reader must use the declared native dtype")
    receipt = {
        **recovery_counts(
            manifest["recovery"]["original_status"],
            status["attempts"],
            status["completed"] - 1,
        ),
        "new_companion_inquiries": status["attempts"] * 5,
        "registered_successful_inquiries": status["completed"] * 5,
        "combined_model_seconds": status["model_seconds"],
        "original_cap_seconds": manifest["spec"]["max_model_seconds"],
        "reader_cast_max_abs": max(c["cast_max_abs"] for c in casts),
        "original_remains_FAILED": H.read(ORIGINAL / "status.json")["state"]
        == "FAILED",
        "formula_change": False,
        "precision_optimization": False,
    }
    H.write(out / "recovery_accounting.json", receipt)
    H.seal(out, "recovery_accounting_seal.json", ("recovery_accounting.json",))
    print(receipt, flush=True)


def verify(out):
    """Audit all original accounts and the byte-identical reused observation."""
    U.verify(out)
    manifest = R.check(out)
    H.check_hashes(out, H.read(out / "recovery_accounting_seal.json"))
    for name in ("b000_blind.npz", "b000_blind.json", "head.npz"):
        if H.sha(out / "01" / name) != H.sha(ORIGINAL / "01" / name):
            raise ValueError("Reused observation bytes differ")
    if H.read(ORIGINAL / "status.json") != manifest["recovery"]["original_status"]:
        raise ValueError("Original failure was overwritten")
    print("RECOVERY AUDIT:original failure/reused bytes/formula preserved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
        args.stage
    ](args.output)
