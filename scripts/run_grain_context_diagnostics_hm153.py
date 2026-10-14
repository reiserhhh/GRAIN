#!/usr/bin/env python3
"""Test context diagnostic shape using sealed states; zero decoder forwards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_context_diagnostics as M

PARENT = ROOT / "results/grain_math_response_v1/conditional_likelihood_paths_r1"
DEFAULT = ROOT / "results/grain_math_response_v1/context_diagnostic_shape_r1"
MODEL = (
    ROOT
    / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
CODE = (
    "grain_research/construction/math_context_diagnostics.py",
    "scripts/run_grain_context_diagnostics_hm153.py",
    "scripts/verify_grain_context_diagnostics_hm153.py",
    "tests/test_grain_context_diagnostics.py",
    "docs/GRAIN_CONTEXT_DIAGNOSTIC_SHAPE_PLAN_HM153.md",
)


def read(path):
    """Read a numerical/public synthetic artifact only."""
    return json.loads(path.read_text())


def write(path, value):
    """Write a new receipt with strict JSON finite/null semantics."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    """Authenticate file bytes without changing prior scientific artifacts."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table(path, rows):
    """Save complete aligned numeric rows, without outcome-based filtering."""
    if not rows:
        raise ValueError("Empty scientific output")
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def authenticate_parent():
    """Require HM152 complete support and all its immutable source/capture seals."""
    m = read(PARENT / "manifest.json")
    for section in ("code", "source"):
        for name, checksum in m[section].items():
            if digest(ROOT / name) != checksum:
                raise ValueError("Parent frozen dependency changed: " + name)
    for name, checksum in {
        **read(PARENT / "capture_seal.json"),
        **read(PARENT / "output_seal.json"),
    }.items():
        if digest(PARENT / name) != checksum:
            raise ValueError("Parent observation changed: " + name)
    if digest(PARENT / "cases.json") != m["cases_sha256"]:
        raise ValueError("Parent cases changed")
    cases = read(PARENT / "cases.json")
    if len(cases) != 972 or len({c["text_id"] for c in cases}) != 702:
        raise ValueError("All original972cases/702articles required")
    if any(c["case_id"] != i or len(c["tokens"]) != 32 for i, c in enumerate(cases)):
        raise ValueError("Missing, reordered or short observation;no imputation")
    return m, cases


def prepare(out):
    """Seal code, finite analysis and same-text pair addresses before projecting."""
    parent, cases = authenticate_parent()
    by_text = {}
    for c in cases:
        if c["pole"] == 0:
            by_text.setdefault((c["text_id"], c["family"]), {})[c["axis"]] = c
    pairs = []
    for _, axes in sorted(by_text.items()):
        e, n = axes["extraversion"], axes["neuroticism"]
        if e["tokens"] != n["tokens"]:
            raise ValueError("Not the identical actual text")
        pairs.append({"pair_id": len(pairs), "E": e["case_id"], "N": n["case_id"]})
    if len(pairs) != 162:
        raise ValueError("All54neutral texts x3families required")
    out.mkdir(parents=True, exist_ok=False)
    (out / "accounts").mkdir()
    (out / "cross_accounts").mkdir()
    write(out / "pairs.json", pairs)
    write(
        out / "manifest.json",
        {
            "packet": "HM153",
            "code": {p: digest(ROOT / p) for p in CODE},
            "parent": {
                str((PARENT / name).relative_to(ROOT)): digest(PARENT / name)
                for name in (
                    "manifest.json",
                    "cases.json",
                    "capture_seal.json",
                    "output_seal.json",
                    "alternate_audit.json",
                )
            },
            "pairs_sha256": digest(out / "pairs.json"),
            "model_weights_sha256": parent["model_weights_sha256"],
            "analysis_seconds_cap": 600,
            "audit_seconds_cap": 120,
            "case_count": 972,
            "second_token_units": 15552,
            "same_text_target_pairs": 162,
            "projected_rows": 38880,
            "decoder_forwards": 0,
            "absolute_tolerance": 1e-4,
            "relative_tolerance": 1e-5,
            "original_scope_complete": False,
        },
    )
    write(out / "status.json", {"status": "PREPARED", "cases_done": 0, "pairs_done": 0})
    print(
        {"prepared_cases": 972, "prepared_pairs": 162, "decoder_forwards": 0},
        flush=True,
    )


def check(out):
    """Verify sealed analysis code and immutable upstream interface."""
    manifest = read(out / "manifest.json")
    for section in ("code", "parent"):
        for name, checksum in manifest[section].items():
            if digest(ROOT / name) != checksum:
                raise ValueError("Frozen dependency changed: " + name)
    if digest(out / "pairs.json") != manifest["pairs_sha256"]:
        raise ValueError("Same-text pair manifest changed")
    return manifest


def weights():
    """Read exact BF16 archive values as FP32; do not instantiate a decoder."""
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    with safe_open(
        MODEL / "model.safetensors", framework="pt", device="cpu"
    ) as archive:
        w = archive.get_tensor("model.embed_tokens.weight").to(torch.float32)
    if tuple(w.shape) != (151936, 1536):
        raise ValueError("Unexpected pinned tied output head")
    return w


def distribution(values):
    """Report descriptive finite-support summaries, not participant inference."""
    a = np.asarray(values, float)
    finite = a[np.isfinite(a)]
    if not len(finite):
        return {"finite": 0, "undefined": len(a), "mean": None, "median": None}
    return {
        "finite": len(finite),
        "undefined": int(len(a) - len(finite)),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def analyze(out):
    """Project all original observations once within a nonrenewing finite budget."""
    import torch

    start = time.monotonic()
    manifest = check(out)
    if read(out / "status.json")["status"] != "PREPARED":
        raise ValueError("No automatic repeat or overwrite of a scientific execution")
    _, cases = authenticate_parent()
    if digest(MODEL / "model.safetensors") != manifest["model_weights_sha256"]:
        raise ValueError("Pinned model weights changed")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Declared MPS projection backend unavailable;no automatic fallback"
        )
    w = weights().to("mps")
    token_rows, case_rows, cross_rows = [], [], []
    counters = {
        "cases_done": 0,
        "pairs_done": 0,
        "projection_batches": 0,
        "projected_rows": 0,
    }

    def checkpoint(status):
        write(
            out / "status.json",
            {
                "status": status,
                **counters,
                "seconds": time.monotonic() - start,
                "decoder_forwards": 0,
            },
        )

    def project(rows):
        if time.monotonic() - start > manifest["analysis_seconds_cap"]:
            raise TimeoutError("Sealed600second budget exhausted;no automatic renewal")
        with torch.inference_mode():
            result = (
                (torch.as_tensor(rows, dtype=torch.float32, device="mps") @ w.T)
                .cpu()
                .numpy()
                .astype(float)
            )
        counters["projection_batches"] += 1
        counters["projected_rows"] += len(rows)
        return result

    def capture(i):
        with np.load(PARENT / "captures" / f"{i:04d}.npz", allow_pickle=False) as a:
            values = {k: a[k] for k in a.files}
        if not np.array_equal(values["lengths"], [32, 32, 16, 16]):
            raise ValueError("Missing aligned native units")
        if np.any(values["bias"] != 0):
            raise ValueError(
                "This pinned tied-head law requires the verified zero bias"
            )
        return values

    checkpoint("RUNNING_OUTPUT_PROJECTIONS_ONLY")
    try:
        for c in cases:
            i = c["case_id"]
            raw = capture(i)
            delta, midpoint, normalizer = M.aligned_states(
                raw["states"], raw["partition"]
            )
            fields = project(np.vstack([delta, midpoint]))
            account = M.field_account(
                fields[:16], fields[16:], normalizer, c["tokens"][16:]
            )
            native = raw["chosen"].astype(float)
            native_k = (
                native[0, 16:]
                - native[1, 16:]
                - native[2, :16]
                + native[3, :16]
                - normalizer
            )
            error = account["reconstructed_k"] - native_k
            if not np.allclose(
                account["reconstructed_k"], native_k, atol=1e-4, rtol=1e-5
            ):
                raise ArithmeticError("Actual token context account failed")
            account["native_k"], account["closure_error"] = native_k, error
            np.savez_compressed(out / "accounts" / f"{i:04d}.npz", **account)
            metadata = {
                k: c[k]
                for k in (
                    "case_id",
                    "text_id",
                    "axis",
                    "family",
                    "situation",
                    "style",
                    "seed",
                    "pole",
                )
            }
            for t in range(16):
                token_rows.append(
                    {
                        **metadata,
                        "token_index": t + 16,
                        "native_token_id": c["tokens"][t + 16],
                        **{k: v[t].item() for k, v in account.items()},
                    }
                )
            case_rows.append(
                {
                    **metadata,
                    "shape_sum": float(account["selected_shape"].sum()),
                    "calibration_sum": float(account["calibration"].sum()),
                    "native_k_sum": float(native_k.sum()),
                    "reconstructed_k_sum": float(account["reconstructed_k"].sum()),
                    "weighted_std_mean": float(account["weighted_std"].mean()),
                    "oscillation_mean": float(account["oscillation"].mean()),
                    "max_closure_error": float(abs(error).max()),
                }
            )
            counters["cases_done"] += 1
            if (i + 1) % 81 == 0:
                checkpoint("RUNNING_OUTPUT_PROJECTIONS_ONLY")
                print(
                    {**counters, "seconds": round(time.monotonic() - start, 2)},
                    flush=True,
                )
        for pair in read(out / "pairs.json"):
            e = cases[pair["E"]]
            er, nr = capture(pair["E"]), capture(pair["N"])
            ed, em, _ = M.aligned_states(er["states"], er["partition"])
            nd, nm, _ = M.aligned_states(nr["states"], nr["partition"])
            fields = project(np.vstack([ed, nd, (em + nm) / 2]))
            account = M.cross_fields(fields[:16], fields[16:32], fields[32:])
            np.savez_compressed(
                out / "cross_accounts" / f"{pair['pair_id']:04d}.npz", **account
            )
            for t in range(16):
                values = {
                    k: (v[t].item() if np.isfinite(v[t]) else None)
                    for k, v in account.items()
                }
                cross_rows.append(
                    {
                        **pair,
                        "text_id": e["text_id"],
                        "family": e["family"],
                        "situation": e["situation"],
                        "style": e["style"],
                        "seed": e["seed"],
                        "token_index": t + 16,
                        **values,
                    }
                )
            counters["pairs_done"] += 1
            if counters["pairs_done"] % 54 == 0:
                checkpoint("RUNNING_OUTPUT_PROJECTIONS_ONLY")
                print(
                    {**counters, "seconds": round(time.monotonic() - start, 2)},
                    flush=True,
                )
        groups = {}
        for r in case_rows:
            if r["pole"]:
                key = tuple(r[k] for k in ("axis", "family", "situation", "style"))
                groups.setdefault(key, {1: [], -1: []})[r["pole"]].append(r)
        conditions = []
        for key, poles in sorted(groups.items()):
            if len(poles[1]) != 3 or len(poles[-1]) != 3:
                raise ValueError("All generation replicates must be retained")
            result = dict(
                zip(("axis", "family", "situation", "style"), key, strict=True)
            )
            for metric in ("shape_sum", "calibration_sum", "native_k_sum"):
                result["delta_" + metric] = float(
                    np.mean([r[metric] for r in poles[1]])
                    - np.mean([r[metric] for r in poles[-1]])
                )
            conditions.append(result)
        if (
            len(conditions) != 108
            or len(token_rows) != 15552
            or len(cross_rows) != 2592
        ):
            raise ValueError("Finite support incomplete")
        table(out / "token_accounts.csv", token_rows)
        table(out / "case_accounts.csv", case_rows)
        table(out / "condition_accounts.csv", conditions)
        table(out / "cross_target_accounts.csv", cross_rows)
        summary = {
            "packet": "HM153",
            "status": "FINITE_ANALYSIS_COMPLETE_NOT_GLOBAL",
            **counters,
            "token_units": len(token_rows),
            "condition_cells": 108,
            "cross_target_units": len(cross_rows),
            "weighted_std": distribution([r["weighted_std"] for r in token_rows]),
            "oscillation": distribution([r["oscillation"] for r in token_rows]),
            "shape_sum": distribution([r["shape_sum"] for r in case_rows]),
            "calibration_sum": distribution([r["calibration_sum"] for r in case_rows]),
            "absolute_shape_sum_mean": float(
                np.mean([abs(r["shape_sum"]) for r in case_rows])
            ),
            "absolute_calibration_sum_mean": float(
                np.mean([abs(r["calibration_sum"]) for r in case_rows])
            ),
            "reference_zero_probabilities": sum(
                r["reference_zero_count"] for r in token_rows
            ),
            "max_native_closure_error": max(r["max_closure_error"] for r in case_rows),
            "cross_target_cosine": distribution(
                [np.nan if r["cosine"] is None else r["cosine"] for r in cross_rows]
            ),
            "negative_cross_target_units": sum(
                r["cosine"] is not None and r["cosine"] < 0 for r in cross_rows
            ),
            "seconds_including_load_IO": time.monotonic() - start,
            "decoder_forwards": 0,
            "new_generation_API_training_private_calls": 0,
            "scope": "Known synthetic persona-conditioned lexical diagnostic field,NOT human trait or author correlation;baseline q isdeclared geometric reference",
            "original_scope_complete": False,
        }
        write(out / "summary.json", summary)
        files = [
            "token_accounts.csv",
            "case_accounts.csv",
            "condition_accounts.csv",
            "cross_target_accounts.csv",
            "summary.json",
        ]
        files += [
            str(p.relative_to(out))
            for directory in ("accounts", "cross_accounts")
            for p in sorted((out / directory).glob("[0-9]*.npz"))
        ]
        write(out / "output_seal.json", {p: digest(out / p) for p in files})
        if time.monotonic() - start > manifest["analysis_seconds_cap"]:
            raise TimeoutError(
                "Sealed analysis budget exceeded during output;preserved"
            )
        checkpoint("COMPLETE_ANALYSIS")
        print(summary, flush=True)
    except Exception as exc:
        checkpoint("FAILED_PRESERVED")
        write(
            out / "failure.json",
            {"type": type(exc).__name__, "error": str(exc), **counters},
        )
        raise


def main():
    """Run either prospective sealing or one finite projection execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze}[args.command](args.out)


if __name__ == "__main__":
    main()
