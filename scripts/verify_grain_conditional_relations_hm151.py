#!/usr/bin/env python3
"""Read-only QR and explicit-coefficient audit of sealed HM151 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/grain_math_response_v1/conditional_relation_circuits_r1"


def load(path):
    """Read a public experiment JSON artifact."""
    return json.loads(path.read_text())


def digest(path):
    """Authenticate immutable bytes without changing scientific receipts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(out):
    """Recompute every relation account without importing the tested module."""
    manifest = load(out / "manifest.json")
    checked_hashes = 0
    for root, section in ((ROOT, "code"), (ROOT, "upstream"), (out, "inputs")):
        for name, checksum in manifest[section].items():
            if digest(root / name) != checksum:
                raise ValueError("Changed sealed artifact: " + name)
            checked_hashes += 1
    for name, checksum in load(out / "output_seal.json").items():
        if digest(out / name) != checksum:
            raise ValueError("Changed scientific output: " + name)
        checked_hashes += 1
    addresses = load(out / "addresses.json")
    with np.load(out / "inputs.npz", allow_pickle=False) as archive:
        inputs = {key: archive[key] for key in archive.files}
    with np.load(out / "certificates.npz", allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    with (out / "relation_certificates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (out / "held_replicate_relations.csv").open() as stream:
        repeats = list(csv.DictReader(stream))
    if len(rows) != 64 or len(repeats) != 4 or len(addresses) != 36:
        raise ValueError("Full declared view/scope/style/replicate account required")
    errors = []
    repeated = {}
    independent_ranks = {}

    def equal(actual, expected):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        errors.append(error)
        if not np.allclose(actual, expected, rtol=1e-8, atol=1e-9):
            raise ArithmeticError(f"Alternate account disagrees: {error}")

    for row in rows:
        view, scope, style, unit = (
            row[k] for k in ("representation", "scope", "style", "unit")
        )
        selected = [
            i for i, a in enumerate(addresses) if scope == "all" or a["role"] == scope
        ]
        full = inputs[view]
        neutral = full[0][:, selected]
        styled = full[manifest["styles"].index(style)][:, selected]
        x = neutral.mean(0) if unit == "three_seed_mean" else neutral[0]
        y = styled.mean(0) if unit == "three_seed_mean" else styled[0]
        key = "__".join((view, scope, style, unit))
        basis, r, _ = qr(x, mode="economic", pivoting=True)
        threshold = float(row["arithmetic_rank_threshold"])
        rank = int(np.count_nonzero(np.abs(np.diag(r)) > threshold))
        if rank != int(row["source_rank"]):
            raise ArithmeticError("Independent QR rank differs: " + key)
        independent_ranks[key] = rank
        q = (
            np.zeros((len(x), len(x)))
            if rank == len(x)
            else (np.eye(len(x)) - basis[:, :rank] @ basis[:, :rank].T)
        )
        equal(q, saved[key + "__q"])
        equal(np.linalg.norm(q @ x), float(row["source_projector_closure"]))
        equal(
            np.linalg.norm(q @ y) / np.linalg.norm(y),
            float(row["relative_relation_defect"]),
        )
        equal(q @ q, q)
        if rank < len(x):
            j = int(np.argmax(np.diag(q)))
            c = q[:, j] / np.linalg.norm(q[:, j])
            equal(c, saved[key + "__witness"])
            a = np.sum(c[:, None] * x, axis=0)
            b = np.sum(c[:, None] * y, axis=0)
            equal(np.sqrt(np.mean(a * a)), float(row["witness_source_rms"]))
            equal(np.sqrt(np.mean(b * b)), float(row["witness_target_rms"]))
            if unit == "seed11_relation":
                a = np.sum(c[None, :, None] * neutral[1:], axis=1)
                b = np.sum(c[None, :, None] * styled[1:], axis=1)
                repeated[(view, scope, style)] = {
                    "neutral_rms": np.sqrt(np.mean(a * a)),
                    "styled_rms": np.sqrt(np.mean(b * b)),
                    "neutral_signed_cross_product": np.sum(a[0] * a[1]),
                    "styled_signed_cross_product": np.sum(b[0] * b[1]),
                    "change_signed_cross_product": np.sum(
                        (b[0] - a[0]) * (b[1] - a[1])
                    ),
                }
    for row in repeats:
        key = tuple(row[k] for k in ("representation", "scope", "style"))
        for field, value in repeated[key].items():
            equal(value, float(row[field]))
    summary = load(out / "summary.json")
    for view in inputs:
        counts = summary["by_view"][view]
        selected = [
            r
            for r in rows
            if r["representation"] == view and r["unit"] == "three_seed_mean"
        ]
        if len(selected) != counts["three_seed_mean_certificates"]:
            raise ArithmeticError("Scope count differs")
        saturated = sum(
            independent_ranks["__".join((view, r["scope"], r["style"], r["unit"]))]
            == int(r["rows"])
            for r in selected
        )
        if saturated != counts["saturated_mean_certificates"]:
            raise ArithmeticError("Saturation count differs")
    return {
        "status": "PASS_ALTERNATE_ARITHMETIC_NOT_THIRD_PARTY",
        "relation_rows": len(rows),
        "held_replicate_rows": len(repeats),
        "authenticated_files": checked_hashes,
        "scalar_array_checks": len(errors),
        "max_absolute_error": max(errors),
        "rank_check": "Pivoted QR diagonal at the unchanged declared arithmetic tolerance",
        "new_model_calls": 0,
        "original_scope_complete": False,
    }


def main():
    """Print the full audit; optionally create an exclusive companion receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.input)
    result["audit_code_sha256"] = digest(Path(__file__))
    if args.receipt:
        with args.receipt.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
