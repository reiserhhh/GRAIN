#!/usr/bin/env python3
"""Result-after HM72 pairing diagnostic; no scoring fit or neural execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_event_field import FEATURES, cosine

PRIMARY = ROOT / "results/grain_math_response_v1/event_field_presentation_r1"
OUT = PRIMARY / "04_cross_order_diagnostic"
CODE = (
    "docs/GRAIN_EVENT_FIELD_CROSS_ORDER_COMPANION_HM72.md",
    "scripts/diagnose_grain_event_field_cross_order.py",
    "tests/test_grain_event_field_cross_order.py",
    "grain_research/construction/math_event_field.py",
    "grain_research/construction/math_cold_relational.py",
)


def sha(path):
    """Bind a scientific input or output to its actual byte content."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    """Read saved JSON without evaluating instructions or object code."""
    return json.loads(Path(path).read_text())


def write(path, value):
    """Write a deterministic generated numerical record."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export(path, rows):
    """Preserve every designed case, including null cosine values."""
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pairing(a, b):
    """Separate identity alignment from two fixed wrong-slot controls."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim not in (1, 2):
        raise ValueError("Comparable vector or event-by-vector fields required")
    return {
        "cosine": cosine(a, b),
        "cyclic_wrong_event_cosine": cosine(a, np.roll(b, 1, axis=0))
        if b.ndim == 2
        else None,
        "reversed_slot_cosine": cosine(a, b[::-1]) if b.ndim == 2 else None,
    }


def dot_parts(source, new):
    """Account for all cross-order dot products without deleting odd responses."""
    s, n = np.asarray(source, float), np.asarray(new, float)
    if s.shape != n.shape or s.shape[0] != 2 or not np.isfinite([s, n]).all():
        raise ValueError("Two complete comparable finite orders required")
    es, os = (s[0] + s[1]) / 2, (s[0] - s[1]) / 2
    en, on = (n[0] + n[1]) / 2, (n[0] - n[1]) / 2
    return {
        name: float(np.vdot(a, b))
        for name, a, b in (
            ("even_even", es, en),
            ("even_odd", es, on),
            ("odd_even", os, en),
            ("odd_odd", os, on),
        )
    }


def prepare():
    """Freeze the result-after question and verified primary data before run."""
    if OUT.exists():
        raise FileExistsError("Preserve an existing companion")
    if (
        read(PRIMARY / "verification.json")["status"]
        != "VERIFIED_FINITE_HM72_NOT_GLOBAL"
    ):
        raise ValueError("Verified primary required")
    seals = read(PRIMARY / "03/output_seal.json")
    for name, digest in seals.items():
        if sha(PRIMARY / name) != digest:
            raise ValueError("Primary output drift")
    (OUT / "00/code").mkdir(parents=True)
    code = {}
    for name in CODE:
        destination = OUT / "00/code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        code[name] = sha(ROOT / name)
    inputs = [PRIMARY / name for name in seals] + [
        PRIMARY / "03/output_seal.json",
        PRIMARY / "verification.json",
    ]
    write(
        OUT / "00/manifest.json",
        {
            "result_after_not_primary": True,
            "code": code,
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
            "new_model_calls_fits": [0, 0],
        },
    )


def check():
    """Refuse silent drift in frozen code or primary observations."""
    manifest = read(OUT / "00/manifest.json")
    for name, digest in (manifest["code"] | manifest["inputs"]).items():
        if sha(ROOT / name) != digest:
            raise ValueError("Input/code drift: " + name)


def summarize(cross, within):
    """Aggregate all cases without assigning a score to undefined directions."""
    summaries = []
    for feature, i, j in itertools.product(FEATURES, range(2), range(2)):
        rows = [
            r
            for r in cross
            if r["feature"] == feature
            and int(r["source_order"]) == i
            and int(r["new_order"]) == j
        ]
        result = {
            "feature": feature,
            "source_order": i,
            "new_order": j,
            "rows": len(rows),
        }
        for field in ("cosine", "cyclic_wrong_event_cosine", "reversed_slot_cosine"):
            values = [float(r[field]) for r in rows if r[field] not in (None, "")]
            result[field + "_defined"] = len(values)
            result[field + "_mean"] = (
                math.fsum(values) / len(values) if values else None
            )
        summaries.append(result)
    for feature, bank in itertools.product(FEATURES, ("source", "new")):
        rows = [r for r in within if r["feature"] == feature and r["bank"] == bank]
        values = [float(r["cosine"]) for r in rows if r["cosine"] not in (None, "")]
        summaries.append(
            {
                "feature": feature,
                "bank": bank,
                "rows": len(rows),
                "cosine_defined": len(values),
                "cosine_mean": math.fsum(values) / len(values) if values else None,
            }
        )
    return summaries


def run():
    """Compute all pairings and retain the algebraic contribution accounts."""
    check()
    if (OUT / "output_seal.json").exists():
        raise FileExistsError("Do not overwrite an analyzed companion")
    cross, within, dots = [], [], []
    with np.load(PRIMARY / "03/profile_vectors.npz", allow_pickle=False) as data:
        for pair, case, feature in itertools.product(range(30), range(8), FEATURES):
            key = f"{pair:02d}_{feature}"
            f, r = data[f"{key}_forward_case{case}"], data[f"{key}_reverse_case{case}"]
            source, new = np.stack((f[0], r[0])), np.stack((f[1], r[1]))
            identity = {"pair": pair, "case": case, "feature": feature}
            for i, j in itertools.product(range(2), repeat=2):
                cross.append(
                    identity
                    | {"source_order": i, "new_order": j}
                    | pairing(source[i], new[j])
                )
            for bank, values in (("source", source), ("new", new)):
                within.append(identity | {"bank": bank} | pairing(*values))
            parts = dot_parts(source, new)
            dots.append(identity | parts)
    export(OUT / "cross_bank.csv", cross)
    export(OUT / "within_bank.csv", within)
    export(OUT / "dot_accounts.csv", dots)
    write(
        OUT / "summary.json",
        {
            "result_after_not_primary": True,
            "rows": [len(cross), len(within), len(dots)],
            "summaries": summarize(cross, within),
            "original_scope_complete": False,
        },
    )
    paths = [
        OUT / name
        for name in (
            "cross_bank.csv",
            "within_bank.csv",
            "dot_accounts.csv",
            "summary.json",
        )
    ]
    write(OUT / "output_seal.json", {p.name: sha(p) for p in paths})
    print("HM72 companion analyzed all cross-order/within-order pairings", flush=True)


def verify():
    """Independently replay dot products and every saved pairing; no neural rerun."""
    check()
    for name, digest in read(OUT / "output_seal.json").items():
        if sha(OUT / name) != digest:
            raise ValueError("Companion output drift")
    with (OUT / "cross_bank.csv").open() as handle:
        cross = list(csv.DictReader(handle))
    with (OUT / "within_bank.csv").open() as handle:
        within = list(csv.DictReader(handle))
    with (OUT / "dot_accounts.csv").open() as handle:
        accounts = list(csv.DictReader(handle))
    if (len(cross), len(within), len(accounts)) != (4800, 2400, 1200):
        raise ValueError("Incomplete designed battery")
    if summarize(cross, within) != read(OUT / "summary.json")["summaries"]:
        raise ValueError("Summary arithmetic replay difference")
    maximum, count = 0.0, 0
    with np.load(PRIMARY / "03/profile_vectors.npz", allow_pickle=False) as data:
        for row in accounts:
            key = f"{int(row['pair']):02d}_{row['feature']}"
            case = int(row["case"])
            f, r = data[f"{key}_forward_case{case}"], data[f"{key}_reverse_case{case}"]
            s, n = np.stack((f[0], r[0])), np.stack((f[1], r[1]))
            parts = dot_parts(s, n)
            for name, value in parts.items():
                np.testing.assert_equal(float(row[name]), value)
            for i, j in itertools.product(range(2), repeat=2):
                a, b = 1 - 2 * i, 1 - 2 * j
                reconstructed = (
                    parts["even_even"]
                    + b * parts["even_odd"]
                    + a * parts["odd_even"]
                    + a * b * parts["odd_odd"]
                )
                alternate = float(
                    np.sum(
                        np.asarray(s[i], np.longdouble)
                        * np.asarray(n[j], np.longdouble),
                        dtype=np.longdouble,
                    )
                )
                np.testing.assert_allclose(
                    reconstructed, alternate, atol=2e-12, rtol=1e-12
                )
                maximum = max(maximum, abs(reconstructed - alternate))
                count += 1
        for row in cross + within:
            key = f"{int(row['pair']):02d}_{row['feature']}"
            case = int(row["case"])
            f, r = data[f"{key}_forward_case{case}"], data[f"{key}_reverse_case{case}"]
            if "bank" in row:
                bank = int(row["bank"] == "new")
                expected = pairing(f[bank], r[bank])
            else:
                expected = pairing(
                    (f, r)[int(row["source_order"])][0],
                    (f, r)[int(row["new_order"])][1],
                )
            for name, value in expected.items():
                if value is None:
                    if row[name] != "":
                        raise ValueError("Undefined cosine was replaced")
                else:
                    np.testing.assert_equal(float(row[name]), value)
                count += 1
    write(
        OUT / "verification.json",
        {
            "status": "VERIFIED_RESULT_AFTER_COMPANION",
            "scalar_checks": count,
            "dot_max_error": maximum,
            "new_calls_fits": [0, 0],
            "same_agent_alternate_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    print(
        f"HM72 companion verified {count} scalar accounts;maxerror={maximum:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    action = parser.parse_args().action
    {"prepare": prepare, "run": run, "verify": verify}[action]()
