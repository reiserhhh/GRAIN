#!/usr/bin/env python3
"""Recover HM155 JSON serialization without editing sealed science or rerunning it."""

from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_qualified_target_field_hm155 as R


def table(path):
    """Read serialized numeric results into ordinary Python scalar types."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    integers = {"group", "bank", "condition", "order", "mask"}
    text = {"mode", "axis", "family"}
    return [
        {
            k: int(v) if k in integers else v if k in text else float(v) if v else None
            for k, v in row.items()
        }
        for row in rows
    ]


def scalar_accounts(out):
    """Audit every action/composition CSV row directly from recorded endpoints."""
    rows = R.read(out / "rows.json")
    indexed = {(r["bank"], r["condition"], r["order"], r["e"], r["n"]): r for r in rows}
    observations = {
        (d["mode"], d["row"]): R.arrays(out / "captures" / f"{d['batch']:03d}.npz")
        for d in R.read(out / "design.json")
    }

    # Use direct per-record logits rather than the runner's cube/character helpers.
    def values(mode, record, axis):
        logits = observations[mode, record["id"]]["selected_logits"].astype(float)
        return (logits[:, 0] - logits[:, 1]).reshape(2, 6)[axis]

    def square_mean(x):
        return math.fsum(float(a) ** 2 for a in np.ravel(x)) / np.size(x)

    checks, maximum = 0, 0.0

    def compare(actual, expected):
        nonlocal checks, maximum
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                raise ValueError("Inconsistent undefined account")
        else:
            error = abs(actual - expected)
            maximum = max(maximum, error)
            if not math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError(
                    f"Alternate scalar account differs: {actual}, {expected}"
                )
        checks += 1

    for entry in table(out / "actions.csv"):
        axis = R.M.AXES.index(entry["axis"])
        group = tuple(entry[k] for k in ("bank", "condition", "order"))
        records = [
            r
            for r in rows
            if tuple(r[k] for k in ("bank", "condition", "order")) == group
        ]
        xs, ys = [], []
        for r in records:
            mode = entry["mode"]
            donor = indexed[
                *group,
                -r["e"] if mode != "n_local" else r["e"],
                -r["n"] if mode != "e_local" else r["n"],
            ]
            b = values("natural", r, axis)
            xs.extend(values(mode, r, axis) - b)
            ys.extend(values("natural", donor, axis) - b)
        xx, yy = math.fsum(a * a for a in xs), math.fsum(a * a for a in ys)
        xy = math.fsum(a * b for a, b in zip(xs, ys, strict=True))
        expected = {
            "rms": math.sqrt(xx / len(xs)),
            "reference_rms": math.sqrt(yy / len(ys)),
            "signed_projection": xy / yy if yy else None,
            "cosine": xy / math.sqrt(xx * yy) if xx and yy else None,
            "relative_remainder": math.sqrt(
                math.fsum((a - b) ** 2 for a, b in zip(xs, ys, strict=True)) / yy
            )
            if yy
            else None,
        }
        for key, value in expected.items():
            compare(entry[key], value)
    for entry in table(out / "composition.csv"):
        axis = R.M.AXES.index(entry["axis"])
        group = tuple(entry[k] for k in ("bank", "condition", "order"))
        records = [
            r
            for r in rows
            if tuple(r[k] for k in ("bank", "condition", "order")) == group
        ]
        b, e, n, both, whole = [
            np.concatenate([values(mode, r, axis) for r in records]) for mode in R.MODES
        ]
        parts = {
            "e": e - b,
            "n": n - b,
            "interaction": both - e - n + b,
            "nonlocal_remainder": whole - both,
            "whole": whole - b,
        }
        for key, value in parts.items():
            compare(entry[key + "_rms"], math.sqrt(square_mean(value)))
        closure = (
            parts["e"]
            + parts["n"]
            + parts["interaction"]
            + parts["nonlocal_remainder"]
            - parts["whole"]
        )
        compare(entry["identity_error"], float(np.max(abs(closure))))
    for entry in table(out / "fields.csv"):
        gram = R.arrays(out / "accounts" / f"{entry['group']:02d}_gram.npz")["gram"]
        offset = 3 * R.M.AXES.index(entry["axis"])
        g = gram[offset : offset + 3, offset : offset + 3]
        compare(
            entry["cross_wording_support"],
            2
            * math.fsum(float(g[i, j]) for i, j in itertools.combinations(range(3), 2))
            / 3,
        )
        compare(
            entry["mean_field_energy"], math.fsum(float(g[i, i]) for i in range(3)) / 3
        )
    masses = []
    for r in rows:
        observation = observations["natural", r["id"]]
        selected = observation["selected_logits"].astype(float)
        masses.extend(
            np.exp(
                logsumexp(selected, axis=1) - observation["normalizer"].astype(float)
            )
        )
    self_error = max(
        float(
            np.max(
                abs(
                    a["selected_logits"]
                    - observations["natural", rid]["selected_logits"]
                )
            )
        )
        for (mode, rid), a in observations.items()
        if mode == "self"
    )
    return (
        {
            "status": "PASS_DIRECT_ENDPOINT_AGGREGATES_NOT_BLIND",
            "checks": checks,
            "max_error": maximum,
            "new_forwards": 0,
        },
        float(np.mean(masses)),
        float(np.min(masses)),
        self_error,
    )


def recover(out=R.DEFAULT):
    """Preserve partial JSON and completed arrays; create an explicit new receipt."""
    start = time.monotonic()
    R.check(out)
    collection = R.read(out / "collection_status.json")
    if collection["status"] != "COLLECTED" or collection["batches_done"] != 164:
        raise RuntimeError("Full captures required; no partial scientific recovery")
    for name, checksum in R.read(out / "capture_seal.json").items():
        if R.H.digest(out / name) != checksum:
            raise ValueError("Capture changed: " + name)
    failure_hash = R.H.digest(out / "summary.json")
    try:
        R.read(out / "summary.json")
    except ValueError:
        pass
    else:
        raise RuntimeError("Expected preserved partial JSON, not a completed summary")
    recovery = out / "serialization_recovery_r1"
    if recovery.exists() or (out / "output_seal.json").exists():
        raise RuntimeError("Do not overwrite or retry existing recovery")
    audit, mass_mean, mass_min, self_error = scalar_accounts(out)
    coefficients = table(out / "characters.csv")
    own = [
        r
        for r in coefficients
        if r["mode"] == "natural"
        and r["mask"] == (5 if r["axis"] == R.M.AXES[0] else 6)
    ]
    if len(own) != 48:
        raise ValueError("Missing predeclared qualification rows")
    summary = {
        "packet": "HM155",
        "records": 32,
        "batches": 164,
        "inquiry_sequences": 1968,
        "qualified_scalar_positive": sum(bool(r["logodds"] > 0) for r in own),
        "qualified_scalar_count": len(own),
        "qualified_scalar_rows": own,
        "actions": table(out / "actions.csv"),
        "composition": table(out / "composition.csv"),
        "fields": table(out / "fields.csv"),
        "answer_mass_mean": mass_mean,
        "answer_mass_min": mass_min,
        "own_donor_selected_error": self_error,
        "analysis_seconds": None,
        "analysis_seconds_note": "Original analysis failed at final JSON serialization; elapsed time not recovered.",
        "human_personality_accuracy": False,
        "writer_reader_laws_identified_as_equal": False,
        "original_scope_complete": False,
        "recovery": "CSV and saved arrays only; no science rerun",
    }
    recovery.mkdir()
    R.H.write_json(recovery / "summary.json", summary)
    R.H.write_json(recovery / "direct_endpoint_audit.json", audit)
    R.H.write_json(
        recovery / "recovery_receipt.json",
        {
            "status": "SERIALIZATION_RECOVERED_NO_SCIENCE_EDIT_OR_RERUN",
            "failure": "TypeError: Object of type int64 is not JSON serializable (qualified_scalar_positive)",
            "preserved_partial_summary_sha256": failure_hash,
            "recovery_script_sha256": R.H.digest(Path(__file__)),
            "new_forwards": 0,
            "refitted_models": 0,
            "seconds": time.monotonic() - start,
            "original_scope_complete": False,
        },
    )
    if R.H.digest(out / "summary.json") != failure_hash:
        raise ValueError("Original partial summary was changed")
    sealed = sorted(
        list((out / "accounts").glob("*"))
        + list(out.glob("*.csv"))
        + [out / "summary.json"]
        + list(recovery.glob("*.json"))
    )
    R.H.write_json(
        out / "output_seal.json",
        {str(p.relative_to(out)): R.H.digest(p) for p in sealed},
    )
    R.H.write_json(
        out / "analysis_status.json",
        {
            "status": "COMPLETE_WITH_SEPARATE_SERIALIZATION_RECOVERY",
            "canonical_summary": "serialization_recovery_r1/summary.json",
            "original_partial_summary_preserved": True,
            "original_scope_complete": False,
        },
    )
    print(
        {
            "qualified_positive": summary["qualified_scalar_positive"],
            "of": len(own),
            "direct_endpoint_audit": audit,
            "seconds": time.monotonic() - start,
        }
    )


if __name__ == "__main__":
    recover()
