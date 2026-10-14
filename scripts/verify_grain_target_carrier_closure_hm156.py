#!/usr/bin/env python3
"""Alternate direct-sum audit of actual HM156 carriers and reader endpoints."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_target_carrier_closure_hm156 as R


def verify(out):
    """Check seals, actual donors, finite contrasts and selected native logits."""
    start, manifest = time.monotonic(), R.check(out)
    parent = ROOT / manifest["parent"]
    R.authenticate_parent(parent)
    for section in ("capture_seal.json", "output_seal.json"):
        for name, checksum in R.R.read(out / section).items():
            if R.H.digest(out / name) != checksum:
                raise ValueError("Immutable result changed: " + name)
    rows, design, queries = (
        R.R.read(out / f"{name}.json") for name in ("rows", "design", "queries")
    )
    natural = R.parent_natural(parent)
    lookup = {
        (r["bank"], r["condition"], r["order"], r["e"], r["n"]): r["id"] for r in rows
    }
    ids = [queries["answers"][name][0] for name in ("Yes", "No")]
    weights = R.R.OLD.weights().cpu()[ids].numpy().astype(float)
    captured, checks, max_native, max_formula, max_donor = {}, 0, 0.0, 0.0, 0.0
    for entry in design:
        if time.monotonic() - start > manifest["audit_seconds_cap"]:
            raise TimeoutError("Finite120second audit cap; no renewal")
        row = rows[entry["row"]]
        stem = out / "captures" / f"{entry['batch']:03d}"
        a, receipt = (
            R.R.arrays(stem.with_suffix(".npz")),
            R.R.read(stem.with_suffix(".json")),
        )
        captured[entry["mode"], row["id"]] = a
        if receipt["sha256"] != R.H.digest(stem.with_suffix(".npz")):
            raise ValueError("Capture receipt mismatch")
        key = (row["bank"], row["condition"], row["order"])
        b = natural[row["id"]]
        e = natural[lookup[*key, -row["e"], row["n"]]]
        n = natural[lookup[*key, row["e"], -row["n"]]]
        both = natural[lookup[*key, -row["e"], -row["n"]]]
        mode = entry["mode"]
        own = row["scopes"]["extraversion" if mode.startswith("e_") else "neuroticism"]
        other = row["scopes"][
            "neuroticism" if mode.startswith("e_") else "extraversion"
        ]
        tail = list(
            range(max(max(s) for s in row["scopes"].values()) + 1, len(row["history"]))
        )
        slots = (
            list(range(len(row["history"])))
            if mode == "self"
            else other
            if mode.endswith("_other")
            else list(range(min(own), len(row["history"])))
            if mode.endswith("_closure")
            else tail
        )
        if a["patched_slots"].tolist() != slots:
            raise ValueError("Saved operation has wrong carrier addresses")
        d = (
            b
            if mode == "self"
            else e
            if mode.startswith("e_")
            else n
            if mode.startswith("n_")
            else both
        )
        for name, saved in (
            ("history_key", "donor_key_slice"),
            ("history_value", "donor_value_slice"),
        ):
            expected = b[name].copy()
            if mode == "both_tail_additive":
                source = (
                    e[name].astype(float)
                    + n[name].astype(float)
                    - b[name].astype(float)
                ).astype(expected.dtype)
            else:
                source = d[name]
            expected[:, :, slots] = source[:, :, slots]
            err = float(np.max(abs(expected[:, :, slots].astype(float) - a[saved])))
            max_donor = max(max_donor, err)
            if not np.array_equal(expected[:, :, slots], a[saved]):
                raise ValueError("Actual donor slice differs from prescribed operation")
            receipt_key = (
                "donor_key_sha256" if name == "history_key" else "donor_value_sha256"
            )
            if R.array_digest(expected) != receipt[receipt_key]:
                raise ValueError("Full actual memory argument hash failed")
            checks += a[saved].size
        observed = a["states"].astype(float) @ weights.T
        max_native = max(
            max_native, float(np.max(abs(observed - a["selected_logits"])))
        )
        if not np.allclose(
            observed,
            a["selected_logits"],
            atol=manifest["selected_atol"],
            rtol=manifest["selected_rtol"],
        ):
            raise ValueError("Native W*h reconstruction failed")
        if not np.array_equal(a["prefix_states"], b["prefix_states"]):
            raise ValueError("Actual prefix changed")
        checks += observed.size + a["prefix_states"].size
    with (out / "characters.csv").open() as f:
        coefficient_rows = list(csv.DictReader(f))
    for record in coefficient_rows:
        group = R.R.GROUPS[int(record["group"])]
        mask, mode = int(record["mask"]), record["mode"]
        terms = []
        for e, n, p in itertools.product((-1, 1), repeat=3):
            row_id = lookup[*group, e, n]
            a = natural[row_id] if mode == "natural" else captured[mode, row_id]
            index = queries["definitions"].index(
                {"axis": record["axis"], "family": record["family"], "person": p}
            )
            sign = math.prod(s for j, s in enumerate((e, n, p)) if mask & (1 << j))
            terms.append(
                sign
                * (
                    float(a["selected_logits"][index, 0])
                    - float(a["selected_logits"][index, 1])
                )
            )
        expected = math.fsum(terms) / 8
        err = abs(expected - float(record["logodds"]))
        max_formula = max(max_formula, err)
        if err > 1e-10:
            raise ValueError("Independent endpoint character sum failed")
        checks += 1
    with (out / "actions.csv").open() as f:
        actions = list(csv.DictReader(f))
    with (out / "interactions.csv").open() as f:
        interactions = list(csv.DictReader(f))
    for gi, group in enumerate(R.R.GROUPS):
        account = R.R.arrays(out / "accounts" / f"{gi:02d}.npz")
        for mode in ("natural", *R.MODES):
            for ie, inn, ip in itertools.product(range(2), repeat=3):
                row_id = lookup[*group, 2 * ie - 1, 2 * inn - 1]
                a = natural[row_id] if mode == "natural" else captured[mode, row_id]
                for axis, family in itertools.product(range(2), range(3)):
                    index = queries["definitions"].index(
                        {
                            "axis": R.R.M.AXES[axis],
                            "family": R.R.M.FAMILIES[family],
                            "person": 2 * ip - 1,
                        }
                    )
                    raw = float(a["selected_logits"][index, 0]) - float(
                        a["selected_logits"][index, 1]
                    )
                    err = abs(
                        raw - account[mode + "_logodds"][ie, inn, ip, axis, family]
                    )
                    max_formula = max(max_formula, err)
                    checks += 1
        for record in (v for v in actions if int(v["group"]) == gi):
            axis, mode = R.R.M.AXES.index(record["axis"]), record["mode"]
            base = account["natural_logodds"][..., axis, :]
            ref = (
                base[::-1]
                if mode.startswith("e_")
                else base[:, ::-1]
                if mode.startswith("n_")
                else base[::-1, ::-1]
            )
            x = (account[mode + "_logodds"][..., axis, :] - base).ravel()
            y = (ref - base).ravel()
            xx = math.fsum(float(v) ** 2 for v in x)
            yy = math.fsum(float(v) ** 2 for v in y)
            xy = math.fsum(float(a) * float(b) for a, b in zip(x, y, strict=True))
            expected = {
                "rms": math.sqrt(xx / len(x)),
                "reference_rms": math.sqrt(yy / len(y)),
                "signed_projection": xy / yy if yy else None,
                "cosine": xy / math.sqrt(xx * yy) if xx and yy else None,
                "relative_remainder": math.sqrt(
                    math.fsum(
                        (float(a) - float(b)) ** 2 for a, b in zip(x, y, strict=True)
                    )
                    / yy
                )
                if yy
                else None,
            }
            for name, value in expected.items():
                if value is None:
                    if record[name] != "":
                        raise ValueError("Undefined action was replaced by a fake zero")
                else:
                    max_formula = max(max_formula, abs(value - float(record[name])))
                checks += 1
        for name in ("history_key", "history_value"):
            terms = {}
            for e, n in itertools.product((-1, 1), repeat=2):
                row_id = lookup[*group, e, n]
                tail = list(
                    range(
                        max(max(s) for s in rows[row_id]["scopes"].values()) + 1,
                        len(rows[row_id]["history"]),
                    )
                )
                terms[e, n] = natural[row_id][name][:, :, tail].astype(float)
            mixed = (terms[-1, -1] - terms[-1, 1] - terms[1, -1] + terms[1, 1]) / 4
            max_formula = max(
                max_formula,
                float(np.max(abs(mixed - account[name + "_tail_characters"][3]))),
            )
            for e, n in itertools.product((-1, 1), repeat=2):
                defect = terms[-e, -n] - (terms[-e, n] + terms[e, -n] - terms[e, n])
                err = float(np.max(abs(defect - 4 * e * n * mixed)))
                max_formula = max(max_formula, err)
                checks += defect.size
        for axis in range(2):
            b = account["natural_logodds"][..., axis, :]
            x, y, r, a = [
                account[mode + "_logodds"][..., axis, :]
                for mode in ("e_tail", "n_tail", "both_tail_real", "both_tail_additive")
            ]
            direct = {
                "total": r - x - y + b,
                "encoding": r - a,
                "reader": a - x - y + b,
            }
            for name, value in direct.items():
                err = float(np.max(abs(value - account[f"{name}_{axis}"])))
                max_formula = max(max_formula, err)
                checks += value.size
            record = next(
                v
                for v in interactions
                if int(v["group"]) == gi and v["axis"] == R.R.M.AXES[axis]
            )
            for name, value in direct.items():
                rms = math.sqrt(
                    math.fsum(float(a) ** 2 for a in value.ravel()) / value.size
                )
                max_formula = max(max_formula, abs(rms - float(record[name + "_rms"])))
                checks += 1
    if max_formula > 1e-10 or max_donor != 0:
        raise ValueError("Finite carrier identities failed")
    result = {
        "status": "PASS_ALTERNATE_FORMULA_NOT_BLIND",
        "checks": checks,
        "max_native_logit_error": max_native,
        "max_formula_error": max_formula,
        "max_donor_slice_error": max_donor,
        "seconds": time.monotonic() - start,
        "new_model_forwards": 0,
    }
    path = out / "alternate_audit.json"
    if not path.exists():
        R.write_json(path, result)
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=R.DEFAULT)
    verify(parser.parse_args().out)
