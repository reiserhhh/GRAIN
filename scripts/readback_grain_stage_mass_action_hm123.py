#!/usr/bin/env python3
"""Alternate-cell audit of staged responses without torch or model calls."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_author_evidence_merge_hm102 as H


def audit(out):
    """Rebuild every owner/focus/sign contrast independently of runner code."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    H.verify_hashes(out, H.load(out / "01/seal.json"))
    pairs = H.load(out / "00/pairs.json")
    lookup = {tuple(p["key"]): p for p in pairs}
    with np.load(out / "02/response_fields.npz", allow_pickle=False) as z:
        data = {n: z[n] for n in z.files}
    r, full, scale = data["readings"], data["full_readings"], data["read_scale"]
    errors = {"contrasts": 0.0, "selectivity": 0.0, "primary": 0.0, "couplings": 0.0}
    rebuilt = {}
    for component in ("actor_odd", "actor_even"):
        rebuilt[component] = {
            n: np.zeros_like(data[f"{n}_{component}"])
            for n in ("early", "late", "total", "interaction")
        }
        for v, w, f, c, q, align, mode, readq in np.ndindex(3, 2, 6, 2, 6, 2, 2, 6):
            cells = {n: [] for n in rebuilt[component]}
            for owner in (-1, 1):
                p = lookup[(v, w, f, 2 * c - 1, owner)]
                values = {n: 0.0 for n in cells}
                for polarity, key in ((-1, "low"), (1, "high")):
                    u = p[key]
                    for focus in range(2):
                        for sign_ix in range(2):
                            original = 1 + q * 4 + align * 2 + sign_ix
                            early_ix, late_ix = (
                                1 + (original - 1) * 2,
                                2 + (original - 1) * 2,
                            )
                            base = r[u, 0, focus, mode, readq]
                            e = r[u, early_ix, focus, mode, readq] - base
                            l = r[u, late_ix, focus, mode, readq] - base
                            t = full[u, original, focus, mode, readq] - base
                            weight = polarity * (1 - 2 * focus) * (2 * sign_ix - 1) / 4
                            for n, value in (
                                ("early", e),
                                ("late", l),
                                ("total", t),
                                ("interaction", t - e - l),
                            ):
                                values[n] += weight * value
                for n, cell in cells.items():
                    cell.append(values[n])
            for n, cell in cells.items():
                target = (
                    (cell[1] - cell[0]) / 2
                    if component == "actor_odd"
                    else sum(cell) / 2
                )
                rebuilt[component][n][v, w, f, c, q, align, mode, readq] = target
        for n, x in rebuilt[component].items():
            errors["contrasts"] = max(
                errors["contrasts"], float(abs(x - data[f"{n}_{component}"]).max())
            )
    with (out / "02/selectivity.csv").open() as stream:
        for row in csv.DictReader(stream):
            n = row["stage"]
            v, w, f, c, align, mode = [
                int(row[k])
                for k in (
                    "variant",
                    "wording",
                    "family",
                    "condition",
                    "alignment",
                    "mode",
                )
            ]
            x = rebuilt["actor_odd"][n][v, w, f, c, f, align, mode] / scale[mode]
            actual = abs(x[f]) - sum(abs(x[j]) for j in range(6) if j != f) / 5
            errors["selectivity"] = max(
                errors["selectivity"], abs(actual - float(row["selectivity"]))
            )
    summary = H.load(out / "02/summary.json")
    for row in summary["primary"]:
        w = row["wording"]
        values = {
            n: x[2, w, :, :, :, 0, 1] / scale[1]
            for n, x in rebuilt["actor_odd"].items()
        }
        for n, value in values.items():
            rms = float(np.sqrt(np.sum(value**2) / value.size))
            errors["primary"] = max(
                errors["primary"], abs(rms - row["RMS_source_scaled"][n])
            )
        denom = float(np.sum(values["total"] ** 2))
        for n in ("early", "late", "interaction"):
            actual = float(np.sum(values[n] * values["total"]) / denom)
            errors["primary"] = max(
                errors["primary"], abs(actual - row["signed_total_alignments"][n])
            )
    with np.load(out / "02/couplings.npz", allow_pickle=False) as z:
        for v, align, mode in np.ndindex(3, 2, 2):
            vectors = {}
            for n in rebuilt["actor_odd"]:
                values = (
                    rebuilt["actor_odd"][n][v, :, :, :, :, align, mode] / scale[mode]
                )
                a = np.stack([values[:, :, :, q].ravel() for q in range(6)])
                vectors[n] = a
                ref = z[f"{n}_{v}_{align}_{mode}"]
                for q, t in np.ndindex(6, 6):
                    value = np.dot(a[q], a[t]) / (
                        np.linalg.norm(a[q]) * np.linalg.norm(a[t])
                    )
                    errors["couplings"] = max(
                        errors["couplings"], abs(value - ref[q, t])
                    )
            for q in range(6):
                a, b = vectors["early"][q], vectors["late"][q]
                value = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                errors["couplings"] = max(
                    errors["couplings"],
                    abs(value - z[f"early_late_cosines_{v}_{align}_{mode}"][q]),
                )
    if max(errors.values()) > 1e-9:
        raise AssertionError(errors)
    H.write(
        out / "02/readback.json",
        {
            "status": "NUMERIC_READBACK_PASSED_NOT_GLOBAL",
            "errors": errors,
            "same_agent_not_third_party": True,
            "audit_code_sha256": H.sha(Path(__file__)),
        },
    )
    H.write(out / "02/seal.json", H.seals(out, out / "02"))
    print(errors, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    audit(p.parse_args().out.resolve())
