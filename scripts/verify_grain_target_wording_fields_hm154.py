#!/usr/bin/env python3
"""Alternate CPU64 arithmetic for HM154, not independent blind replication."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_target_wording_fields_hm154 import (
    DEFAULT,
    PARENT,
    SCALARS,
    capture,
    check,
    digest,
    read,
    weights,
    write,
)


def rows(path):
    """Read complete scientific CSVs without applying outcome filters."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify(out):
    """Check all Gram-derived accounts and six fixed full-vocabulary spots."""
    start = time.monotonic()
    manifest = check(out)
    if read(out / "status.json")["status"] != "COMPLETE_ANALYSIS":
        raise ValueError("Analysis is not terminal complete")
    for name, checksum in read(out / "output_seal.json").items():
        if digest(out / name) != checksum:
            raise ValueError("Scientific output changed: " + name)
    texts = read(out / "texts.json")
    cases = read(PARENT / "cases.json")
    units, selected, text_rows, conditions = [
        rows(out / name)
        for name in (
            "unit_accounts.csv",
            "selected_token_accounts.csv",
            "text_accounts.csv",
            "condition_accounts.csv",
        )
    ]
    if tuple(map(len, (units, selected, text_rows, conditions))) != (864, 5184, 54, 18):
        raise ValueError("Full support missing")
    w = weights().numpy()
    w64 = w.astype(float)
    errors, projection_errors, native_errors = [], [], []

    def equal(actual, expected, projection=False):
        error = float(np.max(abs(np.asarray(actual) - np.asarray(expected))))
        (projection_errors if projection else errors).append(error)
        if not np.allclose(
            actual,
            expected,
            atol=1e-4 if projection else 1e-9,
            rtol=1e-5 if projection else 1e-9,
            equal_nan=True,
        ):
            raise ArithmeticError(f"Alternate numeric mismatch: {error}")

    def guard():
        if time.monotonic() - start > manifest["audit_seconds_cap"]:
            raise TimeoutError("Sealed120second audit exhausted; no renewal")

    m_coeff = np.zeros((3, 6))
    d_coeff = np.zeros((3, 6))
    for f in range(3):
        m_coeff[f, [f, f + 3]] = 0.5
        d_coeff[f, [f, f + 3]] = [0.5, -0.5]
    mean_m, mean_d = m_coeff.mean(0), d_coeff.mean(0)
    signs = ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1))
    spots = {tuple(address) for address in manifest["fixed_full_vocabulary_spots"]}
    pair_indices = ((0, 1), (0, 2), (1, 2))
    for text in texts:
        guard()
        j = text["text_index"]
        with np.load(out / "accounts" / f"{j:04d}.npz", allow_pickle=False) as z:
            account = {k: z[k] for k in z.files}
        raw = [capture(i) for i in text["case_ids"]]
        states = [r["states"].astype(float) for r in raw]
        delta = np.stack(
            [h[0, 16:] - h[1, 16:] - h[2, :16] + h[3, :16] for h in states]
        )
        ids = np.asarray(cases[text["case_ids"][0]]["tokens"][16:])
        chosen = np.einsum("tv,itv->it", w64[ids], delta)
        normalizer = np.stack(
            [
                r["partition"].astype(float)[0, 16:]
                - r["partition"].astype(float)[1, 16:]
                - r["partition"].astype(float)[2, :16]
                + r["partition"].astype(float)[3, :16]
                for r in raw
            ]
        )
        equal(chosen, account["selected_u"], projection=True)
        equal(
            account["selected_shape"] + account["calibration"],
            account["reconstructed_k"],
        )
        equal(account["reference_mean_u"] - normalizer, account["calibration"])
        equal(chosen - normalizer, account["native_k"], projection=True)
        native_errors.append(
            float(np.max(abs(chosen - normalizer - account["native_k"])))
        )
        for t in range(16):
            g = account["gram"][t]
            gd = d_coeff @ g @ d_coeff.T
            common = mean_m @ g @ mean_m
            target = mean_d @ g @ mean_d
            wording = sum((c - mean_m) @ g @ (c - mean_m) for c in m_coeff) / 3
            interaction = sum((c - mean_d) @ g @ (c - mean_d) for c in d_coeff) / 3
            total = sum(g[i, i] for i in range(6)) / 6
            orbit = [
                2 * sum(e[f] * e[h] * gd[f, h] for f, h in pair_indices) / 3
                for e in signs
            ]
            cosines = [
                gd[f, h] / np.sqrt(gd[f, f] * gd[h, h])
                if gd[f, f] > 0 and gd[h, h] > 0
                else np.nan
                for f, h in pair_indices
            ]
            margin = orbit[0] - max(orbit[1:])
            values = {
                "common_energy": common,
                "target_energy": target,
                "wording_energy": wording,
                "interaction_energy": interaction,
                "total_energy": total,
                "intended_support": orbit[0],
                "intended_margin": margin,
                "mean_cross_family_cosine": np.mean(cosines),
                "energy_closure_error": total - common - target - wording - interaction,
                "orbit_mean_error": sum(orbit) / 4,
                "support_identity_error": orbit[0] - 2 * target + interaction,
                **{f"orbit_{i}": value for i, value in enumerate(orbit)},
                **{
                    f"cosine_{f}_{h}": value
                    for (f, h), value in zip(pair_indices, cosines, strict=True)
                },
            }
            row = units[j * 16 + t]
            if int(row["text_index"]) != j or int(row["token_index"]) != t + 16:
                raise ValueError("Text/native position address mismatch")
            for key, value in values.items():
                equal(value, float(row[key]) if row[key] else np.nan)
            for key, truth in (
                ("strict_first", margin > 1e-10),
                ("tied_first", abs(margin) <= 1e-10),
                ("not_first", margin < -1e-10),
            ):
                if (row[key] == "True") != truth:
                    raise ArithmeticError("Partition ordering changed")
            for i, case_id in enumerate(text["case_ids"]):
                sr = selected[(j * 16 + t) * 6 + i]
                if (
                    int(sr["case_id"]),
                    int(sr["native_token_id"]),
                    int(sr["token_index"]),
                ) != (case_id, ids[t], t + 16):
                    raise ValueError("Selected-token source binding changed")
                for key in (
                    "selected_u",
                    "reference_mean_u",
                    "selected_shape",
                    "calibration",
                    "reconstructed_k",
                    "native_k",
                ):
                    equal(float(sr[key]), account[key][i, t])
            if (j, t) in spots:
                fields = delta[:, t] @ w64.T
                midpoint = np.mean([(h[2, t] + h[3, t]) / 2 for h in states], axis=0)
                logits = w64 @ midpoint
                q = np.exp(logits - logsumexp(logits))
                mean = fields @ q
                centered = fields - mean[:, None]
                equal(mean, account["reference_mean_u"][:, t], projection=True)
                equal((centered * q) @ centered.T, g, projection=True)
                equal(q.min(), account["reference_min_probability"][t], projection=True)
    aggregate_keys = (
        SCALARS
        + tuple(f"orbit_{j}" for j in range(4))
        + tuple(f"cosine_{f}_{g}" for f, g in pair_indices)
    )
    for row in text_rows + conditions:
        guard()
        group = [
            r
            for r in units
            if (
                r["text_index"] == row["text_index"]
                if "text_index" in row
                else all(r[k] == row[k] for k in ("situation", "style"))
            )
        ]
        equal(len(group), int(row["native_units"]))
        for key in aggregate_keys:
            available = [float(r[key]) for r in group if r[key]]
            equal(
                np.mean(available) if available else np.nan,
                float(row[key]) if row[key] else np.nan,
            )
        for key in ("strict_first", "tied_first", "not_first"):
            equal(sum(r[key] == "True" for r in group), int(row[key + "_count"]))
    summary = read(out / "summary.json")
    for key in aggregate_keys:
        vals = [float(r[key]) for r in units if r[key]]
        expected = summary["all_unit_means"][key]
        equal(
            np.mean(vals) if vals else np.nan,
            expected if expected is not None else np.nan,
        )
    receipt = {
        "status": "PASS_FINITE_ALTERNATE_NOT_BLIND",
        "algebra_comparisons": len(errors),
        "max_algebra_error": max(errors),
        "max_projection_spot_error": max(projection_errors),
        "max_selected_native_error": max(native_errors),
        "full_vocabulary_spots": len(spots),
        "every_vocabulary_field_independently_recomputed": False,
        "seconds": time.monotonic() - start,
        "decoder_forwards": 0,
        "scientific_output_seal_sha256": digest(out / "output_seal.json"),
        "verifier_sha256": digest(Path(__file__)),
        "original_scope_complete": False,
    }
    guard()
    write(out / "alternate_audit.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    verify(parser.parse_args().out)
