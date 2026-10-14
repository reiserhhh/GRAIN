#!/usr/bin/env python3
"""Alternate CPU64 checks of diagnostic fields, not a blind third-party audit."""

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
from scripts.run_grain_context_diagnostics_hm153 import (
    DEFAULT,
    PARENT,
    check,
    digest,
    read,
    weights,
    write,
)


def rows(path):
    """Load all records with their native case addresses."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify(out):
    """Check every chosen-token account plus fixed full-vocabulary observations."""
    start = time.monotonic()
    m = check(out)
    for name, checksum in read(out / "output_seal.json").items():
        if digest(out / name) != checksum:
            raise ValueError("Scientific output changed: " + name)
    cases = read(PARENT / "cases.json")
    token_rows, case_rows = (
        rows(out / "token_accounts.csv"),
        rows(out / "case_accounts.csv"),
    )
    cross_rows, conditions = (
        rows(out / "cross_target_accounts.csv"),
        rows(out / "condition_accounts.csv"),
    )
    if (len(token_rows), len(case_rows), len(cross_rows), len(conditions)) != (
        15552,
        972,
        2592,
        108,
    ):
        raise ValueError("Full support missing")
    w = weights().numpy()
    errors, native_errors, spot_errors = [], [], []

    def equal(actual, expected, projection=False):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        (spot_errors if projection else errors).append(error)
        if not np.allclose(
            actual,
            expected,
            atol=1e-4 if projection else 1e-9,
            rtol=1e-5 if projection else 1e-9,
        ):
            raise ArithmeticError(f"Alternate account mismatch:{error}")

    def raw(i):
        with np.load(PARENT / "captures" / f"{i:04d}.npz", allow_pickle=False) as z:
            return {k: z[k].astype(float) for k in z.files}

    def delta(z):
        return (
            z["states"][0, 16:]
            - z["states"][1, 16:]
            - z["states"][2, :16]
            + z["states"][3, :16]
        )

    for i, (c, r) in enumerate(zip(cases, case_rows, strict=True)):
        if time.monotonic() - start > m["audit_seconds_cap"]:
            raise TimeoutError("Finite120second audit exhausted")
        z = raw(i)
        d = delta(z)
        a = (
            z["partition"][0, 16:]
            - z["partition"][1, 16:]
            - z["partition"][2, :16]
            + z["partition"][3, :16]
        )
        ids = c["tokens"][16:]
        chosen = np.einsum("ij,ij->i", w[ids].astype(float), d)
        with np.load(out / "accounts" / f"{i:04d}.npz", allow_pickle=False) as archive:
            account = {k: archive[k] for k in archive.files}
        equal(chosen, account["selected_u"], projection=True)
        equal(
            account["selected_shape"] + account["calibration"],
            account["reconstructed_k"],
        )
        equal(account["reference_mean_u"] - a, account["calibration"])
        native_errors.append(float(np.max(abs(chosen - a - account["native_k"]))))
        equal(chosen - a, account["native_k"], projection=True)
        for t, tr in enumerate(token_rows[i * 16 : (i + 1) * 16]):
            if (
                int(tr["case_id"]) != i
                or int(tr["native_token_id"]) != ids[t]
                or int(tr["token_index"]) != t + 16
            ):
                raise ValueError("Native address changed")
            for key, values in account.items():
                equal(float(tr[key]), values[t])
        for key, field in (
            ("shape_sum", "selected_shape"),
            ("calibration_sum", "calibration"),
            ("native_k_sum", "native_k"),
            ("reconstructed_k_sum", "reconstructed_k"),
        ):
            equal(float(r[key]), sum(account[field]))
        if i in (0, 6, 485, 971):
            for t in (0, 15):
                # All vocabulary rows are recomputed only at these predeclared spots.
                u = w.astype(float) @ d[t]
                midpoint = (z["states"][2, t] + z["states"][3, t]) / 2
                logits = w.astype(float) @ midpoint
                p = np.exp(logits - logsumexp(logits))
                mean = float(np.dot(p, u))
                equal(mean, account["reference_mean_u"][t], projection=True)
                equal(
                    np.sqrt(np.dot(p, (u - mean) ** 2)),
                    account["weighted_std"][t],
                    projection=True,
                )
                equal(np.ptp(u), account["oscillation"][t], projection=True)
                for key in ("min_id", "max_id"):
                    j = int(account[key][t])
                    equal(u[j], account[key.replace("id", "u")][t], projection=True)
    for row in conditions:
        values = [
            r
            for r in case_rows
            if all(r[k] == row[k] for k in ("axis", "family", "situation", "style"))
        ]
        for metric in ("shape_sum", "calibration_sum", "native_k_sum"):
            high = [float(r[metric]) for r in values if int(r["pole"]) == 1]
            low = [float(r[metric]) for r in values if int(r["pole"]) == -1]
            if len(high) != 3 or len(low) != 3:
                raise ValueError("Incomplete generation replicate support")
            equal(sum(high) / 3 - sum(low) / 3, float(row["delta_" + metric]))
        equal(
            float(row["delta_shape_sum"]) + float(row["delta_calibration_sum"]),
            float(row["delta_native_k_sum"]),
            projection=True,
        )
    pairs = read(out / "pairs.json")
    for pair in pairs:
        if time.monotonic() - start > m["audit_seconds_cap"]:
            raise TimeoutError("Finite120second audit exhausted")
        j = pair["pair_id"]
        with np.load(
            out / "cross_accounts" / f"{j:04d}.npz", allow_pickle=False
        ) as archive:
            a = {k: archive[k] for k in archive.files}
        equal(
            (a["variance_E"] + a["variance_N"]) / 2,
            a["shared_variance"] + a["specific_variance"],
        )
        for t, tr in enumerate(cross_rows[j * 16 : (j + 1) * 16]):
            for key, values in a.items():
                if np.isnan(values[t]):
                    if tr[key]:
                        raise ArithmeticError("Undefined field cosine imputed")
                else:
                    equal(float(tr[key]), values[t])
        if j in (0, 161):
            er, nr = raw(pair["E"]), raw(pair["N"])
            t = 15
            u, v = w.astype(float) @ delta(er)[t], w.astype(float) @ delta(nr)[t]
            midpoint = (
                er["states"][2, t]
                + er["states"][3, t]
                + nr["states"][2, t]
                + nr["states"][3, t]
            ) / 4
            logits = w.astype(float) @ midpoint
            p = np.exp(logits - logsumexp(logits))
            u, v = u - np.dot(p, u), v - np.dot(p, v)
            equal(np.dot(p, u * v), a["covariance"][t], projection=True)
            equal(
                np.dot(p, u * v) / np.sqrt(np.dot(p, u * u) * np.dot(p, v * v)),
                a["cosine"][t],
                projection=True,
            )
    summary = read(out / "summary.json")
    equal(
        summary["max_native_closure_error"],
        max(abs(float(r["closure_error"])) for r in token_rows),
    )
    receipt = {
        "status": "PASS_FINITE_ALTERNATE_NOT_BLIND",
        "account_comparisons": len(errors),
        "max_account_error": max(errors),
        "max_projection_error": max(spot_errors),
        "max_selected_native_error": max(native_errors),
        "full_vocabulary_main_spots": 8,
        "full_vocabulary_cross_spots": 2,
        "every_vocabulary_field_independently_recomputed": False,
        "seconds": time.monotonic() - start,
        "decoder_forwards": 0,
        "scientific_output_seal_sha256": digest(out / "output_seal.json"),
        "verifier_sha256": digest(Path(__file__)),
        "original_scope_complete": False,
    }
    if time.monotonic() - start > m["audit_seconds_cap"]:
        raise TimeoutError("Finite audit cap exceeded;no automatic renewal")
    write(out / "alternate_audit.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    verify(parser.parse_args().out)
