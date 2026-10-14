#!/usr/bin/env python3
"""Direct finite sums and genuine distribution pairs audit the cone certificate."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_ordered_valuation_hm179 as H


def direct_status(values, resolution):
    """Classify explicit generator sums independently of the scorer."""
    positive = sum(t > resolution for t in values)
    negative = sum(t < -resolution for t in values)
    if positive and negative:
        return "SCALE_DEPENDENT"
    if positive:
        return "ORDINALLY_NONNEGATIVE"
    if negative:
        return "ORDINALLY_NONPOSITIVE"
    return "NUMERICALLY_UNRESOLVED_ZERO"


def main():
    """Recompute all actual endpoints/certificates without cone-scoring functions."""
    out = H.DEFAULT
    spec = R.check(out, ("analysis_seal.json",))["config"]
    z = R.prepared(out)
    with np.load(out / "order_accounts.npz", allow_pickle=False) as a:
        saved = {k: a[k] for k in a.files}
    p = np.empty((24, 2, 5, 10))
    for u, v, subset in np.ndindex(24, 2, 5):
        logits = z["native_logits"][u, v, subset]
        values = [math.exp(float(t - max(logits))) for t in logits]
        den = math.fsum(values)
        p[u, v, subset] = [t / den for t in values]
    native = np.empty((24, 2, 10))
    tails = np.empty((24, 2, 9))
    for u, v, j in np.ndindex(24, 2, 10):
        native[u, v, j] = math.fsum(
            (p[u, v, 2, j], -p[u, v, 1, j], -p[u, v, 3, j], p[u, v, 4, j])
        )
    for u, v, k in np.ndindex(24, 2, 9):
        tails[u, v, k] = math.fsum(native[u, v, k + 1 :])
    errors = {
        "candidate_probability": float(np.max(np.abs(p - saved["probabilities"]))),
        "native_signed_measure": float(
            np.max(np.abs(native - saved["native_measure"]))
        ),
        "native_tails": float(np.max(np.abs(tails - saved["native_tails"]))),
        "positive_distribution": float(
            np.max(
                np.abs((p[:, :, 2] + p[:, :, 4]) / 2 - saved["positive_distribution"])
            )
        ),
        "negative_distribution": float(
            np.max(
                np.abs((p[:, :, 1] + p[:, :, 3]) / 2 - saved["negative_distribution"])
            )
        ),
        "pair_distribution_identity": float(
            np.max(
                np.abs(
                    2
                    * (saved["positive_distribution"] - saved["negative_distribution"])
                    - saved["native_measure"]
                )
            )
        ),
    }
    keys, direct = {}, {}
    for arm, component, target in np.ndindex(2, 5, 5):
        source = z["components"][arm, :, :, component, target]
        for scope, (x, m) in enumerate(H.scopes(source, native)):
            key = (
                spec["arms"][arm],
                spec["scopes"][scope],
                spec["components"][component],
                spec["targets"][target],
            )
            norm = math.sqrt(math.fsum(float(t * t) for t in x))
            if norm <= spec["resolution"]:
                keys[key] = None
                continue
            b = [
                math.fsum(float(x[u] * m[u, j]) for u in range(24)) / norm
                for j in range(10)
            ]
            ts = [math.fsum(b[j] for j in range(k, 10)) for k in range(1, 10)]
            keys[key] = (b, ts)
            direct[key] = (x, m)
    pooled_error = 0.0
    status_errors = 0
    with (out / "pooled_certificates.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        key = tuple(row[k] for k in ("arm", "scope", "component", "target"))
        result = keys[key]
        if result is None:
            status_errors += row["status"] != "UNRESOLVED_SOURCE_ZERO"
            continue
        b, ts = result
        checks = [abs(float(row[f"tail_{k + 1}"]) - ts[k]) for k in range(9)]
        checks += [
            abs(float(row["lower"]) - min(ts)),
            abs(float(row["upper"]) - max(ts)),
            abs(
                float(row["equally_spaced"])
                - math.fsum(j * b[j] / 9 for j in range(10))
            ),
        ]
        pooled_error = max(pooled_error, *checks)
        status_errors += direct_status(ts, spec["resolution"]) != row["status"]
        index = tuple(
            spec[k].index(row[r])
            for k, r in (
                ("arms", "arm"),
                ("scopes", "scope"),
                ("components", "component"),
                ("targets", "target"),
            )
        )
        pooled_error = max(
            pooled_error,
            float(np.max(np.abs(np.asarray(b) - saved["pooled_measure"][index]))),
        )
    individual_error = 0.0
    with (out / "individual_certificates.csv").open() as f:
        individuals = list(csv.DictReader(f))
    for row in individuals:
        key = tuple(row[k] for k in ("arm", "scope", "component", "target"))
        if keys[key] is None:
            status_errors += row["status"] != "UNRESOLVED_SOURCE_ZERO"
            continue
        x, m = direct[key]
        u = int(row["panel_index"])
        if row["author_id"] != str(z["author_ids"][u]):
            raise ValueError("Changed author identity in individual certificate")
        if abs(x[u]) <= spec["resolution"]:
            status_errors += row["status"] != "UNRESOLVED_SOURCE_ZERO"
            continue
        sign = 1 if x[u] > 0 else -1
        ts = [
            math.fsum(sign * float(m[u, j]) for j in range(k, 10)) for k in range(1, 10)
        ]
        status_errors += direct_status(ts, spec["resolution"]) != row["status"]
        individual_error = max(
            individual_error,
            *(abs(float(row[f"tail_{k + 1}"]) - ts[k]) for k in range(9)),
        )
        individual_error = max(
            individual_error,
            abs(float(row["lower"]) - min(ts)),
            abs(float(row["upper"]) - max(ts)),
        )
    errors["pooled_certificates"] = pooled_error
    errors["individual_certificates"] = individual_error
    if (
        max(errors.values()) > spec["resolution"]
        or status_errors
        or len(rows) != 150
        or len(individuals) != 3600
    ):
        raise ValueError("Actual finite function-class certificate disagrees")
    receipt = {
        "candidate_distributions": 240,
        "pooled_certificates": len(rows),
        "individual_certificates": len(individuals),
        "errors": errors,
        "status_errors": status_errors,
        "same_agent_not_blind": True,
        "selected_utility_fit": False,
        "new_neural_calls": 0,
    }
    R.write(out / "alternate_audit.json", receipt)
    R.seal(out, "audit_seal.json", ("alternate_audit.json",))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
