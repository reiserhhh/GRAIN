#!/usr/bin/env python3
"""HM29 independent rational inverse, native score and source-field reconstruction."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_response import write_json

OUT = ROOT / "results/grain_math_response_v1/position_support_verification_r1"
PARENT = ROOT / "results/grain_math_response_v1/position_support_acquisition_r1"


def rational_inverse():
    """Invert the four-position monomial matrix by exact independent elimination."""
    positions = [Fraction(-1), Fraction(-1, 3), Fraction(1, 3), Fraction(1)]
    matrix = [[t**k / 4 for t in positions] for k in range(4)]
    augmented = [
        row + [Fraction(int(k == j)) for j in range(4)] for k, row in enumerate(matrix)
    ]
    for k in range(4):
        pivot = next(j for j in range(k, 4) if augmented[j][k])
        augmented[k], augmented[pivot] = augmented[pivot], augmented[k]
        d = augmented[k][k]
        augmented[k] = [x / d for x in augmented[k]]
        for j in range(4):
            if j != k:
                d = augmented[j][k]
                augmented[j] = [
                    x - d * y for x, y in zip(augmented[j], augmented[k], strict=True)
                ]
    return matrix, [row[4:] for row in augmented]


def score(native, values):
    """Reconstruct digit expectation from the stored native candidate logits."""
    logits = np.asarray(native["candidate_logits"], float)
    exp = np.exp(logits - logits.max())
    return float(sum(p * v for p, v in zip(exp / exp.sum(), values, strict=True)))


def rms(values):
    """Compute finite-family root mean square without a fitted reference."""
    return float(np.sqrt(sum(x * x for x in values) / len(values)))


def verify(parent, out):
    """Check every actual intervention and source-only held transport independently."""
    if out.exists():
        raise FileExistsError("preserve the previous independent reconstruction")
    manifest = read_json(parent / "00/manifest.json")
    spec, alphabet = manifest["spec"], manifest["alphabet"]
    ledger = read_json(parent / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(parent / name) != sha:
            raise ValueError("frozen output changed " + name)
    for name, sha in manifest["files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("frozen external dependency changed " + name)
    design = read_json(parent / "00/design.json")
    source = read_json(parent / "01/source_manifest.json")
    matrix_exact, inverse_exact = rational_inverse()
    matrix, inverse = np.array(matrix_exact, float), np.array(inverse_exact, float)
    null = [Fraction(0), Fraction(-1), Fraction(0), Fraction(1)]
    old_columns = ([1, 0, 0, 0], [1, -1, 1, -1], [1, 1, 1, 1])
    if any(
        sum(Fraction(a) * b for a, b in zip(column, null, strict=True))
        for column in old_columns
    ):
        raise ValueError("null is not orthogonal to old source support")
    native_error, field_error, action_error, metric_error, hook_error = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    baseline, fields = {}, {}
    calls = read_json(parent / "run_status.json")
    if (calls["forward_calls"], calls["backward_calls"], calls["in_flight"]) != (
        336,
        48,
        None,
    ):
        raise ValueError("finite actual-call ledger incomplete")
    with np.load(parent / "00/directions.npz", allow_pickle=False) as data:
        bank = {key: data[key].copy() for key in data.files}
    for row in design:
        p = parent / "01/baselines" / f"unit_{row['unit']:03d}.json"
        meta = read_json(p)
        if (
            meta["address"] != row
            or digest(p.with_suffix(".npz")) != meta["array_sha256"]
        ):
            raise ValueError("baseline address/hash mismatch")
        if len(row["spans"]["A_property"]["positions"]) != 4:
            raise ValueError("actual property span not four tokens")
        native_error = max(
            native_error,
            abs(
                score(meta["native"], alphabet["values"])
                - meta["native"]["conditional_score"]
            ),
        )
        if row["role"] == "source":
            if meta["source_sha256"] is not None:
                raise ValueError("source acquired after fit")
        elif meta["source_sha256"] != digest(parent / "01/source_manifest.json") or any(
            datetime.fromisoformat(meta["acquired_at"])
            <= datetime.fromisoformat(e["frozen_at"])
            for e in source.values()
        ):
            raise ValueError("held observations did not follow source freeze")
        with np.load(p.with_suffix(".npz"), allow_pickle=False) as data:
            baseline[row["unit"]] = {
                "meta": meta,
                "h": data["h"].copy(),
                "g": data["gradient"].astype(float),
            }
    for q, entry in source.items():
        for name, sha in entry["source_hashes"].items():
            if digest(parent / name) != sha:
                raise ValueError("source gradient changed")
        rows = [r for r in design if r["query"] == q and r["role"] == "source"]
        if len(rows) != 8:
            raise ValueError("source family coverage changed")
        mean = sum(baseline[r["unit"]]["g"] for r in rows) / 8
        fields[q] = np.linalg.solve(matrix.T, mean)
        with np.load(parent / entry["file"], allow_pickle=False) as data:
            field_error = max(
                field_error, float(np.max(abs(fields[q] - data["covector"])))
            )
            if int(data["rank"]) != 4:
                raise ValueError("new source did not identify four moment directions")
    with (parent / "02/actions.csv").open() as stream:
        reported = {(int(r["unit"]), r["arm"]): r for r in csv.DictReader(stream)}
    expected = []
    endpoints = 0
    for row in design:
        unit, q = row["unit"], row["query"]
        base = baseline[unit]
        b = bank[q + "_b"] / np.linalg.norm(bank[q + "_b"])
        random = bank["random_b"] / np.linalg.norm(bank["random_b"])
        for arm in spec["arms"]:
            target = (
                np.array([1.0, 0.0, 0.0, 0.0])
                if arm == "known_mean"
                else np.array(null, float)
            )
            hidden = random if arm == "unseen_random" else b
            direction = np.outer(inverse @ target, hidden)
            delta = (
                direction
                / np.linalg.norm(direction)
                * spec["radius_relative_span_norm"]
                * np.linalg.norm(base["h"])
            )
            plus = (base["h"] + delta.astype(np.float32)).astype(np.float32)
            minus = (base["h"] - delta.astype(np.float32)).astype(np.float32)
            half = (plus.astype(float) - minus.astype(float)) / 2
            moments = np.zeros((4, len(b)))
            for k in range(4):
                for j in range(4):
                    moments[k] += matrix[k, j] * half[j]
            scores = {}
            for sign in ("plus", "minus"):
                p = parent / "01/patches" / f"unit_{unit:03d}_{arm}_{sign}.json"
                end = read_json(p)
                scores[sign] = score(end["native"], alphabet["values"])
                native_error = max(
                    native_error, abs(scores[sign] - end["native"]["conditional_score"])
                )
                hook_error = max(hook_error, *end["audit"].values())
                if end["address"] != {"unit": unit, "arm": arm, "sign": sign}:
                    raise ValueError("endpoint address mismatch")
                if end["source_sha256"] != base["meta"]["source_sha256"]:
                    raise ValueError("endpoint source differs from reference")
                endpoints += 1
            action = (scores["plus"] - scores["minus"]) / 2
            even = (scores["plus"] + scores["minus"]) / 2 - score(
                base["meta"]["native"], alphabet["values"]
            )
            predictions = {}
            for name, coef in (
                ("old_joint", bank[q + "_old_joint"]),
                ("old_quotient", bank[q + "_old_quotient"]),
                ("new_source", fields[q]),
            ):
                predictions[name] = float(
                    sum(
                        coef[k, d] * moments[k, d]
                        for k in range(4)
                        for d in range(len(b))
                    )
                )
            predictions["local_derivative"] = float(
                sum(
                    base["g"][j, d] * half[j, d]
                    for j in range(4)
                    for d in range(len(b))
                )
            )
            r = {
                "unit": unit,
                "query": q,
                "role": row["role"],
                "arm": arm,
                "odd_action": action,
                "even_change": even,
                **predictions,
            }
            expected.append(r)
            if (unit, arm) not in reported:
                raise ValueError("missing action")
            for name in ("odd_action", "even_change", *predictions):
                action_error = max(
                    action_error, abs(r[name] - float(reported[unit, arm][name]))
                )
    with (parent / "02/metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    for r in metrics:
        items = [
            x
            for x in expected
            if (x["query"], x["role"], x["arm"]) == (r["query"], r["role"], r["arm"])
        ]
        y = [x["odd_action"] for x in items]
        values = {
            "families": len(items),
            "native_action_rms": rms(y),
            "even_change_rms": rms([x["even_change"] for x in items]),
        }
        for name in ("old_joint", "old_quotient", "new_source", "local_derivative"):
            predictions = [x[name] for x in items]
            values[name + "_rmse"] = rms(
                [p - t for p, t in zip(predictions, y, strict=True)]
            )
            values[name + "_prediction_rms"] = rms(predictions)
            values[name + "_signs"] = sum(
                p * t > 0 for p, t in zip(predictions, y, strict=True)
            )
        for name, v in values.items():
            metric_error = max(metric_error, abs(v - float(r[name])))
    if (len(design), len(expected), endpoints, len(metrics), len(fields)) != (
        48,
        144,
        288,
        27,
        3,
    ):
        raise ValueError("incomplete finite coverage")
    if (
        max(native_error, field_error, action_error, metric_error) > 2e-10
        or hook_error != 0
    ):
        raise ValueError("alternate numerical reconstruction failed")
    out.mkdir(parents=True)
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM29_NOT_GLOBAL",
            "counts": {
                "references": len(design),
                "actions": len(expected),
                "endpoints": endpoints,
                "metrics": len(metrics),
                "fields": len(fields),
            },
            "max_errors": {
                "native": native_error,
                "field": field_error,
                "action": action_error,
                "metric": metric_error,
                "hook": hook_error,
            },
            "exact_O4_inverse": [[str(x) for x in row] for row in inverse_exact],
            "parent_ledger_sha256": digest(parent / "02/artifact_hashes.json"),
            "verifier_sha256": digest(Path(__file__)),
            "independent_algebra_same_agent_not_third_party": True,
            "scope": "finite fixed-reader mechanism; not human personality validity",
        },
    )
    print(read_json(out / "summary.json"), flush=True)


def main():
    """Reconstruct saved outputs without another model call or fitted knob."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.parent, args.out)


if __name__ == "__main__":
    main()
