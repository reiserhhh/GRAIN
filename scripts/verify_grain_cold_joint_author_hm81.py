#!/usr/bin/env python3
"""Alternate cold arithmetic and result-after within/between-condition accounts."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_grain_cold_joint_author_hm81 import (
    OUT,
    check,
    digest,
    read_json,
    write,
)
from scripts.run_grain_math_response import write_csv


def main():
    """Check actual aggregates; do not rerun source fitting or model observations."""
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    check(out)
    for seal in (out / "01/seal.json", out / "03/seal.json"):
        for name, expected in read_json(seal)["files"].items():
            if digest(out / name) != expected:
                raise ValueError("sealed result drift:" + name)
    if read_json(out / "run_status.json")["calls"] != 384:
        raise ValueError("all384 actual calls required")
    companion = out / "02_entrypoint_repair"
    entry = read_json(companion / "manifest.json")
    if (
        digest(Path(entry["live_code"])) != entry["code_sha256"]
        or digest(companion / Path(entry["live_code"]).name)
        != entry["frozen_code_sha256"]
    ):
        raise ValueError("entrypoint correction drift")
    bundle = read_json(out / "00/bundle.json")
    records = [
        json.loads(line)
        for line in (out / "03/observations.jsonl").read_text().splitlines()
    ]
    scores = {
        r["author_key"]: r
        for r in (
            json.loads(line)
            for line in (out / "03/author_profiles.jsonl").read_text().splitlines()
        )
    }
    audit = read_json(out / "03/cold_file_audit.json")
    permitted = {
        str((out / n).resolve())
        for n in (
            "00/bundle.json",
            "03/observations.jsonl",
            "03/author_profiles.jsonl",
            "03/cold_file_audit.json",
        )
    }
    if set(audit["opened"]) - permitted or set(audit["allowed"]) != permitted:
        raise ValueError("unexpected cold process file access")
    errors, rows, features = [], [], {}
    for rec in records:
        event_features = []
        for event in rec["events"]:
            g = [
                (v - m) / s
                for v, m, s in zip(
                    event["grain"],
                    bundle["grain_center"],
                    bundle["grain_scale"],
                    strict=True,
                )
            ]
            q = [
                (v - m) / s
                for v, m, s in zip(
                    event["query"],
                    bundle["query_center"],
                    bundle["query_scale"],
                    strict=True,
                )
            ]
            raw = [
                *g,
                *q,
                *[a * b for a in g for b in q],
                *[q[j] * q[k] for j in range(6) for k in range(j + 1, 6)],
            ]
            event_features.append(
                [
                    (v - m) / s
                    for v, m, s in zip(
                        raw,
                        bundle["feature_center"],
                        bundle["feature_scale"],
                        strict=True,
                    )
                ]
            )
        value = [
            math.fsum(v[j] for v in event_features) / len(event_features)
            for j in range(49)
        ]
        errors.extend(
            abs(a - b)
            for a, b in zip(value, scores[rec["author_key"]]["profile"], strict=True)
        )
        features[rec["author_key"]] = np.asarray(event_features)
        g = (
            np.asarray([e["grain"] for e in rec["events"]]) - bundle["grain_center"]
        ) / bundle["grain_scale"]
        q = (
            np.asarray([e["query"] for e in rec["events"]]) - bundle["query_center"]
        ) / bundle["query_scale"]
        within, between = np.zeros((4, 6)), np.zeros((4, 6))
        for condition in sorted({e["condition"] for e in rec["events"]}):
            mask = np.asarray([e["condition"] == condition for e in rec["events"]])
            gg, qq = g[mask], q[mask]
            weight = mask.mean()
            within += weight * (gg - gg.mean(0)).T @ (qq - qq.mean(0)) / len(gg)
            between += weight * np.outer(gg.mean(0) - g.mean(0), qq.mean(0) - q.mean(0))
        total = np.asarray(scores[rec["author_key"]]["grain_query_covariance"])
        errors.extend(abs(within + between - total).ravel())
        rows.append(
            {
                "author": rec["author_key"],
                "within_condition_covariance_RMS": float(np.sqrt(np.mean(within**2))),
                "between_condition_covariance_RMS": float(np.sqrt(np.mean(between**2))),
                "total_covariance_RMS": float(np.sqrt(np.mean(total**2))),
                "within_between_cross_inner_product": float(np.mean(within * between)),
                "not_variance_fractions_or_personal_response_operators": True,
            }
        )
    for a, b in itertools.combinations(features, 2):
        x, y = features[a], features[b]
        direct = np.sum((x.mean(0) - y.mean(0)) ** 2)
        gram = (x @ x.T).mean() + (y @ y.T).mean() - 2 * (x @ y.T).mean()
        errors.append(abs(direct - gram))
    if max(errors) > 1e-10:
        raise ArithmeticError("alternate arithmetic mismatch")
    dst = out / "04_condition_accounts"
    dst.mkdir(exist_ok=False)
    write_csv(dst / "within_between_condition.csv", rows)
    write(
        dst / "verification.json",
        {
            "status": "VERIFIED_ALTERNATE_HM81_FINITE_INTERFACE",
            "scalar_checks": len(errors),
            "max_error": max(errors),
            "cold_opened_files": sorted(audit["opened"]),
            "model_calls_fits": [0, 0],
            "condition_accounts_result_after": True,
            "same_agent_not_independent_rebuilder": True,
            "code_sha256": digest(Path(__file__).resolve()),
            "original_scope_complete": False,
        },
    )
    print(json.dumps(read_json(dst / "verification.json")), flush=True)


if __name__ == "__main__":
    main()
