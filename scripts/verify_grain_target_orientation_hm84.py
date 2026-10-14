#!/usr/bin/env python3
"""Direct-cell and scalar-sum verification of finite HM84 cone/action accounts."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_target_orientation_hm84 import (
    HM83,
    OUT,
    SOURCE,
    check,
    load,
    sha,
    write,
)


def verify():
    """Verify certificates without solving them again or invoking an observer."""
    check()
    for name, digest in load(OUT / "01/seal.json")["files"].items():
        if sha(OUT / name) != digest:
            raise ValueError("output drift:" + name)
    bundle = load(OUT / "00/bundle.json")
    certificates = load(OUT / "00/certificates.json")
    addresses = load(OUT / "00/addresses.json")
    checks, errors = 0, []

    def compare(a, b, tolerance=1e-9):
        nonlocal checks
        aa, bb = np.asarray(a, float), np.asarray(b, float)
        error = float(np.max(abs(aa - bb)))
        if error > tolerance * (1 + float(np.max(abs(bb)))):
            raise ArithmeticError("alternate value mismatch")
        checks += aa.size
        errors.append(error)

    with np.load(SOURCE / "source.npz") as a:
        p, joint, base, c = (
            a["reference"],
            a["joint49_features"],
            a["marginal10_features"],
            a["conditions"],
        )
        arms = {
            "marginal10": base,
            "joint49": joint,
            "anchor_joint50": np.column_stack((joint, c)),
            "conditional_joint99": np.column_stack((joint, c, c[:, None] * joint)),
        }
    with np.load(OUT / "00/source_arrays.npz") as a:
        for arm, b in arms.items():
            compare(a[arm], b)
            direct = np.array(
                [
                    [
                        float(b[r["high"], k]) - float(b[r["low"], k])
                        for k in range(b.shape[1])
                    ]
                    for r in addresses
                ]
            )
            compare(a[arm + "_D"], direct)
    statuses = {}
    for key, cert in certificates.items():
        fn = bundle["functions"][key]
        b = arms[fn["arm"]]
        ix = cert["indices"]
        expected_ix = [
            i
            for i, r in enumerate(addresses)
            if (
                fn["scope"] == "all_contexts" or (r["actor"] == -1 and r["anchor"] == 1)
            )
            and (fn["target"] == "common" or r["focal"] == fn["target"])
        ]
        if ix != expected_ix:
            raise ValueError("support certificate selected wrong rows")
        d = np.array([b[addresses[i]["high"]] - b[addresses[i]["low"]] for i in ix])
        dual = cert["dual"]
        mixture = np.array(
            [
                math.fsum(float(d[i, k]) * dual[i] for i in range(len(d)))
                for k in range(d.shape[1])
            ]
        )
        if min(dual) < -5e-8:
            raise ArithmeticError("negative certificate weight")
        if cert["w"] is None:
            compare(mixture, np.zeros(d.shape[1]), 5e-8)
            compare(math.fsum(dual), 1.0, 5e-8)
        else:
            w = cert["w"]
            margins = [
                math.fsum(float(v) * coef for v, coef in zip(row, w, strict=True))
                for row in d
            ]
            if min(margins) < 1 - 5e-8 or max(abs(mixture)) > 1 + 5e-8:
                raise ArithmeticError("primal/dual inequality mismatch")
            compare(math.fsum(dual), math.fsum(abs(v) for v in w), 5e-8)
            compare(min(margins), cert["minimum_source_margin"], 5e-8)
            source_center = [
                math.fsum(float(p[i]) * float(b[i, k]) for i in range(len(b)))
                for k in range(b.shape[1])
            ]
            compare(
                fn["offset"],
                -math.fsum(v * coef for v, coef in zip(source_center, w, strict=True)),
            )
            checks += len(margins)
        statuses[cert["status"]] = statuses.get(cert["status"], 0) + 1
    records = [
        json.loads(s) for s in (OUT / "01/observations.jsonl").read_text().splitlines()
    ]
    outputs = [
        json.loads(s) for s in (OUT / "01/scores.jsonl").read_text().splitlines()
    ]
    obs = bundle["observation"]
    g = (np.array([r["grain"] for r in records[:120]]) - obs["grain_center"]) / obs[
        "grain_scale"
    ]
    q = (np.array([r["query"] for r in records[:120]]) - obs["query_center"]) / obs[
        "query_scale"
    ]
    joint = np.array(
        [
            [
                *gg,
                *qq,
                *[a * b for a in gg for b in qq],
                *[qq[j] * qq[k] for j in range(6) for k in range(j + 1, 6)],
            ]
            for gg, qq in zip(g, q, strict=True)
        ]
    )
    joint = (joint - obs["feature_center"]) / obs["feature_scale"]
    c = np.array([r["condition"] for r in records[:120]])
    application = {
        "marginal10": np.column_stack((g, q)),
        "joint49": joint,
        "anchor_joint50": np.column_stack((joint, c)),
        "conditional_joint99": np.column_stack((joint, c, c[:, None] * joint)),
    }
    for key, fn in bundle["functions"].items():
        if fn["w"] is None:
            if any(r["functions"][key]["value"] is not None for r in outputs):
                raise ValueError("infeasible direction got invented scores")
            continue
        b = application[fn["arm"]]
        direct = [
            math.fsum(float(v) * coef for v, coef in zip(row, fn["w"], strict=True))
            + fn["offset"]
            for row in b
        ]
        compare([r["functions"][key]["value"] for r in outputs[:120]], direct)
        if (
            fn["arm"] in ("anchor_joint50", "conditional_joint99")
            and outputs[-1]["functions"][key]["value"] is not None
        ):
            raise ValueError("unknown condition imputed")
    metadata = load(HM83 / "00/application.json")
    for name in ("new_focal_actions.csv", "new_other_person_companion_actions.csv"):
        for row in csv.DictReader((OUT / "01" / name).open()):
            key, lo, hi = row["function"], int(row["low"]), int(row["high"])
            l, h = (
                outputs[lo]["functions"][key]["value"],
                outputs[hi]["functions"][key]["value"],
            )
            if l is None:
                if row["action"] != "":
                    raise ValueError("undefined action got zero")
            else:
                compare(float(row["action"]), h - l)
    for row in csv.DictReader((OUT / "01/author_composition.csv").open()):
        key, t = row["function"], row["changed_evidence_target"]
        values = [r["functions"][key]["value"] for r in outputs[:120]]
        if values[0] is None:
            if row["author_score_contrast"] != "":
                raise ValueError("undefined author score got zero")
            continue
        direct = math.fsum(
            (metadata[i]["x"] if metadata[i]["focal"] == t else 0) * v / 120
            for i, v in enumerate(values)
        )
        compare(float(row["author_score_contrast"]), direct)
        compare(float(row["from_mean_action"]), direct)
    audit = load(OUT / "01/cold_audit.json")
    if len(audit["opened"]) != 4 or set(audit["opened"]) != set(audit["allowed"]):
        raise ValueError("cold4file boundary failed")
    result = {
        "status": "VERIFIED_FINITE_HM84_NOT_GLOBAL",
        "source_cases": len(certificates),
        "source_status_counts": statuses,
        "alternate_values": checks,
        "max_absolute_error": max(errors),
        "cold_files": 4,
        "new_calls": 0,
        "same_agent_not_independent_rebuilder": True,
        "original_scope_complete": False,
    }
    write(OUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        verify()
