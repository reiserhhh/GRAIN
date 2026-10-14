#!/usr/bin/env python3
"""Alternate QR, conditional sums and raw-decoder checks for finite HM83."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_author_contrast_hm82 import load, sha, write
from scripts.run_grain_condition_order_hm83 import OUT, SOURCE, check


def verify(out):
    """Recompute using different formulas; not an independent human re-builder."""
    manifest = check(out)
    for path in (out / "01/seal.json", out / "03/seal.json"):
        for name, expected in load(path)["files"].items():
            if sha(out / name) != expected:
                raise ValueError("output seal drift:" + name)
    errors, checks = [], 0

    def compare(actual, expected):
        nonlocal checks
        actual, expected = np.asarray(actual), np.asarray(expected)
        errors.append(float(np.max(abs(actual - expected))))
        checks += actual.size

    bundle = load(out / "00/bundle.json")
    accounts = list(csv.DictReader((out / "00/order_accounts.csv").open()))
    with np.load(SOURCE / "source.npz") as a:
        p, t, c = a["reference"], a["targets"], a["conditions"]
        arms = {
            "marginal10": a["marginal10_features"],
            "joint49": a["joint49_features"],
            "anchor_joint50": np.column_stack((a["joint49_features"], c)),
            "conditional_joint99": a["conditional_joint99_features"],
        }
    root = np.sqrt(p)
    E = np.column_stack((root, root * c))
    E = E @ E.T
    for account in accounts:
        name = account["arm"]
        b = arms[name]
        Q, _ = np.linalg.qr(
            root[:, None] * np.column_stack((np.ones(len(p)), b)), mode="reduced"
        )
        P = Q @ Q.T
        compiled = bundle["functions"][name]
        value = b @ compiled["beta"] + compiled["offset"]
        compare(value, (P @ (root[:, None] * t)) / root[:, None])
        pc = (P @ (root * c)) / root
        choice = np.empty(len(p))
        for sign in (-1, 1):
            ix = c == sign
            choice[ix] = sum(p[k] * pc[k] for k in np.flatnonzero(ix)) / sum(p[ix])
        v = sum(p[k] * pc[k] ** 2 for k in range(len(p)))
        leakage = sum(p[k] * (pc[k] - choice[k]) ** 2 for k in range(len(p)))
        compare(
            [
                v,
                leakage,
                np.linalg.norm(E @ P - P @ E),
                np.linalg.norm(c[:, None] * P - P * c[None, :]),
            ],
            [
                float(account[k])
                for k in (
                    "c_visibility",
                    "choice_to_within_norm2",
                    "E_commutator_Frobenius",
                    "M_commutator_Frobenius",
                )
            ],
        )
        compare(leakage, v * (1 - v))
        compare(choice, v * c)
    records = [
        json.loads(s) for s in (out / "03/observations.jsonl").read_text().splitlines()
    ]
    outputs = [
        json.loads(s) for s in (out / "03/scores.jsonl").read_text().splitlines()
    ]
    with np.load(out / "03/terminal_arrays.npz") as a:
        g, q, truth = a["grain"], a["query"], a["truth"]
    obs = bundle["observation"]
    gc, qc = (
        (g - obs["grain_center"]) / obs["grain_scale"],
        (q - obs["query_center"]) / obs["query_scale"],
    )
    joint = np.array(
        [
            [
                *gg,
                *qq,
                *[aa * bb for aa in gg for bb in qq],
                *[qq[j] * qq[k] for j in range(6) for k in range(j + 1, 6)],
            ]
            for gg, qq in zip(gc, qc, strict=True)
        ]
    )
    joint = (joint - obs["feature_center"]) / obs["feature_scale"]
    raw = {
        "marginal10": np.column_stack((gc, qc)),
        "joint49": joint,
        "anchor_joint50": np.column_stack((joint, truth[:, 0])),
        "conditional_joint99": np.column_stack(
            (joint, truth[:, 0], truth[:, 0, None] * joint)
        ),
    }
    for arm, b in raw.items():
        values = (
            b @ bundle["functions"][arm]["beta"] + bundle["functions"][arm]["offset"]
        )
        compare(values, [r["arms"][arm]["scores"] for r in outputs[:120]])
    for i in range(720):
        native = load(out / f"01/units/unit_{i:04d}.json")["native"]
        logits = native["candidate_logits"]
        weight = [np.exp(s - max(logits)) for s in logits]
        value = sum(
            w * v for w, v in zip(weight, manifest["alphabet"]["values"], strict=True)
        ) / sum(weight)
        compare(value, q[i // 6, i % 6])
    if len(records) != 121 or len(outputs) != 121:
        raise ValueError("new event output count mismatch")
    if any(
        outputs[-1]["arms"][a]["scores"] is not None
        for a in ("anchor_joint50", "conditional_joint99")
    ):
        raise ValueError("unknown condition imputed")
    audit = load(out / "03/cold_audit.json")
    if len(audit["opened"]) != 4 or audit["opened"] != audit["allowed"]:
        raise ValueError("cold file boundary mismatch")
    if max(errors) > 1e-10:
        raise ArithmeticError("alternate computation mismatch")
    result = {
        "status": "VERIFIED_FINITE_HM83_NOT_GLOBAL",
        "alternate_scalar_checks": checks,
        "max_error": max(errors),
        "actual_new_reads": 720,
        "cold_files": 4,
        "same_agent_alternate_not_independent_rebuilder": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(json.dumps(result), flush=True)


def main():
    """Verify one frozen packet without new inference or fitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(1):
        verify(args.out)


if __name__ == "__main__":
    main()
