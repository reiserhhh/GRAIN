#!/usr/bin/env python3
"""Separate common endorsement from old item shape,without new ranks or fits."""

import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_item_configuration as M
from grain_research.construction import math_output_observability as O
from scripts import run_grain_observed_family_hm170 as R
from scripts.run_grain_item_configuration_hm286 import PARENT, SPEC, parents
from scripts.score_grain_author_belief_cold import load_bundle

PRIMARY = ROOT / SPEC["output"]
DEFAULT = PRIMARY / "common_account_followup_r1"
SCIENCE = (
    "docs/GRAIN_ITEM_COMMON_ACCOUNT_FOLLOWUP_HM286.md",
    "scripts/describe_grain_item_common_hm286.py",
    "tests/test_grain_item_common_hm286.py",
    "grain_research/construction/math_item_configuration.py",
)
NAMES = ("full_item", "keyed_line", "common_endorsement", "within_pole_pattern")


def parts4(frame, direction, signs):
    """Preserve fixed key/common/within directions,not learned nuisance removal."""
    z, v, sign = (np.asarray(x, float) for x in (frame, direction, signs))
    old = M.split_frame(z, v)
    u = v * sign
    if sign.shape != v.shape or not np.isin(sign, (-1, 1)).all() or abs(u @ v) > 1e-12:
        raise ValueError("The unchanged balanced-pole orthogonal directions required")
    common = (z @ u)[..., None] * u
    return np.stack((z, old[1], common, old[2] - common))


def expired(_signal, _frame):
    """Enforce one finite descriptor run without outcome-dependent expansion."""
    raise TimeoutError("Original30-second common-account descriptor cap")


def main():
    """Seal the distinct descriptive question and conserve all primary contacts."""
    parents()
    R.check(PRIMARY, ("computed_seal.json", "verification_seal.json"))
    out = DEFAULT
    out.mkdir(parents=True, exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {
                str((base / p).relative_to(ROOT)): R.digest(base / p)
                for base, names in (
                    (
                        PRIMARY,
                        (
                            "joint_responses.npz",
                            "bank.json",
                            "accounts.npz",
                            "manifest.json",
                            "computed_seal.json",
                            "verification_seal.json",
                        ),
                    ),
                    (PARENT, ("source.npz",)),
                )
                for p in names
            },
            "post_outcome_descriptive_only": True,
            "new_ranks_or_assignments_or_fit": False,
            "seconds_cap": 30,
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "question.json", {"keep_all4_parts": True, "no_primary_replacement": True}
    )
    R.seal(out, "prepared_seal.json", ("question.json",))
    source = load_bundle(PARENT / "source.npz")
    bank, joint = (
        R.read(PRIMARY / "bank.json"),
        load_bundle(PRIMARY / "joint_responses.npz")["joint"],
    )
    old = load_bundle(PRIMARY / "accounts.npz")["accounts"]
    left, right = np.arange(0, 48, 2), np.arange(1, 48, 2)
    e = source["labels"] - source["means"]
    account = np.empty((2, 3, 2, 5, 5, 4, 3))
    rows, errors = [], 0.0
    begin = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(30)
    try:
        with threadpool_limits(limits=1):
            for sp, space in enumerate(SPEC["spaces"]):
                for ref, reference in enumerate(SPEC["references"]):
                    d = source["weights"][sp, ref].mean(1)
                    a, b = (O.centered_scores((d[ix] @ e).T) for ix in (left, right))
                    for op, operation in enumerate(SPEC["operations"]):
                        for q, target in enumerate(SPEC["targets"]):
                            for r, query in enumerate(SPEC["targets"]):
                                z, v, ids = M.item_frame(joint[:, op], bank, r)
                                signs = np.array([bank[i]["pole"] for i in ids])
                                l, rr = (
                                    parts4(z[left], v, signs),
                                    parts4(z[right], v, signs),
                                )
                                k = np.einsum("pid,pjd->pij", l, rr)
                                for p, name in enumerate(NAMES):
                                    values = O.cross_author_accounts(a[q], b[q], k[p])
                                    account[sp, ref, op, q, r, p] = [
                                        values[t]
                                        for t in (
                                            "all_pairs",
                                            "same_author",
                                            "distinct_authors",
                                        )
                                    ]
                                    rows.append(
                                        {
                                            "space": space,
                                            "reference": reference,
                                            "operation": operation,
                                            "source_target": target,
                                            "query": query,
                                            "part": name,
                                            **{t: float(x) for t, x in values.items()},
                                        }
                                    )
                                block = account[sp, ref, op, q, r]
                                previous = old[sp, ref, op, 2, 2, q, r]
                                errors = max(
                                    errors,
                                    float(abs(block[:2] - previous[:2]).max()),
                                    float(abs(block[2] + block[3] - previous[2]).max()),
                                )
            selected = [
                {
                    "target": target,
                    "full_U": float(account[1, 2, 0, q, q, 0, 2]),
                    "key_U": float(account[1, 2, 0, q, q, 1, 2]),
                    "common_U": float(account[1, 2, 0, q, q, 2, 2]),
                    "within_pole_U": float(account[1, 2, 0, q, q, 3, 2]),
                }
                for q, target in enumerate(SPEC["targets"])
            ]
            if errors > 1e-12:
                raise ArithmeticError("Old scientific contacts changed")
            R.table(out / "all_common_accounts.csv", rows)
            R.archive(out / "accounts.npz", accounts=account)
            R.write(
                out / "summary.json",
                {
                    "matching_descriptions": selected,
                    "seconds": time.monotonic() - begin,
                    "old_contact_and_shape_closure_error": errors,
                    "new_ranks_or_assignments_or_fit": False,
                    "post_outcome_descriptive_only": True,
                    "new_neural_API_private_training": 0,
                    "same_agent_not_blind": True,
                    "original_scope_complete": False,
                },
            )
        R.seal(
            out,
            "completed_seal.json",
            ("all_common_accounts.csv", "accounts.npz", "summary.json"),
        )
        print(json.dumps(R.read(out / "summary.json")))
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
