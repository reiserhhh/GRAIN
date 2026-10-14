#!/usr/bin/env python3
"""One source-signed author-local response test on frozen HM284 observations."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_criterion_components as C
from grain_research.construction import math_response_orientation as M
from grain_research.construction import math_source_anchor_exchange as E
from scripts import run_grain_observed_family_hm170 as R
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_response_orientation_hm285.json"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_RESPONSE_ORIENTATION_PLAN_HM285.md",
    "grain_research/construction/math_response_orientation.py",
    "scripts/run_grain_response_orientation_hm285.py",
    "scripts/score_grain_response_orientation_cold.py",
    "tests/test_grain_response_orientation.py",
    "grain_research/construction/math_output_observability.py",
    "grain_research/construction/math_criterion_components.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "scripts/run_grain_observed_family_hm170.py",
)
PARENT = ROOT / R.read(ROOT / CONFIG)["parent"]
DEFAULT = ROOT / R.read(ROOT / CONFIG)["output"]


def parent_check():
    """Authenticate both prior results, without rerunning their science."""
    R.check(
        PARENT, ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    R.check(PARENT / "author_local_followup_r1", ("completed_seal.json",))


def prepare(out):
    """Bind source, same-event evidence and the new question before computing it."""
    parent_check()
    spec = R.read(ROOT / CONFIG)
    source, geom = (
        load_bundle(PARENT / "source.npz"),
        load_bundle(PARENT / "geometry.npz"),
    )
    if (
        source["weights"].shape != (2, 3, 48, 2, 386)
        or source["assignments"].shape != (5, 199, 386)
        or geom["grams"].shape != (4, 480, 480)
        or not np.array_equal(source["windows"], np.tile([0, 1], 24))
    ):
        raise ValueError(
            "Original complete SOURCE/panel/assignment identities required"
        )
    bindings = R.read(PARENT / "event_bindings.json")
    for u in range(24):
        a, b = bindings[2 * u], bindings[2 * u + 1]
        if a["author"] != b["author"] or set(a["comment_ids"]) & set(b["comment_ids"]):
            raise ValueError("Same author with disjoint observed comments required")
    out.mkdir(parents=True, exist_ok=False)
    inputs = [
        PARENT / name
        for name in (
            "source.npz",
            "geometry.npz",
            "event_bindings.json",
            "cold_record.json",
            "manifest.json",
            "prepared_seal.json",
            "computed_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        )
    ]
    follow = PARENT / "author_local_followup_r1"
    inputs.extend(follow / name for name in ("manifest.json", "completed_seal.json"))
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in inputs},
            "post_outcome_new_question_not_previous_primary_replacement": True,
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "registered_question.json",
        {
            "primary": spec["primary"],
            "all_five_targets": spec["targets"],
            "two_sided_old_assignment_ranks": True,
            "centering_is_finite_covariance_not_text_denoising": True,
            "model_reader_response_not_human_author_response": True,
            "application_psychological_labels_used": False,
            "no_metric_or_weight_fit": True,
        },
    )
    R.seal(out, "prepared_seal.json", ("registered_question.json",))
    print(json.dumps({"status": "PREPARED", "source": 386, "panel": 24}))


def expired(_signal, _frame):
    """Terminate the single finite computation without extending its budget."""
    raise TimeoutError("Original60-second orientation cap")


def compute(out):
    """Compute all retained signed accounts and only the fixed five primary ranks."""
    R.check(out)
    if (out / "computed_seal.json").exists() or (out / "failed_compute.json").exists():
        raise FileExistsError("No scientific computation retry")
    z, geometry = (
        load_bundle(PARENT / "source.npz"),
        load_bundle(PARENT / "geometry.npz"),
    )
    config = R.read(ROOT / CONFIG)
    left, right = np.arange(0, 48, 2), np.arange(1, 48, 2)
    values = (z["labels"], z["means"], z["labels"] - z["means"])
    accounts = np.empty((2, 3, 2, 3, 5, 5, 4))
    rows, primaries, nulls, pullbacks, terms = [], [], [], [], []
    began = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(config["seconds_cap"])
    try:
        with threadpool_limits(limits=1):
            for sp, space in enumerate(config["source_spaces"]):
                for ref, reference in enumerate(config["references"]):
                    d = z["weights"][sp, ref].mean(1)
                    a = np.stack([d[left] @ v for v in values])
                    b = np.stack([d[right] @ v for v in values])
                    for op, operation in enumerate(config["operations"]):
                        for q, target in enumerate(config["targets"]):
                            for r, query in enumerate(config["targets"]):
                                i0, i1 = left * 10 + op * 5 + r, right * 10 + op * 5 + r
                                for ch, channel in enumerate(config["channels"]):
                                    k = geometry["grams"][ch, i0, i1]
                                    t = M.oriented_accounts(a[:, :, q], b[:, :, q], k)
                                    accounts[sp, ref, op, :, q, r, ch] = t
                                    row = {
                                        "space": space,
                                        "reference": reference,
                                        "operation": operation,
                                        "source_target": target,
                                        "query": query,
                                        "channel": channel,
                                        "total": float(t[0]),
                                        "composition": float(t[1]),
                                        "within_configuration": float(t[2]),
                                    }
                                    rows.append(row)
                                    if (sp, ref, op, ch) != (1, 2, 0, 0) or q != r:
                                        continue
                                    ell = M.source_pullback(d[left], d[right], k)
                                    e = values[2][:, q]
                                    en = e[z["assignments"][q]]
                                    null = M.oriented_accounts(
                                        en @ d[left].T, en @ d[right].T, k
                                    )
                                    variance = M.orbit_variance(
                                        ell,
                                        C.orbit_covariance(
                                            z["labels"][:, q], z["groups"][:, q]
                                        ),
                                    )
                                    primaries.append(
                                        {
                                            "target": target,
                                            "signed_contact": float(t[2]),
                                            "exact_conditional_mean": 0.0,
                                            "exact_conditional_variance": variance,
                                            "old_reference_mean": float(null.mean()),
                                            "two_sided_conditional_rank": float(
                                                (1 + np.sum(abs(null) >= abs(t[2])))
                                                / 200
                                            ),
                                        }
                                    )
                                    nulls.append(null)
                                    pullbacks.append(ell)
                                    c = (a[2, :, q] + b[2, :, q]) / 2
                                    terms.append((c - c.mean()) * k / 24)
            for row, adjusted in zip(
                primaries,
                E.holm([r["two_sided_conditional_rank"] for r in primaries]),
                strict=True,
            ):
                row["Holm5_rank"] = float(adjusted)
            channel_error = float(
                np.max(abs(accounts[..., 0] - accounts[..., 1:].sum(-1)))
            )
            source_error = float(
                np.max(
                    abs(
                        accounts[:, :, :, 0]
                        - accounts[:, :, :, 1]
                        - accounts[:, :, :, 2]
                    )
                )
            )
            R.table(out / "all_oriented_contacts.csv", rows)
            R.table(out / "primary.csv", primaries)
            R.archive(
                out / "accounts.npz",
                accounts=accounts,
                nulls=np.asarray(nulls),
                pullbacks=np.asarray(pullbacks),
                author_terms=np.asarray(terms),
            )
            R.write(out / "cold_record.json", R.read(PARENT / "cold_record.json"))
            R.write(
                out / "summary.json",
                {
                    "primary": primaries,
                    "compute_seconds": time.monotonic() - began,
                    "all_retained_accounts": len(rows),
                    "old_assignments_per_target": 199,
                    "channel_closure_error": channel_error,
                    "source_component_closure_error": source_error,
                    "new_neural_API_private_training": 0,
                    "all_targets_same_support_no_fit": True,
                    "model_reader_response_not_author_behavior": True,
                    "post_outcome_new_question": True,
                    "original_scope_complete": False,
                },
            )
        R.seal(
            out,
            "computed_seal.json",
            (
                "all_oriented_contacts.csv",
                "primary.csv",
                "accounts.npz",
                "cold_record.json",
                "summary.json",
            ),
        )
        print(json.dumps(R.read(out / "summary.json")))
    except Exception as error:
        R.write(
            out / "failed_compute.json",
            {"error": repr(error), "seconds": time.monotonic() - began},
        )
        raise
    finally:
        signal.alarm(0)


def verify(out):
    """Literal author sums, SOURCE orbit variance, sign reversal and cold readback."""
    parent_check()
    R.check(out, ("computed_seal.json",))
    source, geom = (
        load_bundle(PARENT / "source.npz"),
        load_bundle(PARENT / "geometry.npz"),
    )
    data = load_bundle(out / "accounts.npz")
    errors = {
        "literal_references": 0.0,
        "source_pullback": 0.0,
        "variance": 0.0,
        "real_sign_reversal": 0.0,
    }
    left, right = np.arange(0, 48, 2), np.arange(1, 48, 2)
    d = source["weights"][1, 2].mean(1)
    with threadpool_limits(limits=1):
        for q in range(5):
            k = geom["grams"][0, left * 10 + q, right * 10 + q]
            e = source["labels"][:, q] - source["means"][:, q]
            assignments = np.vstack((np.arange(386), source["assignments"][q]))
            values = np.r_[data["accounts"][1, 2, 0, 2, q, q, 0], data["nulls"][q]]
            for assignment, saved in zip(assignments, values, strict=True):
                a, b = d[left] @ e[assignment], d[right] @ e[assignment]
                c = (a + b) / 2
                direct = sum(float((c[u] - c.mean()) * k[u]) for u in range(24)) / 24
                errors["literal_references"] = max(
                    errors["literal_references"], abs(direct - saved)
                )
                errors["source_pullback"] = max(
                    errors["source_pullback"],
                    abs(data["pullbacks"][q] @ e[assignment] - saved),
                )
            variance = 0.0
            for group in np.unique(source["groups"][:, q]):
                ix = source["groups"][:, q] == group
                ell, y = data["pullbacks"][q, ix], source["labels"][ix, q]
                m, p = len(y), y.mean()
                if m > 1:
                    variance += (
                        p
                        * (1 - p)
                        * m
                        / (m - 1)
                        * float(sum(ell**2) - sum(ell) ** 2 / m)
                    )
                if np.max(abs(e[ix].mean())) > 1e-12:
                    raise ArithmeticError("SOURCE residual not conditionally centered")
            expected = R.read(out / "summary.json")["primary"][q][
                "exact_conditional_variance"
            ]
            errors["variance"] = max(errors["variance"], abs(variance - expected))
            forward = M.oriented_accounts(d[left] @ e, d[right] @ e, k)
            reverse = M.oriented_accounts(-d[left] @ e, -d[right] @ e, k)
            errors["real_sign_reversal"] = max(
                errors["real_sign_reversal"], abs(forward + reverse)
            )
    reply = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_response_orientation_cold.py"),
            "--input",
            str(out / "cold_record.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cold = json.loads(reply.stdout)
    cold_values = np.asarray(cold["oriented_source_query_channels"])
    errors["cold"] = float(np.max(abs(cold_values - data["accounts"][1, 2, 0, 2])))
    summary = R.read(out / "summary.json")
    errors["channel_closure"] = summary["channel_closure_error"]
    errors["source_component_closure"] = summary["source_component_closure_error"]
    if max(errors.values()) > 1e-12:
        raise ArithmeticError("Registered adequate account tolerance failed")
    R.write(out / "cold_readback.json", cold)
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "errors": errors,
            "literal_primary_and_references": 1000,
            "same_agent_not_blind": True,
            "no_new_scientific_collection": True,
        },
    )
    R.seal(out, "verification_seal.json", ("verification.json", "cold_readback.json"))
    print(json.dumps(R.read(out / "verification.json")))


def main():
    """Run a single immutable stage; check and verify never rerun scientific outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        parent_check()
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print(json.dumps({"status": "ALL_OLD_AND_NEW_SEALS_PASS"}))
    else:
        {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
