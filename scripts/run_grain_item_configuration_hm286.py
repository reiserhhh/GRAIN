#!/usr/bin/env python3
"""One fixed signed-item configuration discriminator on HM280/284 evidence."""

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
from grain_research.construction import math_item_configuration as M
from grain_research.construction import math_output_observability as O
from grain_research.construction import math_source_anchor_exchange as E
from scripts import run_grain_observed_family_hm170 as R
from scripts.score_grain_author_belief_cold import load_bundle

CONFIG = "configs/grain_item_configuration_hm286.json"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_ITEM_CONFIGURATION_PLAN_HM286.md",
    "grain_research/construction/math_item_configuration.py",
    "scripts/run_grain_item_configuration_hm286.py",
    "scripts/score_grain_item_configuration_cold.py",
    "tests/test_grain_item_configuration.py",
    "grain_research/construction/math_output_observability.py",
    "grain_research/construction/math_criterion_components.py",
    "grain_research/construction/math_source_anchor_exchange.py",
    "scripts/run_grain_observed_family_hm170.py",
)
SPEC = R.read(ROOT / CONFIG)
PARENT, READER, DEFAULT = (ROOT / SPEC[k] for k in ("parent", "reader", "output"))
KINDS = ("all_pairs", "same_author", "distinct_authors")


def parents():
    """Read back old complete observation/qualification seals without reruns."""
    R.check(
        PARENT, ("computed_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    R.check(
        READER, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )


def prepare(out):
    """Bind old individual-item responses before any SOURCE contact is computed."""
    parents()
    units, bank, bindings = (
        R.read(READER / f) for f in ("units.json", "bank.json", "event_bindings.json")
    )
    if len(units) != 386 or len(bindings) != 48:
        raise ValueError("Original complete observational scope required")
    logits = np.empty((48, 2, 4, 50, 2))
    seen, files = set(), []
    for i, row in enumerate(units):
        p = READER / "batches" / f"{i:03d}.npz"
        files.append(p)
        if row["unit"] < 0:
            continue
        key = (row["unit"], SPEC["operations"].index(row["operation"]), row["context"])
        if key in seen:
            raise ValueError("Duplicated original item family")
        seen.add(key)
        with np.load(p, allow_pickle=False) as archive:
            logits[key] = archive["logits"]
    if seen != set(np.ndindex(48, 2, 4)):
        raise ValueError("Missing registered item observation")
    odds = logits[..., 1] - logits[..., 0]
    joint = (odds[:, :, 0] + odds[:, :, 1] - odds[:, :, 2] - odds[:, :, 3]) / 2
    old = load_bundle(PARENT / "fields.npz")["keyed_joint"]
    key_error = float(abs(joint @ O.key_matrix(bank).T - old).max())
    if key_error > 1e-12 or not np.isfinite(joint).all():
        raise ArithmeticError("Old balanced item response changed")
    out.mkdir(parents=True, exist_ok=False)
    files.extend(
        READER / f
        for f in (
            "manifest.json",
            "units.json",
            "bank.json",
            "event_bindings.json",
            "collected_seal.json",
        )
    )
    files.extend(
        PARENT / f
        for f in (
            "source.npz",
            "fields.npz",
            "cold_record.json",
            "manifest.json",
            "computed_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        )
    )
    R.write(
        out / "manifest.json",
        {
            "config": SPEC,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "historical_exploration_not_new_lockbox": True,
            "application_psychological_labels_read": False,
            "original_scope_complete": False,
        },
    )
    R.archive(out / "joint_responses.npz", joint=joint)
    R.write(out / "bank.json", bank)
    R.write(out / "event_bindings.json", bindings)
    R.write(
        out / "quality.json",
        {
            "source_authors": 386,
            "panel_authors": 24,
            "pairs": 48,
            "items": 50,
            "operations": 2,
            "key_replay_error": key_error,
            "new_neural_API_private_training": 0,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("joint_responses.npz", "bank.json", "event_bindings.json", "quality.json"),
    )
    print(json.dumps(R.read(out / "quality.json")))


def expired(_signal, _frame):
    """Stop the single computation at its finite original budget."""
    raise TimeoutError("Original60-second item configuration cap")


def compute(out):
    """Retain signed cross-moments and test only the fixed5 shape primary contacts."""
    R.check(out)
    if (out / "computed_seal.json").exists() or (out / "failed_compute.json").exists():
        raise FileExistsError("No scientific computation retry")
    z = load_bundle(PARENT / "source.npz")
    joint, bank = (
        load_bundle(out / "joint_responses.npz")["joint"],
        R.read(out / "bank.json"),
    )
    left, right = np.arange(0, 48, 2), np.arange(1, 48, 2)
    source_values = (z["labels"], z["means"], z["labels"] - z["means"])
    accounts = np.empty((2, 3, 2, 3, 3, 5, 5, 3, 3))
    patterns = np.empty((2, 3, 2, 3, 2, 5, 5, 3, 10))
    rows, primary, references, exact, literal_fields = [], [], [], [], []
    begin = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(SPEC["seconds_cap"])
    try:
        with threadpool_limits(limits=1):
            for sp, space in enumerate(SPEC["spaces"]):
                for ref, reference in enumerate(SPEC["references"]):
                    d = z["weights"][sp, ref].mean(1)
                    a0, b0 = (
                        np.stack([d[ix] @ v for v in source_values])
                        for ix in (left, right)
                    )
                    for op, operation in enumerate(SPEC["operations"]):
                        for q, target in enumerate(SPEC["targets"]):
                            a, b = (
                                O.centered_scores(a0[:, :, q]),
                                O.centered_scores(b0[:, :, q]),
                            )
                            aa = np.broadcast_to(a[:, None], (3, 3, 24))
                            bb = np.broadcast_to(b[None, :], (3, 3, 24))
                            for r, query in enumerate(SPEC["targets"]):
                                frame, v, _ = M.item_frame(joint[:, op], bank, r)
                                lparts, rparts = (
                                    M.split_frame(frame[left], v),
                                    M.split_frame(frame[right], v),
                                )
                                ks = M.kernels(frame[left], frame[right], v)
                                for component in range(3):
                                    patterns[sp, ref, op, component, 0, q, r] = (
                                        M.signed_patterns(a0[component, :, q], lparts)
                                    )
                                    patterns[sp, ref, op, component, 1, q, r] = (
                                        M.signed_patterns(b0[component, :, q], rparts)
                                    )
                                for part, name in enumerate(SPEC["response_parts"]):
                                    values = O.cross_author_accounts(aa, bb, ks[part])
                                    for kind, key in enumerate(KINDS):
                                        accounts[
                                            sp, ref, op, :, :, q, r, part, kind
                                        ] = values[key]
                                    rows.append(
                                        {
                                            "space": space,
                                            "reference": reference,
                                            "operation": operation,
                                            "source_target": target,
                                            "query": query,
                                            "response_part": name,
                                            **{
                                                key: float(values[key][2, 2])
                                                for key in KINDS
                                            },
                                        }
                                    )
                                    if (sp, ref, op, part) != (1, 2, 0, 2) or q != r:
                                        continue
                                    e = source_values[2][:, q]
                                    en = e[z["assignments"][q]]
                                    an, bn = (
                                        O.centered_scores(en @ d[ix].T)
                                        for ix in (left, right)
                                    )
                                    null = O.cross_author_accounts(an, bn, ks[part])[
                                        "distinct_authors"
                                    ]
                                    mean = O.orbit_mean(
                                        d[left],
                                        d[right],
                                        C.orbit_covariance(
                                            z["labels"][:, q], z["groups"][:, q]
                                        ),
                                        ks[part],
                                    )
                                    observed = float(values["distinct_authors"][2, 2])
                                    primary.append(
                                        {
                                            "target": target,
                                            "item_shape_U": observed,
                                            "exact_assignment_mean": mean,
                                            "old_reference_mean": float(null.mean()),
                                            "upper_conditional_rank": float(
                                                (1 + np.sum(null >= observed)) / 200
                                            ),
                                            "key_U": float(
                                                accounts[sp, ref, op, 2, 2, q, r, 1, 2]
                                            ),
                                            "full_U": float(
                                                accounts[sp, ref, op, 2, 2, q, r, 0, 2]
                                            ),
                                            "item_shape_D": float(
                                                values["same_author"][2, 2]
                                            ),
                                        }
                                    )
                                    references.append(null)
                                    exact.append(mean)
                                    literal_fields.append(
                                        (frame[left], frame[right], v)
                                    )
            for row, adjusted in zip(
                primary,
                E.holm([p["upper_conditional_rank"] for p in primary]),
                strict=True,
            ):
                row["Holm5_rank"] = float(adjusted)
            channel_error = float(
                abs(
                    accounts[..., 0, :] - accounts[..., 1, :] - accounts[..., 2, :]
                ).max()
            )
            source_error = float(
                abs(
                    accounts[:, :, :, 0, 0] - accounts[:, :, :, 1:, 1:].sum((3, 4))
                ).max()
            )
            old = R.read(PARENT / "cold_record.json")
            readings = np.stack(
                [
                    d @ source_values[2]
                    for d in (
                        z["weights"][1, 2, left].mean(1),
                        z["weights"][1, 2, right].mean(1),
                    )
                ]
            )
            record = {
                "version": "HM286-v1",
                **{
                    key: old[key]
                    for key in ("author_keys", "left_event_ids", "right_event_ids")
                },
                "source_readings": readings.tolist(),
                "item_joint_responses": np.stack(
                    (joint[left, 0], joint[right, 0])
                ).tolist(),
                "item_bank": [
                    {key: row[key] for key in ("item", "target", "pole")}
                    for row in bank
                ],
            }
            R.table(out / "all_contacts.csv", rows)
            R.table(out / "primary.csv", primary)
            R.archive(
                out / "accounts.npz",
                accounts=accounts,
                patterns=patterns,
                references=np.asarray(references),
                exact_means=np.asarray(exact),
            )
            R.archive(
                out / "literal_fields.npz",
                left=np.asarray([x[0] for x in literal_fields]),
                right=np.asarray([x[1] for x in literal_fields]),
                directions=np.asarray([x[2] for x in literal_fields]),
            )
            R.write(out / "cold_record.json", record)
            R.write(
                out / "summary.json",
                {
                    "primary": primary,
                    "compute_seconds": time.monotonic() - begin,
                    "descriptive_rows": len(rows),
                    "old_assignments_per_target": 199,
                    "key_shape_closure_error": channel_error,
                    "source_mixed_closure_error": source_error,
                    "new_neural_API_private_training": 0,
                    "no_fitted_probe": True,
                    "original_scope_complete": False,
                },
            )
        R.seal(
            out,
            "computed_seal.json",
            (
                "all_contacts.csv",
                "primary.csv",
                "accounts.npz",
                "literal_fields.npz",
                "cold_record.json",
                "summary.json",
            ),
        )
        print(json.dumps(R.read(out / "summary.json")))
    except Exception as error:
        R.write(
            out / "failed_compute.json",
            {"error": repr(error), "seconds": time.monotonic() - begin},
        )
        raise
    finally:
        signal.alarm(0)


def verify(out):
    """Literal off-author pair sums and cold signed patterns,without scientific reruns."""
    parents()
    R.check(out, ("computed_seal.json",))
    source = load_bundle(PARENT / "source.npz")
    data, fixture = (
        load_bundle(out / "accounts.npz"),
        load_bundle(out / "literal_fields.npz"),
    )
    d = source["weights"][1, 2].mean(1)
    left, right = np.arange(0, 48, 2), np.arange(1, 48, 2)
    errors = {
        "literal_pairs": 0.0,
        "exact_assignment_mean": 0.0,
        "source_sign_patterns": 0.0,
    }
    with threadpool_limits(limits=1):
        for q in range(5):
            z0, z1, v = (
                fixture["left"][q],
                fixture["right"][q],
                fixture["directions"][q],
            )
            x0, x1 = z0 - (z0 @ v)[:, None] * v, z1 - (z1 @ v)[:, None] * v
            k = x0 @ x1.T
            e = source["labels"][:, q] - source["means"][:, q]
            en = np.vstack((e, e[source["assignments"][q]]))
            a, b = (O.centered_scores(en @ d[ix].T) for ix in (left, right))
            actual = np.r_[
                data["accounts"][1, 2, 0, 2, 2, q, q, 2, 2], data["references"][q]
            ]
            for ix, saved in enumerate(actual):
                direct = sum(
                    float(a[ix, i] * b[ix, j] * (x0[i] @ x1[j]))
                    for i in range(24)
                    for j in range(24)
                    if i != j
                ) / (24 * 23)
                errors["literal_pairs"] = max(
                    errors["literal_pairs"], abs(direct - saved)
                )
            aa, bb = d[left] - d[left].mean(0), d[right] - d[right].mean(0)
            cov = (
                aa
                @ C.orbit_covariance(source["labels"][:, q], source["groups"][:, q])
                @ bb.T
            )
            expected = sum(
                float(cov[i, j] * k[i, j])
                for i in range(24)
                for j in range(24)
                if i != j
            ) / (24 * 23)
            errors["exact_assignment_mean"] = max(
                errors["exact_assignment_mean"], abs(expected - data["exact_means"][q])
            )
            changed = M.signed_patterns(-a[0], M.split_frame(z0, v))
            old = data["patterns"][1, 2, 0, 2, 0, q, q]
            errors["source_sign_patterns"] = max(
                errors["source_sign_patterns"], float(abs(changed + old).max())
            )
    reply = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_item_configuration_cold.py"),
            "--input",
            str(out / "cold_record.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cold = json.loads(reply.stdout)
    patterns = data["patterns"][1, 2, 0, 2]
    errors["cold_patterns"] = float(
        abs(np.asarray(cold["signed_source_query_item_patterns"]) - patterns).max()
    )
    contact = data["accounts"][1, 2, 0, 2, 2, :, :, :, 2]
    errors["cold_contacts"] = float(
        abs(np.asarray(cold["distinct_author_source_query_parts"]) - contact).max()
    )
    summary = R.read(out / "summary.json")
    errors["key_shape_closure"] = summary["key_shape_closure_error"]
    errors["source_mixed_closure"] = summary["source_mixed_closure_error"]
    if max(errors.values()) > 1e-12:
        raise ArithmeticError("Original adequate finite account tolerance fails")
    R.write(out / "cold_readback.json", cold)
    R.write(
        out / "verification.json",
        {
            "PASS": True,
            "errors": errors,
            "literal_primary_and_references": 1000,
            "same_agent_not_blind": True,
        },
    )
    R.seal(out, "verification_seal.json", ("cold_readback.json", "verification.json"))
    print(json.dumps(R.read(out / "verification.json")))


def main():
    """Run each scientific stage once; check verifies seals only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compute", "verify", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        parents()
        R.check(args.out, ("computed_seal.json", "verification_seal.json"))
        print(json.dumps({"status": "ALL_OLD_AND_NEW_SEALS_PASS"}))
    else:
        {"prepare": prepare, "compute": compute, "verify": verify}[args.stage](args.out)


if __name__ == "__main__":
    main()
