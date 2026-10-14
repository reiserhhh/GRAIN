#!/usr/bin/env python3
"""HM22: fixed source score covectors versus native query-conditioned motion."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_native_joint as N
from grain_research.construction import math_query_action_transport as Q
from grain_research.construction.math_action_observability import (
    family_weights,
    moments,
)
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_action_observability import artifact_hashes, csv_rows
from scripts.run_grain_math_paired_readout_transport import check as readout_check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_query_action_transport_v1.json"
PLAN = ROOT / "docs/GRAIN_QUERY_ACTION_TRANSPORT_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/query_action_transport_r1"
CODE = (
    "grain_research/construction/math_query_action_transport.py",
    "scripts/run_grain_math_query_action_transport.py",
    "tests/test_grain_math_query_action_transport.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out, source=False):
    """Check the immutable new packet plus all verified HM21/native ancestors."""
    m = read_json(out / "00/manifest.json")
    for n, sha in m["source_files"].items():
        if digest(ROOT / n) != sha:
            raise ValueError(f"changed HM22 code: {n}")
    for n, sha in m["inputs"].items():
        if digest(out / n) != sha:
            raise ValueError(f"changed HM22 input: {n}")
    old = ROOT / m["spec"]["readout_packet"]
    readout_check(old)
    for n, sha in read_json(out / "00/inherited_artifacts.json").items():
        if digest(old / n) != sha:
            raise ValueError(f"changed HM21 artifact: {n}")
    if (
        digest(ROOT / m["spec"]["readout_verification"] / "summary.json")
        != m["verification_sha256"]
    ):
        raise ValueError("changed HM21 verification")
    if source:
        for n, sha in read_json(out / "01/source_manifest.json")["artifacts"].items():
            if digest(out / n) != sha:
                raise ValueError(f"changed source anchor: {n}")
    return m


def prepare(out):
    """Freeze addresses/code before source moments and application diagnostics."""
    if out.exists():
        raise FileExistsError("preserve prior HM22 packet")
    spec = read_json(SPEC)
    old, verification = (
        ROOT / spec[n] for n in ("readout_packet", "readout_verification")
    )
    readout_check(old)
    v = read_json(verification / "summary.json")
    if (
        v["status"] != "VERIFIED_FINITE_HM21_NOT_GLOBAL_SCOPE_COMPLETE"
        or v["primary_manifest_sha256"] != digest(old / "00/manifest.json")
        or v["verifier_sha256"]
        != digest(ROOT / "scripts/verify_grain_math_paired_readout_transport.py")
    ):
        raise ValueError("unchanged verified HM21 required")
    ledger = read_json(verification / "frozen_primary_artifact_hashes.json")
    for n, sha in ledger.items():
        if digest(old / n) != sha:
            raise ValueError(f"changed inherited artifact: {n}")
    meta = read_json(old / "00/metadata.json")
    if (
        len(meta) != spec["paired_query_rows"]
        or dict(Counter(r["role"] for r in meta)) != spec["role_rows"]
        or [r["row"] for r in meta] != list(range(len(meta)))
    ):
        raise ValueError("original address scope changed")
    (out / "00").mkdir(parents=True)
    for name, value in (
        ("metadata", meta),
        ("alphabet", read_json(old / "00/alphabet.json")),
        ("inherited_artifacts", ledger),
    ):
        write_json(out / "00" / (name + ".json"), value)
    for n in CODE:
        p = out / "00/code" / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / n).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "spec": spec,
            "source_files": {n: digest(ROOT / n) for n in CODE},
            "inputs": artifact_hashes(out / "00", out),
            "verification_sha256": digest(verification / "summary.json"),
            "frozen_before_analysis": True,
            "new_source_fits": 0,
            "new_model_calls": 0,
        },
    )
    print({"status": "HM22_FROZEN", "addresses": len(meta)}, flush=True)


def source(out):
    """Freeze equal-family reference gradients without loading any held readout."""
    spec = check(out)["spec"]
    if (out / "01").exists():
        raise FileExistsError("preserve existing source anchor")
    meta = read_json(out / "00/metadata.json")
    a = np.array(read_json(out / "00/alphabet.json")["values"])
    saved, ledger, source_rows, audit = {}, {}, [], []
    for qi, q in enumerate(spec["queries"]):
        rows = [
            r
            for r in meta
            if r["role"] == "source"
            and r["arm"] == spec["anchor_design"]
            and r["query"] == q
        ]
        if (
            len(rows) != spec["anchor_states_per_query"]
            or len({r["family"] for r in rows}) != spec["source_families"]
        ):
            raise ValueError("source anchor scope changed")
        g = []
        for r in rows:
            p = N.probability(
                read_json(ROOT / r["reference_path"])["native"]["candidate_logits"]
            )
            g.append(p * (a - p @ a))
            ledger[r["reference_path"]] = digest(ROOT / r["reference_path"])
        g = np.array(g)
        anchor, w = Q.source_anchor(rows, g)
        saved[f"q{qi}_anchor"], saved[f"q{qi}_gradients"], saved[f"q{qi}_weights"] = (
            anchor,
            g,
            w,
        )
        saved[f"q{qi}_rows"] = np.array([r["row"] for r in rows])
        count = Counter(r["reference_path"] for r in rows)
        audit.append(
            {
                "query": q,
                "design": spec["anchor_design"],
                "states": len(rows),
                "families": len({r["family"] for r in rows}),
                "distinct_reference_records": len(count),
                "duplicate_address_occurrences": len(rows) - len(count),
                "max_reference_multiplicity": max(count.values()),
                "anchor_sum_gauge_error": abs(float(anchor.sum())),
                "anchor_norm": float(np.linalg.norm(anchor)),
            }
        )
        source_rows.extend(
            {
                **r,
                "anchor_weight": float(w[i]),
                "reference_multiplicity": count[r["reference_path"]],
            }
            for i, r in enumerate(rows)
        )
    (out / "01").mkdir()
    np.savez_compressed(out / "01/anchors.npz", **saved)
    write_csv(out / "01/source_addresses.csv", source_rows)
    write_csv(out / "01/anchor_audit.csv", audit)
    write_json(out / "01/consumed_source_hashes.json", ledger)
    write_json(
        out / "01/source_manifest.json",
        {
            "held_targets_loaded": False,
            "source_anchor_moments": len(audit),
            "source_rows": len(source_rows),
            "distinct_source_reference_files": len(ledger),
            "artifacts": artifact_hashes(out / "01", out),
        },
    )
    print({"status": "HM22_SOURCE_FROZEN", "audit": audit}, flush=True)


def evaluate(out):
    """Evaluate every three-account action and its complete cross-query covariance."""
    spec = check(out, source=True)["spec"]
    if (out / "02").exists():
        raise FileExistsError("preserve evaluated HM22 result")
    start = time.monotonic()
    old = ROOT / spec["readout_packet"]
    meta = read_json(out / "00/metadata.json")
    original = csv_rows(old / "02/actions.csv")
    with np.load(old / "02/endpoints.npz") as p:
        d, gradients = p["displacement"].copy(), p["reference_gradient"].copy()
    with np.load(out / "01/anchors.npz") as p:
        anchors = {q: p[f"q{i}_anchor"].copy() for i, q in enumerate(spec["queries"])}
    parts, actions = np.empty((len(meta), 3)), []
    errors = {
        "action": 0.0,
        "tangent": 0.0,
        "covariance": 0.0,
        "logit_contraction": 0.0,
        "historical_link": 0.0,
    }
    for q in spec["queries"]:
        ids = np.array([r["row"] for r in meta if r["query"] == q])
        curvature = np.array([float(original[i]["curvature"]) for i in ids])
        parts[ids] = Q.accounts(d[ids], gradients[ids], anchors[q], curvature)
    for r, v in zip(meta, parts, strict=True):
        oldr = original[r["row"]]
        if int(oldr["row"]) != r["row"]:
            raise ValueError("HM21 row order changed")
        native, tangent = float(oldr["native"]), float(oldr["tangent"])
        errors["action"] = max(errors["action"], abs(v.sum() - native))
        errors["tangent"] = max(errors["tangent"], abs(v[0] + v[1] - tangent))
        if r["exact_input_noop"] and np.any(v != 0):
            raise ValueError("changed native no-op")
        actions.append(
            {
                **r,
                "B": float(v[0]),
                "G": float(v[1]),
                "C": float(v[2]),
                "native": native,
                "tangent": tangent,
            }
        )
    lookup = {(r["state_index"], r["query"]): r for r in actions}
    first = sorted(
        (r for r in actions if r["query"] == spec["queries"][0]),
        key=lambda r: r["state_index"],
    )
    if [r["state_index"] for r in first] != list(range(spec["unique_state_pairs"])):
        raise ValueError("missing paired native states")
    compiled = {
        (int(r["row"]), r["model"]): r
        for r in csv_rows(ROOT / spec["exchange_packet"] / "02/components.csv")
    }
    oldjoint = {
        (r["role"], r["design"], r["query_a"], r["query_b"]): r
        for r in csv_rows(old / "02/readout_covariance.csv")
    }
    stats, joint, gaps, matrices, matrix_keys = [], [], [], [], []
    for role, design in itertools.product(
        spec["roles"], ("all_states", *sorted({r["arm"] for r in first}))
    ):
        selected = [
            r
            for r in first
            if r["role"] == role and (design == "all_states" or r["arm"] == design)
        ]
        ids = [r["state_index"] for r in selected]
        w = family_weights([r["family"] for r in selected])
        info = {
            "role": role,
            "design": design,
            "states": len(ids),
            "families": len({r["family"] for r in selected}),
        }
        for q in spec["queries"]:
            v = [lookup[s, q] for s in ids]
            values = np.array([[r[n] for n in Q.PARTS] for r in v])
            y = np.array([r["native"] for r in v])
            rms = lambda x, w=w: float(np.sqrt(w @ x**2))
            stats.append(
                {
                    **info,
                    "query": q,
                    "native_rms": rms(y),
                    **{n + "_rms": rms(values[:, i]) for i, n in enumerate(Q.PARTS)},
                    **{
                        n + "_native_rms_ratio": rms(values[:, i]) / rms(y)
                        if rms(y)
                        else None
                        for i, n in enumerate(Q.PARTS)
                    },
                    "B_G_inner": float(w @ (values[:, 0] * values[:, 1])),
                    "B_C_inner": float(w @ (values[:, 0] * values[:, 2])),
                    "G_C_inner": float(w @ (values[:, 1] * values[:, 2])),
                }
            )
        for qa, qb in itertools.combinations(spec["queries"], 2):
            va, vb = ([lookup[s, q] for s in ids] for q in (qa, qb))
            pa, pb = (np.array([[r[n] for n in Q.PARTS] for r in v]) for v in (va, vb))
            v = Q.covariance_accounts(pa, pb, w)
            sigma = Q.displacement_cross_covariance(
                d[[r["row"] for r in va]], d[[r["row"] for r in vb]], w
            )
            contraction = float(anchors[qa] @ sigma @ anchors[qb])
            errors["logit_contraction"] = max(
                errors["logit_contraction"], abs(contraction - v["BB"])
            )
            errors["covariance"] = max(errors["covariance"], v["recombination_error"])
            for n, oldn in (
                ("native_covariance", "native_covariance"),
                ("tangent_covariance", "tangent_covariance"),
                ("link_increment", "link_covariance_increment"),
            ):
                errors["historical_link"] = max(
                    errors["historical_link"],
                    abs(v[n] - float(oldjoint[role, design, qa, qb][oldn])),
                )
            key = {**info, "query_a": qa, "query_b": qb, "matrix_row": len(matrices)}
            matrix_keys.append(key)
            matrices.append(sigma)
            joint.append(
                {
                    **key,
                    **v,
                    "fixed_covariance_from_logit_block": contraction,
                    "logit_gauge_left_error": float(
                        np.max(abs(np.ones(len(anchors[qa])) @ sigma))
                    ),
                    "logit_gauge_right_error": float(
                        np.max(abs(sigma @ np.ones(len(anchors[qb]))))
                    ),
                }
            )
            for model, channel in itertools.product(spec["models"], ("forward", "odd")):
                ca, cb = (
                    np.array([float(compiled[r["row"], model][channel]) for r in items])
                    for items in (va, vb)
                )
                cc = float(moments(np.column_stack((ca, cb)), w)["covariance"][0, 1])
                gap = v["native_covariance"] - cc
                fixed = v["BB"] - cc
                gaps.append(
                    {
                        **info,
                        "query_a": qa,
                        "query_b": qb,
                        "model": model,
                        "channel": channel,
                        "native_covariance": v["native_covariance"],
                        "compiled_covariance": cc,
                        "total_gap": gap,
                        "fixed_gap": fixed,
                        "sensitivity_increment": v["sensitivity_increment"],
                        "link_increment": v["link_increment"],
                        "sensitivity_gap_ratio": v["sensitivity_increment"] / gap
                        if gap
                        else None,
                        "link_gap_ratio": v["link_increment"] / gap if gap else None,
                        "gap_recombination_error": abs(
                            gap
                            - fixed
                            - v["sensitivity_increment"]
                            - v["link_increment"]
                        ),
                    }
                )
    if max(errors.values()) > spec["arithmetic_absolute_tolerance"]:
        raise ValueError(f"HM22 action accounts failed: {errors}")
    (out / "02").mkdir()
    np.savez_compressed(out / "02/accounts.npz", parts=parts)
    np.savez_compressed(
        out / "02/logit_cross_covariance.npz", matrices=np.array(matrices)
    )
    for n, r in (
        ("actions", actions),
        ("action_statistics", stats),
        ("covariance_accounts", joint),
        ("compiled_gap_decomposition", gaps),
        ("logit_matrix_addresses", matrix_keys),
    ):
        write_csv(out / "02" / (n + ".csv"), r)
    write_json(
        out / "02/summary.json",
        {
            "status": "HM22_EVALUATED_NOT_GLOBAL_COMPLETE",
            "action_rows": len(actions),
            "states": len(first),
            "statistics_rows": len(stats),
            "covariance_rows": len(joint),
            "gap_rows": len(gaps),
            "logit_blocks": len(matrices),
            "noop_rows": sum(r["exact_input_noop"] for r in actions),
            "errors": errors,
            "source_anchor_moments": 3,
            "new_source_fits": 0,
            "new_model_calls": 0,
            "seconds": time.monotonic() - start,
            "fixed_native_sign_agreements": sum(
                r["fixed_native_sign_agree"] for r in joint
            ),
            "key_application_full_rows": [
                r
                for r in gaps
                if r["role"] == "application" and r["design"] == "full_property"
            ],
            "scope": "cached patch response and fixed categorical observation units; not human personality validity",
        },
    )
    write_json(out / "02/artifact_hashes.json", artifact_hashes(out, out))
    print(
        {"status": "HM22_EVALUATED", "actions": len(actions), "errors": errors},
        flush=True,
    )


def main():
    """Run one finite stage without concurrent writers or overwrite recovery."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "source", "evaluate", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    with collector_guard(a.out):
        if a.stage == "prepare":
            prepare(a.out)
        elif a.stage == "source":
            source(a.out)
        elif a.stage == "evaluate":
            evaluate(a.out)
        else:
            check(a.out, source=True)
            print("HM22_FREEZE_VALID")


if __name__ == "__main__":
    main()
