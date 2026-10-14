#!/usr/bin/env python3
"""HM21: native paired-logit transport and readout contribution to the joint gap."""

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

from grain_research.construction import math_paired_readout_transport as P
from grain_research.construction.math_action_observability import (
    family_weights,
    moments,
)
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_action_observability import artifact_hashes, csv_rows
from scripts.run_grain_math_endpoint_exchange_orbit import check as exchange_check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_paired_readout_transport_v1.json"
PLAN = ROOT / "docs/GRAIN_PAIRED_READOUT_TRANSPORT_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/paired_readout_transport_r1"
CODE = (
    "grain_research/construction/math_paired_readout_transport.py",
    "grain_research/construction/math_native_joint.py",
    "scripts/run_grain_math_paired_readout_transport.py",
    "tests/test_grain_math_paired_readout_transport.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out):
    """Validate frozen code, inputs, HM20 ancestry and consumed native records."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["source_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"changed HM21 code: {name}")
    for name, sha in m["inputs"].items():
        if digest(out / name) != sha:
            raise ValueError(f"changed HM21 input: {name}")
    old = ROOT / m["spec"]["exchange_packet"]
    exchange_check(old, source=True)
    for name, sha in read_json(out / "00/inherited_artifacts.json").items():
        if digest(old / name) != sha:
            raise ValueError(f"changed HM20 artifact: {name}")
    if (
        digest(ROOT / m["spec"]["exchange_verification"] / "summary.json")
        != m["verification_sha256"]
    ):
        raise ValueError("changed HM20 verification")
    for name, sha in read_json(out / "00/native_record_hashes.json").items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"changed native observation: {name}")
    return m


def prepare(out):
    """Freeze all addressed pairs and the original alphabet before readout analysis."""
    if out.exists():
        raise FileExistsError("preserve prior HM21 output")
    spec = read_json(SPEC)
    old, verification = (
        ROOT / spec[n] for n in ("exchange_packet", "exchange_verification")
    )
    exchange_check(old, source=True)
    v = read_json(verification / "summary.json")
    if (
        v["status"] != "VERIFIED_FINITE_HM20_NOT_GLOBAL_SCOPE_COMPLETE"
        or digest(old / "00/manifest.json") != v["primary_manifest_sha256"]
        or digest(ROOT / "scripts/verify_grain_math_endpoint_exchange_orbit.py")
        != v["verifier_sha256"]
    ):
        raise ValueError("unchanged verified HM20 required")
    ledger = read_json(verification / "frozen_primary_artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(old / name) != sha:
            raise ValueError(f"changed inherited result: {name}")
    meta = read_json(old / "00/metadata.json")
    if (
        len(meta) != spec["paired_query_rows"]
        or dict(Counter(r["role"] for r in meta)) != spec["role_rows"]
        or [r["row"] for r in meta] != list(range(len(meta)))
    ):
        raise ValueError("original observation scope changed")
    alphabet = read_json(ROOT / spec["span_packet"] / "00/manifest.json")["alphabet"]
    if alphabet["labels"] != list(map(str, range(10))) or alphabet["values"] != [
        i / 9 for i in range(10)
    ]:
        raise ValueError("native alphabet changed")
    (out / "00").mkdir(parents=True)
    for name, data in (
        ("metadata", meta),
        ("alphabet", alphabet),
        ("inherited_artifacts", ledger),
        ("native_record_hashes", read_json(old / "02/consumed_native_hashes.json")),
    ):
        write_json(out / "00" / (name + ".json"), data)
    for name in CODE:
        p = out / "00/code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
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
    print({"status": "HM21_FROZEN", "addresses": len(meta)}, flush=True)


def evaluate(out):
    """Decompose all native pairs and every declared joint covariance without fitting."""
    spec = check(out)["spec"]
    if (out / "02").exists():
        raise FileExistsError("preserve evaluated HM21 output")
    start = time.monotonic()
    metadata = read_json(out / "00/metadata.json")
    a = np.array(read_json(out / "00/alphabet.json")["values"])
    old = ROOT / spec["exchange_packet"]
    inventory = {
        int(r["row"]): r for r in csv_rows(old / "02/native_inverse_inventory.csv")
    }
    compiled = {
        (int(r["row"]), r["model"]): r for r in csv_rows(old / "02/components.csv")
    }
    arrays = {
        n: []
        for n in (
            "reference_logits",
            "target_logits",
            "reference_gradient",
            "displacement",
            "digit_path_action",
        )
    }
    actions, errors = [], Counter()
    for address in metadata:
        unit, ref = (
            read_json(ROOT / address[k]) for k in ("unit_path", "reference_path")
        )
        if unit["live_body_error"] != 0 or unit["address"]["query"] != address["query"]:
            raise ValueError("native body/address mismatch")
        r, t = (
            np.array(ref["native"]["candidate_logits"]),
            np.array(unit["native"]["candidate_logits"]),
        )
        v = P.paired_action(r, t, a, spec["quadrature_order"])
        for record, z, key in ((ref, r, "reference"), (unit, t, "target")):
            native = record["native"]
            probability = P.N.probability(z)
            errors["score"] = max(
                errors["score"],
                abs(float(probability @ a) - native["conditional_score"]),
            )
            errors["probability"] = max(
                errors["probability"],
                float(np.max(abs(probability - native["conditional_probabilities"]))),
            )
        errors["native_action"] = max(
            errors["native_action"],
            abs(v["native"] - float(inventory[address["row"]]["native_forward"])),
        )
        errors["gradient_integral"] = max(
            errors["gradient_integral"], abs(v["native"] - v["gradient_integral"])
        )
        errors["hessian_integral"] = max(
            errors["hessian_integral"], abs(v["curvature"] - v["hessian_integral"])
        )
        errors["digit_sum"] = max(
            errors["digit_sum"], abs(v["native"] - v["digit_path_action"].sum())
        )
        if address["exact_input_noop"] and np.max(abs(v["displacement"])) != 0:
            raise ValueError("declared exact input no-op has a changed native endpoint")
        for name, value in (("reference_logits", r), ("target_logits", t)):
            arrays[name].append(value)
        for name in ("reference_gradient", "displacement", "digit_path_action"):
            arrays[name].append(v[name])
        actions.append(
            {
                **address,
                **{k: value for k, value in v.items() if np.ndim(value) == 0},
                "reference_candidate_mass": ref["native"]["candidate_mass"],
                "target_candidate_mass": unit["native"]["candidate_mass"],
            }
        )
    if max(errors.values()) > spec["arithmetic_absolute_tolerance"]:
        raise ValueError(f"fixed readout arithmetic failed: {errors}")
    lookup = {(r["state_index"], r["query"]): r for r in actions}
    first = sorted(
        (r for r in actions if r["query"] == spec["queries"][0]),
        key=lambda r: r["state_index"],
    )
    if [r["state_index"] for r in first] != list(range(spec["unique_state_pairs"])):
        raise ValueError("incomplete query-paired state design")
    stats, covariances, gaps = [], [], []
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
        for query in spec["queries"]:
            values = [lookup[s, query] for s in ids]
            y, l, c = (
                np.array([r[k] for r in values])
                for k in ("native", "tangent", "curvature")
            )
            rms = lambda v, w=w: float(np.sqrt(w @ v**2))
            stats.append(
                {
                    **info,
                    "query": query,
                    "native_rms": rms(y),
                    "tangent_rms": rms(l),
                    "curvature_rms": rms(c),
                    "curvature_native_rms_ratio": rms(c) / rms(y) if rms(y) else None,
                    "native_tangent_inner": float(w @ (y * l)),
                    "tangent_curvature_inner": float(w @ (l * c)),
                    "reference_candidate_mass_min": min(
                        r["reference_candidate_mass"] for r in values
                    ),
                    "target_candidate_mass_min": min(
                        r["target_candidate_mass"] for r in values
                    ),
                    "exact_noops": sum(r["exact_input_noop"] for r in values),
                }
            )
        for qa, qb in itertools.combinations(spec["queries"], 2):
            va, vb = ([lookup[s, q] for s in ids] for q in (qa, qb))
            na, nb = (np.array([r["native"] for r in v]) for v in (va, vb))
            la, lb = (np.array([r["tangent"] for r in v]) for v in (va, vb))
            part = P.covariance_parts(na, nb, la, lb, w)
            errors["covariance"] = max(
                errors["covariance"], part["recombination_error"]
            )
            covariances.append({**info, "query_a": qa, "query_b": qb, **part})
            for model, channel in itertools.product(spec["models"], ("forward", "odd")):
                pa, pb = (
                    np.array([float(compiled[r["row"], model][channel]) for r in v])
                    for v in (va, vb)
                )
                pc = float(moments(np.column_stack((pa, pb)), w)["covariance"][0, 1])
                gap = part["native_covariance"] - pc
                upstream = part["tangent_covariance"] - pc
                gaps.append(
                    {
                        **info,
                        "query_a": qa,
                        "query_b": qb,
                        "model": model,
                        "channel": channel,
                        "native_covariance": part["native_covariance"],
                        "compiled_covariance": pc,
                        "total_gap": gap,
                        "tangent_gap": upstream,
                        "link_increment": part["link_covariance_increment"],
                        "link_gap_ratio": part["link_covariance_increment"] / gap
                        if gap
                        else None,
                        "gap_recombination_error": abs(
                            gap - upstream - part["link_covariance_increment"]
                        ),
                    }
                )
    (out / "02").mkdir()
    np.savez_compressed(
        out / "02/endpoints.npz", **{k: np.array(v) for k, v in arrays.items()}
    )
    write_csv(out / "02/actions.csv", actions)
    write_csv(out / "02/action_statistics.csv", stats)
    write_csv(out / "02/readout_covariance.csv", covariances)
    write_csv(out / "02/compiled_gap_decomposition.csv", gaps)
    write_json(
        out / "02/summary.json",
        {
            "status": "HM21_EVALUATED_NOT_GLOBAL_COMPLETE",
            "action_rows": len(actions),
            "states": len(first),
            "statistics_rows": len(stats),
            "covariance_rows": len(covariances),
            "gap_rows": len(gaps),
            "noop_rows": sum(r["exact_input_noop"] for r in actions),
            "native_record_files": len(read_json(out / "00/native_record_hashes.json")),
            "errors": dict(errors),
            "new_source_fits": 0,
            "new_model_calls": 0,
            "seconds": time.monotonic() - start,
            "all_covariance_sign_agreements": sum(
                r["native_tangent_sign_agree"] for r in covariances
            ),
            "key_application_full_rows": [
                r
                for r in gaps
                if r["role"] == "application" and r["design"] == "full_property"
            ],
            "scope": "cached patch pairs; categorical observation law, not human personality validity",
        },
    )
    write_json(out / "02/artifact_hashes.json", artifact_hashes(out, out))
    print(
        {"status": "HM21_EVALUATED", "actions": len(actions), "errors": dict(errors)},
        flush=True,
    )


def main():
    """Run one finite phase while rejecting concurrent writers and overwrites."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "evaluate", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    args = p.parse_args()
    with collector_guard(args.out):
        if args.stage == "prepare":
            prepare(args.out)
        elif args.stage == "evaluate":
            evaluate(args.out)
        else:
            check(args.out)
            print("HM21_FREEZE_VALID")


if __name__ == "__main__":
    main()
