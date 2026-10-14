#!/usr/bin/env python3
"""HM23: one source-only vector-valued fit with the frozen HM16 input kernel."""

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

from grain_research.construction import math_context_covector as C
from grain_research.construction import math_output_vector_transport as M
from grain_research.construction import math_query_action_transport as Q
from grain_research.construction.math_action_observability import (
    family_weights,
    moments,
)
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_action_observability import artifact_hashes, csv_rows
from scripts.run_grain_math_context_covector import check as context_check
from scripts.run_grain_math_context_covector import load_views
from scripts.run_grain_math_query_action_transport import check as query_check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_token_mean import family_interval
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_output_vector_transport_v1.json"
PLAN = ROOT / "docs/GRAIN_OUTPUT_VECTOR_TRANSPORT_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/output_vector_transport_r1"
CODE = (
    "grain_research/construction/math_output_vector_transport.py",
    "scripts/run_grain_math_output_vector_transport.py",
    "scripts/verify_grain_math_output_vector_transport.py",
    "tests/test_grain_math_output_vector_transport.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def check(out, models=False):
    """Validate frozen source code, input addresses and both verified ancestors."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["source_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"changed HM23 code: {name}")
    for name, sha in m["inputs"].items():
        if digest(out / name) != sha:
            raise ValueError(f"changed HM23 input: {name}")
    spec = m["spec"]
    query_check(ROOT / spec["query_packet"], source=True)
    context_check(ROOT / spec["context_packet"], models=True)
    for kind in ("query", "context"):
        for name, sha in read_json(out / "00" / f"{kind}_artifacts.json").items():
            if digest(ROOT / spec[f"{kind}_packet"] / name) != sha:
                raise ValueError(f"changed {kind} ancestor: {name}")
        if (
            digest(ROOT / spec[f"{kind}_verification"] / "summary.json")
            != m[f"{kind}_verification_sha256"]
        ):
            raise ValueError(f"changed {kind} verification")
    if models:
        for name, sha in read_json(out / "01/model_manifest.json")["artifacts"].items():
            if digest(out / name) != sha:
                raise ValueError(f"changed source vector model: {name}")
        for name, sha in read_json(out / "01/source_files.json").items():
            if digest(ROOT / name) != sha:
                raise ValueError(f"changed source native endpoint: {name}")
    return m


def prepare(out):
    """Freeze the exact original scope and one selected mechanism, not a grid."""
    if out.exists():
        raise FileExistsError("preserve earlier HM23 packet")
    spec = read_json(SPEC)
    if spec["model"] != "receiver_tensor" or spec["alpha"] != 1:
        raise ValueError("fixed inherited receiver model required")
    query_check(ROOT / spec["query_packet"], source=True)
    context_check(ROOT / spec["context_packet"], models=True)
    ledgers, shas = {}, {}
    for kind, packet_id, script in (
        ("query", "HM22", "query_action_transport"),
        ("context", "HM16", "context_covector"),
    ):
        vpath = ROOT / spec[f"{kind}_verification"]
        v = read_json(vpath / "summary.json")
        if v["status"] != f"VERIFIED_FINITE_{packet_id}_NOT_GLOBAL_SCOPE_COMPLETE" or v[
            "verifier_sha256"
        ] != digest(ROOT / f"scripts/verify_grain_math_{script}.py"):
            raise ValueError(f"verified unchanged {kind} ancestor required")
        ledger = read_json(vpath / "frozen_primary_artifact_hashes.json")
        for name, sha in ledger.items():
            if digest(ROOT / spec[f"{kind}_packet"] / name) != sha:
                raise ValueError(f"changed inherited artifact: {kind}/{name}")
        ledgers[kind], shas[kind] = ledger, digest(vpath / "summary.json")
    old = ROOT / spec["query_packet"]
    meta = read_json(old / "00/metadata.json")
    if (
        len(meta) != spec["paired_query_rows"]
        or dict(Counter(r["role"] for r in meta)) != spec["role_rows"]
    ):
        raise ValueError("original scope changed")
    if [r["row"] for r in meta] != list(range(len(meta))):
        raise ValueError("original query row order changed")
    (out / "00").mkdir(parents=True)
    for n in ("metadata", "alphabet"):
        write_json(out / "00" / f"{n}.json", read_json(old / "00" / f"{n}.json"))
    for kind, ledger in ledgers.items():
        write_json(out / "00" / f"{kind}_artifacts.json", ledger)
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
            **{f"{k}_verification_sha256": v for k, v in shas.items()},
            "frozen_before_fit": True,
            "new_model_calls": 0,
            "new_neural_training": False,
        },
    )
    print({"status": "HM23_PREPARED", "addresses": len(meta)}, flush=True)


def inputs(out):
    """Rebuild the unchanged input kernel with source-only coordinates and units."""
    spec = read_json(out / "00/manifest.json")["spec"]
    old = ROOT / spec["context_packet"]
    with np.load(old / "01/source_coordinates.npz") as p:
        parameters = {
            n: p[n].copy() for n in ("origin", "context_scale", "delta_scale")
        }
        states = p["source_state_indices"].copy()
    x, c, _, _, _ = load_views(old, parameters)
    linear, coupling = C.kernel_parts(x, c, x[states], c[states], "receiver_tensor")
    k = linear + coupling
    functions = read_json(old / "01/functions.json")
    units = [functions[q]["receiver_tensor"]["kernel_unit"] for q in spec["queries"]]
    if (
        len(states) != spec["source_states"]
        or len(set(states)) != len(states)
        or len(set(units)) != 1
    ):
        raise ValueError("fixed source support/unit changed")
    return spec, states, k, units[0], functions


def compile(out):
    """Fit one 30-output source system before loading any held endpoint target."""
    check(out)
    if (out / "01").exists():
        raise FileExistsError("preserve frozen source vector model")
    start = time.monotonic()
    spec, states, k, unit, _ = inputs(out)
    meta = read_json(out / "00/metadata.json")
    selected = [r for r in meta if r["role"] == "source"]
    source_ledger, reference, target = {}, [], []
    for r in selected:
        pair = []
        for name in ("reference_path", "unit_path"):
            path = ROOT / r[name]
            source_ledger[r[name]] = digest(path)
            pair.append(read_json(path)["native"]["candidate_logits"])
        reference.append(pair[0])
        target.append(pair[1])
    d = M.source_targets(selected, reference, target, spec["queries"], states.tolist())
    if d.shape != (spec["source_states"], spec["output_columns"]):
        raise ValueError("full source output coordinates required")
    first = {r["state_index"]: r for r in selected if r["query"] == spec["queries"][0]}
    w = family_weights([first[s]["family"] for s in states])
    unit_error = abs(float(w @ np.diag(k[states])) - unit)
    if unit_error > spec["arithmetic_absolute_tolerance"]:
        raise ValueError("inherited kernel unit changed")
    f = M.fit_multi(k[states], d, w, unit, spec["alpha"])
    (out / "01").mkdir()
    np.savez_compressed(
        out / "01/model.npz",
        source_targets=d,
        weights=w,
        states=states,
        weighted_dual=f.pop("weighted_dual"),
        eigenvalues=f.pop("eigenvalues"),
    )
    write_csv(out / "01/source_addresses.csv", selected)
    write_json(out / "01/source_files.json", source_ledger)
    write_json(
        out / "01/function.json",
        {
            **f,
            "kernel_unit_error": unit_error,
            "source_families": len({r["family"] for r in selected}),
            "query_order": spec["queries"],
            "candidate_columns": spec["candidate_columns"],
            "source_only": True,
            "held_targets_loaded": False,
            "shared_system_is_not_learned_output_coupling": True,
            "compile_seconds": time.monotonic() - start,
        },
    )
    write_json(
        out / "01/model_manifest.json",
        {
            "held_targets_loaded": False,
            "new_shared_cpu_systems": 1,
            "right_hand_sides": spec["output_columns"],
            "new_model_calls": 0,
            "source_file_count": len(source_ledger),
            "artifacts": artifact_hashes(out / "01", out),
        },
    )
    print(
        {
            "status": "HM23_SOURCE_FROZEN",
            "source_matrix": list(d.shape),
            "shared_systems": 1,
            "residual": f["system_residual_max"],
            "held_targets_loaded": False,
        },
        flush=True,
    )


def evaluate(out):
    """Keep every held action, reference and vector remainder without model selection."""
    check(out, models=True)
    if (out / "02").exists():
        raise FileExistsError("preserve evaluated vector transport")
    start = time.monotonic()
    spec, _states, kernel, unit, functions = inputs(out)
    meta = read_json(out / "00/metadata.json")
    alphabet = np.array(read_json(out / "00/alphabet.json")["values"])
    with np.load(out / "01/model.npz") as p:
        predicted = M.apply_multi(kernel, p["weighted_dual"], unit)
    with np.load(ROOT / spec["readout_packet"] / "02/endpoints.npz") as p:
        r, t = p["reference_logits"].copy(), p["target_logits"].copy()
    native_delta = M.centered_displacement(r, t)
    with np.load(ROOT / spec["query_packet"] / "02/accounts.npz") as p:
        native_parts = p["parts"].copy()
    native = native_parts.sum(axis=1)
    new, scalar = np.empty(len(meta)), np.empty(len(meta))
    parts, dhat = np.empty_like(native_parts), np.empty_like(native_delta)
    with np.load(ROOT / spec["query_packet"] / "01/anchors.npz") as p:
        anchors = {q: p[f"q{i}_anchor"].copy() for i, q in enumerate(spec["queries"])}
    native_check_error = 0.0
    for qi, q in enumerate(spec["queries"]):
        ids = np.array([v["row"] for v in meta if v["query"] == q])
        ss = np.array([meta[i]["state_index"] for i in ids])
        dhat[ids] = predicted[ss, qi * 10 : (qi + 1) * 10]
        new[ids], parts[ids] = M.readout(r[ids], dhat[ids], alphabet, anchors[q])
        actual_y, actual_parts = M.readout(
            r[ids], native_delta[ids], alphabet, anchors[q]
        )
        native_check_error = max(
            native_check_error,
            float(np.max(abs(actual_y - native[ids]))),
            float(np.max(abs(actual_parts - native_parts[ids]))),
        )
        f = functions[q]["receiver_tensor"]
        with np.load(ROOT / spec["context_packet"] / f["path"]) as p:
            scalar[ids] = kernel[ss] @ p["weighted_dual"] / unit
    errors = {
        "account": float(np.max(abs(parts.sum(axis=1) - new))),
        "zero_sum_motion": float(np.max(abs(dhat.sum(axis=1)))),
        "historical_scalar": 0.0,
        "native_accounts": native_check_error,
        "covariance_recombination": 0.0,
        "matrix_contraction": 0.0,
    }
    old_scalar = {
        (int(v["row"]), v["model"]): v
        for v in csv_rows(
            ROOT
            / read_json(ROOT / spec["query_packet"] / "00/manifest.json")["spec"][
                "exchange_packet"
            ]
            / "02/components.csv"
        )
    }
    actions = []
    for v in meta:
        i = v["row"]
        errors["historical_scalar"] = max(
            errors["historical_scalar"],
            abs(scalar[i] - float(old_scalar[i, "receiver_tensor"]["forward"])),
        )
        if v["exact_input_noop"] and (np.any(dhat[i]) or new[i] or scalar[i]):
            raise ValueError("input no-op no longer gives zero")
        actions.append(
            {
                **v,
                "native": float(native[i]),
                "vector_readout": float(new[i]),
                "scalar_receiver": float(scalar[i]),
                "scalar_remainder": float(native[i] - new[i]),
                "logit_motion_norm": float(np.linalg.norm(native_delta[i])),
                "predicted_motion_norm": float(np.linalg.norm(dhat[i])),
                "motion_error_norm": float(np.linalg.norm(native_delta[i] - dhat[i])),
                **{f"predicted_{n}": float(parts[i, j]) for j, n in enumerate(Q.PARTS)},
                **{
                    f"native_{n}": float(native_parts[i, j])
                    for j, n in enumerate(Q.PARTS)
                },
            }
        )
    lookup = {(v["state_index"], v["query"]): v["row"] for v in meta}
    first = sorted(
        (v for v in meta if v["query"] == spec["queries"][0]),
        key=lambda v: v["state_index"],
    )
    if [v["state_index"] for v in first] != list(range(spec["unique_state_pairs"])):
        raise ValueError("paired state scope changed")
    stats, families, joint, accounts, blocks, addresses = [], [], [], [], [], []
    for role, design in itertools.product(
        spec["roles"], ("all_states", *sorted({v["arm"] for v in first}))
    ):
        selected = [
            v
            for v in first
            if v["role"] == role and (design == "all_states" or v["arm"] == design)
        ]
        ss = [v["state_index"] for v in selected]
        fs = [v["family"] for v in selected]
        w = family_weights(fs)
        info = {
            "role": role,
            "design": design,
            "states": len(ss),
            "families": len(set(fs)),
        }
        for q in spec["queries"]:
            ids = np.array([lookup[s, q] for s in ss])
            ym, sm = (M.scalar_metrics(native[ids], v[ids], w) for v in (new, scalar))
            family_values = []
            for f in sorted(set(fs)):
                mask = np.array([v == f for v in fs])
                delta = float(
                    np.mean(
                        (new[ids][mask] - native[ids][mask]) ** 2
                        - (scalar[ids][mask] - native[ids][mask]) ** 2
                    )
                )
                families.append(
                    {
                        **info,
                        "query": q,
                        "family": f,
                        "observations": int(mask.sum()),
                        "mse_delta_vector_minus_scalar": delta,
                    }
                )
                family_values.append(delta)
            ci = family_interval(family_values, spec)
            motion_mse = float(w @ np.sum((dhat[ids] - native_delta[ids]) ** 2, axis=1))
            motion_norm2 = float(w @ np.sum(native_delta[ids] ** 2, axis=1))
            stats.append(
                {
                    **info,
                    "query": q,
                    **{f"vector_{k}": v for k, v in ym.items()},
                    **{f"scalar_{k}": v for k, v in sm.items()},
                    "motion_rmse": float(np.sqrt(motion_mse)),
                    "motion_relative_rmse": float(np.sqrt(motion_mse / motion_norm2))
                    if motion_norm2
                    else None,
                    "mse_delta_mean": ci[0],
                    "mse_delta_ci_low": ci[1],
                    "mse_delta_ci_high": ci[2],
                    "positive_gain_families": sum(v < 0 for v in family_values),
                }
            )
        for qa, qb in itertools.combinations(spec["queries"], 2):
            ia, ib = (np.array([lookup[s, q] for s in ss]) for q in (qa, qb))
            v = M.remainder_covariance(native[ia], native[ib], new[ia], new[ib], w)
            oldcov = float(
                moments(np.column_stack((scalar[ia], scalar[ib])), w)["covariance"][
                    0, 1
                ]
            )
            errors["covariance_recombination"] = max(
                errors["covariance_recombination"], v["recombination_error"]
            )
            key = {**info, "query_a": qa, "query_b": qb}
            joint.append(
                {
                    **key,
                    **v,
                    "scalar_covariance": oldcov,
                    "vector_covariance_gap": v["native_covariance"] - v["PP"],
                    "scalar_covariance_gap": v["native_covariance"] - oldcov,
                    "vector_native_sign_agree": bool(
                        np.sign(v["PP"]) == np.sign(v["native_covariance"])
                    ),
                    "scalar_native_sign_agree": bool(
                        np.sign(oldcov) == np.sign(v["native_covariance"])
                    ),
                }
            )
            covs = []
            for left, right in (
                (dhat[ia], dhat[ib]),
                (dhat[ia], native_delta[ib] - dhat[ib]),
                (native_delta[ia] - dhat[ia], dhat[ib]),
                (native_delta[ia] - dhat[ia], native_delta[ib] - dhat[ib]),
            ):
                covs.append(Q.displacement_cross_covariance(left, right, w))
            blocks.append(covs)
            addresses.append({**key, "matrix_row": len(blocks) - 1})
            newcov = Q.covariance_accounts(parts[ia], parts[ib], w)
            truecov = Q.covariance_accounts(native_parts[ia], native_parts[ib], w)
            errors["matrix_contraction"] = max(
                errors["matrix_contraction"],
                abs(float(anchors[qa] @ covs[0] @ anchors[qb]) - newcov["BB"]),
            )
            accounts.append(
                {
                    **key,
                    **{f"predicted_{k}": val for k, val in newcov.items()},
                    **{f"native_{k}": val for k, val in truecov.items()},
                    **{
                        f"motion_{n}_fixed_contraction": float(
                            anchors[qa] @ covs[j] @ anchors[qb]
                        )
                        for j, n in enumerate(("PP", "PR", "RP", "RR"))
                    },
                }
            )
    if max(errors.values()) > spec["arithmetic_absolute_tolerance"]:
        raise ValueError(f"HM23 exact-account check failed: {errors}")
    (out / "02").mkdir()
    np.savez_compressed(
        out / "02/predictions.npz",
        source_order_prediction=predicted,
        predicted_delta=dhat,
        native_delta=native_delta,
        predicted_parts=parts,
        native_parts=native_parts,
        native=native,
        predicted=new,
        scalar_receiver=scalar,
    )
    np.savez_compressed(
        out / "02/remainder_covariance_blocks.npz", blocks=np.array(blocks)
    )
    for name, values in (
        ("actions", actions),
        ("action_statistics", stats),
        ("family_comparisons", families),
        ("joint_covariance", joint),
        ("account_covariance", accounts),
        ("matrix_addresses", addresses),
    ):
        write_csv(out / "02" / f"{name}.csv", values)
    write_json(
        out / "02/summary.json",
        {
            "status": "HM23_EVALUATED_NOT_GLOBAL_SCOPE_COMPLETE",
            "action_rows": len(actions),
            "states": len(first),
            "statistics_rows": len(stats),
            "family_rows": len(families),
            "joint_rows": len(joint),
            "account_rows": len(accounts),
            "matrix_blocks": len(blocks),
            "noop_rows": sum(v["exact_input_noop"] for v in meta),
            "new_shared_cpu_systems": 1,
            "source_output_columns": spec["output_columns"],
            "new_model_calls": 0,
            "held_reference_is_observer_cache": True,
            "errors": errors,
            "evaluate_seconds": time.monotonic() - start,
            "vector_native_sign_agreements": sum(
                bool(v["vector_native_sign_agree"]) for v in joint
            ),
            "scalar_native_sign_agreements": sum(
                bool(v["scalar_native_sign_agree"]) for v in joint
            ),
            "application_full_statistics": [
                v
                for v in stats
                if v["role"] == "application" and v["design"] == "full_property"
            ],
            "application_full_joint": [
                v
                for v in joint
                if v["role"] == "application" and v["design"] == "full_property"
            ],
        },
    )
    write_json(out / "02/artifact_hashes.json", artifact_hashes(out, out))
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Run resumable finite stages; existing outputs cannot be overwritten."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "compile", "evaluate", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    collector_guard(ROOT)
    try:
        if args.stage == "check":
            check(args.out, models=True)
            for name, sha in read_json(args.out / "02/artifact_hashes.json").items():
                if digest(args.out / name) != sha:
                    raise ValueError(f"changed HM23 result: {name}")
            print("HM23_FREEZE_VALID", flush=True)
        else:
            {"prepare": prepare, "compile": compile, "evaluate": evaluate}[args.stage](
                args.out
            )
    except BaseException as error:
        if args.out.exists():
            write_json(
                args.out / f"failure_{args.stage}.json",
                {"stage": args.stage, "error": repr(error)},
            )
        raise


if __name__ == "__main__":
    main()
