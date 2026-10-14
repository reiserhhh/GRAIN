#!/usr/bin/env python3
"""HM20: source-first exchange geometry, fixed actions and native-edge inventory."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_action_observability as A
from grain_research.construction import math_context_operator_decomposition as O
from grain_research.construction import math_endpoint_exchange_orbit as E
from grain_research.construction.math_midpoint_polarization import spectrum
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_action_observability import artifact_hashes, csv_rows
from scripts.run_grain_math_context_operator_decomposition import inputs
from scripts.run_grain_math_midpoint_polarization import check as polarization_check
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_endpoint_exchange_orbit_v1.json"
PLAN = ROOT / "docs/GRAIN_ENDPOINT_EXCHANGE_ORBIT_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/endpoint_exchange_orbit_r1"
CODE = (
    "grain_research/construction/math_endpoint_exchange_orbit.py",
    "scripts/run_grain_math_endpoint_exchange_orbit.py",
    "tests/test_grain_math_endpoint_exchange_orbit.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)


def address_hash(value):
    """Hash a complete deterministic context description, not its compressed vector."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def check(out, source=False):
    """Check the new freeze plus the complete verified HM19 ancestry."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["source_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"changed HM20 code: {name}")
    for name, sha in m["inputs"].items():
        if digest(out / name) != sha:
            raise ValueError(f"changed HM20 input: {name}")
    old = ROOT / m["spec"]["polarization_packet"]
    polarization_check(old, source=True)
    for name, sha in read_json(out / "00/inherited_artifacts.json").items():
        if digest(old / name) != sha:
            raise ValueError(f"changed verified HM19 result: {name}")
    if (
        digest(ROOT / m["spec"]["polarization_verification"] / "summary.json")
        != m["verification_sha256"]
    ):
        raise ValueError("changed HM19 verification")
    if source:
        for name, sha in read_json(out / "01/source_manifest.json")[
            "artifacts"
        ].items():
            if digest(out / name) != sha:
                raise ValueError(f"changed source-only orbit: {name}")
    return m


def prepare(out):
    """Freeze all addresses and verified inputs; no target fitting or new observations."""
    if out.exists():
        raise FileExistsError("preserve prior endpoint-exchange output")
    spec = read_json(SPEC)
    old, verification = (
        ROOT / spec[n] for n in ("polarization_packet", "polarization_verification")
    )
    polarization_check(old, source=True)
    v = read_json(verification / "summary.json")
    if (
        v["status"] != "VERIFIED_FINITE_HM19_NOT_GLOBAL_SCOPE_COMPLETE"
        or digest(old / "00/manifest.json") != v["primary_manifest_sha256"]
    ):
        raise ValueError("verified HM19 required")
    if (
        digest(ROOT / "scripts/verify_grain_math_midpoint_polarization.py")
        != v["verifier_sha256"]
    ):
        raise ValueError("changed HM19 verifier")
    ledger = read_json(verification / "frozen_primary_artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(old / name) != sha:
            raise ValueError(f"changed inherited artifact: {name}")
    meta = read_json(old / "00/metadata.json")
    if len(meta) != spec["paired_query_rows"]:
        raise ValueError("address scope changed")
    (out / "00").mkdir(parents=True)
    write_json(out / "00/metadata.json", meta)
    write_json(out / "00/inherited_artifacts.json", ledger)
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
    print({"status": "HM20_FROZEN", "addresses": len(meta)}, flush=True)


def source(out):
    """Freeze balanced source orbit Grams before loading validation/application scores."""
    spec = check(out)["spec"]
    if (out / "01").exists():
        raise FileExistsError("preserve source geometry")
    start = time.monotonic()
    _, _, parameters, states, views, _, _ = inputs(ROOT / spec["operator_packet"])
    x, c, _, _, _ = views
    k = float(parameters["delta_scale"] / parameters["context_scale"])
    meta = {
        r["state_index"]: r
        for r in read_json(out / "00/metadata.json")
        if r["query"] == spec["queries"][0] and r["role"] == "source"
    }
    if set(meta) != set(states.tolist()) or len(states) != spec["source_states"]:
        raise ValueError("source addresses changed")
    (out / "01").mkdir()
    rows, error = [], 0.0
    for design in spec["source_designs"]:
        ids = np.array(
            [s for s in states if design == "all_states" or meta[s]["arm"] == design]
        )
        b = E.orbit_grams(x[ids], c[ids], k)
        xr, cr = E.exchange(x[ids], c[ids], k)
        actual_reverse = E.orbit_grams(xr, cr, k)
        error = max(error, float(np.max(abs(actual_reverse["U"] - b["U_reverse"]))))
        saved = {"state_indices": ids, **b}
        for name in spec["source_measures"]:
            w = (
                np.full(len(ids), 1 / len(ids))
                if name == "uniform"
                else A.family_weights([meta[s]["family"] for s in ids])
            )
            saved[name + "_weights"] = w
            for part in ("U", "V", "W", "stack_U_V", "orbit_U_U_reverse"):
                g = b[part]
                weight = np.tile(w / 2, 2) if len(g) == 2 * len(ids) else w
                d = np.sqrt(weight)
                eig = spectrum(
                    d[:, None] * g * d[None, :], spec["machine_tolerance_multiplier"]
                )
                saved[part + "_" + name + "_eigenvalues"] = eig["eigenvalues"]
                rows.append(
                    {
                        "design": design,
                        "measure": name,
                        "part": part,
                        "original_states": len(ids),
                        "virtual_gram_rows": len(g),
                        "observable_rank": eig["rank"],
                        "tolerance": eig["tolerance"],
                        "eigen_min": eig["eigen_min"],
                        "largest": eig["largest"],
                        "negative_beyond_tolerance": eig["negative_beyond_tolerance"],
                    }
                )
        np.savez_compressed(out / "01" / (design + "_geometry.npz"), **saved)
    write_csv(out / "01/source_spectra.csv", rows)
    rho, metric = E.lifted_metric()
    write_json(
        out / "01/source_manifest.json",
        {
            "held_targets_loaded": False,
            "source_states": len(states),
            "kappa": k,
            "exchange_gram_max_error": error,
            "lifted_rho": rho.tolist(),
            "lifted_metric": metric.tolist(),
            "metric_invariance_error": float(
                np.max(abs(rho.T @ metric @ rho - metric))
            ),
            "virtual_rows_are_new_native_observations": False,
            "source_seconds": time.monotonic() - start,
            "artifacts": artifact_hashes(out / "01", out),
        },
    )
    print(
        {"status": "HM20_SOURCE_FROZEN", "spectra": len(rows), "exchange_error": error},
        flush=True,
    )


def fixed_orbit_actions(x, c, midpoint, shuffled, states, alpha, k, query, model):
    """Evaluate both orientations explicitly with unchanged source coefficients."""
    ctx = {
        "receiver_tensor": c,
        "endpoint_potential": midpoint,
        "shuffled_context": shuffled,
    }[model]
    args = x[states], ctx[states], alpha[query, model], k
    common = O.action_parts(x, c, *args)
    xr, cr = E.exchange(x, c, k)
    common_r = O.action_parts(xr, cr, *args)
    if model == "endpoint_potential":
        f = O.action_parts(x, midpoint, *args)
        r = O.action_parts(-x, midpoint, *args)
        forward, reverse = f["linear"] + f["symmetric"], r["linear"] + r["symmetric"]
        common_forward = common["linear"] + common["symmetric"]
        common_reverse = common_r["linear"] + common_r["symmetric"]
    else:
        actual_context = shuffled if model == "shuffled_context" else c
        ax, ac = E.exchange(x, actual_context, k)
        f, r = O.action_parts(x, actual_context, *args), O.action_parts(ax, ac, *args)
        forward = f["linear"] + f["symmetric"] + f["skew"]
        reverse = r["linear"] + r["symmetric"] + r["skew"]
        common_forward = common["linear"] + common["symmetric"] + common["skew"]
        common_reverse = common_r["linear"] + common_r["symmetric"] + common_r["skew"]
    even, odd = E.project_actions(forward, reverse)
    common_even, common_odd = E.project_actions(common_forward, common_reverse)
    return {
        "forward": forward,
        "reverse_virtual": reverse,
        "even": even,
        "odd": odd,
        "v": common["finite_step"],
        "skew_common": common["skew"]
        if model != "endpoint_potential"
        else np.zeros(len(x)),
        "common_forward": common_forward,
        "common_reverse_virtual": common_reverse,
        "common_even": common_even,
        "common_odd": common_odd,
    }


def native_inventory(out, spec, metadata):
    """Audit actual context addresses; reverse subtraction is explicitly only inferred."""
    capture = read_json(ROOT / spec["span_packet"] / "00/design.json")
    lookup = {(r["doc_id"], r["query"]): r for r in capture}
    consumed, descriptors, rows = {}, [], []
    scores = {}
    for a in metadata:
        addr = lookup[a["receiver_doc"], a["query"]]
        if a["positions"] != addr["spans"][a["span"]]["positions"]:
            raise ValueError("assigned positions differ from original full context")
        unit, ref = (read_json(ROOT / a[p]) for p in ("unit_path", "reference_path"))
        for field in ("unit_path", "reference_path"):
            consumed[a[field]] = digest(ROOT / a[field])
        if unit["live_body_error"] != 0 or unit["address"]["query"] != a["query"]:
            raise ValueError("native observation mismatch")
        context = {
            "input_ids": addr["input_ids"],
            "layer": spec["layer"],
            "positions": a["positions"],
            "span": a["span"],
        }
        descriptor = {
            "row": a["row"],
            "context_sha256": address_hash(context),
            "reference_sha256": a["reference_state_sha256"],
            "target_sha256": a["target_state_sha256"],
        }
        descriptors.append(descriptor)
        y = unit["native"]["conditional_score"] - ref["native"]["conditional_score"]
        scores[a["row"]] = y
        rows.append(
            {
                **a,
                **descriptor,
                "native_forward": y,
                "native_reverse_inferred_only": -y,
                "reference_score": ref["native"]["conditional_score"],
                "target_score": unit["native"]["conditional_score"],
                "complete_input_ids_sha256": address_hash(addr["input_ids"]),
                "reference_record_sha256": consumed[a["reference_path"]],
                "target_record_sha256": consumed[a["unit_path"]],
            }
        )
    index = {
        (r["receiver_doc"], r["donor_doc"], r["query"], r["span"], r["arm"]): r["row"]
        for r in rows
    }
    for row, inv in zip(rows, E.inverse_inventory(descriptors), strict=True):
        row.update(inv)
        ids = row.pop("inverse_rows")
        row["independent_inverse_rows"] = ";".join(map(str, ids))
        row["independent_inverse_count"] = len(ids)
        row["observed_inverse_sum_max"] = max(
            (abs(scores[i] + scores[row["row"]]) for i in ids), default=None
        )
        other = index.get(
            (
                row["donor_doc"],
                row["receiver_doc"],
                row["query"],
                row["span"],
                row["arm"],
            )
        )
        row["swapped_document_candidate"] = other
        row["swapped_document_is_same_context"] = (
            other is not None
            and descriptors[other]["context_sha256"] == row["context_sha256"]
        )
    write_csv(out / "02/native_inverse_inventory.csv", rows)
    write_json(out / "02/consumed_native_hashes.json", consumed)
    return rows, scores


def evaluate(out):
    """Test frozen laws, record all arms and inspect joint action without refitting."""
    spec = check(out, source=True)["spec"]
    if (out / "02").exists():
        raise FileExistsError("preserve evaluated orbit packet")
    start = time.monotonic()
    _, _, parameters, states, views, alpha, _ = inputs(ROOT / spec["operator_packet"])
    x, c, midpoint, shuffled, _ = views
    k = float(parameters["delta_scale"] / parameters["context_scale"])
    addresses = read_json(out / "00/metadata.json")
    first = sorted(
        (r for r in addresses if r["query"] == spec["queries"][0]),
        key=lambda r: r["state_index"],
    )
    if len(first) != spec["unique_state_pairs"] or [
        r["state_index"] for r in first
    ] != list(range(len(x))):
        raise ValueError("all original states required")
    (out / "02").mkdir()
    inventory, native = native_inventory(out, spec, addresses)
    previous = {
        (r["query"], r["model"], int(r["state_index"])): r
        for r in csv_rows(ROOT / spec["polarization_packet"] / "02/components.csv")
        if r["source_projection_design"] == "all_states"
    }
    values, components = {}, []
    errors = {
        "frozen_replay": 0.0,
        "even_identity": 0.0,
        "common_even_identity": 0.0,
        "involution": 0.0,
    }
    xx, cc = E.exchange(*E.exchange(x, c, k), k)
    errors["involution"] = float(max(np.max(abs(xx - x)), np.max(abs(cc - c))))
    for query, model in itertools.product(spec["queries"], spec["models"]):
        v = fixed_orbit_actions(
            x, c, midpoint, shuffled, states, alpha, k, query, model
        )
        by_state = {r["state_index"]: r for r in addresses if r["query"] == query}
        v["native"] = np.array([native[by_state[s]["row"]] for s in range(len(x))])
        values[query, model] = v
        errors["even_identity"] = max(
            errors["even_identity"],
            float(
                np.max(
                    abs(v["even"] + (0 if model == "endpoint_potential" else v["v"]))
                )
            ),
        )
        errors["common_even_identity"] = max(
            errors["common_even_identity"],
            float(np.max(abs(v["common_even"] + v["v"]))),
        )
        for s in range(len(x)):
            old = previous[query, model, s]
            errors["frozen_replay"] = max(
                errors["frozen_replay"],
                abs(v["forward"][s] - float(old["prediction"])),
                abs(v["native"][s] - float(old["native"])),
            )
            components.append(
                {
                    **by_state[s],
                    "model": model,
                    "reverse_status": "virtual_fixed_context"
                    if model == "shuffled_context"
                    else "formula_not_native_measurement",
                    **{name: float(a[s]) for name, a in v.items()},
                }
            )
    if max(errors.values()) > 1e-9:
        raise ValueError(f"exchange action identity failed: {errors}")
    stats, joint = [], []
    for role in ("source", "validation", "application"):
        for design in ("all_states", *sorted({r["arm"] for r in first})):
            selected = [
                r
                for r in first
                if r["role"] == role and (design == "all_states" or r["arm"] == design)
            ]
            ids, w = (
                [r["state_index"] for r in selected],
                A.family_weights([r["family"] for r in selected]),
            )
            info = {
                "role": role,
                "design": design,
                "states": len(ids),
                "families": len({r["family"] for r in selected}),
            }
            for model in spec["models"]:
                for query in spec["queries"]:
                    v = values[query, model]
                    f, e, o, n = (
                        v[p][ids] for p in ("forward", "even", "odd", "native")
                    )
                    original, balanced = E.balanced_cross(e, o, w)
                    rms = lambda a, w=w: float(np.sqrt(w @ (a * a)))
                    stats.append(
                        {
                            **info,
                            "model": model,
                            "query": query,
                            "forward_rms": rms(f),
                            "even_rms": rms(e),
                            "odd_rms": rms(o),
                            "native_rms": rms(n),
                            "forward_rmse": rms(f - n),
                            "odd_rmse": rms(o - n),
                            "odd_minus_forward_rmse": rms(o - n) - rms(f - n),
                            "even_forward_rms_ratio": rms(e) / rms(f)
                            if rms(f) > 0
                            else None,
                            "unbalanced_even_odd_inner": original,
                            "balanced_even_odd_inner": balanced,
                        }
                    )
                for qa, qb in itertools.combinations(spec["queries"], 2):
                    a, b = values[qa, model], values[qb, model]
                    for channel in ("forward", "odd"):
                        pa, pb, na, nb = (
                            a[channel][ids],
                            b[channel][ids],
                            a["native"][ids],
                            b["native"][ids],
                        )
                        m = A.moments(
                            np.column_stack((pa, pb, na - pa, nb - pb, na, nb)), w
                        )["covariance"]
                        recombined = m[0, 1] + m[0, 3] + m[2, 1] + m[2, 3]
                        joint.append(
                            {
                                **info,
                                "model": model,
                                "query_a": qa,
                                "query_b": qb,
                                "channel": channel,
                                "native_covariance": float(m[4, 5]),
                                "compiled_covariance": float(m[0, 1]),
                                "compiled_a_remainder_b": float(m[0, 3]),
                                "remainder_a_compiled_b": float(m[2, 1]),
                                "remainder_covariance": float(m[2, 3]),
                                "joint_gap": float(m[4, 5] - m[0, 1]),
                                "recombination_error": float(abs(m[4, 5] - recombined)),
                            }
                        )
    write_csv(out / "02/components.csv", components)
    write_csv(out / "02/action_statistics.csv", stats)
    write_csv(out / "02/native_remainder_covariance.csv", joint)
    write_json(
        out / "02/summary.json",
        {
            "status": "HM20_ENDPOINT_EXCHANGE_EVALUATED_NOT_GLOBAL_COMPLETE",
            "component_rows": len(components),
            "statistics_rows": len(stats),
            "covariance_rows": len(joint),
            "native_inventory_rows": len(inventory),
            "nontrivial_observed_inverse_rows": sum(
                r["nontrivial_observed_inverse"] for r in inventory
            ),
            "noop_rows": sum(r["exact_input_noop"] for r in inventory),
            "swapped_document_candidates": sum(
                r["swapped_document_candidate"] is not None for r in inventory
            ),
            "same_context_swapped_document_candidates": sum(
                r["swapped_document_is_same_context"] for r in inventory
            ),
            "errors": errors,
            "source_manifest_sha256": digest(out / "01/source_manifest.json"),
            "new_source_fits": 0,
            "new_model_calls": 0,
            "private_text_reads": 0,
            "api_calls": 0,
            "reverse_score_subtraction_is_inferred_not_independently_observed": True,
            "original_full_scope_complete": False,
            "evaluate_seconds": time.monotonic() - start,
        },
    )
    write_json(out / "02/artifact_hashes.json", artifact_hashes(out, out))
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Run a single bounded stage without overwriting any prior experiment."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "source", "evaluate"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    with collector_guard(a.out):
        {"prepare": prepare, "source": source, "evaluate": evaluate}[a.stage](a.out)


if __name__ == "__main__":
    main()
