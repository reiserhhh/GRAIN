#!/usr/bin/env python3
"""HM19 source-first observation polarization and fixed-query action analysis."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_action_observability as A
from grain_research.construction import math_context_operator_decomposition as O
from grain_research.construction import math_midpoint_polarization as M
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_action_observability import artifact_hashes, csv_rows
from scripts.run_grain_math_action_observability import check as observation_check
from scripts.run_grain_math_context_operator_decomposition import inputs
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_midpoint_polarization_v1.json"
PLAN = ROOT / "docs/GRAIN_MIDPOINT_POLARIZATION_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/midpoint_polarization_r1"
CODE = (
    "grain_research/construction/math_midpoint_polarization.py",
    "scripts/run_grain_math_midpoint_polarization.py",
    "tests/test_grain_math_midpoint_polarization.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)
CHANNELS = ("u", "v", "w", "v_supported", "v_new", "prediction", "native", "remainder")


def check(out, source=False):
    """Validate all frozen inputs and optional source-only projection artifacts."""
    manifest = read_json(out / "00/manifest.json")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"changed HM19 source: {name}")
    for name, expected in manifest["inputs"].items():
        if digest(out / name) != expected:
            raise ValueError(f"changed HM19 input: {name}")
    old = ROOT / manifest["spec"]["observation_packet"]
    observation_check(old, geometry=True)
    for name, expected in read_json(out / "00/inherited_artifacts.json").items():
        if digest(old / name) != expected:
            raise ValueError(f"changed HM18 artifact: {name}")
    verification = ROOT / manifest["spec"]["observation_verification"]
    if digest(verification / "summary.json") != manifest["verification_sha256"]:
        raise ValueError("HM18 verification changed")
    if source:
        for name, expected in read_json(out / "01/source_manifest.json")[
            "artifacts"
        ].items():
            if digest(out / name) != expected:
                raise ValueError(f"source projection changed: {name}")
    return manifest


def prepare(out):
    """Seal protocols and inherited verified results, without starting new fitting."""
    if out.exists():
        raise FileExistsError("preserve prior HM19 results")
    spec = read_json(SPEC)
    old = ROOT / spec["observation_packet"]
    observation_check(old, geometry=True)
    verification = ROOT / spec["observation_verification"]
    v = read_json(verification / "summary.json")
    if (
        v["status"] != "VERIFIED_FINITE_HM18_NOT_GLOBAL_SCOPE_COMPLETE"
        or digest(ROOT / "scripts/verify_grain_math_action_observability.py")
        != v["verifier_sha256"]
        or digest(old / "00/manifest.json") != v["primary_manifest_sha256"]
    ):
        raise ValueError("verified HM18 required")
    ledger = read_json(verification / "frozen_primary_artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(old / name) != expected:
            raise ValueError(f"verified inherited artifact changed: {name}")
    meta = read_json(old / "00/metadata.json")
    if len(meta) != spec["paired_query_rows"]:
        raise ValueError("query address scope changed")
    (out / "00").mkdir(parents=True)
    write_json(out / "00/metadata.json", meta)
    write_json(out / "00/inherited_artifacts.json", ledger)
    for name in CODE:
        dest = out / "00/code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "spec": spec,
            "source_files": {n: digest(ROOT / n) for n in CODE},
            "inputs": artifact_hashes(out / "00", out),
            "verification_sha256": digest(verification / "summary.json"),
            "inherited_manifest_sha256": digest(old / "00/manifest.json"),
            "frozen_before_analysis": True,
            "new_source_fits": 0,
            "new_model_calls": 0,
        },
    )
    print({"status": "HM19_FROZEN", "addresses": len(meta)}, flush=True)


def source(out):
    """Freeze U/V/W geometry and U-span projectors before loading held targets."""
    manifest = check(out)
    if (out / "01").exists():
        raise FileExistsError("preserve source freeze")
    start, spec = time.monotonic(), manifest["spec"]
    operator = ROOT / spec["operator_packet"]
    _, _, parameters, states, views, _, _ = inputs(operator)
    x, c, midpoint, _, _ = views
    kappa = float(parameters["delta_scale"] / parameters["context_scale"])
    meta = {
        r["state_index"]: r
        for r in read_json(out / "00/metadata.json")
        if r["query"] == spec["queries"][0] and r["role"] == "source"
    }
    if set(states.tolist()) != set(meta) or len(states) != spec["source_states"]:
        raise ValueError("source addresses changed")
    (out / "01").mkdir()
    rows, errors = [], {"midpoint_gram": 0.0, "skew_invariance": 0.0}
    for design in spec["source_designs"]:
        selected = np.array(
            [s for s in states if design == "all_states" or meta[s]["arm"] == design]
        )
        b = M.gram_blocks(x[selected], c[selected], kappa)
        receiver = A.dyad_grams(x[selected], c[selected])
        endpoint = A.dyad_grams(x[selected], midpoint[selected])
        errors["midpoint_gram"] = max(
            errors["midpoint_gram"], float(np.max(abs(b["W"] - endpoint["S"])))
        )
        errors["skew_invariance"] = max(
            errors["skew_invariance"], float(np.max(abs(receiver["H"] - endpoint["H"])))
        )
        saved = {"state_indices": selected, **b}
        for measure in spec["source_measures"]:
            w = (
                np.full(len(selected), 1 / len(selected))
                if measure == "uniform"
                else A.family_weights([meta[s]["family"] for s in selected])
            )
            p = M.source_projection(
                b["U"], b["UV"], b["V"], w, spec["machine_tolerance_multiplier"]
            )
            saved[measure + "_weights"] = w
            saved[measure + "_coefficient"] = p["coefficient"]
            saved[measure + "_residual"] = p["residual"]
            block = np.block([[b["U"], b["UV"]], [b["UV"].T, b["V"]]])
            for part, gram, weights in [(k, b[k], w) for k in ("U", "V", "W")] + [
                ("stack_U_V", block, np.tile(w, 2)),
                ("V_perp", p["residual"], w),
            ]:
                d = np.sqrt(weights)
                eig = M.spectrum(
                    d[:, None] * gram * d[None, :], spec["machine_tolerance_multiplier"]
                )
                saved[part + "_" + measure + "_eigenvalues"] = eig["eigenvalues"]
                rows.append(
                    {
                        "design": design,
                        "measure": measure,
                        "part": part,
                        "observations": len(selected),
                        "gram_rows": len(gram),
                        "observable_rank": eig["rank"],
                        "tolerance": eig["tolerance"],
                        "eigen_min": eig["eigen_min"],
                        "largest": eig["largest"],
                        "negative_beyond_tolerance": eig["negative_beyond_tolerance"],
                        "range_defect_max": p["range_defect_max"],
                        "residual_trace": float(w @ np.diag(p["residual"])),
                    }
                )
        np.savez_compressed(out / "01" / (design + "_geometry.npz"), **saved)
    write_csv(out / "01/source_spectra.csv", rows)
    write_json(
        out / "01/source_manifest.json",
        {
            "held_targets_loaded": False,
            "new_source_fits": 0,
            "source_states": len(states),
            "source_designs": spec["source_designs"],
            "kappa": kappa,
            "identities_max_errors": errors,
            "source_seconds": time.monotonic() - start,
            "artifacts": artifact_hashes(out / "01", out),
        },
    )
    print(
        {"status": "HM19_SOURCE_FROZEN", "spectra": len(rows), "errors": errors},
        flush=True,
    )


def fixed_actions(x, c, contexts, states, alphas, kappa, queries, models):
    """Evaluate common U/V actions and each previously fitted model unchanged."""
    values = {}
    for query, model in itertools.product(queries, models):
        ctx, alpha = contexts[model], alphas[query, model]
        common = O.action_parts(x, c, x[states], ctx[states], alpha, kappa)
        actual = O.action_parts(x, ctx, x[states], ctx[states], alpha, kappa)
        if model == "endpoint_potential":
            actual.update(skew=np.zeros(len(x)), finite_step=np.zeros(len(x)))
        values[query, model] = {
            "linear": common["linear"],
            "u": common["symmetric"],
            "v": common["finite_step"],
            "w": common["symmetric"] + common["finite_step"],
            "frozen_symmetric": actual["symmetric"],
            "frozen_skew": actual["skew"],
            "prediction": actual["linear"] + actual["symmetric"] + actual["skew"],
            "receiver_potential_same_coefficients": common["linear"]
            + common["symmetric"]
            + common["finite_step"],
        }
    return values


def evaluate(out):
    """Decompose frozen quadratic action and residual covariance on all addresses."""
    manifest = check(out, source=True)
    if (out / "02").exists():
        raise FileExistsError("preserve evaluated packet")
    start, spec = time.monotonic(), manifest["spec"]
    _, old, parameters, states, views, alphas, _ = inputs(
        ROOT / spec["operator_packet"]
    )
    x, c, midpoint, shuffled, _ = views
    kappa = float(parameters["delta_scale"] / parameters["context_scale"])
    contexts = {
        "receiver_tensor": c,
        "endpoint_potential": midpoint,
        "shuffled_context": shuffled,
    }
    values = fixed_actions(
        x, c, contexts, states, alphas, kappa, spec["queries"], spec["models"]
    )
    addresses = read_json(out / "00/metadata.json")
    first = sorted(
        (r for r in addresses if r["query"] == spec["queries"][0]),
        key=lambda r: r["state_index"],
    )
    if len(first) != spec["unique_state_pairs"] or [
        r["state_index"] for r in first
    ] != list(range(len(x))):
        raise ValueError("state scope/order changed")
    native = np.empty(len(addresses))
    for role in ("source", "held"):
        with np.load(old / f"00/{role}_targets.npz") as arrays:
            native[arrays["indices"]] = arrays["y"]
    original = {
        (r["query"], r["model"], int(r["state_index"])): r
        for r in csv_rows(ROOT / spec["operator_packet"] / "02/components.csv")
    }
    components, stats, sharing, covariance = [], [], [], []
    replay_error, recombination_error, native_cov_error = 0.0, 0.0, 0.0
    for source_design in spec["source_designs"]:
        with np.load(out / "01" / (source_design + "_geometry.npz")) as a:
            selected = a["state_indices"].copy()
            coefficient = a[spec["action_projection_measure"] + "_coefficient"].copy()
        current = {}
        for query, model in itertools.product(spec["queries"], spec["models"]):
            v = {name: a.copy() for name, a in values[query, model].items()}
            v["v_supported"] = M.projected_action(
                x, x[selected], c[selected], kappa, coefficient, v["u"][selected]
            )
            v["v_new"] = v["v"] - v["v_supported"]
            by_query = {r["state_index"]: r for r in addresses if r["query"] == query}
            v["native"] = np.array([native[by_query[s]["row"]] for s in range(len(x))])
            v["remainder"] = v["native"] - v["prediction"]
            current[query, model] = v
            for s in range(len(x)):
                meta = by_query[s]
                previous = original[query, model, s]
                replay_error = max(
                    replay_error,
                    abs(v["prediction"][s] - float(previous["predicted_action"])),
                    abs(v["native"][s] - float(previous["observed_action"])),
                )
                recombination_error = max(
                    recombination_error,
                    abs(v["w"][s] - v["u"][s] - v["v"][s]),
                    abs(v["v"][s] - v["v_supported"][s] - v["v_new"][s]),
                )
                components.append(
                    {
                        **meta,
                        "source_projection_design": source_design,
                        "model": model,
                        **{name: float(a[s]) for name, a in v.items()},
                    }
                )
        for role in ("source", "validation", "application"):
            for design in ("all_states", *sorted({r["arm"] for r in first})):
                members = [
                    r
                    for r in first
                    if r["role"] == role
                    and (design == "all_states" or r["arm"] == design)
                ]
                selected = [r["state_index"] for r in members]
                weights = A.family_weights([r["family"] for r in members])
                info = {
                    "role": role,
                    "design": design,
                    "source_projection_design": source_design,
                    "observations": len(selected),
                    "families": len({r["family"] for r in members}),
                }
                for model in spec["models"]:
                    for query in spec["queries"]:
                        v = current[query, model]
                        mom = A.moments(
                            np.column_stack([v[p][selected] for p in CHANNELS]), weights
                        )
                        rms = np.sqrt(np.maximum(0, np.diag(mom["gram"])))
                        stats.append(
                            {
                                **info,
                                "query": query,
                                "model": model,
                                **{
                                    p + "_rms": float(rms[j])
                                    for j, p in enumerate(CHANNELS)
                                },
                                **{
                                    p + "_mean": float(mom["mean"][j])
                                    for j, p in enumerate(CHANNELS)
                                },
                                "v_new_v_rms_ratio": float(rms[4] / rms[1])
                                if rms[1] > 0
                                else None,
                                "v_new_prediction_rms_ratio": float(rms[4] / rms[5])
                                if rms[5] > 0
                                else None,
                                "v_new_supported_inner": float(mom["gram"][4, 3]),
                            }
                        )
                    for qa, qb in itertools.combinations(spec["queries"], 2):
                        va, vb = current[qa, model], current[qb, model]
                        for part in CHANNELS:
                            mom = A.moments(
                                np.column_stack(
                                    (va[part][selected], vb[part][selected])
                                ),
                                weights,
                            )
                            g, cov = mom["gram"], mom["covariance"]
                            sharing.append(
                                {
                                    **info,
                                    "model": model,
                                    "query_a": qa,
                                    "query_b": qb,
                                    "part": part,
                                    "inner_product": float(g[0, 1]),
                                    "covariance": float(cov[0, 1]),
                                    "cosine": A.cosine(g[0, 1], g[0, 0], g[1, 1]),
                                    "correlation": A.cosine(
                                        cov[0, 1], cov[0, 0], cov[1, 1]
                                    ),
                                }
                            )
                        mom = A.moments(
                            np.column_stack(
                                (
                                    va["prediction"][selected],
                                    vb["prediction"][selected],
                                    va["remainder"][selected],
                                    vb["remainder"][selected],
                                    va["native"][selected],
                                    vb["native"][selected],
                                )
                            ),
                            weights,
                        )
                        cov = mom["covariance"]
                        explained = cov[0, 1] + cov[0, 3] + cov[2, 1] + cov[2, 3]
                        native_cov_error = max(
                            native_cov_error, abs(cov[4, 5] - explained)
                        )
                        covariance.append(
                            {
                                **info,
                                "model": model,
                                "query_a": qa,
                                "query_b": qb,
                                "native_covariance": float(cov[4, 5]),
                                "compiled_covariance": float(cov[0, 1]),
                                "compiled_a_remainder_b": float(cov[0, 3]),
                                "remainder_a_compiled_b": float(cov[2, 1]),
                                "remainder_covariance": float(cov[2, 3]),
                                "joint_gap": float(cov[4, 5] - cov[0, 1]),
                                "recombination_error": float(
                                    abs(cov[4, 5] - explained)
                                ),
                            }
                        )
    if max(replay_error, recombination_error, native_cov_error) > 1e-9:
        raise ValueError("frozen action/recombination identity failed")
    (out / "02").mkdir()
    for name, rows in {
        "components": components,
        "action_statistics": stats,
        "target_sharing": sharing,
        "native_remainder_covariance": covariance,
    }.items():
        write_csv(out / "02" / (name + ".csv"), rows)
    write_json(
        out / "02/summary.json",
        {
            "status": "HM19_MIDPOINT_POLARIZATION_EVALUATED_NOT_GLOBAL_COMPLETE",
            "paired_query_rows": len(addresses),
            "unique_state_pairs": len(first),
            "component_rows": len(components),
            "statistics_rows": len(stats),
            "sharing_rows": len(sharing),
            "covariance_rows": len(covariance),
            "frozen_prediction_replay_max_error": replay_error,
            "recombination_max_error": recombination_error,
            "native_covariance_recombination_max_error": native_cov_error,
            "source_manifest_sha256": digest(out / "01/source_manifest.json"),
            "new_source_fits": 0,
            "new_model_calls": 0,
            "api_calls": 0,
            "private_text_reads": 0,
            "original_full_scope_complete": False,
            "evaluate_seconds": time.monotonic() - start,
        },
    )
    write_json(out / "02/artifact_hashes.json", artifact_hashes(out, out))
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Execute one finite stage without overwriting previous results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "source", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with collector_guard(args.out):
        {"prepare": prepare, "source": source, "evaluate": evaluate}[args.stage](
            args.out
        )


if __name__ == "__main__":
    main()
