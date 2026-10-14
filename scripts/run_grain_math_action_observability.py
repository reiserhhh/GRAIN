#!/usr/bin/env python3
"""HM18: frozen-source observation kernels and design-dependent target sharing."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_action_observability as M
from grain_research.construction import math_context_operator_decomposition as O
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_context_operator_decomposition import (
    check as operator_check,
)
from scripts.run_grain_math_context_operator_decomposition import inputs
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_action_observability_v1.json"
PLAN = ROOT / "docs/GRAIN_ACTION_OBSERVABILITY_PLAN_V1.md"
DEFAULT = ROOT / "results/grain_math_response_v1/action_observability_r1"
CODE = (
    "grain_research/construction/math_action_observability.py",
    "scripts/run_grain_math_action_observability.py",
    "tests/test_grain_math_action_observability.py",
    str(SPEC.relative_to(ROOT)),
    str(PLAN.relative_to(ROOT)),
)
PARTS = (*M.CHANNELS, "predicted_action", "observed_action", "coupling")


def csv_rows(path):
    """Read addressed tabular results, never treating sidecar files as tables."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def artifact_hashes(folder, relative):
    """Hash packet outputs while excluding external-volume metadata sidecars."""
    return {
        str(p.relative_to(relative)): digest(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not p.name.startswith("._")
    }


def check(out, geometry=False):
    """Verify canonical HM17 code, all accepted outputs, and this packet's freeze."""
    manifest = read_json(out / "00/manifest.json")
    for name, expected in manifest["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"changed HM18 source: {name}")
    for name, expected in manifest["inputs"].items():
        if digest(out / name) != expected:
            raise ValueError(f"changed HM18 input: {name}")
    packet = ROOT / manifest["spec"]["operator_packet"]
    operator_check(packet, geometry=True)
    for name, expected in read_json(out / "00/operator_artifacts.json").items():
        if digest(packet / name) != expected:
            raise ValueError(f"changed verified HM17 artifact: {name}")
    verified = ROOT / manifest["spec"]["operator_verification"]
    if digest(verified / "summary.json") != manifest["verification_sha256"]:
        raise ValueError("HM17 verification changed")
    if geometry:
        for name, expected in read_json(out / "01/source_manifest.json")[
            "artifacts"
        ].items():
            if digest(out / name) != expected:
                raise ValueError(f"changed source observation geometry: {name}")
    return manifest


def prepare(out):
    """Freeze designs, addresses and inherited results before any new diagnostics."""
    if out.exists():
        raise FileExistsError("preserve prior HM18 packet")
    spec = read_json(SPEC)
    old = ROOT / spec["operator_packet"]
    inherited = operator_check(old, geometry=True)
    verifier = ROOT / spec["operator_verification"]
    verified = read_json(verifier / "summary.json")
    if (
        verified["status"] != "VERIFIED_FINITE_HM17_NOT_GLOBAL_SCOPE_COMPLETE"
        or digest(ROOT / "scripts/verify_grain_math_context_operator_decomposition.py")
        != verified["verifier_sha256"]
        or digest(old / "00/manifest.json") != verified["primary_manifest_sha256"]
    ):
        raise ValueError("canonical verified HM17 required")
    ledger = read_json(verifier / "frozen_primary_artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(old / name) != expected:
            raise ValueError(f"verified input changed: {name}")
    context = ROOT / inherited["spec"]["context_packet"]
    metadata = read_json(context / "00/metadata.json")
    if len(metadata) != spec["paired_query_rows"]:
        raise ValueError("original addresses missing")
    first = [r for r in metadata if r["query"] == spec["queries"][0]]
    if len({r["state_index"] for r in first}) != spec["unique_state_pairs"]:
        raise ValueError("original state pairs missing")
    if len({r["arm"] for r in first}) != 7:
        raise ValueError("original intervention arms missing")
    if spec["queries"] != inherited["spec"]["queries"]:
        raise ValueError("queries changed")
    (out / "00").mkdir(parents=True)
    write_json(out / "00/operator_artifacts.json", ledger)
    write_json(out / "00/metadata.json", metadata)
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    write_json(
        out / "00/manifest.json",
        {
            "spec": spec,
            "source_files": {n: digest(ROOT / n) for n in CODE},
            "inputs": artifact_hashes(out / "00", out),
            "verification_sha256": digest(verifier / "summary.json"),
            "inherited_manifest_sha256": digest(old / "00/manifest.json"),
            "frozen_before_analysis": True,
            "new_model_calls": 0,
            "new_source_fits": 0,
        },
    )
    print({"status": "HM18_DESIGNS_FROZEN", "addresses": len(metadata)}, flush=True)


def source(out):
    """Construct empirical operator observation kernels without reading held outputs."""
    manifest = check(out)
    if (out / "01").exists():
        raise FileExistsError("preserve frozen source geometry")
    start, spec = time.monotonic(), manifest["spec"]
    packet = ROOT / spec["operator_packet"]
    _, _, parameters, states, views, alphas, baselines = inputs(packet)
    x, c, midpoint, shuffled, _ = views
    kappa = float(parameters["delta_scale"] / parameters["context_scale"])
    with np.load(packet / "01/source_support.npz") as arrays:
        support_rank = int(arrays["rank"])
    if len(states) != spec["source_states"]:
        raise ValueError("source support changed")
    meta = {
        r["state_index"]: r
        for r in read_json(out / "00/metadata.json")
        if r["query"] == spec["queries"][0] and r["role"] == "source"
    }
    if set(states.tolist()) != set(meta):
        raise ValueError("source addresses do not match source operators")
    contexts = {
        "receiver_tensor": c,
        "endpoint_potential": midpoint,
        "shuffled_context": shuffled,
    }
    values = {}
    for query in spec["queries"]:
        values[query, "full_linear"] = np.column_stack(
            (x[states] @ baselines[query], np.zeros((len(states), 3)))
        )
        for kind, ctx in contexts.items():
            parts = O.action_parts(
                x[states],
                ctx[states],
                x[states],
                ctx[states],
                alphas[query, kind],
                kappa,
            )
            if kind == "endpoint_potential":
                parts.update(
                    skew=np.zeros(len(states)), finite_step=np.zeros(len(states))
                )
            values[query, kind] = np.column_stack([parts[p] for p in M.CHANNELS])
    (out / "01").mkdir()
    np.savez_compressed(
        out / "01/source_actions.npz",
        states=states,
        **{q + "__" + k: v for (q, k), v in values.items()},
    )
    spectra, channel_grams, diagnostics = [], [], []
    dimensions = {
        "A": support_rank**2,
        "S": support_rank * (support_rank + 1) // 2,
        "H": support_rank * (support_rank - 1) // 2,
    }
    for design in spec["source_designs"]:
        pos = np.array(
            [
                j
                for j, s in enumerate(states)
                if design == "all_states" or meta[s]["arm"] == design
            ]
        )
        families = [meta[states[j]]["family"] for j in pos]
        weights = {
            "uniform": np.full(len(pos), 1 / len(pos)),
            "family_balanced": M.family_weights(families),
        }
        for kind, ctx in contexts.items():
            grams = M.dyad_grams(x[states[pos]], ctx[states[pos]])
            saved = {"state_indices": states[pos]}
            for part, gram in grams.items():
                saved[part + "_raw"] = gram
                for measure, w in weights.items():
                    weighted = M.weighted_design_gram(gram, w)
                    eigen = M.spectrum(weighted, spec["machine_tolerance_multiplier"])
                    saved[part + "_" + measure + "_eigenvalues"] = eigen["eigenvalues"]
                    saved[measure + "_weights"] = w
                    spectra.append(
                        {
                            "design": design,
                            "context": kind,
                            "part": part,
                            "measure": measure,
                            "observations": len(pos),
                            "families": len(set(families)),
                            "support_rank": support_rank,
                            "operator_dimension": dimensions[part],
                            "observable_rank": eigen["rank"],
                            "numerical_null_dimension": dimensions[part]
                            - eigen["rank"],
                            "tolerance": eigen["tolerance"],
                            "smallest_positive": eigen["smallest_positive"],
                            "largest": eigen["largest"],
                            "eigen_min": float(eigen["eigenvalues"][0]),
                        }
                    )
            np.savez_compressed(
                out / "01" / f"{design}_{kind}_observation_grams.npz", **saved
            )
        w = weights["family_balanced"]
        for (query, kind), v in values.items():
            info = {
                "design": design,
                "query": query,
                "model": kind,
                "observations": len(pos),
                "families": len(set(families)),
            }
            mom = M.moments(v[pos], w)
            for i, j in itertools.product(range(4), repeat=2):
                channel_grams.append(
                    {
                        **info,
                        "part_a": M.CHANNELS[i],
                        "part_b": M.CHANNELS[j],
                        "inner_product": float(mom["gram"][i, j]),
                        "covariance": float(mom["covariance"][i, j]),
                    }
                )
            diagnostics.append(
                {
                    **info,
                    **M.action_diagnostics(
                        v[pos], w, spec["machine_tolerance_multiplier"]
                    ),
                }
            )
    write_csv(out / "01/observation_spectra.csv", spectra)
    write_csv(out / "01/source_channel_grams.csv", channel_grams)
    write_csv(out / "01/source_action_diagnostics.csv", diagnostics)
    write_json(
        out / "01/source_manifest.json",
        {
            "held_targets_loaded": False,
            "new_source_fits": 0,
            "support_rank": support_rank,
            "source_states": len(states),
            "source_designs": spec["source_designs"],
            "source_seconds": time.monotonic() - start,
            "artifacts": artifact_hashes(out / "01", out),
        },
    )
    print(
        {
            "status": "SOURCE_OBSERVATION_GEOMETRY_FROZEN",
            "spectra": len(spectra),
            "held_targets_loaded": False,
        },
        flush=True,
    )


def evaluate(out):
    """Compare all frozen channels and native actions with declared family measures."""
    manifest = check(out, geometry=True)
    if (out / "02").exists():
        raise FileExistsError("preserve observation diagnostics")
    start, spec = time.monotonic(), manifest["spec"]
    packet = ROOT / spec["operator_packet"]
    rows = csv_rows(packet / "02/components.csv")
    if len(rows) != spec["paired_query_rows"] * len(spec["models"]):
        raise ValueError("frozen component scope changed")
    addresses = read_json(out / "00/metadata.json")
    address = {r["row"]: r for r in addresses}
    for row in rows:
        original = address[int(row["row"])]
        for key in (
            "query",
            "role",
            "arm",
            "family",
            "state_index",
            "unit_path",
            "reference_path",
        ):
            if str(row[key]) != str(original[key]):
                raise ValueError("inherited component address mismatch")
    frob = {
        (r["model"], r["query_a"], r["query_b"], r["part"]): r
        for r in csv_rows(packet / "01/cross_target_geometry.csv")
    }
    table = {(r["query"], r["model"], int(r["state_index"])): r for r in rows}
    first = sorted(
        (r for r in addresses if r["query"] == spec["queries"][0]),
        key=lambda r: r["state_index"],
    )
    diagnostics, sharing, covariance_comparison = [], [], []
    source_error = 0.0
    with np.load(out / "01/source_actions.npz") as arrays:
        source_states = arrays["states"]
        for query, kind in itertools.product(spec["queries"], spec["models"]):
            previous = arrays[query + "__" + kind]
            actual = np.array(
                [
                    [float(table[query, kind, s][p]) for p in M.CHANNELS]
                    for s in source_states
                ]
            )
            source_error = max(source_error, float(np.max(abs(previous - actual))))
    if source_error > 1e-10:
        raise ValueError("source freeze disagrees with inherited actions")
    for role in ("source", "validation", "application"):
        for design in ("all_states", *sorted({r["arm"] for r in first})):
            members = [
                r
                for r in first
                if r["role"] == role and (design == "all_states" or r["arm"] == design)
            ]
            indices = [r["state_index"] for r in members]
            w = M.family_weights([r["family"] for r in members])
            info = {
                "role": role,
                "design": design,
                "observations": len(indices),
                "families": len({r["family"] for r in members}),
            }
            for kind in spec["models"]:
                by_query = {}
                for query in spec["queries"]:
                    current = [table[query, kind, s] for s in indices]
                    v = np.array([[float(r[p]) for p in M.CHANNELS] for r in current])
                    diagnostics.append(
                        {
                            **info,
                            "model": kind,
                            "query": query,
                            **M.action_diagnostics(
                                v, w, spec["machine_tolerance_multiplier"]
                            ),
                        }
                    )
                    by_query[query] = {
                        p: np.array([float(r[p]) for r in current])
                        for p in PARTS
                        if p != "coupling"
                    }
                    by_query[query]["coupling"] = v[:, 1] + v[:, 2]
                for qa, qb in itertools.combinations(spec["queries"], 2):
                    for part in PARTS:
                        a, b = by_query[qa][part], by_query[qb][part]
                        mom = M.moments(np.column_stack((a, b)), w)
                        g, cov = mom["gram"], mom["covariance"]
                        op_part = {
                            "linear": "b",
                            "symmetric": "S",
                            "skew": "H",
                            "coupling": "A",
                        }.get(part)
                        operator = frob.get((kind, qa, qb, op_part), {})
                        fc = (
                            float(operator["cosine"])
                            if operator.get("cosine")
                            else None
                        )
                        ac = M.cosine(g[0, 1], g[0, 0], g[1, 1])
                        sharing.append(
                            {
                                **info,
                                "model": kind,
                                "query_a": qa,
                                "query_b": qb,
                                "part": part,
                                "inner_product": float(g[0, 1]),
                                "covariance": float(cov[0, 1]),
                                "rms_a": float(np.sqrt(g[0, 0])),
                                "rms_b": float(np.sqrt(g[1, 1])),
                                "action_cosine": ac,
                                "action_correlation": M.cosine(
                                    cov[0, 1], cov[0, 0], cov[1, 1]
                                ),
                                "frobenius_cosine": fc,
                                "cosine_sign_reversal": fc is not None
                                and ac is not None
                                and fc * ac < 0,
                            }
                        )
                    native = M.moments(
                        np.column_stack(
                            (
                                by_query[qa]["observed_action"],
                                by_query[qb]["observed_action"],
                            )
                        ),
                        w,
                    )
                    compiled = M.moments(
                        np.column_stack(
                            (
                                by_query[qa]["predicted_action"],
                                by_query[qb]["predicted_action"],
                            )
                        ),
                        w,
                    )
                    covariance_comparison.append(
                        {
                            **info,
                            "model": kind,
                            "query_a": qa,
                            "query_b": qb,
                            "native_covariance": float(native["covariance"][0, 1]),
                            "compiled_covariance": float(compiled["covariance"][0, 1]),
                            "native_correlation": M.cosine(
                                native["covariance"][0, 1],
                                native["covariance"][0, 0],
                                native["covariance"][1, 1],
                            ),
                            "compiled_correlation": M.cosine(
                                compiled["covariance"][0, 1],
                                compiled["covariance"][0, 0],
                                compiled["covariance"][1, 1],
                            ),
                            "native_mean_a": float(native["mean"][0]),
                            "native_mean_b": float(native["mean"][1]),
                            "compiled_mean_a": float(compiled["mean"][0]),
                            "compiled_mean_b": float(compiled["mean"][1]),
                            "native_sd_a": float(np.sqrt(native["covariance"][0, 0])),
                            "native_sd_b": float(np.sqrt(native["covariance"][1, 1])),
                            "compiled_sd_a": float(
                                np.sqrt(compiled["covariance"][0, 0])
                            ),
                            "compiled_sd_b": float(
                                np.sqrt(compiled["covariance"][1, 1])
                            ),
                        }
                    )
    write_csv(out / "02/action_diagnostics.csv", diagnostics)
    write_csv(out / "02/target_sharing.csv", sharing)
    write_csv(out / "02/native_compiled_covariance.csv", covariance_comparison)
    write_json(
        out / "02/summary.json",
        {
            "status": "HM18_ACTION_OBSERVABILITY_EVALUATED_NOT_GLOBAL_COMPLETE",
            "paired_query_rows": len(addresses),
            "unique_state_pairs": len(first),
            "diagnostic_rows": len(diagnostics),
            "sharing_rows": len(sharing),
            "covariance_rows": len(covariance_comparison),
            "source_action_replay_max_error": source_error,
            "source_manifest_sha256": digest(out / "01/source_manifest.json"),
            "new_model_calls": 0,
            "new_source_fits": 0,
            "api_calls": 0,
            "private_text_reads": 0,
            "native_cycles_observed": False,
            "accuracy_selected_rank": False,
            "original_full_scope_complete": False,
            "evaluate_seconds": time.monotonic() - start,
        },
    )
    write_json(out / "02/artifact_hashes.json", artifact_hashes(out, out))
    print(read_json(out / "02/summary.json"), flush=True)


def main():
    """Run one finite stage, preserving previous frozen outputs."""
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
