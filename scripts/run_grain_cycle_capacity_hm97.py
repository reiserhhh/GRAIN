#!/usr/bin/env python3
"""Measure multiple independent cycles; no score-fitting or accuracy grid."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_cycle_capacity as M
from grain_research.vector_mechanisms.navigation import MODEL
from scripts.run_grain_author_current_hm93 import observe, sentence
from scripts.run_grain_body_relation_graph_hm89 import collect, material_tokens
from scripts.run_grain_hierarchical_query_hm88 import unit
from scripts.run_grain_observer_parity_hm96 import (
    FC,
    load,
    seal,
    sha,
    table,
    verify,
    write,
)
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

SPEC = ROOT / "configs/grain_cycle_capacity_hm97.json"
CODE = (
    "configs/grain_cycle_capacity_hm97.json",
    "docs/GRAIN_MULTI_CYCLE_CAPACITY_PLAN_HM97.md",
    "grain_research/construction/math_cycle_capacity.py",
    "scripts/run_grain_cycle_capacity_hm97.py",
    "scripts/verify_grain_cycle_capacity_hm97.py",
    "tests/test_grain_math_cycle_capacity.py",
    "scripts/run_grain_author_current_hm93.py",
    "scripts/run_grain_body_relation_graph_hm89.py",
    "scripts/run_grain_hierarchical_query_hm88.py",
    "scripts/run_grain_query_evidence_hm86.py",
    "scripts/run_grain_observer_parity_hm96.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_math_construct_routing.py",
    "scripts/run_grain_upstream_routes.py",
    "grain_research/construction/math_channel_boundary_transport.py",
    "grain_research/construction/math_cold_evidence.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/vector_mechanisms/prefix.py",
    "suica_core/suica.py",
    "suica_core/v7_observations.py",
    "suica_core/v7_psychometric.py",
    "suica_core/v7_operator_controls.py",
)


def settings():
    """Return the sole fixed protocol and immutable output destination."""
    spec = load(SPEC)
    return spec, ROOT / spec["output"]


def check(out):
    """Require unchanged code, declared numeric ancestry and public inputs."""
    manifest = load(out / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Freeze all18 states and every ordered pair before observing outcomes."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve previous or partial packets")
    spec, _ = settings()
    source = ROOT / spec["source"]
    old = load(source / "00/manifest.json")
    verify(ROOT, old["code"] | old["source"])
    verify(source, old["inputs"])
    verify(
        source,
        {
            n: h
            for n, h in load(source / "02/output_seal.json").items()
            if not Path(n).name.startswith("._")
        },
    )
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM93_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM93 required")
    bundle = load(source / "00/bundle.json")
    material = load(ROOT / spec["material"])
    if bundle["targets"] != spec["targets"] or material["targets"] != spec["targets"]:
        raise ValueError("All six frozen targets required")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = load(ROOT / spec["source_material"])["system"]
    states = [
        {"state": 3 * family + level, "family": target, "level": level}
        for family, target in enumerate(spec["targets"])
        for level in range(3)
    ]
    contexts, rows = [], []
    for lex, frame in itertools.product(range(2), spec["frames"]):
        context = len(contexts)
        words = [material["behaviours"][s["family"]][lex][s["level"]] for s in states]
        contexts.append(
            {"context": context, "lexicon": lex, "frame": frame, "actions": words}
        )
        for i, j in itertools.product(range(spec["states"]), repeat=2):
            body = "\n".join(
                [
                    sentence("K1", words[i], frame),
                    sentence("K2", words[j], frame),
                    "On occasion K3, Person A checked the date on a wall calendar.",
                ]
            )
            tokens = material_tokens(
                body, spec["targets"][0], system, tokenizer, [words[i], words[j]]
            )
            if not all(tokens["action_groups"]):
                raise ValueError("Empty actual action token span")
            if not rows:
                other = material_tokens(
                    body, spec["targets"][-1], system, tokenizer, [words[i], words[j]]
                )
                if other["input_ids"] != tokens["input_ids"]:
                    raise ValueError("Analysis target contaminated body-only capture")
            rows.append(
                {
                    "unit": len(rows),
                    "context": context,
                    "previous": i,
                    "current": j,
                    **tokens,
                }
            )
    if len(rows) != spec["max_forwards"]:
        raise ValueError("Incomplete ordered-pair support")
    (out / "00").mkdir(parents=True)
    write(out / "00/states.json", states)
    write(out / "00/contexts.json", contexts)
    write(out / "00/rows.json", rows)
    write(out / "00/bundle.json", bundle)
    files = [
        source / n
        for n in (
            "00/manifest.json",
            "00/bundle.json",
            "02/kernels.npz",
            "02/output_seal.json",
            "verification.json",
        )
    ]
    files.extend(
        ROOT / spec[n] for n in ("material", "source_material", "bundle", "runtime")
    )
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "inputs": seal(out, out / "00"),
            "model": old["model"],
            "no_application_query_answer_fit": True,
            "public_designed_only": True,
            "no_new_predictor": True,
            "scope_complete": False,
            "cycle_dimension": (spec["states"] - 1) * (spec["states"] - 2) // 2,
        },
    )
    print("PREPARED HM97", len(rows), "public bodies", flush=True)


def score(out):
    """Run official event scoring without refitting its reference population."""
    manifest = check(out)
    spec = manifest["spec"]
    contexts = load(out / "00/contexts.json")
    dest = out / "02"
    dest.mkdir()
    records, meta = [], []
    for c in contexts:
        texts = [
            (s, position, sentence(f"K{position + 1}", action, c["frame"]))
            for s, action in enumerate(c["actions"])
            for position in range(2)
        ]
        texts.append(
            (-1, 2, "On occasion K3, Person A checked the date on a wall calendar.")
        )
        for s, position, text in texts:
            uid = f"node{len(records):04d}"
            records.append({"author": uid, "body": text})
            meta.append(
                {
                    "user_id": uid,
                    "context": c["context"],
                    "state": s,
                    "position": position,
                }
            )
    frame = canonicalize_comments(
        pd.DataFrame(records), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    runtime = joblib.load(ROOT / spec["runtime"])
    bundle = read_factor_bundle(ROOT / spec["bundle"])
    units, _, scores = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=runtime,
        bundle=bundle,
    )
    if (
        len(units) != len(records)
        or not scores.score_status.eq("SCORED").all()
        or not scores.n_units.eq(1).all()
    ):
        raise ValueError("Every declared state must produce one official observation")
    ids = [r["user_id"] for r in meta]
    units = units.set_index("user_id").loc[ids].reset_index()
    scores = scores.set_index("user_id").loc[ids].reset_index()
    svd = runtime.svd.transform(runtime.vectorizer.transform(units["text"]))
    factors = scores[FC].to_numpy(float)
    np.savez_compressed(dest / "official_coordinates.npz", factors=factors, svd=svd)
    write(
        dest / "nodes.json",
        [{**m, "text": str(t)} for m, t in zip(meta, units["text"], strict=True)],
    )
    scores.to_csv(dest / "official_scores.csv", index=False)
    write(
        dest / "summary.json",
        {
            "actual_official_units": len(records),
            "bundle_id": bundle.bundle_id,
            "fit_calls": 0,
            "neural_calls": 0,
        },
    )
    write(out / "02_seal.json", seal(out, dest))
    print("SCORED HM97", len(records), "official nodes", flush=True)


def analyze(out):
    """Measure missing target action and positive-law witnesses, not accuracy."""
    manifest = check(out)
    spec = manifest["spec"]
    verify(out, load(out / "02_seal.json"))
    verify(
        out,
        {
            n: h
            for n, h in load(out / "01/raw_seal.json").items()
            if not Path(n).name.startswith("._")
        },
    )
    if load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("All real captures must be complete")
    start = time.monotonic()
    dest = out / "03"
    dest.mkdir()
    rows, bundle = load(out / "00/rows.json"), load(out / "00/bundle.json")
    n = spec["states"]
    kernels = np.zeros((4, n, n, 18))
    for row in rows:
        fields, _ = observe(unit(out, row["unit"]), row, bundle)
        kernels[row["context"], row["previous"], row["current"]] = fields.ravel()
    nodes = load(out / "02/nodes.json")
    coordinates = np.load(out / "02/official_coordinates.npz")
    lookup = {(r["context"], r["state"], r["position"]): i for i, r in enumerate(nodes)}
    generic = np.random.default_rng(spec["generic_seed"]).normal(size=(n, 4))
    all_arrays = {"target_kernel": kernels, "generic_points": generic}
    capacity, accounts, geometries, invariant_errors = [], [], [], []
    plus = np.zeros((4, 3, 18, n, n))
    minus = np.zeros_like(plus)
    old = np.load(ROOT / spec["source"] / "02/kernels.npz")["kernel"]
    replay_error = 0.0
    for context in range(4):
        ix0 = [lookup[context, s, 0] for s in range(n)]
        ix1 = [lookup[context, s, 1] for s in range(n)]
        iz = lookup[context, -1, 2]
        lex, frame = context // 2, context % 2
        for family in range(6):
            block = kernels[
                context, 3 * family : 3 * family + 3, 3 * family : 3 * family + 3
            ]
            replay_error = max(
                replay_error,
                float(
                    np.max(np.abs(block.reshape(3, 3, 3, 6) - old[lex, frame, family]))
                ),
            )
        fc = M.cycles(kernels[context])[2]
        fedges = M.edge_columns(fc)
        f_basis, f_singular, f_cutoff = M.orthogonal_basis(
            fedges, spec["rank_rtol"], spec["rank_atol"]
        )
        all_arrays[f"target_cycle_{context}"] = fc
        all_arrays[f"target_singular_{context}"] = f_singular
        norm = np.linalg.norm(fedges, axis=0)
        normalized = np.divide(
            fedges,
            norm,
            out=np.zeros_like(fedges),
            where=norm > spec["arithmetic_atol"],
        )
        gram = normalized.T @ normalized
        for a, b in itertools.combinations(range(18), 2):
            geometries.append(
                {
                    "context": context,
                    "channel_a": a,
                    "channel_b": b,
                    "cycle_cosine": float(gram[a, b]),
                }
            )
        for bi, bank in enumerate(spec["banks"]):
            if bank == "generic4":
                x, z, position_error = generic, np.zeros(4), 0.0
            else:
                key = "factors" if bank == "grain_factor4" else "svd"
                x, z = coordinates[key][ix0], coordinates[key][iz]
                position_error = float(np.max(np.abs(x - coordinates[key][ix1])))
                if position_error > spec["arithmetic_atol"]:
                    raise ValueError(
                        "Common node coordinates failed; preserve partial output"
                    )
            a = M.area_bank(x, z)
            result = M.decompose(
                a, kernels[context], spec["rank_rtol"], spec["rank_atol"]
            )
            aff_basis, _, _ = M.orthogonal_basis(
                x - x.mean(0), spec["rank_rtol"], spec["rank_atol"]
            )
            principal = np.linalg.svd(result["basis"].T @ f_basis, compute_uv=False)
            anchor_error = float(
                np.max(
                    np.abs(
                        result["area_cycle"]
                        - M.cycles(M.area_bank(x, np.zeros_like(z)))[2]
                    )
                )
            )
            all_arrays[f"points_{context}_{bi}"] = x
            all_arrays[f"anchor_{context}_{bi}"] = z
            for name in (
                "area_cycle",
                "shared",
                "residual",
                "singular",
                "coefficients",
            ):
                all_arrays[f"{name}_{context}_{bi}"] = result[name]
            all_arrays[f"principal_cosines_{context}_{bi}"] = principal
            capacity.append(
                {
                    "context": context,
                    "bank": bank,
                    "node_dimension": x.shape[1],
                    "affine_rank": aff_basis.shape[1],
                    "area_channels": a.shape[-1],
                    "area_cycle_rank": result["basis"].shape[1],
                    "cycle_dimension": manifest["cycle_dimension"],
                    "exterior_rank_expected": aff_basis.shape[1]
                    * (aff_basis.shape[1] - 1)
                    // 2,
                    "target_cycle_rank": f_basis.shape[1],
                    "rank_cutoff": result["cutoff"],
                    "target_rank_cutoff": f_cutoff,
                    "position_error": position_error,
                    "anchor_cycle_error": anchor_error,
                }
            )
            for q in range(18):
                r, observed = result["residual"][..., q], result["shared"][..., q]
                plus[context, bi, q], minus[context, bi, q], scale = M.legal_witness(
                    r, spec["arithmetic_atol"]
                )
                delta = plus[context, bi, q] - minus[context, bi, q]
                total_energy = float(np.sum(fc[..., q] ** 2))
                residual_energy = float(np.sum(r**2))
                action = float(np.sum(delta * kernels[context, ..., q]))
                area_action = np.einsum("ij,ijb->b", delta, a)
                static_error = float(np.max(np.abs(delta.sum(1) @ x)))
                witness_errors = [
                    float(np.max(np.abs(delta.sum(1)))),
                    float(np.max(np.abs(area_action))),
                    static_error,
                    abs(action - 2 * scale * residual_energy),
                    anchor_error,
                    float(np.max(np.abs(result["area_cycle"].sum(1)))),
                ]
                invariant_errors.extend(witness_errors)
                accounts.append(
                    {
                        "context": context,
                        "bank": bank,
                        "field": spec["fields"][q // 6],
                        "target": spec["targets"][q % 6],
                        "cycle_norm": np.sqrt(total_energy),
                        "observed_energy_fraction": float(
                            np.sum(observed**2) / total_energy
                        )
                        if total_energy
                        else None,
                        "residual_energy_fraction": residual_energy / total_energy
                        if total_energy
                        else None,
                        "residual_norm": np.sqrt(residual_energy),
                        "witness_exists": bool(scale),
                        "target_contrast": action,
                        "area_contrast_max": float(np.max(np.abs(area_action))),
                        "static_contrast_max": static_error,
                        "scale": scale,
                        "min_pair_mass": float(
                            min(plus[context, bi, q].min(), minus[context, bi, q].min())
                        ),
                        "identity_error": abs(action - 2 * scale * residual_energy),
                    }
                )
    all_arrays["witness_plus"], all_arrays["witness_minus"] = plus, minus
    np.savez_compressed(dest / "measured_capacity.npz", **all_arrays)
    table(dest / "capacity.csv", capacity)
    table(dest / "target_accounts.csv", accounts)
    table(dest / "target_cycle_geometry.csv", geometries)
    summary = {
        "status": "MEASURED_AWAITING_ALTERNATE_VERIFICATION",
        "actual_forwards": len(rows),
        "model_seconds": load(out / "01/status.json")["model_seconds"],
        "analysis_seconds": time.monotonic() - start,
        "official_units": len(nodes),
        "target_account_rows": len(accounts),
        "capacity_rows": capacity,
        "invariant_error_max": max(invariant_errors),
        "HM93_overlap_replay_error": replay_error,
        "no_new_predictor_fit": True,
        "witness_is_post_observation_existence_not_validation": True,
        "original_scope_complete": False,
    }
    write(dest / "summary.json", summary)
    write(out / "03_seal.json", seal(out, dest))
    print(
        "ANALYZED HM97",
        len(accounts),
        "target accounts; max identity error",
        max(invariant_errors),
        flush=True,
    )


def main():
    """Run one explicit stage; never auto-expand a failed finite packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "score", "analyze"))
    args = parser.parse_args()
    _, out = settings()
    with threadpool_limits(limits=1):
        {"prepare": prepare, "collect": collect, "score": score, "analyze": analyze}[
            args.stage
        ](out)


if __name__ == "__main__":
    main()
