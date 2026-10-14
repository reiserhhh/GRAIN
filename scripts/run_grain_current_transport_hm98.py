#!/usr/bin/env python3
"""Transport source target cochains before observing fresh body kernels."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_current_transport as M
from grain_research.construction.math_cycle_capacity import cycles
from grain_research.vector_mechanisms.navigation import MODEL
from scripts.run_grain_author_current_hm93 import observe, sentence
from scripts.run_grain_body_relation_graph_hm89 import collect, material_tokens
from scripts.run_grain_cycle_capacity_hm97 import CODE as SOURCE_CODE
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

SPEC = ROOT / "configs/grain_current_transport_hm98.json"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_current_transport_hm98.json",
            "docs/GRAIN_CURRENT_TRANSPORT_PLAN_HM98.md",
            "grain_research/construction/math_current_transport.py",
            "scripts/run_grain_current_transport_hm98.py",
            "scripts/score_grain_current_transport_cold.py",
            "scripts/verify_grain_current_transport_hm98.py",
            "tests/test_grain_math_current_transport.py",
            *SOURCE_CODE,
        )
    )
)


def settings():
    """Return one fixed protocol, with no automatic search expansion."""
    spec = load(SPEC)
    return spec, ROOT / spec["output"]


def check(out):
    """Check frozen code, source numerical assets and public application inputs."""
    manifest = load(out / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    verify(out, manifest["inputs"])
    return manifest


def stage_seal(out, number):
    """Verify only declared stage seals, never AppleDouble sidecars."""
    hashes = load(out / f"{number:02d}_seal.json")
    verify(out, {n: h for n, h in hashes.items() if not Path(n).name.startswith("._")})


def prepare(out):
    """Freeze new action strings and every pair before reading their kernels."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve old and partial packets")
    spec, _ = settings()
    source = ROOT / spec["source"]
    old = load(source / "00/manifest.json")
    verify(ROOT, old["code"] | old["source"])
    verify(source, old["inputs"])
    for number in (2, 3, 4, 5, 6):
        stage_seal(source, number)
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM97_NOT_GLOBAL"
    ):
        raise ValueError("Actual verified HM97 required")
    bundle = load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"]:
        raise ValueError("All six source targets required")
    old_actions = set(
        itertools.chain.from_iterable(
            c["actions"] for c in load(source / "00/contexts.json")
        )
    )
    actions = [a for target in spec["targets"] for a in spec["behaviours"][target]]
    if len(actions) != 18 or old_actions.intersection(actions):
        raise ValueError("Application actions must be new and complete")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    system = load(ROOT / spec["source_material"])["system"]
    rows = []
    for context, frame in enumerate(spec["frames"]):
        for i, j in itertools.product(range(18), repeat=2):
            body = "\n".join(
                [
                    sentence("K1", actions[i], frame),
                    sentence("K2", actions[j], frame),
                    "On occasion K3, Person A checked the date on a wall calendar.",
                ]
            )
            tokens = material_tokens(
                body, spec["targets"][0], system, tokenizer, [actions[i], actions[j]]
            )
            if not all(tokens["action_groups"]):
                raise ValueError("Missing actual action-token span")
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
        raise ValueError("Incomplete fixed pair support")
    (out / "00").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/bundle.json", bundle)
    write(out / "00/actions.json", actions)
    files = [
        source / n
        for n in (
            "00/manifest.json",
            "00/bundle.json",
            "03/measured_capacity.npz",
            "03_seal.json",
            "verification.json",
        )
    ]
    files += [ROOT / spec[n] for n in ("bundle", "runtime", "source_material")]
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "inputs": seal(out, out / "00"),
            "model": old["model"],
            "new_public_bodies": len(rows),
            "all_targets_retained": True,
            "cold_before_new_reader": True,
            "original_scope_complete": False,
        },
    )
    print("PREPARED HM98", len(rows), "new public bodies", flush=True)


def score(out):
    """Run the inherited official scorer on new event units, without fitting."""
    manifest = check(out)
    spec = manifest["spec"]
    actions = load(out / "00/actions.json")
    dest = out / "02"
    dest.mkdir()
    records, nodes = [], []
    for context, frame in enumerate(spec["frames"]):
        for state, action in enumerate(actions):
            for position in range(2):
                uid = f"hm98node{len(nodes):03d}"
                text = sentence(f"K{position + 1}", action, frame)
                records.append({"author": uid, "body": text})
                nodes.append(
                    {
                        "user_id": uid,
                        "context": context,
                        "state": state,
                        "position": position,
                    }
                )
        uid = f"hm98node{len(nodes):03d}"
        records.append(
            {
                "author": uid,
                "body": "On occasion K3, Person A checked the date on a wall calendar.",
            }
        )
        nodes.append({"user_id": uid, "context": context, "state": -1, "position": 2})
    frame = canonicalize_comments(
        pd.DataFrame(records), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    runtime, bundle = (
        joblib.load(ROOT / spec["runtime"]),
        read_factor_bundle(ROOT / spec["bundle"]),
    )
    units, _, scores = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=runtime,
        bundle=bundle,
    )
    if (
        len(units) != len(nodes)
        or not scores.score_status.eq("SCORED").all()
        or not scores.n_units.eq(1).all()
    ):
        raise ValueError("Every new node must score exactly once")
    ids = [n["user_id"] for n in nodes]
    units = units.set_index("user_id").loc[ids].reset_index()
    scores = scores.set_index("user_id").loc[ids].reset_index()
    svd = runtime.svd.transform(runtime.vectorizer.transform(units["text"]))
    factors = scores[FC].to_numpy(float)
    coordinates = {}
    lookup = {(n["context"], n["state"], n["position"]): i for i, n in enumerate(nodes)}
    position_error = 0.0
    for context, bank in itertools.product(range(2), spec["banks"]):
        array = factors if bank == "factor4" else svd
        ix = [lookup[context, s, 0] for s in range(18)]
        other = [lookup[context, s, 1] for s in range(18)]
        coordinates[f"points_{context}_{bank}"] = array[ix]
        position_error = max(
            position_error, float(np.max(np.abs(array[ix] - array[other])))
        )
    if position_error > spec["arithmetic_atol"]:
        raise ValueError("Position changes lexical event coordinates")
    np.savez_compressed(dest / "application_coordinates.npz", **coordinates)
    scores.to_csv(dest / "official_scores.csv", index=False)
    write(
        dest / "nodes.json",
        [{**n, "text": str(t)} for n, t in zip(nodes, units["text"], strict=True)],
    )
    write(
        dest / "summary.json",
        {
            "official_units": len(nodes),
            "position_error": position_error,
            "fit_calls": 0,
            "bundle_id": bundle.bundle_id,
        },
    )
    write(out / "02_seal.json", seal(out, dest))
    print("SCORED HM98", len(nodes), "official units", flush=True)


def compile_source(out):
    """Compile measured source relations before application reader observations."""
    manifest = check(out)
    stage_seal(out, 2)
    if (out / "01").exists():
        raise ValueError("Compile before collecting any new target kernels")
    source = ROOT / manifest["spec"]["source"]
    measured = np.load(source / "03/measured_capacity.npz", allow_pickle=False)
    dest = out / "03"
    dest.mkdir()
    arrays = {"kernel": measured["target_kernel"][:2]}
    for context, bank in itertools.product(range(2), ("factor4", "svd24")):
        arrays[f"points_{context}_{bank}"] = measured[
            f"points_{context}_{int(bank == 'svd24')}"
        ]
    np.savez_compressed(dest / "compiled_bundle.npz", **arrays)
    write(
        dest / "summary.json",
        {
            "source_contexts": [0, 1],
            "source_states": 18,
            "fit_calls": 0,
            "source_only": True,
            "compiled_before_new_capture": True,
        },
    )
    write(out / "03_seal.json", seal(out, dest))
    print("COMPILED HM98 source-only relation kernel", flush=True)


def predict(out):
    """Run a separate, audited process without application target access."""
    check(out)
    stage_seal(out, 2)
    stage_seal(out, 3)
    if (out / "01").exists():
        raise ValueError("Cold scoring must precede new target capture")
    dest = out / "04"
    dest.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_grain_current_transport_cold.py"),
            str(out / "03/compiled_bundle.npz"),
            str(out / "02/application_coordinates.npz"),
            str(dest / "predictions.npz"),
            str(dest / "audit.json"),
        ],
        check=True,
        cwd=ROOT,
    )
    write(out / "04_seal.json", seal(out, dest))


def collect_after_cold(out):
    """Read new bodies only after the cold numerical result is immutable."""
    check(out)
    stage_seal(out, 4)
    collect(out)


def evaluate(out):
    """Compare current observations, not a teacher-fit leaderboard."""
    manifest = check(out)
    spec = manifest["spec"]
    for number in (2, 3, 4):
        stage_seal(out, number)
    verify(
        out,
        {
            n: h
            for n, h in load(out / "01/raw_seal.json").items()
            if not Path(n).name.startswith("._")
        },
    )
    if load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("All actual captures required")
    begin = time.monotonic()
    dest = out / "05"
    dest.mkdir()
    native = np.zeros((2, 18, 18, 18))
    bundle = load(out / "00/bundle.json")
    rows = load(out / "00/rows.json")
    for row in rows:
        field, _ = observe(unit(out, row["unit"]), row, bundle)
        native[row["context"], row["previous"], row["current"]] = field.ravel()
    source = np.load(out / "03/compiled_bundle.npz", allow_pickle=False)
    predictions = np.load(out / "04/predictions.npz", allow_pickle=False)
    d, plus, minus, triples = M.fundamental_currents(18)
    arrays = {"native_kernel": native, "current": d, "plus": plus, "minus": minus}
    accounts, contrasts, map_records, errors = [], [], [], []
    for context in range(2):
        fs = source["kernel"][context]
        nc = cycles(native[context])[2]
        src_cycle = cycles(fs)[2]
        native_contrasts = np.einsum("aij,ijq->aq", plus - minus, native[context])
        arrays[f"native_cycle_{context}"] = nc
        arrays[f"native_contrasts_{context}"] = native_contrasts
        for arm in (*spec["banks"], "uniform_null", "known_correspondence"):
            t = (
                predictions[f"map_{context}_{arm}"]
                if arm in spec["banks"]
                else np.full((18, 18), 1 / 18)
                if arm == "uniform_null"
                else np.eye(18)
            )
            transported = (
                predictions[f"kernel_{context}_{arm}"]
                if arm in spec["banks"]
                else M.pullback(fs, t)
            )
            predicted_cycle = cycles(transported)[2]
            pc = np.einsum("aij,ijq->aq", plus - minus, transported)
            _, singular, exterior = M.centered_map_spectrum(t)
            arrays[f"map_{context}_{arm}"] = t
            arrays[f"kernel_{context}_{arm}"] = transported
            arrays[f"singular_{context}_{arm}"] = singular
            arrays[f"exterior_{context}_{arm}"] = exterior
            arrays[f"contrasts_{context}_{arm}"] = pc
            cutoff = max(spec["rank_atol"], spec["rank_rtol"] * singular[0])
            map_records.append(
                {
                    "context": context,
                    "arm": arm,
                    "centered_node_rank": int(np.count_nonzero(singular > cutoff)),
                    "largest_node_singular": float(singular[0]),
                    "smallest_nonzero_node_singular": float(
                        singular[singular > cutoff].min()
                    )
                    if np.any(singular > cutoff)
                    else 0,
                    "largest_exterior_singular": float(exterior.max()),
                    "mean_corresponding_state_mass": float(np.trace(t) / 18),
                    "design_address_top1_count": int(
                        np.count_nonzero(np.argmax(t, axis=1) == np.arange(18))
                    ),
                    "mean_entropy": float(
                        -np.sum(t * np.log(np.maximum(t, 1e-300))) / 18
                    ),
                    "is_oracle_diagnostic": arm == "known_correspondence",
                }
            )
            for current, p, m in zip(d, plus, minus, strict=True):
                dp = M.pushforward(current, t)
                pp, mp = M.pushforward(p, t), M.pushforward(m, t)
                errors.extend(
                    [
                        float(np.max(np.abs(dp.sum(1)))),
                        float(np.max(np.abs(dp + dp.T))),
                        float(np.max(np.abs(pp.sum(1) - pp.sum(0)))),
                        abs(float(pp.sum()) - 1),
                        float(np.max(np.abs(pp.sum(1) - mp.sum(1)))),
                        float(np.max(np.abs(pp + pp.T - mp - mp.T))),
                        float(
                            np.max(
                                np.abs(
                                    np.einsum("ij,ijq->q", p - m, transported)
                                    - np.einsum(
                                        "ij,ijq->q", M.pushforward(p - m, t), fs
                                    )
                                )
                            )
                        ),
                    ]
                )
                if min(pp.min(), mp.min()) < 0:
                    raise ValueError("Transport produced a negative pair law")
            gauge = np.arange(18.0)[:, None] * np.arange(1.0, 19.0)[None, :]
            gauge_kernel = gauge[:, None] - gauge[None, :]
            errors.append(float(np.max(np.abs(cycles(M.pullback(gauge_kernel, t))[2]))))
            for q in range(18):
                actual, pred = nc[..., q], predicted_cycle[..., q]
                norm = np.linalg.norm(actual)
                pred_norm = np.linalg.norm(pred)
                fnorm = np.linalg.norm(src_cycle[..., q])
                active = np.abs(native_contrasts[:, q]) > spec["arithmetic_atol"]
                pred_active = np.abs(pc[:, q]) > spec["arithmetic_atol"]
                accounts.append(
                    {
                        "context": context,
                        "arm": arm,
                        "field": spec["fields"][q // 6],
                        "target": spec["targets"][q % 6],
                        "native_cycle_norm": float(norm),
                        "transport_cycle_norm": float(pred_norm),
                        "transport_over_source_norm": float(pred_norm / fnorm)
                        if fnorm
                        else None,
                        "cycle_cosine": float(
                            np.sum(actual * pred) / (norm * pred_norm)
                        )
                        if min(norm, pred_norm) > spec["arithmetic_atol"]
                        else None,
                        "relative_observer_defect": float(
                            np.linalg.norm(actual - pred) / norm
                        )
                        if norm
                        else None,
                        "native_nonzero_author_contrasts": int(active.sum()),
                        "transport_nonzero_author_contrasts": int(pred_active.sum()),
                        "same_sign_author_contrasts": int(
                            np.count_nonzero(
                                active
                                & pred_active
                                & (native_contrasts[:, q] * pc[:, q] > 0)
                            )
                        ),
                    }
                )
                for a, triple in enumerate(triples):
                    contrasts.append(
                        {
                            "context": context,
                            "arm": arm,
                            "field": spec["fields"][q // 6],
                            "target": spec["targets"][q % 6],
                            "cycle": a,
                            "nodes": "-".join(map(str, triple)),
                            "native_contrast": float(native_contrasts[a, q]),
                            "transport_contrast": float(pc[a, q]),
                        }
                    )
    if max(errors) > spec["arithmetic_atol"]:
        raise ValueError("Structural transport mismatch; retain failed outputs")
    np.savez_compressed(dest / "measured_transport.npz", **arrays)
    table(dest / "target_accounts.csv", accounts)
    table(dest / "author_contrasts.csv", contrasts)
    table(dest / "map_diagnostics.csv", map_records)
    summary = {
        "status": "MEASURED_AWAITING_ALTERNATE_VERIFICATION",
        "actual_forwards": len(rows),
        "model_seconds": load(out / "01/status.json")["model_seconds"],
        "official_units": load(out / "02/summary.json")["official_units"],
        "cold_audit": load(out / "04/audit.json"),
        "analysis_seconds": time.monotonic() - begin,
        "structural_error_max": max(errors),
        "positive_author_law_pairs": len(triples),
        "target_accounts": len(accounts),
        "author_contrast_rows": len(contrasts),
        "no_application_fitting": True,
        "original_scope_complete": False,
        "psychological_validity_claim": False,
    }
    write(dest / "summary.json", summary)
    write(out / "05_seal.json", seal(out, dest))
    print(
        "EVALUATED HM98",
        len(accounts),
        "accounts;structure error",
        max(errors),
        flush=True,
    )


def main():
    """Run one explicit stage without automatically restarting a failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "score", "compile", "predict", "collect", "evaluate"),
    )
    args = parser.parse_args()
    _, out = settings()
    with threadpool_limits(limits=1):
        {
            "prepare": prepare,
            "score": score,
            "compile": compile_source,
            "predict": predict,
            "collect": collect_after_cold,
            "evaluate": evaluate,
        }[args.stage](out)


if __name__ == "__main__":
    main()
