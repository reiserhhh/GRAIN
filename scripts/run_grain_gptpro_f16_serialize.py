#!/usr/bin/env python3
"""Build public calibrated F16 bundles, then replay them in sealed cold processes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as dyn_text
from grain_research.gptpro_foundation_v3 import f16_dynamic_transport as dyn_law
from grain_research.gptpro_foundation_v3 import f16_serialize as pub
from grain_research.gptpro_foundation_v3 import f16_t1_source as t1_source
from grain_research.gptpro_foundation_v3 import f16_t1_text as t1_text
from grain_research.gptpro_foundation_v3 import f16_t1_transport as t1_law
from grain_research.gptpro_foundation_v3 import f16_text_transport as text_law
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.run_grain_entailment_reference import MODEL, REVISION
from scripts.run_grain_gptpro_f08_nli_reader import QUERIES
from scripts.run_grain_gptpro_f12_transport import transform
from scripts.run_grain_gptpro_f16_dynamic_transport import source_records
from scripts.run_grain_gptpro_f16_t1_transport import fit_all, serial
from scripts.run_grain_gptpro_f16_t2 import group

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_gptpro_foundation_v3"
F16 = BASE / "F16"
SEEDS = (20260915, 20260916, 20260917)
WORLDS = ("identity", "global", "conditional", "nonlinear")
MODES = {w: ("global", "conditional") for w in WORLDS}
MODES["nonlinear"] = ("global", "conditional", "cubic")
FORWARD = {
    "identity": "y'=y",
    "global": "y'=1+1.5y",
    "conditional": "y'=z+(1+0.25z)y",
    "nonlinear": "y'=y+0.1y^3",
}
T1_RUN = F16 / "t1_transport_full_r1"
DYN_RUN = F16 / "dynamic_transport_full_r1"
TEXT_RUN = F16 / "text_transport_full_r1"
CODE = [
    "grain_research/gptpro_foundation_v3/f16_serialize.py",
    "scripts/run_grain_gptpro_f16_serialize.py",
    "scripts/audit_grain_gptpro_f16_serialize.py",
    "tests/test_grain_gptpro_f16_serialize.py",
    "docs/GRAIN_F16_SERIALIZE_PROTOCOL_V1.md",
]
REASONS = {
    "channel_counts": "old-bank source-validation counts inherited from t1_text_full_r1;the transported channel draws come from pilot-fit counts in calibration.counts",
    "calibration": "legacy NOT_EVALUATED label;superseded by the public calibration and known_validation blocks",
    "reader_hash": "hash over the OLD 96-text materials and OLD channel draws;does not identify the changed reader",
    "reader_bundle_sha256": "file hash of the OLD reader bundle t1_text_full_r1;superseded",
    "reference_mu": "old reader/old bank source reference;not the same target function under the changed reader",
    "reference_sigma": "old reader/old bank source reference;not the same target function under the changed reader",
    "reference_authors": "count attached to the historical reference",
    "reference": "old reader/old bank source references for T2/T3;not the same target function",
    "source_reference_authors": "count attached to the historical reference",
}
SCORE_FIELDS = {
    "text_t1": (
        "version",
        "scorer_code_hash",
        "world_contract",
        "seed",
        "qmc_power",
        "source_components",
        "source_models",
        "materials",
        "qualification_rate",
    ),
    "text_dynamic": (
        "version",
        "scorer_code_hash",
        "world_contract",
        "future_conditions",
        "target_ids",
        "object_classes",
        "seed",
        "qmc_power",
        "source_components",
        "source_models",
        "materials",
        "qualification_rate",
    ),
}


def rel(path):
    """Repository-relative provenance paths."""
    return str(Path(path).resolve().relative_to(ROOT))


def save(path, value):
    """Immutable strict JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(serial(value), sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    """Immutable JSON lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for row in rows:
            stream.write(
                json.dumps(serial(row), sort_keys=True, allow_nan=False) + "\n"
            )


def load(path):
    """Read JSON."""
    return json.loads(Path(path).read_text())


def manifest_hash(run, relative):
    """Authenticate a reused artifact against its original run manifest."""
    manifest = load(run / "output_manifest.json")
    digest = sha256_file(run / relative)
    if manifest.get(relative) != digest:
        raise ValueError(f"reused artifact not authenticated by manifest: {relative}")
    return digest


def wilson(coverage, n):
    """Wilson95 interval recomputed from archived coverage (T1 did not store it)."""
    v = 1.96**2
    center = (coverage + v / (2 * n)) / (1 + v / n)
    half = 1.96 * np.sqrt(coverage * (1 - coverage) / n + v / (4 * n * n)) / (1 + v / n)
    return center - half, center + half


def records(frame):
    """JSON-safe rows with NaN as null."""
    return json.loads(frame.to_json(orient="records"))


def prepare(output):
    """Freeze code and every reused input hash before any new scoring."""
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for seed in SEEDS:
        for p in [
            F16 / f"t1_source_full_r1/{seed}/bundle.json",
            F16 / f"t2_cold_full_r1/{seed}/bundle.json",
            BASE / f"F12/transport_full_r1/observations_{seed}_pilot.npz",
            BASE / f"F12/transport_full_r1/observations_{seed}_application.npz",
            BASE / f"F04/source_full_r1/L_{seed}/source.jsonl",
            BASE / f"F05/source_full_r2/core_{seed}/source.jsonl",
            DYN_RUN / f"{seed}/pilot_observations.npz",
            DYN_RUN / f"{seed}/application_observations.npz",
            DYN_RUN / f"{seed}/reference.json",
            TEXT_RUN / f"scores/{seed}/T1_primary_bundle.json",
            TEXT_RUN / f"scores/{seed}/dynamic_primary_bundle.json",
            F16 / f"t1_text_full_r1/{seed}/bundle.json",
        ]:
            inputs[rel(p)] = sha256_file(p)
    for p in [
        TEXT_RUN / "reader_scores.csv",
        TEXT_RUN / "channels.json",
        TEXT_RUN / "missing_anchor_controls.json",
        TEXT_RUN / "protocol_before_reading.json",
        TEXT_RUN / "output_manifest.json",
        T1_RUN / "output_manifest.json",
        DYN_RUN / "output_manifest.json",
    ]:
        inputs[rel(p)] = sha256_file(p)
    save(
        output / "protocol_before_run.json",
        {
            "packet": "F16-R03-SERIALIZE",
            "protocol": "docs/GRAIN_F16_SERIALIZE_PROTOCOL_V1.md",
            "code": {p: sha256_file(ROOT / p) for p in CODE if (ROOT / p).exists()},
            "inputs": inputs,
            "seeds": SEEDS,
            "fresh_text_reference_rendering": "SeedSequence([seed,172,int(dynamic)])",
            "tolerances": {
                "cold_replay": 1e-10,
                "numeric_t1_independent": 1e-9,
                "numeric_dynamic_independent": 1e-6,
                "reference_replay": 1e-10,
            },
            "new_source_fits": 0,
            "new_neural_calls": 0,
            "truth_files_opened": 0,
        },
    )
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for p in CODE:
        if (ROOT / p).exists():
            shutil.copyfile(ROOT / p, snapshot / Path(p).name)


def verify_frozen(output):
    """Refuse changed code or reused inputs after preparation."""
    protocol = load(output / "protocol_before_run.json")
    for key in ("code", "inputs"):
        for path, digest in protocol[key].items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError(f"changed frozen {key}: {path}")


def numeric_validation(route, seed, world, mode, estimator, target):
    """Attach archived finite SIM metrics, including failed gates, to one bundle."""
    run = T1_RUN if route == "numeric_t1" else DYN_RUN
    frame = pd.read_csv(run / "metrics_by_seed.csv")
    keep = (frame.world == world) & (frame.estimator == estimator)
    if route == "numeric_dynamic":
        keep &= frame.target == target
    rows = frame[keep & (frame.seed == seed)]
    same = rows[rows.arm == mode]
    references = rows[rows.arm.isin(["uncalibrated", "known_inverse"])]
    pooled = pd.read_csv(run / "metrics.csv")
    pooled = pooled[(pooled.world == world) & (pooled.estimator == estimator)]
    if route == "numeric_dynamic":
        pooled = pooled[pooled.target == target]
    this = records(same)
    if route == "numeric_t1":
        for row in this:
            if row["coverage"] is not None:
                low, high = wilson(row["coverage"], row["supported"])
                row.update(
                    wilson_low=low,
                    wilson_high=high,
                    gate=bool(low <= 0.95 <= high),
                    gate_note="Wilson95 recomputed from archived coverage by this packet",
                )
    failures = [
        f"seed {r['seed']} coverage gate failed" for r in this if r.get("gate") is False
    ]
    return {
        "scope": "finite SIM evaluation on the declared shifted application population;not human validity",
        "this_bundle": this,
        "same_world_information_references": records(references),
        "pooled_three_seed_means": records(pooled),
        "failed_gates": failures,
        "retained_route_failures": [
            "nonlinear cubic inverse refused NONMONOTONE_INVERSE in every seed",
            "missing-condition pilot anchors refused MISSING_ANCHOR (4 worlds x 3 seeds)",
            "nonlinear world: global/conditional affine calibrators are misspecified;coverage well below .95 retained",
        ],
    }


def numeric_calibration(route, seed, world, mode, refits):
    """Copy archived calibrators and prove they re-derive from pilot rows."""
    run = T1_RUN if route == "numeric_t1" else DYN_RUN
    relative = f"{seed}/{world}_{mode}.json"
    digest = manifest_hash(run, relative)
    stored = load(run / relative)
    refit = json.loads(json.dumps(serial(refits[(world, mode)])))
    reproduced = refit == stored
    if not reproduced:
        raise ValueError(f"calibrator refit mismatch {route} {relative}")
    pilot = (
        BASE / f"F12/transport_full_r1/observations_{seed}_pilot.npz"
        if route == "numeric_t1"
        else DYN_RUN / f"{seed}/pilot_observations.npz"
    )
    plugin = stored["plugin"]
    arrays = np.load(pilot)
    pilot_old, z = arrays["old"], arrays["conditions"]
    pilot_new = transform(pilot_old, z, world)
    indices = load(run / f"{seed}/pilot_bootstrap_indices.json")
    return {
        "kind": "numeric_affine_pilot",
        "mode": mode,
        "status": plugin["status"],
        "plugin": plugin,
        "draws": stored["draws"],
        "pilot_fit_pairs": {
            "old": pilot_old[:60].tolist(),
            "new": pilot_new[:60].tolist(),
            "conditions": z.tolist(),
        },
        "bootstrap_indices": indices,
        "draw_count": len(stored["draws"]),
        "rejected_draws": sum(d["status"] != "READY" for d in stored["draws"]),
        "data_provenance": {
            "role": "pilot",
            "file": rel(pilot),
            "sha256": sha256_file(pilot),
            "fit_authors": "first 60 of 100 paired pilot authors;last 40 validation only",
            "bootstrap": "200 whole-author resamples of the 60 fit authors,SeedSequence([seed,163,12])",
            "calibrator_file": rel(run / relative),
            "calibrator_sha256": digest,
            "refit_reproduced_exactly": reproduced,
        },
    }


def t1_reference_replay(engine, seed):
    """Re-score F04 source authors with the transported kernel formula."""
    people = group(BASE / f"F04/source_full_r1/L_{seed}/source.jsonl", "source")
    values = []
    for author in sorted(people):
        rows = sorted(people[author], key=lambda r: r["t"])
        z = np.array([r["z"] for r in rows], float)
        y = np.array([r["y"] for r in rows], float)
        kernels = t1_law.source_kernels(engine, z)
        values.append(
            float(
                t1_law.integrate(kernels, t1_law.cell_means(y, z)[None, None])["score"][
                    0
                ]
            )
        )
    values = np.array(values)
    return float(values.mean()), float(values.std(ddof=1)), len(values)


def dynamic_reference_replay(models, seed):
    """Re-score F05 core source authors with the compiled conditional-Gaussian rule."""
    _, ids, sy, z = source_records(seed)
    kernels = [dyn_law.compile_source(m, z, 64) for m in models]
    reference = dyn_law.collapse(
        dyn_law.integrate(kernels, sy[..., None], np.ones((1, 1))), slice(None)
    )
    scores = reference["mean"]
    return scores.mean(0), scores.std(0, ddof=1), len(ids)


def build_numeric(output, seed, refusals, provenance):
    """Build numeric T1 and dynamic bundles for every world/mode/estimator/target."""
    built = []
    t1_engine = load(F16 / f"t1_source_full_r1/{seed}/bundle.json")
    t1_source.verify_bundle(t1_engine)
    mu, sigma, count = t1_reference_replay(t1_engine, seed)
    t1_ref_error = max(
        abs(mu - t1_engine["reference_mu"]), abs(sigma - t1_engine["reference_sigma"])
    )
    provenance.append(
        {
            "seed": seed,
            "check": "numeric_t1_reference_replay",
            "max_error": t1_ref_error,
            "authors": count,
        }
    )
    if t1_ref_error > 1e-10:
        raise ValueError(
            "inherited T1 reference is not reproduced by the transported formula"
        )
    pilot = np.load(BASE / f"F12/transport_full_r1/observations_{seed}_pilot.npz")
    indices, t1_refits = fit_all(pilot["old"], pilot["conditions"], seed)
    stored_indices = load(T1_RUN / f"{seed}/pilot_bootstrap_indices.json")
    if serial(indices) != stored_indices:
        raise ValueError("T1 bootstrap indices not reproduced")
    dyn_source_path = DYN_RUN / f"{seed}/source_bundle.json"
    dyn_source = load(dyn_source_path)
    original = F16 / f"t2_cold_full_r1/{seed}/bundle.json"
    if sha256_file(dyn_source_path) != sha256_file(original):
        raise ValueError("dynamic transport source bundle is not the frozen t2 source")
    models = [x["model"] for x in dyn_source["source_models"]]
    ref = load(DYN_RUN / f"{seed}/reference.json")
    rmu, rsd, rcount = dynamic_reference_replay(models, seed)
    dyn_ref_error = float(
        max(np.abs(rmu - ref["mu"]).max(), np.abs(rsd - ref["sigma"]).max())
    )
    provenance.append(
        {
            "seed": seed,
            "check": "numeric_dynamic_reference_replay",
            "max_error": dyn_ref_error,
            "authors": rcount,
        }
    )
    if dyn_ref_error > 1e-10:
        raise ValueError("fresh dynamic reference not reproduced")
    dyn_pilot = np.load(DYN_RUN / f"{seed}/pilot_observations.npz")
    dyn_indices, dyn_refits = fit_all(dyn_pilot["old"], dyn_pilot["conditions"], seed)
    if serial(dyn_indices) != load(DYN_RUN / f"{seed}/pilot_bootstrap_indices.json"):
        raise ValueError("dynamic bootstrap indices not reproduced")
    dyn_engine = {
        "formula": "conditional-Gaussian64eta-source-cut-v1",
        "eta_nodes": 64,
        "source_components": len(models),
        "source_models": models,
    }
    for route, refits in (("numeric_t1", t1_refits), ("numeric_dynamic", dyn_refits)):
        for world in WORLDS:
            observation = {
                "route": "numeric",
                "definition": {
                    "channel_world": world,
                    "declared_forward_map": FORWARD[world],
                    "scorer_uses_forward_map": False,
                    "source_unit_reader": t1_source.READER_HASH
                    if route == "numeric_t1"
                    else "F05 core numeric response",
                    "observation": "changed-reader numeric response per event",
                },
                "cache_sha256": None,
            }
            for mode in MODES[world]:
                calibration = numeric_calibration(route, seed, world, mode, refits)
                provenance.append(
                    {
                        "seed": seed,
                        "check": f"{route}_calibrator_refit",
                        "world": world,
                        "mode": mode,
                        "exact": True,
                    }
                )
                targets = ("T1",) if route == "numeric_t1" else ("T2", "T3")
                for estimator in ("mixture", "plugin"):
                    for target in targets:
                        name = f"{route}_{seed}_{world}_{mode}_{estimator}_{target}"
                        if route == "numeric_t1":
                            reference = {
                                "kind": "INHERITED_COMMON_UNIT",
                                "mu": t1_engine["reference_mu"],
                                "sigma": t1_engine["reference_sigma"],
                                "authors": t1_engine["source_reference_authors"],
                                "application_derived": False,
                                "scope": "F04 L source authors in the original numeric response unit;same four-cell T1 function;calibration maps changed-reader observations into this unit;shifted application population is not renormalized",
                                "provenance": {
                                    "source_bundle": rel(
                                        F16 / f"t1_source_full_r1/{seed}/bundle.json"
                                    ),
                                    "replay_max_error": t1_ref_error,
                                },
                                "historical_positions": [],
                            }
                            engine = t1_engine
                            source = {
                                "formula_version": t1_engine["formula_version"],
                                "components": len(t1_engine["source_models"]),
                                "source_bundle": rel(
                                    F16 / f"t1_source_full_r1/{seed}/bundle.json"
                                ),
                                "source_bundle_sha256": sha256_file(
                                    F16 / f"t1_source_full_r1/{seed}/bundle.json"
                                ),
                                "source_observations_sha256": t1_engine["source_hash"],
                                "source_fits_sha256": t1_engine["source_fits_hash"],
                                "refit_in_this_packet": False,
                            }
                            extra = {
                                "source_components": 200,
                                "calibration_draws": 200
                                if estimator == "mixture"
                                else 1,
                                "components": 40000 if estimator == "mixture" else 200,
                                "formula": "mean=mean_j mean_s(a_s+w_s'v_j);var=mean_s V_s+mean_j Var_s+Var_j mean_s",
                                "numerical_error": "closed-form Gaussian kernels;no quadrature",
                            }
                            unit = "original numeric response unit;endpoint contrast"
                        else:
                            index = 0 if target == "T2" else 1
                            reference = {
                                "kind": "FRESH_SOURCE_SCORED",
                                "mu": ref["mu"][index],
                                "sigma": ref["sigma"][index],
                                "authors": ref["source_authors"],
                                "application_derived": False,
                                "scope": "F05 core source authors scored by the same conditional-Gaussian 64-node rule;legacy Laplace reference not reused",
                                "provenance": {
                                    "reference_file": rel(
                                        DYN_RUN / f"{seed}/reference.json"
                                    ),
                                    "sha256": sha256_file(
                                        DYN_RUN / f"{seed}/reference.json"
                                    ),
                                    "replay_max_error": dyn_ref_error,
                                },
                                "historical_positions": [],
                            }
                            engine = dyn_engine
                            source = {
                                "formula_version": dyn_engine["formula"],
                                "components": len(models),
                                "source_bundle": rel(original),
                                "source_bundle_sha256": sha256_file(original),
                                "source_observations_sha256": dyn_source["source_hash"][
                                    "source"
                                ],
                                "refit_in_this_packet": False,
                            }
                            extra = {
                                "source_components": 200,
                                "calibration_draws": 200
                                if estimator == "mixture"
                                else 1,
                                "components": 40000 if estimator == "mixture" else 200,
                                "eta_quadrature": "64-node Gauss-Hermite per source;(a,b,state) exact Gaussian conditioning",
                                "archived_quadrature_diagnostic": load(
                                    DYN_RUN / f"{seed}/{world}_{mode}_quadrature.json"
                                )
                                if (
                                    DYN_RUN / f"{seed}/{world}_{mode}_quadrature.json"
                                ).exists()
                                else None,
                            }
                            unit = "original numeric latent/response unit (F05 core)"
                        try:
                            bundle = pub.build_bundle(
                                route=route,
                                estimator=estimator,
                                seed=seed,
                                target=target,
                                unit=unit,
                                observation=observation,
                                source=source,
                                calibration=calibration,
                                reference=reference,
                                engine=engine,
                                known_validation=numeric_validation(
                                    route, seed, world, mode, estimator, target
                                ),
                                integration_extra=extra,
                            )
                        except pub.Refusal as refusal:
                            refusals.append(
                                {
                                    "name": name,
                                    "code": refusal.code,
                                    "detail": refusal.detail,
                                    "kind": "build",
                                }
                            )
                            continue
                        path = output / "bundles" / f"{name}.json"
                        save(path, bundle)
                        built.append(
                            {
                                "name": name,
                                "route": route,
                                "seed": seed,
                                "world": world,
                                "mode": mode,
                                "estimator": estimator,
                                "target": target,
                                "sha256": sha256_file(path),
                                "bundle_id": bundle["bundle_id"],
                            }
                        )
        for world in WORLDS:
            run = T1_RUN if route == "numeric_t1" else DYN_RUN
            for certificate in load(run / f"{seed}/missing_anchors.json"):
                refusals.append(
                    {
                        "name": f"{route}_{seed}_{certificate['world']}_missing_negative_condition_anchor",
                        "code": certificate["status"],
                        "detail": certificate.get("required_anchor"),
                        "kind": "build_certificate_reused",
                    }
                )
            break
    return built


def text_bank():
    """Actual readout table from the text transport packet."""
    return pd.read_csv(TEXT_RUN / "reader_scores.csv").to_dict("records")


def render_source(groups, bank, seed, dynamic_target):
    """Render source authors onto source-role materials with a declared stream."""
    pools = {
        (b, q): [
            r
            for r in bank
            if r["role"] == "source" and r["pole"] == b and r["qualification"] == q
        ]
        for b in (0, 1)
        for q in (0, 1)
    }
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 172, int(dynamic_target)])
    )
    target_time = 47 if dynamic_target else 31
    contract = (
        dyn_text.CONTRACT if dynamic_target else "F04_L_thresholded_text_eligibility_v1"
    )
    inputs = []
    for author in sorted(groups):
        history = []
        for r in groups[author][: target_time + 1]:
            q = int(rng.integers(2))
            b = int(r["y"] >= 0) if q else int(rng.integers(2))
            choices = pools[b, q]
            material = choices[int(rng.integers(len(choices)))]
            obs = {k: r[k] for k in ("author_id", "event_id", "t", "z")}
            obs["material_id"] = material["material_id"]
            history.append(
                {
                    "record_id": r["event_id"],
                    "recorded_at": r["t"],
                    "op": "new",
                    "observation": obs,
                }
            )
        inputs.append(
            {
                "author_id": author,
                "target_time": target_time,
                "world_contract": contract,
                "history": history,
            }
        )
    return inputs


def text_validation(seed, target, estimator):
    """Archived finite text-transport metrics for this bundle and comparators."""
    frame = pd.read_csv(TEXT_RUN / "metrics_by_seed.csv")
    arm = f"changed_pilot_{estimator}"
    rows = frame[(frame.seed == seed) & (frame.target == target)]
    pooled = pd.read_csv(TEXT_RUN / "metrics.csv")
    pooled = pooled[pooled.target == target]
    contrasts = [
        c for c in load(TEXT_RUN / "paired_contrasts.json") if c["target"] == target
    ]
    return {
        "scope": "finite controlled-text SIM targets;shared 380-material bank;not human personality accuracy",
        "this_bundle": records(rows[rows.arm == arm]),
        "comparators_same_seed": records(rows[rows.arm != arm]),
        "pooled_three_seed_means": records(pooled),
        "paired_contrasts": contrasts,
        "failed_gates": [],
        "retained_route_failures": [
            "old observation law under the changed reader worsens T3 MSE .407543->.485925 (uncalibrated arm)",
            "pilot calibration does not reach the known-pool reference (finite pilot does not identify the application law exactly)",
            "four missing pole/qualification anchor controls refused MISSING_ANCHOR",
            "T2 improvement interval crosses zero;not promoted",
        ],
    }


def build_text(output, seed, refusals, provenance):
    """Reconcile legacy payloads, score fresh source references, then build bundles."""
    built = []
    bank = text_bank()
    laws = load(TEXT_RUN / "channels.json")
    pilot_rows = [
        r for r in bank if r["role"] == "pilot" and r["pilot_partition"] == "fit"
    ]
    recount = text_law.calibrated_channel(
        pilot_rows, "nli_qualification", required_role="pilot", partition="fit"
    )
    if recount["counts"] != laws["pilot"]["counts"]:
        raise ValueError("pilot channel counts not reproduced from reader_scores.csv")
    cache = json.loads(json.dumps(text_law.observation_cache(bank, "new")))
    mixture_draws = text_law.channels(
        laws["pilot"], np.random.SeedSequence([seed, 170, 1]), 200, "mixture"
    )
    plugin_draws = text_law.channels(laws["pilot"], 0, 200, "plugin")
    outer = {
        "protocol_before_reading_sha256": sha256_file(
            TEXT_RUN / "protocol_before_reading.json"
        ),
        "reader_scores_sha256": manifest_hash(TEXT_RUN, "reader_scores.csv"),
        "output_manifest_sha256": sha256_file(TEXT_RUN / "output_manifest.json"),
    }
    definition = {
        "reader": "changed qualification reader (F16-R03-TEXT new arm)",
        "nli_model": MODEL,
        "nli_revision": REVISION,
        "queries": QUERIES,
        "pole_readout": "p_ent(positive)/(p_ent(positive)+p_ent(negative))",
        "qualification_readout": "p_ent(qualification query)",
        "category": "2*(pole>=.5)+(qualification>=.5)",
        "engine_slot_alias": "engine slots grain_qualification and nli_qualification both carry the NLI qualification readout;the historical GRAIN comment-view head is not used",
        "cold_path": "registered cached materials only;a new text requires running this declared reader in a separate reading stage",
    }
    for route, file_name, source_path in (
        (
            "text_t1",
            "T1_primary_bundle.json",
            BASE / f"F04/source_full_r1/L_{seed}/source.jsonl",
        ),
        (
            "text_dynamic",
            "dynamic_primary_bundle.json",
            BASE / f"F05/source_full_r2/core_{seed}/source.jsonl",
        ),
    ):
        relative = f"scores/{seed}/{file_name}"
        primary_sha = manifest_hash(TEXT_RUN, relative)
        primary = load(TEXT_RUN / relative)
        if primary["materials"] != cache:
            raise ValueError("primary reader cache differs from reader_scores.csv")
        for key in ("hybrid_joint", "nli_joint"):
            if not np.array_equal(
                np.asarray(primary["channel_draws"][key]), mixture_draws
            ):
                raise ValueError(
                    "primary channel draws not regenerated from pilot counts"
                )
        provenance.append(
            {
                "seed": seed,
                "check": f"{route}_cache_counts_draws_regenerated",
                "exact": True,
            }
        )
        dynamic_target = route == "text_dynamic"
        stale = [k for k in pub.STALE_TEXT_FIELDS if k in primary]
        reference_keys = (
            ["reference", "source_reference_authors"]
            if dynamic_target
            else ["reference_mu", "reference_sigma", "reference_authors"]
        )
        moved = {k: primary[k] for k in stale + reference_keys}
        legacy_check = None
        if "reader_hash" in primary:
            import hashlib

            def legacy_formula(materials, channels):
                return hashlib.sha256(
                    json.dumps(
                        {"materials": materials, "channels": channels}, sort_keys=True
                    ).encode()
                ).hexdigest()

            old = load(F16 / f"t1_text_full_r1/{seed}/bundle.json")
            legacy_check = {
                "legacy_inner_reader_hash": primary["reader_hash"],
                "recomputed_over_old_reader_bundle": legacy_formula(
                    old["materials"], old["channel_draws"]
                ),
                "recomputed_over_changed_reader_payload": legacy_formula(
                    primary["materials"], primary["channel_draws"]
                ),
            }
            legacy_check["identifies_old_reader"] = (
                legacy_check["legacy_inner_reader_hash"]
                == legacy_check["recomputed_over_old_reader_bundle"]
            )
            legacy_check["identifies_changed_reader"] = (
                legacy_check["legacy_inner_reader_hash"]
                == legacy_check["recomputed_over_changed_reader_payload"]
            )
            if (
                legacy_check["identifies_changed_reader"]
                or not legacy_check["identifies_old_reader"]
            ):
                raise ValueError("unexpected legacy reader hash semantics")
        groups = group(source_path, "source")
        if len(groups) != 300:
            raise ValueError("source panel incomplete")
        source_inputs = render_source(groups, bank, seed, dynamic_target)
        write_jsonl(
            output / "fresh_reference" / f"{route}_{seed}_source_inputs.jsonl",
            source_inputs,
        )
        for estimator, draws in (("mixture", mixture_draws), ("plugin", plugin_draws)):
            engine = {k: copy.deepcopy(v) for k, v in primary.items() if k not in moved}
            engine["channel_draws"] = {
                k: draws.tolist() for k in ("hybrid_joint", "nli_joint")
            }
            for key in SCORE_FIELDS[route]:
                if pub.object_hash(engine[key]) != pub.object_hash(primary[key]):
                    raise ValueError(f"score-determining field changed: {key}")
            if dynamic_target:
                engine["reference"] = {
                    t: {"mu": 0.0, "sigma": 1.0} for t in ("T2", "T3")
                }
                scores = dyn_text.cold_scores(source_inputs, engine)
                fresh = {}
                for t in ("T2", "T3"):
                    v = np.array([s["targets"][t]["score"] for s in scores])
                    fresh[t] = {"mu": float(v.mean()), "sigma": float(v.std(ddof=1))}
                engine["reference"] = fresh
                engine["source_reference_authors"] = len(scores)
            else:
                engine.update(reference_mu=0.0, reference_sigma=1.0)
                scores = t1_text.cold_scores(source_inputs, engine)
                v = np.array([s["score"] for s in scores])
                fresh = {"T1": {"mu": float(v.mean()), "sigma": float(v.std(ddof=1))}}
                engine.update(
                    reference_mu=fresh["T1"]["mu"],
                    reference_sigma=fresh["T1"]["sigma"],
                    reference_authors=len(scores),
                )
            scores_path = (
                output
                / "fresh_reference"
                / f"{route}_{seed}_{estimator}_source_scores.json"
            )
            save(scores_path, scores)
            calibration = {
                "kind": "categorical_channel_pilot",
                "estimator": estimator,
                "status": "SUPPORTED",
                "counts": laws["pilot"]["counts"],
                "prior": "Jeffreys row Dirichlet(counts+0.5) draws"
                if estimator == "mixture"
                else "plugin (counts+0.5)/row sum repeated;calibration uncertainty NOT propagated",
                "generator": {
                    "function": "f16_t1_text.channel_draws",
                    "seed_sequence": [seed, 170, 1],
                    "draws": 200,
                },
                "data_provenance": {
                    "role": "pilot",
                    "partition": "fit",
                    "material_ids": laws["pilot"]["material_ids"],
                    "reader_scores_sha256": outer["reader_scores_sha256"],
                    "recount_reproduced_exactly": True,
                },
            }
            legacy = {
                "frozen_primary_payload": {
                    "path": rel(TEXT_RUN / relative),
                    "sha256": primary_sha,
                    "authenticated_by": "text_transport_full_r1/output_manifest.json",
                },
                "outer_packet": outer,
                "score_determining_fields": {
                    k: pub.object_hash(primary[k]) for k in SCORE_FIELDS[route]
                },
                "score_determining_fields_identical": True,
                "channel_draws": "identical to primary (mixture)"
                if estimator == "mixture"
                else "plugin law regenerated from the same pilot counts;primary payload holds mixture draws",
                "moved_out": {
                    k: {
                        "value_sha256": pub.object_hash(old_value),
                        "value": old_value,
                        "reason": REASONS[k],
                    }
                    for k, old_value in moved.items()
                },
                "inherited_inner_reader_hash_check": legacy_check,
                "authentication": "the changed reader is authenticated by the outer packet manifests and the recomputed public reader_id;no inherited inner reader field is used",
            }
            targets = ("T1",) if route == "text_t1" else ("T2", "T3")
            for target in targets:
                name = f"{route}_{seed}_{estimator}_{target}"
                historical = (
                    primary["reference"][target]
                    if dynamic_target
                    else {
                        "mu": primary["reference_mu"],
                        "sigma": primary["reference_sigma"],
                    }
                )
                reference = {
                    "kind": "FRESH_SOURCE_SCORED",
                    "mu": fresh[target]["mu"],
                    "sigma": fresh[target]["sigma"],
                    "authors": len(scores),
                    "application_derived": False,
                    "scope": "300 archived source authors rendered on 80 source-role controlled materials,read by the changed reader,scored by this bundle's engine;not a population,human or application norm",
                    "provenance": {
                        "rendering": f"SeedSequence([{seed},172,{int(dynamic_target)}])",
                        "source_file": rel(source_path),
                        "source_sha256": sha256_file(source_path),
                        "source_inputs_sha256": sha256_file(
                            output
                            / "fresh_reference"
                            / f"{route}_{seed}_source_inputs.jsonl"
                        ),
                        "source_scores_sha256": sha256_file(scores_path),
                    },
                    "historical_positions": [
                        {
                            "label": "old reader,old 96-text bank source reference",
                            "mu": historical["mu"],
                            "sigma": historical["sigma"],
                            "status": "NOT_SAME_TARGET_FUNCTION;not used for positions",
                        }
                    ],
                }
                if dynamic_target:
                    reference["companion"] = fresh["T3" if target == "T2" else "T2"]
                observation = {
                    "route": "controlled_text",
                    "definition": definition,
                    "cache_sha256": pub.text_materials_digest(engine["materials"]),
                    "cache_provenance": {
                        **outer,
                        "materials": len(engine["materials"]),
                    },
                }
                source = {
                    "formula_version": engine["version"],
                    "components": engine["source_components"],
                    "source_bundle_sha256": primary.get(
                        "source_bundle_hash", primary.get("source_bundle_sha256")
                    ),
                    "refit_in_this_packet": False,
                }
                diagnostics = [
                    d
                    for d in load(TEXT_RUN / f"scores/{seed}/diagnostics.json")
                    if d["target"] == ("dynamic" if dynamic_target else "T1")
                ]
                extra = {
                    "source_components": 200,
                    "channel_draws": 200,
                    "components": 200,
                    "pairing": "b-th source model with b-th channel draw;independent product law approximated by 200 paired draws,NOT 40000 Cartesian",
                    "qmc": {
                        "nodes_per_component": 2 ** engine["qmc_power"],
                        "scramble": "SeedSequence([seed,164,b,0])"
                        if dynamic_target
                        else "SeedSequence([seed,161,b,0])",
                        "state": "ADF moment projection" if dynamic_target else None,
                    },
                    "archived_numerical_diagnostics_first_last_author": diagnostic_summary(
                        diagnostics, target, dynamic_target
                    )
                    if estimator == "mixture"
                    else "not archived for plugin",
                }
                try:
                    bundle = pub.build_bundle(
                        route=route,
                        estimator=estimator,
                        seed=seed,
                        target=target,
                        unit=engine["unit"],
                        observation=observation,
                        source=source,
                        calibration=calibration,
                        reference=reference,
                        engine=engine,
                        known_validation=text_validation(seed, target, estimator),
                        legacy_reconciliation=legacy,
                        integration_extra=extra,
                    )
                except pub.Refusal as refusal:
                    refusals.append(
                        {
                            "name": name,
                            "code": refusal.code,
                            "detail": refusal.detail,
                            "kind": "build",
                        }
                    )
                    continue
                path = output / "bundles" / f"{name}.json"
                save(path, bundle)
                built.append(
                    {
                        "name": name,
                        "route": route,
                        "seed": seed,
                        "estimator": estimator,
                        "target": target,
                        "sha256": sha256_file(path),
                        "bundle_id": bundle["bundle_id"],
                        "fresh_reference_inputs": rel(
                            output
                            / "fresh_reference"
                            / f"{route}_{seed}_source_inputs.jsonl"
                        ),
                        "fresh_reference_scores": rel(scores_path),
                    }
                )
    return built


def diagnostic_summary(diagnostics, target, dynamic_target):
    """Max score change for doubled QMC nodes and replaced channel draws."""
    out = {}
    for d in diagnostics:

        def value(row):
            return row["targets"][target]["score"] if dynamic_target else row["score"]

        out["double_nodes_max_abs"] = max(
            abs(value(a) - value(b))
            for a, b in zip(d["base"], d["double_nodes"], strict=True)
        )
        out["alternate_channel_draws_max_abs"] = max(
            abs(value(a) - value(b))
            for a, b in zip(d["base"], d["alternate_channel_draws"], strict=True)
        )
    return out


def text_refusals(refusals):
    """Actual attempted constructions that must refuse."""
    laws = load(TEXT_RUN / "channels.json")
    controls = load(TEXT_RUN / "missing_anchor_controls.json")
    base = {
        "route": "text_t1",
        "estimator": "mixture",
        "seed": SEEDS[0],
        "target": "T1",
        "unit": "u",
        "observation": {
            "route": "controlled_text",
            "definition": {},
            "cache_sha256": None,
        },
        "source": {},
        "reference": {"kind": "FRESH_SOURCE_SCORED", "application_derived": False},
        "engine": {},
        "known_validation": {},
    }
    attempts = [
        (
            "text_known_pool_law",
            {
                "kind": "categorical_channel_pilot",
                "estimator": "mixture",
                "status": laws["known_pool"]["status"],
                "counts": laws["known_pool"]["counts"],
                "data_provenance": {"role": "application"},
            },
        )
    ]
    attempts.append(
        (
            "text_oracle_labels",
            {
                "kind": "categorical_channel_pilot",
                "estimator": "mixture",
                "status": "SUPPORTED",
                "data_provenance": {"role": "oracle"},
            },
        )
    )
    for name, control in controls.items():
        attempts.append(
            (
                f"text_{name}",
                {
                    "kind": "categorical_channel_pilot",
                    "estimator": "mixture",
                    "status": control["status"],
                    "counts": control["counts"],
                    "data_provenance": {"role": "pilot"},
                },
            )
        )
    for name, calibration in attempts:
        try:
            pub.build_bundle(calibration=calibration, **base)
            raise ValueError(f"{name} was not refused")
        except pub.Refusal as refusal:
            refusals.append(
                {
                    "name": name,
                    "code": refusal.code,
                    "detail": refusal.detail,
                    "kind": "build",
                }
            )
    try:
        pub.build_bundle(
            calibration={
                "kind": "numeric_affine_pilot",
                "status": "READY",
                "plugin": {"status": "READY"},
                "draws": [],
                "data_provenance": {"role": "pilot"},
            },
            **dict(
                base,
                route="numeric_t1",
                estimator="plugin",
                reference={
                    "kind": "INHERITED_COMMON_UNIT",
                    "application_derived": True,
                },
            ),
        )
        raise ValueError("application reference was not refused")
    except pub.Refusal as refusal:
        refusals.append(
            {
                "name": "application_derived_reference",
                "code": refusal.code,
                "detail": refusal.detail,
                "kind": "build",
            }
        )


def build(output):
    """Build all bundles and refusals; no application truth is opened."""
    verify_frozen(output)
    refusals, provenance, built = [], [], []
    for seed in SEEDS:
        started = time.monotonic()
        built += build_numeric(output, seed, refusals, provenance)
        built += build_text(output, seed, refusals, provenance)
        print(
            json.dumps(
                {
                    "stage": "build",
                    "seed": seed,
                    "bundles": len(built),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    text_refusals(refusals)
    save(output / "bundle_index.json", built)
    save(output / "refusals.json", refusals)
    save(output / "provenance_checks.json", provenance)


def requests_for(entry, bundle):
    """Observation envelopes for the canonical application panel, no truth."""
    seed, route = entry["seed"], entry["route"]
    if route == "numeric_t1":
        arr = np.load(
            BASE / f"F12/transport_full_r1/observations_{seed}_application.npz"
        )
    elif route == "numeric_dynamic":
        arr = np.load(DYN_RUN / f"{seed}/application_observations.npz")
    if route.startswith("numeric"):
        z = arr["conditions"]
        new = transform(arr["old"], z, entry["world"])
        return [
            pub.make_request(
                bundle,
                {
                    "author_id": f"app_{seed}_{i:03d}",
                    "records": [
                        {"t": k, "z": float(z[k]), "y": float(new[i, k])}
                        for k in range(len(z))
                    ],
                },
            )
            for i in range(len(new))
        ]
    name = "T1_inputs.jsonl" if route == "text_t1" else "dynamic_inputs.jsonl"
    raw = [
        json.loads(x)
        for x in (TEXT_RUN / f"scores/{seed}/{name}").read_text().splitlines()
    ]
    return [pub.make_request(bundle, r) for r in raw]


def run_sealed(bundle_path, digest, input_path, output_path, probe=None):
    """Launch one fresh sealed process with single-thread BLAS."""
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    command = [
        sys.executable,
        "-m",
        "grain_research.gptpro_foundation_v3.f16_serialize",
        "score",
        "--bundle",
        str(bundle_path),
        "--sha256",
        digest,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if probe:
        command += ["--probe", str(probe)]
    started = time.monotonic()
    done = subprocess.run(
        command, env=env, capture_output=True, text=True, cwd=str(ROOT), check=False
    )
    return {
        "returncode": done.returncode,
        "stderr": done.stderr[-4000:],
        "seconds": time.monotonic() - started,
    }


def cold(output, workers):
    """Replay every bundle on its canonical application inputs in sealed processes."""
    verify_frozen(output)
    index = load(output / "bundle_index.json")
    jobs = []
    for entry in index:
        bundle_path = output / "bundles" / f"{entry['name']}.json"
        if sha256_file(bundle_path) != entry["sha256"]:
            raise ValueError("bundle changed after build")
        bundle = load(bundle_path)
        req = output / "cold" / f"{entry['name']}.requests.jsonl"
        write_jsonl(req, requests_for(entry, bundle))
        jobs.append(
            (entry, bundle_path, req, output / "cold" / f"{entry['name']}.scores.jsonl")
        )
        if entry["route"].startswith("text"):
            ref_req = output / "cold" / f"{entry['name']}.reference_requests.jsonl"
            source_inputs = [
                json.loads(x)
                for x in (ROOT / entry["fresh_reference_inputs"])
                .read_text()
                .splitlines()
            ]
            write_jsonl(ref_req, [pub.make_request(bundle, r) for r in source_inputs])
            jobs.append(
                (
                    entry,
                    bundle_path,
                    ref_req,
                    output / "cold" / f"{entry['name']}.reference_scores.jsonl",
                )
            )
    log = []

    def launch(job):
        entry, bundle_path, req, out = job
        result = run_sealed(bundle_path, entry["sha256"], req, out)
        result.update(name=entry["name"], output=rel(out))
        print(
            json.dumps(
                {k: result[k] for k in ("name", "output", "returncode", "seconds")}
            ),
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        log = list(pool.map(launch, jobs))
    save(output / "cold_process_log.json", log)
    if any(r["returncode"] for r in log):
        raise ValueError("a sealed cold process failed")


def rejections(output):
    """Actual wrong-contract, tamper, stale-field, support and sealing attempts."""
    verify_frozen(output)
    index = load(output / "bundle_index.json")
    pick = {
        route: next(
            e
            for e in index
            if e["route"] == route
            and e["seed"] == SEEDS[0]
            and e["estimator"] == "mixture"
            and e.get("world", "conditional") == "conditional"
            and e.get("mode", "conditional") == "conditional"
            and e["target"] in ("T1", "T3")
        )
        for route in pub.ROUTE_TARGETS
    }
    folder = output / "rejections"
    folder.mkdir()
    cases = []
    other_t2 = next(
        e
        for e in index
        if e["route"] == "text_dynamic"
        and e["seed"] == SEEDS[0]
        and e["estimator"] == "mixture"
        and e["target"] == "T2"
    )
    for route, entry in pick.items():
        bundle = load(output / "bundles" / f"{entry['name']}.json")
        good = requests_for(entry, bundle)[0]
        variants = []
        wrong_reader = copy.deepcopy(good)
        wrong_reader["contract"]["reader_id"] = pub.object_hash(
            {"legacy": "old reader"}
        )
        variants.append(("wrong_reader", wrong_reader, "WRONG_READER"))
        wrong_target = copy.deepcopy(good)
        wrong_target["contract"]["target_id"] = (
            "T2" if bundle["target"]["target_id"] != "T2" else "T3"
        )
        variants.append(("wrong_target", wrong_target, "WRONG_TARGET"))
        wrong_version = copy.deepcopy(good)
        wrong_version["contract"]["bundle_id"] = load(
            output / "bundles" / f"{other_t2['name']}.json"
        )["bundle_id"]
        variants.append(("wrong_bundle_version", wrong_version, "WRONG_VERSION"))
        wrong_clock = copy.deepcopy(good)
        wrong_clock["contract"]["target_time"] += 1
        variants.append(("wrong_clock", wrong_clock, "WRONG_CLOCK"))
        truth = copy.deepcopy(good)
        truth["data"]["truth"] = 0.0
        variants.append(("hidden_truth_field", truth, "HIDDEN_OR_UNDECLARED_FIELD"))
        if route == "numeric_t1":
            partial = copy.deepcopy(good)
            partial["data"]["records"] = [
                r for r in partial["data"]["records"] if r["z"] != -1.0
            ]
            variants.append(
                (
                    "incomplete_condition_support",
                    partial,
                    "INCOMPLETE_CONDITION_SUPPORT",
                )
            )
            off = copy.deepcopy(good)
            off["data"]["records"][0]["z"] = 0.5
            variants.append(("uncalibrated_condition", off, "UNSUPPORTED_CONDITION"))
        elif route == "numeric_dynamic":
            gap = copy.deepcopy(good)
            gap["data"]["records"] = gap["data"]["records"][:-1]
            variants.append(("missing_last_step", gap, "INCOMPLETE_CLOCK_SUPPORT"))
            late = copy.deepcopy(good)
            late["data"]["records"][-1]["t"] = 48
            variants.append(("event_after_cutoff", late, "WRONG_CLOCK"))
        else:
            unknown = copy.deepcopy(good)
            unknown["data"]["history"][0]["observation"]["material_id"] = (
                "unregistered_new_text"
            )
            variants.append(("unregistered_text", unknown, "UNREGISTERED_TEXT"))
            future = copy.deepcopy(good)
            future["data"]["history"][-1]["recorded_at"] = (
                future["data"]["target_time"] + 5
            )
            variants.append(("report_after_cutoff", future, "WRONG_CLOCK"))
        req = folder / f"{route}_requests.jsonl"
        write_jsonl(req, [v[1] for v in variants])
        out = folder / f"{route}_scores.jsonl"
        probe = F16 / (
            "t1_transport_full_r1/20260915/evaluation_hashes.json"
            if route == "numeric_t1"
            else "dynamic_transport_full_r1/20260915/application_truth.npz"
            if route == "numeric_dynamic"
            else "text_transport_full_r1/scores/20260915/T1_oracle_rendering.json"
        )
        result = run_sealed(
            output / "bundles" / f"{entry['name']}.json",
            entry["sha256"],
            req,
            out,
            probe=probe,
        )
        if result["returncode"]:
            raise ValueError(result["stderr"])
        rows = [json.loads(x) for x in out.read_text().splitlines()]
        manifest = load(str(out) + ".manifest.json")
        for (label, _, expected), row in zip(variants, rows, strict=True):
            cases.append(
                {
                    "route": route,
                    "case": label,
                    "expected": expected,
                    "status": row["status"],
                    "code": row.get("code"),
                    "pass": row.get("code") == expected,
                }
            )
        cases.append(
            {
                "route": route,
                "case": "sealed_undeclared_truth_read",
                "expected": "REFUSED",
                "status": "REFUSED" if manifest["probe"]["refused"] else "READ",
                "code": manifest["probe"].get("error"),
                "pass": manifest["probe"]["refused"],
            }
        )
        for label, obj, expected in bundle_mutations(bundle, route):
            try:
                pub.verify(obj)
                code = None
            except pub.Refusal as refusal:
                code = refusal.code
            cases.append(
                {
                    "route": route,
                    "case": label,
                    "expected": expected,
                    "status": "REFUSED" if code else "ACCEPTED",
                    "code": code,
                    "pass": code == expected,
                }
            )
    save(folder / "rejection_cases.json", cases)
    failed = [c for c in cases if not c["pass"]]
    if failed:
        raise ValueError(f"rejection expectation failed: {failed}")


def bundle_mutations(bundle, route):
    """Bundle-level tamper cases; re-identified variants isolate the semantic check."""

    def reidentify(obj):
        obj["bundle_id"] = pub.body_id(obj)
        return obj

    text = route.startswith("text")
    tampered = copy.deepcopy(bundle)
    if text:
        for key in ("hybrid_joint", "nli_joint"):
            draws = tampered["engine"]["channel_draws"][key]
            draws[0], draws[1] = draws[1], draws[0]
    else:
        tampered["calibration"]["plugin"]["coefficients"][0] += 1e-6
    cases = [
        (
            "tampered_calibration_without_reidentification",
            tampered,
            "BUNDLE_HASH_MISMATCH",
        ),
        (
            "tampered_calibration_reidentified",
            reidentify(copy.deepcopy(tampered)),
            "CALIBRATION_DRAWS_NOT_REPRODUCIBLE",
        ),
    ]
    norm = copy.deepcopy(bundle)
    norm["reference_policy"]["application_derived"] = True
    cases.append(
        (
            "application_reference_reidentified",
            reidentify(norm),
            "WRONG_REFERENCE_POLICY",
        )
    )
    if route != "numeric_dynamic":
        edited = copy.deepcopy(bundle)
        edited["reference_policy"]["mu"] += 0.1
        cases.append(
            (
                "reference_edited_reidentified",
                reidentify(edited),
                "WRONG_REFERENCE_POLICY",
            )
        )
    if text:
        stale = copy.deepcopy(bundle)
        stale["engine"]["reader_hash"] = (
            "1fb0e38ef472a406ce4d9476a0bc5d09a393d0d6e75b7a703cc034664fff8709"
        )
        cases.append(
            (
                "stale_inner_reader_hash_restored",
                reidentify(stale),
                "STALE_LEGACY_FIELD_PRESENT",
            )
        )
        swap = copy.deepcopy(bundle)
        first = min(swap["engine"]["materials"])
        item = swap["engine"]["materials"][first]
        item["grain_qualification"] = item["nli_qualification"] = (
            1 - item["nli_qualification"]
        )
        cases.append(
            ("reader_cache_edited_reidentified", reidentify(swap), "WRONG_READER")
        )
    else:
        pair = copy.deepcopy(bundle)
        pair["calibration"]["pilot_fit_pairs"]["new"][0][0] += 1e-3
        cases.append(
            (
                "pilot_pair_edited_reidentified",
                reidentify(pair),
                "CALIBRATION_DRAWS_NOT_REPRODUCIBLE",
            )
        )
    version = copy.deepcopy(bundle)
    version["schema_version"] = "F16_public_source_compilation_v1"
    cases.append(("wrong_schema_version", version, "WRONG_BUNDLE_VERSION"))
    code = copy.deepcopy(bundle)
    code["code_manifest"]["f16_t1_text.py"] = "0" * 64
    cases.append(("engine_code_hash_changed", reidentify(code), "CODE_CHANGED"))
    return cases


def main():
    """Run one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "build", "cold", "reject"), required=True
    )
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage == "build":
        build(args.output)
    elif args.stage == "cold":
        cold(args.output, args.workers)
    else:
        rejections(args.output)


if __name__ == "__main__":
    main()
