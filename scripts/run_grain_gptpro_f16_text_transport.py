#!/usr/bin/env python3
"""Freeze materials, read actual texts, calibrate and score all three F16 targets."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from grain_research.gptpro_foundation_v3 import f16_dynamic_text as dynamic
from grain_research.gptpro_foundation_v3 import f16_t1_text as static
from grain_research.gptpro_foundation_v3 import f16_text_transport as transport
from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.run_grain_gptpro_f08_nli_reader import QUERIES
from scripts.run_grain_gptpro_f11_grain_reader import predict_readout
from scripts.run_grain_gptpro_f16_t1_text import save, write_inputs
from scripts.run_grain_gptpro_f16_t2 import group

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_gptpro_foundation_v3"
SEEDS = (20260915, 20260916, 20260917)
RUNTIME = (
    ROOT
    / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
)
HEAD = BASE / "F11/grain_comment_reader_r1/readout_bundle.json"


def prepare(output):
    """Freeze raw material text and all planned estimator inputs before reading."""
    output.mkdir(parents=True, exist_ok=False)
    bank = pd.DataFrame(transport.material_bank())
    old = pd.read_csv(BASE / "F08/nli_reader_r1/reader_scores.csv")
    if set(bank.text) & set(old.text):
        raise ValueError("new bank overlaps old materials")
    bank.to_csv(output / "materials.csv", index=False)
    code = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f16_text_transport.py",
        ROOT / "grain_research/gptpro_foundation_v3/f16_t1_text.py",
        ROOT / "grain_research/gptpro_foundation_v3/f16_dynamic_text.py",
        ROOT / "scripts/run_grain_evidence_calibration.py",
        ROOT / "scripts/run_grain_gptpro_f08_nli_reader.py",
        ROOT / "tests/test_grain_gptpro_f16_text_transport.py",
        ROOT / "docs/GRAIN_F16_TEXT_TRANSPORT_PROTOCOL_V1.md",
    ]
    inputs = [RUNTIME, HEAD]
    for seed in SEEDS:
        inputs += [
            BASE / f"F16/t1_text_full_r1/{seed}/bundle.json",
            BASE / f"F16/dynamic_text_full_r1/{seed}/bundle.json",
            BASE / f"F04/source_full_r1/L_{seed}/application.jsonl",
            BASE / f"F05/source_full_r2/core_{seed}/application.jsonl",
        ]
    save(
        output / "protocol_before_reading.json",
        {
            "code": {str(p): sha256_file(p) for p in code},
            "inputs": {str(p): sha256_file(p) for p in inputs},
            "bank_sha256": sha256_file(output / "materials.csv"),
            "arms": transport.ARMS,
            "seeds": SEEDS,
            "queries": QUERIES,
            "material_roles": {
                "source": 80,
                "pilot_fit": 60,
                "pilot_validation": 40,
                "application": 200,
            },
            "neural_pairs": 1140,
            "neural_training": False,
            "API_calls": 0,
            "application_per_seed": 200,
            "source_components": 200,
            "source_reader_integration": "200 independently sampled paired cut components,not40000Cartesianproduct",
            "numeric_source_refits": 0,
            "scope": "controlledpublictexts on reusedSIMtargets;not natural/human personality accuracy",
        },
    )
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for p in code:
        (snapshot / p.name).write_bytes(p.read_bytes())


def verify(output):
    """Refuse silently changed primary code, materials or frozen source inputs."""
    contract = json.loads((output / "protocol_before_reading.json").read_text())
    for key in ("code", "inputs"):
        for path, digest in contract[key].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"changed frozen {key}: {path}")
    if sha256_file(output / "materials.csv") != contract["bank_sha256"]:
        raise ValueError("changed material bank")


def read_materials(output):
    """Run pinned NLI and historical GRAIN on the new raw texts, without labels."""
    import joblib

    from scripts.run_grain_evidence_calibration import infer_pairs

    if (output / "reader_scores.csv").exists():
        raise ValueError("readings already exist; do not rerun")
    bank = pd.read_csv(output / "materials.csv")
    pairs = pd.DataFrame(
        [
            {
                "audit_id": f"{r.material_id}:{name}",
                "sentence": r.text,
                "item_text": query,
            }
            for r in bank.itertuples()
            for name, query in QUERIES.items()
        ]
    )
    pairs.to_csv(output / "nli_inputs.csv", index=False)
    began = time.monotonic()
    readings = infer_pairs(pairs)
    readings.to_csv(output / "nli_probabilities.csv", index=False)
    runtime = joblib.load(RUNTIME)
    head = json.loads(HEAD.read_text())
    features = runtime.transform(bank.text.tolist())
    raw, clipped = predict_readout(features, head)
    np.savez_compressed(output / "grain_features.npz", features=features, raw=raw)
    indexed = readings.set_index("audit_id")
    rows = []
    for i, r in enumerate(bank.to_dict("records")):
        material = r["material_id"]
        pos, neg = [
            float(indexed.loc[f"{material}:{k}", "p_entailment"])
            for k in ("positive", "negative")
        ]
        if pos + neg <= 0:
            raise ValueError("undefined NLI content ratio")
        rows.append(
            dict(
                r,
                pole_readout=pos / (pos + neg),
                nli_qualification=float(
                    indexed.loc[f"{material}:qualification", "p_entailment"]
                ),
                grain_qualification=float(clipped[i, 1]),
                grain_pole=float(clipped[i, 0]),
            )
        )
    pd.DataFrame(rows).to_csv(output / "reader_scores.csv", index=False)
    save(
        output / "reading_summary.json",
        {
            "actual_NLI_pairs": len(readings),
            "actual_GRAIN_texts": len(features),
            "elapsed_seconds": time.monotonic() - began,
            "reader_scores_sha256": sha256_file(output / "reader_scores.csv"),
        },
    )
    print(
        json.dumps({"stage": "actual_text_reading_complete", "pairs": len(readings)}),
        flush=True,
    )


def calibrate(output):
    """Freeze source/pilot channels and validation diagnostics without target truth."""
    rows = pd.read_csv(output / "reader_scores.csv").to_dict("records")
    source = [r for r in rows if r["role"] == "source"]
    pilot = [r for r in rows if r["role"] == "pilot" and r["pilot_partition"] == "fit"]
    app = [r for r in rows if r["role"] == "application"]
    laws = {
        "old": transport.calibrated_channel(
            source, "grain_qualification", required_role="source"
        ),
        "pilot": transport.calibrated_channel(
            pilot, "nli_qualification", required_role="pilot", partition="fit"
        ),
        "known_pool": transport.calibrated_channel(
            app, "nli_qualification", required_role="application"
        ),
    }
    controls = {}
    for q in (0, 1):
        controls[f"missing_q{q}"] = transport.calibrated_channel(
            [r for r in pilot if r["qualification"] != q],
            "nli_qualification",
            required_role="pilot",
            partition="fit",
        )
    for b in (0, 1):
        controls[f"missing_pole{b}"] = transport.calibrated_channel(
            [r for r in pilot if r["pole"] != b],
            "nli_qualification",
            required_role="pilot",
            partition="fit",
        )
    save(output / "channels.json", laws)
    save(output / "missing_anchor_controls.json", controls)
    save(
        output / "channel_diagnostics.json",
        {k: transport.channel_diagnostics(v) for k, v in laws.items()},
    )
    measures = []
    for role, rs in (
        ("validation", [r for r in rows if r["pilot_partition"] == "validation"]),
        ("application_descriptive", app),
    ):
        for reader, qkey in (
            ("old", "grain_qualification"),
            ("new", "nli_qualification"),
        ):
            for target, key in (("pole", "pole_readout"), ("qualification", qkey)):
                p = np.array([r[key] for r in rs])
                y = np.array([r[target] for r in rs])
                measures.append(
                    {
                        "partition": role,
                        "reader": reader,
                        "target": target,
                        "n": len(rs),
                        "brier": float(np.mean((p - y) ** 2)),
                        "accuracy": float(np.mean((p >= 0.5) == y)),
                    }
                )
        for law in ("old", "pilot"):
            e = transport.channels(laws[law], 0, 1, "plugin")[0]
            probabilities = [
                e[
                    r["pole"],
                    r["qualification"],
                    static.category(r["pole_readout"], r["nli_qualification"]),
                ]
                for r in rs
            ]
            measures.append(
                {
                    "partition": role,
                    "reader": f"new_with_{law}_law",
                    "target": "joint_category",
                    "n": len(rs),
                    "log_loss": float(-np.mean(np.log(probabilities))),
                }
            )
    pd.DataFrame(measures).to_csv(output / "reader_diagnostics.csv", index=False)


def render_panel(groups, bank, seed, dynamic_target):
    """Generate text addresses from observations only; no latent author truth."""
    pools = {
        (b, q): [
            r
            for r in bank
            if r["role"] == "application" and r["pole"] == b and r["qualification"] == q
        ]
        for b in (0, 1)
        for q in (0, 1)
    }
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 169, int(dynamic_target)])
    )
    target_time = 47 if dynamic_target else 31
    contract = (
        dynamic.CONTRACT if dynamic_target else "F04_L_thresholded_text_eligibility_v1"
    )
    inputs, references = [], []
    for author in sorted(groups):
        history, oracle = [], []
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
            oracle.append({"t": r["t"], "z": r["z"], "category": 2 * b + q})
        inputs.append(
            {
                "author_id": author,
                "target_time": target_time,
                "world_contract": contract,
                "history": history,
            }
        )
        references.append(oracle)
    return inputs, references


def scored_arm(inputs, labels, base, laws, bank, seed, arm, dynamic_target, count):
    """Construct compatible observed inputs and fixed channel for exactly one arm."""
    reader = "old" if arm == "old_matched" else "new"
    mode = "plugin" if arm == "changed_pilot_plugin" else "mixture"
    law_key = "old" if arm in ("old_matched", "changed_uncalibrated") else "pilot"
    if arm == "changed_known_pool":
        law_key, mode = "known_pool", "known_pool"
    channel = transport.channels(
        laws[law_key],
        np.random.SeedSequence([seed, 170, 0 if law_key == "old" else 1]),
        count,
        mode,
    )
    cache = transport.observation_cache(bank, reader)
    if arm == "oracle_labels":
        channel = np.repeat(static.oracle_channel()[None], count, axis=0)
        lookup = {r["material_id"]: r for r in bank}
        # Explicit information reference, excluded from operational calibration.
        cache = {
            k: dict(
                v,
                pole_readout=float(lookup[k]["pole"]),
                grain_qualification=float(lookup[k]["qualification"]),
                nli_qualification=float(lookup[k]["qualification"]),
            )
            for k, v in cache.items()
        }
    bundle = transport.bundle_for(base, cache, channel)
    return transport.score_panel(inputs, bundle, dynamic_target), bundle


def run_seed(arguments):
    """Score all fixed arms, saving each finished arm without opening target truth."""
    output, seed, smoke = arguments
    output = Path(output)
    began = time.monotonic()
    bank = pd.read_csv(output / "reader_scores.csv").to_dict("records")
    laws = json.loads((output / "channels.json").read_text())
    folder = output / ("smoke_scores" if smoke else "scores") / str(seed)
    folder.mkdir(parents=True, exist_ok=False)
    count = 3 if smoke else 200
    checks = []
    for dynamic_target, name in ((False, "T1"), (True, "dynamic")):
        bundle_path = BASE / (
            f"F16/dynamic_text_full_r1/{seed}/bundle.json"
            if dynamic_target
            else f"F16/t1_text_full_r1/{seed}/bundle.json"
        )
        base = json.loads(bundle_path.read_text())
        base["source_models"] = base["source_models"][:count]
        base["source_components"] = count
        if smoke:
            base["qmc_power"] = 4
        archive = BASE / (
            f"F05/source_full_r2/core_{seed}"
            if dynamic_target
            else f"F04/source_full_r1/L_{seed}"
        )
        groups = group(archive / "application.jsonl", "application")
        if smoke:
            groups = {a: groups[a] for a in sorted(groups)[:3]}
        if not smoke and len(groups) != 200:
            raise ValueError("full application panel missing")
        inputs, labels = render_panel(groups, bank, seed, dynamic_target)
        write_inputs(folder / f"{name}_inputs.jsonl", inputs)
        save(folder / f"{name}_oracle_rendering.json", labels)
        for arm in transport.ARMS:
            records, bundle = scored_arm(
                inputs, labels, base, laws, bank, seed, arm, dynamic_target, count
            )
            save(folder / f"{name}_{arm}.json", records)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "target": name,
                        "arm": arm,
                        "elapsed": time.monotonic() - began,
                        "smoke": smoke,
                    }
                ),
                flush=True,
            )
            if arm == "changed_pilot_mixture":
                save(folder / f"{name}_primary_bundle.json", bundle)
                # Fixed IDs, not selected by scores or truth. Numerical diagnostics only.
                subset = [inputs[0], inputs[-1]]
                fine = copy.deepcopy(bundle)
                fine["qmc_power"] += 1
                more = transport.score_panel(subset, fine, dynamic_target)
                alternate = copy.deepcopy(bundle)
                second = transport.channels(
                    laws["pilot"], np.random.SeedSequence([seed, 170, 99]), count
                )
                alternate["channel_draws"] = {
                    k: second.tolist() for k in ("hybrid_joint", "nli_joint")
                }
                another = transport.score_panel(subset, alternate, dynamic_target)
                checks.append(
                    {
                        "target": name,
                        "indices": [0, len(inputs) - 1],
                        "base": [records[0], records[-1]],
                        "double_nodes": more,
                        "alternate_channel_draws": another,
                    }
                )
    save(folder / "diagnostics.json", checks)
    save(
        folder / "complete.json",
        {
            "status": "SCORED_WITHOUT_TARGET_TRUTH",
            "elapsed_seconds": time.monotonic() - began,
            "source_components": count,
            "smoke": smoke,
        },
    )
    return seed


def evaluate(output):
    """Read latent evaluation files only after all seeds and arms have completed."""
    flat = []
    for seed in SEEDS:
        folder = output / "scores" / str(seed)
        if not (folder / "complete.json").exists():
            raise ValueError("all scoring must finish before evaluation")
    for seed in SEEDS:
        folder = output / "scores" / str(seed)
        for is_dynamic, name in ((False, "T1"), (True, "dynamic")):
            archive = BASE / (
                f"F05/source_full_r2/core_{seed}"
                if is_dynamic
                else f"F04/source_full_r1/L_{seed}"
            )
            truth = {
                r["author_id"]: r
                for r in json.loads((archive / "truth.json").read_text())
                if r["role"] == "application"
            }
            groups = group(archive / "application.jsonl", "application")
            inputs = [
                json.loads(x)
                for x in (folder / f"{name}_inputs.jsonl").read_text().splitlines()
            ]
            for arm in transport.ARMS:
                results = json.loads((folder / f"{name}_{arm}.json").read_text())
                for raw, result in zip(inputs, results, strict=True):
                    a = raw["author_id"]
                    y = truth[a]
                    if is_dynamic:
                        future = groups[a][48:52]
                        if [r["t"] for r in future] != [48, 49, 50, 51]:
                            raise ValueError("future clock mismatch")
                        values = {
                            "T2": y["states"][47],
                            "T3": float(
                                np.mean(
                                    [
                                        r["y"] + y["b"] * (z - r["z"])
                                        for z, r in zip(
                                            dynamic.PLAN, future, strict=True
                                        )
                                    ]
                                )
                            ),
                        }
                    else:
                        values = {"T1": y["means"][-1] - y["means"][0]}
                    for target, value in values.items():
                        score = result["targets"][target] if is_dynamic else result
                        flat.append(
                            {
                                "seed": seed,
                                "author_id": a,
                                "arm": arm,
                                "target": target,
                                "truth": value,
                                **{
                                    k: score[k]
                                    for k in ("score", "variance", "lower", "upper")
                                },
                            }
                        )
    frame = pd.DataFrame(flat)
    frame.to_csv(output / "predictions.csv", index=False)
    metrics = []
    for (seed, target, arm), g in frame.groupby(["seed", "target", "arm"]):
        p, y = g.score.to_numpy(), g.truth.to_numpy()
        metrics.append(
            {
                "seed": seed,
                "target": target,
                "arm": arm,
                "n": len(g),
                "mse": float(np.mean((p - y) ** 2)),
                "bias": float(np.mean(p - y)),
                "r": float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-12 else None,
                "coverage": float(np.mean((g.lower <= g.truth) & (g.truth <= g.upper))),
                "width": float(np.mean(g.upper - g.lower)),
            }
        )
    by_seed = pd.DataFrame(metrics)
    by_seed.to_csv(output / "metrics_by_seed.csv", index=False)
    by_seed.groupby(["target", "arm"])[
        ["mse", "bias", "r", "coverage", "width"]
    ].mean().to_csv(output / "metrics.csv")
    contrasts = []
    for target in ("T1", "T2", "T3"):
        subset = frame[frame.target == target].copy()
        subset["se"] = (subset.score - subset.truth) ** 2
        wide = subset.pivot(index=["seed", "author_id"], columns="arm", values="se")
        for comparison in (
            "changed_uncalibrated",
            "changed_pilot_plugin",
            "changed_known_pool",
            "old_matched",
            "oracle_labels",
        ):
            delta = wide.changed_pilot_mixture - wide[comparison]
            rng = np.random.default_rng(171)
            rep = np.zeros(2000)
            for seed in SEEDS:
                d = delta.loc[seed].to_numpy()
                rep += d[rng.integers(len(d), size=(2000, len(d)))].mean(1) / len(SEEDS)
            contrasts.append(
                {
                    "target": target,
                    "contrast": f"changed_pilot_mixture-minus-{comparison}",
                    "delta_mse": float(delta.mean()),
                    "lower": float(np.quantile(rep, 0.025)),
                    "upper": float(np.quantile(rep, 0.975)),
                    "scope": "stratified author bootstrap,conditional on reusedSIMauthors,fixed materialpool andfits",
                }
            )
    save(output / "paired_contrasts.json", contrasts)
    save(
        output / "summary.json",
        {
            "status": "ACTUAL_TEXT_READER_TRANSPORT_EXECUTED_AUDIT_PENDING",
            "prediction_rows": len(frame),
            "source_components": 200,
            "channel_draws": 200,
            "combination": "independent paired cutMC,notCartesian",
            "T1_authors": 600,
            "shared_T2_T3_authors": 600,
            "materials": 380,
            "neural_pairs": 1140,
            "public_serialization": "NEXT_PACKET",
            "goal_complete": False,
        },
    )
    hashes = {
        str(p.relative_to(output)): sha256_file(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "output_manifest.json"
    }
    save(output / "output_manifest.json", hashes)


def main():
    """Run only requested phase, keeping neural reading and truth evaluation apart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "read", "calibrate", "smoke", "score", "evaluate", "cold"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()
    if args.stage == "cold":
        bundle = json.loads(args.bundle.read_text())
        inputs = [json.loads(x) for x in args.input.read_text().splitlines()]
        save(args.output, transport.score_panel(inputs, bundle, args.dynamic))
        return
    if args.stage == "prepare":
        prepare(args.output)
        return
    verify(args.output)
    if args.stage == "read":
        read_materials(args.output)
    elif args.stage == "calibrate":
        calibrate(args.output)
    elif args.stage == "smoke":
        run_seed((args.output, SEEDS[0], True))
    elif args.stage == "score":
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(run_seed, [(args.output, s, False) for s in SEEDS]))
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(args.output.resolve()),
                "--stage",
                "evaluate",
            ],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            check=True,
        )
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
