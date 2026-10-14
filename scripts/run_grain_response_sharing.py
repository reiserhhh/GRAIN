#!/usr/bin/env python3
"""Bounded source-response sharing experiment with sealed observation stages."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.response_sharing import (
    apply_function,
    compile_function,
    learn_response,
    source_neighbors,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_joint_conditional_observation import dump, read, sha

DEFAULT = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1/response_sharing_r1"
CODE = [
    "configs/grain_response_sharing.json",
    "docs/GRAIN_RESPONSE_SHARING_PLAN_V1.md",
    "grain_research/response_sharing.py",
    "scripts/run_grain_response_sharing.py",
    "scripts/run_grain_evidence_calibration.py",
    "scripts/run_grain_entailment_reference.py",
    "scripts/run_grain_joint_conditional_observation.py",
    "grain_research/vector_mechanisms/stage_file_guard.py",
    "configs/grain_joint_conditional_observation.json",
]
TARGETS = {
    "level": [0.2] * 5,
    "finite_change": [0, -1, 0, 1, 0],
    "new_condition": [0, 0, 0, 1, 0],
}


def begin(out, stage):
    """Verify sealed code and create a new stage exactly once."""
    for p, digest in read(out / "freeze.json").items():
        assert sha(ROOT / p) == digest, p
    folder = out / stage
    folder.mkdir(exist_ok=False)
    return read(out / "config.json"), folder


def prepare(out):
    """Publish fixed condition/phrase bank before observing model responses."""
    cfg = read(ROOT / CODE[0])
    previous = read(ROOT / CODE[-1])
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "freeze.json", {p: sha(ROOT / p) for p in CODE})
    dump(out / "config.json", cfg)
    rows = []
    for j, z in enumerate(cfg["conditions"]):
        for sign in [-1, 1]:
            for v in range(6):
                body = previous["worry_high" if sign == 1 else "worry_low"][v]
                rows.append(
                    {
                        "case_id": len(rows),
                        "condition_index": j,
                        "condition": z,
                        "sign": sign,
                        "variant": v,
                        "sentence": f"I was dealing with a routine request at work. I had {int(25 + 20 * z)} minutes available for the work. "
                        + body,
                    }
                )
    bank = pd.DataFrame(rows)
    bank[bank.variant.isin(cfg["source_variants"])].to_csv(
        out / "source_bank.csv", index=False
    )
    bank[bank.variant.isin(cfg["held_variants"])].to_csv(
        out / "held_bank.csv", index=False
    )
    dump(
        out / "bank_freeze.json",
        {name: sha(out / name) for name in ["source_bank.csv", "held_bank.csv"]},
    )
    print("60 public texts frozen:40 source,20 held", flush=True)


def cold(out, source):
    """Actually run pinned NLI on new texts; source calibration precedes held calls."""
    from scripts.run_grain_evidence_calibration import MODEL, REVISION, infer_pairs

    cfg, folder = begin(out, "cold_source" if source else "cold_held")
    if not source:
        assert (
            sha(out / "calibrate/meter.json")
            == read(out / "calibrate/seal.json")["meter"]
        )
    name = "source_bank.csv" if source else "held_bank.csv"
    assert sha(out / name) == read(out / "bank_freeze.json")[name]
    bank = pd.read_csv(out / name)
    requests = pd.DataFrame(
        [
            {
                "audit_id": f"{r.case_id}_{q}",
                "case_id": r.case_id,
                "polarity": q,
                "sentence": r.sentence,
                "item_text": hypothesis,
            }
            for r in bank.itertuples()
            for q, hypothesis in enumerate(cfg["hypotheses"])
        ]
    )
    begun = time.monotonic()
    result = requests.merge(infer_pairs(requests), on="audit_id", validate="one_to_one")
    elapsed = time.monotonic() - begun
    result.to_csv(folder / "nli_pairs.csv", index=False)
    wide = result.pivot(index="case_id", columns="polarity", values="p_entailment")
    bank["raw_value"] = bank.case_id.map(wide[0] - wide[1])
    bank.to_csv(folder / "readings.csv", index=False)
    dump(
        folder / "manifest.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "actual_pairs": len(result),
            "seconds": elapsed,
            "within_budget": elapsed <= cfg["max_neural_seconds"],
        },
    )
    dump(
        folder / "seal.json",
        {
            "readings.csv": sha(folder / "readings.csv"),
            "nli_pairs.csv": sha(folder / "nli_pairs.csv"),
        },
    )
    print(f"{len(result)} actual NLI pairs in {elapsed:.3f}s", flush=True)


def calibrate(out):
    """Restore signed event units with source anchors only, by recorded condition."""
    _, folder = begin(out, "calibrate")
    source = out / "cold_source/readings.csv"
    denied = install(
        [source],
        [folder / "meter.json", folder / "guard.json", folder / "seal.json"],
        [out / "cold_held", out / "held_bank.csv"],
        [ROOT],
    )
    f, meter = pd.read_csv(source), {}
    for j, group in f.groupby("condition_index"):
        means = group.groupby("sign").raw_value.mean()
        offset, scale = (means[1] + means[-1]) / 2, (means[1] - means[-1]) / 2
        if not np.isfinite(scale) or abs(scale) < 1e-6:
            raise ValueError("Source sign contrast does not define a usable unit")
        meter[str(j)] = {"offset": float(offset), "scale": float(scale)}
    dump(folder / "meter.json", meter)
    dump(folder / "guard.json", {"denied": denied})
    dump(folder / "seal.json", {"meter": sha(folder / "meter.json")})
    print("5 source units sealed before held inference", flush=True)


def make_functions(cfg, n, world, rng):
    """Declared mechanism worlds with legal event probabilities and local variation."""
    a = rng.uniform(-cfg["level_halfwidth"], cfg["level_halfwidth"], size=(n, 1))
    b = rng.uniform(-cfg["response_halfwidth"], cfg["response_halfwidth"], size=(n, 1))
    local = rng.uniform(-1, 1, size=(n, 5))
    if world == "independent_response":
        f = a + cfg["independent_halfwidth"] * local
    else:
        pattern = (
            cfg["smooth_pattern"]
            if world == "smooth_shared"
            else cfg["crossed_pattern"]
        )
        f = a + b * np.array(pattern) + cfg["local_halfwidth"] * local
    assert np.max(np.abs(f)) < 1
    return f


def read_simulated_events(functions, lookup, variants, events, halves, rng):
    """Map each generated event to an actual frozen text reading, without label input to fitting."""
    shape = (len(functions), 5, halves, events)
    signs = (rng.random(shape) < (1 + functions[:, :, None, None]) / 2).astype(int)
    choices = rng.integers(len(variants), size=shape)
    j = np.arange(5)[None, :, None, None]
    v = np.array(variants)[choices]
    values = lookup[j, signs, v]
    indices = j * 12 + signs * 6 + v
    return values.mean(-1), indices


def generate(out):
    """Seal same-event phrase panels and isolated observed-condition files."""
    cfg, folder = begin(out, "generate")
    meter = read(out / "calibrate/meter.json")
    frames = []
    for phase in ["cold_source", "cold_held"]:
        for name, digest in read(out / phase / "seal.json").items():
            assert sha(out / phase / name) == digest
        frames.append(pd.read_csv(out / phase / "readings.csv"))
    bank = pd.concat(frames).sort_values("case_id")
    bank["value"] = [
        (row.raw_value - meter[str(row.condition_index)]["offset"])
        / meter[str(row.condition_index)]["scale"]
        for row in bank.itertuples()
    ]
    bank.to_csv(folder / "bank_values.csv", index=False)
    lookup = bank.value.to_numpy().reshape(5, 2, 6)
    sources, truth = {}, {}
    applications = {stage: {} for stage in cfg["stages"]}
    for seed in cfg["seeds"]:
        for world in cfg["worlds"]:
            key = f"s{seed}_{world}"
            f_source = make_functions(
                cfg, cfg["source_authors"], world, np.random.default_rng(seed)
            )
            f_app = make_functions(
                cfg,
                cfg["application_authors"],
                world,
                np.random.default_rng(seed + 100000),
            )
            y, ids = read_simulated_events(
                f_source,
                lookup,
                cfg["source_variants"],
                cfg["events_per_half"],
                2,
                np.random.default_rng(seed + 200000),
            )
            sources[key + "_halves"], sources[key + "_cases"] = y, ids
            truth[key + "_source"], truth[key + "_application"] = f_source, f_app
            for style, variants in [
                ("source_style", cfg["source_variants"]),
                ("held_style", cfg["held_variants"]),
            ]:
                y, ids = read_simulated_events(
                    f_app,
                    lookup,
                    variants,
                    cfg["events_per_half"],
                    1,
                    np.random.default_rng(seed + 300000),
                )
                for stage, obs in cfg["stages"].items():
                    applications[stage][key + "_" + style] = y[:, obs, 0]
                    applications[stage][key + "_" + style + "_cases"] = ids[:, obs, 0]
    np.savez_compressed(folder / "source.npz", **sources)
    np.savez_compressed(folder / "truth.npz", **truth)
    for stage, arrays in applications.items():
        np.savez_compressed(folder / f"application_{stage}.npz", **arrays)
    dump(folder / "seal.json", {p.name: sha(p) for p in folder.glob("*.npz")})
    print(
        "9 paired worlds; source,application stages,truth sealed separately", flush=True
    )


def fit(out):
    """Learn all permitted source structures once, with truth and app reads denied."""
    cfg, folder = begin(out, "fit")
    datafile = out / "generate/source.npz"
    denied = install(
        [datafile],
        [folder / n for n in ["models.json", "guard.json", "seal.json"]],
        [out / "generate/truth.npz"]
        + [out / f"generate/application_{s}.npz" for s in cfg["stages"]],
        [ROOT],
    )
    models = {}
    with np.load(datafile, allow_pickle=False) as data:
        for seed in cfg["seeds"]:
            for world in cfg["worlds"]:
                key = f"s{seed}_{world}"
                y = data[key + "_halves"]
                models[key] = learn_response(y, cfg["conditions"], cfg["noise_floor"])
                permutation = np.random.default_rng(seed + 400000).permutation(len(y))
                models[key + "_unpaired"] = learn_response(
                    y, cfg["conditions"], cfg["noise_floor"], permutation
                )
    dump(folder / "models.json", models)
    dump(folder / "guard.json", {"denied": denied})
    dump(folder / "seal.json", {"models": sha(folder / "models.json")})
    print("18 paired/unpaired source models sealed", flush=True)


def apply(out, stage):
    """Apply frozen functions without access to that person's hidden conditions."""
    cfg, folder = begin(out, "apply_" + stage)
    model_file, input_file = (
        out / "fit/models.json",
        out / f"generate/application_{stage}.npz",
    )
    assert sha(model_file) == read(out / "fit/seal.json")["models"]
    assert sha(input_file) == read(out / "generate/seal.json")[input_file.name]
    blocked = [out / "generate/truth.npz"] + [
        out / f"generate/application_{s}.npz" for s in cfg["stages"] if s != stage
    ]
    denied = install(
        [model_file, input_file],
        [
            folder / n
            for n in ["programs.json", "predictions.npz", "guard.json", "seal.json"]
        ],
        blocked,
        [ROOT],
    )
    models, programs, predictions = read(model_file), {}, {}
    with np.load(input_file, allow_pickle=False) as data:
        for seed in cfg["seeds"]:
            for world in cfg["worlds"]:
                key = f"s{seed}_{world}"
                for arm in cfg["arms"]:
                    model = (
                        models[key + "_unpaired"]
                        if arm == "unpaired_source"
                        else models[key]
                    )
                    p = compile_function(model, cfg["stages"][stage], arm)
                    programs[key + "_" + arm] = p
                    for style in ["source_style", "held_style"]:
                        scored = apply_function(p, data[key + "_" + style])
                        for name, values in scored.items():
                            predictions[f"{key}_{arm}_{style}_{name}"] = values
    dump(folder / "programs.json", programs)
    np.savez_compressed(folder / "predictions.npz", **predictions)
    dump(folder / "guard.json", {"denied": denied})
    dump(
        folder / "seal.json",
        {n: sha(folder / n) for n in ["programs.json", "predictions.npz"]},
    )
    print(
        f"{stage}:45 functions,observed{cfg['stages'][stage]},hidden input denied",
        flush=True,
    )


def evaluate(out):
    """Open truth after all three applications, retaining every seed/world/arm."""
    cfg, folder = begin(out, "evaluate")
    rows, neighbors, kernels = [], [], []
    models = read(out / "fit/models.json")
    for stage in cfg["stages"]:
        for n, digest in read(out / f"apply_{stage}/seal.json").items():
            assert sha(out / f"apply_{stage}/{n}") == digest
    with np.load(out / "generate/truth.npz", allow_pickle=False) as truth:
        for stage in cfg["stages"]:
            programs = read(out / f"apply_{stage}/programs.json")
            with np.load(
                out / f"apply_{stage}/predictions.npz", allow_pickle=False
            ) as predictions:
                for seed in cfg["seeds"]:
                    for world in cfg["worlds"]:
                        key = f"s{seed}_{world}"
                        actual = truth[key + "_application"]
                        for arm in cfg["arms"]:
                            p = programs[key + "_" + arm]
                            for style in ["source_style", "held_style"]:
                                estimate = predictions[f"{key}_{arm}_{style}_value"]
                                for target, direction in TARGETS.items():
                                    c = np.array(direction)
                                    t, pred = actual @ c, estimate @ c
                                    error = pred - t
                                    variance = float(c @ p["covariance"] @ c)
                                    r = (
                                        float(np.corrcoef(pred, t)[0, 1])
                                        if np.std(pred) > 1e-12
                                        else np.nan
                                    )
                                    rows.append(
                                        {
                                            "seed": seed,
                                            "world": world,
                                            "stage": stage,
                                            "arm": arm,
                                            "style": style,
                                            "target": target,
                                            "mse": np.mean(error**2),
                                            "bias": error.mean(),
                                            "correlation": r,
                                            "working_variance": variance,
                                            "coverage": np.mean(
                                                np.abs(error)
                                                <= 1.959964 * np.sqrt(max(0, variance))
                                            ),
                                            "personal_response_rms": np.sqrt(
                                                np.mean(
                                                    predictions[
                                                        f"{key}_{arm}_{style}_personal_response"
                                                    ]
                                                    ** 2
                                                )
                                            ),
                                        }
                                    )
        with (
            np.load(out / "generate/source.npz", allow_pickle=False) as source,
            np.load(
                out / "generate/application_response.npz", allow_pickle=False
            ) as app,
        ):
            for seed in cfg["seeds"]:
                for world in cfg["worlds"]:
                    key = f"s{seed}_{world}"
                    observed = cfg["stages"]["response"]
                    reference = source[key + "_halves"].mean(-1)[:, observed]
                    for style in ["source_style", "held_style"]:
                        for rule in ["level", "response", "random"]:
                            probe = app[key + "_" + style]
                            ref = reference
                            if rule == "level":
                                center = observed.index(cfg["stages"]["center"][0])
                                ref, probe = ref[:, [center]], probe[:, [center]]
                            ids = source_neighbors(
                                ref, probe, rule, np.random.default_rng(seed + 500000)
                            )
                            c = np.array(TARGETS["finite_change"])
                            d = (
                                truth[key + "_source"][ids]
                                - truth[key + "_application"]
                            ) @ c
                            neighbors.append(
                                {
                                    "seed": seed,
                                    "world": world,
                                    "style": style,
                                    "rule": rule,
                                    "hidden_response_difference_mse": np.mean(d**2),
                                }
                            )
                    k = np.array(models[key]["response_kernel"])
                    for i in range(5):
                        for j in range(5):
                            kernels.append(
                                {
                                    "seed": seed,
                                    "world": world,
                                    "i": i,
                                    "j": j,
                                    "response_covariance": k[i, j],
                                    "response_distance_sq": k[i, i]
                                    + k[j, j]
                                    - 2 * k[i, j],
                                }
                            )
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "metrics_by_seed.csv", index=False)
    frame.groupby(["world", "stage", "arm", "style", "target"], dropna=False)[
        ["mse", "correlation", "working_variance", "coverage", "personal_response_rms"]
    ].mean().reset_index().to_csv(folder / "comparison.csv", index=False)
    pd.DataFrame(neighbors).to_csv(folder / "neighbor_comparison.csv", index=False)
    pd.DataFrame(kernels).to_csv(folder / "response_kernels.csv", index=False)
    print(
        "810 metrics;54 matching diagnostics;all conditions/worlds retained", flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "prepare",
            "cold_source",
            "calibrate",
            "cold_held",
            "generate",
            "fit",
            "apply_center",
            "apply_response",
            "apply_added",
            "evaluate",
        ],
    )
    p.add_argument("--output", type=Path, default=DEFAULT)
    args = p.parse_args()
    out = args.output.resolve()
    if args.stage.startswith("cold_"):
        cold(out, args.stage == "cold_source")
    elif args.stage.startswith("apply_"):
        apply(out, args.stage.removeprefix("apply_"))
    else:
        globals()[args.stage](out)
