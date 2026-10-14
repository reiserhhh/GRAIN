#!/usr/bin/env python3
"""Source-calibrated GRAIN/query observations into one conditional person function."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.joint_conditional_observation import (
    build_meter,
    calibrate,
    conditional_function,
    decoder,
    factorial,
    joint_probabilities,
    predict_function,
    read_events,
)
from grain_research.vector_mechanisms.stage_file_guard import install

DEFAULT = (
    ROOT
    / "results/grain_vector_mechanisms/mechanism_todo_v1/joint_conditional_observation_r1"
)
PREVIOUS = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1/query_moment_r1"
CODE = [
    "configs/grain_joint_conditional_observation.json",
    "docs/GRAIN_JOINT_CONDITIONAL_OBSERVATION_PLAN_V1.md",
    "grain_research/joint_conditional_observation.py",
    "grain_research/query_moment_operator.py",
    "scripts/run_grain_joint_conditional_observation.py",
    "scripts/score_grain_query_moments.py",
    "scripts/run_grain_query_moment_operator.py",
    "scripts/run_grain_evidence_calibration.py",
    "grain_research/vector_mechanisms/prefix.py",
]


def read(path):
    """Read declared JSON artifacts."""
    return json.loads(Path(path).read_text())


def dump(path, value):
    """Write finite metadata and numeric arrays without pickles."""

    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)

    Path(path).write_text(
        json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n"
    )


def sha(path):
    """Stream hashes without deserializing source artifacts."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def begin(out, stage):
    """Prevent source drift or accidental stage overwrite."""
    for path, digest in read(out / "freeze.json").items():
        if sha(ROOT / path) != digest:
            raise ValueError("Frozen input changed: " + path)
    folder = out / stage
    folder.mkdir(exist_ok=False)
    return read(out / "config.json"), folder


def prepare(out):
    """Freeze all public texts and old source operators before new inference."""
    from scripts.run_grain_native_material_reader import BUNDLE, REPRESENTATION
    from scripts.run_grain_real_author_path import MODEL

    cfg = read(ROOT / CODE[0])
    out.mkdir(parents=True, exist_ok=False)
    paths = [ROOT / p for p in CODE] + [
        BUNDLE,
        REPRESENTATION,
        MODEL / "config.json",
        PREVIOUS / "models/freeze.json",
    ]
    paths.extend(
        PREVIOUS / "models" / n for n in read(PREVIOUS / "models/freeze.json")["files"]
    )
    dump(out / "freeze.json", {str(p.relative_to(ROOT)): sha(p) for p in paths})
    dump(out / "config.json", cfg)
    rows = []
    for z in cfg["conditions"]:
        for s in [-1, 1]:
            for t in [-1, 1]:
                for v in range(6):
                    text = (
                        f"I was dealing with a routine request at work. I had {int(25 + 20 * z)} minutes available for the work. "
                        + cfg["worry_high" if s == 1 else "worry_low"][v]
                        + " "
                        + cfg["confirm_high" if t == 1 else "confirm_low"][v]
                        + " "
                        + "After that part was over, I put the papers back in their folder and noted which practical steps I had completed."
                    )
                    source = (
                        z in cfg["source_conditions"] and v in cfg["source_variants"]
                    )
                    rows.append(
                        {
                            "case_id": f"p{len(rows):03d}",
                            "condition": z,
                            "worry": s,
                            "reassurance": t,
                            "variant": v,
                            "source": source,
                            "text": text,
                        }
                    )
    bank = pd.DataFrame(rows)
    assert len(bank) == cfg["prefix_budget"] and bank.source.sum() == 32
    bank.to_csv(out / "public_bank.csv", index=False)
    dump(out / "bank_freeze.json", {"sha256": sha(out / "public_bank.csv")})
    print("prepare: 32 source / 40 new texts frozen", flush=True)


def cold(out, source, device):
    """Raw text to fixed GRAIN/query coordinates then NLI; no teacher target reads."""
    import gc

    import joblib
    import torch

    from grain_research.query_moment_operator import apply_operator, observation_moments
    from grain_research.vector_mechanisms.prefix import capture_prefix
    from grain_research.vector_mechanisms.trace import DecoderObserver
    from scripts.run_grain_entailment_reference import MODEL as NLI_MODEL
    from scripts.run_grain_entailment_reference import REVISION
    from scripts.run_grain_evidence_calibration import infer_pairs
    from scripts.run_grain_native_material_reader import BUNDLE, REPRESENTATION
    from scripts.run_grain_query_moment_operator import load_models
    from scripts.run_grain_real_author_path import MODEL
    from scripts.score_grain_query_moments import raw_coordinates
    from suica_core.v7_psychometric import read_factor_bundle

    cfg, folder = begin(out, "cold_source" if source else "cold_application")
    assert sha(out / "public_bank.csv") == read(out / "bank_freeze.json")["sha256"]
    if not source:
        assert (
            sha(out / "fit/meters.json") == read(out / "fit/seal.json")["meters_sha256"]
        )
    bank = pd.read_csv(out / "public_bank.csv")
    items = bank[bank.source == source]
    models = load_models(PREVIOUS)
    representation, bundle = joblib.load(REPRESENTATION), read_factor_bundle(BUNDLE)
    allowed = [
        MODEL.resolve(),
        out.resolve(),
        (PREVIOUS / "models").resolve(),
        BUNDLE.parent.resolve(),
        REPRESENTATION.parent.resolve(),
    ]
    denied = []

    def guard(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to((ROOT / "results").resolve()) and not any(
            p.is_relative_to(a) for a in allowed
        ):
            denied.append(str(p))
            raise PermissionError(
                "Current reader cannot access historical private state/teacher cache"
            )

    sys.addaudithook(guard)
    for name in [
        "logic_complete_function_extraction_v1/private_jobs.json",
        "logic_fixed_seed_attention_v1/private_jobs.json",
    ]:
        try:
            (ROOT / "results/grain_vector_mechanisms" / name).read_bytes()
        except PermissionError:
            pass
        else:
            raise AssertionError("Teacher probe not rejected")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    started = time.monotonic()
    observer = DecoderObserver.load(
        str(MODEL), device=device, dtype="float32", max_tokens=1024
    )
    observer.model.requires_grad_(False)
    load_seconds = time.monotonic() - started
    started = time.monotonic()
    rows, traces = [], []
    for item in items.itertuples():
        ids, positions, g = raw_coordinates(
            [item.text], observer.tokenizer, representation, bundle
        )
        prefix, trace = capture_prefix(observer, ids)
        h = prefix["boundary"][positions].astype(float)
        values = list(g.mean(axis=0))
        quadratic, within = [], []
        for query in ["neuroticism", "confirmation_seeking"]:
            v, _ = apply_operator(g, h, models[query])
            km = observation_moments(g, h, models[query])["within_covariance"]
            quadratic.append(float(v[1]))
            within.append(float(np.trace(models[query]["within_matrix"] @ km)))
        values.extend(quadratic + within)
        rows.append(
            dict(case_id=item.case_id, **dict(zip(cfg["channels"][:8], values)))
        )
        np.savez_compressed(
            folder / (item.case_id + ".npz"), h=h, g=g, ids=ids, positions=positions
        )
        traces.append(dict(case_id=item.case_id, **trace))
        if time.monotonic() - started > cfg["max_cold_seconds"]:
            raise TimeoutError("Cold budget exceeded")
    calls = observer.calls
    prefix_seconds = time.monotonic() - started
    del observer, prefix, h
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    requests = []
    for item in items.itertuples():
        for query, hypotheses in enumerate(cfg["hypotheses"]):
            for pole, hypothesis in enumerate(hypotheses):
                requests.append(
                    {
                        "audit_id": f"{item.case_id}q{query}p{pole}",
                        "case_id": item.case_id,
                        "query": query,
                        "pole": pole,
                        "sentence": item.text,
                        "item_text": hypothesis,
                    }
                )
    requests = pd.DataFrame(requests)
    begun = time.monotonic()
    nli = requests.merge(infer_pairs(requests), on="audit_id", validate="one_to_one")
    nli.to_csv(folder / "nli_pairs.csv", index=False)
    nli_seconds = time.monotonic() - begun
    frame = pd.DataFrame(rows)
    for query, channel in enumerate(cfg["channels"][8:]):
        p = nli[nli["query"] == query].pivot(
            index="case_id", columns="pole", values="p_entailment"
        )
        frame[channel] = frame.case_id.map(p[0] - p[1])
    frame.to_csv(folder / "observations.csv", index=False)
    assert calls == len(items) and len(denied) == 2
    dump(
        folder / "runtime.json",
        {
            "status": "complete",
            "prefix_calls": calls,
            "nli_pairs": len(nli),
            "load_seconds": load_seconds,
            "prefix_seconds": prefix_seconds,
            "nli_seconds": nli_seconds,
            "nli_model": NLI_MODEL,
            "nli_revision": REVISION,
            "denied": denied,
            "traces": traces,
            "full_teacher_calls": 0,
            "new_neural_training": False,
            "observations_sha256": sha(folder / "observations.csv"),
        },
    )
    print(
        f"cold: {calls} actual prefix calls / {len(nli)} actual NLI pairs", flush=True
    )


def fit(out):
    """Calibrate from the 32 source texts, never using new condition/style outputs."""
    cfg, folder = begin(out, "fit")
    allowed = [out / "public_bank.csv", out / "cold_source/observations.csv"]
    outputs = [
        folder / name
        for name in [
            "calibration.json",
            "meters.json",
            "meter_diagnostics.csv",
            "guard.json",
            "seal.json",
        ]
    ]
    denied = install(
        allowed,
        outputs,
        [out / "cold_application/observations.csv", out / "generate/truth.npz"],
        [ROOT],
    )
    bank = pd.read_csv(allowed[0])
    bank = bank[bank.source]
    data = bank.merge(pd.read_csv(allowed[1]), on="case_id", validate="one_to_one")
    model = calibrate(
        data[cfg["channels"]],
        data.condition,
        data[["worry", "reassurance"]],
        cfg["covariance_shrinkage"],
    )
    meters, rows = {}, []
    for name, index in cfg["arms"].items():
        for diagonal in [False, True] if name == "joint_all" else [False]:
            arm = name + ("_diagonal" if diagonal else "")
            for z in cfg["conditions"]:
                d = build_meter(
                    model, z, index, diagonal=diagonal, tolerance=cfg["svd_tolerance"]
                )
                meters[f"{arm}__{z}"] = d
                s = d["singular_values"]
                rows.append(
                    {
                        "arm": arm,
                        "condition": z,
                        "rank": d["rank"],
                        "observation_rank": d["observation_rank"],
                        "largest_singular": float(s[0]),
                        "smallest_singular": float(s[-1]),
                        "resolution_error": float(
                            np.max(np.abs(d["resolution"] - np.eye(3)))
                        ),
                        "working_variance_trace": float(
                            np.trace(d["working_covariance"])
                        ),
                    }
                )
    dump(folder / "calibration.json", model)
    dump(folder / "meters.json", meters)
    pd.DataFrame(rows).to_csv(folder / "meter_diagnostics.csv", index=False)
    dump(
        folder / "guard.json",
        {"denied": denied, "source_texts": len(data), "new_author_labels_read": False},
    )
    dump(
        folder / "seal.json",
        {
            "meters_sha256": sha(folder / "meters.json"),
            "calibration_sha256": sha(folder / "calibration.json"),
        },
    )
    print("fit: meters frozen from source text only", flush=True)


def generate(out):
    """Create separate synthetic event identities and author truth for two styles."""
    cfg, folder = begin(out, "generate")
    bank = pd.read_csv(out / "public_bank.csv")
    lookup = {
        (r.condition, r.worry, r.reassurance, r.variant): int(r.case_id[1:])
        for r in bank.itertuples()
    }
    evidence, truth = {}, {}
    z = np.array(cfg["conditions"], float)
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    for seed in cfg["seeds"]:
        rng = np.random.default_rng(seed)
        a = rng.uniform(
            -np.array(cfg["level_halfwidths"]),
            cfg["level_halfwidths"],
            (cfg["authors"], 3),
        ) + [0.05, -0.03, 0.02]
        b = rng.uniform(
            -np.array(cfg["slope_halfwidths"]),
            cfg["slope_halfwidths"],
            (cfg["authors"], 3),
        )
        m = a[:, None, :] + z[None, :, None] * b[:, None, :]
        p = joint_probabilities(m)
        u = rng.random((cfg["authors"], 3, cfg["events_per_condition"]))
        c = (u[:, :, :, None] > np.cumsum(p, axis=2)[:, :, None, :]).sum(axis=3)
        assert c.max() < 4
        vp = rng.random(u.shape)
        truth[f"s{seed}_level"] = a
        truth[f"s{seed}_slope"] = b
        truth[f"s{seed}_condition_moments"] = m
        for style, variants in [
            ("source_style", cfg["source_variants"]),
            ("held_style", cfg["held_variants"]),
        ]:
            v = np.array(variants)[(vp * len(variants)).astype(int)]
            ids = np.empty_like(c)
            for author in range(cfg["authors"]):
                for zi, cond in enumerate(z):
                    for ei in range(cfg["events_per_condition"]):
                        ss, tt = corners[c[author, zi, ei]]
                        ids[author, zi, ei] = lookup[(cond, ss, tt, v[author, zi, ei])]
            evidence[f"s{seed}_{style}_case_indices"] = ids
    np.savez_compressed(folder / "evidence.npz", **evidence)
    np.savez_compressed(folder / "truth.npz", **truth)
    print("generate: same-author paired styles, hidden middle condition", flush=True)


def apply(out):
    """Freeze personal functions from endpoint events, not middle condition or truth."""
    cfg, folder = begin(out, "apply")
    mp = out / "fit/meters.json"
    assert sha(mp) == read(out / "fit/seal.json")["meters_sha256"]
    reads = [
        mp,
        out / "cold_source/observations.csv",
        out / "cold_application/observations.csv",
        out / "generate/evidence.npz",
    ]
    outputs = [folder / p for p in ["functions.npz", "guard.json", "seal.json"]]
    denied = install(reads, outputs, [out / "generate/truth.npz"], [ROOT])
    meters = read(mp)
    data = pd.concat([pd.read_csv(reads[1]), pd.read_csv(reads[2])]).sort_values(
        "case_id"
    )
    values = data[cfg["channels"]].to_numpy(float)
    arms = list(cfg["arms"]) + ["joint_all_diagonal"]
    result = {}
    with np.load(reads[3], allow_pickle=False) as ev:
        for source in ev.files:
            ids = ev[source]
            name = source.removesuffix("_case_indices")
            for arm in arms:
                events = np.stack(
                    [
                        read_events(values[ids[:, i]], meters[f"{arm}__{z}"])
                        for i, z in enumerate(cfg["conditions"])
                    ],
                    axis=1,
                )
                # Only endpoints enter all three restrictions of the personal function.
                for mode in ["joint", "level", "slope"]:
                    f = conditional_function(events[:, [0, 2]], mode=mode)
                    k = f"{name}__{arm}__{mode}__"
                    result[k + "level"] = f["level"]
                    result[k + "finite_change"] = 2 * f["slope"]
                    result[k + "new_condition"] = predict_function(f, 0.5)
                    if mode == "joint":
                        result[k + "level_material_variance"] = np.diagonal(
                            f["level_covariance"], axis1=1, axis2=2
                        )
                        result[k + "new_condition_material_variance"] = np.diagonal(
                            (
                                0.25 * f["material_mean_covariance"][:, 0]
                                + 2.25 * f["material_mean_covariance"][:, 1]
                            )
                            / 4,
                            axis1=1,
                            axis2=2,
                        )
                result[f"{name}__{arm}__held_direct_reading"] = events[:, 1].mean(
                    axis=1
                )
    np.savez_compressed(folder / "functions.npz", **result)
    dump(
        folder / "seal.json",
        {"functions_sha256": sha(folder / "functions.npz"), "meters_sha256": sha(mp)},
    )
    dump(
        folder / "guard.json",
        {
            "denied": denied,
            "truth_read": False,
            "endpoint_only_function_fit": True,
            "middle_reading_separate": True,
            "arrays": len(result),
        },
    )
    print("apply: individual level/change/new-condition functions frozen", flush=True)


def evaluate(out):
    """Inspect all fixed families and affine/duplicate behavior after score freeze."""
    cfg, folder = begin(out, "evaluate")
    seal = read(out / "apply/seal.json")
    assert sha(out / "apply/functions.npz") == seal["functions_sha256"]
    meters = read(out / "fit/meters.json")
    bank = pd.read_csv(out / "public_bank.csv")
    data = pd.concat(
        [
            pd.read_csv(out / "cold_source/observations.csv"),
            pd.read_csv(out / "cold_application/observations.csv"),
        ]
    )
    frame = bank.merge(data, on="case_id", validate="one_to_one").sort_values("case_id")
    values = frame[cfg["channels"]].to_numpy(float)
    outcomes, actions, transforms = [], [], []
    arms = list(cfg["arms"]) + ["joint_all_diagonal"]
    with (
        np.load(out / "apply/functions.npz", allow_pickle=False) as scores,
        np.load(out / "generate/truth.npz", allow_pickle=False) as truth,
    ):
        for seed in cfg["seeds"]:
            for style in ["source_style", "held_style"]:
                for arm in arms:
                    for mode in ["joint", "level", "slope"]:
                        base = f"s{seed}_{style}__{arm}__{mode}__"
                        for target in ["level", "finite_change", "new_condition"]:
                            actual = (
                                truth[f"s{seed}_level"]
                                if target == "level"
                                else 2 * truth[f"s{seed}_slope"]
                                if target == "finite_change"
                                else truth[f"s{seed}_condition_moments"][:, 1]
                            )
                            estimated = scores[base + target]
                            for j, moment in enumerate(cfg["targets"]):
                                err = estimated[:, j] - actual[:, j]
                                corr = (
                                    float(
                                        np.corrcoef(estimated[:, j], actual[:, j])[0, 1]
                                    )
                                    if np.std(estimated[:, j]) > 1e-12
                                    else None
                                )
                                outcomes.append(
                                    {
                                        "seed": seed,
                                        "style": style,
                                        "arm": arm,
                                        "mode": mode,
                                        "target": target,
                                        "moment": moment,
                                        "mse": float(np.mean(err**2)),
                                        "bias": float(err.mean()),
                                        "correlation": corr,
                                        "predicted_variance": float(
                                            np.var(estimated[:, j], ddof=1)
                                        ),
                                        "true_variance": float(
                                            np.var(actual[:, j], ddof=1)
                                        ),
                                    }
                                )
        for arm in arms:
            for z in cfg["conditions"]:
                m = meters[f"{arm}__{z}"]
                f = frame[frame.condition == z]
                y = read_events(f[cfg["channels"]].to_numpy(), m)
                actual = factorial(f[["worry", "reassurance"]])[:, 1:]
                for i, row in enumerate(f.itertuples()):
                    for j, target in enumerate(cfg["targets"]):
                        actions.append(
                            {
                                "case_id": row.case_id,
                                "condition": z,
                                "source": row.source,
                                "variant": row.variant,
                                "arm": arm,
                                "target": target,
                                "estimated": float(y[i, j]),
                                "actual": float(actual[i, j]),
                                "error": float(y[i, j] - actual[i, j]),
                            }
                        )
    # Frozen unit transformation and exact channel duplication, not refitting the meter.
    for z in cfg["conditions"]:
        m = meters[f"joint_all__{z}"]
        l, r, b = (np.array(m[k]) for k in ["loading", "covariance", "raw_offset"])
        transform = np.diag([0.5, 2, -1, 3, 0.1, 10, 0.25, 4, 2, -2])
        td = decoder(
            transform @ b,
            transform @ l,
            transform @ r @ transform.T,
            tolerance=cfg["svd_tolerance"],
        )
        y = values[np.array(frame.condition) == z]
        original = y @ np.array(m["weights"]).T + m["offset"]
        transported = (y @ transform.T) @ td["weights"].T + td["offset"]
        repeat = np.vstack([np.eye(10), np.eye(10)[6]])
        dd = decoder(
            repeat @ b,
            repeat @ l,
            repeat @ r @ repeat.T,
            tolerance=cfg["svd_tolerance"],
        )
        repeated = (y @ repeat.T) @ dd["weights"].T + dd["offset"]
        transforms.append(
            {
                "condition": z,
                "unit_max_error": float(np.abs(original - transported).max()),
                "duplicate_max_error": float(np.abs(original - repeated).max()),
                "duplicate_information_error": float(
                    np.abs(np.array(m["information"]) - dd["information"]).max()
                ),
                "rank": dd["rank"],
            }
        )
    pd.DataFrame(outcomes).to_csv(folder / "metrics_by_seed.csv", index=False)
    summary = (
        pd.DataFrame(outcomes)
        .groupby(["style", "arm", "mode", "target", "moment"], sort=False)[
            ["mse", "bias", "correlation", "predicted_variance", "true_variance"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(folder / "comparison.csv", index=False)
    pd.DataFrame(actions).to_csv(folder / "event_calibration_actions.csv", index=False)
    dump(folder / "invariance.json", transforms)
    print(
        summary[(summary["mode"] == "joint") & (summary.target == "new_condition")][
            ["style", "arm", "moment", "mse", "bias", "correlation"]
        ].to_string(index=False),
        flush=True,
    )


def main():
    """Run one bounded stage at a time; source freeze precedes new text observation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "cold_source",
            "fit",
            "cold_application",
            "generate",
            "apply",
            "evaluate",
        ],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    args = parser.parse_args()
    if args.stage.startswith("cold_"):
        cold(args.output.resolve(), args.stage == "cold_source", args.device)
    else:
        globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
