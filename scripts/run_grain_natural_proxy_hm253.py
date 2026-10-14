#!/usr/bin/env python3
"""One frozen natural three-view bridge experiment, never a precision search."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_natural_proxy as M
from scripts import run_grain_observed_family_hm170 as R

CONFIG = "configs/grain_natural_proxy_hm253.json"
DEFAULT = ROOT / "results/grain_math_response_v1/natural_proxy_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_NATURAL_PROXY_PLAN_HM253.md",
    "grain_research/construction/math_natural_proxy.py",
    "scripts/run_grain_natural_proxy_hm253.py",
    "scripts/score_grain_natural_proxy_cold.py",
    "scripts/verify_grain_natural_proxy_hm253.py",
    "tests/test_grain_natural_proxy.py",
    "scripts/run_grain_observed_family_hm170.py",
)


def arrays(path):
    """Read only selected licensed numerical derivatives, no private material."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check(out, stages=()):
    """Verify this packet's source/code identities and immutable stage outputs."""
    return R.check(out, stages)


def prepare(out):
    """Join existing public source criteria to disjoint original comment units."""
    cfg = R.read(ROOT / CONFIG)
    ep, sp = (ROOT / cfg[k] for k in ("event_parent", "source_parent"))
    for parent in (ep, sp):
        expected = R.read(parent / "prepared_seal.json")["observations.npz"]
        if R.digest(parent / "observations.npz") != expected:
            raise ValueError("Changed licensed numerical observation identity")
    e, s = arrays(ep / "observations.npz"), arrays(sp / "observations.npz")
    ids, folds = s["source_indices"], s["folds"]
    if len(ids) != 386 or len(np.unique(ids)) != 386 or s["labels"].shape != (386, 5):
        raise ValueError("Changed source author or five-criterion scope")
    roles = np.concatenate(
        [e[k] for k in ("source_train", "source_validation", "held")]
    )
    if len(np.unique(roles)) != 624 or not set(ids) <= set(e["source_train"]):
        raise ValueError("Source/public cohort roles disagree")
    order = e["original_units"]
    if not np.all(np.sort(order, axis=1) == np.arange(36)):
        raise ValueError("Invalid original event identity permutation")
    inverse = np.argsort(order, axis=1)
    original = np.take_along_axis(e["coordinates"], inverse[:, :, None], axis=1)
    replay = original[ids, :24].reshape(386, 2, 12, 15) - s["vectors"] @ s["mapping"]
    if np.max(np.abs(replay)) > 2e-12:
        raise ValueError("Public criteria and coordinate event units disagree")
    coordinates, conditions, units = (
        e["coordinates"][ids],
        e["conditions"][ids],
        order[ids],
    )
    if coordinates.shape != (386, 36, 15) or not np.isin(s["labels"], [0, 1]).all():
        raise ValueError("Registered natural observations/criteria changed")
    features = M.view_features(coordinates, conditions, units)
    counts = np.stack([conditions[:, v::3].sum(1) for v in range(3)], axis=1)
    lengths = np.stack([e["lengths"][ids, v::3].mean(1) for v in range(3)], axis=1)
    timestamps = e["times"][ids]
    quality = {
        "authors": 386,
        "public_cached_events": 386 * 36,
        "views": 3,
        "events_per_view": 12,
        "features": 32,
        "fold_counts": np.bincount(folds).tolist(),
        "label_prevalence": s["labels"].mean(0).tolist(),
        "label_units": "historical binary Big5 high/low groups, NOT continuous scores",
        "mean_top20_share_by_view": (counts / 12).mean(0).tolist(),
        "median_span_days_by_view": [
            float(
                np.median(
                    (timestamps[:, v::3][:, -1] - timestamps[:, v::3][:, 0]) / 86400
                )
            )
            for v in range(3)
        ],
        "median_chronological_third_span_days": [
            float(
                np.median((timestamps[:, 12 * v + 11] - timestamps[:, 12 * v]) / 86400)
            )
            for v in range(3)
        ],
        "coordinate_replay_error": float(np.max(np.abs(replay))),
        "disjoint_event_views": True,
        "independent_proxy_errors_established": False,
        "new_private_raw_labels_or_model": False,
        "original_scope_complete": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for parent in (ep, sp):
        for filename in ("observations.npz", "prepared_seal.json", "manifest.json"):
            path = parent / filename
            inputs[str(path.relative_to(ROOT))] = R.digest(path)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "inputs": inputs,
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
        },
    )
    R.archive(
        out / "observations.npz",
        features=features,
        labels=s["labels"],
        folds=folds,
        source_indices=ids,
        units=units,
        coordinates=coordinates,
        conditions=conditions,
        counts=counts,
        lengths=lengths,
    )
    R.archive(out / "cold_inputs.npz", features=features, folds=folds)
    R.write(out / "quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_inputs.npz", "quality.json"),
    )
    print(json.dumps(quality))


def compile_programs(data, cfg):
    """Fit exactly three source observation laws per author fold, all targets."""
    programs = []
    for f in cfg["source_folds"]:
        train, held = M.author_fold(data["folds"], f)
        for w, z, t in cfg["bridges"]:
            program = M.construct(
                data["features"][train],
                data["labels"][train],
                w,
                z,
                cfg["relative_rank_tolerance"],
            )
            program.update(
                {
                    "source_fold": f,
                    "w": w,
                    "z": z,
                    "t": t,
                    "source_authors": len(train),
                    "held_authors": len(held),
                }
            )
            for method in ("bridge", "ordinary"):
                program[method + "_unused_source_moments"] = M.moment_error(
                    data["features"][train],
                    data["labels"][train],
                    program[method],
                    w,
                    t,
                    program["source_rms"],
                )
            programs.append(program)
    return programs


def compile_stage(out):
    """Freeze criterion-linked source coefficients before held-score analysis."""
    cfg = check(out)["config"]
    data = arrays(out / "observations.npz")
    start = time.perf_counter()
    programs = compile_programs(data, cfg)
    R.write(
        out / "bundle.json",
        {
            "targets": cfg["targets"],
            "programs": programs,
            "contract": "candidate linear proxy moments; no latent/personality identification certified",
        },
    )
    R.write(
        out / "compile_runtime.json",
        {
            "seconds": time.perf_counter() - start,
            "programs": len(programs),
            "source_folds": 2,
            "application_answers_used": False,
        },
    )
    R.seal(out, "compiled_seal.json", ("bundle.json", "compile_runtime.json"))
    print("Compiled six fixed source programs; held answers not used")


def predictions(features, folds, programs):
    """Route each author only to the OTHER-fold source program, label-free."""
    result = np.full((2, 3, 2, len(features), 5), np.nan)
    for p in programs:
        _, held = M.author_fold(folds, p["source_fold"])
        for method_index, method in enumerate(("bridge", "ordinary")):
            for view_index, view in enumerate((p["w"], p["t"])):
                result[method_index, p["w"], view_index, held] = features[
                    held, view
                ] @ np.asarray(p[method])
    if not np.isfinite(result).all():
        raise ValueError("Missing cold score or author source/application overlap")
    return result


def analyze(out):
    """Use sealed cold readings and all fixed nulls; never choose a best rule."""
    cfg = check(out, ("compiled_seal.json", "cold_seal.json"))["config"]
    d = arrays(out / "observations.npz")
    programs = R.read(out / "bundle.json")["programs"]
    score = arrays(out / "cold_scores.npz")["scores"]
    y, f, x = d["labels"], d["folds"], d["features"]
    start = time.perf_counter()
    observed_r = np.array(
        [
            [
                [
                    [M.correlation(y[:, q], score[m, c, v, :, q]) for q in range(5)]
                    for v in range(2)
                ]
                for c in range(3)
            ]
            for m in range(2)
        ],
        float,
    )
    # Linear source operators are fixed by vectors; only criteria move in nulls.
    operators = []
    for p in programs:
        train, _ = M.author_fold(f, p["source_fold"])
        a, b = x[train, p["w"]], x[train, p["z"]]
        operators.append(
            (
                p,
                train,
                np.linalg.pinv(
                    b.T @ a / len(train), rcond=cfg["relative_rank_tolerance"]
                )
                @ b.T
                / len(train),
                np.linalg.pinv(a, rcond=cfg["relative_rank_tolerance"]),
            )
        )
    rng = np.random.default_rng(cfg["profile_null_seed"])
    null_r = np.empty((cfg["null_replicates"], 2, 3, 2, 5))
    for repeat in range(cfg["null_replicates"]):
        permuted = M.permute_profiles(y, f, rng)
        changed = []
        for p, train, op_bridge, op_ordinary in operators:
            changed.append(
                {
                    **p,
                    "bridge": op_bridge @ permuted[train],
                    "ordinary": op_ordinary @ permuted[train],
                }
            )
        null_score = predictions(x, f, changed)
        for method in range(2):
            for cycle in range(3):
                for view in range(2):
                    for q in range(5):
                        null_r[repeat, method, cycle, view, q] = M.correlation(
                            permuted[:, q], null_score[method, cycle, view, :, q]
                        )
        if time.perf_counter() - start > cfg["CPU_seconds_cap"]:
            raise TimeoutError("Finite CPU cap reached; no extended precision queue")
    rows = []
    agreement = []
    moments = []
    for mi, method in enumerate(("bridge", "ordinary")):
        for cycle, (w, z, t) in enumerate(cfg["bridges"]):
            donors = np.full(len(y), -1, int)
            for sf in (0, 1):
                train, held = M.author_fold(f, sf)
                local = M.matched_donors(
                    d["counts"][held, t],
                    d["lengths"][held, t],
                    np.median(d["lengths"][train, t]),
                )
                eligible = local >= 0
                donors[held[eligible]] = held[local[eligible]]
            eligible = donors >= 0
            for q, target in enumerate(cfg["targets"]):
                same = M.correlation(
                    score[mi, cycle, 0, eligible, q], score[mi, cycle, 1, eligible, q]
                )
                stranger = M.correlation(
                    score[mi, cycle, 0, eligible, q],
                    score[mi, cycle, 1, donors[eligible], q],
                )
                agreement.append(
                    {
                        "method": method,
                        "cycle": cycle,
                        "target": target,
                        "eligible_authors": int(eligible.sum()),
                        "same_author_r": same,
                        "matched_stranger_r": stranger,
                        "same_minus_stranger": same - stranger
                        if same is not None and stranger is not None
                        else None,
                    }
                )
                for vi, view in enumerate((w, t)):
                    p = score[mi, cycle, vi, :, q]
                    r = observed_r[mi, cycle, vi, q]
                    rows.append(
                        {
                            "method": method,
                            "cycle": cycle,
                            "reading": "W" if vi == 0 else "T",
                            "view": view,
                            "target": target,
                            "authors": len(y),
                            "pearson_r": float(r),
                            "AUC": M.auc(y[:, q], p),
                            "bias": float((p - y[:, q]).mean()),
                            "score_min": float(p.min()),
                            "score_max": float(p.max()),
                            "fold0_r": M.correlation(y[f == 0, q], p[f == 0]),
                            "fold1_r": M.correlation(y[f == 1, q], p[f == 1]),
                            "directional_profile_null_p": float(
                                (1 + np.sum(null_r[:, mi, cycle, vi, q] >= r))
                                / (cfg["null_replicates"] + 1)
                            ),
                            "five_target_max_abs_p": float(
                                (
                                    1
                                    + np.sum(
                                        np.max(np.abs(null_r[:, mi, cycle, vi]), axis=1)
                                        >= abs(r)
                                    )
                                )
                                / (cfg["null_replicates"] + 1)
                            ),
                        }
                    )
        for p in programs:
            train, held = M.author_fold(f, p["source_fold"])
            for sample_name, sample in (("source_unused", train), ("held", held)):
                instruments = (
                    (p["t"],) if sample_name == "source_unused" else (p["z"], p["t"])
                )
                for instrument in instruments:
                    error = M.moment_error(
                        x[sample],
                        y[sample],
                        p[method],
                        p["w"],
                        instrument,
                        p["source_rms"],
                    )
                    for q, target in enumerate(cfg["targets"]):
                        moments.append(
                            {
                                "method": method,
                                "source_fold": p["source_fold"],
                                "cycle": p["w"],
                                "sample": sample_name,
                                "instrument": instrument,
                                "target": target,
                                "normalized_moment_rms": error["normalized_rms"][q],
                                "calibration_bias": error["calibration_bias"][q],
                            }
                        )
    R.table(out / "target_links.csv", rows)
    R.table(out / "author_agreement.csv", agreement)
    R.table(out / "unused_moments.csv", moments)
    R.archive(out / "profile_nulls.npz", correlations=null_r)
    overview = {}
    for method in ("bridge", "ordinary"):
        third = [r for r in rows if r["method"] == method and r["reading"] == "T"]
        overview[method] = {
            "third_view_r_mean": float(np.mean([r["pearson_r"] for r in third])),
            "third_view_r_by_cycle": [
                [r["pearson_r"] for r in third if r["cycle"] == c] for c in range(3)
            ],
            "positive_third_links": sum(r["pearson_r"] > 0 for r in third),
            "five_target_adjusted_third_links": sum(
                r["pearson_r"] > 0 and r["five_target_max_abs_p"] <= 0.05 for r in third
            ),
            "same_minus_stranger_mean": float(
                np.mean(
                    [
                        r["same_minus_stranger"]
                        for r in agreement
                        if r["method"] == method
                    ]
                )
            ),
            "source_unused_moment_mean": float(
                np.mean(
                    [
                        r["normalized_moment_rms"]
                        for r in moments
                        if r["method"] == method and r["sample"] == "source_unused"
                    ]
                )
            ),
            "held_moment_mean": float(
                np.mean(
                    [
                        r["normalized_moment_rms"]
                        for r in moments
                        if r["method"] == method and r["sample"] == "held"
                    ]
                )
            ),
        }
    drift = []
    for w in range(3):
        bank = [p for p in programs if p["w"] == w]
        drift.append(
            {
                "cycle": w,
                "bridge_coefficient_difference_norm": np.linalg.norm(
                    np.asarray(bank[0]["bridge"]) - bank[1]["bridge"], axis=0
                ).tolist(),
                "ordinary_coefficient_difference_norm": np.linalg.norm(
                    np.asarray(bank[0]["ordinary"]) - bank[1]["ordinary"], axis=0
                ).tolist(),
            }
        )
    summary = {
        "authors": len(y),
        "source_criterion_targets": cfg["targets"],
        "candidate_links": overview,
        "coefficient_drift": drift,
        "null_replicates": cfg["null_replicates"],
        "analysis_seconds": time.perf_counter() - start,
        "new_model_API_private_training": 0,
        "original_scope_complete": False,
        "qualification": "frozen public derivatives; independent events not independent errors; exploratory cross-fit not untouched psych validation; no J-test or clipped scores",
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "target_links.csv",
            "author_agreement.csv",
            "unused_moments.csv",
            "profile_nulls.npz",
            "summary.json",
        ),
    )
    print(json.dumps(summary))


def main():
    """Execute one exclusive stage; preserve existing results rather than retry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compile", "analyze", "check"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "check":
        check(
            out,
            (
                "compiled_seal.json",
                "cold_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        )
        print("HM253 all science/input/output seals PASS")
    else:
        {"prepare": prepare, "compile": compile_stage, "analyze": analyze}[args.stage](
            out
        )


if __name__ == "__main__":
    main()
