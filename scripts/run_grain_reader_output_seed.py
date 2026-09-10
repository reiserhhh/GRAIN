#!/usr/bin/env python3
"""NEXT38: cached output-seed and suffix-transport diagnostics, no neural calls."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.native_material_reader import cosine, direction
from grain_research.vector_mechanisms.output_seed import (
    binary_gain,
    coefficient_actions,
    projection_fraction,
    row_span,
    score_seed,
    source_gains,
)
from grain_research.vector_mechanisms.reader_ensemble import unpack_models
from scripts.analyze_grain_position_reader import action_metrics
from scripts.analyze_grain_query_residual_input import (
    verify_predictions as verify_old_predictions,
)
from scripts.run_grain_native_material_reader import OLD, QUERIES
from scripts.run_grain_native_material_reader import verify as verify_old_native
from scripts.run_grain_reader_new_author_transport import (
    packet_arrays,
    verify_hashes,
    verify_prediction_freeze,
)
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

PLAN = ROOT / "docs/GRAIN_READER_OUTPUT_SEED_PLAN_V1.md"
NATIVE = BASE / "logic_native_material_reader_v1_r2"
READERS = BASE / "logic_query_residual_input_v1_r2"
CONTENT = BASE / "logic_content_coefficients_v1"
NEW = BASE / "logic_reader_new_author_transport_v1"


def load_assets():
    """Verify already computed endpoints and read only required frozen output rows."""
    verify_old_native(NATIVE)
    verify_old_predictions(READERS)
    verify_prediction_freeze(NEW)
    verify_hashes(NEW, read(NEW / "analysis/manifest.json")["native_packet_hashes"])
    if read(NEW / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("NEXT37 must be complete")
    checkpoint = MODEL / "model.safetensors"
    if sha(checkpoint) != read(OLD / "manifest.json")["model_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    old_members = read(OLD / "private_inputs.json")
    new_members = read(NEW / "private_design.json")
    if (
        len(old_members) != 4
        or len(new_members) != 8
        or {a["author_id"] for a in old_members} & {a["author_id"] for a in new_members}
    ):
        raise ValueError("Old4/new8 author contract changed")
    units, files = (
        {},
        [
            PLAN,
            Path(__file__),
            ROOT / "grain_research/vector_mechanisms/output_seed.py",
            MODEL / "config.json",
            OLD / "private_inputs.json",
            NATIVE / "manifest.json",
            READERS / "manifest.json",
            READERS / "prediction_freeze.json",
            READERS / "private_membership.json",
            NEW / "manifest.json",
            NEW / "prediction_freeze.json",
            NEW / "analysis/manifest.json",
        ],
    )
    alphabets = {}
    for cohort, members, directory in (
        ("source4", old_members, NATIVE),
        ("new8", new_members, NEW),
    ):
        for author in members:
            i = author["index"]
            for q in QUERIES:
                grad_path = directory / "private_packets" / f"a{i:02d}_{q}.json"
                meta, arrays = packet_arrays(grad_path)
                prefix_path = (
                    OLD / "private_packets" / f"unit_{i:02d}_{q}.json"
                    if cohort == "source4"
                    else NEW / "private_prefix" / f"a{i:02d}_{q}.json"
                )
                full = read(prefix_path)["full"]
                alphabet = author["queries"][q]["alphabet"]
                if q in alphabets and alphabet != alphabets[q]:
                    raise ValueError("Candidate order or values differ across authors")
                alphabets[q] = alphabet
                units[cohort, i, q] = {
                    "gradient": arrays["body_gradient"].astype(float),
                    "full": full,
                    "score": meta["score"],
                    "alphabet": alphabet,
                }
                files.extend([grad_path, grad_path.with_suffix(".npz"), prefix_path])
    ids = sorted({i for a in alphabets.values() for i in a["token_ids"]})
    with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
        key = "lm_head.weight"
        if key not in set(f.keys()):
            if not read(MODEL / "config.json").get("tie_word_embeddings"):
                raise ValueError("Absent untied LM head")
            key = "model.embed_tokens.weight"
        view = f.get_slice(key)
        weights = {i: view[i : i + 1].float().numpy()[0].astype(float) for i in ids}
    return (
        units,
        alphabets,
        weights,
        files,
        {
            "tensor": key,
            "rows": len(ids),
            "shape": view.get_shape(),
            "checkpoint_sha256": sha(checkpoint),
        },
    )


def summarize(frame, keys, columns):
    """Aggregate finite values without exposing author identifiers or text."""
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key, strict=True)) | {"authors": len(group)}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            for stat in ("mean", "min", "max"):
                row[column + "_" + stat] = (
                    float(getattr(values, stat)()) if len(values) else None
                )
        rows.append(row)
    return pd.DataFrame(rows)


def binary_diagnostic(units, files):
    """Read source-only converted programs and evaluator-only binary target scaling."""
    folds = {
        a["held"]: a["source"]
        for a in read(READERS / "private_membership.json")["folds"]
    }
    gains = source_gains(
        {i: units["source4", i, "TF_cont"]["score"] for i in range(4)}, folds
    )
    rows, replay_max = [], 0.0
    with np.load(READERS / "private_predictions.npz", allow_pickle=False) as z:
        old_predictions = {f: z[f] for f in z.files if f.startswith("TF_cont")}
    with np.load(NEW / "private_predictions.npz", allow_pickle=False) as z:
        new_predictions = {f: z[f] for f in z.files if f.startswith("TF_cont")}
    with np.load(NEW / "private_models/TF_cont__TF_cont.npz", allow_pickle=False) as z:
        ensemble = unpack_models(dict(z))
    files += [
        READERS / "private_predictions.npz",
        NEW / "private_predictions.npz",
        NEW / "private_models/TF_cont__TF_cont.npz",
    ]
    existing = {
        "source4": pd.read_csv(READERS / "analysis/private_actions.csv"),
        "new8": pd.read_csv(NEW / "analysis/private_actions.csv"),
    }
    for cohort, count in (("source4", 4), ("new8", 8)):
        for i in range(count):
            inp = (
                CONTENT / f"private_inputs_{i}.npz"
                if cohort == "source4"
                else NEW / "private_inputs" / f"a{i:02d}.npz"
            )
            files.append(inp)
            with np.load(inp, allow_pickle=False) as z:
                h, g = z["hidden"], z["grain"]
            dirs = np.stack([direction(h, g, k)[0] for k in range(4)])
            unit = units[cohort, i, "TF_cont"]
            actual = np.einsum("kth,th->k", dirs, unit["gradient"])
            if cohort == "source4":
                path = READERS / "TF_cont" / f"private_model_{i}_residual_query.npz"
                files.append(path)
                with np.load(path, allow_pickle=False) as z:
                    v = z["basis"]
                pred = coefficient_actions(
                    old_predictions[f"TF_cont__a{i}__residual_query"], v, dirs
                )
                converted = pred / gains[i]
                arm = "residual_query"
            else:
                coefs = new_predictions[f"TF_cont__a{i}__TF_cont"]
                component, offset = [], 0
                for model in ensemble:
                    rank = len(model["basis"])
                    component.append(
                        coefficient_actions(
                            coefs[:, offset : offset + rank], model["basis"], dirs
                        )
                    )
                    offset += rank
                component = np.stack(component)
                pred = component.mean(0)
                converted = (
                    component / np.asarray([gains[s] for s in range(4)])[:, None]
                ).mean(0)
                arm = "TF_cont"
            saved = existing[cohort]
            saved = saved[
                (saved["query"] == "TF_cont")
                & (saved.arm == arm)
                & (saved["index"] == i)
            ].sort_values("factor")
            discrepancy = max(
                float(abs(actual - saved.actual.to_numpy()).max()),
                float(abs(pred - saved.predicted.to_numpy()).max()),
            )
            replay_max = max(replay_max, discrepancy)
            if discrepancy > 1e-10:
                raise ValueError("Source/new frozen action replay changed")
            gain = float(binary_gain(unit["score"]))
            for k in range(4):
                rows.append(
                    {
                        "cohort": cohort,
                        "index": i,
                        "factor": f"whole:F{k + 1:02d}",
                        "gain": gain,
                        "saturated": gain < 1e-6,
                        "division_supported": gain > 1e-12,
                        "actual": actual[k],
                        "predicted": pred[k],
                        "source_converted_prediction": converted[k],
                        "contrast_target": actual[k] / gain if gain > 1e-12 else None,
                        "current_gain_reweighted_prediction": pred[k] / gain
                        if gain > 1e-12
                        else None,
                    }
                )
    frame, comparisons = pd.DataFrame(rows), []
    for cohort, group in frame.groupby("cohort", sort=True):
        for mode, target, predicted, endpoint_used in (
            ("original_probability", "actual", "predicted", False),
            (
                "source_gain_contrast",
                "contrast_target",
                "source_converted_prediction",
                False,
            ),
            (
                "current_gain_error_reweighting",
                "contrast_target",
                "current_gain_reweighted_prediction",
                True,
            ),
        ):
            valid = (
                group
                if mode == "original_probability"
                else group[group.division_supported]
            )
            comparisons.append(
                {
                    "cohort": cohort,
                    "mode": mode,
                    "authors": valid["index"].nunique(),
                    "unsupported_authors": group["index"].nunique()
                    - valid["index"].nunique(),
                    "predictor_uses_current_score": endpoint_used,
                    **action_metrics(valid[target], valid[predicted]),
                }
            )
    return frame, pd.DataFrame(comparisons), gains, replay_max


def run(name):
    """Run one bounded no-fit packet; preserve outputs if the wall boundary fails."""
    out = BASE / name
    out.mkdir(exist_ok=False)
    loading = time.monotonic()
    units, alphabets, weights, files, output_info = load_assets()
    load_seconds = time.monotonic() - loading
    start = time.monotonic()
    states, rows, pairs, arrays = {}, [], [], {}
    score_error = probability_error = sum_error = fd_error = 0.0
    for (cohort, i, q), unit in units.items():
        w = np.stack([weights[token] for token in alphabets[q]["token_ids"]])
        full = unit["full"]
        z, values = full["candidate_logits"], alphabets[q]["values"]
        seed = score_seed(z, values, w)
        score_error = max(score_error, abs(seed["score"] - full["conditional_score"]))
        probability_error = max(
            probability_error,
            float(abs(seed["p"] - full["conditional_probabilities"]).max()),
        )
        sum_error = max(sum_error, abs(float(seed["candidate_seed"].sum())))
        # A fixed normalized value contrast checks the full softmax differential.
        delta = np.asarray(values, float) - np.mean(values)
        delta /= np.linalg.norm(delta)
        eps = 1e-5
        finite = (
            score_seed(np.asarray(z) + eps * delta, values, w)["score"]
            - score_seed(np.asarray(z) - eps * delta, values, w)["score"]
        ) / (2 * eps)
        fd_error = max(fd_error, abs(finite - seed["candidate_seed"] @ delta))
        j = unit["gradient"]
        snorm, jnorm = (
            float(np.linalg.norm(seed["hidden_seed"])),
            float(np.linalg.norm(j)),
        )
        rows.append(
            {
                "cohort": cohort,
                "index": i,
                "query": q,
                "score": seed["score"],
                "max_probability": float(seed["p"].max()),
                "min_probability": float(seed["p"].min()),
                "candidate_seed_norm": float(np.linalg.norm(seed["candidate_seed"])),
                "hidden_seed_norm": snorm,
                "gradient_norm": jnorm,
                "gradient_density_rms": float(len(j) * np.sqrt(np.mean(j**2))),
                "transport_norm_ratio": jnorm / snorm if snorm else None,
                "candidate_mass": full["candidate_mass"],
                "canonical_score_discrepancy": abs(unit["score"] - seed["score"]),
                "tokens": len(j),
                "saturated_binary": q == "TF_cont"
                and binary_gain(unit["score"]) < 1e-6,
            }
        )
        states[cohort, i, q] = seed
        for field in ("p", "candidate_seed", "hidden_seed"):
            arrays[f"{cohort}__a{i}__{q}__{field}"] = seed[field]
    if max(score_error, probability_error, sum_error) > 1e-12 or fd_error > 1e-8:
        raise ValueError("Softmax/seed identity replay failed")
    for cohort, count in (("source4", 4), ("new8", 8)):
        for i in range(count):
            for qa, qb in itertools.combinations(QUERIES, 2):
                sa, sb = states[cohort, i, qa], states[cohort, i, qb]
                ja, jb = (
                    units[cohort, i, qa]["gradient"],
                    units[cohort, i, qb]["gradient"],
                )
                pairs.append(
                    {
                        "cohort": cohort,
                        "index": i,
                        "query_a": qa,
                        "query_b": qb,
                        "same_alphabet": alphabets[qa] == alphabets[qb],
                        "candidate_seed_cosine": cosine(
                            sa["candidate_seed"], sb["candidate_seed"]
                        )
                        if alphabets[qa] == alphabets[qb]
                        else None,
                        "hidden_seed_cosine": cosine(
                            sa["hidden_seed"], sb["hidden_seed"]
                        ),
                        "body_gradient_cosine": cosine(ja, jb),
                        "mean_reader_cosine": cosine(ja.sum(0), jb.sum(0)),
                    }
                )
    coverage = []
    for q in QUERIES:
        basis = row_span(
            np.stack([states["source4", i, q]["hidden_seed"] for i in range(4)])
        )
        arrays[q + "__source_seed_basis"] = basis
        for i in range(8):
            coverage.append(
                {
                    "query": q,
                    "index": i,
                    "source_authors": 4,
                    "rank": len(basis),
                    "seed_energy_covered": projection_fraction(
                        states["new8", i, q]["hidden_seed"], basis
                    ),
                }
            )
    bf, comparisons, gains, action_error = binary_diagnostic(units, files)
    summary_rows, pair_rows, cv = (
        pd.DataFrame(rows),
        pd.DataFrame(pairs),
        pd.DataFrame(coverage),
    )
    summary_rows.to_csv(out / "private_units.csv", index=False)
    pair_rows.to_csv(out / "private_query_pairs.csv", index=False)
    cv.to_csv(out / "private_seed_coverage.csv", index=False)
    bf.to_csv(out / "private_binary_actions.csv", index=False)
    comparisons.to_csv(out / "binary_comparison.csv", index=False)
    np.savez_compressed(out / "private_seeds.npz", **arrays)
    summarize(
        summary_rows,
        ["cohort", "query"],
        [
            "score",
            "max_probability",
            "hidden_seed_norm",
            "gradient_norm",
            "gradient_density_rms",
            "transport_norm_ratio",
            "candidate_mass",
            "canonical_score_discrepancy",
        ],
    ).to_csv(out / "seed_norms.csv", index=False)
    summarize(
        pair_rows,
        ["cohort", "query_a", "query_b"],
        [
            "candidate_seed_cosine",
            "hidden_seed_cosine",
            "body_gradient_cosine",
            "mean_reader_cosine",
        ],
    ).to_csv(out / "query_geometry.csv", index=False)
    summarize(cv, ["query"], ["rank", "seed_energy_covered"]).to_csv(
        out / "seed_coverage.csv", index=False
    )
    elapsed = time.monotonic() - start
    summary = {
        "status": "COMPLETE" if elapsed <= 120 else "FAILED_WALL_BOUND",
        "authors": 12,
        "source_authors": 4,
        "new_authors": 8,
        "queries": 3,
        "units": 36,
        "pair_rows": 36,
        "output_weights": output_info,
        "source_binary_gains": {str(k): v for k, v in gains.items()},
        "score_replay_max": score_error,
        "probability_replay_max": probability_error,
        "seed_sum_max": sum_error,
        "finite_derivative_error_max": fd_error,
        "canonical_score_discrepancy_max": float(
            summary_rows.canonical_score_discrepancy.max()
        ),
        "binary_action_replay_max": action_error,
        "saturated_binary_authors": int(summary_rows.saturated_binary.sum()),
        "unsupported_binary_authors": int(
            bf[~bf.division_supported][["cohort", "index"]].drop_duplicates().shape[0]
        ),
        "new_neural_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "loading_seconds": load_seconds,
        "numeric_seconds": elapsed,
        "files": {
            p.name: sha(p)
            for p in sorted(out.iterdir())
            if p.suffix in (".csv", ".npz") and not p.name.startswith(".")
        },
    }
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_READER_OUTPUT_SEED_V1",
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "model_sha256": output_info["checkpoint_sha256"],
            "budget_seconds_after_loading": 120,
            "native_prefix_seeds_vs_canonical_gradient_approximation": True,
        },
    )
    write(out / "summary.json", summary)
    print(pd.read_csv(out / "query_geometry.csv").to_string(index=False))
    print(comparisons.to_string(index=False))
    print(summary)
    if elapsed > 120:
        raise TimeoutError(
            "Numeric wall budget exceeded; retain packet without relaunch"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="logic_reader_output_seed_v1")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.name)
