#!/usr/bin/env python3
"""Run the bounded M13 raw-evidence to numerical-change explanation packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.explanation_chain import (
    condition_profile,
    decompose_query,
    make_packet,
    public_bank,
)
from grain_research.query_moment_operator import apply_operator
from grain_research.vector_mechanisms.interpretation_evidence import validate_packet
from grain_research.vector_mechanisms.prefix import capture_prefix
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.run_grain_native_material_reader import BUNDLE, FC, REPRESENTATION
from scripts.run_grain_query_moment_operator import load_models
from scripts.run_grain_real_author_path import MODEL
from scripts.run_grain_within_comment_channels import lexical_slices
from scripts.score_grain_query_moments import raw_coordinates
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

BASE = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1"
OLD = BASE / "query_moment_r1"
CONFIG = ROOT / "configs/grain_explanation_chain.json"
CRITERION = (
    ROOT
    / "results/grain_vector_mechanisms/logic_material_coupling_v1_r2/analysis_r1/criterion_links.csv"
)


def sha(path):
    """Hash a file without loading large representations into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    """Write a generated JSON artifact, rejecting nonfinite values."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def slice_scores(text, representation, bundle):
    """Score four lossless slices and independently replay the FA posterior."""
    parts = lexical_slices(text)
    if len(parts) != 4 or "".join(parts) != text:
        raise ValueError("Four complete slices required")
    rows = [{"author": f"s{i}", "body": part} for i, part in enumerate(parts)]
    frame = canonicalize_comments(
        pd.DataFrame(rows), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    _, features, scored = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=representation,
        bundle=bundle,
    )
    if not scored.score_status.eq("SCORED").all():
        raise ValueError("Unsupported slice")
    order = [r["author"] for r in rows]
    scores = scored.set_index("user_id").loc[order, FC].to_numpy(float)
    x = features.set_index("user_id").loc[order, bundle.feature_names].to_numpy(float)
    x = np.where(np.isfinite(x), x, np.asarray(bundle.feature_impute))
    x = (x - bundle.feature_center) / bundle.feature_scale - bundle.fa_mean
    loading = np.asarray(bundle.factor_loadings)
    inverse_noise_loading = loading / np.asarray(bundle.noise_variance)
    posterior = np.linalg.solve(
        np.eye(4) + inverse_noise_loading @ loading.T, inverse_noise_loading @ x.T
    ).T
    replay = (posterior * bundle.factor_signs - bundle.norm_mean) / bundle.norm_scale
    np.testing.assert_allclose(scores, replay, atol=1e-8, rtol=1e-8)
    return parts, scores, float(np.max(np.abs(scores - replay)))


def comparisons(frame):
    """Make complete paired contrasts; never select a favorable wording."""
    contrasts = []
    for variable, group, lo, hi in [
        ("pole", ["identity", "wording", "minutes"], -1, 1),
        ("minutes", ["identity", "wording", "pole"], 5, 45),
        ("identity", ["wording", "pole", "minutes"], "intention", "reported_action"),
        (
            "identity",
            ["wording", "pole", "minutes"],
            "reported_action",
            "current_utterance",
        ),
    ]:
        for _, rows in frame.groupby(group):
            a, b = (
                rows.loc[rows[variable].eq(lo)].iloc[0],
                rows.loc[rows[variable].eq(hi)].iloc[0],
            )
            for measure in ["grain_f03", "query"]:
                contrasts.append(
                    {
                        "variable": variable,
                        "from": lo,
                        "to": hi,
                        "case_low": a["case"],
                        "case_high": b["case"],
                        "measure": measure,
                        "delta": b[measure] - a[measure],
                        "identity": a["identity"],
                        "wording": int(a.wording),
                    }
                )
    return pd.DataFrame(contrasts)


def run(out):
    """Freeze new stimuli, observe them once and bind explanations to exact scores."""
    import torch

    config = json.loads(CONFIG.read_text())
    bank = public_bank()
    if len(bank) != config["cases"]:
        raise ValueError("Unexpected bank size")
    out.mkdir(parents=True, exist_ok=False)
    (out / "states").mkdir()
    (out / "cards").mkdir()
    write(out / "public_bank.json", bank)
    write(out / "config.json", config)
    dependency_paths = [
        Path(__file__),
        ROOT / "grain_research/explanation_chain.py",
        ROOT / "docs/GRAIN_EXPLANATION_CHAIN_PLAN_V1.md",
        CONFIG,
        ROOT / "scripts/score_grain_query_moments.py",
        ROOT / "grain_research/query_moment_operator.py",
        ROOT / "grain_research/vector_mechanisms/prefix.py",
        ROOT / "grain_research/vector_mechanisms/interpretation_evidence.py",
        ROOT / "configs/grain/interpretation_evidence.schema.json",
        BUNDLE,
        REPRESENTATION,
        MODEL / "config.json",
        OLD / "models/freeze.json",
        OLD / "models/confirmation_seeking.npz",
        CRITERION,
        out / "public_bank.json",
    ]
    frozen = {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
        for p in dependency_paths
    }
    write(
        out / "freeze.json",
        {"files": frozen, "purpose": "M13 bounded computational interpretation"},
    )
    operator_hash = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode()
    ).hexdigest()
    model = load_models(OLD)[config["query"]]
    representation, bundle = joblib.load(REPRESENTATION), read_factor_bundle(BUNDLE)
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(config["torch_threads"])
    start = time.monotonic()
    observer = DecoderObserver.load(
        str(MODEL), device=config["device"], dtype="float32", max_tokens=1024
    )
    observer.model.requires_grad_(False)
    load_seconds = time.monotonic() - start
    start = time.monotonic()
    rows, slices, profiles, traces, errors = [], [], [], [], []
    for row in bank:
        if time.monotonic() - start > config["max_seconds"]:
            raise TimeoutError("Bounded M13 budget exceeded; keep partial artifacts")
        text, key = row["text"], row["case"]
        parts, gs, fa_error = slice_scores(text, representation, bundle)
        ids, positions, g = raw_coordinates(
            [text], observer.tokenizer, representation, bundle
        )
        prefix, trace = capture_prefix(observer, ids)
        h = prefix["boundary"][positions].astype(float)
        np.savez_compressed(
            out / "states" / f"{key}.npz",
            g=g,
            h=h,
            positions=positions,
            input_ids=np.asarray(ids),
            slice_scores=gs,
        )
        values, _ = apply_operator(g, h, model)
        offset, between, within = decompose_query(g, h, model)
        q = float(values[2])
        replay_error = abs(q - offset - between.sum() - within.sum())
        if replay_error > config["numerical_tolerance"]:
            raise AssertionError("Query decomposition did not replay")
        values_dict = {"grain_f03": float(gs[:, 2].mean()), "query": q}
        packet = make_packet(row, values_dict, {k: operator_hash for k in values_dict})
        trusted = {m["id"]: m for m in packet["measurements"]}
        validate_packet(packet, {key: text}, trusted)
        write(out / "cards" / f"{key}.json", packet)
        (out / "cards" / f"{key}.md").write_text(
            f"# {key}: {row['identity']}, {row['minutes']} minutes\n\n"
            f"> {row['target_span']}\n\n"
            f"F03 uniform slice mean: {values_dict['grain_f03']:.8f}. "
            f"Query potential: {q:.8f} = {offset:.8f} (source anchor) + "
            f"{between.sum():.8f} (between-cell) + {within.sum():.8f} (within-cell).\n\n"
            f"Candidate: {packet['candidates'][0]['hypothesis']}. "
            "Identity is from the controlled stimulus manifest, not inferred from the score. "
            "Neither number is a probability or clinical/personality conclusion.\n"
        )
        cursor = 0
        for si, (part, scores) in enumerate(zip(parts, gs)):
            slices.append(
                {
                    "case": key,
                    "slice": si,
                    "start": cursor,
                    "end": cursor + len(part),
                    "text": part,
                    **dict(zip(["g1", "g2", "g3", "g4"], scores)),
                }
            )
            cursor += len(part)
        pd.DataFrame(
            {"model_token_position": positions, "between": between, "within": within}
        ).to_csv(out / "states" / f"{key}_contributions.csv", index=False)
        rows.append(
            {k: v for k, v in row.items() if k not in ["text", "target_span"]}
            | values_dict
            | {
                "anchor": offset,
                "between": float(between.sum()),
                "within": float(within.sum()),
                "token_f03_mean": float(g[:, 2].mean()),
                "tokens": len(positions),
            }
        )
        errors.append({"case": key, "fa_error": fa_error, "query_error": replay_error})
        traces.append({"case": key, **trace})
        print(key, values_dict, flush=True)
    if observer.calls != config["max_prefix_calls"]:
        raise AssertionError("Unexpected prefix calls")
    frame = pd.DataFrame(rows)
    contrast = comparisons(frame)
    for keys, endpoints in frame.groupby(["identity", "pole", "wording"]):
        endpoints = endpoints.set_index("minutes")
        for measure in ["grain_f03", "query"]:
            profile = condition_profile(
                endpoints.loc[5, measure], endpoints.loc[45, measure]
            )
            profiles.append(
                dict(zip(["identity", "pole", "wording"], keys))
                | {
                    "measure": measure,
                    "case_low": endpoints.loc[5, "case"],
                    "case_high": endpoints.loc[45, "case"],
                    **profile,
                }
            )
    for name, data in [
        ("readings", frame),
        ("contrasts", contrast),
        ("slice_scores", pd.DataFrame(slices)),
        ("condition_profiles", pd.DataFrame(profiles)),
        ("numerical_replay", pd.DataFrame(errors)),
    ]:
        data.to_csv(out / f"{name}.csv", index=False)
    criterion = pd.read_csv(CRITERION)
    criterion.loc[criterion.factor.eq("whole:F03")].to_csv(
        out / "historical_f03_criterion_links.csv", index=False
    )
    summaries = []
    for keys, data in contrast.groupby(
        ["variable", "from", "to", "measure"], sort=False
    ):
        summaries.append(
            dict(zip(["variable", "from", "to", "measure"], keys))
            | {
                "pairs": len(data),
                "positive": int(data.delta.gt(0).sum()),
                "mean_delta": float(data.delta.mean()),
                "min_delta": float(data.delta.min()),
                "max_delta": float(data.delta.max()),
            }
        )
    for path, digest in frozen.items():
        if sha(ROOT / path) != digest:
            raise AssertionError("Frozen dependency changed during execution: " + path)
    write(
        out / "summary.json",
        {
            "status": "complete",
            "cases": len(frame),
            "prefix_calls": observer.calls,
            "load_seconds": load_seconds,
            "execution_seconds": time.monotonic() - start,
            "raw_score_identity_metadata_used": False,
            "meaning_source": "authored stimulus manifest",
            "fa_max_error": max(x["fa_error"] for x in errors),
            "query_max_error": max(x["query_error"] for x in errors),
            "contrasts": summaries,
            "human_behavior_observed": False,
            "new_neural_training": False,
            "traces": traces,
        },
    )
    write(
        out / "artifact_manifest.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and not p.name.startswith("._")
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "explanation_chain_r1")
    args = parser.parse_args()
    run(args.out)
