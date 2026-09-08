#!/usr/bin/env python3
"""Construct conditional anchor functions from cached prefix vectors, not human labels."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_baseline import (
    attention_moments,
    author_folds,
    grouped_predictions,
    project_boundary,
)
from grain_research.vector_mechanisms.configuration_source import full_head
from grain_research.vector_mechanisms.level_response import correlation
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_author_discovery import scalar_metrics
from scripts.run_grain_level_response import question_mask
from scripts.run_grain_prefix_source_connection import LEVEL
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write
from scripts.run_grain_source_remainder import covariance

SOURCE = BASE / "source_anchor_diagnosis_v1"
PREFIX = BASE / "prefix_source_connection_v1"
PLAN = ROOT / "docs/GRAIN_CONDITIONAL_BASELINE_CONSTRUCTION_PLAN_V1.md"
FAMILIES = (
    "evidence_mean",
    "native_PQ",
    "anchored_attention",
    "anchored_moments",
)


def prepare(out):
    """Freeze all source arrays and deterministic author folds before construction."""
    if read(SOURCE / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Completed source anchors required")
    design = read(SOURCE / "private_design.json")
    if len(design) != 48 or len({r["author_id"] for r in design}) != 48:
        raise ValueError("Exact 48-author panel required")
    for query in ("TF_cont", "neuroticism"):
        rows = [r for r in design if r["query"] == query]
        if len(rows) != 24:
            raise ValueError("24 authors per query required")
        folds = author_folds([r["author_id"] for r in rows])
        for row, fold in zip(rows, folds, strict=True):
            row["fold"] = int(fold)
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_features").mkdir()
    write(out / "private_design.json", design)
    sources = {}
    for row in design:
        packet = SOURCE / "private_packets" / f"author_{row['index']:03d}.json"
        metadata = read(packet)
        arrays = packet.with_suffix(".npz")
        if metadata["arrays_sha256"] != sha(arrays):
            raise ValueError("Source boundary archive changed")
        sources[str(packet.relative_to(ROOT))] = sha(packet)
        sources[str(arrays.relative_to(ROOT))] = metadata["arrays_sha256"]
    files = [
        Path(__file__),
        PLAN,
        ROOT / "grain_research/vector_mechanisms/conditional_baseline.py",
        ROOT / "grain_research/vector_mechanisms/configuration_source.py",
        ROOT / "grain_research/vector_mechanisms/prefix.py",
        ROOT / "scripts/run_grain_level_response.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_CONDITIONAL_BASELINE_V1",
            "design_sha256": sha(out / "private_design.json"),
            "reference_sha256": sha(LEVEL / "reference.npz"),
            "model_sha256": read(SOURCE / "manifest.json")["model_sha256"],
            "sources": sources,
            "code_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "families": list(FAMILIES),
            "alpha": 1.0,
            "folds": 4,
            "max_seconds": 600,
            "source_outcome_sha256": sha(
                SOURCE / "private_anchor_source_connection.csv"
            ),
            "model_scores_sha256": sha(PREFIX / "private_scores_before_outcomes.csv"),
            "human_outcomes_for_fitting": False,
            "expected_prefix_suffix_gradient_calls": 0,
        },
    )
    print({"status": "PREPARED", "authors": 48, "views": 144, "folds": 4})


def verify(out):
    """Verify immutable construction inputs without exposing private source records."""
    manifest = read(out / "manifest.json")
    for path, digest in (
        (out / "private_design.json", manifest["design_sha256"]),
        (LEVEL / "reference.npz", manifest["reference_sha256"]),
        *[(ROOT / p, h) for p, h in manifest["code_hashes"].items()],
    ):
        if sha(path) != digest:
            raise ValueError("Frozen construction input changed")
    return manifest


def extract(out):
    """Execute only local QKV algebra on each cached boundary and freeze features."""
    m = verify(out)
    done = out / "extraction_status.json"
    if done.exists() and read(done)["status"] == "COMPLETE":
        print("Feature construction already complete")
        return
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Pinned checkpoint changed")
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        reference = dict(a)
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    now, old = observer.manifest(), read(SOURCE / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
    ):
        if now[key] != old[key]:
            raise ValueError("Frozen observer changed")
    write(out / "observer.json", now)
    handles, forbidden_calls = [], []

    def reject(name):
        """Fail immediately if any decoder block or scoring path is invoked."""

        def hook(_module, _args):
            forbidden_calls.append(name)
            raise RuntimeError("Forbidden forward in cached algebra: " + name)

        return hook

    modules = [("root", observer.model), ("decoder", observer.model.model)]
    modules.extend(
        (f"block_{i}", block) for i, block in enumerate(observer.model.model.layers)
    )
    block = observer.model.model.layers[17]
    modules.extend(
        [
            ("attention_module", block.self_attn),
            ("attention_output", block.self_attn.o_proj),
            ("layer_mlp", block.mlp),
            ("final_norm", observer.model.model.norm),
            ("lm_head", observer.model.lm_head),
        ]
    )
    for name, module in modules:
        handles.append(module.register_forward_pre_hook(reject(name)))
    start, count, projections, replay_error = time.monotonic(), 0, 0, 0.0
    try:
        for row in read(out / "private_design.json"):
            dst = out / "private_features" / f"author_{row['index']:03d}.npz"
            if dst.exists():
                count += 1
                continue
            if time.monotonic() - start > m["max_seconds"]:
                write(done, {"status": "PAUSED_RESOURCE_BOUNDARY", "authors": count})
                return
            packet = SOURCE / "private_packets" / dst.name
            if sha(packet) != m["sources"][str(packet.relative_to(ROOT))]:
                raise ValueError("Frozen cached boundary changed")
            query = row["query"]
            features = {family: [] for family in FAMILIES}
            mass, entropy = [], []
            with np.load(packet, allow_pickle=False) as arrays:
                for view, material in enumerate(row["views"]):
                    native = arrays[f"view_{view}"]
                    _, _, encoded, p = question_mask(
                        observer.tokenizer, material["texts"], query
                    )
                    if len(encoded["input_ids"]) != len(native):
                        raise ValueError("Cached state and fixed token layout differ")
                    e = ~p
                    e[-1] = False
                    anchored = native.copy()
                    anchored[p] = reference[f"{query}__Z"]
                    anchored[-1] = reference[f"{query}__h"]
                    cache = project_boundary(observer.model, anchored)
                    moments = attention_moments(cache, e)
                    error = float(
                        np.max(np.abs(moments["anchored_attention"] - full_head(cache)))
                    )
                    replay_error = max(replay_error, error)
                    if error > 1e-10:
                        raise ValueError(
                            "Moment and exact full-head reconstruction disagree"
                        )
                    features["evidence_mean"].append(native[e].mean(0))
                    features["native_PQ"].append(
                        np.concatenate([native[p].mean(0), native[-1]])
                    )
                    for family in FAMILIES[2:]:
                        features[family].append(moments[family])
                    mass.append(moments["mass"])
                    entropy.append(moments["entropy"])
                    projections += 1
            np.savez_compressed(dst, **{k: np.asarray(v) for k, v in features.items()})
            write(
                dst.with_suffix(".json"),
                {
                    "arrays_sha256": sha(dst),
                    "views": 3,
                    "mass_mean": float(np.mean(mass)),
                    "entropy_mean": float(np.mean(entropy)),
                },
            )
            count += 1
            if count % 8 == 0:
                print(
                    {
                        "status": "EXTRACTING",
                        "authors": count,
                        "seconds": time.monotonic() - start,
                    },
                    flush=True,
                )
        write(
            done,
            {
                "status": "COMPLETE",
                "authors": count,
                "qkv_projections_this_run": projections,
                "prefix_suffix_calls": observer.calls,
                "gradient_calls": 0,
                "forbidden_calls": forbidden_calls,
                "moment_head_max_error": replay_error,
                "seconds": time.monotonic() - start,
                "human_outcomes_read": False,
            },
        )
        print(read(done))
    finally:
        for handle in handles:
            handle.remove()


def fit(out):
    """Fit only measured b targets in fixed author folds and lock every prediction."""
    m = verify(out)
    if read(out / "extraction_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete feature bank required")
    design = read(out / "private_design.json")
    source_path = PREFIX / "private_scores_before_outcomes.csv"
    if sha(source_path) != m["model_scores_sha256"]:
        raise ValueError("Old model-score file changed")
    model_scores = pd.read_csv(source_path, dtype={"author_id": str}).set_index(
        ["author_id", "query"]
    )
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as a:
        refs = dict(a)
    outputs, metrics = [], []
    for query in ("TF_cont", "neuroticism"):
        rows = [r for r in design if r["query"] == query]
        targets, features = [], {family: [] for family in FAMILIES}
        for row in rows:
            packet = SOURCE / "private_packets" / f"author_{row['index']:03d}.json"
            if sha(packet) != m["sources"][str(packet.relative_to(ROOT))]:
                raise ValueError("Frozen model targets changed")
            targets.append(np.mean([v["anchor"] for v in read(packet)["views"]]))
            path = out / "private_features" / packet.with_suffix(".npz").name
            if sha(path) != read(path.with_suffix(".json"))["arrays_sha256"]:
                raise ValueError("Frozen features changed")
            with np.load(path, allow_pickle=False) as a:
                for family in FAMILIES:
                    if len(a[family]) != 3:
                        raise ValueError("Three fixed views required")
                    features[family].append(a[family].mean(0))
        targets = np.asarray(targets)
        folds = np.asarray([r["fold"] for r in rows])
        bref = float(refs[f"{query}__b"])
        reference_mse = float(np.mean((targets - bref) ** 2))
        for family in FAMILIES:
            prediction, constant = grouped_predictions(
                features[family], targets, folds, alpha=m["alpha"]
            )
            mse = float(np.mean((prediction - targets) ** 2))
            mean_mse = float(np.mean((constant - targets) ** 2))
            metrics.append(
                {
                    "query": query,
                    "method": family,
                    "authors": len(rows),
                    "features": len(features[family][0]),
                    "anchor_r": correlation(targets, prediction),
                    "anchor_rmse": float(np.sqrt(mse)),
                    "mse_over_fold_mean": mse / mean_mse,
                    "mse_over_old_ref": mse / reference_mse,
                }
            )
            for i, row in enumerate(rows):
                old = model_scores.loc[(row["author_id"], query)]
                outputs.append(
                    {
                        "author_id": row["author_id"],
                        "query": query,
                        "fold": row["fold"],
                        "method": family,
                        "anchor": float(targets[i]),
                        "anchor_hat": float(prediction[i]),
                        "fold_mean_anchor": float(constant[i]),
                        "base_ref": bref,
                        "PQ": float(old.PQ),
                        "native": float(old.native),
                        "constructed": float(prediction[i] + old.PQ - bref),
                    }
                )
    pd.DataFrame(outputs).to_csv(
        out / "private_predictions_before_outcomes.csv", index=False
    )
    pd.DataFrame(metrics).to_csv(out / "anchor_fidelity.csv", index=False)
    write(
        out / "predictions_frozen.json",
        {
            "sha256": sha(out / "private_predictions_before_outcomes.csv"),
            "target": "measured conditional model anchor only",
            "human_labels_used": False,
            "families": list(FAMILIES),
            "alpha": m["alpha"],
            "selection": "none; primary anchored_attention fixed in plan",
        },
    )
    print(pd.DataFrame(metrics).to_string(index=False))


def evaluate(out):
    """Join exposed source Y after freezing, preserving all functions and components."""
    m = verify(out)
    path = out / "private_predictions_before_outcomes.csv"
    if sha(path) != read(out / "predictions_frozen.json")["sha256"]:
        raise ValueError("Frozen predictions changed")
    source_path = SOURCE / "private_anchor_source_connection.csv"
    if sha(source_path) != m["source_outcome_sha256"]:
        raise ValueError("Exposed source link changed")
    frame = pd.read_csv(path, dtype={"author_id": str})
    source = pd.read_csv(source_path, dtype={"author_id": str})
    frame = frame.merge(
        source[
            [
                "author_id",
                "query",
                "y_true",
                "configuration_response",
                "transport_remainder",
            ]
        ],
        on=["author_id", "query"],
        validate="many_to_one",
    )
    if len(frame) != 48 * len(FAMILIES):
        raise ValueError("Complete source-only connection required")
    frame["baseline_error"] = frame.anchor - frame.anchor_hat
    frame["new_remainder"] = frame.native - frame.constructed
    if not np.allclose(
        frame.new_remainder,
        frame.baseline_error + frame.transport_remainder,
        atol=1e-12,
    ):
        raise ValueError("Conditional function ledger does not close")
    rows, ledger = [], []
    for (query, method), g in frame.groupby(["query", "method"]):
        for name in ("PQ", "native", "constructed"):
            metric = scalar_metrics(
                g.y_true.to_numpy(), g[name].to_numpy(), binary=query == "TF_cont"
            )
            metric.update(
                query=query,
                method=method,
                readout=name,
                native_fidelity=correlation(g.native.to_numpy(), g[name].to_numpy()),
            )
            rows.append(metric)
        for name in (
            "PQ",
            "anchor_hat",
            "constructed",
            "baseline_error",
            "transport_remainder",
            "new_remainder",
        ):
            ledger.append(
                {
                    "query": query,
                    "method": method,
                    "component": name,
                    "r_y": correlation(g.y_true.to_numpy(), g[name].to_numpy()),
                    "cov_y": covariance(g.y_true.to_numpy(), g[name].to_numpy()),
                }
            )
    frame.to_csv(out / "private_source_connection.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "source_metrics.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "covariance_ledger.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "scope": "24 source-exposed authors/query; frozen four-fold model-anchor construction, not new human-label confirmation",
            "cost": read(out / "extraction_status.json"),
            "anchor_metrics": pd.read_csv(out / "anchor_fidelity.csv").to_dict(
                "records"
            ),
            "source_metrics": rows,
            "ledger": ledger,
        },
    )
    print(
        pd.DataFrame(rows)[
            ["query", "method", "readout", "pearson_r", "native_fidelity"]
        ].to_string(index=False)
    )


def main():
    """Keep feature extraction, model-target fitting and source linkage distinct."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "extract", "fit", "evaluate"))
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "conditional_baseline_construction_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError(
            "Private results must stay under the ignored research directory"
        )
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    {"prepare": prepare, "extract": extract, "fit": fit, "evaluate": evaluate}[
        args.phase
    ](out)


if __name__ == "__main__":
    main()
