#!/usr/bin/env python3
"""CON03: model-only selection, portable replay, and one frozen external pass."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_runtime import ConditionalRuntime
from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    quality,
    read_json,
    write_json,
)
from scripts.analyze_grain_conditional_navigation import NATURAL, ROUTES
from scripts.analyze_grain_state_transport import external_quality
from scripts.export_grain_navigation_runtime import replay
from scripts.fit_grain_conditional_navigation import DEFAULT as FIT
from scripts.run_grain_conditional_augmentation import DEFAULT as AUG
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_source_span_extension import DEFAULT as SPAN
from scripts.run_grain_state_transport import input_units
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = BASE / "conditional_navigation_v1/CON-03/runtime_external_r1"


def qualification(folder, input_kind):
    """Use matched source and model responses only; no external columns are read."""
    s = pd.read_csv(folder / "source_response_metrics.csv").query(
        "target=='native' and split=='author_holdout'"
    )
    n = pd.read_csv(folder / "natural_response_metrics.csv").query("target=='native'")
    d = pd.read_csv(folder / "paired_error_intervals.csv")
    keys = ["query", "method", "axis"]
    x = (
        s[keys + ["r", "mse_over_zero"]]
        .merge(
            n[keys + ["r", "mse_over_zero"]],
            on=keys,
            suffixes=("_source", "_natural"),
            validate="one_to_one",
        )
        .merge(d[keys + ["delta_mse_vs_A_source"]], on=keys, validate="one_to_one")
    )
    x["input_kind"] = input_kind
    x["eligible"] = (
        (x.r_source > 0)
        & (x.mse_over_zero_source < 1)
        & (x.r_natural > 0)
        & (x.delta_mse_vs_A_source < 0)
    )
    return x


def prepare(out):
    """Freeze every qualifying program, not a human-label-selected winner."""
    if out.exists() or not out.is_relative_to(
        (BASE / "conditional_navigation_v1").resolve()
    ):
        raise ValueError("Choose a fresh ignored export directory")
    decision = pd.concat(
        [qualification(FIT, "PQ"), qualification(AUG / "fit", "PQ_plus_evidence")],
        ignore_index=True,
    )
    eligible = decision[decision.eligible]
    # E's one fixed augmentation is a failed candidate in this completed packet.
    # A future passing augmented query needs a new raw-excerpt external manifest.
    if (eligible.input_kind != "PQ").any():
        raise ValueError(
            "Do not silently run an augmented query on PQ-only external inputs"
        )
    selection = {q: sorted(g.method.unique()) for q, g in eligible.groupby("query")}
    if not selection:
        raise ValueError(
            "No useful program; record non-execution rather than open external labels"
        )
    frozen = read_json(FIT / "construction_freeze.json")
    if any(digest(FIT / n) != sha for n, sha in frozen["files"].items()):
        raise ValueError("Source fit changed")
    out.mkdir(parents=True)
    decision.to_csv(out / "selection_evidence.csv", index=False)
    selection_sources = {
        str((p / n).relative_to(ROOT)): digest(p / n)
        for p in (FIT, AUG / "fit")
        for n in (
            "construction_freeze.json",
            "source_response_metrics.csv",
            "natural_response_metrics.csv",
            "paired_error_intervals.csv",
        )
    }
    write_json(
        out / "selection_freeze.json",
        {
            "selection": selection,
            "sources": selection_sources,
            "rule": "same-axis positive source r and MSE/zero<1, positive exposed-native r and MSE better than A_source",
            "external_labels_or_scores_accessed": False,
            "human_selection": False,
            "code_sha256": digest(Path(__file__)),
        },
    )
    bundle = out / "bundle"
    (bundle / "compiled").mkdir(parents=True)
    (bundle / "grain_runtime").mkdir()
    original = BASE / "navigation_v1/NAV-11/runtime_export_r1/bundle"
    original_m = read_json(original / "manifest.json")
    file_names = []
    for name in (
        "context.safetensors",
        "config.json",
        "tokenizer.json",
        "queries.json",
    ):
        if digest(original / name) != original_m["files"][name]:
            raise ValueError("Frozen neural scaffold changed")
        shutil.copy2(original / name, bundle / name)
        file_names.append(name)
    modules = (
        "__init__.py",
        "conditional_runtime.py",
        "conditional_models.py",
        "conditional_inputs.py",
        "navigation_runtime.py",
        "context_program.py",
        "projection_program.py",
        "state_transport.py",
    )
    for name in modules:
        dst = "grain_runtime/" + name
        shutil.copy2(ROOT / "grain_research/vector_mechanisms" / name, bundle / dst)
        file_names.append(dst)
    shutil.copy2(ROOT / "scripts/run_grain_conditional_runtime.py", bundle / "run.py")
    file_names.append("run.py")
    for q, methods in selection.items():
        dst = f"compiled/inputs_{q}.npz"
        shutil.copy2(FIT / f"private_inputs_{q}.npz", bundle / dst)
        file_names.append(dst)
        for method in methods:
            dst = f"compiled/{q}_{method}.npz"
            shutil.copy2(FIT / f"program_{q}_{method}.npz", bundle / dst)
            file_names.append(dst)
    write_json(
        bundle / "manifest.json",
        {
            "protocol": "CON03_CONDITIONAL_RUNTIME_V1",
            "selection": selection,
            "files": {n: digest(bundle / n) for n in file_names},
            "selection_sha256": digest(out / "selection_freeze.json"),
            "runtime_layers": 4,
            "full_teacher": False,
            "human_labels_fit": False,
            "private_author_derived_anchors": True,
            "public_redistribution_authorized": False,
            "installed_versions": {
                n: importlib.metadata.version(n)
                for n in (
                    "numpy",
                    "scipy",
                    "torch",
                    "transformers",
                    "tokenizers",
                    "safetensors",
                    "threadpoolctl",
                )
            },
        },
    )
    predictions = pd.read_csv(FIT / "private_natural_predictions.csv")
    if (
        digest(FIT / "private_natural_predictions.csv")
        != read_json(FIT / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("Expected frozen scores changed")
    worlds = read_json(NATURAL / "private_worlds.json")
    author = worlds[0]["author_id"]
    inputs, expected = [], []
    for w in worlds:
        if w["author_id"] != author:
            continue
        for q, methods in selection.items():
            identity = w["world_id"] + "::" + q
            inputs.append({"id": identity, "query": q, "texts": w["texts"]})
            for method in methods:
                g = predictions[
                    (predictions.world_id == w["world_id"])
                    & (predictions["query"] == q)
                    & (predictions.method == method)
                ]
                if len(g) != 1:
                    raise ValueError("Missing replay score")
                expected.append(
                    {
                        "id": identity + "::" + method,
                        "program": "conditional",
                        "score": float(g.iloc[0].score),
                    }
                )
    (out / "private_inputs.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in inputs),
        encoding="utf-8",
    )
    write_json(out / "expected.json", expected)
    write_json(
        out / "export.json",
        {
            "status": "EXPORTED",
            "bundle_manifest_sha256": digest(bundle / "manifest.json"),
            "input_sha256": digest(out / "private_inputs.jsonl"),
            "expected_sha256": digest(out / "expected.json"),
            "replay_rows_per_program": len(expected),
            "replay_forwards": len(inputs),
            "absolute_score_tolerance": 1e-5,
            "human_labels_fit": False,
        },
    )
    print(
        {
            "status": "EXPORTED",
            "selection": selection,
            "replay_scores": len(expected),
            "replay_forwards": len(inputs),
        },
        flush=True,
    )


def collect(out, seconds):
    """One exact-membership external traversal; all selected variants share activations."""
    if read_json(out / "replay_conditional/summary.json")["status"] != "PASS":
        raise ValueError("Independent raw runtime must pass first")
    bundle = out / "bundle"
    manifest = read_json(bundle / "manifest.json")
    units, hashes = input_units("external")
    refs = set(read_json(SPAN / "private_membership.json")["reference_authors"])
    selected = [
        u
        for u in units
        if u["query"] in manifest["selection"] and u["author_id"] not in refs
    ]
    if len(selected) != 156 or sum(len(u["views"]) for u in selected) != 284:
        raise ValueError("This packet expects64 TF +46 chat +46 task units, no E")
    dest = out / "external"
    expected = {
        "bundle_sha256": digest(bundle / "manifest.json"),
        "selection_sha256": digest(out / "selection_freeze.json"),
        "input_sha256": hashes,
        "unit_count": len(selected),
        "view_count": 284,
        "code_sha256": digest(Path(__file__)),
        "labels_before_freeze": False,
        "human_fit": False,
        "MEPS_fit": False,
    }
    if not dest.exists():
        (dest / "private_units").mkdir(parents=True)
        write_json(dest / "manifest.json", expected)
        write_json(
            dest / "run_status.json", {"status": "PREPARED", "calls": 0, "seconds": 0}
        )
    elif read_json(dest / "manifest.json") != expected:
        raise ValueError("External manifest changed")
    prior = read_json(dest / "run_status.json")
    if prior["status"] == "COMPLETE":
        return
    runtime = ConditionalRuntime(bundle)
    start, rows = time.monotonic(), []
    for u in selected:
        path = dest / "private_units" / f"unit_{u['index']:04d}.json"
        identity = {k: u[k] for k in ("index", "dataset", "author_id", "query")}
        if path.exists():
            meta, z = checked_arrays(path)
            if meta["identity"] != identity:
                raise ValueError("External unit changed")
        else:
            if time.monotonic() - start >= seconds:
                break
            values = [
                runtime.score_ids(v["input_ids"], v["question_mask"], u["query"])
                for v in u["views"]
            ]
            names = list(runtime.models[u["query"]])
            array = np.asarray([[v[n] for n in names] for v in values])
            save_unit(path, {"identity": identity, "methods": names}, scores=array)
            meta, z = checked_arrays(path)
        for name, score in zip(meta["methods"], z["scores"].mean(0), strict=True):
            rows.append(identity | {"method": name, "score": float(score)})
        if len(rows) % 72 == 0:
            print(
                {
                    "phase": "EXTERNAL",
                    "score_rows": len(rows),
                    "new_calls": runtime.calls,
                    "seconds": round(time.monotonic() - start, 2),
                },
                flush=True,
            )
    complete = len(rows) == sum(
        len(manifest["selection"][u["query"]]) for u in selected
    )
    status = {
        "status": "COMPLETE" if complete else "PAUSED",
        "score_rows": len(rows),
        "calls": prior["calls"] + runtime.calls,
        "seconds": prior["seconds"] + time.monotonic() - start,
    }
    write_json(dest / "run_status.json", status)
    if complete:
        pd.DataFrame(rows).to_csv(dest / "predictions.csv", index=False)
        write_json(
            dest / "prediction_freeze.json",
            {
                "sha256": digest(dest / "predictions.csv"),
                "manifest_sha256": digest(dest / "manifest.json"),
                "human_labels_joined": False,
            },
        )
        write_json(dest / "dependencies.json", runtime.dependency_report())
    print(status, flush=True)


def analyze(out):
    """Join exposed external criteria only after prediction and program freezes."""
    dest = out / "external"
    path = dest / "predictions.csv"
    if (
        read_json(dest / "run_status.json")["status"] != "COMPLETE"
        or digest(path) != read_json(dest / "prediction_freeze.json")["sha256"]
    ):
        raise ValueError("External predictions not complete/frozen")
    analysis = dest / "analysis"
    if analysis.exists():
        raise ValueError("Do not rerun external selection")
    frame = pd.read_csv(path, dtype={"author_id": str})
    keys = ["dataset", "author_id", "query"]
    member = set(map(tuple, frame[keys].to_numpy()))
    cp = ROUTES / "external/predictions.csv"
    if digest(cp) != read_json(cp.parent / "prediction_freeze.json")["sha256"]:
        raise ValueError("Frozen controls changed")
    controls = pd.read_csv(cp, dtype={"author_id": str}).query("reference=='R_synth'")
    controls = controls[[tuple(v) in member for v in controls[keys].to_numpy()]]
    for method, g in controls.groupby("method"):
        if set(map(tuple, g[keys].to_numpy())) != member:
            raise ValueError("Control membership differs")
    frame = pd.concat([frame, controls.drop(columns="reference")], ignore_index=True)
    ref = BASE / "navigation_v1/NAV-10/reference_transfer_r1"
    source = pd.read_csv(
        ref / "private_source_external.csv", dtype={"author_id": str}
    ).query("reference=='R_synth'")
    mood = pd.read_csv(
        ref / "private_counseling_predictions.csv", dtype={"participant_code": str}
    ).query("reference=='R_synth'")
    joined = frame[frame.dataset == "PANDORA_source"].merge(
        source[["author_id", "query", "score", "y_true", "prediction"]].rename(
            columns={"score": "full_function", "prediction": "native"}
        ),
        on=["author_id", "query"],
        validate="many_to_one",
    )
    chat = frame[frame.dataset != "PANDORA_source"].merge(
        mood[["participant_code", "view", "query", "score", "native_score"]].rename(
            columns={
                "participant_code": "author_id",
                "view": "dataset",
                "score": "full_function",
                "native_score": "native",
            }
        ),
        on=keys,
        validate="many_to_one",
    )
    criterion = BASE / "counseling_mood_m30c_v1/private_mood_scores.csv"
    if (
        digest(criterion)
        != read_json(criterion.parent / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("External labels changed")
    labels = pd.read_csv(criterion, dtype={"participant_code": str}).pivot(
        index="participant_code", columns="scale", values="score"
    )
    chat["y_true"] = [
        (
            float(
                labels.loc[
                    r.author_id,
                    f"{'post' if r.dataset == 'free_chat' else 'mid'}_adapted_dejected5",
                ]
            )
            - 1
        )
        / 5
        for r in chat.itertuples()
    ]
    all_rows = pd.concat([joined, chat], ignore_index=True)
    if (
        len(all_rows) != len(frame)
        or all_rows.duplicated(keys + ["method"]).any()
        or all_rows[["score", "native", "full_function", "y_true"]].isna().any().any()
    ):
        raise ValueError("External join mismatch")
    metrics, paired = [], []
    for (dataset, q, method), g in all_rows.groupby(["dataset", "query", "method"]):
        metrics.append(
            {
                "dataset": dataset,
                "query": q,
                "method": method,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **external_quality(g.score, g.native, g.y_true, binary=q == "TF_cont"),
            }
        )
    for (dataset, q), g in all_rows.groupby(["dataset", "query"]):
        pivot = g.pivot(index="author_id", columns="method", values="score")
        target = (
            g.drop_duplicates("author_id")
            .set_index("author_id")
            .reindex(pivot.index)
            .y_true
        )
        for method in read_json(out / "bundle/manifest.json")["selection"][q]:
            for control in ("A_source", "B_token", "B_context4"):
                paired.append(
                    {
                        "dataset": dataset,
                        "query": q,
                        "method": method,
                        "control": control,
                        **external_quality(
                            pivot[method], pivot[control], target, binary=q == "TF_cont"
                        ),
                    }
                )
    analysis.mkdir()
    all_rows.to_csv(analysis / "private_joined.csv", index=False)
    pd.DataFrame(metrics).to_csv(analysis / "external_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(analysis / "paired_comparisons.csv", index=False)
    write_json(
        analysis / "summary.json",
        {
            "status": "COMPLETE",
            "prediction_sha256": digest(path),
            "control_sha256": digest(cp),
            "criterion_sha256": digest(criterion),
            "source_reference_sha256": digest(ref / "private_source_external.csv"),
            "mood_reference_sha256": digest(ref / "private_counseling_predictions.csv"),
            "models_or_signs_selected_from_external": False,
            "code_sha256": digest(Path(__file__)),
            "membership": "64 TF;46 freechat;46 task; previously exposed criteria, not fresh participants",
        },
    )
    print(
        pd.DataFrame(metrics)[
            [
                "dataset",
                "query",
                "method",
                "n",
                "full_function_r",
                "function_external_r",
                "function_auc",
            ]
        ]
        .round(5)
        .to_string(index=False),
        flush=True,
    )


def main():
    """Execute frozen export, independent replay, one collection, or post-freeze analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "replay", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--seconds", type=int, default=600)
    a = parser.parse_args()
    out = a.out.resolve()
    if not out.is_relative_to((BASE / "conditional_navigation_v1").resolve()):
        raise ValueError("Use private ignored artifacts")
    with threadpool_limits(limits=2):
        if a.action == "prepare":
            prepare(out)
        elif a.action == "replay":
            replay(out, "conditional")
        elif a.action == "collect":
            with collector_guard(out / "external"):
                collect(out, a.seconds)
        else:
            analyze(out)


if __name__ == "__main__":
    main()
