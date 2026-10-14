#!/usr/bin/env python3
"""Fresh-process raw-text replay and arithmetic audit of the semantic link."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from grain_research.semantic_link import opportunity_numeric, validate_addresses

BASE = ROOT / "results/grain_vector_mechanisms"
DEFAULT = BASE / "predecessor_integration_v1/material"


def run(out):
    """Reconstruct scoring without teacher caches, then independently check summaries."""
    summary = json.loads((out / "summary.json").read_text())
    contract = json.loads((out / "contract.json").read_text())
    for name, sha in summary["artifacts"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed output artifact: " + name)
    for path, sha in (
        (ROOT / "grain_research/semantic_link.py", contract["module_sha256"]),
        (ROOT / "scripts/run_grain_semantic_link.py", contract["script_sha256"]),
        (
            ROOT / "docs/GRAIN_SAME_ADDRESS_SEMANTIC_LINK_PLAN_V1.md",
            contract["plan_sha256"],
        ),
    ):
        if digest(path) != sha:
            raise ValueError("Changed frozen source: " + str(path))
    cold = {"active": True, "denied": 0}

    def guard(event, args):
        """Forbid access to teacher/target arrays during actual raw-text inference."""
        if event == "open" and cold["active"] and isinstance(args[0], (str, bytes)):
            name = str(args[0])
            if any(
                token in name
                for token in (
                    "paraphrase_cache",
                    "private_arrays.npz",
                    "private_inputs.npz",
                    "private_held_predictions.npz",
                )
            ):
                cold["denied"] += 1
                raise PermissionError("Teacher/target access blocked in cold inference")

    sys.addaudithook(guard)
    try:
        (DEFAULT / "private_arrays.npz").open("rb")
    except PermissionError:
        pass
    else:
        raise ValueError("Cold inference guard failed")
    probe_denials = cold["denied"]
    start = time.monotonic()
    data = json.loads((DEFAULT / "private_material.json").read_text())
    comments = validate_addresses(data)
    indices = [
        i
        for i, s in enumerate(data["sentences"])
        if data["bundles"][s["author_index"]]["official_folds"]["big5"] == 0
    ]
    sentences = [data["sentences"][i] for i in indices]
    runtime_path = (
        ROOT
        / "results/v7_multiview_projection/e1_v72_full_20260715/artifacts/common_source_comment_representation.joblib"
    )
    if digest(runtime_path) != contract["runtime_sha256"]:
        raise ValueError("Frozen runtime changed")
    runtime = joblib.load(runtime_path)
    with threadpool_limits(limits=2):
        g = runtime.transform([s["text"] for s in sentences])
        z = opportunity_numeric(sentences, comments)
        sub = np.asarray(
            [
                comments[s["author_index"]][s["comment_index"]]["subreddit"]
                for s in sentences
            ],
            str,
        )
        models = list(contract["models"]) + ["permuted_source"]
        outputs = {
            mode: joblib.load(out / f"{mode}.joblib").predict(g, z, sub)
            for mode in models
        }
    elapsed = time.monotonic() - start
    if cold["denied"] != probe_denials:
        raise ValueError("Scoring tried to read a teacher/target cache")
    loaded_neural = [
        m
        for m in ("torch", "transformers", "sentence_transformers")
        if m in sys.modules
    ]
    cold["active"] = False
    errors = []
    with np.load(out / "private_held_predictions.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(indices, arrays["sentence_index"])
        y, authors = arrays["target"], arrays["authors"]
        for mode in models:
            errors.append(float(np.max(abs(outputs[mode] - arrays[mode]))))
    if max(errors) > 1e-10:
        raise ValueError("Raw-text prediction replay differs")
    # Independently recompute author-mean correlations and author-balanced losses.
    expected = pd.read_csv(out / "affinity_by_facet.csv")
    scale = joblib.load(out / "joint_linear.joblib").y_scaler_.scale_
    ids = sorted(set(authors))
    checks = []
    for mode in models:
        pred = outputs[mode]
        am_y = np.stack([y[authors == a].mean(0) for a in ids])
        am_p = np.stack([pred[authors == a].mean(0) for a in ids])
        am_mse = np.stack(
            [((y[authors == a] - pred[authors == a]) ** 2).mean(0) for a in ids]
        )
        table = expected[expected.model == mode].set_index(["function", "facet"])
        for j, name in enumerate(contract["target_names"]):
            reference = table.loc[tuple(name.split(":"))]
            ar = np.corrcoef(am_y[:, j], am_p[:, j])[0, 1]
            checks.extend(
                [
                    abs(ar - reference.author_affinity_r),
                    abs(am_mse[:, j].mean() - reference.mse),
                    abs(am_mse[:, j].mean() / scale[j] ** 2 - reference.normalized_mse),
                ]
            )
    if np.nanmax(checks) > 1e-10:
        raise ValueError("Independent metric arithmetic differs")
    result = {
        "status": "PASS",
        "scope": "fresh-process raw-text replay plus independent arithmetic; not independent researchers or semantic validity",
        "held_sentences": len(indices),
        "held_authors": len(ids),
        "models_replayed": models,
        "raw_replay_max_error": max(errors),
        "independent_metric_max_error": float(np.nanmax(checks)),
        "cold_raw_scoring_wall_seconds": elapsed,
        "teacher_guard_probe_denials": probe_denials,
        "teacher_access_during_scoring": cold["denied"] - probe_denials,
        "neural_libraries_loaded_during_scoring": loaded_neural,
        "run_summary_sha256": digest(out / "summary.json"),
        "audit_script_sha256": digest(Path(__file__)),
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/semantic_link_r2"
    )
    run(parser.parse_args().output)
