#!/usr/bin/env python3
"""Reconstruct existing source references and cold-score permitted cached vectors.

No new instruments, labels, raw texts or encoder calls. This audits the actual
observed-expression reference, not Gaussian T1/T2/T3 or psychological validity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from scripts.score_grain_conditional_reference import restore_reference, score_packet


def rebuild_reference(conditions, scores, authors, source):
    """Independently reconstruct author-balanced source cells using group sums."""
    source = np.asarray(source, int)
    if len(source) != len(set(source)) or not len(source):
        raise ValueError("unique nonempty source required")
    mask = np.isin(authors, source)
    frame = pd.DataFrame(np.asarray(scores)[mask])
    features = list(frame.columns)
    frame["author"] = np.asarray(authors)[mask]
    frame["condition"] = np.asarray(conditions)[mask]
    if set(frame.author) != set(source):
        raise ValueError("missing source author")
    groups = frame.groupby(["author", "condition"], sort=True)
    counts = groups.size()
    cell_means = groups[features].mean()
    weights = counts / counts.groupby(level="author").transform("sum") / len(source)
    mass = weights.groupby(level="condition").sum().sort_index()
    mass /= mass.sum()
    means = cell_means.groupby(level="condition").mean().reindex(mass.index)
    support = cell_means.groupby(level="condition").size().reindex(mass.index)
    keys = mass.index.tolist()
    core = np.zeros(len(keys))
    top = sorted(range(len(keys)), key=lambda i: (-support.iloc[i], keys[i]))[:5]
    core[top] = mass.iloc[top] / mass.iloc[top].sum()
    return {
        "keys": keys,
        "weights": mass.to_numpy(),
        "mean": means.to_numpy(),
        "source_authors_per_cell": support.to_numpy(),
        "common5_weights": core,
        "source_indices": source,
    }


def compare_reference(rebuilt, archived) -> dict:
    """Fail on source membership, condition identity, or numerical reference drift."""
    if rebuilt["keys"] != archived["keys"]:
        raise ValueError("condition identity mismatch")
    errors = {}
    for key in (
        "weights",
        "mean",
        "common5_weights",
        "source_authors_per_cell",
        "source_indices",
    ):
        if np.asarray(rebuilt[key]).shape != np.asarray(archived[key]).shape:
            raise ValueError(f"reference shape mismatch: {key}")
        error = float(np.max(np.abs(rebuilt[key] - archived[key])))
        if error > 1e-12:
            raise ValueError(f"source reference mismatch: {key}")
        errors[key] = error
    return errors


def main() -> None:
    """Verify all units and held profiles, with one isolated cold input per corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1"
    reference_root = base / "conditional_reference_r2"
    directions = reference_root / "private_directions.npz"
    with np.load(directions, allow_pickle=False) as a:
        direction, targets = a["direction"], a["targets"]
    rows, inputs = [], {str(directions): sha256_file(directions)}
    for domain in ("pandora", "essays"):
        root = reference_root / domain
        local = base / "local_bilinear_r2" / domain
        required = [
            root / "private_reference.json",
            root / "private_units.npz",
            root / "private_scores_full.npz",
            root / "private_scores_common5.npz",
            local / "private_units.npz",
        ]
        inputs.update({str(p): sha256_file(p) for p in required})
        archived = restore_reference(json.loads(required[0].read_text()))
        with np.load(required[1], allow_pickle=False) as a:
            scores, authors, conditions, source, held = [
                a[k] for k in ("scores", "authors", "conditions", "source", "held")
            ]
        with np.load(required[4], allow_pickle=False) as a:
            h = a["h"].astype(float)
        norms = np.linalg.norm(h, axis=1)
        if not np.isfinite(h).all() or np.any(norms < 1e-12):
            raise ValueError("invalid raw encoder coordinates")
        replay = (h / norms[:, None]) @ direction
        vector_error = float(np.max(np.abs(scores - replay)))
        if vector_error > 1e-12:
            raise ValueError("frozen vector normalization/projection mismatch")
        ref_errors = compare_reference(
            rebuild_reference(conditions, scores, authors, source), archived
        )
        cold_error, max_score_error, checked = 0.0, 0.0, 0
        coverage = {}
        for target in ("full", "common5"):
            with np.load(
                root / f"private_scores_{target}.npz", allow_pickle=False
            ) as a:
                previous = {k: a[k] for k in a.files}
            values, complete = [], 0
            for u in held:
                ix = authors == u
                s = score_packet(
                    conditions[ix], h[ix], archived, direction, targets, target
                )
                for key in (
                    "actual",
                    "context",
                    "departure",
                    "unseen",
                    "center",
                    "reference_target_mean",
                    "lower",
                    "upper",
                    "coverage",
                    "missing",
                    "unknown_material_mass",
                ):
                    error = float(np.max(np.abs(np.asarray(s[key]) - previous[key][u])))
                    max_score_error = max(max_score_error, error)
                if (
                    s["complete"] != bool(previous["complete"][u])
                    or (s["point"] is None) == s["complete"]
                ):
                    raise ValueError(
                        "unsupported point invented or supported point omitted"
                    )
                values.append(s["coverage"])
                complete += int(s["complete"])
                checked += 1
            coverage[target] = {
                "median": float(np.median(values)),
                "complete_held_authors": complete,
            }
        if max_score_error > 1e-10:
            raise ValueError(
                "archived held score differs from cached-vector cold interface"
            )
        # Temporary standalone inputs keep IDs/text/source data outside this process.
        with tempfile.TemporaryDirectory(prefix="grain-f14-cold-") as temp:
            temp = Path(temp)
            ref_path, d_path, x_path, out_path = [
                temp / name
                for name in (
                    "reference.json",
                    "directions.npz",
                    "input.npz",
                    "output.json",
                )
            ]
            ref_path.write_bytes(required[0].read_bytes())
            d_path.write_bytes(directions.read_bytes())
            ix = authors == held[0]
            np.savez_compressed(x_path, conditions=conditions[ix], vectors=h[ix])
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/score_grain_conditional_reference.py"),
                    "--reference",
                    str(ref_path),
                    "--directions",
                    str(d_path),
                    "--input",
                    str(x_path),
                    "--output",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            cold = json.loads(out_path.read_text())
            warm = score_packet(conditions[ix], h[ix], archived, direction, targets)
            for key in warm:
                if cold[key] != warm[key]:
                    raise ValueError(f"cold output mismatch: {key}")
            if proc.returncode:
                raise ValueError("cold process failed")
        rows.append(
            {
                "dataset": domain,
                "units": len(authors),
                "source_authors": len(source),
                "held_authors": len(held),
                "source_reference_errors": ref_errors,
                "normalized_vector_readout_max_error": vector_error,
                "held_target_objects": checked,
                "score_replay_max_error": max_score_error,
                "cold_processes": 1,
                "cold_full_object_error": cold_error,
                "coverage": coverage,
            }
        )
    result = {
        "status": "SOURCE_REFERENCE_AND_CACHED_VECTOR_REPLAY_COMPLETE",
        "datasets": rows,
        "input_hashes": inputs,
        "scope": "Existing conditional observed-expression interface only; cached vectors, not new raw-text reading or T1/T2/T3 integration",
        "bounds": "Unit-ball completion envelopes, not psychological confidence intervals",
        "new_fits_or_model_calls": 0,
        "new_text_or_questionnaire_values": False,
        "F14_parent_complete": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": result["status"], "datasets": rows}), flush=True)


if __name__ == "__main__":
    main()
