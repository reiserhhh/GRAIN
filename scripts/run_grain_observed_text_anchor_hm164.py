#!/usr/bin/env python3
"""Connect cached vectors to actually observed expression, without new model calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observed_text_anchor as M
from grain_research.construction.math_semantic_anchor import tensor_features

DEFAULT = ROOT / "results/grain_math_response_v1/observed_text_anchor_r1"
CONFIG = ROOT / "configs/grain_observed_text_anchor_hm164.json"
SCIENCE = (
    "grain_research/construction/math_observed_text_anchor.py",
    "grain_research/construction/math_semantic_anchor.py",
    "scripts/run_grain_observed_text_anchor_hm164.py",
    "configs/grain_observed_text_anchor_hm164.json",
    "tests/test_grain_observed_text_anchor.py",
    "docs/GRAIN_OBSERVED_TEXT_ANCHOR_PLAN_HM164.md",
)


def digest(path):
    """Hash a scientific input without altering its original provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    """Read explicitly registered public generated materials or numeric metadata."""
    return json.loads(path.read_text())


def write(path, value):
    """Create one new artifact; do not overwrite scientific outputs."""
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write("\n")


def table(path, rows):
    """Preserve all descriptive cells in an inspectable table."""
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def authenticated(config):
    """Align all702 actual articles to vectors, checking historical seals first."""
    source = ROOT / config["source"]
    for filename in ("generation_seal.json", "observation_seal.json"):
        for name, expected in read(source / filename).items():
            if digest(source / name) != expected:
                raise ValueError("Historical input changed: " + name)
    for filename in ("generation_status.json", "observation_status.json"):
        if read(source / filename)["status"] != "COMPLETE":
            raise ValueError("Incomplete historical input")
    rows = read(source / "materials.json")
    if (
        digest(source / "materials.json")
        != read(source / "manifest.json")["materials_sha256"]
    ):
        raise ValueError("Historical design seal mismatch")
    records = [
        r
        for path in sorted((source / "batches").glob("[0-9][0-9][0-9].json"))
        for r in read(path)
    ]
    if len(rows) != 702 or [r["text_id"] for r in rows] != [
        r["text_id"] for r in records
    ]:
        raise ValueError("All actual articles must match vector addresses")
    expected_tasks = set(
        config["source_tasks"]
        + config["development_tasks"]
        + config["application_tasks"]
    )
    if {r["situation"] for r in rows} != expected_tasks:
        raise ValueError("Unregistered task scope")
    return source, rows, records


def prepare(out):
    """Seal new observation rules and old inputs before any new outcome analysis."""
    config = read(CONFIG)
    source, rows, _ = authenticated(config)
    inputs = [
        source / name
        for name in (
            "manifest.json",
            "materials.json",
            "generation_seal.json",
            "observation_seal.json",
            "numeric_observations.npz",
        )
    ] + [ROOT / config["grain_units"]]
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "packet": "HM164",
            "rows": len(rows),
            "config": config,
            "science_hashes": {name: digest(ROOT / name) for name in SCIENCE},
            "input_hashes": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
            "stage": "SEALED_BEFORE_ANALYSIS",
            "human_observations": 0,
            "new_neural_forwards": 0,
            "private_inputs": 0,
        },
    )
    print(json.dumps({"prepared": str(out), "rows": len(rows)}))


def check(out):
    """Reject post-seal scientific edits and changed historical artifacts."""
    manifest = read(out / "manifest.json")
    for section in ("science_hashes", "input_hashes"):
        for name, expected in manifest[section].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Sealed file changed: " + name)
    return manifest


def feature_views(config, source_mask):
    """Use cached public article vectors; never load an encoder or new text model."""
    with np.load(
        ROOT / config["source"] / "numeric_observations.npz", allow_pickle=False
    ) as data:
        raw = {
            key: data[key].astype(float)
            for key in (
                "qwen_input_mean",
                "qwen_block16_mean",
                "grain4_slices",
            )
        }
    result, units = {}, {}
    for key in ("qwen_input_mean", "qwen_block16_mean"):
        x = raw[key]
        if not np.isfinite(x).all() or len(x) != len(source_mask):
            raise ValueError("Incomplete vector observations: " + key)
        center = x[source_mask].mean(0)
        scale = float(np.sqrt(np.mean(np.sum((x[source_mask] - center) ** 2, axis=1))))
        if scale <= 0:
            raise ValueError("No source representation support")
        result[key] = (x - center) / scale
        units[key] = {"center": center, "scale": scale}
    with np.load(ROOT / config["grain_units"], allow_pickle=False) as old:
        z = (raw["grain4_slices"] - old["origin"]) / old["scale"][None, :, None]
        if not np.isfinite(z).all():
            raise ValueError("Incomplete GRAIN slices; do not silently drop articles")
        result["grain_tensor625"] = tensor_features(z)
        units["grain_tensor625"] = {"center": old["origin"], "scale": old["scale"]}
    return result, units


def analyze(out):
    """Construct actual observed anchors, then apply cold scores without label fit."""
    manifest = check(out)
    config = manifest["config"]
    _, rows, records = authenticated(config)
    start = time.monotonic()
    source_mask = np.array([r["role"] == config["source_role"] for r in rows])
    if {r["situation"] for r in rows if r["role"] == config["source_role"]} != set(
        config["source_tasks"]
    ):
        raise ValueError("Source task leakage")
    observations = [
        M.lexical_observations(r["text"], config["observations"]) for r in records
    ]
    counts = np.stack([v[0] for v in observations])
    rates = np.stack([v[1] for v in observations])
    y, center, scale, anchor_support = M.source_units(rates, source_mask)
    features, units = feature_views(config, source_mask)
    conditions = ["|".join(str(r[k]) for k in config["condition_keys"]) for r in rows]
    source_conditions = [
        c for c, mask in zip(conditions, source_mask, strict=True) if mask
    ]
    programs = {
        key: M.anchor_program(x[source_mask], y[source_mask], source_conditions)
        for key, x in features.items()
    }
    saved = {
        "observable_center": center,
        "observable_scale": scale,
        "observable_support": anchor_support,
    }
    for key, p in programs.items():
        for field in ("origin", "total", "within", "between", "norm", "supported"):
            saved[f"{key}_{field}"] = p[field]
        for field, value in units[key].items():
            saved[f"{key}_feature_{field}"] = value
    np.savez_compressed(out / "programs.npz", **saved)
    write(
        out / "source_program_seal.json", {"programs.npz": digest(out / "programs.npz")}
    )
    # The cold scorer receives only fixed source programs and numeric features.
    readings = {
        key: {arm: M.cold_score(p, features[key], arm) for arm in config["arms"]}
        for key, p in programs.items()
    }
    np.savez_compressed(
        out / "cold_readings.npz",
        **{
            f"{key}_{arm}": x
            for key, arms in readings.items()
            for arm, x in arms.items()
        },
    )
    np.savez_compressed(
        out / "actual_observations.npz",
        counts=counts,
        rates=rates,
        source_mask=source_mask,
    )
    observed_rows = []
    for i, row in enumerate(rows):
        for j, definition in enumerate(config["observations"]):
            observed_rows.append(
                {
                    "text_id": row["text_id"],
                    "role": row["role"],
                    "situation": row["situation"],
                    "style": row["style"],
                    "anchor": definition["id"],
                    "count": int(counts[i, j]),
                    "per100_words": float(rates[i, j]),
                    "words": observations[i][2],
                    "finish_reason": records[i]["finish_reason"],
                    "body_sha256": hashlib.sha256(
                        records[i]["text"].encode()
                    ).hexdigest(),
                }
            )
    table(out / "observed_counts.csv", observed_rows)
    pairs, meta = M.persona_pairs(rows)
    changes_y = M.pair_changes(y, pairs)
    changes_rate = M.pair_changes(rates, pairs)
    effects, summary_cells = [], []
    for key, arms in readings.items():
        for arm, scores in arms.items():
            changes = M.pair_changes(scores, pairs)
            for i, m in enumerate(meta):
                for j, definition in enumerate(config["observations"]):
                    effects.append(
                        m
                        | {
                            "representation": key,
                            "arm": arm,
                            "anchor": definition["id"],
                            "observed_rate_change": float(changes_rate[i, j]),
                            "observed_source_unit_change": float(changes_y[i, j]),
                            "witness_change": float(changes[i, j]),
                        }
                    )
            for axis in ("extraversion", "neuroticism"):
                for role in ("discovery", "development", "evaluation"):
                    mask = np.array(
                        [m["axis"] == axis and m["role"] == role for m in meta]
                    )
                    for j, definition in enumerate(config["observations"]):
                        comp = M.signed_comparison(changes_y[mask, j], changes[mask, j])
                        summary_cells.append(
                            {
                                "axis": axis,
                                "role": role,
                                "representation": key,
                                "arm": arm,
                                "anchor": definition["id"],
                                "pairs": int(mask.sum()),
                                **comp,
                                "mean_observed_rate_change": float(
                                    changes_rate[mask, j].mean()
                                ),
                                "mean_witness_change": float(
                                    np.nanmean(changes[mask, j])
                                ),
                            }
                        )
    table(out / "paired_effects.csv", effects)
    table(out / "summary_cells.csv", summary_cells)
    condition_cells = []
    for axis in ("extraversion", "neuroticism"):
        for task in sorted({m["situation"] for m in meta}):
            for style in ("neutral", "newsletter", "spelling"):
                mask = np.array(
                    [
                        m["axis"] == axis
                        and m["situation"] == task
                        and m["style"] == style
                        for m in meta
                    ]
                )
                for j, definition in enumerate(config["observations"]):
                    values = changes_rate[mask, j]
                    condition_cells.append(
                        {
                            "axis": axis,
                            "situation": task,
                            "style": style,
                            "anchor": definition["id"],
                            "pairs": int(mask.sum()),
                            "mean_observed_rate_change": float(values.mean()),
                            "positive": int((values > 1e-12).sum()),
                            "negative": int((values < -1e-12).sum()),
                            "zero": int((abs(values) <= 1e-12).sum()),
                        }
                    )
    table(out / "condition_cells.csv", condition_cells)
    elapsed = time.monotonic() - start
    result = {
        "packet": "HM164",
        "status": "FINITE_OBSERVABLE_ANCHOR_CONSTRUCTION_COMPLETE",
        "rows": len(rows),
        "source_rows": int(source_mask.sum()),
        "actual_pairs": len(pairs),
        "anchor_count": len(config["observations"]),
        "new_neural_forwards": 0,
        "human_observations": 0,
        "independent_future_author_events": 0,
        "refusals_retained": sum(bool(r["audit"]["refusal"]) for r in records),
        "length_caps_retained": sum(r["finish_reason"] == "length" for r in records),
        "condition_association_retained": True,
        "decomposition_errors": {
            key: p["decomposition_error"] for key, p in programs.items()
        },
        "between_over_total_norms": {
            key: (
                np.linalg.norm(p["between"], axis=1)
                / np.where(p["norm"] > 0, p["norm"], 1)
            ).tolist()
            for key, p in programs.items()
        },
        "elapsed_seconds": elapsed,
        "cpu_seconds_cap": config["cpu_seconds_cap"],
        "within_cap": elapsed <= config["cpu_seconds_cap"],
        "scope": config["scope"],
        "original_scope_complete": False,
    }
    write(out / "summary.json", result)
    files = sorted(
        p for p in out.iterdir() if p.is_file() and p.name != "manifest.json"
    )
    write(out / "output_seal.json", {p.name: digest(p) for p in files})
    print(json.dumps(result, allow_nan=False))


def audit(out):
    """Verify source coefficients by pair differences, independently of constructor."""
    manifest = check(out)
    config = manifest["config"]
    for name, expected in read(out / "output_seal.json").items():
        if digest(out / name) != expected:
            raise ValueError("Output seal mismatch: " + name)
    _, rows, records = authenticated(config)
    mask = np.array([r["role"] == config["source_role"] for r in rows])
    features, _ = feature_views(config, mask)
    with np.load(out / "programs.npz", allow_pickle=False) as program:
        rates = np.stack(
            [
                M.lexical_observations(r["text"], config["observations"])[1]
                for r in records
            ]
        )
        y = (rates[mask] - program["observable_center"]) / program["observable_scale"]
        errors = {}
        for key, xall in features.items():
            x = xall[mask]
            alternate = np.zeros((y.shape[1], x.shape[1]))
            for i in range(len(x)):
                alternate += (y[i] - y).T @ (x[i] - x)
            alternate /= 2 * len(x) ** 2
            stored = program[f"{key}_total"]
            error = float(np.max(abs(alternate - stored)))
            if not np.allclose(alternate, stored, atol=1e-10, rtol=1e-10):
                raise ValueError("Alternate covariance disagrees")
            errors[key] = error
    result = {
        "status": "PASS",
        "alternate_total_covariance_max_errors": errors,
        "source_events": int(mask.sum()),
        "same_agent_not_blind": True,
    }
    write(out / "alternate_audit.json", result)
    print(json.dumps(result))


def main():
    """Expose explicit finite stages; never restart or overwrite an existing run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "analyze", "audit"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze, "audit": audit}[args.stage](args.out)


if __name__ == "__main__":
    main()
