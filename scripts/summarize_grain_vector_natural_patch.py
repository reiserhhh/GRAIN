#!/usr/bin/env python3
"""Author-clustered functional effects, not psychological accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.analysis import grouped_bootstrap
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json


def aggregate(frame, keys, values):
    """Weight authors equally after averaging their repeated conditions."""
    rows = []
    for key, sub in frame.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key, strict=True))
        row["rows"] = len(sub)
        for value in values:
            result = grouped_bootstrap(sub[value], sub["author_id"])
            row.update({value + "_" + k: v for k, v in result.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Require complete controls; compare paired candidate/random effects."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    read = lambda name: json.loads((a.parent_dir / name).read_text(encoding="utf-8"))
    manifest, runtime = read("manifest.json"), read("run_status.json")
    if runtime["status"] != "COMPLETE" or runtime["completed_jobs"] != 48:
        raise ValueError(
            "Full analysis requires every frozen job, not selected completions"
        )
    if any(
        runtime[k] != 0
        for k in ("self_sham_max_abs", "restore_max_abs", "prefix_max_error")
    ):
        raise ValueError("Instrument control failed")
    frame = pd.read_csv(a.parent_dir / "interventions.csv")
    base = pd.read_csv(a.parent_dir / "baselines.csv")
    geometry = pd.read_csv(a.parent_dir / "geometry.csv")
    ids = ["case_id", "author_id", "donor_id", "rule", "task", "site", "span"]
    if len(frame) != 2496 or len(base) != 48 or frame.author_id.nunique() != 6:
        raise ValueError("Unexpected paired experiment population")
    if frame.duplicated(ids + ["arm", "rank", "alpha"]).any():
        raise ValueError("Duplicated intervention rows")
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new analysis directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    frame["abs_delta"] = frame.delta.abs()
    frame["natural_delta"] = frame.natural_donor_margin - frame.baseline_margin
    frame["toward_donor"] = (frame.oriented_delta > 0).astype(float)
    frame["decision_flip"] = (
        (frame.patched_margin > 0) != (frame.baseline_margin > 0)
    ).astype(float)
    # Fractions are not defined for the near-zero natural contrast subset.
    frame["large_natural_contrast"] = (frame.natural_delta.abs() >= 1).astype(float)
    aggregate(
        frame,
        ["task", "rule", "site", "span", "arm", "rank", "alpha"],
        [
            "delta",
            "abs_delta",
            "oriented_delta",
            "toward_donor",
            "decision_flip",
            "recovery_ratio",
            "large_natural_contrast",
            "relative_effect_norm",
        ],
    ).to_csv(out / "effects_by_rule.csv", index=False)
    aggregate(
        frame,
        ["task", "site", "span", "arm", "rank", "alpha"],
        [
            "delta",
            "abs_delta",
            "oriented_delta",
            "toward_donor",
            "decision_flip",
            "recovery_ratio",
            "large_natural_contrast",
            "relative_effect_norm",
        ],
    ).to_csv(out / "effects_pooled_rules.csv", index=False)
    candidates = frame[(frame.arm == "candidate") & (frame.alpha == 1)]
    random = frame[frame.arm.str.startswith("random_")]
    random = random.groupby(ids + ["rank"], as_index=False)[
        ["abs_delta", "oriented_delta", "actual_effect_norm", "requested_effect_norm"]
    ].mean()
    paired = candidates.merge(
        random,
        on=ids + ["rank"],
        suffixes=("_candidate", "_random"),
        validate="one_to_one",
    )
    for metric in ("abs_delta", "oriented_delta"):
        paired[metric + "_advantage"] = (
            paired[metric + "_candidate"] - paired[metric + "_random"]
        )
    for kind in ("actual", "requested"):
        paired[kind + "_norm_relative_difference"] = (
            paired[kind + "_effect_norm_random"]
            - paired[kind + "_effect_norm_candidate"]
        ) / paired[kind + "_effect_norm_candidate"].clip(lower=1e-12)
    paired.to_csv(out / "candidate_random_pairs.csv", index=False)
    aggregate(
        paired,
        ["task", "site", "span", "rank"],
        [
            "abs_delta_advantage",
            "oriented_delta_advantage",
            "actual_norm_relative_difference",
            "requested_norm_relative_difference",
        ],
    ).to_csv(out / "candidate_random_summary.csv", index=False)
    geometry["current_difference_norm"] = geometry.current_difference_norm.astype(float)
    aggregate(
        geometry,
        ["rule", "site"],
        [
            "rank2_energy",
            "rank4_energy",
            "current_difference_norm",
        ],
    ).to_csv(out / "current_geometry.csv", index=False)
    base["abs_natural_delta"] = base.natural_delta.abs()
    base["donor_decision_flip"] = (
        (base.baseline_margin > 0) != (base.natural_donor_margin > 0)
    ).astype(float)
    aggregate(
        base,
        ["task", "rule"],
        [
            "baseline_margin",
            "natural_donor_margin",
            "natural_delta",
            "abs_natural_delta",
            "donor_decision_flip",
        ],
    ).to_csv(out / "natural_context_contrast.csv", index=False)
    # These are two different tasks, so collateral movement is not accuracy loss.
    collateral = frame.pivot(
        index=[
            "case_id",
            "author_id",
            "donor_id",
            "rule",
            "site",
            "span",
            "arm",
            "rank",
            "alpha",
        ],
        columns="task",
        values="abs_delta",
    ).reset_index()
    collateral["target_minus_collateral_abs"] = (
        collateral.disposition - collateral.current_guidance
    )
    aggregate(
        collateral,
        ["site", "span", "arm", "rank", "alpha"],
        [
            "disposition",
            "current_guidance",
            "target_minus_collateral_abs",
        ],
    ).to_csv(out / "collateral_effects.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "code_sha256": code_fingerprint(ROOT),
        "runtime": runtime,
        "authors": 6,
        "current_comments": 12,
        "paired_contexts": 24,
        "requested_norm_match_max_relative_error": float(
            paired.requested_norm_relative_difference.abs().max()
        ),
        "actual_norm_match_max_relative_error": float(
            paired.actual_norm_relative_difference.abs().max()
        ),
        "psychological_labels_used": False,
        "limitations": [
            "Six independent authors; repeats and forwards are not additional participants",
            "Pair-specific basis is not a universal personality or author subspace",
            "Current token alignment does not remove historical length/position effects",
            "No human truth labels for disposition; collateral changes are not proven accuracy damage",
            "Bootstrap intervals are descriptive and not corrected for the exploratory comparison family",
            "Only one small open-weight model; no general claim about ChatGPT reasoning",
        ],
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
