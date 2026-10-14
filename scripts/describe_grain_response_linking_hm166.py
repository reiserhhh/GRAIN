#!/usr/bin/env python3
"""Post-outcome description of frozen response relinks; no new scorer or fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_response_function as M
from scripts import run_grain_response_function_transport_hm166 as R


def describe(output_name):
    """Separate link comparators and signed components without changing primary data."""
    source = R.DEFAULT
    manifest = R.check(source)
    for name, expected in R.A.read(source / "analysis_seal.json").items():
        if R.A.digest(source / name) != expected:
            raise ValueError("Sealed primary analysis changed: " + name)
    with np.load(source / "source.npz", allow_pickle=False) as data:
        groups = data["groups"]
    articles = R.A.read(source / "source_articles.json")
    pairs = R.A.read(source / "complete_pairs.json")
    with np.load(source / "paired_accounts.npz", allow_pickle=False) as data:
        readings, observed = data["cold"], data["observed"]
    relinks = []
    for shift in range(1, 6):
        donors = np.arange(len(articles))
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            donors[ids] = np.roll(ids, shift)
        relinks.append(
            {
                "shift": shift,
                "same_article": int(sum(i == j for i, j in enumerate(donors))),
                "same_persona_pole": sum(
                    articles[i]["pole"] == articles[j]["pole"]
                    for i, j in enumerate(donors)
                ),
                "same_seed": sum(
                    articles[i]["seed"] == articles[j]["seed"]
                    for i, j in enumerate(donors)
                ),
                "profiles": len(articles),
            }
        )
    components, cancellation = [], []
    for role in ("all", "development", "evaluation"):
        for axis in ("all", "extraversion", "neuroticism"):
            ids = np.array(
                [
                    (role == "all" or p["role"] == role)
                    and (axis == "all" or p["axis"] == axis)
                    for p in pairs
                ]
            )
            for channel, channel_name in enumerate(R.CHANNELS):
                for opportunity, opportunity_name in enumerate(
                    manifest["config"]["opportunities"]
                ):
                    for field, field_name in enumerate(("even", "odd")):
                        y = observed[ids, channel, opportunity, field]
                        values = readings[ids, :, channel, opportunity, field]
                        base = {
                            "role": role,
                            "axis": axis,
                            "channel": channel_name,
                            "opportunity": opportunity_name,
                            "order_field": field_name,
                        }
                        for shift in range(1, 6):
                            # A relink preserves every block mean, so between is unchanged.
                            linked_within = values[:, shift + 3] - values[:, 3]
                            components.append(
                                base
                                | {
                                    "shift": shift,
                                    "preserves_persona_pole": shift % 2 == 0,
                                }
                                | M.signed_alignment(linked_within, y)
                            )
                        full, within, between = values[:, 0], values[:, 2], values[:, 3]
                        products = [
                            float(np.mean(v * y)) for v in (full, within, between)
                        ]
                        cancellation.append(
                            base
                            | {
                                "full_product": products[0],
                                "within_product": products[1],
                                "between_product": products[2],
                                "signed_product_error": abs(
                                    products[0] - products[1] - products[2]
                                ),
                                "score_decomposition_error": float(
                                    np.max(abs(full - within - between))
                                ),
                            }
                        )
    if Path(output_name).name != output_name:
        raise ValueError("A single new child directory name is required")
    out = source / output_name
    out.mkdir(exist_ok=False)
    R.A.table(out / "source_relink_addresses.csv", relinks)
    R.A.table(out / "relinked_within_alignment.csv", components)
    R.A.table(out / "signed_component_accounts.csv", cancellation)
    summary = {
        "timing": "POST_OUTCOME; descriptive arithmetic, not prospective evidence",
        "source_relink_addresses": relinks,
        "evaluation_probability_even": [
            row
            for row in components
            if row["role"] == "evaluation"
            and row["axis"] != "all"
            and row["channel"] == "probability"
            and row["order_field"] == "even"
        ],
        "decomposition_error": max(
            row["score_decomposition_error"] for row in cancellation
        ),
        "product_error": max(row["signed_product_error"] for row in cancellation),
        "new_forwards": 0,
        "original_scope_complete": False,
    }
    R.A.write(out / "summary.json", summary)
    inputs = (
        "manifest.json",
        "analysis_seal.json",
        "source.npz",
        "source_articles.json",
        "complete_pairs.json",
        "paired_accounts.npz",
    )
    R.A.write(
        out / "manifest.json",
        {
            "parent": str(source.relative_to(ROOT)),
            "input_hashes": {name: R.A.digest(source / name) for name in inputs},
            "script_sha256": R.A.digest(Path(__file__)),
            "timing": summary["timing"],
            "new_fit": False,
            "new_scoring_program": False,
        },
    )
    R.A.write(
        out / "output_seal.json",
        {
            name: R.A.digest(out / name)
            for name in (
                "source_relink_addresses.csv",
                "relinked_within_alignment.csv",
                "signed_component_accounts.csv",
                "summary.json",
            )
        },
    )
    print(json.dumps({"described": str(out), "addresses": relinks}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", default="post_outcome_link_description_r1")
    describe(parser.parse_args().output_name)
