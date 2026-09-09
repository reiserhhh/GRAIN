#!/usr/bin/env python3
"""Analyze fixed query, temporal and natural evidence interventions with paired effects."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    digest,
    quality,
    read_json,
    write_json,
)
from scripts.analyze_grain_multiquery_sensitivity import cluster_r_interval
from scripts.run_grain_evidence_packets import PACKET_DIRS


def read_packet(packet, independent=None):
    """Validate prediction bytes and join exact world coordinates without criteria."""
    if read_json(packet / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete model collection first")
    path = packet / "predictions.csv"
    if digest(path) != read_json(packet / "prediction_freeze.json")["sha256"]:
        raise ValueError("Fixed-function predictions changed")
    frame = pd.read_csv(path)
    if independent is not None:
        p = independent / "predictions.csv"
        if read_json(independent / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Complete independent program execution first")
        if digest(p) != read_json(independent / "prediction_freeze.json")["sha256"]:
            raise ValueError("Independent prediction freeze changed")
        frame = pd.concat([frame, pd.read_csv(p)], ignore_index=True)
    world = pd.DataFrame(read_json(packet / "private_worlds.json")).drop(
        columns="texts"
    )
    result = frame.merge(world, on="world_id", validate="many_to_one")
    if (
        len(result) != len(frame)
        or result.duplicated(["world_id", "query", "readout"]).any()
    ):
        raise ValueError("World join or prediction identity failure")
    return result


def pair_contrasts(frame, axis, low, high, base_keys):
    """Take exact partner differences and preserve available additive components."""
    rows = []
    for key, g in frame.groupby(base_keys, dropna=False):
        if len(g) != 2 or set(g[axis]) != {low, high}:
            raise ValueError("Expected a complete matched intervention pair")
        a, b = g[g[axis] == low].iloc[0], g[g[axis] == high].iloc[0]
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(base_keys, key, strict=True)) | {"axis": axis}
        for column in (
            "score",
            "mean",
            "configuration",
            "final",
            "projection1",
            "projection2",
        ):
            if column in frame and pd.notna(a[column]) and pd.notna(b[column]):
                row[column + "_delta"] = float(b[column] - a[column])
        rows.append(row)
    return pd.DataFrame(rows)


def response_rows(frame, kind):
    """Build semantically declared paired edits, never unpaired score changes."""
    basic = ["query", "readout", "scene_id", "split"]
    if kind == "identity":
        items = []
        for axis, low, high in (
            ("actor", "self", "other"),
            ("opportunity", "voluntary", "assigned"),
            ("realization", 0, 1),
        ):
            fixed = [c for c in ("actor", "opportunity", "realization") if c != axis]
            items.append(pair_contrasts(frame, axis, low, high, basic + fixed))
        return pd.concat(items, ignore_index=True)
    if kind == "time":
        return pd.concat(
            [
                pair_contrasts(
                    frame, "negative_now", False, True, basic + ["reverse_display"]
                ),
                pair_contrasts(
                    frame, "reverse_display", False, True, basic + ["negative_now"]
                ),
            ],
            ignore_index=True,
        )
    items = []
    for low, high, axis in (
        ("self_report", "other_report", "actor"),
        ("endorsed", "denied", "endorsement"),
        ("today", "past", "time"),
        ("original", "remove", "removal"),
    ):
        part = pair_contrasts(
            frame[frame.condition.isin([low, high])], "condition", low, high, basic
        )
        part["axis"] = axis
        items.append(part)
    return pd.concat(items, ignore_index=True)


def compare_responses(responses):
    """Compare each readout to the same native event delta and cluster by source unit."""
    id_cols = [c for c in responses if not c.endswith("_delta") and c != "readout"]
    target = responses[responses.readout == "native"][id_cols + ["score_delta"]].rename(
        columns={"score_delta": "native_delta"}
    )
    # Different factorial axes use explicit NaNs for their missing fixed coordinates.
    joined = responses[responses.readout != "native"].merge(
        target, on=id_cols, validate="many_to_one"
    )
    if len(joined) != len(responses[responses.readout != "native"]):
        raise ValueError("Missing native response partner")
    rows = []
    for (q, readout, split, axis), g in joined.groupby(
        ["query", "readout", "split", "axis"]
    ):
        rows.append(
            {
                "query": q,
                "readout": readout,
                "split": split,
                "axis": axis,
                "n_pairs": len(g),
                "independent_units": g.scene_id.nunique(),
                **cluster_r_interval(g.native_delta, g.score_delta, g.scene_id),
                "mae": float(abs(g.native_delta - g.score_delta).mean()),
                "mean_native_delta": float(g.native_delta.mean()),
                "mean_readout_delta": float(g.score_delta.mean()),
                "mean_abs_native_delta": float(abs(g.native_delta).mean()),
                "mean_abs_readout_delta": float(abs(g.score_delta).mean()),
                "sign_agreement": float(
                    (np.sign(g.native_delta) == np.sign(g.score_delta)).mean()
                ),
            }
        )
    return joined, pd.DataFrame(rows)


def instrument_matrix(frame, responses, definitions):
    """Separate framing, response coding, levels and evidence-response agreement."""
    rows = []
    for readout, g in frame.groupby("readout"):
        for split, part in g.groupby("split"):
            levels = part.pivot(index="world_id", columns="query", values="score")
            for qa, qb in itertools.combinations(sorted(levels), 2):
                da, db = definitions[qa], definitions[qb]
                change = (
                    "coding"
                    if da["framing"] == db["framing"]
                    else "framing"
                    if da["coding"] == db["coding"]
                    else "both"
                )
                rows.append(
                    {
                        "query_a": qa,
                        "query_b": qb,
                        "change": change,
                        "readout": readout,
                        "split": split,
                        "axis": "levels",
                        **quality(levels[qa], levels[qb]),
                    }
                )
            r = responses[(responses.readout == readout) & (responses.split == split)]
            for axis, values in r.groupby("axis"):
                fixed = [
                    c for c in ("actor", "opportunity", "realization") if c != axis
                ]
                grid = values.pivot(
                    index=["scene_id", *fixed], columns="query", values="score_delta"
                )
                for qa, qb in itertools.combinations(sorted(grid), 2):
                    da, db = definitions[qa], definitions[qb]
                    change = (
                        "coding"
                        if da["framing"] == db["framing"]
                        else "framing"
                        if da["coding"] == db["coding"]
                        else "both"
                    )
                    rows.append(
                        {
                            "query_a": qa,
                            "query_b": qb,
                            "change": change,
                            "readout": readout,
                            "split": split,
                            "axis": axis,
                            **quality(grid[qa], grid[qb]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize(packet, out, independent=None):
    """Write aggregate fidelity, matched effects and instrument-specific decisions."""
    if out.exists():
        raise FileExistsError("Choose a new analysis ID")
    manifest = read_json(packet / "manifest.json")
    kind = manifest["kind"]
    frame = read_packet(packet, independent)
    response = response_rows(frame, kind)
    joined, metrics = compare_responses(response)
    levels = []
    reference = frame[frame.readout == "native"][["world_id", "query", "score"]].rename(
        columns={"score": "native"}
    )
    aligned = frame[frame.readout != "native"].merge(
        reference, on=["world_id", "query"], validate="many_to_one"
    )
    for (q, readout, split), g in aligned.groupby(["query", "readout", "split"]):
        levels.append(
            {
                "query": q,
                "readout": readout,
                "split": split,
                "independent_units": g.scene_id.nunique(),
                **quality(g.native, g.score),
            }
        )
    out.mkdir(parents=True)
    pd.DataFrame(levels).to_csv(out / "level_metrics.csv", index=False)
    response.to_csv(out / "paired_responses.csv", index=False)
    joined.to_csv(out / "response_components.csv", index=False)
    metrics.to_csv(out / "response_metrics.csv", index=False)
    compliance = (
        frame[frame.readout == "native"]
        .groupby(["query", "split"], as_index=False)
        .agg(
            mean_candidate_mass=("candidate_mass", "mean"),
            min_candidate_mass=("candidate_mass", "min"),
            native_sd=("score", "std"),
            native_min=("score", "min"),
            native_max=("score", "max"),
        )
    )
    compliance.to_csv(out / "candidate_mass.csv", index=False)
    if kind == "identity":
        table = instrument_matrix(frame, response, read_json(packet / "queries.json"))
        table.to_csv(out / "query_identity_matrix.csv", index=False)
        coords = manifest["question_tokens"]
        write_json(
            out / "coordinate_compatibility.json",
            {
                "query_lengths": {q: len(t) for q, t in coords.items()},
                "pairwise": [
                    {"query_a": a, "query_b": b, "same_tokens": coords[a] == coords[b]}
                    for a, b in itertools.combinations(coords, 2)
                ],
                "own_reference_constructed_per_question": True,
                "alphabet_polarity": "0/A low or I;1/B high or E",
                "original_continuous_Big5_and_letter_EI_unchanged": True,
            },
        )
        native = frame[frame.readout == "native"].pivot(
            index=["world_id", "split"], columns="query", values="score"
        )
        native["coding_effect_big5"] = (
            native.big5_binary_letters - native.big5_binary_digits
        )
        native["coding_effect_mbti"] = (
            native.mbti_binary_letters - native.mbti_binary_digits
        )
        native["framing_effect_digits"] = (
            native.mbti_binary_digits - native.big5_binary_digits
        )
        native["framing_effect_letters"] = (
            native.mbti_binary_letters - native.big5_binary_letters
        )
        native["framing_x_coding"] = (
            native.coding_effect_mbti - native.coding_effect_big5
        )
        native.reset_index().to_csv(out / "framing_coding_terms.csv", index=False)
        print(
            table.query("split=='held_scene' and readout=='native'")
            .round(4)
            .to_string(index=False),
            flush=True,
        )
    if kind == "time":
        raw = response.groupby(
            ["query", "readout", "split", "axis"], as_index=False
        ).agg(
            mean_delta=("score_delta", "mean"),
            mean_abs_delta=("score_delta", lambda x: abs(x).mean()),
            sd_delta=("score_delta", "std"),
        )
        raw.to_csv(out / "time_order_effects.csv", index=False)
        rectangles = []
        for key, g in frame.groupby(["query", "readout", "scene_id", "split"]):
            p = g.set_index(["negative_now", "reverse_display"]).score
            rectangles.append(
                dict(zip(["query", "readout", "scene_id", "split"], key, strict=True))
                | {
                    "time_x_order": float(
                        p.loc[True, True]
                        - p.loc[True, False]
                        - p.loc[False, True]
                        + p.loc[False, False]
                    )
                }
            )
        pd.DataFrame(rectangles).to_csv(
            out / "time_order_interactions.csv", index=False
        )
        print(
            raw.query("split=='held_scene'").round(4).to_string(index=False), flush=True
        )
    if kind == "evidence":
        print(
            metrics.query(
                "readout in ['R_synth','token_semantic_order','context4_mean']"
            )[
                [
                    "query",
                    "readout",
                    "axis",
                    "r",
                    "mean_abs_native_delta",
                    "mean_abs_readout_delta",
                ]
            ]
            .round(4)
            .to_string(index=False),
            flush=True,
        )
    write_json(
        out / "summary.json",
        {
            "status": "COMPLETE_FIXED_PACKET_ANALYSIS",
            "kind": kind,
            "units": manifest["units"],
            "independent_sources": frame.scene_id.nunique(),
            "prediction_sha256": digest(packet / "predictions.csv"),
            "independent_prediction_sha256": digest(independent / "predictions.csv")
            if independent
            else None,
            "code_sha256": digest(Path(__file__)),
            "bootstrap_cluster": "author or scene skeleton, not rendering",
            "bootstrap_draws": 1000,
            "human_labels_used": False,
            "scope": "local instrument/evidence response, not personality accuracy or clinical validity",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", choices=PACKET_DIRS, default="identity")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--independent", type=Path)
    a = parser.parse_args()
    source = a.source or BASE / "navigation_v1" / PACKET_DIRS[a.packet]
    summarize(source, a.out or source / "analysis", a.independent)
