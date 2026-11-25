#!/usr/bin/env python3
"""Offline worked study. Run from the repository root: python -m examples.handbook_study DIR."""

import argparse
import json
from pathlib import Path

import numpy as np
from grain_core.liquid_reference_error import population_io
from threadpoolctl import threadpool_limits

from grain_core import liquid_instrument as reader


def corpus(role, count):
    # Target writers use identical words with different ordering, by construction.
    phrases = {
        "plan": ["I wrote a list of steps.", "I checked the route on paper."],
        "explore": ["I tried another route.", "I followed a different trail."],
    }
    settings = (
        "station",
        "square",
        "bridge",
        "market",
        "school",
        "museum",
        "garden",
        "shop",
        "hall",
        "library",
        "park",
        "harbor",
    )
    episodes = []
    for person in range(count):
        for v, view in enumerate("AB"):
            for c, condition in enumerate(("departure", "detour")):
                if role == "target":
                    seq = [0, 0, 0, 1, 1, 1] if person == 0 else [0, 1] * 3
                else:
                    seq = [
                        int((j * (person % 3 + 1) + c + v) % 7 < 1 + person % 6)
                        for j in range(6)
                    ]
                place = settings[
                    (c + v + (0 if role == "target" else person)) % len(settings)
                ]
                snippets = [
                    f"At the {place}: " + phrases[("plan", "explore")[i]][(c + v) % 2]
                    for i in seq
                ]
                author = f"{role}-{person:02d}"
                episodes.append(
                    {
                        "id": f"{author}-{view}-{c}",
                        "author": author,
                        "condition": condition,
                        "carrier": "route-note",
                        "view": view,
                        "events": snippets,
                    }
                )
    doc = {"language": "en", "clock": "opportunity", "episodes": episodes}
    if role == "calibration":
        doc.update(
            population="Synthetic route-note writers", namespace="handbook-fiction"
        )
    return doc


def run(out, inputs=None):
    docs = {
        role: reader.load(inputs / f"{role}.json") if inputs else corpus(role, n)
        for role, n in (("discovery", 12), ("calibration", 10), ("target", 2))
    }
    seen = set()
    for role, doc in docs.items():
        parsed = reader.episodes(doc, 0)
        authors = {meta["author"] for meta, _ in parsed}
        if seen & authors:
            raise ValueError(f"{role}: an author occurs in another cohort")
        if any(len(rows) < 2 for _, rows in parsed):
            raise ValueError(
                f"{role}: the worked step-length measure requires two events per episode"
            )
        seen.update(authors)
    for key in ("namespace", "population"):
        value = docs["calibration"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"calibration.json: declare a nonempty {key}")
    out.mkdir(parents=True, exist_ok=False)
    for role, doc in docs.items():
        reader.save(out / f"{role}.json", doc)
    items = reader.episodes(reader.load(out / "discovery.json"), 0)
    atlas = reader.fit_atlas(
        [t for _, rows in items for _, _, t in rows], 4, "gmm_soft"
    )
    conditions = sorted({m["condition"] for m, _ in items})
    feats = reader.participant_features(
        items, reader.native_paths(atlas, items), conditions
    )
    ref = {
        "atlas": atlas,
        "factors": reader.fit_axes(feats),
        "conditions": conditions,
        "cut": "whole-event",
        "source": reader.digest(reader.load(out / "discovery.json")),
        "namespace": docs["calibration"]["namespace"],
        "discovery_members": sorted(
            {
                reader.digest([docs["calibration"]["namespace"], m["author"]])
                for m, _ in items
            }
        ),
    }
    reader.save(out / "native-reference.json", ref)
    ref = reader.load(out / "native-reference.json")
    table, _, _ = measure(ref, docs["calibration"], docs["calibration"]["population"])
    if np.any(np.ptp([r["values"] for r in table["units"]], axis=0) <= 1e-10):
        raise ValueError(
            "a teaching axis has no variation in the calibration population"
        )
    reader.save(out / "calibration-units.json", table)
    reader.save(out / "chart.json", population_io("fit", table))
    return score(out, out, docs["target"])


def measure(ref, doc, population):
    items = reader.episodes(doc, 0)
    paths = reader.native_paths(ref["atlas"], items)
    by_author = {}
    for (meta, _), path in zip(items, paths, strict=True):
        x = np.array(path)
        if len(x) < 2:
            raise ValueError(
                f"{meta['id']}: step length requires two events per episode"
            )
        if np.any(x[:, -1] == 1):
            raise ValueError(
                f"{meta['id']}: uncovered event; inspect the frozen vocabulary"
            )
        # Equal episode weights; never form a step across two episodes.
        by_author.setdefault(meta["author"], []).append(
            [x[:, 0].mean(), np.linalg.norm(np.diff(x, axis=0), axis=1).mean()]
        )
    axes = [f"mean:{ref['atlas']['labels'][0]}", "mean:within_episode_step_l2"]
    table = {
        "population": population,
        "namespace": ref["namespace"],
        "axes": axes,
        "basis": reader.digest([ref, axes]),
        "units": [
            {"id": a, "values": np.mean(rows, axis=0).tolist()}
            for a, rows in by_author.items()
        ],
    }
    return table, items, paths


def score(out, frozen, doc):
    ref = reader.load(frozen / "native-reference.json")
    chart = reader.load(frozen / "chart.json")
    table, items, paths = measure(ref, doc, chart["population"])
    source = set(ref["discovery_members"])
    if any(
        reader.digest([ref["namespace"], r["id"]]) in source for r in table["units"]
    ):
        raise ValueError("a target was a discovery author")
    positions = population_io("locate", table, chart=chart)
    families = reader.participant_features(items, paths, ref["conditions"])
    reader.save(out / "target-units.json", table)
    reader.save(out / "target-families.json", reader.scores(families, ref["factors"]))
    reader.save(out / "positions.json", positions)
    return positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory; will not overwrite")
    parser.add_argument(
        "--inputs",
        type=Path,
        help="directory of three prepared English episode JSON files",
    )
    parser.add_argument(
        "--frozen-from",
        type=Path,
        help="directory with the saved native reference and chart",
    )
    parser.add_argument(
        "--new", type=Path, help="new target episode JSON; pair with --frozen-from"
    )
    args = parser.parse_args()
    if bool(args.frozen_from) != bool(args.new) or (args.inputs and args.new):
        parser.error("use --inputs alone, or --frozen-from and --new together")
    with threadpool_limits(limits=1):
        if args.new:
            args.output.mkdir(parents=True, exist_ok=False)
            result = score(args.output, args.frozen_from, reader.load(args.new))
        else:
            result = run(args.output, args.inputs)
        print(json.dumps(result, indent=2))
