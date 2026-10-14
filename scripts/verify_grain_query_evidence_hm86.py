#!/usr/bin/env python3
"""HM86 alternate arithmetic: direct token sums, no primary reading helper."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import MODEL, read_json
from scripts.run_grain_author_contrast_hm82 import write
from scripts.run_grain_query_evidence_hm86 import OUT, READOUTS, check_hashes, unit


def verify(out):
    """Reconstruct every application vector, calibration and forbidden-call trace."""
    manifest = read_json(out / "00/manifest.json")
    check_hashes(ROOT, {**manifest["code"], **manifest["source"]})
    check_hashes(MODEL, manifest["model"])
    check_hashes(out, read_json(out / "01/raw_seal.json"))
    check_hashes(out, read_json(out / "02/output_seal.json"))
    rows = read_json(out / "00/rows.json")
    with np.load(out / "01/calibration.npz", allow_pickle=False) as arr:
        dirs, refs, source_means = (
            arr[k].copy() for k in ("directions", "raw_queries", "source_means")
        )
    with np.load(out / "02/observations.npz", allow_pickle=False) as arr:
        vectors, projected, masses = (
            arr[k].copy() for k in ("vectors", "projections", "masses")
        )
    maximum, checks = 0.0, 0
    means_by_body = {}
    targets = manifest["spec"]["targets"]

    def compare(actual, expected):
        nonlocal maximum, checks
        if np.shape(actual) != np.shape(expected) or not np.isfinite(actual).all():
            raise ValueError("Shape/nonfinite verification failure")
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        checks += np.size(actual)

    delta = (
        source_means[:, 1, 0]
        - source_means[:, 0, 0]
        + source_means[:, 1, 1]
        - source_means[:, 0, 1]
    ) / 2
    compare(dirs, delta / np.sqrt(np.sum(delta * delta, axis=(1, 2)))[:, None, None])
    for row in rows:
        raw = unit(out, row["unit"])
        if row["stage"] == "source":
            t, p, a = (
                manifest["spec"]["targets"].index(row["target"]),
                int(row["polarity"] > 0),
                int(row["actor"] == "B"),
            )
            actual = np.repeat(
                raw["v"][row["groups"][0]].astype(float).mean(axis=0),
                len(raw["q"]) // raw["v"].shape[1],
                axis=0,
            )
            compare(actual, source_means[t, p, a])
            continue
        if row["stage"] == "reference":
            compare(
                raw["raw_query"], refs[manifest["spec"]["targets"].index(row["target"])]
            )
            continue
        c, corner, q = (
            row["context"],
            row["corner"],
            manifest["spec"]["targets"].index(row["query"]),
        )
        means_by_body[c, corner] = np.stack(
            [
                np.repeat(
                    raw["v"][group].astype(float).mean(axis=0),
                    len(raw["q"]) // raw["v"].shape[1],
                    axis=0,
                )
                for group in row["groups"]
            ],
            axis=1,
        ).astype(float)
        unrotated = refs[q]
        rotated = np.concatenate(
            (
                -unrotated[:, unrotated.shape[1] // 2 :],
                unrotated[:, : unrotated.shape[1] // 2],
            ),
            axis=1,
        )
        reference = (unrotated * raw["last_cos"] + rotated * raw["last_sin"]) * float(
            raw["scaling"]
        )
        for view, query in enumerate((raw["q"], reference)):
            for head in range(len(query)):
                ratio = len(query) // raw["v"].shape[1]
                kh, vh = (
                    raw["k"][:, head // ratio].astype(float),
                    raw["v"][:, head // ratio].astype(float),
                )
                logits = np.sum(kh * query[head].astype(float), axis=1)
                exp = np.exp(logits - logits.max())
                prob = exp / exp.sum()
                means = np.stack([vh[g].mean(axis=0) for g in row["groups"]])
                mg = np.array([sum(prob[i] for i in g) for g in row["groups"]])
                total = np.array(
                    [
                        sum(prob[i] * vh[i, d] for g in row["groups"] for i in g)
                        for d in range(vh.shape[1])
                    ]
                )
                event = np.sum(mg[:, None] * means, axis=0)
                pair = total - event
                uniform = vh[np.concatenate(row["groups"])].mean(axis=0)
                actual = np.stack(
                    (
                        total,
                        total / mg.sum(),
                        event,
                        pair,
                        event / mg.sum(),
                        pair / mg.sum(),
                        uniform,
                    )
                )
                compare(actual, vectors[c, corner, q, view, :, head])
                compare(mg, masses[c, corner, q, view, head])
            direct = np.array(
                [
                    [
                        sum((vectors[c, corner, q, view, channel] * d).ravel())
                        for d in dirs
                    ]
                    for channel in range(len(READOUTS))
                ]
            )
            compare(direct, projected[c, corner, q, view])
    cube = np.array(list(itertools.product((-1, 1), repeat=4)), float)

    def records(name):
        with (out / "02" / name).open() as stream:
            return list(csv.DictReader(stream))

    for row in records("query_selectivity.csv"):
        c, view = int(row["context"]), int(row["view"] == "neutral_reference")
        channel, direction = (
            READOUTS.index(row["readout"]),
            targets.index(row["target"]),
        )
        g = 1 if row["actor"] == "A" else -1
        ix = cube[:, 2] == g
        s = projected[c, ix, :, view, channel, direction]
        effect = np.sum(s * cube[ix, 1, None], axis=0) / 4
        match = effect[direction]
        other = (effect.sum() - match) / 5
        compare(
            np.array([match, other, match - other]),
            np.array(
                [
                    float(row[k])
                    for k in ("matched_action", "other_query_mean", "selectivity")
                ]
            ),
        )
    epsilon = manifest["spec"]["epsilon"]
    for row in records("author_composition.csv"):
        c, q = int(row["context"]), targets.index(row["query"])
        view, channel, direction = (
            int(row["view"] == "neutral_reference"),
            READOUTS.index(row["readout"]),
            targets.index(row["direction"]),
        )
        values = projected[c, :, q, view, channel, direction]
        char = cube[:, 1] * cube[:, 2]
        pplus, pminus = (1 + epsilon * char) / 16, (1 - epsilon * char) / 16
        coefficient = sum(char * values) / 16
        count = cube[:, 3] == -cube[:, 2]
        equal_count = sum(values[count] * char[count]) * 2 * epsilon / 8
        compare(
            np.array(
                [sum(pplus * values) - sum(pminus * values), coefficient, equal_count]
            ),
            np.array(
                [
                    float(row[k])
                    for k in (
                        "law_contrast",
                        "yg_coefficient",
                        "equal_name_count_law_contrast",
                    )
                ]
            ),
        )
    for row in records("finite_actions.csv"):
        c, q = int(row["context"]), targets.index(row["query"])
        view, low, high, direction = (
            int(row["view"] == "neutral_reference"),
            int(row["low_corner"]),
            int(row["high_corner"]),
            targets.index(row["direction"]),
        )
        ml, mh = masses[c, low, q, view], masses[c, high, q, view]
        vl, vh = means_by_body[c, low], means_by_body[c, high]
        value = np.sum(((mh + ml) / 2)[..., None] * (vh - vl), axis=1)
        allocation = np.sum((mh - ml)[..., None] * (vh + vl) / 2, axis=1)
        pairing = vectors[c, high, q, view, 3] - vectors[c, low, q, view, 3]
        actual = vectors[c, high, q, view, 0] - vectors[c, low, q, view, 0]
        scalars = np.array(
            [np.sum(a * dirs[direction]) for a in (value, allocation, pairing, actual)]
        )
        compare(
            scalars,
            np.array(
                [float(row[k]) for k in ("value", "allocation", "pairing", "actual")]
            ),
        )
    if maximum > manifest["spec"]["arithmetic_tolerance"]:
        raise ValueError("Alternate arithmetic disagreement: " + str(maximum))
    summary = {
        "status": "VERIFIED_FINITE_HM86_NOT_GLOBAL",
        "alternate_scalar_checks": int(checks),
        "max_error": maximum,
        "units": len(rows),
        "no_upper_or_answer_execution": True,
        "calibration_before_application": (out / "01/calibration.json")
        .stat()
        .st_mtime_ns
        < (out / "01/units/unit_0030.json").stat().st_mtime_ns,
        "same_agent_not_independent_rebuilder": True,
        "original_scope_complete": False,
    }
    if not summary["calibration_before_application"]:
        raise ValueError("Source calibration was not sealed first")
    path = out / "verification.json"
    if path.exists():
        if read_json(path) != summary:
            raise ValueError("Previous verification changed")
    else:
        write(path, summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    verify(parser.parse_args().out)
