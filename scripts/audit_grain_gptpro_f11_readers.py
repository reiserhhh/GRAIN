#!/usr/bin/env python3
"""Recompute the real-reader interface from input tables, not production helpers."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    """Load a CSV into dictionaries for a transparent numeric replay."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def corr(a, b):
    """Calculate Pearson from centered scalar sums without corrcoef."""
    a, b = np.asarray(a), np.asarray(b)
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def main():
    """Audit all scalar scores, matched edit differences and fixed-method ANOVA."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--lexical", required=True, type=Path)
    parser.add_argument("--nli", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = {
        name: {r["material_id"]: r for r in read(path)}
        for name, path in (("lexical", args.lexical), ("nli", args.nli))
    }
    scores = read(args.input / "paired_scores.csv")
    maximum = 0.0
    indexed = {}
    for r in scores:
        raw = original[r["reader"]][r["material_id"]]
        p, q = float(raw["pole_score"]), float(raw["qualification_score"])
        expected = {"pole": p, "qualification": q, "signed_qualified": 2 * p * q - q}[
            r["target"]
        ]
        maximum = max(maximum, abs(expected - float(r["score"])))
        indexed[
            (
                r["role"],
                r["target"],
                r["reader"],
                r["pole"],
                r["variant"],
                r["subject_status"],
            )
        ] = r
    edits = read(args.input / "edit_signatures.csv")
    for r in edits:
        key = tuple(r[k] for k in ("role", "target", "reader", "pole", "variant"))
        old = indexed[(*key, "actual")]
        new = indexed[(*key, r["edit"].removeprefix("actual_to_"))]
        for output, field in (
            ("observed_delta", "score"),
            ("designed_delta", "designed_target"),
        ):
            maximum = max(
                maximum, abs(float(new[field]) - float(old[field]) - float(r[output]))
            )
    for r in read(args.input / "fixed_method_decomposition.csv"):
        vectors = []
        for reader in ("lexical", "nli"):
            vectors.append(
                np.array(
                    [
                        float(x["score"])
                        for x in scores
                        if x["role"] == r["role"]
                        and x["target"] == r["target"]
                        and x["reader"] == reader
                    ]
                )
            )
        a, b = vectors
        middle, difference = (a + b) / 2, (a - b) / 2
        values = [np.var(middle), np.mean(difference) ** 2, np.var(difference)]
        for field, value in zip(
            (
                "material_variance",
                "reader_fixed_offset_variance",
                "material_reader_interaction_variance",
            ),
            values,
            strict=True,
        ):
            maximum = max(maximum, abs(value - float(r[field])))
    for r in read(args.input / "agreement.csv"):
        groups = [
            [
                x
                for x in scores
                if x["role"] == r["role"]
                and x["target"] == r["target"]
                and x["reader"] == reader
            ]
            for reader in ("lexical", "nli")
        ]
        a, b = [np.array([float(x["score"]) for x in group]) for group in groups]
        truth = np.array([float(x["designed_target"]) for x in groups[0]])
        values = {
            "reader_r": corr(a, b),
            "reader_rmse": np.sqrt(np.mean((a - b) ** 2)),
            "lexical_mse": np.mean((a - truth) ** 2),
            "nli_mse": np.mean((b - truth) ** 2),
            "error_r": corr(a - truth, b - truth),
        }
        for field, value in values.items():
            maximum = max(maximum, abs(value - float(r[field])))
    for r in read(args.input / "wording_halves.csv"):
        selected = [
            x for x in scores if all(x[k] == r[k] for k in ("role", "target", "reader"))
        ]
        keys = sorted({(x["pole"], x["subject_status"]) for x in selected})
        halves = [
            np.array(
                [
                    np.mean(
                        [
                            float(x["score"])
                            for x in selected
                            if (x["pole"], x["subject_status"]) == key
                            and int(x["variant"]) % 2 == parity
                        ]
                    )
                    for key in keys
                ]
            )
            for parity in (0, 1)
        ]
        maximum = max(
            maximum,
            abs(corr(*halves) - float(r["wording_half_r"])),
            abs(
                np.sqrt(np.mean((halves[0] - halves[1]) ** 2))
                - float(r["wording_half_rmse"])
            ),
        )
    if maximum > 1e-12:
        raise ValueError("reader replay failed")
    summary = {
        "status": "PASS_SCALAR_EDIT_DECOMPOSITION_REPLAY",
        "score_rows": len(scores),
        "edit_rows": len(edits),
        "max_error": maximum,
        "independent_agent": False,
        "scope": "scores, edits, variance decomposition, agreement and wording halves replayed",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
