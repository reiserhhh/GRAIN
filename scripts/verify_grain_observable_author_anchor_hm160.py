#!/usr/bin/env python3
"""Read-only rational/count/endpoint audit with alternative computations."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from fractions import Fraction as F
from math import isqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    """Complete read-only hash check."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational_array(table):
    """Parse exact saved fractions without using the primary math module."""
    return [[[F(x) for x in outcome] for outcome in condition] for condition in table]


def enumerate_law(support, mass, rates, selected=False):
    """Independent C/R/action/outcome enumeration of a full observation law."""
    result = [[[F(0)] * 2 for _ in range(2)] for _ in range(4)]
    for i, (p, w) in enumerate(zip(support, mass, strict=True)):
        for c in (0, 1):
            for r in (0, 1):
                history = w * (p if c else 1 - p) * (p if r else 1 - p)
                h = 2 * c + (c ^ r)
                for q in (0, 1):
                    assignment = F(int(q == i)) if selected else F(1, 2)
                    for y in (0, 1):
                        result[h][q][y] += (
                            history
                            * assignment
                            * (rates[i][q] if y else 1 - rates[i][q])
                        )
    return result


def exact_profiles(table):
    """Exact conditional ratios/reference/gain from a rational observed law."""
    history = [sum(sum(row, F(0)) for row in conditions) for conditions in table]
    assert sum(history) == 1
    values = [[row[1] / sum(row) for row in conditions] for conditions in table]
    reference = [sum(history[h] * values[h][q] for h in range(4)) for q in range(2)]
    gain = [
        sum(history[h] * (values[h][q] - reference[q]) ** 2 for h in range(4))
        for q in range(2)
    ]
    return history, values, reference, gain


def decode_exact(vectors, alphabet):
    """Alternative equality-address lookup, not the primary distance decoder."""
    equal = np.all(vectors[:, None, :] == alphabet[None, :, :], axis=-1)
    assert np.all(equal.sum(axis=1) == 1)
    return np.nonzero(equal)[1]


def main():
    """Check seals, all finite records, profiles and separate endpoint outcomes."""
    start = time.monotonic()
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "results/grain_math_response_v1/observable_author_anchor_r1"
    )
    manifest = json.loads((out / "manifest.json").read_text())
    for path, expected in (manifest["code"] | manifest["inputs"]).items():
        assert sha(path) == expected, path
    assert sha(out / "alphabet.npz") == manifest["alphabet_sha256"]
    summary = json.loads((out / "summary.json").read_text())
    for path, expected in summary["outputs"].items():
        assert sha(out / path) == expected, path
    exact = json.loads((out / "exact.json").read_text())
    p, w = [F(1, 20), F(9, 20), F(19, 20)], [F(1, 2), F(3, 10), F(1, 5)]
    law = enumerate_law(p, w, [[x, x * x] for x in p])
    assert law == rational_array(exact["main_joint"])
    latent = [x * x for x in p]
    inverse = [F(isqrt(x.numerator), isqrt(x.denominator)) for x in latent]
    assert enumerate_law(inverse, w, [[x, x * x] for x in inverse]) == law
    history, probability, reference, gain = exact_profiles(law)
    errors = []
    for h in range(4):
        for q in range(2):
            target = exact["main_profiles"][h]["targets"][q]
            errors.extend(
                (
                    abs(float(probability[h][q]) - target["probability"]),
                    abs(float(reference[q]) - target["reference"]),
                    abs(float(probability[h][q] - reference[q]) - target["relative"]),
                )
            )
    assert max(errors) < 1e-12
    assert np.allclose(
        list(map(float, gain)), exact["exact_endpoint_brier_gain"], atol=1e-14
    )
    # Latent coordinate statements are evaluation-only, never source bundle inputs.
    posterior = [w_i * x * (1 - x) / history[3] for x, w_i in zip(p, w, strict=True)]
    for name, coordinates in (("p", p), ("p_squared", latent)):
        prior = sum(a * b for a, b in zip(w, coordinates, strict=True))
        conditional = sum(a * b for a, b in zip(posterior, coordinates, strict=True))
        entry = exact["latent_gauge_witness"][name]
        assert F(entry["reference"]) == prior
        assert F(entry["conditional_mean"]) == conditional
        assert F(entry["relative"]) == conditional - prior
    alias_laws = []
    for world, rates in enumerate(
        (
            [[F(1, 5), F(3, 10)], [F(7, 10), F(4, 5)]],
            [[F(1, 5), F(9, 10)], [F(1, 10), F(4, 5)]],
        )
    ):
        laws = {}
        for selected, regime in (
            (False, "randomized_condition"),
            (True, "self_selected_condition"),
        ):
            alternative = enumerate_law(
                [F(1, 4), F(3, 4)], [F(1, 2)] * 2, rates, selected
            )
            assert alternative == rational_array(
                exact["selection_alias"][str(world)][regime]["table"]
            )
            laws[regime] = alternative
        alias_laws.append(laws)
    assert (
        alias_laws[0]["self_selected_condition"]
        == alias_laws[1]["self_selected_condition"]
    )
    assert (
        alias_laws[0]["randomized_condition"] != alias_laws[1]["randomized_condition"]
    )
    with np.load(out / "alphabet.npz", allow_pickle=False) as data:
        alphabet = data["alphabet"]
    source_authors, application_authors, scalar_checks = 0, 0, 0
    fields = {
        "version",
        "regime",
        "target_ids",
        "history_mass",
        "probabilities",
        "reference",
        "support",
    }
    for directory in sorted(out.glob("seed_*")):
        with np.load(directory / "source_observed.npz", allow_pickle=False) as data:
            assert set(data.files) == {"author_id", "history", "endpoint"}
            assert len(np.unique(data["author_id"])) == len(data["author_id"])
            source_ids = data["author_id"].copy()
            h = decode_exact(data["history"], alphabet)
            endpoints = decode_exact(data["endpoint"], alphabet)
        counts = Counter(
            (int(a), int(b // 2), int((b // 2) ^ (b % 2)))
            for a, b in zip(h, endpoints, strict=True)
        )
        rebuilt = np.array(
            [
                [[counts[(a, q, y)] for y in range(2)] for q in range(2)]
                for a in range(4)
            ]
        )
        with np.load(directory / "source_counts.npz", allow_pickle=False) as data:
            assert np.array_equal(rebuilt, data["counts"])
        probability = rebuilt[:, :, 1] / rebuilt.sum(axis=2)
        mass = rebuilt.sum(axis=(1, 2)) / rebuilt.sum()
        reference = mass @ probability
        program = json.loads((directory / "source_bundle.json").read_text())
        assert set(program) == fields
        errors.extend(
            np.ravel(
                np.abs(probability - np.asarray(program["probabilities"]))
            ).tolist()
        )
        errors.extend(np.abs(reference - np.asarray(program["reference"])).tolist())
        with np.load(directory / "application_input.npz", allow_pickle=False) as data:
            assert set(data.files) == {"author_id", "history"}
            assert not len(np.intersect1d(source_ids, data["author_id"]))
            app_h = decode_exact(data["history"], alphabet)
        cold = probability[app_h]
        with np.load(directory / "cold_predictions.npz", allow_pickle=False) as data:
            assert np.array_equal(cold, data["probabilities"])
        with np.load(
            directory / "application_evaluation_only.npz", allow_pickle=False
        ) as data:
            y = data["outcomes"]
            endpoint_address = decode_exact(
                data["endpoints"].reshape(-1, alphabet.shape[1]), alphabet
            ).reshape(-1, 2)
            assert np.array_equal(
                endpoint_address // 2, np.broadcast_to(np.arange(2), y.shape)
            )
            assert np.array_equal((endpoint_address // 2) ^ (endpoint_address % 2), y)
        seed = int(directory.name.split("_")[1])
        for q in range(2):
            item = next(
                row
                for row in summary["endpoint_checks"]
                if row["seed"] == seed and row["condition"] == q
            )
            brier = float((cold[:, q] ** 2 - 2 * cold[:, q] * y[:, q] + y[:, q]).mean())
            base = float(
                (reference[q] ** 2 - 2 * reference[q] * y[:, q] + y[:, q]).mean()
            )
            errors.extend(
                (
                    abs(item["brier"] - brier),
                    abs(item["fixed_reference_brier"] - base),
                    abs(item["brier_gain"] - (base - brier)),
                )
            )
        source_authors += len(h)
        application_authors += len(app_h)
        scalar_checks += cold.size
    assert source_authors == summary["source_authors"]
    assert application_authors == summary["new_application_authors"]
    assert max(errors) < 1e-12
    result = {
        "status": "PASS",
        "source_authors_recounted": source_authors,
        "application_authors_recounted": application_authors,
        "cold_scalar_checks": scalar_checks,
        "exact_tables_reenumerated": 5,
        "maximum_alternate_error": max(errors),
        "same_agent_not_blind": True,
        "wall_seconds": time.monotonic() - start,
    }
    destination = out / "alternate_audit.json"
    if not destination.exists():
        with destination.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        saved = json.loads(destination.read_text())
        assert {k: v for k, v in saved.items() if k != "wall_seconds"} == {
            k: v for k, v in result.items() if k != "wall_seconds"
        }
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
