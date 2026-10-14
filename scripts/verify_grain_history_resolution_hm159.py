#!/usr/bin/env python3
"""Rational/ordered-event and factorial-moment audit, without scientific edits."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import sys
import time
from fractions import Fraction as F
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    """Read-only complete artifact hash."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posterior(frequency, trials, k):
    """Alternate factorial-moment implementation, not the Bernstein compiler."""
    moments = [
        sum(
            float(frequency[s]) * comb(s, j) / comb(trials, j)
            for s in range(j, trials + 1)
        )
        for j in range(trials + 1)
    ]

    def integral(power):
        return sum(
            (-1) ** j * comb(4 - k, j) * moments[k + j + power] for j in range(5 - k)
        )

    denominator = integral(0)
    if denominator <= 0:
        return None, None
    mean = integral(1) / denominator
    variance = integral(2) / denominator - mean**2 if trials >= 6 else None
    return mean, variance


def main():
    """Audit all source records, cold profiles, exact laws and seals."""
    start = time.monotonic()
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "results/grain_math_response_v1/history_resolution_r1"
    )
    manifest = json.loads((out / "manifest.json").read_text())
    for path, expected in (manifest["code"] | manifest["inputs"]).items():
        assert digest(Path(path)) == expected, path
    assert digest(out / "alphabet.npz") == manifest["alphabet_sha256"]
    summary = json.loads((out / "summary.json").read_text())
    for path, expected in summary["outputs"].items():
        assert digest(out / path) == expected, path
    exact = json.loads((out / "exact.json").read_text())
    support = list(map(F, exact["support"]))
    masses = [list(map(F, exact[name])) for name in ("even_mass", "odd_mass")]
    comparisons = 0
    for bits in itertools.product((0, 1), repeat=4):
        probability = [
            sum(
                w * p ** sum(bits) * (1 - p) ** (4 - sum(bits))
                for p, w in zip(support, mass, strict=True)
            )
            for mass in masses
        ]
        assert probability[0] == probability[1]
        comparisons += 1
    joint_laws = []
    for mass in masses:
        law = [F(0)] * 7
        for sequence in itertools.product(range(4), repeat=3):
            # Address is (C,Y), not (C,R). Convert before counting two trials.
            successes = sum(
                address // 2 + ((address // 2) ^ (address % 2)) for address in sequence
            )
            probability = sum(
                w * p**successes * (1 - p) ** (6 - successes)
                for p, w in zip(support, mass, strict=True)
            )
            law[successes] += probability
        joint_laws.append(law)
    assert joint_laws[0] != joint_laws[1]
    for index, world in enumerate(("even", "odd")):
        assert list(map(str, joint_laws[index])) == exact["fingerprints"][world]["6"]
    source_checks, event_count, errors = 0, 0, []
    rows = list(csv.DictReader((out / "sampled_profiles.csv").open()))
    for directory in sorted(p for p in out.iterdir() if p.is_dir()):
        with np.load(directory / "observed_events.npz", allow_pickle=False) as data:
            fixed, free = data["fixed"], data["free"]
        assert np.all((fixed >= 0) & (fixed <= 3)) and np.all((free >= 0) & (free <= 3))
        r_fixed = (fixed // 2) ^ (fixed % 2)
        r_free = (free // 2) ^ (free % 2)
        observations = {
            "fixed4": (4, r_fixed[:, :4].sum(axis=1)),
            "fixed5": (5, r_fixed.sum(axis=1)),
            "joint3": (6, (free // 2 + r_free).sum(axis=1)),
            "choice_only3": (3, (free // 2).sum(axis=1)),
        }
        assert np.array_equal(
            fixed // 2, np.broadcast_to(np.arange(5) % 2, fixed.shape)
        )
        for design, (n, counts) in observations.items():
            bundle = json.loads((directory / f"{design}_bundle.json").read_text())
            np.testing.assert_array_equal(
                bundle["frequencies"],
                np.bincount(counts, minlength=n + 1) / len(counts),
            )
            source_checks += 1
            if n >= 5:
                world, seed, authors = directory.name.rsplit("_", 2)
                selected = [
                    r
                    for r in rows
                    if r["world"] == world
                    and r["seed"] == seed
                    and r["source_authors"] == authors
                    and r["design"] == design
                ]
                assert len(selected) == 5
                for row in selected:
                    mean, variance = posterior(
                        bundle["frequencies"], n, int(row["successes"])
                    )
                    errors.append(abs(mean - float(row["mean"])))
                    if variance is not None:
                        errors.append(abs(variance - float(row["variance"])))
        event_count += fixed.size + free.size
    assert event_count == summary["new_vector_events"]
    assert max(errors) < 1e-11
    a4 = np.array(
        [
            [comb(4, k) * float(p) ** k * (1 - float(p)) ** (4 - k) for p in support]
            for k in range(5)
        ]
    )
    frequency = a4 @ np.array(masses[0], float)
    for k in range(5):
        with np.load(out / f"finite_bound_{k}.npz", allow_pickle=False) as data:
            for witness in data["witnesses"]:
                np.testing.assert_allclose(a4 @ witness, frequency, atol=1e-10)
                assert abs(witness.sum() - 1) < 1e-10 and min(witness) >= -1e-10
    report = {
        "status": "PASS",
        "ordered_four_event_aliases": comparisons,
        "ordered_joint_sequences_per_world": 64,
        "source_frequency_checks": source_checks,
        "new_events_recounted": event_count,
        "profile_scalar_checks": len(errors),
        "max_factorial_vs_Bernstein_error": max(errors),
        "CPU_seconds": time.monotonic() - start,
        "new_forwards": 0,
        "audit_scope": "Same-agent alternate arithmetic and complete artifact hash checks; not blind independent reconstruction",
    }
    with (out / "alternate_audit.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
