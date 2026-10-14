#!/usr/bin/env python3
"""Replay HM10B2 companion statistics with alternate grouping/count formulas."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json, write_json


def table(path):
    """Read output addresses without importing companion calculations."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify(source, diagnostic, out):
    """Recompute energy, fixed-condition effects and family-CI endpoints."""
    if out.exists():
        raise FileExistsError("preserve previous verification")
    checks, error = defaultdict(int), 0.0

    def compare(actual, expected):
        nonlocal error
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite or mismatched verification values")
        error = max(error, float(np.max(abs(a - b))))

    for directory in (source / "03", diagnostic):
        ledger = read_json(directory / "artifact_hashes.json")
        base = source if directory.name == "03" else diagnostic
        for name, expected in ledger.items():
            if digest(base / name) != expected:
                raise ValueError(f"output changed: {name}")
            checks["hashes"] += 1
    manifest = read_json(diagnostic / "diagnostic_manifest.json")
    code = ROOT / "scripts/diagnose_grain_math_target_orbits.py"
    if digest(code) != manifest["diagnostic_code_sha256"]:
        raise ValueError("diagnostic source changed")
    chars = table(source / "02/characters.csv")
    groups = defaultdict(dict)
    for row in chars:
        groups[row["role"], row["query"], int(row["term"])][int(row["family"])] = row
    for row in table(diagnostic / "character_spectra.csv"):
        cells = list(groups[row["role"], row["query"], int(row["term"])].values())
        c = np.array([json.loads(r["natural_coefficient"]) for r in cells])
        n = np.array([float(r["native_coefficient"]) for r in cells])
        coherent = float(np.einsum("i,i->", c.sum(axis=0), c.sum(axis=0)) / len(c) ** 2)
        ne = float(np.mean([float(r["natural_energy"]) for r in cells]))
        se = float(np.mean([float(r["native_energy"]) for r in cells]))
        compare(
            [ne, coherent, coherent / ne, se, n.mean(), n.mean() ** 2 / se],
            [
                float(row[k])
                for k in (
                    "natural_mean_energy",
                    "natural_mean_coefficient_energy",
                    "natural_coherence_ratio",
                    "native_mean_energy",
                    "native_mean_coefficient",
                    "native_coherence_ratio",
                )
            ],
        )
        checks["spectrum_values"] += 6
    partition = {
        "main": {1, 2, 4, 8, 16},
        "matched_owner_binding": {5, 10},
        "other_order_interaction": set(range(17, 32)),
        "other_nonorder_interaction": set(range(1, 16)) - {1, 2, 4, 5, 8, 10},
    }
    for row in table(diagnostic / "component_energy.csv"):

        def energy(terms, view=row):
            return sum(
                np.mean(
                    [
                        float(r[f"{view['space']}_energy"])
                        for r in groups[view["role"], view["query"], t].values()
                    ]
                )
                for t in terms
            )

        part = energy(partition[row["component"]])
        total = energy(range(1, 32))
        compare(
            [part, total, part / total],
            [float(row[k]) for k in ("energy", "total_centered_energy", "fraction")],
        )
        checks["component_energy_values"] += 3
    readouts = table(source / "02/readouts.csv")
    cubes = defaultdict(dict)
    for row in readouts:
        if row["kind"] == "full":
            cubes[int(row["family"]), row["query"]][int(row["mask"])] = float(
                row["native"]
            )
    for row in table(diagnostic / "fixed_self_trust_cross_effects.csv"):
        base = 2 * int(row["trust_polarity"]) + 16 * int(row["sentence_order"])
        cube = cubes[int(row["family"]), "pre_trust_positive"]
        a, b = cube[base + 1] - cube[base], cube[base + 5] - cube[base + 4]
        compare(
            [a, b, a - b],
            [
                float(row[k])
                for k in ("self_affect_delta", "other_affect_delta", "binding_contrast")
            ],
        )
        checks["fixed_cross_values"] += 3
    split = table(source / "02/conditional_link_split.csv")
    for row in table(diagnostic / "binding_link_split.csv"):
        cells = [
            r for r in split if r["query"] == row["query"] and r["axis"] == row["axis"]
        ]
        compare(
            [
                sum(float(r[k]) for r in cells) / len(cells)
                for k in ("binding_contrast", "link_binding", "representation_binding")
            ],
            [
                float(row[k])
                for k in ("binding_contrast", "link_binding", "representation_binding")
            ],
        )
        checks["binding_split_values"] += 3
    primary = read_json(source / "00/manifest.json")
    values = np.asarray(primary["alphabet"]["values"])
    basis = hadamard(32).astype(float)
    for qi, query in enumerate(primary["spec"]["queries"]):
        c = np.array(
            [
                [
                    json.loads(groups["source", query, t][f]["natural_coefficient"])
                    for t in range(32)
                ]
                for f in range(8)
            ]
        )
        # A complete Hadamard law independently reconstructs the actual logits.
        cold_c = np.array(
            [
                [
                    json.loads(
                        groups["application", query, t][f]["natural_coefficient"]
                    )
                    for t in range(32)
                ]
                for f in range(12, 16)
            ]
        )
        cold_z = np.einsum("gt,ftd->fgd", basis, cold_c)
        actual = np.array(
            [[cubes[f, query][g] for g in range(32)] for f in range(12, 16)]
        )
        rng = np.random.default_rng(2012 + qi)
        gains = []
        for _ in range(2000):
            source_counts = np.bincount(rng.integers(0, 8, 8), minlength=8) / 8
            coefficients = np.einsum("f,ftd->td", source_counts, c)
            losses = []
            for terms in (
                [1, 2, 4, 8, 16],
                [1, 2, 4, 8, 16, 5, 10],
                list(range(1, 32)),
            ):
                delta = (basis[:, terms] - 1) @ coefficients[terms]
                prediction = softmax(cold_z[:, :1] + delta, axis=-1) @ values
                losses.append(((prediction[:, 1:] - actual[:, 1:]) ** 2).mean(axis=1))
            app_counts = np.bincount(rng.integers(0, 4, 4), minlength=4) / 4
            gains.append(
                [
                    app_counts @ (losses[0] - losses[2]),
                    app_counts @ (losses[1] - losses[2]),
                ]
            )
        ends = np.quantile(gains, [0.025, 0.975], axis=0).T
        rows = [
            r
            for r in table(diagnostic / "secondary_full_law_intervals.csv")
            if r["query"] == query
        ]
        compare(ends, [[float(r["ci_low"]), float(r["ci_high"])] for r in rows])
        checks["secondary_ci_endpoints"] += 4
    if error > 1e-12:
        raise ValueError(f"diagnostic disagreement: {error}")
    out.mkdir(parents=True)
    result = {
        "status": "PASS",
        "checks": dict(checks),
        "max_error": error,
        "new_model_calls": 0,
        "same_agent_alternate_formula": True,
        "independent_human_review": False,
    }
    write_json(out / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/target_orbits_r2",
    )
    p.add_argument(
        "--diagnostic",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/target_orbits_diagnostic_r1",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "results/grain_math_response_v1/target_orbits_diagnostic_verification_r1",
    )
    a = p.parse_args()
    verify(a.source, a.diagnostic, a.out)
