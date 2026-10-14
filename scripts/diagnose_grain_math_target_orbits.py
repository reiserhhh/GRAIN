#!/usr/bin/env python3
"""Result-after mechanism diagnosis of frozen HM10B2; no new model calls."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json, write_json
from scripts.run_grain_math_response import write_csv

DEFAULT = ROOT / "results/grain_math_response_v1/target_orbits_r2"
NAMES = {1: "A", 2: "T", 4: "U", 8: "V", 16: "S"}
MAIN = [1, 2, 4, 8, 16]
BINDING = MAIN + [5, 10]
FULL = list(range(1, 32))


def table(path):
    """Read a frozen CSV without touching the primary output."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def term_name(term):
    """Name the address characters, not psychological constructs."""
    return ":".join(name for bit, name in NAMES.items() if term & bit) or "constant"


def component(term):
    """Partition nonconstant terms without choosing terms from their results."""
    if term == 0:
        raise ValueError("constant is not a centered-energy component")
    if term in MAIN:
        return "main"
    if term in (5, 10):
        return "matched_owner_binding"
    if term & 16:
        return "other_order_interaction"
    return "other_nonorder_interaction"


def score(theta, values):
    """Evaluate the fixed answer alphabet by stable exponentiation."""
    z = np.asarray(theta, float)
    exp = np.exp(z - z.max(axis=-1, keepdims=True))
    return (exp / exp.sum(axis=-1, keepdims=True)) @ values


def anchored(anchor, coefficients, terms):
    """Apply source character displacements while retaining the cold anchor."""
    if np.shape(coefficients) != (32, len(anchor)):
        raise ValueError("matched complete natural-coordinate coefficients required")
    basis = hadamard(32).astype(float)
    return anchor + (basis[:, terms] - basis[0, terms]) @ coefficients[terms]


def family_intervals(source, application, actual, values, seed=2012, draws=2000):
    """Resample source and application families for two fixed, secondary contrasts.

    These full-law comparisons were added after primary results were seen.
    They are diagnostics, not a new prespecified or independent confirmation.
    """
    source, application, actual = map(np.asarray, (source, application, actual))
    if len(source) != 8 or len(application) != 4 or actual.shape != (4, 32):
        raise ValueError("frozen 8-source/4-application family design required")
    rng = np.random.default_rng(seed)
    gains = np.zeros((draws, 2))
    for draw in range(draws):
        coeff = source[rng.integers(0, 8, 8)].mean(axis=0)
        losses = []
        for terms in (MAIN, BINDING, FULL):
            prediction = np.array(
                [score(anchored(z[0], coeff, terms), values) for z in application]
            )
            losses.append(((prediction - actual) ** 2)[:, 1:].mean(axis=1))
        chosen = rng.integers(0, 4, 4)
        gains[draw] = [
            (losses[0] - losses[2])[chosen].mean(),
            (losses[1] - losses[2])[chosen].mean(),
        ]
    return np.quantile(gains, [0.025, 0.975], axis=0).T


def diagnose(source, out):
    """Separate actual binding, scalar-link effects, order and cold completion."""
    if out.exists():
        raise FileExistsError("preserve earlier diagnostics")
    primary_hashes = read_json(source / "03/artifact_hashes.json")
    for name, expected in primary_hashes.items():
        if digest(source / name) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    m = read_json(source / "00/manifest.json")
    for name, expected in m["source_files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    queries = m["spec"]["queries"]
    values = np.asarray(m["alphabet"]["values"], float)
    out.mkdir(parents=True)
    write_json(
        out / "diagnostic_manifest.json",
        {
            "status": "RESULT_AFTER_DIAGNOSTIC",
            "source": str(source),
            "source_manifest_sha256": digest(source / "00/manifest.json"),
            "source_artifact_ledger_sha256": digest(source / "03/artifact_hashes.json"),
            "diagnostic_code_sha256": digest(Path(__file__)),
            "added_neural_calls": 0,
            "new_model_fit": False,
            "new_prediction_arms": False,
            "secondary_comparisons_not_prespecified": True,
            "bootstrap_draws": 2000,
            "bootstrap_seed": 2012,
        },
    )
    chars = table(source / "02/characters.csv")
    readouts = table(source / "02/readouts.csv")
    split = table(source / "02/conditional_link_split.csv")
    primary_binding = table(source / "03/binding_summary.csv")
    spectra, energies = [], []
    grouped = defaultdict(list)
    for r in chars:
        grouped[r["role"], r["query"], int(r["term"])].append(r)
    for (role, query, term), rows in grouped.items():
        c = np.array([json.loads(r["natural_coefficient"]) for r in rows])
        native = np.array([float(r["native_coefficient"]) for r in rows])
        natural_energy = float(np.mean(np.sum(c**2, axis=1)))
        scalar_energy = float(np.mean(native**2))
        spectra.append(
            {
                "role": role,
                "query": query,
                "term": term,
                "term_name": term_name(term),
                "degree": term.bit_count(),
                "component": component(term) if term else "constant",
                "families": len(rows),
                "natural_mean_energy": natural_energy,
                "natural_mean_coefficient_energy": float(np.sum(c.mean(axis=0) ** 2)),
                "natural_coherence_ratio": float(np.sum(c.mean(axis=0) ** 2))
                / natural_energy
                if natural_energy
                else None,
                "native_mean_energy": scalar_energy,
                "native_mean_coefficient": float(native.mean()),
                "native_coherence_ratio": float(native.mean() ** 2) / scalar_energy
                if scalar_energy
                else None,
            }
        )
    for role in ("source", "validation", "application"):
        for query in queries:
            rows = [
                r
                for r in spectra
                if r["role"] == role and r["query"] == query and r["term"]
            ]
            for name in sorted({r["component"] for r in rows}):
                selected = [r for r in rows if r["component"] == name]
                for space in ("natural", "native"):
                    total = sum(r[f"{space}_mean_energy"] for r in rows)
                    part = sum(r[f"{space}_mean_energy"] for r in selected)
                    energies.append(
                        {
                            "role": role,
                            "query": query,
                            "space": space,
                            "component": name,
                            "terms": len(selected),
                            "energy": part,
                            "total_centered_energy": total,
                            "fraction": part / total if total else None,
                        }
                    )
    law_split, max_split_error = [], 0.0
    for query in queries:
        for axis in ("anxiety", "trust"):
            selected = [r for r in split if r["query"] == query and r["axis"] == axis]
            primary = next(
                r for r in primary_binding if r["query"] == query and r["axis"] == axis
            )
            row = {"query": query, "axis": axis, "conditional_tables": len(selected)}
            for key in ("binding_contrast", "link_binding", "representation_binding"):
                row[key] = float(np.mean([float(r[key]) for r in selected]))
            err = max(
                abs(row["binding_contrast"] - float(primary["binding_contrast"])),
                abs(
                    row["binding_contrast"]
                    - row["link_binding"]
                    - row["representation_binding"]
                ),
            )
            max_split_error = max(max_split_error, err)
            row["reconstruction_error"] = err
            law_split.append(row)
    # Keep the trust evidence fixed to self while changing affect, person and order.
    cells = {
        (int(r["family"]), r["query"], int(r["mask"])): float(r["native"])
        for r in readouts
        if r["kind"] == "full"
    }
    fixed_cross, fixed_summary = [], []
    for family in range(16):
        role = (
            "source" if family < 8 else "validation" if family < 12 else "application"
        )
        for t in (0, 1):
            for s in (0, 1):
                base = 2 * t + 16 * s
                self_delta = (
                    cells[family, "pre_trust_positive", base | 1]
                    - cells[family, "pre_trust_positive", base]
                )
                other_delta = (
                    cells[family, "pre_trust_positive", base | 5]
                    - cells[family, "pre_trust_positive", base | 4]
                )
                fixed_cross.append(
                    {
                        "family": family,
                        "role": role,
                        "trust_polarity": t,
                        "sentence_order": s,
                        "trust_owner": "self",
                        "self_affect_delta": self_delta,
                        "other_affect_delta": other_delta,
                        "binding_contrast": self_delta - other_delta,
                    }
                )
    for t in (0, 1):
        for s in (0, 1):
            rows = [
                r
                for r in fixed_cross
                if r["trust_polarity"] == t and r["sentence_order"] == s
            ]
            fixed_summary.append(
                {
                    "trust_polarity": t,
                    "sentence_order": s,
                    "families": len(rows),
                    "self_affect_delta": float(
                        np.mean([r["self_affect_delta"] for r in rows])
                    ),
                    "other_affect_delta": float(
                        np.mean([r["other_affect_delta"] for r in rows])
                    ),
                    "binding_contrast": float(
                        np.mean([r["binding_contrast"] for r in rows])
                    ),
                    "negative_self_families": sum(
                        r["self_affect_delta"] < 0 for r in rows
                    ),
                    "negative_binding_families": sum(
                        r["binding_contrast"] < 0 for r in rows
                    ),
                }
            )
    coefficients, units, native_tables = {}, {}, {}
    for family in range(16):
        for query in queries:
            rows = [
                r for r in chars if int(r["family"]) == family and r["query"] == query
            ]
            coefficients[family, query] = np.array(
                [
                    json.loads(r["natural_coefficient"])
                    for r in sorted(rows, key=lambda r: int(r["term"]))
                ]
            )
            native_tables[family, query] = np.array(
                [cells[family, query, g] for g in range(32)]
            )
    for row in read_json(source / "00/design.json"):
        with np.load(
            source / "01/units" / f"unit_{row['unit']:04d}.npz", allow_pickle=False
        ) as a:
            z = a["logits"].astype(float)
        units[row["text_id"], row["query"]] = z - z.mean()
    doc_tables = defaultdict(dict)
    for row in read_json(source / "00/documents.json"):
        if row["kind"] == "full":
            for query in queries:
                doc_tables[row["family"], query][row["mask"]] = units[
                    row["text_id"], query
                ]
    intervals = []
    for qi, query in enumerate(queries):
        ci = family_intervals(
            [coefficients[f, query] for f in range(8)],
            [
                np.array([doc_tables[f, query][g] for g in range(32)])
                for f in range(12, 16)
            ],
            [native_tables[f, query] for f in range(12, 16)],
            values,
            seed=2012 + qi,
        )
        for baseline, ends in zip(
            ("logit_main", "logit_owner_binding"), ci, strict=True
        ):
            intervals.append(
                {
                    "query": query,
                    "baseline": baseline,
                    "arm": "logit_full_law",
                    "gain_units": "MSE_baseline_minus_MSE_full",
                    "ci_low": float(ends[0]),
                    "ci_high": float(ends[1]),
                    "result_after": True,
                    "source_families": 8,
                    "application_families": 4,
                }
            )
    if max_split_error > 1e-12:
        raise ValueError("link/representation diagnosis changed the primary contrast")
    outputs = {
        "character_spectra.csv": spectra,
        "component_energy.csv": energies,
        "binding_link_split.csv": law_split,
        "fixed_self_trust_cross_effects.csv": fixed_cross,
        "fixed_self_trust_cross_summary.csv": fixed_summary,
        "secondary_full_law_intervals.csv": intervals,
    }
    for name, rows in outputs.items():
        write_csv(out / name, rows)
    summary = {
        "status": "RESULT_AFTER_MECHANISM_DIAGNOSIS_COMPLETE",
        "added_model_calls": 0,
        "verified_primary_hashes": len(primary_hashes),
        "max_link_split_error": max_split_error,
        "binding_link_split": law_split,
        "fixed_self_trust_cross_effects": fixed_summary,
        "secondary_full_law_intervals": intervals,
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/target_orbits_diagnostic_r1",
    )
    args = parser.parse_args()
    diagnose(args.source, args.out)
