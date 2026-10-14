#!/usr/bin/env python3
"""Read-only HM24 summaries: fiber effects, tangent failure, and query covectors."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_decoder_fibers import split_gradient
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_fibers_r2 import OUT, check
from scripts.run_grain_math_response import write_csv, write_json

DEFAULT = ROOT / "results/grain_math_response_v1/decoder_fibers_analysis_r1"


def rms(x):
    """Return a physical-unit RMS, preserving zero rather than inflating ratios."""
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def cosine(x, y):
    """Return a geometric cosine or None when the requested direction is zero."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / norm) if norm else None


def analyze(packet, out):
    """Aggregate all fixed units without fitting or selecting a causal direction."""
    check(packet)
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, sha in ledger.items():
        if digest(packet / name) != sha:
            raise ValueError("HM24 result changed: " + name)
    if out.exists():
        raise FileExistsError("preserve the analysis output")
    out.mkdir(parents=True)
    (out / "analysis.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "primary_artifact_hashes.json", ledger)
    records = [
        read_json(packet / "01/units" / f"unit_{r['hm24_unit']:03d}.json")
        for r in read_json(packet / "00/design.json")
    ]
    gradients, geometry, effects, property_rows, sharing = {}, [], [], [], []
    for r in records:
        address = r["address"]
        with np.load(packet / "01/units" / f"unit_{address['hm24_unit']:03d}.npz") as p:
            g = {name: p["g_" + name].copy() for name in address["capture"]["spans"]}
        pieces = {
            name: split_gradient(v, 3 if name.endswith("property") else 0)
            for name, v in g.items()
        }
        gradients[address["family"], address["query"]] = (g, pieces)
        total = sum(float(np.sum(v * v)) for v in g.values())
        for name, piece in pieces.items():
            energy = float(np.sum(g[name] ** 2))
            null_energy = float(np.sum(piece["null"] ** 2))
            geometry.append(
                {
                    "unit": address["hm24_unit"],
                    "family": address["family"],
                    "role": address["role"],
                    "query": address["query"],
                    "span": name,
                    "tokens": len(g[name]),
                    "rank": piece["rank"],
                    "gradient_energy": energy,
                    "null_gradient_energy": null_energy,
                    "fraction_of_six_span_gradient_energy": energy / total,
                    "null_fraction_within_span": null_energy / energy
                    if energy
                    else None,
                }
            )
    roles = ("all", "source", "validation", "application")
    queries = check(packet)["spec"]["queries"]
    for role in roles:
        for query in queries:
            selected = [
                r
                for r in records
                if r["address"]["query"] == query
                and (role == "all" or r["address"]["role"] == role)
            ]
            for arm in ("visible", "null", "random_null"):
                directions = [
                    next(d for d in r["directions"] if d["arm"] == arm)
                    for r in selected
                ]
                actual = np.array([d["central_action"] for d in directions])
                tangent = np.array([d["gradient_action"] for d in directions])
                leak = np.array([d["visible_leak_action"] for d in directions])
                effects.append(
                    {
                        "role": role,
                        "query": query,
                        "arm": arm,
                        "n": len(selected),
                        "native_rms": rms(actual),
                        "gradient_rms": rms(tangent),
                        "gradient_error_rms": rms(actual - tangent),
                        "visible_leak_rms": rms(leak),
                        "max_visible_leak_abs": float(np.max(abs(leak))),
                        "max_moment_change": max(
                            d["moment_change_max"] for d in directions
                        ),
                        "sign_agreements": int(np.sum(actual * tangent > 0)),
                        "minimum_native_abs": float(np.min(abs(actual))),
                    }
                )
            native = np.array([r["full_native_action"] for r in selected])
            tangent = np.array([r["full_tangent_action"] for r in selected])
            property_rows.append(
                {
                    "role": role,
                    "query": query,
                    "n": len(selected),
                    "native_rms": rms(native),
                    "tangent_rms": rms(tangent),
                    "tangent_error_rms": rms(native - tangent),
                    "tangent_native_r": float(np.corrcoef(native, tangent)[0, 1]),
                    "sign_agreements": int(np.sum(native * tangent > 0)),
                    "maximum_pullback_difference": max(
                        abs(r["full_tangent_action"] - r["full_moment_pullback_action"])
                        for r in selected
                    ),
                }
            )
    for family in sorted({r["address"]["family"] for r in records}):
        for q1, q2 in itertools.combinations(queries, 2):
            (g1, p1), (g2, p2) = gradients[family, q1], gradients[family, q2]
            for kind in ("full", "visible", "null"):
                a = np.concatenate(
                    [(g1[n] if kind == "full" else p1[n][kind]).ravel() for n in g1]
                )
                b = np.concatenate(
                    [(g2[n] if kind == "full" else p2[n][kind]).ravel() for n in g2]
                )
                sharing.append(
                    {
                        "family": family,
                        "query_1": q1,
                        "query_2": q2,
                        "kind": kind,
                        "cosine": cosine(a, b),
                    }
                )
    write_csv(out / "span_gradient_geometry.csv", geometry)
    write_csv(out / "finite_fiber_effects.csv", effects)
    write_csv(out / "property_tangent_summary.csv", property_rows)
    write_csv(out / "cross_query_geometry.csv", sharing)
    summary = {
        "status": "ANALYZED_FINITE_HM24_NOT_GLOBAL",
        "units": len(records),
        "geometric_rows": len(geometry),
        "effect_rows": len(effects),
        "property_rows": len(property_rows),
        "sharing_rows": len(sharing),
        "all_units_property_sign_agreements": sum(
            r["full_native_action"] * r["full_tangent_action"] > 0 for r in records
        ),
        "new_model_calls": 0,
        "new_fits": 0,
        "selection": "first pair per family: A axis, polarity0, order0; not the full old design",
        "geometry_not_psychological_variance": True,
        "null_direction_adaptive_not_a_prediction_algorithm": True,
        "reference_context_fixed_off_manifold_patches": True,
    }
    write_json(out / "summary.json", summary)
    write_json(
        out / "artifact_hashes.json",
        {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    )
    print(summary, flush=True)
    print(
        {
            "application_effects": [r for r in effects if r["role"] == "application"],
            "application_properties": [
                r for r in property_rows if r["role"] == "application"
            ],
        },
        flush=True,
    )


def main():
    """Write one bounded numerical diagnosis, preserving the primary packet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    analyze(args.packet, args.out)


if __name__ == "__main__":
    main()
