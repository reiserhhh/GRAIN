#!/usr/bin/env python3
"""Exploratory response-space comparison after HM105's vector-span result."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_qualified_transport import vector_cosine
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_text_response_hm104 as T

OUT = ROOT / "results/grain_math_response_v1/qualified_transport_r1"


def sectors(table):
    """Keep all four orthogonal product-design components, without selection."""
    f = np.asarray(table, float)
    if f.ndim != 2 or not f.size or not np.isfinite(f).all():
        raise ValueError("Finite complete action-by-target table required")
    grand = np.full_like(f, f.mean())
    action = np.broadcast_to(f.mean(1, keepdims=True), f.shape) - grand
    target = np.broadcast_to(f.mean(0, keepdims=True), f.shape) - grand
    coupled = f - grand - action - target
    return {"global": grand, "action": action, "target": target, "coupled": coupled}


def main():
    """Describe response relations using sealed actual tables, not predictor fit."""
    if (
        H.load(OUT / "verification.json")["status"]
        != "VERIFIED_FINITE_HM105_NOT_GLOBAL"
    ):
        raise ValueError("Verified actual text experiment required")
    H.verify_hashes(OUT, H.load(OUT / "output_seal.json"))
    dest = OUT / "04_response_sectors"
    dest.mkdir()
    H.write(
        dest / "manifest.json",
        {
            "classification": "EXPLORATORY_AFTER_PRIMARY_HM105_RESULTS_OPENED",
            "question": "Can observable target-specific relations agree despite weak raw-carrier span transport?",
            "basis": "Four complete orthogonal product-design sectors,all retained,no selection",
            "source": {
                "summary.json": H.sha(OUT / "summary.json"),
                "03/interactions.csv": H.sha(OUT / "03/interactions.csv"),
            },
            "code": {
                "scripts/analyze_grain_qualified_response_sectors_hm105.py": H.sha(
                    Path(__file__)
                ),
                "grain_research/construction/math_qualified_transport.py": H.sha(
                    ROOT / "grain_research/construction/math_qualified_transport.py"
                ),
            },
            "new_fits": 0,
            "new_model_calls": 0,
        },
    )
    targets = H.load(OUT / "00/reader.json")["targets"]
    tables = {
        (b, k, c): np.full((6, 6), np.nan)
        for b, k, c in itertools.product(range(3), range(2), range(2))
    }
    with (OUT / "03/interactions.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["source_span"] == "s7":
                key = int(row["bank"]), int(row["bridge"]), int(row["carrier"])
                tables[key][int(row["axis"]), targets.index(row["target"])] = float(
                    row["actual"]
                )
    split, source_only = {}, {}
    rebuild, orthogonality, parseval = 0.0, 0.0, 0.0
    for key, table in tables.items():
        parts = sectors(table)
        split[key] = parts
        alternate_coupled = (
            table
            - table.mean(1, keepdims=True)
            - table.mean(0, keepdims=True)
            + table.mean()
        )
        rebuild = max(
            rebuild,
            float(np.max(np.abs(sum(parts.values()) - table))),
            float(np.max(np.abs(parts["coupled"] - alternate_coupled))),
        )
        for a, b in itertools.combinations(parts.values(), 2):
            orthogonality = max(orthogonality, abs(float(np.vdot(a, b))))
        parseval = max(
            parseval,
            abs(float(np.sum(table**2) - sum(np.sum(p**2) for p in parts.values()))),
        )
        if key[0] == 0:
            source_only[key] = {
                "full_norm": float(np.linalg.norm(table)),
                "sector_norms": {n: float(np.linalg.norm(p)) for n, p in parts.items()},
            }
    rows = []
    for bank, bridge, carrier in itertools.product(range(3), range(2), range(2)):
        source, application = split[0, bridge, carrier], split[bank, bridge, carrier]
        full_source, full_application = (
            tables[0, bridge, carrier],
            tables[bank, bridge, carrier],
        )
        for name, dim in (("global", 1), ("action", 5), ("target", 5), ("coupled", 25)):
            a, b = source[name], application[name]
            rows.append(
                {
                    "bank": bank,
                    "bridge": bridge,
                    "carrier": carrier,
                    "sector": name,
                    "dimension": dim,
                    "cosine": vector_cosine(a, b),
                    "source_norm": float(np.linalg.norm(a)),
                    "application_norm": float(np.linalg.norm(b)),
                    "source_energy_share": float(np.sum(a**2) / np.sum(full_source**2)),
                    "application_energy_share": float(
                        np.sum(b**2) / np.sum(full_application**2)
                    ),
                    "full_table_cosine": vector_cosine(full_source, full_application),
                }
            )
    H.table(dest / "sectors.csv", rows)
    summary = {
        "status": "VERIFIED_EXPLORATORY_RESPONSE_SECTORS_NOT_PSYCHOLOGICAL_VALIDITY",
        "classification": "AFTER_PRIMARY_RESULTS",
        "new_model_calls": 0,
        "new_predictor_fits": 0,
        "rebuild_max": rebuild,
        "orthogonality_max": orthogonality,
        "parseval_max": parseval,
        "source_only_norms": {str(k): v for k, v in source_only.items()},
        "banks": {},
    }
    for bank in range(3):
        summary["banks"][str(bank)] = {
            name: {
                "cosine": T.stat(
                    [
                        r["cosine"]
                        for r in rows
                        if r["bank"] == bank and r["sector"] == name
                    ]
                ),
                "energy_share": T.stat(
                    [
                        r["application_energy_share"]
                        for r in rows
                        if r["bank"] == bank and r["sector"] == name
                    ]
                ),
            }
            for name in ("global", "action", "target", "coupled")
        }
        summary["banks"][str(bank)]["full_table_cosine"] = T.stat(
            [
                r["full_table_cosine"]
                for r in rows
                if r["bank"] == bank and r["sector"] == "global"
            ]
        )
    if max(rebuild, orthogonality, parseval) > 1e-10:
        raise ValueError("Complete-sector arithmetic identity failed")
    H.write(dest / "summary.json", summary)
    H.write(OUT / "response_sector_seal.json", H.seals(OUT, dest))
    print("HM105 RESPONSE SECTORS", summary, flush=True)


if __name__ == "__main__":
    main()
