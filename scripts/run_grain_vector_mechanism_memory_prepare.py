#!/usr/bin/env python3
"""Choose 12 development currents, frozen retrieval and prior-only memory donors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context import QUESTIONS, match_donors
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.data_views import stable_key
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_baselines import MODELS, encode_local
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Use only source IDs/material and a frozen E5 retrieval rule, no outcomes."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--e5-parent", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="mps")
    a = p.parse_args()
    source = json.loads(a.views.read_text(encoding="utf-8"))
    parent = json.loads((a.e5_parent / "manifest.json").read_text(encoding="utf-8"))
    if parent["source_view_sha256"] != material_hash(source) or any(
        b["split"] != "discovery" for b in source
    ):
        raise ValueError("Only identical discovery material is permitted")
    if (
        json.loads((a.e5_parent / "run_status.json").read_text(encoding="utf-8"))[
            "status"
        ]
        != "COMPLETE"
    ):
        raise ValueError("Incomplete E5 source")
    if parent["authors"] != [b["author_id"] for b in source]:
        raise ValueError("Encoder source author order mismatch")
    comparison = [b for b in source if b["discovery_role"] == "comparison"]
    chosen = sorted(
        comparison,
        key=lambda b: stable_key(b["author_id"], "grain_vm03_memory_20260906"),
    )[:6]
    pool = [
        {
            "author_id": b["author_id"],
            "split": "discovery_comparison",
            "history": b["views"][0],
            "current": b["views"][2][:2],
        }
        for b in comparison
    ]
    matches = match_donors(pool, lambda text: len(text.split()))
    by_id = {b["author_id"]: b for b in pool}
    unit_vectors = np.load(a.e5_parent / "unit_vectors.npz")["units"]
    cases, queries = [], []
    for b in chosen:
        for ci, current in enumerate(b["views"][2][:2]):
            cases.append(
                {
                    "case_id": f"m{len(cases):02d}",
                    "author_id": b["author_id"],
                    "current_index": ci,
                    "current": current,
                    "history": b["views"][0],
                }
            )
            queries.extend(
                QUESTIONS[task] + "\nCurrent writing:\n" + current["body"]
                for task in QUESTIONS
            )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    q, encoder = encode_local(queries, MODELS["e5_large"], a.device)
    for i, c in enumerate(cases):
        uid = c["author_id"]
        c["retrieval"] = {}
        index = parent["authors"].index(uid)
        for j, task in enumerate(QUESTIONS):
            scores = unit_vectors[index, :12] @ q[2 * i + j]
            top = np.argsort(-scores, kind="stable")[:3]
            c["retrieval"][task] = {
                "indices": sorted(map(int, top)),
                "scores": scores.tolist(),
            }
        c["random3"] = sorted(
            sorted(
                range(12),
                key=lambda j: stable_key(
                    c["history"][j]["comment_id"], c["case_id"] + "_random"
                ),
            )[:3]
        )
        c["donors"] = {}
        for rule in ("other_length", "other_condition"):
            match = next(
                r for r in matches if r["author_id"] == uid and r["rule"] == rule
            )
            c["donors"][rule] = (
                {**match, "history": by_id[match["donor_id"]]["history"]}
                if match["status"] == "MATCHED"
                else match
            )
    write_json(a.output_dir / "cases.json", cases)
    write_json(
        a.output_dir / "summarization_sources.json",
        [{"author_id": b["author_id"], "history": b["views"][0]} for b in chosen],
    )
    manifest = {
        "protocol": "GRAIN_VM03_NATURAL_MEMORY_V1",
        "source_view_sha256": material_hash(source),
        "case_sha256": material_hash(cases),
        "e5_parent_identity": parent["identity"],
        "authors": len(chosen),
        "currents": len(cases),
        "donor_pool": len(pool),
        "selection": "Six comparison authors by grain_vm03_memory_20260906 ID hash; first two view-2 comments",
        "donor_matching_units": "whitespace words, not exact tokens; prior/thread separation required",
        "query_encoder": encoder,
        "labels_used": False,
        "code_sha256": code_fingerprint(ROOT),
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("query_encoder",)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
