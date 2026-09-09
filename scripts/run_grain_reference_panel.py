#!/usr/bin/env python3
"""NAV-08: freeze a two-view, 12-reference/12-evaluation author panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    QUERIES,
    digest,
    read_json,
    tokenize_world,
    write_json,
)
from scripts.run_grain_evidence_packets import code_hashes, collect, evaluate
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = BASE / "navigation_v1/NAV-08/author_panel_r1"


def prepare(out):
    """Preserve old author roles and exact tokens when reusing actual states."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Choose a fresh reference packet")
    old = BASE / "shared_author_profiles_m25_v1"
    material = BASE / "multiquery_sensitivity_s1_r1"
    edits = BASE / "navigation_v1/NAV-04/natural_evidence_r1"
    sources = [
        old / "private_design.json",
        material / "private_material.json",
        edits / "private_design.json",
        edits / "private_worlds.json",
    ]
    if digest(sources[0]) != read_json(old / "manifest.json")["input_sha256"]:
        raise ValueError("Old author design changed")
    if digest(sources[1]) != read_json(material / "manifest.json")["material_sha256"]:
        raise ValueError("Old material changed")
    em = read_json(edits / "manifest.json")
    for path in sources[2:]:
        if digest(path) != em["input_sha256"][path.name]:
            raise ValueError("Natural source changed")
    design_old = read_json(sources[0])
    roles = {}
    for d in design_old:
        if d["author_id"] in roles and roles[d["author_id"]] != d["role"]:
            raise ValueError("Author role conflict")
        roles[d["author_id"]] = d["role"]
    material = read_json(sources[1])
    if len(material) != 48 or len(roles) != 24:
        raise ValueError("Expected24 authors/two views")
    tf = {
        (d["author_id"], d["view"]): d
        for d in design_old
        if d["query"] == "TF_cont" and d["condition"] == "full"
    }
    worlds_old = {w["world_id"]: w for w in read_json(sources[3])}
    reuse_edits = {
        (worlds_old[d["world_id"]]["author_id"], d["query"]): d
        for d in read_json(sources[2])
        if worlds_old[d["world_id"]]["condition"] == "original"
    }
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    worlds, design, reuses, tokens = [], [], [], {}
    out.mkdir(parents=True)
    (out / "private_packets").mkdir()
    (out / "reference_units").mkdir()
    for row in material:
        author, view = row["author_id"], row["view"]
        world = {
            "world_id": f"panel{row['panel_index']:02d}_view{view}",
            "author_id": author,
            "view": view,
            "role": roles[author],
            "texts": row["texts"],
        }
        worlds.append(world)
        for q in QUERIES:
            _, _, enc, mask, alphabet = tokenize_world(tok, row["texts"], q)
            d = {
                "index": len(design),
                "world_id": world["world_id"],
                "query": q,
                "input_ids": enc["input_ids"],
                "mask": mask.tolist(),
                "alphabet": alphabet,
            }
            qt = np.asarray(d["input_ids"])[mask].tolist()
            if q in tokens and tokens[q] != qt:
                raise ValueError("Within-query coordinates changed")
            tokens[q] = qt
            native, z, path = None, None, None
            if view == 0 and (author, q) in reuse_edits:
                previous = reuse_edits[author, q]
                path = edits / "private_packets" / f"unit_{previous['index']:04d}.json"
                meta, z = checked_arrays(path)
                if (
                    previous["input_ids"] != d["input_ids"]
                    or previous["mask"] != d["mask"]
                ):
                    raise ValueError("Natural reuse tokens mismatch")
                if any(meta[k] != previous[k] for k in ("index", "world_id", "query")):
                    raise ValueError("Natural reuse identity mismatch")
                native = meta["native"]
            elif q == "TF_cont":
                previous = tf[author, view]
                path = old / "private_packets" / f"cell_{previous['index']:03d}.json"
                meta, z = checked_arrays(path)
                if (
                    previous["input_ids"] != d["input_ids"]
                    or previous["question_mask"] != d["mask"]
                ):
                    raise ValueError("Old T/F reuse tokens mismatch")
                if any(
                    meta[k] != previous[k]
                    for k in (
                        "index",
                        "author_id",
                        "view",
                        "query",
                        "role",
                        "condition",
                    )
                ):
                    raise ValueError("Old T/F identity mismatch")
                native = {
                    "conditional_score": meta["native_score"],
                    "candidate_mass": None,
                }
            if path is not None:
                dest = out / "private_packets" / f"unit_{d['index']:04d}.json"
                save_unit(
                    dest,
                    {k: d[k] for k in ("index", "world_id", "query")}
                    | {
                        "native": native,
                        "reused_from": str(path.relative_to(ROOT)),
                        "source_sha256": digest(path),
                    },
                    P=z["P"],
                    Q=z["Q"],
                )
                reuses.append(
                    {
                        "index": d["index"],
                        "source": str(path.relative_to(ROOT)),
                        "sha256": digest(path),
                        "array_sha256": digest(path.with_suffix(".npz")),
                    }
                )
            design.append(d)
    for name, value in (
        ("private_worlds.json", worlds),
        ("private_design.json", design),
        ("queries.json", {}),
        ("private_eligibility.json", []),
    ):
        write_json(out / name, value)
    write_json(
        out / "manifest.json",
        {
            "protocol": "NAV08_FIXED_AUTHOR_REFERENCE_V1",
            "kind": "author_panel",
            "units": len(design),
            "worlds": len(worlds),
            "queries": list(QUERIES),
            "question_tokens": tokens,
            "synthetic_references": {},
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "code_sha256": code_hashes(),
            "prepare_code_sha256": digest(Path(__file__)),
            "model_sha256": digest(MODEL / "model.safetensors"),
            "input_sha256": {
                n: digest(out / n)
                for n in (
                    "private_worlds.json",
                    "private_design.json",
                    "queries.json",
                    "private_eligibility.json",
                )
            },
            "old_reference_sha256": {
                n: digest(OUTPUT / n)
                for n in ("real_reference.npz", "synthetic_reference.npz")
            },
            "reused": reuses,
            "new_calls_expected": len(design) - len(reuses),
            "human_labels_used": False,
            "MEPS_used": False,
            "old_lockbox_access": False,
            "missing_candidate_mass": "retained unknown for old numeric-only source; never imputed1",
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "units": len(reuses)},
    )
    print(
        {
            "status": "PREPARED",
            "units": len(design),
            "reused": len(reuses),
            "new_calls": len(design) - len(reuses),
        },
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "collect", "evaluate"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--seconds", type=int, default=1200)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.out)
    else:
        if (
            digest(Path(__file__))
            != read_json(a.out / "manifest.json")["prepare_code_sha256"]
        ):
            raise ValueError("Reference preparation code changed")
        with collector_guard(a.out):
            collect(a.out, a.seconds) if a.action == "collect" else evaluate(a.out)
