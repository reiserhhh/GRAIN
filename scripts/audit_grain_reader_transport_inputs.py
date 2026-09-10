#!/usr/bin/env python3
"""Freeze source-disjoint material membership and native-state coverage, without Y."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_real_author_path import BASE, read, sha, write

PLAN = ROOT / "docs/GRAIN_READER_NEW_AUTHOR_TRANSPORT_PLAN_V1.md"
SALT = "GRAIN_NEXT37_READER_TRANSPORT_V1"


def select_members(material, excluded):
    """Fix four per provenance group by ID hash, independent of values or outcomes."""
    rows = [r for r in material if r["view"] == 0]
    if len({r["author_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate author view0")
    if not set(excluded).issubset({r["author_id"] for r in rows}):
        raise ValueError("Old sources absent from material")
    candidates = [r for r in rows if r["author_id"] not in excluded]
    groups = ("mbti_only", "big5_only")
    if {r["source_group"] for r in candidates} != set(groups):
        raise ValueError("Unexpected source populations")
    selected, counts = [], {}
    for group in groups:
        pool = [r for r in candidates if r["source_group"] == group]
        counts[group] = len(pool)
        if len(pool) < 4:
            raise ValueError("Insufficient predeclared source group")
        pool.sort(
            key=lambda r: hashlib.sha256(
                f"{SALT}|{r['author_id']}".encode()
            ).hexdigest()
        )
        selected.extend(pool[:4])
    return selected, counts


def inventory(name):
    """Check existing assets and bind the next8 participants without neural work."""
    old = BASE / "multiquery_sensitivity_s1_r1"
    native = BASE / "full_state_edit_connection_v1"
    members = BASE / "logic_query_residual_input_v1_r2/private_membership.json"
    manifest = read(old / "manifest.json")
    material_path = old / "private_material.json"
    if sha(material_path) != manifest["material_sha256"]:
        raise ValueError("Source material changed")
    selected, counts = select_members(
        read(material_path), {r["author_id"] for r in read(members)["authors"]}
    )
    if counts != {"mbti_only": 10, "big5_only": 10}:
        raise ValueError("Frozen remaining20 coverage changed")
    packets, shapes, dependencies = (
        {},
        {},
        [PLAN, Path(__file__), members, material_path, old / "manifest.json"],
    )
    selected_ids = {r["author_id"] for r in selected}
    for p in sorted((old / "private_packets").glob("panel_*.json")):
        m = read(p)
        if m["author_id"] not in selected_ids or m["view"] != 0:
            continue
        npz = p.with_suffix(".npz")
        if sha(npz) != m["state_sha256"]:
            raise ValueError("Selected old scalar cache changed")
        with np.load(npz) as z:
            shapes[m["author_id"]] = {
                q: list(z[q + "__full"].shape) for q in m["queries"]
            }
        packets[m["author_id"]] = str(p.relative_to(ROOT))
        dependencies.extend([p, npz])
    if set(packets) != selected_ids or any(
        s != [1536] for v in shapes.values() for s in v.values()
    ):
        raise ValueError("Expected scalar-state cache coverage changed")
    inputs_path = native / "private_inputs.json"
    if sha(inputs_path) != read(native / "manifest.json")["input_sha256"]:
        raise ValueError("Old native membership changed")
    old_native = read(inputs_path)
    native_ids = {r["author_id"] for r in old_native}
    if selected_ids & native_ids:
        raise ValueError("Selected authors unexpectedly share old native reader inputs")
    dependencies.extend([inputs_path, native / "manifest.json"])
    records = [
        {
            "index": i,
            "author_id": r["author_id"],
            "source_group": r["source_group"],
            "panel_index": r["panel_index"],
            "view": 0,
            "comments": len(r["texts"]),
            "s1_packet": packets[r["author_id"]],
            "text_list_sha256": hashlib.sha256(
                json.dumps(
                    r["texts"], ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        for i, r in enumerate(selected)
    ]
    out = BASE / name
    out.mkdir(exist_ok=False)
    write(out / "private_selection.json", {"salt": SALT, "authors": records})
    summary = {
        "status": "PREPARED_NOT_COLLECTED",
        "remaining_authors": sum(counts.values()),
        "remaining_by_source_group": counts,
        "selected_authors": 8,
        "selected_per_group": 4,
        "queries": 3,
        "old_native_author_overlap": 0,
        "selected_s1_packets": len(packets),
        "s1_full_state_shape": [1536],
        "selected_usable_full_token_state_packets": 0,
        "needed_prefix_captures": 24,
        "needed_scalar_backwards": 24,
        "fixed_finite_difference_suffix_forwards": 6,
        "new_neural_calls": 0,
        "human_labels_read": False,
    }
    write(out / "summary.json", summary)
    write(
        out / "manifest.json",
        {
            "inputs": {str(p.relative_to(ROOT)): sha(p) for p in dependencies},
            "files": {
                p: sha(out / p) for p in ("private_selection.json", "summary.json")
            },
        },
    )
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="logic_reader_transport_inventory_v1")
    inventory(parser.parse_args().name)
