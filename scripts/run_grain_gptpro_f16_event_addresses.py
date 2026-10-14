#!/usr/bin/env python3
"""Restore exact PANDORA vector/event times and build a deduplicated X entry.

No questionnaire values or MEPS data. Reads authorized cached PANDORA material
for content hashes/lengths; raw corpora are metadata-projected. No encoder calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f14_metadata import sha256_file
from grain_research.gptpro_foundation_v3.f16_event_addresses import (
    event_address_frame,
    text_hash,
    verify_event_matches,
    x_event_tables,
)
from grain_research.vector_mechanisms.cohort import ID_SEED
from grain_research.vector_mechanisms.data_views import stable_key


def write(path, obj):
    """Save finite aggregate JSON or explicitly private metadata output."""
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main():
    """Preserve existing source/held order and expose provenance-only findings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    f14 = ROOT / "results/grain_gptpro_foundation_v3/F14/metadata_full_r2"
    base = ROOT / "results/grain_vector_mechanisms"
    material_root = base / "predecessor_integration_v1/material"
    local = base / "all_plan_closure_v1/local_bilinear_r2/pandora"
    ref = base / "all_plan_closure_v1/conditional_reference_r2/pandora"
    files = [
        material_root / "material_manifest.json",
        material_root / "private_material.json",
        local / "private_authors.json",
        local / "private_addresses.csv",
        local / "material.json",
        ref / "private_units.npz",
        f14 / "data_source_inventory.csv",
        f14 / "summary.json",
    ]
    input_hashes = {str(path): sha256_file(path) for path in files}
    code = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f16_event_addresses.py",
    ]
    write(
        a.output / "contract_before_read.json",
        {
            "inputs": input_hashes,
            "code": {str(path): sha256_file(path) for path in code},
            "scope": "same cached PANDORA36-comment material and registered Xmetadata; no labels/MEPS/model calls",
            "source_held_unchanged": True,
            "parent_promotion": False,
            "text_qualification": "authored publication only; not subject/actuality inference",
        },
    )
    material_manifest = json.loads(files[0].read_text())
    if (
        material_manifest["files"]["private_material.json"]
        != input_hashes[str(files[1])]
    ):
        raise ValueError("changed original material")
    old_meta = json.loads((local / "material.json").read_text())
    for name in ("private_authors.json", "private_addresses.csv"):
        if sha256_file(local / name) != old_meta["files"][name]:
            raise ValueError("changed original vector material mapping")
    material = json.loads(files[1].read_text())
    names = json.loads(files[2].read_text())
    with np.load(ref / "private_units.npz", allow_pickle=False) as ar:
        authors, conditions, source, held = [
            ar[k] for k in ("authors", "conditions", "source", "held")
        ]
    mapped = event_address_frame(
        material, names, pd.read_csv(files[3]), authors, conditions
    )
    del material
    inventory = pd.read_csv(f14 / "data_source_inventory.csv", keep_default_na=False)
    raw_row = inventory.loc[inventory.dataset.eq("pandora")].iloc[0]
    raw = Path(raw_row.source_path)
    before = raw.stat()
    if sha256_file(raw) != raw_row["hash"]:
        raise ValueError("PANDORA raw source changed since F14")
    wanted_authors = set(mapped.author_key)
    wanted_events = set(mapped.event_key)
    author_hashes = {}
    matches, total = [], 0
    cols = ["author", "id", "link_id", "subreddit", "created_utc"]
    for frame in pd.read_csv(
        raw, usecols=cols, dtype="string", keep_default_na=False, chunksize=100_000
    ):
        total += len(frame)
        for author in frame.author.unique():
            if author not in author_hashes:
                author_hashes[author] = stable_key(str(author), ID_SEED)[:24]
        selected = frame.loc[
            frame.author.map(author_hashes).isin(wanted_authors)
        ].copy()
        selected["event_key"] = selected.id.map(
            lambda v: stable_key(str(v), ID_SEED)[:24]
        )
        selected = selected.loc[selected.event_key.isin(wanted_events)]
        if len(selected):
            matches.append(
                pd.DataFrame(
                    {
                        "event_key": selected.event_key,
                        "author_key": selected.author.map(author_hashes),
                        "thread_key": selected.link_id.map(
                            lambda v: stable_key(str(v), ID_SEED)[:24]
                        ),
                        "condition": selected.subreddit,
                        "timestamp_unix_seconds": pd.to_numeric(
                            selected.created_utc, errors="raise"
                        ),
                    }
                )
            )
        if total % 2_000_000 == 0:
            print(
                json.dumps({"stage": "PANDORA_metadata_event_join", "rows": total}),
                flush=True,
            )
    after = raw.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ) or total != raw_row.rows:
        raise ValueError("PANDORA input changed during scan")
    found = pd.concat(matches, ignore_index=True)
    matched = verify_event_matches(mapped, found)
    mapped["split"] = np.where(
        mapped.author_index.isin(source), "source", "application"
    )
    if set(source) & set(held) or set(source) | set(held) != set(mapped.author_index):
        raise ValueError("source/application partition mismatch")
    mapped.to_csv(a.output / "private_pandora_event_addresses.csv", index=False)
    spans = mapped.groupby("author_index").timestamp_unix_seconds.agg(["min", "max"])
    gaps = (
        mapped.groupby("author_index").timestamp_unix_seconds.diff().dropna().to_numpy()
        / 86400
    )
    days = (spans["max"] - spans["min"]).to_numpy() / 86400
    x_paths = [
        Path(v) for v in inventory.loc[inventory.dataset.eq("x_market"), "source_path"]
    ]
    x_columns = ["post_id", "account_id", "timestamp", "symbol", "query_group", "lang"]
    frames = []
    for row in inventory.loc[inventory.dataset.eq("x_market")].itertuples(index=False):
        path = Path(row.source_path)
        if sha256_file(path) != row.hash:
            raise ValueError("X source changed since F14")
        frames.append(
            pd.read_csv(path, usecols=x_columns, dtype="string", keep_default_na=False)
        )
    x_raw = pd.concat(frames, ignore_index=True)
    events, tags = x_event_tables(x_raw)
    event_key = {v: text_hash("F16_X_EVENT::" + str(v))[:24] for v in events.post_id}
    author_key = {
        v: text_hash("F16_X_AUTHOR::" + str(v))[:24] for v in events.account_id.unique()
    }
    if len(set(event_key.values())) != len(event_key) or len(
        set(author_key.values())
    ) != len(author_key):
        raise ValueError("X anonymous key collision")
    events["post_id"] = events.post_id.map(event_key)
    events["account_id"] = events.account_id.map(author_key)
    tags["post_id"] = tags.post_id.map(event_key)
    events.rename(columns={"post_id": "event_key", "account_id": "author_key"}).to_csv(
        a.output / "private_x_events.csv", index=False
    )
    tags.rename(columns={"post_id": "event_key"}).to_csv(
        a.output / "private_x_collection_tags.csv", index=False
    )
    old_counts = x_raw.groupby("account_id").size()
    old_counts.index = old_counts.index.map(author_key)
    unique_counts = events.groupby("account_id").size()
    eligibility = []
    # Fixed text-count examples; no personality outcomes or parameter selection.
    for threshold in (4, 12, 36):
        old_set = set(old_counts.index[old_counts.ge(threshold)])
        new_set = set(unique_counts.index[unique_counts.ge(threshold)])
        eligibility.append(
            {
                "required_events": threshold,
                "row_eligible": len(old_set),
                "event_eligible": len(new_set),
                "duplicate_only_eligible": len(old_set - new_set),
            }
        )
    summary = {
        "status": "EVENT_ADDRESS_AND_COLLECTION_SEPARATION_COMPLETE",
        "pandora": matched
        | {
            "authors": len(names["ids"]),
            "source_authors": len(source),
            "held_authors": len(held),
            "vector_units": len(mapped),
            "raw_rows_scanned": total,
            "days_span_min_median_max": np.quantile(days, [0, 0.5, 1]).tolist(),
            "adjacent_gap_days_min_median_max": np.quantile(gaps, [0, 0.5, 1]).tolist(),
            "equal_time_gaps": int(np.sum(gaps == 0)),
            "calendar_sequence_is_regular_T2_T3_grid": False,
        },
        "x_market": {
            "files": len(x_paths),
            "collected_rows": len(x_raw),
            "events": len(events),
            "authors": len(author_key),
            "collection_relations_including_partial": len(tags),
            "complete_relations": int(tags.complete_condition.sum()),
            "partial_relations": int((~tags.complete_condition).sum()),
            "eligibility_by_unique_event": eligibility,
            "exposure_denominator_known": False,
        },
        "new_model_calls": 0,
        "MEPS_used": False,
        "questionnaire_values_read": False,
        "raw_identifiers_saved": False,
        "F16_parent_complete": False,
        "remaining": [
            "actual same-target reader qualification/unit/error integration",
            "F11 GRAIN same-material readings",
            "target-matched transport and canonical construction",
        ],
        "seconds": time.monotonic() - began,
    }
    write(a.output / "summary.json", summary)
    write(
        a.output / "output_manifest.json",
        {p.name: sha256_file(p) for p in sorted(a.output.iterdir()) if p.is_file()},
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
