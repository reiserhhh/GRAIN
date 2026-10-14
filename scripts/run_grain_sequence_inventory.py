#!/usr/bin/env python
"""Packet 14 Part D: inventory of existing permitted PANDORA time-ordered author sequences (metadata and cached
vectors only; no text read, no raw 17.6M-row rescan, no MEPS). Writes an inventory and a frozen next-packet manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PERSONA = Path("/Volumes/mobile3/projects/project persona")
OUT = ROOT / "results/grain_gptpro_foundation_v3/F14/sequence_inventory_r1"
EA = ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv"
UNITS = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/pandora/private_units.npz"
C2 = PERSONA / "analysis/output/situation_emotion_dynamics/20260613_041057/comment_emotion_features.csv"
F14 = ROOT / "results/grain_gptpro_foundation_v3/F14/metadata_full_r2/pandora_availability.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def q(s):
    return {k: float(v) for k, v in s.quantile([0, .1, .5, .9, 1]).rename(lambda x: f"q{int(x * 100)}").items()}


def main() -> int:
    if OUT.exists():
        raise FileExistsError("sequence_inventory_r1 exists")
    OUT.mkdir(parents=True)
    ea = pd.read_csv(EA)
    with np.load(UNITS, allow_pickle=False) as u:
        scores, authors, conds, source, held = u["scores"], u["authors"], u["conditions"], u["source"], u["held"]
    aligned = bool(len(ea) == len(scores) and np.array_equal(ea.author_index.to_numpy(), authors) and np.array_equal(ea.condition.astype(str).to_numpy(), conds.astype(str)))
    ea = ea.sort_values(["author_index", "timestamp_unix_seconds", "event_key"])
    lens = ea.groupby("author_index").size()
    ties = ea.groupby("author_index").timestamp_unix_seconds.apply(lambda s: int(s.duplicated().sum()))
    gaps = ea.groupby("author_index").timestamp_unix_seconds.diff().dropna() / 86400
    cond_counts = ea.condition.value_counts()
    src_auth = set(ea[ea.split == "source"].author_index)
    top = {}
    src_counts = ea[ea.split == "source"].condition.value_counts()
    for K in (10, 20, 50):
        keep = set(src_counts.index[:K])
        top[f"top{K}_source_venues_coverage_held"] = float(ea[ea.split != "source"].condition.isin(keep).mean())
    # temporal split within held authors: first 70% history, last 30% evaluation
    held_df = ea[ea.split != "source"].copy()
    held_df["rank"] = held_df.groupby("author_index").cumcount()
    held_df["n"] = held_df.groupby("author_index").author_index.transform("size")
    hist = held_df[held_df["rank"] < (0.7 * held_df["n"]).astype(int)]
    ev = held_df[held_df["rank"] >= (0.7 * held_df["n"]).astype(int)]
    seen = ev.merge(hist[["author_index", "condition"]].drop_duplicates(), on=["author_index", "condition"], how="left", indicator=True)
    inv = {
        "event_address_file": {"path": str(EA.relative_to(ROOT)), "sha256": sha(EA), "rows": int(len(ea)), "unique_event_keys": int(ea.event_key.nunique()),
                               "authors": int(ea.author_index.nunique()), "source_authors": int(len(src_auth)), "held_authors": int(ea[ea.split != "source"].author_index.nunique()),
                               "split_labels": sorted(ea.split.unique().tolist()), "missing_timestamps": int(ea.timestamp_unix_seconds.isna().sum()),
                               "within_author_tied_timestamps": int(ties.sum()), "authors_with_ties": int((ties > 0).sum()),
                               "events_per_author": q(lens), "gap_days": q(gaps), "distinct_venues": int(cond_counts.size), **top,
                               "held_eval_window_events": int(len(ev)), "held_eval_events_in_venue_seen_in_own_history": float((seen["_merge"] == "both").mean()),
                               "text_read": False, "note": "hashed keys, venue label, timestamp, body hash only"},
        "cached_vectors": {"path": str(UNITS.relative_to(ROOT)), "sha256": sha(UNITS), "shape": list(scores.shape), "aligned_with_event_addresses": aligned,
                           "finite": bool(np.isfinite(scores).all()), "source_held_arrays": [int(len(source)), int(len(held))],
                           "semantics": "30 cached per-event scores from grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2; column meaning must be pinned from that runner before use"},
        "full_pandora_metadata_F14": {k: v for k, v in json.loads(F14.read_text()).items() if k in ("rows", "authors", "events", "conditions", "repeated_event_keys", "duplicate_event_extra_rows", "missing_time_rows", "invalid_time_rows", "all_opportunities_observed", "opportunity_reason")},
    }
    if C2.exists():
        c2 = pd.read_csv(C2, usecols=["user_id", "comment_id", "created_utc", "subreddit"])
        dup = c2.duplicated(["user_id", "comment_id"]).sum()
        inv["c2_mcd_input_metadata"] = {"path": str(C2), "sha256": sha(C2), "rows": int(len(c2)), "authors": int(c2.user_id.nunique()), "duplicate_user_comment_rows": int(dup),
                                        "missing_created_utc": int(c2.created_utc.isna().sum()),
                                        "note": "duplicate keys conflict on consumed fields (P05 c2_dedup); requires the FIRST/LAST version rule; not preferred over the deduplicated event-address file"}
    (OUT / "inventory.json").write_text(json.dumps(inv, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in inv.items()}, indent=1)[:3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
