#!/usr/bin/env python
"""Packet 24 Part A: narrow packet-23 acceptance repairs (docs/GRAIN_P23_REPAIRS_PROTOCOL_V1.md).
(1) machine-readable manifest hash chain; (2) metadata-only GRAIN comment-order check (IDs/view index/timestamps only,
never body fields); (3) adjunct cold-scoring bundle + fresh-process sealed cold scorer. 0 neural calls, no fitting."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_P23_REPAIRS_PROTOCOL_V1.md"
PROTOCOL_SHA = "460e0c656b2789150d40636634ee2ceb4033d2e68d0dcef0eb2c09ba9258809d"
VM = ROOT / "results/grain_vector_mechanisms"
F14 = ROOT / "results/grain_gptpro_foundation_v3/F14"
OUT = F14 / "raw_grain_history_repairs_r1"
PY = sys.executable


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol changed")
    for rel, hh in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        if sha(ROOT / rel) != hh:
            raise RuntimeError(f"binding mismatch {rel}")


def id_sequences(path, wanted):
    """Traverse a view file for author_id and ordered (comment_id, view_index, timestamp) ONLY; body fields untouched."""
    data = json.loads(Path(path).read_text())
    out = {}
    for b in data:
        a = b["author_id"]
        if a in wanted:
            out[a] = [(str(c["comment_id"]), vi, float(c["timestamp"])) for vi, view in enumerate(b["views"]) for c in view]
    del data
    return out


def main():
    t0 = time.time(); started = datetime.now(timezone.utc).isoformat()
    verify()
    if OUT.exists():
        raise FileExistsError("raw_grain_history_repairs_r1 exists")
    OUT.mkdir(parents=True)
    # ---------------- (1) hash chain
    mm = json.loads((VM / "predecessor_integration_v1/material/material_manifest.json").read_text())
    common = json.loads((VM / "vm05_preflight_r1/common_material_manifest.json").read_text())
    pre = json.loads((VM / "vm05_preflight_r1/manifest.json").read_text())
    par = json.loads((VM / "vm05_full_source_cohort_r1/manifest.json").read_text())
    chain = {"links": [
        {"from": "material_manifest.sources.common_manifest", "to": "sha256(vm05_preflight_r1/common_material_manifest.json)", "equal": mm["sources"]["common_manifest"] == sha(VM / "vm05_preflight_r1/common_material_manifest.json")},
        {"from": "material_manifest.sources.parent_manifest", "to": "sha256(vm05_full_source_cohort_r1/manifest.json)", "equal": mm["sources"]["parent_manifest"] == sha(VM / "vm05_full_source_cohort_r1/manifest.json")},
        {"from": "common_material_manifest.parent_identity", "to": "vm05_preflight_r1/manifest.json identity (writer line 226)", "equal": common["parent_identity"] == pre["identity"]},
        {"from": "vm05_preflight_r1/manifest.json source_identity", "to": "vm05_full_source_cohort_r1/manifest.json identity", "equal": pre["source_identity"] == par["identity"]}],
        "historical_false_gate": "F14/raw_grain_history_r1 results.json 'common_parent_identity_equals_parent_identity'=false compared common.parent_identity with the COHORT parent identity; it never entered grain_ok and stays unmodified"}
    chain["all_links_equal"] = all(l["equal"] for l in chain["links"])
    (OUT / "hash_chain.json").write_text(json.dumps(chain, indent=1) + "\n")
    # ---------------- (2) metadata-only comment order
    with np.load(VM / "predecessor_integration_v1/material/private_arrays.npz", allow_pickle=False) as z:
        ids = z["author_ids"].astype(str)
    wanted = set(ids)
    ea = pd.read_csv(ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv",
                     usecols=["author_index", "unit", "author_key", "event_key", "view_index", "timestamp_unix_seconds"], dtype={"author_key": str, "event_key": str})
    ea = ea.sort_values(["author_index", "unit"])
    seq_pre = id_sequences(VM / "vm05_full_source_cohort_r1/author_views.json", wanted)       # preflight read("author_views.json")
    seq_mat = id_sequences(VM / "vm05_full_source_cohort_r1/big5_only_author_views.json", wanted)  # material constructor source
    res_o = {"authors": len(ids), "units": 0, "preflight_source_authors_found": len(seq_pre), "material_source_authors_found": len(seq_mat),
             "unit_mismatch_preflight_vs_address": 0, "unit_mismatch_material_vs_address": 0, "view_index_mismatch": 0, "timestamp_mismatch": 0, "author_key_mismatch": 0}
    for a_idx, a in enumerate(ids):
        e = ea[ea.author_index == a_idx]
        res_o["units"] += len(e)
        res_o["author_key_mismatch"] += int((e.author_key != a).sum())
        ek, vi, ts = e.event_key.tolist(), e.view_index.tolist(), e.timestamp_unix_seconds.to_numpy()
        for name, seq in (("preflight", seq_pre.get(a)), ("material", seq_mat.get(a))):
            if seq is None or len(seq) != len(ek):
                res_o[f"unit_mismatch_{name}_vs_address"] += len(ek); continue
            res_o[f"unit_mismatch_{name}_vs_address"] += sum(s[0] != k for s, k in zip(seq, ek))
            if name == "preflight":
                res_o["view_index_mismatch"] += sum(s[1] != v for s, v in zip(seq, vi))
                res_o["timestamp_mismatch"] += int(np.sum(np.abs(np.array([s[2] for s in seq]) - ts) > 1e-6))
    res_o["order_verified"] = bool(res_o["units"] == 624 * 36 and all(res_o[k] == 0 for k in ("unit_mismatch_preflight_vs_address", "unit_mismatch_material_vs_address", "view_index_mismatch", "timestamp_mismatch", "author_key_mismatch")))
    res_o["fields_read"] = ["author_id", "views[*][*].comment_id", "view index", "views[*][*].timestamp"]
    res_o["fields_not_read"] = "body and all other comment/questionnaire fields (never accessed, printed, scored or exported)"
    (OUT / "comment_order_check.json").write_text(json.dumps(res_o, indent=1) + "\n")
    # ---------------- (3) adjunct bundle (copies; no re-estimation) + fresh sealed cold process
    p22 = np.load(F14 / "history_increment_r1/coefficients.npz"); p23 = np.load(F14 / "raw_grain_history_r1/private_bundle.npz")
    with np.load(VM / "all_plan_closure_v1/conditional_reference_r2/private_directions.npz", allow_pickle=False) as z:
        direction = np.asarray(z["direction"], float)
    adj = {k: p23[k] for k in p23.files}
    adj.update(direction=direction, keyed_mu=p22["mu"], keyed_sd=p22["sd"],
               provenance_json=np.array(json.dumps({"keyed_mu_sd": "copied from F14/history_increment_r1/coefficients.npz (sha " + sha(F14 / "history_increment_r1/coefficients.npz") + ")",
                                                     "readouts_pca_scalers": "copied from F14/raw_grain_history_r1/private_bundle.npz (sha " + sha(F14 / "raw_grain_history_r1/private_bundle.npz") + ")",
                                                     "direction": "conditional_reference_r2/private_directions.npz (sha " + sha(VM / "all_plan_closure_v1/conditional_reference_r2/private_directions.npz") + ")"})))
    bpath = OUT / "private_cold_adjunct_bundle.npz"
    np.savez(bpath, **adj)
    cold_out = OUT / "private_cold_held_losses.csv"
    cmd = [PY, str(ROOT / "scripts/cold_score_grain_r08_raw_grain_bundle.py"), str(bpath), str(VM / "predecessor_integration_v1/material/private_arrays.npz"),
           str(VM / "vm05_preflight_r1/grain_frozen_features.npz"), str(ROOT / "results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv"),
           str(F14 / "pandora_venue_oom_r1/author_roles.json"), str(cold_out)]
    tc = time.time(); pr = subprocess.run(cmd, capture_output=True, text=True, cwd="/")
    log = {"returncode": pr.returncode, "seconds": time.time() - tc, "stdout": pr.stdout[-4000:], "stderr": pr.stderr[-2000:], "cwd": "/"}
    if pr.returncode == 0:
        cold = pd.read_csv(cold_out); orig = pd.read_csv(F14 / "raw_grain_history_r1/private_held_event_losses.csv")
        mg = cold.merge(orig, on=["author", "t"], suffixes=("_cold", ""))
        log["rows_compared"] = int(len(mg))
        log["max_abs_diff_by_arm"] = {a: float(np.abs(mg[f"mse_{a}_cold"] - mg[f"mse_{a}"]).max()) for a in ("KEYED_STATIC", "RAW_STATIC", "GRAIN_STATIC", "RAW_ORDERED")}
        log["opened"] = json.loads(pr.stdout.strip().splitlines()[-1])
    (OUT / "cold_process_log.json").write_text(json.dumps(log, indent=1) + "\n")
    res = {"protocol_sha256": PROTOCOL_SHA, "neural_calls": 0, "hash_chain_all_equal": chain["all_links_equal"], "comment_order_verified": res_o["order_verified"],
           "cold_returncode": pr.returncode, "cold_max_abs_diff": max(log.get("max_abs_diff_by_arm", {"x": float("nan")}).values()), "seconds": time.time() - t0, "started": started}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
