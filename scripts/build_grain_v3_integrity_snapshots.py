#!/usr/bin/env python
"""Additive, dated CURRENT integrity snapshots (packet 13 Part A3). Reads result roots; writes only the snapshot root.

A snapshot authenticates today's bytes so later changes are detectable. It is never a historical run-time manifest,
a pre-registration, or evidence of immutability since creation, and every snapshot says so. A root with any run-time
manifest is not snapshotted, so no old manifest is replaced and no mismatch is blessed. Private-marked content
(private/MEPS/mcd_cache) is not snapshotted.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grain_research.gptpro_foundation_v3 import revision_stage as rs  # noqa: E402

OUT = rs.RES / "F00/integrity_snapshots_r1"
PRIORITY = ["F00/common_record_adapter_r2", "F00/common_record_adapter_r1", "F00/stage_orchestrator_r1",
            "F13/binary_source_full_r2", "F13/recovery3_r1"]
PRIVATE_MARKERS = ("private", "meps", "mcd_cache")


def main() -> int:
    if OUT.exists():
        raise FileExistsError("integrity_snapshots_r1 exists; snapshots are additive and dated, never rewritten")
    reg = rs.load_registry()
    candidates = PRIORITY + sorted(k for k, e in reg["revisions"].items() if e["kind"] == "CURRENT_REVISION" and k not in PRIORITY)
    (OUT / "snapshots").mkdir(parents=True)
    date = datetime.now(timezone.utc).date().isoformat()
    index, skipped = [], []
    for key in candidates:
        root = rs.resolve_root(key)
        if not root.is_dir():
            skipped.append({"root": key, "reason": "root missing"})
            continue
        if rs.manifest_entries(root):
            skipped.append({"root": key, "reason": "has a run-time manifest; not snapshotted"})
            continue
        files = rs.tree_files(root)
        if any(m in str(p.relative_to(rs.REPO)).lower() for p in files for m in PRIVATE_MARKERS):
            skipped.append({"root": key, "reason": "private-marked content present; not snapshotted"})
            continue
        snap = {"root": key, "snapshot_date": date, "created_at": datetime.now(timezone.utc).isoformat(), "kind": "CURRENT_INTEGRITY_SNAPSHOT",
                "historical_runtime_manifest_available": False, "priority": key in PRIORITY,
                "files": {str(p.relative_to(root)): rs.sha(p) for p in files}, "file_count": len(files), "bytes": sum(p.stat().st_size for p in files),
                "excluded_prefixes": [], "authority": "NEXT_PACKET_13_PROVENANCE_AND_EQUIVALENCE.md Part A3",
                "meaning": "authenticates the bytes present on the snapshot date; not provenance, not pre-registration, not immutability since creation"}
        rs.snapshot_path(key).write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        index.append({"root": key, "file_count": snap["file_count"], "bytes": snap["bytes"], "priority": snap["priority"]})
    (OUT / "index.json").write_text(json.dumps({"snapshot_date": date, "snapshots": index, "skipped": skipped,
                                                "registry_sha256": rs.sha(rs.REGISTRY)}, indent=1) + "\n")
    print(json.dumps({"snapshotted": len(index), "files": sum(i["file_count"] for i in index), "bytes": sum(i["bytes"] for i in index),
                      "priority_snapshotted": [i["root"] for i in index if i["priority"]],
                      "skipped_reasons": {r: sum(1 for s in skipped if s["reason"] == r) for r in {s["reason"] for s in skipped}},
                      "private_skipped": [s["root"] for s in skipped if s["reason"].startswith("private")]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
