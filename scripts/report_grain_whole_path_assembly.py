#!/usr/bin/env python3
"""Publish only audited cohort aggregates from the NEXT55 private packet."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_whole_path_assembly import BASE, read, sha, verify_hashes, write


def publish():
    """Use explicit summary inventories; never export author rows or hidden states."""
    out = BASE / "logic_whole_path_assembly_v1"
    public = ROOT / "reports/grain_vector_mechanisms"
    for analysis, audit_name, name, count in (
        ("analysis", "replay_audit.json", "whole_path_assembly_tables", 12),
        (
            "relocation_analysis",
            "relocation_replay_audit.json",
            "whole_path_relocation_tables",
            9,
        ),
    ):
        audit = read(out / audit_name)
        summary = read(out / analysis / "summary.json")
        verify_hashes(out / analysis, summary["files"])
        if audit["status"] != "PASS":
            raise ValueError("Publication requires a completed cache audit")
        names = [n for n in summary["files"] if not n.startswith("private_")]
        if len(names) != count:
            raise ValueError("Aggregate inventory changed")
        dst = public / name
        dst.mkdir(exist_ok=True)
        for n in names:
            src = out / analysis / n
            tab = pd.read_csv(src)
            if not {"cohort"} <= set(tab.columns) or any(
                c in tab.columns
                for c in ("author_key", "text", "user_id", "participant_id")
            ):
                raise ValueError("Only cohort aggregates may be published")
            (dst / n).write_bytes(src.read_bytes())
        write(dst / "audit.json", audit)
        write(dst / "inventory.json", {n: sha(dst / n) for n in names + ["audit.json"]})
        print({"folder": name, "aggregate_tables": count, "audit": "PASS"})


if __name__ == "__main__":
    publish()
