#!/usr/bin/env python3
"""Unified --stage entry for all 100 GRAIN V3 atomic cards (F00-04 adoption).

Verify-only: replays existing numbers by checking registered artifacts, their
output manifests and dependency states; it never reruns experiments. States:
VERIFIED (all artifacts exist and every manifest entry re-hashes), PRESENT
(artifacts exist, some without a checkable manifest), MISSING, and
DEPENDENCY_INCOMPLETE. States are bookkeeping, not scientific verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/grain_v3_card_registry.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_card(card, entry):
    """Artifact existence and manifest re-hash for one card."""
    if not entry["artifacts"]:
        return {"card": card, "state": "MISSING", "artifacts": 0}
    verified = unverifiable = mismatches = 0
    for rel in entry["artifacts"]:
        p = ROOT / rel
        man = p / "output_manifest.json" if p.is_dir() else None
        if man and man.exists():
            data = json.loads(man.read_text())
            if isinstance(data, dict) and isinstance(data.get("files"), list):
                items = [(f.get("path"), f.get("sha256")) for f in data["files"] if isinstance(f, dict)]
            elif isinstance(data, list):
                items = [(f.get("path"), f.get("sha256")) for f in data if isinstance(f, dict)]
            elif isinstance(data, dict):
                items = list(data.items())
            else:
                items = []
            for name, digest in items:
                if not isinstance(name, str) or Path(name).name.startswith("._"):
                    continue  # AppleDouble filesystem metadata on this volume, not research output
                target = p / name if (p / name).is_file() else ROOT / name
                if isinstance(digest, str) and len(digest) == 64 and target.is_file():
                    verified += 1
                    mismatches += sha(target) != digest
                else:
                    unverifiable += 1
        else:
            unverifiable += 1
    state = "MISMATCH" if mismatches else ("VERIFIED" if unverifiable == 0 else "PRESENT")
    return {"card": card, "state": state, "artifacts": len(entry["artifacts"]), "hash_verified": verified, "hash_mismatch": mismatches, "unverifiable_entries": unverifiable}


def main():
    """--stage CARD or --all; writes task_state.json when --output is given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs="*", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--verify", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    reg = json.loads(REGISTRY.read_text())["cards"]
    cards = sorted(reg) if args.all else (args.stage or [])
    results = {c: check_card(c, reg[c]) for c in sorted(reg)}
    for c, r in results.items():
        deps = reg[c]["depends_on"]
        bad = [d for d in deps if results[d]["state"] in ("MISSING", "MISMATCH")]
        if bad and r["state"] not in ("MISSING", "MISMATCH"):
            r["state"] = "DEPENDENCY_INCOMPLETE"
            r["incomplete_dependencies"] = bad
    out = {c: results[c] for c in cards}
    summary = {}
    for r in out.values():
        summary[r["state"]] = summary.get(r["state"], 0) + 1
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "task_state.json").write_text(json.dumps({"summary": summary, "cards": out, "registry_sha256": sha(REGISTRY)}, indent=1) + "\n")
    print(json.dumps({"summary": summary, "non_verified": {c: r["state"] for c, r in out.items() if r["state"] != "VERIFIED"}}, indent=1))


if __name__ == "__main__":
    main()
