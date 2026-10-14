#!/usr/bin/env python3
"""Write a local, uncommitted content manifest (sha256) for the mechanism theory package."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/grain_measurement_foundation"
PAPER = ROOT / "docs/paper"
FILES = [
    PAPER / "GRAIN_MECHANISM_THEORY_PACKAGE_V1.md",
    PAPER / "GRAIN_MECHANISM_SUPPLEMENT_EN_V1.md",
    PAPER / "GRAIN_MECHANISM_SUPPLEMENT_JA_V1.md",
    PAPER / "GRAIN_MECHANISM_SUPPLEMENT_ZH_V1.md",
    PAPER / "GRAIN_MECHANISM_EVIDENCE_PROVENANCE.md",
    PAPER / "GRAIN_THEORY_MANUSCRIPT_CROSSWALK.md",
    PAPER / "GRAIN_THEORY_MANUSCRIPT_EN_V1.md",
    PAPER / "GRAIN_THEORY_MANUSCRIPT_JA_V1.md",
    PAPER / "GRAIN_THEORY_MANUSCRIPT_ZH_V1.md",
    ROOT / "docs/GRAIN_CONTEXTUAL_CONSTRUCTION_HANDBOOK_V1.md",
    ROOT / "docs/GRAIN_MECHANISM_TODO_20260913.md",
    ROOT / "docs/GRAIN_INDEPENDENT_CODING_SPEC_V1.md",
    ROOT / "docs/GRAIN_CLAUDE_TODO_CLOSURE_HANDOFF_20260913.md",
    ROOT / "docs/GRAIN_CLAUDE_ALL_PLAN_LOGIC_RESEARCH_HANDOFF_20260914.md",
    ROOT / "docs/GRAIN_ALL_PLAN_REQUIREMENT_CROSSWALK_V1.md",
    ROOT / "docs/GRAIN_CURRENT_SYSTEM_LOGIC_V1.md",
    ROOT / "docs/GRAIN_EXECUTABLE_RESEARCH_PLAN_V2.md",
    REPORTS / "GRAIN_ALL_PLAN_LOGIC_FINDINGS_V1.md",
    REPORTS / "CLAUDE_TODO_CLOSURE_AUDIT.md",
    REPORTS / "CLAUDE_TODO_CLOSURE_REPORT.md",
] + sorted(
    p
    for p in REPORTS.glob("*_REPORT.md")
    if not p.name.startswith("._") and not p.name.startswith("CLAUDE_")
) + [REPORTS / "LOCAL_INSTRUMENT_EXECUTION_REPAIR.md"]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in FILES:
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "present": path.exists(),
                "sha256": sha(path) if path.exists() else None,
                "bytes": path.stat().st_size if path.exists() else None,
            }
        )
    payload = {
        "package": "GRAIN mechanism theory package V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": rows,
        "present": sum(r["present"] for r in rows),
        "missing": [r["path"] for r in rows if not r["present"]],
        "git_actions": "none: no commit, tag, push or lockbox opening",
    }
    (args.output / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = ["| file | sha256 | bytes |", "|---|---|---:|"]
    lines += [f"| `{r['path']}` | `{r['sha256']}` | {r['bytes']} |" for r in rows if r["present"]]
    (args.output / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    print({"present": payload["present"], "missing": payload["missing"]})


if __name__ == "__main__":
    main()
