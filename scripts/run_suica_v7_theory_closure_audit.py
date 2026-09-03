#!/usr/bin/env python3
"""Audit the registered GRAIN V7 theoretical-closure evidence set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from suica_core.v7_closure import audit_v7_theory_closure  # noqa: E402
from suica_core.v7_governance import write_artifact_inventory  # noqa: E402


def _report(result: dict[str, object]) -> str:
    evidence = result["evidence"]
    rows = "\n".join(
        f"| {row['id']} | {row['status']} | tracked aggregate + source hashes |"
        for row in evidence
    )
    return (
        "# GRAIN V7 Theory Closure Audit\n\n"
        f"Status: {result['status']}\n\n"
        "## Evidence\n\n"
        "| Evidence | Corrected decision | Portable release snapshot |\n"
        "| --- | --- | --- |\n"
        f"{rows}\n\n"
        "## Evidence lattice\n\n"
        f"{json.dumps(result['evidence_lattice'], ensure_ascii=False, indent=2)}\n\n"
        "## Failures\n\n"
        f"{json.dumps(result['failures'], ensure_ascii=False, indent=2)}\n\n"
        "## Boundary\n\n"
        f"{result['claim_boundary']}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v7_theory_closure.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = audit_v7_theory_closure(config, repository_root=ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "closure_decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_report(result), encoding="utf-8")
    write_artifact_inventory(args.output_dir, args.output_dir / "artifact_inventory.json")
    print(json.dumps({"status": result["status"], "n_failures": result["n_failures"]}, indent=2))
    return 0 if result["n_failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
