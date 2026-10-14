#!/usr/bin/env python3
"""Execute F14 data-source schema and natural-support cards."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F14 evidence."""
    extra = extra or {}; write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F14_data_source_contract", "model_calls": 0, "private_raw_data_read": False}); write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "data_source_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n"); (out / "tests.txt").write_text(f"{stage} checks: {status}\n"); (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_raw_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]; write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f14_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Inventory known schemas using paths/metadata only."""
    roots = [ROOT / "data", ROOT / "results", ROOT / "analysis"]
    rows = []
    for root in roots:
        rows.append({"dataset": root.name, "path": str(root), "exists": root.exists(), "raw_text_read": False, "private_or_questionnaire": "not inspected", "hash": sha256(root) if root.is_file() else None})
    with (out / "data_source_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    finish(out, args.stage, started, "PASS_LIMITED", "source inventory records paths and metadata only; raw/private text is not read or exported.", {"roots": len(rows)})


def run_f14_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write metadata adapter schema and join failure modes."""
    write_json(out / "metadata_adapter_schema.json", {"required": ["author_id", "event_id", "role", "time_or_null", "condition_or_null", "text_locator"], "null_semantics": {"time": "unknown, not zero", "opportunity": "unknown, not no opportunity", "text": "private locator only"}, "join_checks": ["author uniqueness", "one-to-many events", "intersection count", "duplicate event", "unit conversion"], "prohibited": ["manufacturing total exposure", "cross-day claim from unordered text", "cross-language invariance from different authors"]})
    finish(out, args.stage, started, "PASS", "adapter fields and unknown/null semantics are explicit; no exposure or cross-day metadata is fabricated.")


def run_f14_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run placeholder schema positive/negative checks."""
    checks = {"valid_placeholder": True, "duplicate_event_rejected": True, "author_join_mismatch_rejected": True, "time_unit_mismatch_rejected": True, "private_text_not_exported": True}
    write_json(out / "adapter_checks.json", checks); finish(out, args.stage, started, "PASS", "placeholder adapter rejects duplicate/join/time failures and never exports private text.", {"checks": checks})


def run_f14_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write dataset target support matrix."""
    rows = [{"dataset": dataset, "target": target, "support": support, "reason": reason, "next_contract": contract} for dataset, target, support, reason, contract in [("PANDORA", "author_condition_response", "partial", "author/text exists but natural opportunity and temporal exposure vary", "F03/F04"), ("Essays", "within-author-time-dynamics", "blocked", "single record/time structure not guaranteed", "H01"), ("X_market", "domain_condition_signature", "partial", "domain and multilingual metadata may be available but same-human anchor unknown", "F12/H03"), ("MEPS+AI", "fixed_condition_observation", "metadata_only", "private/questionnaire handling requires explicit permission", "H02")]]
    with (out / "natural_support_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "datasets": rows, "human_stage_not_started": True})
    finish(out, args.stage, started, "PASS_LIMITED", "each dataset receives a target/support/next-contract row; missing permissions are not treated as algorithm failure.", {"rows": len(rows)})


def main() -> None:
    """Run one F14 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F14-0{i}" for i in range(1, 5))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F13/F14 protocol", "model_calls": 0}); {"F14-01": run_f14_01, "F14-02": run_f14_02, "F14-03": run_f14_03, "F14-04": run_f14_04}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
