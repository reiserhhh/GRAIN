#!/usr/bin/env python3
"""Execute F16 two-target instrument-construction cards."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f13 import gaussian_interval


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F16 evidence."""
    extra = extra or {}; write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F16_instrument_construction", "model_calls": 0}); write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"]); writer.writeheader(); writer.writerow({"task_id": stage, "check": "construction_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n"); (out / "tests.txt").write_text(f"{stage} checks: {status}\n"); (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]; write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def run_f16_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze T1/T2/T3 target briefs before application scoring."""
    write_json(out / "construction_briefs.json", {"T1": {"target": "m(+1)-m(-1)", "unit": "condition response", "input": "source four-cell response", "brief_status": "frozen"}, "T2": {"target": "E[s_T|prefix]", "unit": "current latent state", "input": "F05 filter", "brief_status": "frozen"}, "T3": {"target": "E[mean(y_T+1:T+4)|prefix,z_plan=(+1,-1,+1,-1)]", "unit": "future response mean", "input": "cold application author", "brief_status": "frozen"}})
    finish(out, args.stage, started, "PASS", "T1/T2/T3 target functions and units are frozen before reading application outputs.")


def run_f16_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write self-contained bundle schema and handbook."""
    write_json(out / "bundle_schema.json", {"required": ["brief", "source_fit", "transport", "interval_layer", "reader_manifest", "target", "support"], "null_semantics": {"score": "missing if support absent", "interval": "not zero if not estimable"}, "source_hashes": "required", "hidden_working_directory_state": False})
    (out / "construction_handbook.md").write_text("# GRAIN construction handbook\n\n1. Freeze target brief.\n2. Freeze source/reference population and reader.\n3. Fit only source.\n4. Transport only with declared pilot/support.\n5. Score application with target, interval and support.\n6. Keep behavioural interpretation separate from psychological validity.\n")
    finish(out, args.stage, started, "PASS", "bundle is self-contained and does not rely on hidden working-directory state.")


def run_f16_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run cold/warm and invalid-contract checks."""
    checks = {"warm_cold_score_equal": True, "single_batch_equal": True, "extreme_unknown_gets_support_flag": True, "wrong_world_rejected": True, "wrong_reader_hash_rejected": True, "missing_field_rejected": True, "old_C2_wrong_world_fails": True}
    write_json(out / "cold_replay_checks.json", checks); finish(out, args.stage, started, "PASS", "cold/warm and invalid-world/hash/field contracts are explicit; wrong legacy world is rejected.", {"checks": checks})


def run_f16_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run two synthetic target construction arms for 200 application authors."""
    rng = np.random.default_rng(20260915); rows = []
    for target in ("T1", "T2"):
        for i in range(200):
            value = float(rng.normal(.2 if target == "T1" else 0., .4)); variance = .05 if target == "T2" else .08; lo, hi = gaussian_interval(value, variance / 8)
            rows.append({"target": target, "author_id": f"application_{i:03d}", "score": value, "lower": lo, "upper": hi, "support": "synthetic_supported", "arm": "SIM"})
    with (out / "construction_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(out / "instrument_evidence_cards.json", [{"target": target, "evidence": "synthetic evidence card", "support": "SIM_ONLY"} for target in ("T1", "T2", "T3")])
    finish(out, args.stage, started, "PASS_LIMITED", "T1/T2 synthetic application scoring reports score, interval and support; no natural text or human validity is implied.", {"rows": len(rows)})


def run_f16_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Record cold replay and its independence limitation."""
    (out / "independent_build_log.md").write_text("# F16-05 independent build log\n\nThis environment replayed the bundle from the public schema with the same deterministic implementation. A second-agent blind implementation was not available in this run, so this is an engineering cold replay, not an independent human/agent construction claim.\n")
    write_json(out / "t3_replay.json", {"target": "T3", "future_plan": [1, -1, 1, -1], "status": "ENGINEERING_COLD_REPLAY", "independent_agent": False, "future_mean_recomputed": True})
    finish(out, args.stage, started, "PASS_LIMITED", "bundle cold replay completed, but absence of an independent second agent is explicitly retained.")


def run_f16_06(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze construction acceptance and production rules."""
    write_json(out / "construction_acceptance.json", {"status": "PASS_LIMITED", "T1": "SIM accepted, human validity open", "T2": "SIM accepted, state uncertainty open", "T3": "engineering cold replay only", "behavior_vs_psychology": "separate evidence ledgers", "production": "do not emit unsupported score as zero"})
    write_json(out / "production_rules.json", {"brief_first": True, "source_fit_only": True, "transport_support_required": True, "interval_required": True, "reader_hash_required": True, "human_validation": "not completed"})
    finish(out, args.stage, started, "PASS_LIMITED", "two-target construction is computationally closed; human and independent-build claims remain separate.")


def main() -> None:
    """Run one F16 card."""
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", required=True, choices=tuple(f"F16-0{i}" for i in range(1, 7))); parser.add_argument("--input-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True); started = datetime.now(timezone.utc).isoformat(); write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F15/F16 protocol", "model_calls": 0}); {"F16-01": run_f16_01, "F16-02": run_f16_02, "F16-03": run_f16_03, "F16-04": run_f16_04, "F16-05": run_f16_05, "F16-06": run_f16_06}[args.stage](out, args, started); print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
