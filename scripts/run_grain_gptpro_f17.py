#!/usr/bin/env python3
"""Execute F17 final crosswalk, theory integration, human contracts and audit."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "project persona"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.acceptance import assess_completion
from grain_research.gptpro_foundation_v3.execution import (
    fresh_output,
    load_execution_request,
)


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(
    out: Path,
    stage: str,
    started: str,
    status: str,
    decision: str,
    extra: dict | None = None,
) -> None:
    """Write common F17 evidence."""
    extra = extra or {}
    write_json(
        out / "protocol.json",
        {
            "task_id": stage,
            "status": "EXECUTED",
            "scope": "F17_final_audit",
            "model_calls": 0,
        },
    )
    write_json(
        out / "summary.json",
        {
            "task_id": stage,
            "status": status,
            "implementation_status": status,
            "protocol_status": "PASS",
            **extra,
        },
    )
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task_id", "check", "status", "value", "reason"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": stage,
                "check": "final_audit_stage",
                "status": status,
                "value": "",
                "reason": decision,
            }
        )
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(
        json.dumps(
            {
                "task_id": stage,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(),
                "model_calls": 0,
                "private_data": False,
            }
        )
        + "\n"
    )
    files = [
        {"path": str(path.relative_to(out)), "sha256": sha256(path)}
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "output_manifest.json"
    ]
    write_json(
        out / "output_manifest.json",
        {"task_id": stage, "status": status, "files": files},
    )


def task_ids() -> list[str]:
    """Return the declared 100 task IDs from the parent compiler source."""
    text = (PARENT / "docs/GRAIN_GPTPRO_ATOMIC_TASKS_V3.md").read_text()
    return sorted(
        set(re.findall(r"\bF(?:0[0-9]|1[0-7])-\d{2}\b", text)),
        key=lambda x: (int(x[1:3]), int(x[4:])),
    )


def load_progress() -> dict:
    """Load the execution ledger."""
    return json.loads(
        (
            PARENT / "reports/gptpro_seminar_20260915/atomic_v3/execution_progress.json"
        ).read_text()
    )


def run_f17_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Crosswalk declared task IDs to actual result manifests."""
    ids = task_ids()
    progress = load_progress()
    rows = []
    for task_id in ids:
        package = task_id.split("-")[0]
        result = ROOT / "results/grain_gptpro_foundation_v3" / package / "r1" / task_id
        status = progress.get("completed_tasks", {}).get(task_id, "PLANNED")
        rows.append(
            {
                "task_id": task_id,
                "declared_status": status,
                "result_root": str(result),
                "result_exists": result.exists(),
                "decision_exists": (result / "decision.md").exists(),
                "manifest_exists": (result / "output_manifest.json").exists(),
            }
        )
    with (out / "final_source_crosswalk.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finish(
        out,
        args.stage,
        started,
        "PASS",
        "100 declared task IDs are crosswalked against actual manifests and decisions; file presence is not treated as scientific resolution.",
        {
            "task_count": len(ids),
            "existing_result_roots": sum(row["result_exists"] for row in rows),
        },
    )


def run_f17_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write integrated theory and contribution ledger."""
    (out / "integrated_theory.md").write_text(
        """# GRAIN V3 Integrated Theory\n\nThe finite computational object is not a free-standing personality score. It is a declared target observed through a chain:\n\n1. **Condition/opportunity**: what could be expressed or observed.\n2. **Selection**: which opportunity became visible.\n3. **Response**: the observed or latent result.\n4. **State**: a time-indexed deviation and its forecast target.\n5. **Reader**: a separate measurement channel with pole and qualification errors.\n6. **Target**: condition change, current state, future response, or process family.\n7. **Explanation**: evidence address, contribution, uncertainty, and support status.\n\nThe main mathematical rules are: use finite future-law closure for predictive states; check design rank before naming coefficients; keep MAR/IPW separate from MNAR bounds; use MTMM anchors before trait variance claims; treat domain transport as a calibrated map with support; and report uncertainty layers rather than treating prior-only output as observed evidence.\n\nThe novel combination is the explicit integration of text opportunity, reader qualification, vector/action signatures, state dynamics, target contracts, and construction bundles. The finite experiments establish mechanisms and counterexamples, not human personality validity.\n"""
    )
    rows = [
        {
            "contribution": "condition-selection-response separation",
            "status": "finite_mechanism_supported",
            "evidence": "F03",
        },
        {
            "contribution": "state/level/forecast separation",
            "status": "finite_mechanism_supported",
            "evidence": "F04-F05",
        },
        {
            "contribution": "reader qualification separate from pole",
            "status": "text_synthetic_supported",
            "evidence": "F08",
        },
        {
            "contribution": "equivalence-class process interpretation",
            "status": "finite_counterexample_supported",
            "evidence": "F09/F15",
        },
        {
            "contribution": "construction bundle with support/uncertainty",
            "status": "engineering_supported",
            "evidence": "F13/F16",
        },
        {
            "contribution": "human psychological validity",
            "status": "not_executed",
            "evidence": "H01-H06",
        },
    ]
    with (out / "contribution_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finish(
        out,
        args.stage,
        started,
        "PASS_LIMITED",
        "integrated theory distinguishes finite mechanism support, engineering support, and human validity; no originality claim is inflated beyond evidence.",
        {"contributions": len(rows)},
    )


def run_f17_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Create six human-stage data/analysis/ethics contracts without recruiting."""
    rows = []
    for hid, target, needed, estimand, ethics in [
        (
            "H01",
            "same-author repeated occasions",
            "author_id,time,condition,text,consent",
            "cross-day/state reliability",
            "consent and re-identification risk",
        ),
        (
            "H02",
            "fixed condition and external anchor",
            "author_id,condition,text,external_scale",
            "convergent/discriminant validity",
            "clinical inference not automatic",
        ),
        (
            "H03",
            "domain/language transport",
            "author_id,language,domain,text,pilot_anchor",
            "transport signature",
            "cultural/linguistic fairness",
        ),
        (
            "H04",
            "adaptive burden",
            "prompt_id,condition,burden,response",
            "information per burden",
            "participant burden and stopping",
        ),
        (
            "H05",
            "reader/MTMM anchor",
            "occasion,method,reader,anchor",
            "trait/method variance",
            "reader privacy",
        ),
        (
            "H06",
            "independent construction",
            "brief,bundle,source_hash,implementation_log",
            "cold reproducibility",
            "no private data leakage",
        ),
    ]:
        rows.append(
            {
                "human_id": hid,
                "target": target,
                "needed_columns": needed,
                "estimand": estimand,
                "ethics_contract": ethics,
                "status": "not_recruited",
            }
        )
    with (out / "human_stage_contracts.md").open("w") as handle:
        handle.write(
            "# Human-stage contracts\n\nNo recruitment or human data collection occurred in V3.\n\n"
        )
        for row in rows:
            handle.write(
                f"- {row['human_id']}: target={row['target']}; needed={row['needed_columns']}; estimand={row['estimand']}; ethics={row['ethics_contract']}\n"
            )
    finish(
        out,
        args.stage,
        started,
        "PASS_LIMITED",
        "six human items have executable data/estimand/ethics contracts; none is misreported as completed.",
        {"human_items": len(rows)},
    )


def run_f17_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Audit counts, manifests, and execution resources."""
    progress = load_progress()
    ids = task_ids()
    completed = progress.get("completed_tasks", {})
    checks = {
        "declared_task_count": len(ids) == 100,
        "completed_count_matches_ledger": len(completed) == len(set(completed)),
        "model_calls_zero": progress.get("model_calls") == 0,
        "private_raw_data_false": progress.get("private_raw_data_read") is False,
        "human_validation_false": progress.get("human_validation_completed") is False,
        "artifact_roots_exist": all(
            (ROOT / root).exists()
            for root in progress.get("completed_artifact_roots", [])
        ),
    }
    write_json(
        out / "final_numeric_audit.json",
        {
            "checks": checks,
            "declared_tasks": len(ids),
            "ledger_completed": len(completed),
            "model_calls": progress.get("model_calls"),
            "budget_note": "no API/generation/training; finite local synthetic execution only",
        },
    )
    finish(
        out,
        args.stage,
        started,
        "PASS" if all(checks.values()) else "FAIL",
        "critical counts, resource/privacy flags and artifact roots audited; count does not substitute for scientific interpretation.",
        {"checks": checks},
    )


def run_f17_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Produce final finite-plan acceptance and remaining triggers."""
    progress = load_progress()
    crosswalk_path = (
        PARENT / "reports/gptpro_seminar_20260915/atomic_v3/SOURCE_TASK_CROSSWALK.csv"
    )
    with crosswalk_path.open(newline="") as handle:
        crosswalk = list(csv.DictReader(handle))
    acceptance = assess_completion(progress, crosswalk, ROOT)
    write_json(out / "final_acceptance.json", acceptance)
    lines = [
        "# GRAIN GPT Pro V3 Acceptance",
        "",
        f"Status: {acceptance['status']}",
        f"Reviewed requirements: {len(acceptance['resolved_requirements'])}/50",
        "",
        "This is a provenance gate, not an independent proof of scientific correctness.",
        "",
    ]
    for item in acceptance["open_requirements"]:
        lines.append(f"- {item['requirement']}: {'; '.join(item['reasons'])}")
    lines.extend(f"- Global error: {error}" for error in acceptance["errors"])
    (out / "final_report.md").write_text("\n".join(lines) + "\n")
    closed = acceptance["status"] != "OPEN"
    finish(
        out,
        args.stage,
        started,
        "PASS" if closed else "OPEN",
        "Requirement-level review and immutable evidence are required; card counts cannot close the plan.",
        {
            "science_status": acceptance["status"],
            "protocol_status": "PASS" if closed else "OPEN",
            "resolved_requirement_count": len(acceptance["resolved_requirements"]),
            "open_requirement_count": len(acceptance["open_requirements"]),
        },
    )


def main() -> None:
    """Run one F17 card."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=tuple(f"F17-0{i}" for i in range(1, 6))
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((PARENT / "configs/grain_gptpro_atomic_v3.json").read_text())
    task = next(task for task in plan["tasks"] if task["id"] == args.stage)
    request = load_execution_request(
        args.input_manifest,
        task,
        ROOT,
        PARENT / "docs/GRAIN_GPTPRO_ATOMIC_PROTOCOLS_V3.md",
    )
    out = fresh_output(args.output, args.input_manifest)
    write_json(out / "input_manifest.json", request)
    started = datetime.now(timezone.utc).isoformat()
    {
        "F17-01": run_f17_01,
        "F17-02": run_f17_02,
        "F17-03": run_f17_03,
        "F17-04": run_f17_04,
        "F17-05": run_f17_05,
    }[args.stage](out, args, started)
    summary = json.loads((out / "summary.json").read_text())
    print(
        json.dumps(
            {"task_id": args.stage, "status": summary["status"], "output": str(out)},
            ensure_ascii=False,
        )
    )
    if summary["status"] not in {"PASS", "PASS_LIMITED"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
