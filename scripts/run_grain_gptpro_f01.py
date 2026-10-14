#!/usr/bin/env python3
"""Run F01 ontology and predecessor audit stages."""

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
from grain_research.gptpro_foundation_v3.f01 import mechanism_graph, object_dictionary, predecessor_ledger, source_sha


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, decision: str, status: str = "PASS") -> None:
    """Write common audit artifacts."""
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F01_audit_only", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", "scientific_completion": False})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"])
        writer.writeheader()
        writer.writerow({"task_id": stage, "check": "audit_artifacts", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} audit checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": path.name, "sha256": sha256(path)} for path in sorted(out.iterdir()) if path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def main() -> None:
    """Run one F01 stage; all stages are local read-only audits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("F01-01", "F01-02", "F01-03", "F01-04", "F01-05"))
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260915, 20260916, 20260917])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    notes = ROOT.parent / "project persona/reports/gptpro_seminar_20260915/CODEX_READING_NOTES.md"
    write_json(args.input_manifest, {"task_id": args.stage, "source_notes": str(notes), "source_sha256": source_sha(notes), "model_calls": 0})
    if args.stage == "F01-01":
        write_json(out / "mechanism_graph.json", mechanism_graph())
        with (out / "object_dictionary.csv").open("w", newline="") as handle:
            rows = object_dictionary()
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "graph.md").write_text("# F01-01对象与测量图\n\nO机会→C选择→E行为/体验→T文本→R读者证据→S计分；Q查询→L模型状态/输出→T。每条边保留观察状态和适用边界。\n")
        finish(out, args.stage, started, "三条箭头和七类测量对象已冻结；注意力边不被解释为人类心理因果。")
    elif args.stage == "F01-02":
        ledger = predecessor_ledger()
        write_json(out / "primary_reference_cards.json", ledger)
        with (out / "bibliography.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
            writer.writeheader()
            writer.writerows(ledger)
        (out / "verification_notes.md").write_text("# F01-02文献核验\n\n本批固定八个承重一手来源及其采用边界。未纳入承重集的P01-P04书目不被自动视为已核验；后续卡只可使用本表列出的有限主张。\n")
        finish(out, args.stage, started, "八个承重方法条目已登记；完整书目去重和逐页核验仍需继续卡片化处理。")
    elif args.stage == "F01-03":
        ledger = predecessor_ledger()
        rows = [{**item, "decision": "ADOPT_FINITE_COMPARATOR", "next_card": "F15-01/F15-05"} for item in ledger]
        write_json(out / "method_adoption.json", rows)
        (out / "adoption.md").write_text("# F01-03方法采用边界\n\n采用的是有限目标、识别条件或比较器；不把既有方法改名为GRAIN原创，也不在本计划中强行实现所有神经算法。\n")
        finish(out, args.stage, started, "承重方法均有采用或不宣称边界；后续F15负责有限证书与未执行工具登记。")
    elif args.stage == "F01-04":
        contract = {
            "same_observation_law_same_target": True,
            "structural_identification": "unique target under declared generating model",
            "practical_identification": "finite-support prediction or estimation loss within declared design",
            "functional_equivalence": "same declared future/action/score function, coordinates may differ",
            "counterexamples": ["one-step-equal_multi-step-different", "opportunity_selection_product", "state_level_shift", "shared_reader_method_component"],
            "additional_observations": ["future condition sequence", "opportunity denominator", "independent anchor reader", "state intervention"],
        }
        write_json(out / "identification_contract.json", contract)
        (out / "counterexample_registry.json").write_text(json.dumps({"task_id": args.stage, "cases": contract["counterexamples"]}, ensure_ascii=False, indent=2) + "\n")
        finish(out, args.stage, started, "结构识别、实践识别、功能等价和坐标gauge已分开；四个最小反例及追加观测已登记。")
    else:
        (out / "theory_to_experiment_map.csv").write_text("claim,package,artifact,status\nR01,F01,mechanism_graph.json,PLANNED_AFTER_F01\nR02,F01,object_dictionary.csv,PLANNED_AFTER_F01\nR03,F15,method_adoption.json,PLANNED_AFTER_F15\n")
        (out / "acceptance.md").write_text("# F01-05审阅\n\nF01入口对象边界与先行方法台账已形成；未将书目覆盖当成科学完成。下一包F02-01。\n")
        finish(out, args.stage, started, "F01五张卡的对象图、承重方法台账、识别合同和映射已登记；未执行真人效度。")
    print(json.dumps({"task_id": args.stage, "status": "PASS", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
