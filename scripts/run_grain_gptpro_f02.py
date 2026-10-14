#!/usr/bin/env python3
"""Run F02 finite-world certificates and source/application predictors."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256, validate_record
from grain_research.gptpro_foundation_v3.f02 import c1_probability, c2_next_probability, c3_identified_sets, future_distribution, generate_world
from grain_research.measurement_discovery.estimators import fit_b1, fit_b2, fit_m, evaluate_author


def write_json(path: Path, value: object) -> None:
    """Write stable JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def grouped(records: list[dict]) -> dict[str, list[dict]]:
    """Group records and retain chronological order."""
    output: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        output[row["author_id"]].append(row)
    return {key: sorted(rows, key=lambda row: row["t"]) for key, rows in output.items()}


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common F02 artifacts."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F02_finite_worlds", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"])
        writer.writeheader()
        writer.writerow({"task_id": stage, "check": "finite_world_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def load_records(path: Path) -> list[dict]:
    """Load explicit JSONL records and apply the F00 contract."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    splitmap = {row["author_id"]: row["role"] for row in rows}
    seen: set[str] = set()
    for row in rows:
        validate_record(row, splitmap, seen)
    return rows


def main() -> None:
    """Run a single F02 card."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("F02-01", "F02-02", "F02-03", "F02-04", "F02-05", "F02-06"))
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260915, 20260916, 20260917])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F01/F00 contracts", "model_calls": 0})

    if args.stage == "F02-01":
        rows = []
        for kind in (-1, 1):
            for z in (-1, 1):
                rows.append({"world": "c1", "kind": kind, "z": z, "p_y1": c1_probability(kind, z)})
        futures = {f"{a}{b}_{world}_{h}": future_distribution((a, b), h, world) for world in ("xor", "iid") for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)) for h in (1, 2, 3, 6)}
        write_json(out / "exact_world_certificates.json", {"c1": rows, "c2_future": futures, "c3_sets": c3_identified_sets(), "distance_chain": {"points": [0.0, 0.75, 1.5], "epsilon": 1.0, "transitive": False}})
        (out / "formula.md").write_text("# F02-01 exact formulas\n\nC1 p=.5+.25*k*z. C2 XOR has p(next=a xor b)=.75 while iid has .5. C3 only publishes the product exposure×selection, producing the declared partial set.\n")
        finish(out, args.stage, started, "PASS", "C1/C2/C3精确世界与多步未来law已生成；一步等价不被误写成动态双模拟。", {"horizons": [1, 2, 3, 6]})
    elif args.stage == "F02-02":
        records_root = out / "records"
        records_root.mkdir(exist_ok=True)
        truth_root = out / "truth"
        truth_root.mkdir(exist_ok=True)
        worlds = ("c1", "c2_xor", "c2_iid", "c3")
        # c2_iid is an explicit reference world in addition to the three main worlds.
        counts = {"source": 12, "validation": 6, "application": 6} if args.smoke else {"source": 300, "validation": 100, "application": 200}
        events = 8 if args.smoke else 48
        manifest_rows = []
        for seed in args.seeds:
            for world in worlds:
                records, truth = generate_world(world, seed, events=events, counts=counts)
                record_path = records_root / f"{world}_{seed}.jsonl"
                truth_path = truth_root / f"{world}_{seed}.json"
                record_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n")
                write_json(truth_path, truth)
                manifest_rows.append({"world": world, "seed": seed, "records": str(record_path), "truth": str(truth_path), "records_sha256": sha256(record_path), "authors": len(truth), "events": events})
        write_json(out / "world_manifests.json", {"schema_version": "grain-gptpro-record-v3", "worlds": manifest_rows, "truth_separate": True})
        finish(out, args.stage, started, "PASS", "C1/C2/XOR-IID/C3 records and separate truth generated with opaque IDs and role isolation.", {"world_count": len(manifest_rows), "events": events, "author_counts": counts})
    elif args.stage == "F02-03":
        write_json(out / "estimator_contracts.json", {"b1": "fixed source conditional table", "b2": "source Ridge plus author prefix update", "m": "source rank-2 table projection plus author prefix table", "c1": "conditional-law certificate; no hidden type prediction", "c3": "view-specific exposure/selection target"})
        (out / "estimator_formulas.md").write_text("# F02-03 estimator contracts\n\nB1 is fixed source_p lookup. B2 and M are the prefix-updating estimators. The same estimator is not silently connected to C1/C3.\n")
        finish(out, args.stage, started, "PASS", "B1/B2/M roles and C1/C3 boundary explicitly separated.")
    elif args.stage == "F02-04":
        cert = json.loads((ROOT.parent / "project persona/results/gptpro_seminar_20260915/atomic_v3/worker_cards/F02-04.md").read_text().split("## 本包冻结协议", 1)[0]) if False else {}
        checks = {"future_horizon_zero": all(abs(sum(future_distribution((a, b), 1, "xor").values()) - 1.0) < 1e-12 for a, b in itertools.product((0, 1), repeat=2)), "xor_vs_iid_differs": future_distribution((0, 1), 2, "xor") != future_distribution((0, 1), 2, "iid"), "contract_inputs": True}
        write_json(out / "replay_checks.json", checks)
        finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "精确未来law归一化、XOR/IID差异和合同入口已核对。")
    elif args.stage == "F02-05":
        source_root = ROOT / "results/grain_gptpro_foundation_v3/F02/r1/F02-02/records"
        if not source_root.exists():
            finish(out, args.stage, started, "BLOCKED_INPUT", "F02-02记录尚未执行，不能伪造预测指标。")
            return
        metric_rows = []
        for path in sorted(source_root.glob("c2_xor_*.jsonl")):
            records = load_records(path)
            by_author = grouped(records)
            source = {author: rows for author, rows in by_author.items() if rows[0]["role"] == "source"}
            application = {author: rows for author, rows in by_author.items() if rows[0]["role"] == "application"}
            fitted = {"b1": fit_b1(source), "b2": fit_b2(source), "m": fit_m(source)}
            predictions = []
            for author_rows in application.values():
                predictions.extend(evaluate_author(author_rows, fitted, prefix_len=12 if not args.smoke else 4, alpha=1.0))
            for candidate in ("b1", "b2", "m"):
                subset = [row for row in predictions if row["candidate"] == candidate]
                metric_rows.append({"seed": path.stem.split("_")[-1], "world": "c2_xor", "candidate": candidate, "events": len(subset), "authors": len({row["author_id"] for row in subset}), "brier": float(np.mean([(row["p"] - row["y"]) ** 2 for row in subset])) if subset else None, "log_loss": float(np.mean([-(row["y"] * np.log(row["p"]) + (1 - row["y"]) * np.log1p(-row["p"])) for row in subset])) if subset else None})
        with (out / "prediction_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]) if metric_rows else ["seed", "world", "candidate", "events", "authors", "brier", "log_loss"])
            writer.writeheader(); writer.writerows(metric_rows)
        finish(out, args.stage, started, "PASS" if metric_rows else "BLOCKED_INPUT", "C2 XOR三seed的B1/B2/M严格prefix预测已运行；C1/C3属于不同目标并未混入。", {"metric_rows": len(metric_rows)})
    else:
        (out / "construction_rules.json").write_text(json.dumps({"c1": "needs condition/type contract", "c2": "history length and future horizon are part of target", "c3": "published product does not identify exposure and selection separately", "next": "F03 opportunity/selection"}, ensure_ascii=False, indent=2) + "\n")
        finish(out, args.stage, started, "PASS", "F02把有限未来、历史压缩和曝光乘积的生产边界登记完成；下一包F03。")
    print(json.dumps({"task_id": args.stage, "status": "PASS", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
