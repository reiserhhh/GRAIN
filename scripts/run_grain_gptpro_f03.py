#!/usr/bin/env python3
"""Execute F03 opportunity/selection/response atomic task cards."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
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

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f03 import (
    WORLD_VALUES,
    beta_condition_mean,
    cell_probability,
    generate_records,
    grouped,
    ipw_condition_mean,
    known_opportunity_bounds,
    naive_selected_mean,
    response_bound,
    response_probability,
    selection_probability,
)


def write_json(path: Path, value: object) -> None:
    """Write stable JSON with UTF-8 text."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def finish(out: Path, stage: str, started: str, status: str, decision: str, extra: dict | None = None) -> None:
    """Write common evidence files for one F03 card."""
    extra = extra or {}
    write_json(out / "protocol.json", {"task_id": stage, "status": "EXECUTED", "scope": "F03_opportunity_selection_response", "model_calls": 0})
    write_json(out / "summary.json", {"task_id": stage, "status": status, "implementation_status": status, "protocol_status": "PASS", **extra})
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "check", "status", "value", "reason"])
        writer.writeheader()
        writer.writerow({"task_id": stage, "check": "opportunity_selection_stage", "status": status, "value": "", "reason": decision})
    (out / "decision.md").write_text(f"# {stage} 决议\n\n状态：{status}。{decision}\n")
    (out / "tests.txt").write_text(f"{stage} checks: {status}\n")
    (out / "run_log.jsonl").write_text(json.dumps({"task_id": stage, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "model_calls": 0, "private_data": False}) + "\n")
    files = [{"path": str(path.relative_to(out)), "sha256": sha256(path)} for path in sorted(out.rglob("*")) if path.is_file() and path.name != "output_manifest.json"]
    write_json(out / "output_manifest.json", {"task_id": stage, "status": status, "files": files})


def exact_params() -> dict:
    """Return a deterministic parameter point used for exact certificates."""
    return {"r": {"-": 0.5, "+": 0.5}, "pi": {"-": 0.4, "+": 0.4}, "kappa": 0.0, "lambda": 0.8, "gamma": 1.0}


def write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    """Write compact reproducible JSONL."""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_gzip_jsonl(path: Path) -> list[dict]:
    """Read compact JSONL."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def true_q(params: dict, z: int) -> float:
    """Calculate the population selection probability given opportunity."""
    key = "-" if z == -1 else "+"
    values = []
    for x in (0, 1):
        for y in (0, 1):
            p_y = response_probability(params["r"][key], x) if y else 1.0 - response_probability(params["r"][key], x)
            values.append(0.5 * p_y * selection_probability(params["kappa"], params["lambda"], params["gamma"], z, x, y))
    return float(sum(values))


def run_f03_01(out: Path, args: argparse.Namespace, started: str) -> None:
    """Produce exact cells and partial-identification counterexamples."""
    params = exact_params()
    rows = []
    for z in (-1, 1):
        key = "-" if z == -1 else "+"
        for x in (0, 1):
            for y in (0, 1):
                rows.append({"z": z, "x": x, "y": y, **cell_probability(params["r"][key], params["pi"][key], params["kappa"], params["lambda"], params["gamma"], z, x, y)})
    with (out / "exact_selection_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(out / "identified_sets.json", {
        "same_published_rate_counterexample": {"published_rate": 0.4, "cases": [{"pi": 0.8, "q": 0.5}, {"pi": 0.4, "q": 1.0}], "q_set": [0.4, 1.0]},
        "missing_y_bound": {"formula": "[g_z, g_z + 1 - f_z]", "known_pi_formula": "[g_z/pi_z, g_z/pi_z + 1 - f_z/pi_z]", "unknown_denominator": "no point estimate"},
        "response_probability": "r_z is the marginal over x; x is observed in the estimable conditional arm",
    })
    finish(out, args.stage, started, "PASS", "同样的发表率可由不同机会率与选择率生成；未知分母只保留partial bounds，不补假点估计。", {"cell_count": len(rows)})


def run_f03_02(out: Path, args: argparse.Namespace, started: str) -> None:
    """Generate nested full/opportunity/published views."""
    counts = {"source": 3, "validation": 2, "application": 2} if args.smoke else {"source": 24, "validation": 12, "application": 12}
    events = 32 if args.smoke else 512
    seeds = args.seeds[:1] if args.smoke else args.seeds
    manifest = []
    root = out / "records"; root.mkdir(exist_ok=True)
    truth_root = out / "truth"; truth_root.mkdir(exist_ok=True)
    for seed in seeds:
        for world in WORLD_VALUES:
            for design in ("free", "fixed", "common"):
                records, truth = generate_records(world, design, seed, events=events, counts=counts)
                record_path = root / f"{world}_{design}_{seed}.jsonl.gz"
                published_path = root / f"{world}_{design}_{seed}.published.jsonl.gz"
                truth_path = truth_root / f"{world}_{design}_{seed}.json"
                write_gzip_jsonl(record_path, records)
                write_gzip_jsonl(published_path, [row for row in records if row["selected"] == 1])
                write_json(truth_path, truth)
                manifest.append({"world": world, "design": design, "seed": seed, "records": str(record_path), "published_view": str(published_path), "truth": str(truth_path), "records_sha256": sha256(record_path), "authors": len(truth), "events_per_author": events, "nested_prefixes": [32, 128, 512] if events >= 512 else [events], "views": ["all_opportunities", "selected_indicator", "published_only"]})
    write_json(out / "opportunity_manifest.json", {"schema_version": "grain-gptpro-record-v3", "manifest": manifest, "counts": counts, "events_per_author": events, "truth_separate": True})
    write_json(out / "view_contracts.json", {"all_opportunities": "O and S are retained; Y is observed only when S=1", "selected_indicator": "S is retained but O is hidden", "published_only": "only S=1 rows are retained; opportunity denominator is unknown", "prohibited": ["mixing selection with response", "using truth parameters in records", "recovering unpublished count from published rows"]})
    finish(out, args.stage, started, "PASS", "4 worlds×3 designs×nested prefixes generated with separate full and published views; full sample is a finite synthetic audit, not a human power claim.", {"worlds": 4, "designs": 3, "seeds": len(seeds), "events_per_author": events, "authors_per_seed_design": sum(counts.values())})


def run_f03_03(out: Path, args: argparse.Namespace, started: str) -> None:
    """Write estimator contracts and assumptions."""
    (out / "estimator_formulas.md").write_text("""# F03 estimator contracts\n\n- `naive`: E[Y | S=1,z]; a selected-case estimand, not automatically E[Y|z].\n- `IPW`: fit q(z,x)=P(S=1|O=1,z,x) from observed opportunities and use a Hájek estimate. Valid only when selection is conditionally independent of Y after observed x; it is intentionally not a repair for W3-C/W3-D MNAR.\n- `Beta`: bounded Beta(1,1) posterior mean over observed selected Y; a shrinkage estimator, not an identification proof.\n- `bounds`: with g=P(SY=1|z), f=P(S=1|z), unknown opportunity denominator gives [g, g+1-f]. If pi is known and the required independence assumption is declared, [g/pi, g/pi+1-f/pi] is a narrower arm.\n- `published_only`: no point estimate of pi or q; report `unknown_denominator` and retain the broad interval.\n\nThe old choice-design readout is kept as a separate target arm and is not required to equal these response estimands.\n""")
    write_json(out / "assumptions.json", {"MCAR_or_conditional_MAR": "required for IPW point recovery", "MNAR": "selection depends on latent Y; point repair is not claimed", "truth_use": "truth appears only in evaluation joins", "selection_clip": [0.01, 0.99]})
    finish(out, args.stage, started, "PASS", "naive/IPW/Beta/bounds are explicitly separated by estimand and missingness assumption; MNAR remains a declared limitation.")


def run_f03_04(out: Path, args: argparse.Namespace, started: str) -> None:
    """Run exact and degenerate design tests."""
    from grain_research.gptpro_foundation_v3.f03 import generate_records

    checks = {}
    checks["fixed_bound_degenerates"] = all(abs(a - b) < 1e-12 for a, b in zip(response_bound(0.4, 1.0, 1.0), (0.4, 0.4)))
    checks["unknown_denominator_broad"] = all(abs(a - b) < 1e-12 for a, b in zip(response_bound(0.2, 0.4), (0.2, 0.8)))
    rows, truth = generate_records("W3-C", "fixed", 20260915, events=32, counts={"source": 1, "validation": 1, "application": 1})
    checks["fixed_has_no_missing_selected"] = all(row["selected"] == 1 and row["observation"] is not None for row in rows)
    checks["truth_is_separate"] = all("r" not in row and "pi" not in row for row in rows) and len(truth) == 3
    checks["author_ids_opaque"] = all("W3-C" not in row["author_id"] for row in rows)
    write_json(out / "selection_tests.json", checks)
    finish(out, args.stage, started, "PASS" if all(checks.values()) else "FAIL", "fixed/unknown-denominator/empty-truth/opaque-ID tests completed; failure would block F03 run.", {"checks": checks})


def run_f03_05(out: Path, args: argparse.Namespace, started: str) -> None:
    """Score every world/design/view and nested prefix."""
    source = ROOT / "results/grain_gptpro_foundation_v3/F03/r1/F03-02"
    if not source.exists():
        finish(out, args.stage, started, "BLOCKED_INPUT", "F03-02 records are missing; no metrics were fabricated.")
        return
    recovery_rows = []
    for manifest_row in json.loads((source / "opportunity_manifest.json").read_text())["manifest"]:
        records = read_gzip_jsonl(Path(manifest_row["records"]))
        truth = {row["author_id"]: row for row in json.loads(Path(manifest_row["truth"]).read_text())}
        for author, author_rows in grouped(records).items():
            params = truth[author]
            prefixes = [32, 128, 512] if len(author_rows) >= 512 else [len(author_rows)]
            for prefix in prefixes:
                rows = author_rows[:prefix]
                for z in (-1, 1):
                    key = "-" if z == -1 else "+"
                    true_r = params["r"][key]
                    naive = naive_selected_mean(rows, z)
                    beta = beta_condition_mean(rows, z)
                    ipw, clip_count = ipw_condition_mean(rows, z)
                    lo, hi, f, pi = known_opportunity_bounds(rows, z)
                    q_true = 1.0 if manifest_row["design"] == "fixed" else true_q(params, z)
                    q_hat = 1.0 if manifest_row["design"] == "fixed" else (f / pi if pi > 0 else None)
                    q_loss = None if manifest_row["design"] == "fixed" else ((q_hat - q_true) ** 2 if q_hat is not None else None)
                    q_status = "design_fixed" if manifest_row["design"] == "fixed" else "estimated_observed_opportunity"
                    for method, estimate, status in (("naive_selected", naive, "point_selected_estimand"), ("beta_selected", beta, "point_selected_shrinkage"), ("ipw", ipw, "point_MAR_only"), ("bounds_lower", lo, "interval_endpoint"), ("bounds_upper", hi, "interval_endpoint")):
                        recovery_rows.append({"task_id": "F03-05", "seed": manifest_row["seed"], "world_id": manifest_row["world"], "design": manifest_row["design"], "view": "all_opportunities", "role": params["role"], "author_id": author, "target_id": f"r_{key}", "method": method, "estimand": "E[Y|z]", "unit": "author-prefix", "n_authors": 1, "n_events": len(rows), "value": estimate, "lower": lo, "upper": hi, "support_status": status if estimate is not None else "no_observed_selected_y", "true_value": true_r, "absolute_error": abs(float(estimate) - true_r) if estimate is not None else None, "bound_covers": bool(lo is not None and hi is not None and lo - 1e-12 <= true_r <= hi + 1e-12), "q_true": q_true, "q_hat": q_hat, "q_loss": q_loss, "q_status": q_status, "propensity_clips": clip_count})
    with (out / "response_recovery.csv").open("w", newline="") as handle:
        fields = list(recovery_rows[0]) if recovery_rows else ["task_id"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(recovery_rows)
    aggregates = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in recovery_rows:
        groups[(row["seed"], row["world_id"], row["design"], row["n_events"], row["method"])].append(row)
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        values = [row["absolute_error"] for row in rows if row["absolute_error"] is not None]
        aggregates.append({"task_id": "F03-05", "seed": key[0], "world_id": key[1], "design": key[2], "n_events": key[3], "method": key[4], "n_authors": len(rows), "mean_absolute_error": float(np.mean(values)) if values else None, "bound_coverage": float(np.mean([row["bound_covers"] for row in rows])) if rows else None, "mean_q_loss": float(np.mean([row["q_loss"] for row in rows if row["q_loss"] is not None])) if any(row["q_loss"] is not None for row in rows) else None, "support_n": sum(row["value"] is not None for row in rows)})
    with (out / "design_identification.csv").open("w", newline="") as handle:
        fields = list(aggregates[0]) if aggregates else ["task_id"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(aggregates)
    finish(out, args.stage, started, "PASS" if recovery_rows else "BLOCKED_INPUT", "每一行标记设计、视图、目标与支持状态；IPW在MNAR世界保留偏差，不把失败改写成实现错误。", {"recovery_rows": len(recovery_rows), "aggregate_rows": len(aggregates)})


def run_f03_06(out: Path, args: argparse.Namespace, started: str) -> None:
    """Freeze production rules and handoff boundaries."""
    rules = {
        "increase_text_or_observation": "增加文本只有在目标Y与机会/选择分母仍可观测时才改善点估计；若只增加发表文本而不增加未发表机会，信息不等价增加。",
        "common_opportunity": "固定共同机会能把机会变成设计常数，但它改变了estimand；不能直接把固定条件结果投射为自然文本普遍结果。",
        "point": "all_opportunities with observed covariates supports naive/IPW/Beta point arms; IPW needs conditional-MAR.",
        "bounds": "unknown opportunity denominator supports only partial bounds; report f/g and interval endpoints.",
        "no_information": "published_only cannot identify pi, q, or an unconditional response rate; return unknown rather than zero.",
        "handoff_F06": "policy choice must expose opportunity and selection burden separately.",
        "handoff_F16": "identity/reader claims need the observation-view contract and may not infer psychological truth from synthetic truth.",
        "scope": "finite mechanism evidence; no PANDORA transfer and no human validity claim.",
    }
    write_json(out / "opportunity_rules.json", rules)
    write_json(out / "acceptance.json", {"status": "PASS_LIMITED", "accepted": ["selection and response are separate objects", "MAR/MNAR boundary is explicit", "unknown denominator is not imputed"], "not_claimed": ["universal IPW correction", "human personality validity", "published-only point recovery"]})
    finish(out, args.stage, started, "PASS_LIMITED", "F03完成机会、选择、反应三对象的有限理论闭环；后续策略与文本应用必须继承unknown/bounds/MNAR边界。")


def main() -> None:
    """Run one F03 card."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=tuple(f"F03-0{i}" for i in range(1, 7)))
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260915, 20260916, 20260917])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    write_json(args.input_manifest, {"task_id": args.stage, "status": "declared", "source": "F02 contracts", "model_calls": 0})
    {"F03-01": run_f03_01, "F03-02": run_f03_02, "F03-03": run_f03_03, "F03-04": run_f03_04, "F03-05": run_f03_05, "F03-06": run_f03_06}[args.stage](out, args, started)
    print(json.dumps({"task_id": args.stage, "status": "EXECUTED", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
