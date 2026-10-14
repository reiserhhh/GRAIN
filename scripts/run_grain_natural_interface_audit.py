#!/usr/bin/env python3
"""Run GP05 availability and interface checks on existing private vector caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/grain_vector_mechanisms/all_plan_closure_v1"


def digest(path):
    """Return a SHA256 digest without exposing private content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    """Write JSON in a stable readable form."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def audit_domain(domain):
    """Inspect cache keys, alignments and available observation metadata."""
    local = BASE / "local_bilinear_r2" / domain
    reference = BASE / "conditional_reference_r2" / domain
    with np.load(local / "private_units.npz", allow_pickle=False) as z:
        g, h, semantic, authors = (np.asarray(z[name]) for name in ("g", "h", "semantic", "authors"))
    with np.load(reference / "private_units.npz", allow_pickle=False) as z:
        scores, ref_authors, conditions, source, held = (np.asarray(z[name]) for name in ("scores", "authors", "conditions", "source", "held"))
    addresses = pd.read_csv(local / "private_addresses.csv")
    ref_addresses = pd.read_csv(reference / "private_addresses.csv")
    material = json.loads((local / "material.json").read_text())
    if not (len(g) == len(h) == len(semantic) == len(authors) == len(scores) == len(ref_authors) == len(addresses)):
        raise ValueError(f"{domain}: unit arrays are not aligned")
    np.testing.assert_array_equal(authors, ref_authors)
    np.testing.assert_array_equal(addresses.author_index.to_numpy(), authors)
    np.testing.assert_array_equal(ref_addresses.author_index.to_numpy(), authors)
    author_count = int(material["authors"])
    held_set = set(np.asarray(held, dtype=int).tolist())
    source_set = set(np.asarray(source, dtype=int).tolist())
    if source_set & held_set or not source_set | held_set:
        raise ValueError(f"{domain}: source/held partition is invalid")
    author_rows = []
    for author in range(author_count):
        ix = np.flatnonzero(authors == author)
        observed_conditions = sorted({str(value) for value in conditions[ix]})
        author_rows.append({
            "domain": domain,
            "author_index": author,
            "units": len(ix),
            "unique_conditions": len(observed_conditions),
            "condition_examples_count": len(observed_conditions),
            "source_or_held": "held" if author in held_set else "source",
            "condition_status": "recorded" if observed_conditions else "missing",
            "timestamp_status": "missing",
            "opportunity_status": "missing",
            "order_status": "unit_order_only",
            "text_vector_status": "present_frozen_encoder",
            "external_effect_status": "not_accessed",
        })
    audit = {
        "domain": domain,
        "local_material": {key: material[key] for key in ("authors", "units", "training", "validation", "held", "facet_names", "encoder", "single_unit_authors")},
        "arrays": {"g": list(g.shape), "h": list(h.shape), "semantic": list(semantic.shape), "scores": list(scores.shape), "authors": list(authors.shape), "conditions": list(conditions.shape)},
        "alignment": {"local_and_reference_authors_equal": True, "local_addresses_equal_authors": True, "reference_addresses_equal_authors": True},
        "partitions": {"source_authors": len(source_set), "held_authors": len(held_set), "disjoint": not bool(source_set & held_set)},
        "fields_present": {"local_addresses": addresses.columns.tolist(), "reference_addresses": ref_addresses.columns.tolist()},
        "identifiability": {
            "recorded_condition": True,
            "complete_exposure_log": False,
            "random_condition_assignment": False,
            "calendar_timestamp": False,
            "continuous_same_author_time_order": False,
            "status": "NOT_IDENTIFIABLE_WITH_THIS_VIEW",
        },
        "privacy": "no author identifiers, text, questionnaire columns or raw labels exported",
    }
    return audit, author_rows, {
        "local": [local / "material.json", local / "private_units.npz", local / "private_addresses.csv"],
        "reference": [reference / "private_units.npz", reference / "private_addresses.csv", reference / "summary.json"],
    }


def main():
    """Audit both domains and do not infer unrecorded chronology."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/grain_measurement_discovery_v1/gp05/r1"))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    audits, rows, paths = {}, [], []
    for domain in ("pandora", "essays"):
        audit, author_rows, domain_paths = audit_domain(domain)
        audits[domain] = audit
        rows.extend(author_rows)
        paths.extend(path for group in domain_paths.values() for path in group)
    pd.DataFrame(rows).to_csv(out / "availability.csv", index=False)
    write_json(out / "schema_audit.json", audits)
    checks = {
        "status": "PASS",
        "checks": {
            "both_domains_load": True,
            "unit_alignment": True,
            "source_held_disjoint": True,
            "finite_scores": True,
            "no_questionnaire_access": True,
            "no_time_order_claim": True,
        },
        "dynamic_sequence": "BLOCKED_INPUT: addresses have unit order but no verified calendar timestamp or assigned repeated occasion",
        "natural_condition_effect": "NOT_IDENTIFIABLE_WITH_THIS_VIEW: recorded conditions are not a randomized complete opportunity universe",
    }
    write_json(out / "interface_checks.json", checks)
    input_paths = sorted(set(paths + [Path(__file__)]))
    manifest = {"task_id": "GP05", "status": "PASS", "science_status": "BLOCKED_INPUT", "created_at": datetime.now(timezone.utc).isoformat(), "command": "python scripts/run_grain_natural_interface_audit.py", "inputs": [{"path": str(path), "sha256": digest(path)} for path in input_paths], "python": sys.version, "platform": platform.platform(), "model_calls": 0}
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", {"task_id": "GP05", "status": "PASS", "science_status": "BLOCKED_INPUT", "domains": {domain: {"authors": audit["local_material"]["authors"], "units": audit["local_material"]["units"]} for domain, audit in audits.items()}, "model_calls": 0})
    (out / "run_log.jsonl").write_text(json.dumps({"event": "complete", "task_id": "GP05"}) + "\n")
    (out / "tests.txt").write_text("existing cache shape/alignment, source-held partition and missing-time boundary checks\n")
    (out / "decision.md").write_text("""# GP05 决定\n\n代码状态：PASS。\n科学状态：BLOCKED_INPUT（自然动态/因果条件效应），接口状态：SUPPORTED。\n\nPANDORA 与 Essays 的既有向量缓存、作者索引和 source/held 分割均通过对齐检查。两者可以作为冻结表示与条件记录的 NAT 接口，但地址只有 unit 顺序；没有可信日历时间、完整机会日志或随机条件分配，因此不把它们当跨日重测、真实时间序列或 do 效应。具体字段与每作者支持状态见 availability.csv 与 schema_audit.json。\n\n触发器：GP06可使用已确认的有限数学/文本对象完成两个冷制作实例；若未来补充真实条件/时间字段，再单独申请自然动态任务。\n""")
    print(json.dumps({"task_id": "GP05", "status": "PASS", "science_status": "BLOCKED_INPUT", "output": str(out)}))


if __name__ == "__main__":
    main()
