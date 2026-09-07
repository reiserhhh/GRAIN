#!/usr/bin/env python3
"""Verify a complete frozen reciprocity packet and decompose its question effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.reciprocity import (
    SYSTEM,
    fit_directions,
    material_identity,
)
from grain_research.vector_mechanisms.reciprocity_audit import (
    check_panel,
    component_summary,
    gap_changes,
    polarity_components,
)
from scripts.run_grain_vector_mechanism_controls import write_json


def audit(parent: Path, out: Path):
    """Audit immutable records without calling a model or using any human labels."""
    status = json.loads((parent / "run_status.json").read_text())
    if status["status"] != "COMPLETE":
        raise ValueError("Incomplete run cannot be accepted")
    manifest = json.loads((parent / "manifest.json").read_text())
    identity = manifest.pop("identity")
    if material_hash(manifest) != identity or status["identity"] != identity:
        raise ValueError("Run identity mismatch")
    material = json.loads((parent / "materials.json").read_text())
    if (
        material["system"] != SYSTEM
        or material_identity(material["rows"]) != manifest["material_identity"]
    ):
        raise ValueError("Material changed")
    with zipfile.ZipFile(parent / "source_snapshot.zip") as saved:
        for path, expected in manifest["code"].items():
            if hashlib.sha256(saved.read(path)).hexdigest() != expected:
                raise ValueError("Source snapshot mismatch")
    config = manifest["config"]
    baseline = pd.read_csv(parent / "baselines.csv")
    effects = pd.read_csv(parent / "interventions.csv")
    panel = check_panel(material["rows"], config, baseline, effects)
    if panel["effect_rows"] != status["intervention_rows"]:
        raise ValueError("Status row count mismatch")
    if len(baseline) != status["cases"] or status["calls"] > config["maximum_calls"]:
        raise ValueError("Status case count or budget mismatch")
    checks = json.loads((parent / "instrument_checks.json").read_text())
    maximum = max(row["max_logit_error"] for row in checks)
    if (
        maximum > config["replay_tolerance"]
        or maximum != status["instrument_max_error"]
    ):
        raise ValueError("Instrument check mismatch")
    states = {}
    for row in material["rows"]:
        if row["split"] == "development":
            with np.load(
                parent / "activation_cache" / f"{row['case_id']}.npz"
            ) as arrays:
                states[row["case_id"]] = {s: arrays[s] for s in config["sites"]}
    derived = fit_directions(material["rows"], states, config["sites"])
    with np.load(parent / "development_directions.npz") as arrays:
        if set(arrays.files) != set(derived):
            raise ValueError("Direction keys mismatch")
        error = max(float(np.max(abs(arrays[k] - v))) for k, v in derived.items())
    if error > 1e-10:
        raise ValueError("Direction is not the development-only paired mean")
    components = polarity_components(effects)
    summaries = component_summary(components)
    gaps = gap_changes(baseline, effects)
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Fresh audit directory required")
    components.to_csv(out / "query_polarity_components.csv", index=False)
    summaries.to_csv(out / "query_polarity_summary.csv", index=False)
    gaps.to_csv(out / "discrimination_gap_changes.csv", index=False)
    paths = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/reciprocity_audit.py",
    ]
    with zipfile.ZipFile(
        out / "audit_source.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    result = {
        "status": "PASS_PACKET_INTEGRITY",
        "parent_identity": identity,
        **panel,
        "instrument_max_error": maximum,
        "direction_rebuild_max_error": error,
        "no_neural_forwards": True,
        "scientific_promotion": "REQUIRES_CONTROL_INTERPRETATION",
        "scope": "post-panel diagnostic decomposition, not predeclared causal identification",
        "audit_code": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    write_json(out / "audit.json", result)
    lines = [
        "# 交叉干预完整性与问法分解",
        "",
        f"{result['status']}；{len(baseline)} baselines，{len(effects)} interventions。",
        "",
        "polarity_even=(正向问法效应+反向问法的语义计分效应)/2；",
        "polarity_odd=(正向问法效应-反向问法的语义计分效应)/2。",
        "在简单加性模型下 odd 与通用 Yes 偏置相容，但它不是答案偏置的唯一因果估计；",
        "even 也不自动等于心理概念。其他依赖问法的计算同样可以产生这些成分。",
        "这是既有控制的事后诊断分解，不重新拟合模型或改变实验条件。",
        "",
        "| 家族 | 来源 | 层 | 成分 | signal | signal-random | 配对95% CI |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in summaries.to_dict("records"):
        lines.append(
            f"| {r['family']} | {r['source']} | {r['site']} | {r['component']} | "
            f"{r['signal']:.4f} | {r['signal_minus_random']:.4f} | "
            f"[{r['paired_ci_low']:.4f}, {r['paired_ci_high']:.4f}] |"
        )
    lines += [
        "",
        "均为固定小面板的 pointwise 场景 bootstrap，非总体心理效度区间。",
        "discrimination_gap_changes.csv 另列同一方向施加到高低材料后是否真的增加两类",
        "可区分性。全体分数一起移动不等于预测改善。G/R 的原始效应量尺不同。",
        "完整性 PASS 不是共享人格机制 PASS，也不自动启动 GRAIN 作者群体分析。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    """Require an explicit completed parent and never overwrite an audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.parent_dir, args.output_dir)


if __name__ == "__main__":
    main()
