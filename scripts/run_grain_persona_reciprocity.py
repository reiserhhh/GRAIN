#!/usr/bin/env python3
"""Bounded local persona-expression / author-recognition cross-intervention pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.reciprocity import (
    SYSTEM,
    analysis_tables,
    arm_schedule,
    build_material,
    fit_directions,
    material_identity,
    random_direction,
)
from grain_research.vector_mechanisms.trace import DecoderObserver, Patch
from scripts.run_grain_vector_mechanism_controls import write_json

EXPECTED_METADATA = {
    "config.json": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


class BudgetExceeded(RuntimeError):
    """Bounded research must preserve partial records without claiming completion."""


def file_hash(path: Path) -> str:
    """Stream large cached weights without duplicating them in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config):
    """Reject silent protocol changes and unbounded/invalid dose schedules."""
    if config["position"] != "last_prompt_token":
        raise ValueError("This pilot only implements last_prompt_token")
    if not config["sites"] or len(set(config["sites"])) != len(config["sites"]):
        raise ValueError("Sites must be unique and nonempty")
    doses = config["doses"]
    if len(set(doses)) != len(doses) or not {-1, 1} <= set(doses):
        raise ValueError("Unique doses must include signed unit interventions")
    if not np.isfinite(doses).all() or max(abs(x) for x in doses) > 2:
        raise ValueError("Unbounded or invalid dose")
    for key in ("random_replicates", "maximum_calls", "maximum_seconds", "threads"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"Invalid positive budget {key}")


def export_tables(out, baselines, effects):
    """Persist complete records, even when a later condition fails."""
    pd.DataFrame(baselines).to_csv(out / "baselines.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "interventions.csv", index=False)


def report(out: Path, summary: dict, capability: pd.DataFrame, matrix: pd.DataFrame):
    """Write an explicit exploratory readout without automatically selecting a site."""
    lines = [
        "# Persona 表达／识别交叉干预：P1",
        "",
        f"状态：{summary['status']}；smoke={summary['smoke']}。",
        "合成 social-initiation 机制面板，不是完整 Big Five，也不是真人人格预测。",
        "没有自由生成打分：G 为自然语言行为候选的整句似然比较，R 为 Yes/No 语义比较。",
        "R_direct 是行为事实识别；R_trait 是模型的人格归因，两者不能混作同一种能力。",
        "所有方向仅从 development 场景提取，未依据 evaluation 选层或选方向。",
        "",
        "## 原生任务检查",
        "",
        "| 数据 | 变体 | 场景 | 绝对正确率 | high-low 正向对比例 | 平均差 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in capability.to_dict("records"):
        lines.append(
            f"| {r['split']} | {r['variant']} | {r['n_scenarios']} | "
            f"{r['accuracy']:.3f} | {r['positive_pair_fraction']:.3f} | "
            f"{r['high_minus_low_gap']:.4f} |"
        )
    lines += [
        "",
        "R_fact 的两极材料都应答 Yes，其 high-low 差不是人格效应。G 的绝对阈值会受",
        "候选句先验／长度影响；主检查是同场景 high-low 差，另保存每 token 分数。",
        "R_trait 的正确率仅指与操纵方向一致，不是真人人格真值的预测准确率。",
        "",
        "## 交叉干预",
        "",
        "slope=(score(+v)-score(-v))/2，按场景聚合；CI 是探索性 pointwise bootstrap。",
        "表内效应是各自任务的对数似然差，不能将 G/R 数值当成同量尺后混合平均。",
        "",
        "| 来源→任务 | 层 | signal slope | signal-random | 配对95% CI |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for r in matrix.to_dict("records"):
        lo, hi = r["paired_ci_low"], r["paired_ci_high"]
        interval = (
            f"[{lo:.4f}, {hi:.4f}]" if lo is not None and hi is not None else "NA"
        )
        lines.append(
            f"| {r['source']}→{r['variant']} | {r['site']} | {r['signal_slope']:.4f} | "
            f"{r['signal_minus_random']:.4f} | {interval} |"
        )
    lines += [
        "",
        "## 解释顺序",
        "",
        "1. G→G 与 R→R 是各自正控制；跨任务非零还不足以证明共享心理概念。",
        "2. G→R_negative 应在交换 Yes/No 的语义计分后保留同向；否则优先考虑答案偏置。",
        "   R_trait 与 R_trait_negative 另作同样比较，不能以事实控制通过替代人格归因控制。",
        "3. R_owner 检查行为归属，R_fact 检查无关事实输出漂移；随机方向是等范数控制。",
        "4. G_behavior 检查只有 trait 词才能触发的表面关联。每个变体单独报告，不挑最好行。",
        "5. 最终裁决需综合这些控制；该程序不自动放行真人群体或心理标签使用。",
        "",
        "限制：固定答案候选、少量场景、通用句式、单一模型和最后提示位置；",
        "不是完整电路定位、自由生成复现、统计显著的人格能力或 GRAIN 群体优势。",
        "GRAIN 作者加权原语已实现，但本面板不伪造作者数据来声称已经验证该优势。",
        "",
        f"总 forward={summary['calls']}，秒数={summary['seconds']:.1f}；模型与源码见 manifest。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    """Run fresh immutable materials offline, with per-case checkpoints and limits."""
    config = json.loads(args.config.read_text())
    validate_config(config)
    rows = build_material(smoke=args.smoke)
    if args.smoke:
        config |= {"sites": ["block_16"], "random_replicates": 1}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError(
            "Fresh output directory required; previous results are immutable"
        )
    config["smoke"] = args.smoke
    config["prepare_only"] = args.prepare_only
    selected = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/reciprocity.py",
        ROOT / "grain_research/vector_mechanisms/trace.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "grain_research/vector_mechanisms/analysis.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "tests/test_grain_persona_reciprocity.py",
        ROOT / "docs/GRAIN_PERSONA_RECIPROCITY_P1_PROTOCOL.md",
        args.config.resolve(),
    ]
    manifest = {
        "config": config,
        "material_identity": material_identity(rows),
        "model_revision": config["revision"],
        "code": {str(p.relative_to(ROOT)): file_hash(p) for p in selected},
        "evidence_scope": "synthetic mechanism localization; not personality validity",
        "condition_matrix": "fixed, all scenarios and both polarities; no success filtering",
        "private_data_read": False,
        "model_training": False,
        "direction_estimation": "development mean paired activation differences",
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    write_json(out / "materials.json", {"system": SYSTEM, "rows": rows})
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in selected:
            z.write(path, path.relative_to(ROOT))
    if args.prepare_only:
        write_json(
            out / "run_status.json", {"status": "PREPARED_ONLY", "cases": len(rows)}
        )
        print(f"PREPARED cases={len(rows)} identity={manifest['identity']}", flush=True)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start, observer = time.monotonic(), None
    baselines, effects, checks = [], [], []
    try:
        write_json(
            out / "run_status.json",
            {"status": "PREFLIGHT", "identity": manifest["identity"]},
        )
        expected = EXPECTED_METADATA | {"model.safetensors": config["model_sha256"]}
        actual = {name: file_hash(args.model_dir / name) for name in expected}
        if actual != expected:
            raise ValueError("Cached model/tokenizer differs from pinned original")
        if config["device"] == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable; no automatic remote fallback")
        if config["device"] == "cuda" and torch.cuda.mem_get_info()[0] < 8_000_000_000:
            raise ValueError("CUDA occupied; no displacement of other jobs")
        torch.manual_seed(config["seed"])
        torch.set_num_threads(config["threads"])
        observer = DecoderObserver.load(
            str(args.model_dir.resolve()),
            device=config["device"],
            dtype=config["dtype"],
            max_tokens=config["max_tokens"],
            local_files_only=True,
        )
        if not set(config["sites"]) <= set(observer.sites):
            raise ValueError("Requested site unavailable in cached observer")
        observer.sites = {s: observer.sites[s] for s in config["sites"]}
        write_json(out / "model_manifest.json", observer.manifest() | {"files": actual})

        def budget(cost=1):
            """Check before a complete score calculation, not after exceeding calls."""
            if (
                observer.calls + cost > config["maximum_calls"]
                or time.monotonic() - start > config["maximum_seconds"]
            ):
                raise BudgetExceeded("Call or wall-time budget exhausted")
            if (
                config["device"] == "mps"
                and torch.mps.current_allocated_memory()
                > config["max_mps_allocated_gib"] * 1024**3
            ):
                raise BudgetExceeded("MPS allocated-memory guard reached")

        states, tokens, baseline_lookup = {}, {}, {}
        cache = out / "activation_cache"
        cache.mkdir()

        def score(row, ids, patch=None, logits=None):
            """Use one shared forward for single-token answers, full sequence otherwise."""
            answers = [row["high_answer"], row["low_answer"]]
            answer_ids = [observer.candidate_ids(ids, a) for a in answers]
            if all(len(x) == 1 for x in answer_ids):
                if logits is None:
                    budget()
                    _, logits = observer.capture(ids, patch=patch)
                margin = float(logits[answer_ids[0][0]] - logits[answer_ids[1][0]])
                return margin, margin
            budget(2)
            scores = [observer.score(ids, a, patch=patch) for a in answers]
            return scores[0] - scores[1], scores[0] / len(answer_ids[0]) - scores[
                1
            ] / len(answer_ids[1])

        for i, row in enumerate(rows):
            ids = observer.prompt_ids(SYSTEM, row["user"])
            budget()
            captured, logits = observer.capture(ids)
            margin, norm = score(row, ids, logits=logits)
            if not np.isfinite([margin, norm]).all() or not all(
                np.isfinite(v).all() for v in captured.values()
            ):
                raise ValueError("Non-finite baseline")
            case = row["case_id"]
            states[case], tokens[case] = captured, ids
            np.savez_compressed(cache / f"{case}.npz", **captured)
            record = {
                k: row[k]
                for k in (
                    "case_id",
                    "group_id",
                    "split",
                    "task",
                    "variant",
                    "polarity",
                    "expected_sign",
                )
            } | {
                "margin": margin,
                "mean_token_margin": norm,
                "n_prompt_tokens": len(ids),
                "prompt_token_sha256": material_hash(ids),
                "high_answer_tokens": len(
                    observer.candidate_ids(ids, row["high_answer"])
                ),
                "low_answer_tokens": len(
                    observer.candidate_ids(ids, row["low_answer"])
                ),
            }
            baselines.append(record)
            baseline_lookup[case] = margin
            if i < 2:
                for site in config["sites"]:
                    budget(2)
                    _, sham = observer.capture(
                        ids, patch=Patch(site, len(ids) - 1, captured[site][0])
                    )
                    _, restored = observer.capture(ids)
                    error = max(
                        float(np.max(abs(sham - logits))),
                        float(np.max(abs(restored - logits))),
                    )
                    checks.append(
                        {"case_id": case, "site": site, "max_logit_error": error}
                    )
                    if error > config["replay_tolerance"]:
                        raise ValueError("Self-patch/restore instrument failed")
            if (i + 1) % 12 == 0:
                export_tables(out, baselines, effects)
                print(
                    f"BASELINE {i + 1}/{len(rows)} calls={observer.calls}", flush=True
                )
        directions = fit_directions(rows, states, config["sites"])
        np.savez_compressed(out / "development_directions.npz", **directions)
        write_json(out / "instrument_checks.json", checks)
        export_tables(out, baselines, effects)
        capability, _ = analysis_tables(pd.DataFrame(baselines), pd.DataFrame())
        capability.to_csv(out / "capability_summary.csv", index=False)
        randoms = {
            (source, site, replica): random_direction(
                directions[f"{source}__{site}"],
                f"{config['seed']}:{source}:{site}:{replica}",
            )
            for source in ("G", "R")
            for site in config["sites"]
            for replica in range(config["random_replicates"])
        }
        np.savez_compressed(
            out / "random_controls.npz",
            **{"__".join(map(str, k)): v for k, v in randoms.items()},
        )
        targets = [r for r in rows if r["split"] == "evaluation"]
        for i, row in enumerate(targets):
            case, ids = row["case_id"], tokens[row["case_id"]]
            for site in config["sites"]:
                current = states[case][site][0]
                for source in ("G", "R"):
                    direction = directions[f"{source}__{site}"]
                    for kind, replica, dose in arm_schedule(
                        config["doses"], config["random_replicates"]
                    ):
                        delta = dose * (
                            direction
                            if kind == "signal"
                            else randoms[source, site, replica]
                        )
                        patch = Patch(site, len(ids) - 1, current + delta)
                        margin, norm = score(row, ids, patch=patch)
                        if not np.isfinite([margin, norm]).all():
                            raise ValueError("Non-finite intervention")
                        effects.append(
                            {
                                k: row[k]
                                for k in (
                                    "case_id",
                                    "group_id",
                                    "variant",
                                    "task",
                                    "polarity",
                                )
                            }
                            | {
                                "source": source,
                                "site": site,
                                "kind": kind,
                                "replica": replica,
                                "dose": dose,
                                "margin": margin,
                                "mean_token_margin": norm,
                                "delta": margin - baseline_lookup[case],
                                "effect_norm": float(np.linalg.norm(delta)),
                                "relative_to_state_norm": float(
                                    np.linalg.norm(delta) / np.linalg.norm(current)
                                ),
                            }
                        )
            export_tables(out, baselines, effects)
            progress = {
                "status": "RUNNING",
                "completed_cases": i + 1,
                "total_cases": len(targets),
                "calls": observer.calls,
                "seconds": time.monotonic() - start,
            }
            write_json(out / "run_status.json", progress)
            print(
                f"INTERVENTION {i + 1}/{len(targets)} calls={observer.calls} seconds={progress['seconds']:.1f}",
                flush=True,
            )
        for row in (targets[0], targets[-1]):
            margin, _ = score(row, tokens[row["case_id"]])
            error = abs(margin - baseline_lookup[row["case_id"]])
            checks.append(
                {
                    "case_id": row["case_id"],
                    "site": "post_panel_restore",
                    "max_logit_error": error,
                }
            )
            if error > config["replay_tolerance"]:
                raise ValueError("Final restoration failed")
        write_json(out / "instrument_checks.json", checks)
        capability, matrix = analysis_tables(
            pd.DataFrame(baselines), pd.DataFrame(effects)
        )
        matrix.to_csv(out / "reciprocity_summary.csv", index=False)
        summary = {
            "status": "COMPLETE",
            "smoke": args.smoke,
            "identity": manifest["identity"],
            "cases": len(rows),
            "intervention_rows": len(effects),
            "calls": observer.calls,
            "seconds": time.monotonic() - start,
            "instrument_max_error": max(r["max_logit_error"] for r in checks),
            "no_personality_validity_claim": True,
            "group_pathway_experiment": "NOT_RUN",
        }
        write_json(out / "run_status.json", summary)
        report(out, summary, capability, matrix)
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        export_tables(out, baselines, effects)
        write_json(
            out / "run_status.json",
            {
                "status": "PARTIAL_BUDGET"
                if isinstance(exc, BudgetExceeded)
                else "FAILED",
                "error": str(exc),
                "calls": observer.calls if observer else 0,
                "seconds": time.monotonic() - start,
                "complete_intervention_rows": len(effects),
            },
        )
        (out / "error_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    """Use only an explicitly supplied local model; never download or call an API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/grain_persona_reciprocity_v1.json",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "results/grain_vector_mechanisms/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
