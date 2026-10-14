#!/usr/bin/env python3
"""Test source-frozen within-author condition borrowing on existing PANDORA units."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.conditional_reference import completion_envelope
from grain_research.conditional_response import (
    complete_functional,
    condition_kernel,
    evaluate_hidden_cells,
    fit_response_reference,
    observed_cells,
)
from scripts.prepare_grain_predecessor_reference import BASE, DEFAULT, digest, write
from scripts.run_grain_conditional_reference import export_reference

PRIOR = BASE / "all_plan_closure_v1/conditional_reference_r2"
GEOMETRY = BASE / "all_plan_closure_v1/local_bilinear_r2/pandora"
PLAN = ROOT / "docs/GRAIN_CONDITIONAL_RESPONSE_PLAN_V1.md"
REPORT = ROOT / "reports/grain_measurement_foundation/CONDITIONAL_RESPONSE_REPORT.md"
SEED = 913


def load_material():
    """Verify reference addresses and source roles without accessing human targets."""
    material = json.loads((DEFAULT / "private_material.json").read_text())
    bundles = material["bundles"]
    ids = [b["author_id"] for b in bundles]
    if len(set(ids)) != 624 or any(b["source_group"] != "big5_only" for b in bundles):
        raise ValueError("Expected the unchanged 624 Big5-only source authors")
    names = json.loads((PRIOR / "pandora/private_authors.json").read_text())
    if names["ids"] != ids:
        raise ValueError("Author order changed")
    with np.load(PRIOR / "pandora/private_units.npz", allow_pickle=False) as z:
        c, x, a, source, held = (
            z[k] for k in ("conditions", "scores", "authors", "source", "held")
        )
    with np.load(GEOMETRY / "private_units.npz", allow_pickle=False) as z:
        g = np.asarray(z["g"], float)
        np.testing.assert_array_equal(a, z["authors"])
    metadata = json.loads((GEOMETRY / "material.json").read_text())
    for name, sha in metadata["files"].items():
        if digest(GEOMETRY / name) != sha:
            raise ValueError("Changed frozen GRAIN unit cache")
    np.testing.assert_array_equal(
        c, [v["subreddit"] for b in bundles for view in b["views"] for v in view]
    )
    folds = np.array([b["official_folds"]["big5"] for b in bundles])
    np.testing.assert_array_equal(source, np.flatnonzero(folds != 0))
    np.testing.assert_array_equal(held, np.flatnonzero(folds == 0))
    np.testing.assert_array_equal(np.bincount(a), np.repeat(36, 624))
    direction = np.load(PRIOR / "private_directions.npz", allow_pickle=False)
    contract = json.loads((PRIOR / "contract.json").read_text())
    inputs = [
        DEFAULT / "private_material.json",
        DEFAULT / "material_manifest.json",
        PRIOR / "contract.json",
        PRIOR / "audit.json",
        PRIOR / "pandora/private_units.npz",
        PRIOR / "pandora/private_authors.json",
        GEOMETRY / "private_units.npz",
        PRIOR / "private_directions.npz",
    ]
    if json.loads((PRIOR / "audit.json").read_text()).get("status") != "PASS":
        raise ValueError("Inherited conditional reference requires a passing audit")
    return (
        c,
        x,
        g,
        a,
        folds,
        direction["direction"],
        direction["targets"],
        contract["facets"],
        {str(p): digest(p) for p in inputs},
    )


def author_metrics(rows, actual, prediction, noise):
    """Average facets and cells within author before aggregating across people."""
    frame = pd.DataFrame(rows)
    delta = prediction - actual
    frame["mse"] = (delta**2).mean(1)
    frame["nmse"] = (delta**2 / noise).mean(1)
    frame["mae"] = abs(delta).mean(1)
    result = frame.groupby("author_index")[["mse", "nmse", "mae"]].mean()
    result["target_cells"] = frame.groupby("author_index").size()
    result["has_personal_support"] = (
        frame.groupby("author_index").other_known_cells.max() > 0
    )
    return result.reset_index()


def evaluate(out, name, c, x, a, panel, ref, kernel, facets):
    """Save complete private predictions and public aggregate reference losses."""
    rows, actual, prior, personal, model_sd = evaluate_hidden_cells(
        c, x, a, panel, ref, kernel
    )
    prediction = prior + personal
    np.savez_compressed(
        out / f"private_{name}_predictions.npz",
        actual=actual,
        prior=prior,
        personal=personal,
        prediction=prediction,
        model_sd=model_sd,
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f"private_{name}_cells.csv", index=False)
    am = author_metrics(rows, actual, prediction, ref["noise"])
    am.to_csv(out / f"private_{name}_authors.csv", index=False)
    by_trait = []
    for j, facet in enumerate(facets):
        d = prediction[:, j] - actual[:, j]
        table = pd.DataFrame(
            {
                "author": frame.author_index,
                "mse": d**2,
                "mae": abs(d),
                "nmse": d**2 / ref["noise"][j],
            }
        )
        means = table.groupby("author").mean().mean()
        by_trait.append({"arm": name, "facet": facet, **means.to_dict()})
    pd.DataFrame(by_trait).to_csv(out / f"{name}_by_facet.csv", index=False)
    supported = am.loc[am.has_personal_support]
    return {
        "arm": name,
        "authors": len(am),
        "target_cells": len(rows),
        "supported_authors": len(supported),
        "prior_only_target_cells": int(frame.other_known_cells.eq(0).sum()),
        **{k: float(am[k].mean()) for k in ("nmse", "mse", "mae")},
        "supported_nmse": float(supported.nmse.mean()) if len(supported) else None,
    }, am


def paired(left, right, name, n=2000):
    """Paired author bootstrap; never resample dependent cells as people."""
    np.testing.assert_array_equal(left.author_index, right.author_index)
    d = left.nmse.to_numpy() - right.nmse.to_numpy()
    draws = np.random.default_rng(SEED).integers(0, len(d), (n, len(d)))
    interval = np.quantile(d[draws].mean(1), [0.025, 0.975])
    return {
        "contrast": name,
        "authors": len(d),
        "delta_nmse": float(d.mean()),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "bootstrap_draws": n,
    }


def completion_panel(out, name, c, x, a, panel, ref, kernel, d, contrasts):
    """Export observed and borrowed contributions under both previously fixed targets."""
    summary = []
    for target, weight in (("full", "weights"), ("common5", "common5_weights")):
        result, support = [], []
        for uid in panel:
            cells = observed_cells(c[a == uid], x[a == uid], ref)
            packet = complete_functional(
                ref,
                kernel,
                cells["indices"],
                cells["means"],
                cells["counts"],
                ref[weight],
            )
            lower, upper, _, _ = completion_envelope(
                packet["observed"], packet["missing_mass"], d, contrasts
            )
            projected = packet["model_completed"] @ contrasts
            support.append(
                {
                    "author_index": int(uid),
                    "coverage": packet["coverage"],
                    "missing_mass": packet["missing_mass"],
                    "known_cells": len(cells["indices"]),
                    "unknown_units": cells["unknown_units"],
                    "empirical_complete": packet["empirical_point"] is not None,
                    "envelope_violations": int(
                        np.sum(
                            (projected < lower - 1e-12) | (projected > upper + 1e-12)
                        )
                    ),
                }
            )
            result.append(packet)
        keys = [
            "observed",
            "missing_source_prior",
            "missing_personal_adjustment",
            "model_completed",
            "model_latent_sd",
        ]
        arrays = {k: np.vstack([r[k] for r in result]) for k in keys}
        np.savez_compressed(out / f"private_{name}_{target}_completion.npz", **arrays)
        sf = pd.DataFrame(support)
        sf.to_csv(out / f"private_{name}_{target}_support.csv", index=False)
        summary.append(
            {
                "arm": name,
                "target": target,
                "authors": len(panel),
                "median_coverage": float(sf.coverage.median()),
                "empirical_complete": int(sf.empirical_complete.sum()),
                "envelope_violations": int(sf.envelope_violations.sum()),
                "mean_abs_personal_adjustment": float(
                    abs(arrays["missing_personal_adjustment"]).mean()
                ),
                "mean_abs_source_prior": float(
                    abs(arrays["missing_source_prior"]).mean()
                ),
                "mean_latent_model_sd": float(arrays["model_latent_sd"].mean()),
            }
        )
    return summary


def run(out):
    """Select within two fixed model families on source authors and evaluate once."""
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    c, x, g, a, folds, d, contrasts, facets, inputs = load_material()
    train, valid, held, source = [
        np.flatnonzero(mask)
        for mask in (folds >= 2, folds == 1, folds == 0, folds != 0)
    ]
    write(
        out / "contract.json",
        {
            "protocol": "GRAIN_CONDITIONAL_RESPONSE_V1",
            "plan_sha256": digest(PLAN),
            "runner_sha256": digest(Path(__file__)),
            "module_sha256": digest(ROOT / "grain_research/conditional_response.py"),
            "inputs": inputs,
            "facets": facets,
            "train_authors": train.tolist(),
            "validation_authors": valid.tolist(),
            "held_authors": held.tolist(),
            "human_target_arrays_read": False,
            "new_neural_calls": 0,
            "held_role": "existing_development_fold_0_not_new_lockbox",
            "selection": "source_validation_author_mean_normalized_squared_error",
            "seed": SEED,
            "interval_scope": "paired empirical author error difference; model SD separately conditional on frozen moments",
        },
    )
    valout, heldout = out / "source_validation", out / "held"
    valout.mkdir()
    heldout.mkdir()
    ref = fit_response_reference(c, x, g, a, train)
    write(valout / "private_reference.json", export_reference(ref))
    candidates, selected = [], {}
    for family, pairs in (
        ("intercept", [(v, 0) for v in (0.1, 1)]),
        ("grain_gp", [(v, w) for v in (0.1, 1) for w in (0.1, 1)]),
    ):
        family_results = []
        for alpha, beta in pairs:
            name = f"{family}_a{alpha}_b{beta}"
            result, _ = evaluate(
                valout,
                name,
                c,
                x,
                a,
                valid,
                ref,
                condition_kernel(ref, alpha, beta),
                facets,
            )
            result.update(family=family, alpha=alpha, beta=beta)
            candidates.append(result)
            family_results.append(result)
        winner = min(family_results, key=lambda r: (r["nmse"], r["alpha"], r["beta"]))
        selected[family] = {k: winner[k] for k in ("alpha", "beta", "nmse")}
    pd.DataFrame(candidates).to_csv(out / "source_selection.csv", index=False)
    write(out / "selected.json", selected)
    print("Source-only selection saved; beginning fixed held comparison", flush=True)
    ref = fit_response_reference(c, x, g, a, source)
    write(heldout / "private_reference.json", export_reference(ref))
    inherited = json.loads((PRIOR / "pandora/private_reference.json").read_text())
    for key in ("mean", "weights", "common5_weights"):
        np.testing.assert_array_equal(ref[key], inherited[key])
    arms = {
        "source_prior": (0, 0),
        **{k: (v["alpha"], v["beta"]) for k, v in selected.items()},
        "shuffled_geometry": (
            selected["grain_gp"]["alpha"],
            selected["grain_gp"]["beta"],
        ),
    }
    metric, authors, completions = [], {}, []
    permutation = np.random.default_rng(SEED).permutation(len(ref["keys"]))
    np.save(out / "private_descriptor_permutation.npy", permutation)
    for name, (alpha, beta) in arms.items():
        kernel = condition_kernel(
            ref, alpha, beta, permutation if name == "shuffled_geometry" else None
        )
        row, am = evaluate(heldout, name, c, x, a, held, ref, kernel, facets)
        metric.append(row)
        authors[name] = am
        completions.extend(
            completion_panel(heldout, name, c, x, a, held, ref, kernel, d, contrasts)
        )
        print(
            f"{name}: authors={row['authors']}, hidden_cells={row['target_cells']}, NMSE={row['nmse']:.6f}",
            flush=True,
        )
    comparisons = [
        paired(authors[left], authors[right], f"{left}_minus_{right}")
        for left, right in (
            ("intercept", "source_prior"),
            ("grain_gp", "intercept"),
            ("grain_gp", "source_prior"),
            ("grain_gp", "shuffled_geometry"),
        )
    ]
    pd.DataFrame(metric).to_csv(out / "held_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_author_intervals.csv", index=False)
    pd.DataFrame(completions).to_csv(out / "completion_summary.csv", index=False)
    support_rows = []
    for uid in held:
        cells = observed_cells(c[a == uid], x[a == uid], ref)
        support_rows.append(
            {
                "author_index": int(uid),
                "all_cells": cells["all_cells"],
                "known_cells": len(cells["indices"]),
                "unknown_cells": cells["unknown_cells"],
                "unknown_units": cells["unknown_units"],
            }
        )
    pd.DataFrame(support_rows).to_csv(out / "private_held_support.csv", index=False)
    gm = next(r for r in comparisons if r["contrast"] == "grain_gp_minus_intercept")
    sh = next(
        r for r in comparisons if r["contrast"] == "grain_gp_minus_shuffled_geometry"
    )
    promote_geometry = gm["ci_high"] < 0 and sh["ci_high"] < 0
    summary = {
        "status": "COMPLETE",
        "source_authors": len(source),
        "train_authors": len(train),
        "validation_authors": len(valid),
        "held_authors": len(held),
        "conditions": len(ref["keys"]),
        "noise_dof": ref["noise_dof"],
        "bandwidth": ref["bandwidth"],
        "selected": selected,
        "metrics": metric,
        "paired": comparisons,
        "completion": completions,
        "promote_condition_geometry": promote_geometry,
        "essays_boundary": "one_recorded_condition_cannot_separate_author_intercept_and_condition_field",
        "wall_seconds": time.monotonic() - start,
    }
    write(out / "summary.json", summary)
    lines = [
        "# 条件响应模型构造报告",
        "",
        "2026-09-13。C01/C12/C15/C17 续批；主 goal 继续 ACTIVE。",
        "",
        "## 本轮完成",
        "",
        "用624名作者的22,464条原评论检验整组条件隐藏；不读取人格量表标签。",
        f"来源训练/验证/最终来源/开发留出={len(train)}/{len(valid)}/{len(source)}/{len(held)}。",
        f"固定来源条件{len(ref['keys'])}个；30维是IPIP-keyed语义响应坐标，不是直接问卷得分。",
        "",
        "| 方法 | 作者数 | 隐藏条件数 | NMSE | MSE | MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metric:
        lines.append(
            f"| {r['arm']} | {r['authors']} | {r['target_cells']} | {r['nmse']:.6f} | {r['mse']:.8f} | {r['mae']:.6f} |"
        )
    lines += [
        "",
        "NMSE以来源作者条件内单位残差方差逐维归一化，先作者内再作者间平均。",
        "不是Pearson r/AUC；数值越低越好。整组隐藏后目标评论不进入该作者的条件输入。",
        "",
        "| 配对差（负数更好） | delta NMSE | 作者bootstrap 95%区间 |",
        "|---|---:|---:|",
    ]
    for r in comparisons:
        lines.append(
            f"| {r['contrast']} | {r['delta_nmse']:.6f} | [{r['ci_low']:.6f}, {r['ci_high']:.6f}] |"
        )
    lines += [
        "",
        "## 公式与计分出口",
        "",
        "标准高斯过程条件分布提供：mu_T + K_TS (K_SS + diag(1/n))^-1 (m_S-mu_S)。",
        "参考 [GPML第2章，式2.23/2.24/2.38/2.40](https://gaussianprocess.org/gpml/chapters/RW2.pdf)。",
        "条件描述符使用来源GRAIN24条件均值，RBF中位距离规则固定；噪声模型是假设而非人格误差定理。",
        "",
        "固定nu输出严格拆分为：已观测贡献 + 缺测条件的来源参照 + 缺测条件的个人支持修正。",
        "模型补全不改变原先缺测经验point=None，也不改变覆盖率。每项保留单人向量/条件/贡献。",
        "model_latent_sd只含给定来源矩及核的潜在缺测函数不确定性；不含来源估计或真实问卷误差。",
        "",
        "| 模型 | 目标 | 覆盖中位 | 完整经验目标 | 平均个人补全幅度 | 范围越界数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in completions:
        lines.append(
            f"| {r['arm']} | {r['target']} | {r['median_coverage']:.4f} | {r['empirical_complete']} | {r['mean_abs_personal_adjustment']:.6f} | {r['envelope_violations']} |"
        )
    lines += [
        "",
        "## 制作规则与下一步",
        "",
        f"当前固定门槛是否支持条件几何升级：{promote_geometry}。",
        "只依据本轮预先声明的GP相对截距及置乱几何比较，不继续搜核/带宽/融合权重。",
        "来源未出现条件保留原观测与unknown标志，不伪造来源条件或默认平均人。",
        "自然社群条件是作者选择与他人构成的观测条件，不能当作随机刺激的因果响应。",
        "Essays只有一个已记录条件，不足以分离截距和跨条件函数；不拆段冒充新条件。",
        "",
        "模型给出可复算的借力方式；未来环境可复用制作规则并重新冻结条件参照。",
        "完整Study B、真实机会集合、路径/心理解释及独立复制仍由总台账继续推进。",
        "",
        "## 可复跑产物",
        "",
        f"私有目录：`{out.relative_to(ROOT)}`；原文/身份/向量均不公开。",
        "主入口：`scripts/run_grain_conditional_response.py --output <new-private-directory>`。",
        "`contract.json`记录输入与代码hash；`selected.json`和`source_selection.csv`记录来源选参。",
        "`held_metrics.csv`、`paired_author_intervals.csv`、`completion_summary.csv`为汇总。",
        "`held/private_reference.json`为冻结模型，`private_*`文件保留所有重算地址和值。",
        f"主运行 {summary['wall_seconds']:.2f} 秒，零神经调用；算术审计另见audit.json。",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    return summary


def main():
    """Run one bounded response-field reference with refusal to overwrite outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        summary = run(args.output.resolve())
    print(
        json.dumps(
            {"status": summary["status"], "wall_seconds": summary["wall_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
