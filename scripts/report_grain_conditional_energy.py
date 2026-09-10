#!/usr/bin/env python3
"""Publish aggregate conditional-moment evidence and exact cache reproduction."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_grain_logic_review import edge_inventory
from scripts.run_grain_conditional_energy import verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def publish():
    """Whitelist four aggregate tables; never export author actions or text states."""
    out, repeated = [
        BASE / n
        for n in ("logic_conditional_energy_v1", "logic_conditional_energy_replay_v1")
    ]
    checks = {}
    for folder in (out, repeated):
        verified(folder)
        manifest = read(folder / "analysis_r2/manifest.json")
        if manifest["analyzer_sha256"] != sha(
            ROOT / "scripts/analyze_grain_conditional_energy.py"
        ) or manifest["prediction_freeze_sha256"] != sha(
            folder / "prediction_freeze.json"
        ):
            raise ValueError("Analysis/program provenance mismatch")
    for name, digest in read(out / "prediction_freeze.json")["files"].items():
        checks["construction/" + name] = (
            digest == sha(out / name) == sha(repeated / name)
        )
    for name, digest in read(out / "analysis_r2/summary.json")["files"].items():
        checks["analysis/" + name] = (
            digest
            == sha(out / "analysis_r2" / name)
            == sha(repeated / "analysis_r2" / name)
        )
        if name != "energy_comparison.csv" and sha(out / "analysis" / name) != digest:
            raise ValueError(
                "Constant-fold display correction changed other statistics"
            )
    old = pd.read_csv(out / "analysis/energy_comparison.csv").drop(
        columns="within_fold_r"
    )
    current = pd.read_csv(out / "analysis_r2/energy_comparison.csv").drop(
        columns="within_fold_r"
    )
    pd.testing.assert_frame_equal(old, current)
    if len(checks) != 9 or not all(checks.values()):
        raise ValueError("Nine numerical files must replay exactly")
    dest = ROOT / "reports/grain_vector_mechanisms/conditional_energy_tables_v1"
    dest.mkdir(exist_ok=True)
    for name in (
        "energy_comparison.csv",
        "energy_paired_mse.csv",
        "moment_geometry.csv",
        "score_link_comparison.csv",
    ):
        shutil.copyfile(out / "analysis_r2" / name, dest / name)
    summary = read(out / "analysis_r2/summary.json")
    write(dest / "summary.json", {k: v for k, v in summary.items() if k != "files"})
    write(dest / "construction_summary.json", read(out / "construction_summary.json"))
    write(
        dest / "reproduction.json",
        {
            "exact_files": checks,
            "constant_fold_correction_only": True,
            "new_neural_calls": 0,
            "human_labels_read": False,
        },
    )
    data = pd.read_csv(dest / "energy_comparison.csv")
    table = []
    for (arm, pair), g in data.groupby(["arm", "pair"]):
        g = g.set_index("statistic")
        table.append(
            f"| {arm} / {pair} | {g.loc['old_vector_energy', 'r']:.3f} | {g.loc['old_vector_energy', 'within_fold_r']:.3f} | {g.loc['old_vector_energy', 'mse_over_zero']:.3f} | {g.loc['energy', 'mse_over_zero']:.3f} | {g.loc['pair_energy_mean', 'mse_over_zero']:.3f} |"
        )
    report = """# 平均方向之外的强度：二阶矩能算，但不是末端评分的主要缺口

2026-09-10，第十八轮，NEXT25有限实验完成。**平均方向与作用强度是不同的
数学对象；现有相邻文本读法已经保留一些强度差异，新增分布读法主要恢复
群体量级，未提供额外个体读取优势。** 下一步已转向完整条件输入，不继续
第二矩或softmax校准网格。

## 1. 相同材料与明确目标

24名旧N源作者、4个作者互斥折、相邻/远隔两种评论对、原bag/position两种
Fourier输入。原投影、源几何、旧预测和训练均值全部精确重放（最大误差0）。
目标是原生模型中心化混合logits Z，不是人的神经质自评。人类标签、原文、
MEPS及lockbox均未读取；0新神经调用、0优化、16个确定性源平滑器构造。

旧向量预测m的强度缺口被精确拆成：

    ||Z||²-||m||² = ||Z-m||² + 2<m,Z-m>。

相邻bag的平均交叉项+1.50e-5，远隔bag为-1.58e-5；相邻position为-1.39e-5。
它确实不总为0。因此不能把“预测幅度变小”全部命名为被消除的噪声。

## 2. 一套权重同时算均值、强度、协方差

用同折之外18名同类文本对的非负核权重，加一个零作用参考原子，得到一个
有限条件分布候选。由同一套权重计算mu、二阶矩M、协方差V及energy：

    energy = ||mu||² + trace(V)。

两股相反的作用可使mu=0而energy>0，数学测试直接复现。数据中的矩恒等式
最大误差9.89e-17；最小协方差特征值-1.1e-20，是浮点量级。其trace(V)
描述这个源混合分布，不是独立识别的个人情绪波动或心理噪声。

下表每行24作者；r为预测强度与原生交互强度的相关，**不是人格量表相关**。
MSE/零越低越好；最后一列是源内“平方范数的平均”，不是“均值的平方范数”。

| 原输入/文本对 | 旧强度r | 旧强度折内r | 旧强度MSE/零 | 新二阶强度MSE/零 | 源平均强度MSE/零 |
|---|---:|---:|---:|---:|---:|
TABLE

相邻已有读法强度r约.48，折内仍约.44；保留它作为局部读取对象。远隔新
二阶强度比旧预测均值范数的误差低，但不优于源平均强度。配对区间也没有
支持其超过这个平均基线。新强度的折内r相邻约.207、远隔.035/.056，而
pooled r为负，说明折间量级不能直接当成个体方向。全部28行统计与16个
点区间比较保留，不用最佳一行挑方法。

## 3. 多知道一个分布，并不意味着最终评分大变

在实际z_add上比较“先平均logits再评分”和“逐原子评分后平均”。
二者差异RMS相邻约1.1e-6、远隔约5.7e-8，只有原生交互RMS的约.165%/.034%。
Hessian×协方差二阶公式能重建它，剩余高阶RMS约5.2e-10/1.4e-11。
这个数学近似很准确，但它针对的是一个很小的分项。

所以结论不是“只要补二阶矩就能补回人格推论”。新分布读法不替换旧signed
reader；读取方向、强度与最终评分分三本账。上述评分仍借用当前作者
z_add，下一步应实际移除这个端点依赖。

## 4. 对GRAIN理论更有意义的连接

固定模型与完整输入H时，Z=F(H)是确定函数。把H压成X后，不同完整状态
可以共享同一个X，于是Var(Z|X)>0。这部分差异可能来自没有输入的条件，
而不是应当删除的噪声。加入条件在理想条件期望下会降低平均预测风险，
但并不保证每个群体、每次有限样本拟合都改善。

这连接了GRAIN“把情境作为前提”与路径读取：完整接口应明确M（正文均值）、
R（其余携带状态）、L（布局）以及片段配置。NEXT26已冻结：用这些输入构造
z_0和单评论作用，再接已有mixed读法，得到不借用当前作者端点的完整条件
文本对分数。它仍需原块16编码，不会被称为已经完全脱离LLM。

理论与一手文献见[条件矩理论](../../docs/GRAIN_CONDITIONAL_MOMENT_THEORY_V1.md)；
下一执行计划见[完整条件接口](../../docs/GRAIN_COMPLETE_PAIR_INTERFACE_PLAN_V1.md)。

## 5. 复现与状态

46项相关测试（本轮17项），9个构造/分析文件精确重放。旧分析对折内恒定
基线产生约1e-17的相关；修正为数学上应报告的NaN，其余数值完全不变，旧包
仍保留。现有SWIG弃用警告不影响测试；未声称全仓库测试。
公开文件只有聚合表，无逐作者标识、作用向量、正文或模型。

```bash
PY="../project persona/.venv/bin/python"
"$PY" scripts/run_grain_conditional_energy.py --name logic_conditional_energy_user_replay
"$PY" scripts/analyze_grain_conditional_energy.py --output-name logic_conditional_energy_user_replay
"$PY" -m pytest -q tests/test_grain_conditional_energy.py tests/test_grain_conditional_energy_analysis.py tests/test_grain_joint_readout.py tests/test_grain_action_geometry.py tests/test_grain_retained_logit.py
```

计划11e9062，实现d1a69e3；NEXT26计划bf21c94。均本地提交，未push。
NEXT25本次候选已完成；NEXT26尚未执行，持续goal仍ACTIVE。
""".replace("TABLE", "\n".join(table))
    target = ROOT / "reports/grain_vector_mechanisms/CONDITIONAL_ENERGY_REPORT_V1.md"
    target.write_text(report, encoding="utf-8")
    (ROOT / "docs/GRAIN_SYSTEM_LOGIC_EDGES_V1.json").write_text(
        json.dumps(edge_inventory(), indent=2) + "\n", encoding="utf-8"
    )
    print(
        {
            "report": str(target.relative_to(ROOT)),
            "exact_file_replays": len(checks),
            "public_tables": 4,
        }
    )


if __name__ == "__main__":
    publish()
