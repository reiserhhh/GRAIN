#!/usr/bin/env python3
"""Generate the aggregate logic review and machine-readable edge inventory."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grain_logic_levels import BASE, sha, write


def edge_inventory():
    """Extract explicit logical edges, not invented proof status, from the atlas."""
    path = ROOT / "docs/GRAIN_SYSTEM_LOGIC_ATLAS_V1.md"
    edges = []
    for line in path.read_text().splitlines():
        if not re.match(r"\| L\d\d \|", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5:
            raise ValueError("Logic table schema changed")
        edges.append(
            dict(
                zip(
                    (
                        "id",
                        "implication",
                        "type_and_status",
                        "alternative_or_next_link",
                        "evidence_entry",
                    ),
                    cells,
                    strict=True,
                )
            )
        )
    if [r["id"] for r in edges] != [f"L{i:02d}" for i in range(1, 32)]:
        raise ValueError("Missing or duplicate logic edge")
    return {
        "atlas_sha256": sha(path),
        "finite_inventory_not_exhaustiveness_proof": True,
        "edges": edges,
    }


def render(out, *, levels_name="levels_r1", witnesses_name="witnesses_r2"):
    """Publish aggregate-only findings; never expose author-level cached records."""
    levels, witnesses = out / levels_name, out / witnesses_name
    design = json.loads((levels / "manifest.json").read_text())
    if design["bootstrap"] != 1999 or design["permutations"] != 3999:
        raise ValueError(
            "This full report requires the frozen resampling counts, not a smoke packet"
        )
    for folder in (levels, witnesses):
        if json.loads((folder / "status.json").read_text())["status"] != "COMPLETE":
            raise ValueError("Incomplete experiment packet")
    links = pd.read_csv(levels / "level_links.csv")
    cov = pd.read_csv(levels / "covariance_accounting.csv")
    coord = pd.read_csv(levels / "coordinate_accounting.csv")
    native = pd.read_csv(
        witnesses / "native_endpoint_accounting.csv", keep_default_na=False
    )
    if len(links) != 1530 or len(cov) != 306 or len(coord) != 162:
        raise ValueError("Unexpected finite analysis coverage")
    focus = links[
        links["query"].eq("TF_cont")
        & links.feature.isin(["whole:F02", "level::whole:F02", "disp::whole:F02"])
    ]
    display = focus.pivot(index="feature", columns="level", values="r")[
        ["between", "within", "transition", "criterion"]
    ]
    counts = {level: int(g.q.lt(0.05).sum()) for level, g in links.groupby("level")}
    pivot = links.pivot(index=["query", "feature"], columns="level", values="r")
    opposite = pivot[
        (pivot.between * pivot.within < 0)
        & pivot.between.abs().gt(0.1)
        & pivot.within.abs().gt(0.1)
    ]
    tf = coord[coord["query"].eq("TF_cont") & coord.level.eq("between")]
    rotation = tf[tf["transform"].str.startswith("rotation_")]
    disp = focus[
        focus.feature.eq("disp::whole:F02") & focus.level.eq("criterion")
    ].iloc[0]
    report = f"""# GRAIN 全系统逻辑审查：首轮实证与理论修正

2026-09-10。首轮 LOG00–05 已交付；持续研究 goal 仍 ACTIVE。
不是所有逻辑已完成，不修改任何旧 lockbox/发布评分器。

## 核心发现

**同一因子可以包含不同作用的信息；但要在相同统计层级比较后才能这么说。**
F02 的作者平均水平明显接到模型 T/F 判断，而文内离散度更偏向当前真实 T/F
标签的候选连接。重新比较没有抹掉上一轮发现，而是让这一解释有了正确依据。
我们还把原生向量干预区分为共同查询位移和作者异质响应，并给坐标变化后的
作用定义了明确搬运规则。

## 1. 全系统已整理什么

[总逻辑图谱](../../docs/GRAIN_SYSTEM_LOGIC_ATLAS_V1.md) 将31条关键边放在
三套系统中：作者表达与群体测量、LLM查询推论、心理解释/效标。
覆盖 MCD、rind/choice、V6动态、V7观察平面、V8微观至宏观、4W、当前固定
功能与便携导航，而不只审计上一轮的回归公式。
[研究计划](../../docs/GRAIN_LOGIC_RESEARCH_PLAN_V1.md) 记录问题、替代机制、
文献用途和执行顺序。[理论研讨及公式修正](../../docs/GRAIN_LOGIC_SEMINAR_20260910.md)
记录 sol/xhigh 的数学建议及主 agent 的独立复核。

## 2. LOG01：同级比较，而非用变化否定位置

276名 reference-disjoint 作者，213 Big5-only / 63 MBTI-only，三视图；12个
冻结因子 + 文内22个统计量 × 九个查询。完成{len(links)}格关联、1999次
作者bootstrap、3999次作者整体置换。没有新模型调用或预测调参。

| 特征 | 作者间 r(G均值,f均值) | 作者内偏移 r | 相邻变化 r | 作者间 r(G均值,Y) |
|---|---:|---:|---:|---:|
"""
    for feature, row in display.iterrows():
        report += f"| {feature} | {row['between']:.4f} | {row['within']:.4f} | {row['transition']:.4f} | {row['criterion']:.4f} |\n"
    report += f"""
这里 f 是固定查询函数，Y 是一次自评标签，不是实际行为或原生概率。
文内离散度与Y的r={disp.r:.4f}，作者bootstrap95%CI
[{disp.lower:.4f},{disp.upper:.4f}]，全306格效标家族q={disp.q:.4f}，保留为
探索性连接。其与f的同级相关只有+.073；这才是“当前线性读出与效标关联的
方面部分分离”的适当比较。低线性相关不等于表征完全没有该信息。

BH<.05格数：{json.dumps(counts, ensure_ascii=False)}。
每一level是单独306格家族，不同于旧108/162格的q；新家族不会反向改写旧检验。
有{len(opposite)}格作者间/作者内相关异号且各自|r|>.1，这只是描述性计数。
例如native:F04/T-F是-.273与+.187。不存在必须跨级同号的理论要求。

精确经验协方差采用相同权重及ddof=0，最大闭合误差
{cov.closure_error.abs().max():.2e}。F02/T-F的总相关-.472，可分为共同分母下的
作者间贡献-.353和作者内贡献-.119；两项不是分别标准化的层级相关，也不是
人格方差比例。一次标签Y的作者内协方差恒为0是设计结果，不能据此删掉动态。
这一拆分对应 [Curran & Bauer 2011](https://pubmed.ncbi.nlm.nih.gov/19575624/)
讨论的不同层级效应问题。

## 3. LOG02：可以稳定追踪作用，不必固定宇宙中唯一一条轴

真实缓存上的162次坐标比较完成。T/F作者层面的单轴最大|r|从原.548，在八个
正交坐标中变为{rotation.max_coordinate_abs_r_after.min():.3f}–{rotation.max_coordinate_abs_r_after.max():.3f}，
但共同12维子空间的样本内投影R²始终{tf.in_sample_projection_r2.iloc[0]:.6f}。
这不是新的外推精度，后一个数是样本内几何记账。

所有查询/层级的固定函数搬运最大误差{coord.transported_function_max_error.max():.2e}，
协变惩罚重拟合误差{coord.covariant_refit_max_error.max():.2e}。
非正交缩放后如果仍直接套相同的isotropic Ridge惩罚，结果确实会变；那是
换了统计模型而非发现新的心理结构。CKA等表示比较的适用不变性也需明确，
参见 [Kornblith et al. 2019](https://proceedings.mlr.press/v97/kornblith19a.html)。

因此下一代接口可研究**坐标协变的作用/子空间及其分布统计**，不必先要求
每个F编号成为全局唯一可命名轴。此次只验证搬运规则，不宣称低秩机制已发现。

## 4. LOG05：方向起作用，和方向解释个体差异，是两条边

仅重用12名旧干预作者的real/null正负端点，0次新神经计算。

| 方向/输出坐标 | 平均奇项 | 奇项作者SD | 共同能量占比 | 平均绝对偶项 |
|---|---:|---:|---:|---:|
"""
    for row in native.itertuples():
        report += f"| {row.arm}/{row.coordinate} | {row.mean_odd:.6f} | {row.sd_odd:.6f} | {row.common_odd_energy_fraction:.3%} | {row.mean_abs_even:.6f} |\n"
    report += """
奇项=(plus-minus)/2，偶项=(plus+minus)/2-native，共同能量=平均奇项²，
异质能量=奇项方差，二者精确相加。真实方向仍有12/12同向作用；约94.1%的
log-odds响应能量是共同平移。这不是94.1%的人格，也不否定既有干预效果。
它意味着当前干预主要测得**共同单位响应**，本身并不检验12人的自然位置差异。
这同样不能被当作“方向没有人格信息”：一个完美的线性测量轴也会对相同位移
作出完全相同响应。第五个有限例子同时给出level/criterion r=1和100%共同响应。
因此保留共享算子，检验个体在该算子下的位置/行程；不强求每个人梯度都不同。
log-odds平均绝对偶项约.000072，远小于奇项.027251；此幅度
下不应把非线性门控当作主要新发现。仅有一档步幅，不能据此证明全局线性。

已知恒定logit平移的合成例子也会产生概率偶项和异质响应，但logit偶项为0。
这是为何需要分开“输出坐标曲率”与“内部条件门控”。读取缓存时另修正了
pandas把字面arm名null解释成NaN的问题，未改动旧结果。

## 5. LOG04：公式接得上，哪一步仍可能缺逻辑

五个有限例子已执行：层级相关可以异号；恒定logit平移可产生概率曲率；
样本均值误差趋零但事件残差不消失；删除查询可读余项会破坏作用计算。
第五个说明共同梯度与有效个体轴完全兼容。例子不是对人体的模拟验证，
只用于定位必要前提。原四例r1保留，当前增加第五例的packet为witnesses_r2。

关键理论升级是保留两项：

    dR = dH - D_q dG
    Delta F = integral (D_q^T grad F)^T dG + integral grad F^T dR.

源拟合D不是已知dH/dG。如果同一G允许不同H或历史，第一项的协向量也不
一定是G上的单值函数。保留第二项后，可以研究其结构并扩充观察对象，而
不以“有残差”宣布失败，也不以“写了余项”宣布所有机制已解释。
[路径归因](https://proceedings.mlr.press/v70/sundararajan17a.html)与
[因果抽象](https://arxiv.org/abs/2106.02997)分别提供归因和干预的标准参照；
两者都不能替我们保证人类构念意义。

## 6. 下一轮顺序与实际开放问题

1. 同级的level/dispersion/路径作用组成局部接口，优先检验自然文本变化中的
   作者特异作用；不再仅凭共同方向12/12同向宣布个体机制。
2. 查询可读余项作为新增GRAIN观察通道，检验它与现有分布统计能否稳定连接。
   先复用现有上下文/函数缓存，不全量调人格预测器。
3. 在目前可观察范围拆开条件选择、文本表达、自评通道；独立实际行为目前
   未观测，保留为需要新增数据的连接，而不是让模型猜一个“行为真值”。
4. 明确B0零Transformer、B4四层前端、f上层P/Q三种构造/执行依赖；两条
   数学运行路线都保留。新机制应服务可运行解释，而不是只增加安全规则。

对照 [Funder的判断信息链](https://rap.ucr.edu/FunderPR1995.pdf)，本项目现在
可以把“表达信息有相关”“模型检测/使用”“固定程序保留”“实际效标连接”
逐项变成研究对象。余下任务是发展这些边，不是重新证明所有历史算法有效。

## 7. 重放与产物

```bash
PY='../project persona/.venv/bin/python'
"$PY" -m pytest -q tests/test_grain_logic_levels.py tests/test_grain_dynamic_channels.py
"$PY" scripts/run_grain_logic_levels.py --output results/grain_vector_mechanisms/logic_closure_v1/levels_reproduction
"$PY" scripts/run_grain_logic_witnesses.py --output results/grain_vector_mechanisms/logic_closure_v1/witnesses_reproduction
"$PY" scripts/report_grain_logic_review.py --levels-name levels_reproduction --witnesses-name witnesses_reproduction
```

不带参数生成原始报告时读取levels_r1/witnesses_r2；上述命令读取新重放目录。
源路径/代码/计划hash、全部关联、协方差与坐标核验表位于
`results/grain_vector_mechanisms/logic_closure_v1/`。输入均为已有私有数字缓存；
没有读取/上传新的真人文本、问卷原题、密钥。公开只存aggregate报告与逻辑目录。
旧动态报告的同级解释已加更新注；原数值、原实验族和已提交历史保留。
"""
    target = ROOT / "reports/grain_vector_mechanisms/SYSTEM_LOGIC_REVIEW_V1.md"
    target.write_text(report)
    inventory = edge_inventory()
    write(ROOT / "docs/GRAIN_SYSTEM_LOGIC_EDGES_V1.json", inventory)
    write(
        out / "review_manifest.json",
        {
            "report_sha256": sha(target),
            "atlas_sha256": inventory["atlas_sha256"],
            "logic_edges": len(inventory["edges"]),
            "level_associations": len(links),
            "coordinate_cases": len(coord),
            "inputs": {
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    levels / "level_links.csv",
                    levels / "covariance_accounting.csv",
                    levels / "coordinate_accounting.csv",
                    witnesses / "native_endpoint_accounting.csv",
                ]
            },
        },
    )
    print("Wrote aggregate logic review and 31-edge inventory")


def main():
    """Generate the current finite review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "logic_closure_v1")
    parser.add_argument("--levels-name", default="levels_r1")
    parser.add_argument("--witnesses-name", default="witnesses_r2")
    args = parser.parse_args()
    render(args.input, levels_name=args.levels_name, witnesses_name=args.witnesses_name)


if __name__ == "__main__":
    main()
