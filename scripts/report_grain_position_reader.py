#!/usr/bin/env python3
"""Publish aggregate position-reader findings after exact numerical replay."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_grain_logic_review import edge_inventory
from scripts.run_grain_position_reader import verified
from scripts.run_grain_real_author_path import BASE, read, sha, write


def publish(tests):
    """Verify frozen constructions and publish no participant-level records."""
    runs = [
        BASE / n
        for n in ("logic_position_reader_v1", "logic_position_reader_v1_replay")
    ]
    checks = {}
    for out in runs:
        verified(out)
        freeze = read(out / "prediction_freeze.json")
        if freeze["manifest_sha256"] != sha(out / "manifest.json"):
            raise ValueError("Construction manifest changed")
        for name, digest in freeze["files"].items():
            if sha(out / name) != digest:
                raise ValueError("Construction file changed")
        if read(out / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Incomplete construction")
        for sub, script in (
            ("analysis", "analyze_grain_position_reader.py"),
            ("mode_analysis", "analyze_grain_position_modes.py"),
        ):
            m = read(out / sub / "manifest.json")
            if m["script_sha256"] != sha(ROOT / "scripts" / script) or m[
                "prediction_freeze_sha256"
            ] != sha(out / "prediction_freeze.json"):
                raise ValueError("Analysis provenance changed")
            if read(out / sub / "summary.json")["status"] != "COMPLETE":
                raise ValueError("Incomplete analysis")
    a, b = runs
    for name, digest in read(a / "prediction_freeze.json")["files"].items():
        checks["construction/" + name] = digest == sha(b / name)
    for sub in ("analysis", "mode_analysis"):
        for name, digest in read(a / sub / "summary.json")["files"].items():
            checks[sub + "/" + name] = (
                digest == sha(a / sub / name) == sha(b / sub / name)
            )
    if len(checks) != 18 or not all(checks.values()):
        raise ValueError("Eighteen numerical artifacts must reproduce exactly")
    dest = ROOT / "reports/grain_vector_mechanisms/position_reader_tables_v1"
    dest.mkdir(exist_ok=True)
    allowed = {
        "reader_comparison.csv",
        "factor_actions.csv",
        "coverage.csv",
        "mode_addition.csv",
    }
    for sub in ("analysis", "mode_analysis"):
        summary = read(a / sub / "summary.json")
        for name in summary["files"]:
            if name not in allowed:
                raise ValueError("Only declared aggregate tables may be published")
            shutil.copyfile(a / sub / name, dest / name)
        write(
            dest / (sub + "_summary.json"),
            {k: v for k, v in summary.items() if k != "files"},
        )
    write(dest / "construction_summary.json", read(a / "construction_summary.json"))
    write(
        dest / "reproduction.json",
        {
            "exact_files": checks,
            "tests_passed": tests,
            "constructions": 2,
            "new_neural_calls": 0,
            "construction_compute_seconds": [
                read(p / "run_status.json")["seconds"] for p in runs
            ],
            "timing_scope": "Numerical construction after loading; not total preparation and analysis time",
        },
    )
    d = pd.read_csv(dest / "reader_comparison.csv")
    labels = {
        "local_svd": "本字段8方向（当前梯度压缩）",
        "shared_svd": "其他3作者8方向（当前系数oracle）",
        "source_position": "源系数+位置（不读当前梯度）",
        "uniform": "本字段平均读法（诊断）",
    }
    rows = []
    for r in d.itertuples():
        if r.method in ("local_svd", "shared_svd") and r.rank != 8:
            continue
        label = labels[r.method]
        if r.method == "source_position":
            label += f" degree={r.rank}"
        rows.append(
            f"| {r.query} | {label} | {r.gradient_error_ratio_mean:.3f} | {r.action_r:.3f} | {r.action_mse_over_zero:.3f} | {r.same_sign}/{r.actions} |"
        )
    modes = pd.read_csv(dest / "mode_addition.csv")
    n = modes[
        (modes["query"] == "neuroticism")
        & (modes.method == "shared_svd")
        & (modes.rank_from == 4)
    ].iloc[0]
    report = """# 共享方向可保留作用，真正缺口是内容相关的局部读取权重

2026-09-10。第二十二轮，NEXT31有限比较完成；整体逻辑goal继续ACTIVE。

**原生GRAIN作用可以用少量隐藏方向近似，且部分方向可从其他作者借用。**
但把每个位置的权重也换成源作者的简单位置函数，作用几乎消失。这把下一
机制定位为“这个位置的内容应被怎样读取”，而非仅看位置序号或继续增加维数。

## 1. 做了什么

复用4名旧作者、T/F、neuroticism、confirmation_seeking三个查询的12个完整
正文梯度字段。每作者4个已冻结GRAIN方向，48个原生切线作用；选中正文长
794–1004 token，隐藏宽1536。查询是模型判断目标，不是新发现的人格因子。

- 自身梯度固定rank1/2/4/8压缩，诊断小方向集合是否足够。
- 每次3源/1留出，源梯度生成共享8方向；留出梯度提供系数，只测方向可借用性。
- 相同3源拟合常数和三次Legendre位置函数；预测时只给长度，不给留出梯度。
- 12共享基底拟合、24位置函数、12自身分解，零新神经调用或人类量表读取。

旧2/2参考角色已记录；本轮4人LOO是新定义的探索性分割，不是新独立人群。
48个作用不是48名被试。原生切线复算最大误差1.03e-14。完整数值构造约2.1秒，
不含缓存加载、准备和分析；两次构造完成，18个数值文件逐字节一致。

## 2. 结果：不能把oracle系数当已实现的预测器

field error是每作者梯度平方误差/真实梯度能量再平均。action error是16个
配对作用的SSE/真实作用平方和，零预测=1。r为描述性作用相关，非人格效度。

| 查询 | 方法 | field error | action r | action error | 同号 |
|---|---|---:|---:|---:|---:|
__TABLE__

自身8方向保留约75%–78%梯度能量，作用r=.952/.998/.990。借用其他作者的
8方向时，仅保留约26%–44%梯度能量，仍有r=.862/.987/.977；相对作用平方
误差却为.314/.319/.100，故高相关不等于数值全部还原。这是方向词汇的可用性证据。

逐因子表也保留：共享rank8的T/F-F03仅r=.499，N-F03的平方误差1.011，
尽管N-F03相关.855。因此pooled高r不能代替每个因子的作者比较；下一步需
同时改善局部系数及其作用，而不是只重现不同因子之间的量级差异。

真正不读留出梯度的三次位置函数只在T/F有部分相关(.498)，其error仍.922；
N/confirmation几乎零改善。位置指选中正文的ordinal midpoint，不是绝对
token距离，也没有给位置系数读取内容或GRAIN。结果支持进入内容条件化，
不是宣称任何位置算法都不可能有效。

## 3. 新解释：能量更准而目标作用更差，来自有符号抵消

18次固定nested-rank增量中，4次梯度误差下降但作用误差上升。N共享rank4→8
是清晰实例：field error .840→.738，而action error __OLD__→__NEW__。

    e_new = e_old + delta
    error_new = error_old + ||delta||²/||y||² + 2<e_old,delta>/||y||²

本例新增作用能量__ADD__，交叉项__CROSS__；18个恒等式最大误差2.22e-16。
因此新增方向不一定是噪声，也不一定是增益；投影优化全字段平方误差，实际
GRAIN读数关心另一个半范数。这不是据留出结果改选rank4，所有固定rank保留。
完整公式与反例见[理论](../../docs/GRAIN_POSITION_READER_THEORY_V1.md)。

## 4. 这怎样连接GRAIN和向量路径

GRAIN给同一材料一个可定位的表达方向D_k，查询梯度J_q给出对此方向的
局部读取作用<J_q,D_k>。现在把J拆成“共享隐藏方向V + 每个token的系数a”，
两系统的接口就不再只是总分相关，而是同位置的有符号乘积及其求和。
当前是全文局部导数，不是attention权重或已验证的有限编辑效应。

下一[固定计划](../../docs/GRAIN_CONTENT_COEFFICIENT_READER_PLAN_V1.md)只做两个
源拟合：内容向量+位置，以及加入同位置GRAIN及其交互；全部使用源作者
梯度作系数目标，留出作者梯度只评价。这样直接测试GRAIN是否能辅助局部
读取，不启动新神经训练、泛化调参或外部量表重命名。

精神分析/交流分析的辅助解释继续沿用[同材料解释协议](../../docs/GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)：
数值测量、模型敏感度、候选心理解释分别保存；关系事件需要实际上下轮，
不能把token系数直接命名为防御机制或ego-state。该解释协议本轮未调用LLM。

## 5. 复现与数学依据

主要入口：`run_grain_position_reader.py`、`analyze_grain_position_reader.py`、
`analyze_grain_position_modes.py`、`report_grain_position_reader.py`。
输入与梯度、基底、系数和逐人结果保留在gitignored results；本目录只发布汇总。
__TESTS__项聚焦测试通过，两个旧SWIG警告。反例测试最初把浮点零写成严格
相等，修正为1e-12容差后通过；没有因此修改实验结果。无全测试套件声明。

随机低秩近似遵循[Halko、Martinsson与Tropp (2011)](https://arxiv.org/abs/0909.4061)
的数值线性代数方向。共享基底+位置系数是矩阵因子化，不是新心理量表的
因子确认。目标作用误差半范数与交叉项来自标准内积代数；本研究的新工作
是把它们接到同材料GRAIN/原生查询作用并测出可用范围，不宣称首创SVD。
"""
    replacements = {
        "__TABLE__": "\n".join(rows),
        "__TESTS__": str(tests),
        "__OLD__": f"{n.old_action_error:.6f}",
        "__NEW__": f"{n.new_action_error:.6f}",
        "__ADD__": f"{n.added_action_energy:.6f}",
        "__CROSS__": f"{n.prior_added_cross:.6f}",
    }
    for key, value in replacements.items():
        report = report.replace(key, value)
    (dest.parent / "POSITION_SPECIFIC_READER_REPORT_V1.md").write_text(report)
    (ROOT / "docs/GRAIN_SYSTEM_LOGIC_EDGES_V1.json").write_text(
        json.dumps(edge_inventory(), indent=2) + "\n"
    )
    print({"status": "PUBLISHED", "exact_numerical_files": len(checks), "tests": tests})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tests", required=True, type=int)
    publish(p.parse_args().tests)
