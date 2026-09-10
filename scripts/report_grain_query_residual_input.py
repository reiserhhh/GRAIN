#!/usr/bin/env python3
"""Publish NEXT35 paired aggregates after replay and provenance verification."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_grain_query_residual_input import verify_predictions
from scripts.run_grain_real_author_path import BASE, read, sha, write


def publish(tests):
    """Retain all paired arms; export aggregates only, never source token assets."""
    runs = [BASE / ("logic_query_residual_input_v1_r2" + s) for s in ("", "_replay")]
    for out in runs:
        verify_predictions(out)
        if read(out / "run_status.json")["status"] != "COMPLETE":
            raise ValueError("Incomplete construction")
        m = read(out / "analysis/manifest.json")
        if m["script_sha256"] != sha(
            ROOT / "scripts/analyze_grain_query_residual_input.py"
        ):
            raise ValueError("Analysis code changed")
        if m["prediction_freeze_sha256"] != sha(out / "prediction_freeze.json"):
            raise ValueError("Analysis prediction binding changed")
        if read(out / "analysis/summary.json")["status"] != "COMPLETE":
            raise ValueError("Incomplete analysis")
    a, b = runs
    checks = {
        "construction/" + p: digest == sha(a / p) == sha(b / p)
        for p, digest in read(a / "prediction_freeze.json")["files"].items()
    }
    checks.update(
        {
            "analysis/" + p: digest
            == sha(a / "analysis" / p)
            == sha(b / "analysis" / p)
            for p, digest in read(a / "analysis/summary.json")["files"].items()
        }
    )
    if len(checks) != 43 or not all(checks.values()):
        raise ValueError("All 43 numerical artifacts must reproduce exactly")
    abandoned = BASE / "logic_query_residual_input_v1"
    relocation = {
        p: digest == sha(abandoned / p)
        for p, digest in read(a / "prediction_freeze.json")["files"].items()
    }
    if not all(relocation.values()):
        raise ValueError("Path relocation altered numerical construction")
    dest = ROOT / "reports/grain_vector_mechanisms/query_residual_input_tables_v1"
    dest.mkdir(exist_ok=True)
    allowed = {
        "reader_comparison.csv",
        "factor_actions.csv",
        "error_account.csv",
        "source_fits.csv",
        "source_singulars.csv",
        "prior_comparators.csv",
    }
    for p in allowed:
        shutil.copyfile(a / "analysis" / p, dest / p)
    write(dest / "construction_summary.json", read(a / "construction_summary.json"))
    summary = read(a / "analysis/summary.json")
    write(
        dest / "analysis_summary.json",
        {k: v for k, v in summary.items() if k != "files"},
    )
    times = [read(out / "run_status.json")["seconds"] for out in runs]
    write(
        dest / "reproduction.json",
        {
            "numerical_files": checks,
            "tests_passed": tests,
            "constructions": 2,
            "construction_seconds_after_input_loading": times,
            "unreviewed_path_collision_numerical_match": relocation,
            "historical_runner_restored_from": "0c940fd",
            "new_neural_calls": 0,
        },
    )
    d = pd.read_csv(dest / "reader_comparison.csv")
    old = pd.read_csv(dest / "prior_comparators.csv")
    table, geometry = [], []
    for q in ("TF_cont", "neuroticism", "confirmation_seeking"):
        rows = pd.concat(
            [old[(old["query"] == q) & (old.arm == "content")], d[d["query"] == q]]
        )
        for r in rows.itertuples():
            table.append(
                f"| {q} | {r.arm} | {r.action_r:.3f} | {r.action_mse_over_zero:.3f} | {r.field_error_mean:.3f} | {r.factor_centered_action_r:.3f} | {r.predicted_action_rms / r.actual_action_rms:.1%} |"
            )
    for r in pd.read_csv(dest / "source_fits.csv").itertuples():
        geometry.append(
            f"| {r.query} | {r.arm} | {r.extra_energy_mean:.2%} | {r.coefficient_error_mean:.3f} | {r.held_outside_fraction_mean:.3%} |"
        )
    report = """# 查询定向的低能量输入方向，改善同材料 GRAIN 作用读取

2026-09-10。第二十四轮 NEXT35 完成；整套系统逻辑研究 goal 继续 ACTIVE。

**找到的机制：通用方差压缩漏掉了一部分可用于查询读取的方向。**
同样多加8个输入方向，查询定向子空间只保留约3%的残差能量，普通残差PCA
保留约7%；前者却在三个查询上都有更好的留出 GRAIN 作用读取。它把用户的
“同一材料可有两套读法”推进为一个源构造、当前目标禁读的数值程序。

## 1. 实验与结果

同4名旧作者、3个查询、每人4个旧 GRAIN 方向；每次3源/1留出。只读旧数值
缓存，不读新文本、心理量表或临床数据。原始4人的2/2角色保留在私有台账；
本次LOO是探索性复用，不把16作用当16名独立作者。

保留原36列内容/位置设计，统一增加8列，alpha=.01不变。方差组用源残差
PCA；查询组用源残差与原系数预测误差的交叉协方差。24个头均只拟合3源。
当前作者运行只读固定模型与隐藏状态/GRAIN输入包，梯度和旧结果读取被拒绝。

| 查询 | 读法 | 作用r | 作用SSE/零 | 完整梯度误差比 | 因子中心化作者r | 预测/原生RMS |
|---|---|---:|---:|---:|---:|---:|
__TABLE__

表中相关是对原模型局部方向导数的复现，**不是对人类T/F或神经质量表的
预测相关**。因子中心化后仍有改善；T/F为.844，N为.539，寻求确认为.596。
方向信息改善同时伴随幅度低估；没有在留出结果上调比例以制造“恢复成功”。
逐因子完整表保留例外：F03的T/F排序未改善，另两查询F03误差仍大于零预测。

## 2. 输入方差不等于读取价值

| 查询 | 补充方向 | 覆盖残差能量 | 源系数误差/能量 | 留出坐标超源单维极值比例 |
|---|---|---:|---:|---:|
__GEOMETRY__

两组均44列、rank8；新方向与原PCA32的最大重叠小于1.2e-16。查询定向的
源误差由原content的.944/.954/.959下降为.802/.798/.838。它不是“原空间
调alpha”，而是纳入了另一种输入信息。留出单维越界比例很低，但这不等于
联合支持集相同；本文不把坐标范围检查当泛化证明。

## 3. 两种“接近”不相同

三个查询的完整梯度误差都比方差组大，沿GRAIN方向的作用误差却都更小。
这在数学上并不矛盾：完整场使用Frobenius范数，GRAIN用特定方向构成的
测量半范数。沿当前方向不可见的误差可能很大，同时可见部分更有用。
因此先保留可用方向读取，而不是因整张梯度图更不像就丢弃本轮发现。
有符号误差账保持精确：N查询的作用误差为.319+2.134-1.579=.874。
不能把相互抵消的部分称为独立心理信号，也不称其余部分为无用噪声。

这一设计借鉴[多输出降秩回归](https://pmc.ncbi.nlm.nih.gov/articles/PMC3444519/)
将输入压缩和响应结构相结合的思路。本实现是交叉协方差子空间加Ridge，
不是完整PLS或该文降秩Ridge估计器。[数学定义](../../docs/GRAIN_QUERY_RESIDUAL_INPUT_THEORY_V1.md)
将输入方向、输出方向和GRAIN测量方向明确分开。

## 4. 下一机制：共享读法还是查询专用？

[NEXT36](../../docs/GRAIN_QUERY_DIRECTION_TRANSFER_PLAN_V1.md)冻结查询间方向
互换。在同一源折内，将T/F、N、寻求确认各自找到的8维输入方向互相借用，
接收查询仍用自己的源目标拟合。最多24个新增小Ridge头，无新神经调用。
对角线优胜支持专用输入；异查询也有效支持共用读取结构；混合结果定位
可共享与专用边界。先检验这项机制，不再沿本面板重复搜rank/alpha。

这是小面板中的数学机制发现。当前隐藏向量仍来自固定模型块16，源方向
也由模型导数构造；提取出的数值读法运行时免后缀/反向传播，不等于从生
文本开始完全脱离编码器。GRAIN计分、外部效标、辅助解释协议保持独立。

## 5. 重放与修正记录

规范r2两次构造耗时__TIMES__秒（不含输入加载和分析）；4残差PCA、12小
交叉协方差SVD、24Ridge，每次0新神经调用。35构造+8分析=43个数值文件
逐字节相同，__TESTS__项聚焦测试通过。两处禁读探针均生效，24独立预测
与本进程逐值相同。私有token字段、模型、作者ID不发布；仅6个汇总CSV。

首次入口误用了历史同名脚本；发现后恢复为0c940fd版本，独立入口改为
`run_grain_query_residual_input.py`，参数和比较不变。初次结果未审阅，旧包
留作历史记录；重新冻结的r2与旧包35构造数值一致。旧包不作为当前可重跑
证据入口。修正提交c689ca6，分析预冻结89a25a5，不改历史commit或lockbox。

重跑：`python scripts/run_grain_query_residual_input.py --name <new_private_name>`；
分析：`python scripts/analyze_grain_query_residual_input.py --source <new_private_name>`。
数值源包与原冻结环境需本地可用；无后台长跑待续。
"""
    report = report.replace("__TABLE__", "\n".join(table)).replace(
        "__GEOMETRY__", "\n".join(geometry)
    )
    report = report.replace("__TIMES__", "/".join(f"{t:.2f}" for t in times)).replace(
        "__TESTS__", str(tests)
    )
    (dest.parent / "QUERY_RESIDUAL_INPUT_REPORT_V1.md").write_text(report)
    print({"status": "PUBLISHED", "exact_files": len(checks), "tests": tests})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", type=int, required=True)
    publish(parser.parse_args().tests)
