#!/usr/bin/env python3
"""Publish aggregate reading-level evidence and verify the numerical replay."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write

TABLES = (
    "source_fidelity.csv",
    "criterion_links.csv",
    "affine_error_accounting.csv",
    "held_change_metrics.csv",
    "held_level_metrics.csv",
    "reader_geometry.csv",
)


def publish(source):
    """Keep author records private and require the current-code replay to match."""
    replay = source / "linted_replay"
    for folder in (source, replay):
        if read(folder / "status.json")["status"] != "COMPLETE":
            raise ValueError("Both original and current-code runs must complete")
    expected = read(replay / "manifest.json")
    current = {
        "script_sha256": ROOT / "scripts/run_grain_reading_levels.py",
        "core_sha256": ROOT / "grain_research/vector_mechanisms/reading_levels.py",
        "plan_sha256": ROOT / "docs/GRAIN_LOGIC_READING_LEVEL_PLAN_V1.md",
    }
    if any(sha(path) != expected[key] for key, path in current.items()):
        raise ValueError("Replay does not represent the current code and plan")
    names = (
        *TABLES,
        "loss_accounts.json",
        *read(source / "prediction_freeze.json")["files"],
    )
    matches = {name: sha(source / name) == sha(replay / name) for name in names}
    if not all(matches.values()):
        raise ValueError("Numerical replay differs")
    comparisons = source / "comparisons_v1"
    comparison_manifest = read(comparisons / "manifest.json")
    if comparison_manifest["prediction_freeze_sha256"] != sha(
        source / "prediction_freeze.json"
    ):
        raise ValueError("Paired comparison belongs to another prediction packet")
    public = ROOT / "reports/grain_vector_mechanisms/reading_level_tables_v1"
    public.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        table = pd.read_csv(source / name)
        if {"author_id", "text", "y_true"}.intersection(table.columns):
            raise ValueError("Author-level information cannot be published")
        shutil.copyfile(source / name, public / name)
    shutil.copyfile(
        comparisons / "paired_comparisons.csv", public / "paired_comparisons.csv"
    )
    audit = {
        "numerical_replay_byte_matches": matches,
        "prediction_freeze": read(source / "prediction_freeze.json"),
        "current_run_hashes": {key: sha(path) for key, path in current.items()},
        "comparison_script_sha256": comparison_manifest["script_sha256"],
        "publisher_sha256": sha(__file__),
        "paired_comparison_rows": comparison_manifest["rows"],
        "new_neural_calls": 0,
        "new_criterion_fits": 0,
    }
    write(public / "replay_provenance.json", audit)
    write(source / "code_replay_audit.json", audit)
    links = pd.read_csv(source / "criterion_links.csv")
    account = pd.read_csv(source / "affine_error_accounting.csv")
    delta = pd.read_csv(source / "held_change_metrics.csv")
    paired = pd.read_csv(comparisons / "paired_comparisons.csv")
    levels = pd.read_csv(source / "source_fidelity.csv")
    nine = links.pivot(index="query", columns=["arm", "target"], values="pearson_r")
    nine.columns = ["_".join(c) for c in nine.columns]
    nine = nine[["pooled_c", "pooled_r", "between_c", "between_r"]]
    tf = account.query(
        "query=='TF_cont' and arm=='pooled' and target=='r' and level=='pooled_oof'"
    ).iloc[0]
    fc = account.query(
        "query=='TF_cont' and arm=='pooled' and target=='r' and level=='within_cv_fold'"
    ).iloc[0]
    primary = paired.query(
        "target=='r' and (scope=='source_author_mean' or scope=='held_e1')"
    )
    natural = delta.query("edit=='e1' and estimate=='r' and target=='native_delta'")
    source_fidelity = levels.query("target=='r' and query in ['TF_cont','neuroticism']")
    report = f"""# GRAIN逻辑研究：读取层级、幅度误差与九查询扩展

2026-09-10。第五轮有限实验完成，持续goal ACTIVE。接在
[真实作者效标连接](CONFIGURATION_CRITERION_LOGIC_REPORT_V1.md)之后。

## 1. 本轮发现及决定

**T/F剩余效标关联不是简单的分数幅度问题，也没有被单独的作者均值训练解开。**
但同一轻量数学接口已经扩展到全部九个查询，保留了E/A/O的一部分真实自评联系。
接下来值得找的是条件和材料配置如何进入读取器，不是继续放大分数或试alpha。

- T/F余项OOF误差贡献+.1475中，幅度项仅+.0068，非仿射剩余项+.1406。
  这里“非仿射”指不能用当前标量预测的线性重标定消去，不等于已发现非线性机制。
- 保留同一E5输入和惩罚后，作者间专用读取器的T/F余项效标r从.2489降至.2034。
  配对差值-.0454，95%CI[-.0982,.0030]。不替换原pooled读取器。
- 九查询扩展的pooled余项E/A/O自评r为.1894/.1484/.2061。没有用真实Y训练。
  36项本轮关联BH筛查中只有O的两种读取器q<.05；其他局部关联仍是探索性信息。
- N的within读取器自然删除响应r=.788，高于pooled .733的点估计；配对差+.055，
  CI[-.098,.358]，而实际组件变化忠实度从.841降至.720。因此不单凭最高相关换模型。

## 2. 同一输入，三个不同估计对象

源样本276作者：213 Big5-only、63 MBTI-only，每人三个12评论视图。
九个query分别训练，五折按作者划分，seed=20260910。另用原有12位源不重叠
作者的full/E1/E2材料检验T/F和N。12人材料是已反复使用的研究面板，不是新盲评。
三个视图是抽样材料，不是多次临床会话，更不意味着人格应在三次采样中恒定。

固定训练目标为查询配置c，以及源折GRAIN参照后的R=c-m_G。每折的GRAIN
映射、E5特征中心和尺度均仅由训练作者拟合；query目标来自已冻结模型函数。
这里c是查询token位置上的隐状态配置，不是作者时间轴运动。历史12个GRAIN
因子也保持原定义，没有把新E5回归头重命名为GRAIN因子。

令A为每作者J=3视图的均值投影，W=I-A；AW=0。对误差v，有：

    ||v||² = ||Av||² + ||Wv||²
    pooled MSE = between MSE + within MSE.

因此共享一个w是在同时求两类误差的折中，而不是证明两类规律必然相同。
本轮比较：

    pooled:  mean_uv (t_uv - b - w' x_uv)² + .01||w||²
    between: mean_u  (tbar_u - b - w' xbar_u)² + .01||w||²
    within:  mean_uv ((t_uv-tbar_u) - w'(x_uv-xbar_u))² + .01||w||².

三者保留同一源训练视图的特征中心/trace尺度，避免归一化改变成为隐藏实验臂。
Ridge alpha按实际损失样本数缩放。within没有可识别的作者水平截距，故只预测
中心化视图或材料差分，不能拿它给新作者输出一个绝对人格位置。

这一分层来自[Curran与Bauer(2011)](https://curran.web.unc.edu/wp-content/uploads/sites/6785/2015/03/CurranBauer2011.pdf)
的个体内/个体间区分，以及[Mundlak(1978)](https://people.stern.nyu.edu/wgreene/Econometrics/Mundlak-1978.pdf)
对重复单位异质性的处理思想。当前是抽样视图的预测问题，不声称复现其纵向
心理模型或实现因果面板估计。五个合成测试含“作者间正斜率、作者内负斜率”例子。

真实数据54个损失分解最大误差5.85e-18。最终源R的between/within系数余弦
T/F=.178、N=.249，表明有限拟合下的读取规则确实不同；差异本身不保证改进。

{source_fidelity[["query", "arm", "level", "r"]].to_markdown(index=False, floatfmt=".4f")}

作者均值是每人三视图均值，view0是单个12评论视图，within是中心化视图。
三个地址不互当同一种测试。within相关不把视图当独立被试做显著性判断。

## 3. T/F遗漏关联的幅度与非仿射部分

只用已冻结的目标t和预测p描述t=a+b*p+e，不用效标Y决定b。有：

    Cov(t-p,Y) = (b-1)Cov(p,Y) + Cov(e,Y),  Cov(e,p)=0.

全部贡献除以共同sd(t)sd(Y)。T/F pooled R给出：

- b={tf.alignment_slope:.6f}；原读出/目标r={tf.teacher_fidelity_r:.6f}。
- 总误差关联贡献{tf.total_error_criterion_contribution:.6f}
  =幅度{tf.amplitude_contribution:.6f}+剩余{tf.non_affine_contribution:.6f}。
- 剩余e与效标r={tf.non_affine_criterion_r:.6f}。
- 去掉CV折间均值后仍有剩余贡献{fc.non_affine_contribution:.6f}、e/Y r={fc.non_affine_criterion_r:.6f}。

这是描述性协方差记账，不是部署校准、信息比例或新泛化成绩。剩余可能来自
均值表征损失、读取算子不足、采样误差或教师的不稳定性；结果将简单幅度解释
降级，但没有把剩余强行命名为“未发现的人格”。N旧删除变化只有12.9%幅度是
另一个地址，不能借它解释T/F作者水平上的+.147。

## 4. 九查询固定接口与真实效标

以下是作者OOF的量表Pearson r。c是完整配置目标，r是GRAIN参照后的余项目标。
真实标签只在全部预测和模型文件冻结之后读取。扩展由先前E/A/O发现推动，
但实际计算包含所有九query，没有选择性省略负值或用标签翻转符号。

{nine.to_markdown(floatfmt=".4f")}

T/F pooled R AUC=.6538，between=.6216；J/P保留原方向和负关联。
S/N少数类只有4人，不据其AUC做路线升级。Big5和MBTI来自不同5/4查询人群，
这张表不构成同人Big5-MBTI桥接。所有Y早已用于源探索，不是新的独立效度样本。

## 5. 自然材料变化与源位置不能互相替代

E1已有12作者的固定原生模型输出变化关联：

{natural[["query", "arm", "r", "r_lower", "r_upper"]].to_markdown(index=False, floatfmt=".4f")}

paired比较按同一作者整体重采样1999次，条件于现有拟合结果：

{primary[["scope", "query", "endpoint", "candidate", "delta_r", "delta_r_lower", "delta_r_upper"]].to_markdown(index=False, floatfmt=".4f")}

N的between程序对E1原生变化相对pooled差-.211，区间[-.435,-.038]，这是明确的
局部退化。within N在原生变化关联上的点估计更好，但组件MSE/零变化=.904，
差于pooled .816；并不同时改善函数忠实度、幅度和效标。E2全部结果保留在CSV，
只有原生变化缓存，没有P/Q组件真值；不补造缺失的组件指标。

## 6. 对整套逻辑的升级

系统现在明确保留三类连接：

1. 表达分布/查询配置中的信息是否可以被一个输入程序计算出来。
2. 同一程序在人际位置与材料变化两个对象上分别保留什么信息。
3. 保留的组件与作者效标、完整模型输出分别怎样关联，以及组合是否抵消关联。

这与GRAIN“条件前置、不要把差异整体当噪声”的原则一致：between和within
都保留。原pooled程序仍是简洁的工作接口；分层程序是机制对照，不是自动升级。

**下一步：** 用相同P/Q端点的条件缓存比较均值相同而条件/配置不同的材料，
找到当前输入表示不可区分、但目标函数可区分的实际例子；再决定应加入哪个
条件统计或读取操作。不是用Y训练新残差头。组件组合卡NEXT10也继续保留，
不因本轮九查询表完成就宣告逻辑穷尽。

历史[S6](S6_MEAN_NULL_PATH_REPORT.md)已证明block16内保持评论均值的干预仍能
改变模型判断，因此不重做同一证明。需要补的接口是：该内部配置机制与当前
独立E5评论均值输入之间有什么对应。两者不是同一个向量空间，不能借用S6
直接宣称当前E5输入的遗漏来源已被识别。优先检索同内容重排的实际端点；
旧CON否认/时间缓存也只在作者、材料、query、目标函数逐项吻合时才接入。

## 7. 可复现交付

- 方案：`docs/GRAIN_LOGIC_READING_LEVEL_PLAN_V1.md`，执行前固定。
- 核心/runner：`reading_levels.py`、`run_grain_reading_levels.py`。
- 配对比较：`analyze_grain_reading_level_comparisons.py`，24行主比较。
- 私人缓存：`results/grain_vector_mechanisms/logic_reading_levels_v1/`。
- 本报告汇总表：`reading_level_tables_v1/`；不包含作者ID、原文或量表逐人值。
- 首跑4.03秒、当前代码回放4.13秒；0新神经调用、0真实效标拟合。
- 旧T/F/N的828条pooled预测重放误差<1e-16；本轮{len(matches)}个数值文件逐字节复现。
- 本轮相关测试70通过，其中5个新增；Ruff和diff检查另记录于执行台账。
- 没有改发布计分器、旧lockbox、MEPS用途，没有API上传或push。

从独立仓库运行（需要已有私人缓存，不随公共仓库分发）：

```bash
"../project persona/.venv/bin/python" scripts/run_grain_reading_levels.py --output-dir results/grain_vector_mechanisms/logic_reading_levels_v1
"../project persona/.venv/bin/python" scripts/analyze_grain_reading_level_comparisons.py
"../project persona/.venv/bin/python" scripts/run_grain_reading_levels.py --output-dir results/grain_vector_mechanisms/logic_reading_levels_v1/linted_replay
"../project persona/.venv/bin/python" scripts/report_grain_reading_levels.py
```

runner不覆盖现有实验目录；已有包可直接读取，不必为表格完整重跑。
"""
    target = ROOT / "reports/grain_vector_mechanisms/READING_LEVEL_LOGIC_REPORT_V1.md"
    target.write_text(report, encoding="utf-8")
    print(f"Published {target.name}; {len(matches)} numerical replay files match")


def main():
    """Build a report only from completed, provenance-matched fixed packets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "logic_reading_levels_v1")
    args = parser.parse_args()
    publish(args.source)


if __name__ == "__main__":
    main()
