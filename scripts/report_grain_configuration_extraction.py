#!/usr/bin/env python3
"""Publish aggregate configuration-extraction findings and replay frozen programs."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.run_grain_natural_transport import FACTORS, QUERIES

SOURCE = BASE / "logic_configuration_extraction_v1"
RELATIONS = BASE / "logic_configuration_relations_v1/operator_aligned_r3"
PUBLIC = ROOT / "reports/grain_vector_mechanisms/configuration_extraction_tables_v1"


def replay():
    """Reapply all eight saved programs; do not fit or read private author labels."""
    freeze = read(SOURCE / "prediction_freeze.json")
    if sha(SOURCE / "private_held_predictions.csv") != freeze["held_sha256"]:
        raise ValueError("Frozen predictions changed")
    meta = pd.read_csv(SOURCE / "private_feature_rows.csv")
    part = meta[meta.group == "held"]
    ids = part.index.to_numpy(int)
    with np.load(SOURCE / "private_features.npz", allow_pickle=False) as z:
        flat, offsets, tokens = z["bags"], z["offsets"], z["token"]
    bags = [flat[offsets[i] : offsets[i + 1]] for i in ids]
    old = pd.read_csv(SOURCE / "private_held_predictions.csv")
    error, hashes, checked = 0.0, {}, 0
    for (q, arm), observed in old.groupby(["query", "arm"]):
        path = SOURCE / f"private_{q}_{arm}.joblib"
        stored = joblib.load(path)
        model, gm = stored["input"], stored["grain"]
        p = model.predict(bags, tokens[ids, list(QUERIES).index(q)])
        g = part[FACTORS].to_numpy() @ gm["coef"] + gm["intercept"]
        values = np.column_stack([p, p[:, 0] - g, g])
        key = pd.MultiIndex.from_frame(part[["author_id", "edit"]])
        prior = observed.set_index(["author_id", "edit"]).loc[
            key, ["predicted_c", "predicted_r", "derived_r", "grain_part"]
        ]
        error = max(error, float(np.max(np.abs(values - prior.to_numpy()))))
        hashes[path.name] = sha(path)
        checked += len(prior)
    if error > 1e-10 or checked != 288:
        raise ValueError("Saved programs no longer reproduce original predictions")
    result = {
        "prediction_rows": checked,
        "max_absolute_error": error,
        "additional_model_fits": 0,
        "new_neural_calls": 0,
        "model_hashes": hashes,
        "current_module_sha256": sha(
            ROOT / "grain_research/vector_mechanisms/configuration_extraction.py"
        ),
    }
    write(SOURCE / "code_replay_audit.json", result)
    return result


def table(frame, cols):
    """Render only explicitly chosen aggregate columns."""
    return frame[cols].to_markdown(index=False, floatfmt=".4f")


def main():
    """Validate and export a finite report without changing research endpoints."""
    if any(
        read(p / "status.json")["status"] != "COMPLETE" for p in (SOURCE, RELATIONS)
    ):
        raise ValueError("Complete source analyses required")
    with threadpool_limits(limits=2):
        audit = replay()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / "metrics.csv"] + [
        RELATIONS / name
        for name in (
            "source_view0_metrics.csv",
            "query_relations.csv",
            "functional_semantic_directions.csv",
            "evidence_mixture_metrics.csv",
            "residual_operator_metrics.csv",
        )
    ]
    for path in paths:
        frame = pd.read_csv(path)
        if set(frame.columns) & {"author_id", "text", "body", "label", "user_id"}:
            raise ValueError("Only aggregate numeric tables may be published")
        frame.to_csv(PUBLIC / path.name, index=False)
    manifest = read(RELATIONS / "manifest.json")
    m = pd.read_csv(SOURCE / "metrics.csv")
    source = pd.read_csv(RELATIONS / "source_view0_metrics.csv")
    fidelity = m.query(
        "split=='held' and level=='e1' and target=='r_delta' and estimate=='predicted_r'"
    )
    native = m.query(
        "split=='held' and estimate=='predicted_r' and target=='native_delta'"
    )
    source = source.query("estimate=='predicted_c'")
    levels = m.query("split=='held' and level=='full' and estimate=='predicted_c'")
    levels = source.merge(
        levels, on=["query", "arm"], suffixes=("_source_view0", "_held_full")
    )
    relations = pd.read_csv(RELATIONS / "query_relations.csv")
    selected = relations.query(
        "level=='author' and ((query2=='neuroticism') or (query1=='neuroticism') or (query1=='agreeableness' and query2=='extraversion') or (query1=='SN_cont' and query2=='TF_cont'))"
    )
    record = {
        "source_manifest_sha256": sha(SOURCE / "feature_manifest.json"),
        "relation_manifest_sha256": sha(RELATIONS / "manifest.json"),
        "private_source": str(SOURCE.relative_to(ROOT)),
        "private_relations": str(RELATIONS.relative_to(ROOT)),
        "replay_rows": audit["prediction_rows"],
        "max_replay_error": audit["max_absolute_error"],
        "runtime_seconds": read(SOURCE / "status.json")["seconds"],
        "new_neural_calls": 0,
        "files": {p.name: sha(PUBLIC / p.name) for p in paths},
    }
    write(PUBLIC / "provenance.json", record)
    report = f"""# GRAIN 逻辑研究：轻量配置提取与共同作用算子

2026-09-10。持续 goal **ACTIVE**。NEXT05、NEXT04b 完成一次固定比较；
NEXT03b 新增可拆解的评论级功能读数。没有更新发布评分器或打开旧 lockbox。

## 1. 本轮推进了哪条主线

**已把一个有用的查询配置读数，从完整模型状态搬到固定 E5 向量上的线性程序。**
神经质查询中，直接预测的配置余项变化与实际余项变化
**r=.841 [.602,.953]**；与原生模型判断变化 **r=.733 [.209,.926]**。
这些来自同一12位源样本外作者的真实评论删减，不是心理量表准确率。
目标余项的变化幅度仍明显收缩：预测RMS=.002364、实际=.018274，约12.9%；
相对零变化基准MSE=.816，即减少18.4%。因此是可用的变化读数，不是等幅重建。

纯 token 嵌入查表在完整文本的神经质配置水平上 r=.744，却无法跟随同样的
原生变化（r=-.106）。**一个人的位置读得出，不等于证据改变的作用也读得出。**
T/F的新轻量程序没有重现此前完整P/Q的自然变化优势，保留为另一种输入支持需求。

## 2. 固定任务、实际输入与依赖

- 源276作者：213 Big5-only、63 MBTI-only；每人3视图、每视图12评论。
  五折按作者留出，所有同作者视图同折。只学习固定模型函数，不读取人格标签。
- 独立评分12作者，每人full/E1/E2三份已有材料。此前用过此评估组，本轮不称
  全新盲评。E1/E2为预先固定的材料删除，不是人的人格随时间改变。
- 教师配置 c 是固定问题token位置P的中心化排列读数，不是注意力权重或
  作者跨日动态。R=c-m_G，m_G=b+beta^T G为源作者均值拟合的GRAIN部分。
- 本轮R使用 **between参照**。此前报告主transition余项的.775/.642不可偷换
  为本轮同一估计量；此前between余项与原生变化为.741/.713。
- 历史GRAIN仍是冻结的12因子 TF-IDF/SVD24 系统，并非新E5程序改名。
- 四个固定输入程序：E5评论均值；均值+极性相关加权；均值+256随机傅里叶
  分布特征；纯token嵌入均值+问题相似度加权。最后一个是新B0式控制，不是
  旧已发布B0成绩的重跑。现有B4缓存的材料/问题不匹配，记为未做，不记失败。
- 两个输出头均为相同几何下的Ridge。每折只在训练视图拟合中心/等迹块尺度、
  RFF带宽及系数；alpha=.01乘训练视图数，种子20260910，无挑参。
- E5新文本仍需要E5编码器；缓存运行无需新增神经前向。token控制只读固定
  Qwen输入嵌入表，0个上下文层。均不生成自然语言推理过程。

全部276份源材料token签名核对通过。先冻结所有预测，再联接已有原生终点。
实际完整比较耗时{record["runtime_seconds"]:.2f}秒，新增神经调用0。

## 3. 水平、自然变化、幅度各有不同答案

下表**均为每人12评论的单个完整视图**。源OOF是view0，避免把源3视图作者
均值与held单视图混为一谈。源作者3视图均值和文内中心化结果另存完整CSV。

{table(levels, ["query", "arm", "r_source_view0", "r_held_full"])}

N的E5均值水平：源单视图.511、held.495；不是拿源3视图均值.635宣称迁移
退化。token路线：源单视图.357、held.744，说明这12人水平结果不能代表其
一般稳定性。T/F源.783、held.414，且自然变化为负，不能靠水平分数替代变化。

**直接预测R的E1变化忠实度**（目标是R变化，不是完整原生F）：

{table(fidelity, ["query", "arm", "r", "r_lower", "r_upper", "actual_rms", "predicted_rms", "mse_over_zero", "sign_match"])}

**同一R预测变化与完整原生模型变化的关联**：

{table(native, ["query", "arm", "level", "r", "r_lower", "r_upper"])}

E5三种N程序方向相近，没有充分理由升级到复杂加权/分布程序。E2上关联很弱，
所以本次局部成功对应E1证据变化，不能把低语义相关材料直接删掉作为无信息。
原生关联只报告r，不把一个分量当完整原生分数计算“预测MSE”。
所有CI按完整作者对bootstrap1,999次；未以view为独立作者扩大样本。

## 4. 从黑箱差额到一条评论可计算的函数

E5均值路线可以精确写成（配置头或余项头分别成立）：

    psi_q(h_j) = w_q^T h_j
    score_q(T) = b_q + (1/n) sum_j psi_q(h_j)
    Delta score_q = (k/n) [mean_kept(psi_q) - mean_removed(psi_q)].

这里h_j是完整评论向量。删掉k条评论时，两个均值之差和删除比例完全解释这个
已学习程序的变化。实际缓存最大闭合误差
{manifest["evidence_maps"]["max_removal_replay_error"]:.3g}。
因此我们得到“什么材料贡献于这个查询读数”的显式函数，而非只知道一个相关数。

N余项权重与正负人格词极性方向cosine=.083；N配置头为.081。学习的功能方向
并非简单的人格词距离。高维中的小cosine不等于信息独立或新的正交心理构念。
固定语义极性均值的E1变化与原生响应r=.143；学习的余项函数为.733。
这个差别支持研究“模型怎样使用证据”，而不仅是“证据离人格词多近”。
上述psi是轻量程序的精确分解，不冒充LLM内部真实注意力或人的心理因果贡献。

## 5. 为什么两种减法会不同

以S_X表示在同一训练输入、中心化、正则项下，响应向量到新样本预测的固定
线性学习算子。Ridge对响应是线性的，含截距时该结论仍成立：

    R_direct = S_X(c - m_G) = S_X(c) - S_X(m_G)
    R_derived = S_X(c) - m_G
    R_direct - R_derived = m_G - S_X(m_G).

直接预测余项，并不神秘地发现第四块信息；它减掉的是**同一输入表示可重现的
GRAIN部分**。事后减法减掉的是完整GRAIN观测，两者相差GRAIN的表示搬运误差。
本轮N direct变化忠实度.841，对比derived .393。由哪种误差相消造成优势仍应
分开检验，而不能把较高r命名为“更纯人格”。这是可计算的跨表示连接机制。
实际减法闭合误差{manifest["residual_operator_account"]["max_subtraction_identity_error"]:.3g}；
独立合成测试核对了Ridge响应线性性，不只验证一条代数重排。

## 6. 九个查询中，什么共享、什么特异

全部5个Big5查询在同一213作者上、4个MBTI查询在同一63作者上分别计算，
共16对x作者/个体内两个层级=32条关联。**未把两个人群拼成九查询同人矩阵。**
下表是源作者3视图均值的配置c、GRAIN表示部分m_G、余项R之间的对应关系：

{table(selected, ["query1", "query2", "authors", "configuration_r", "grain_r", "residual_r", "residual_r_lower", "residual_r_upper"])}

N与其余Big5查询的余项关联在作者水平很小（-.006至.103），但A/E余项仍为
.560，A/O为.455。E/N的GRAIN部分r=.784，余项只有.049。
同一个共享表示不等于所有查询剩余部分共享；“共有+查询特异”有具体实例。
文内N/C余项r=.244，作者水平约-.006，再次说明特异性取决于所问的层级。
这些是**模型查询组件**的关系，不是实际Big5量表关系或新的H2桥接成功。

所有分解保留交叉/抵消项：

    Cov(c) = Cov(m_G) + Cov(R) + Cov(m_G,R) + Cov(R,m_G).

最大误差{manifest["query_relations"]["covariance_identity_max_error"]:.3g}。
源整体拟合的关系图是描述性结构，不等于留出数据验证或独立因果中介。

## 7. 文献、发现顺序与下一问题

- [Szabo等(2015)](https://proceedings.mlr.press/v38/szabo15.html)为分布嵌入加
  Ridge及“两层采样”提供理论先例；促使我们同时明确作者数、评论数和统计层级。
  其有条件一致性定理不保证当前表示包含全部人格信息。
- [Rahimi与Recht(2007)](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
  为随机傅里叶核近似提供依据。本轮256维RFF没有带来普遍优势，不再扩网格。
- [Ilse等(2018)](https://proceedings.mlr.press/v80/ilse18a.html)是学习型多实例
  加权的先例。本轮只是固定语义加权，不把它称为训练了注意力模型或因果解释。

实验先冻结[四路线方案](../../docs/GRAIN_LOGIC_CONFIGURATION_EXTRACTION_PLAN_V1.md)，
再看结果。评论功能式、减法搬运解释、单视图层级核对是结果后的数学/描述性
跟进，不回写为事前假设。r1关系分析的bootstrap未先排序作者，r2统一顺序，
r3新增减法解释；数值点估计与源模型不变，早版本保留。没有用held结果重新拟合。

下一条主要连接是：**这些已经固定的技术读数究竟带来哪些真实量表关联，
以及与GRAIN的共享/剩余信息如何分配？** 沿源作者OOF预测及既有许可量表做
探索性联接，先解剖信息，不换目标或训练精度网格。T/F若发展条件响应程序，
先需要匹配的构造材料；本轮12人不作为训练修补材料。

## 8. 交付与复现

私人数值/模型：`results/grain_vector_mechanisms/logic_configuration_extraction_v1/`
及`logic_configuration_relations_v1/operator_aligned_r3/`，仍gitignored。
公开的是本报告与[完整汇总表](configuration_extraction_tables_v1/)。
当前代码重放{audit["prediction_rows"]}条原始保存预测，最大差
{audit["max_absolute_error"]:.3g}；无新拟合。源预测冻结及缓存签名均留档。

```bash
# 从独立GRAIN仓库执行；输出目录必须新建，避免覆盖证据。
"../project persona/.venv/bin/python" scripts/run_grain_configuration_extraction.py \\
  --output-dir results/grain_vector_mechanisms/logic_configuration_extraction_replay
"../project persona/.venv/bin/python" scripts/analyze_grain_configuration_logic.py \\
  --source results/grain_vector_mechanisms/logic_configuration_extraction_replay \\
  --output-dir results/grain_vector_mechanisms/logic_configuration_relations_replay
"../project persona/.venv/bin/python" scripts/report_grain_configuration_extraction.py
```

最后一条只渲染当前已声明原始运行，不自动改用上述replay目录。测试命令及
实际通过数量记录在活跃goal台账；私人数据缺失时仍可运行合成单元测试。
"""
    path = (
        ROOT
        / "reports/grain_vector_mechanisms/CONFIGURATION_EXTRACTION_LOGIC_REPORT_V1.md"
    )
    path.write_text(report, encoding="utf-8")
    print(path.relative_to(ROOT))
    print(record)


if __name__ == "__main__":
    main()
