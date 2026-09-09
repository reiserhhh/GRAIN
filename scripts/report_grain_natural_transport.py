#!/usr/bin/env python3
"""Publish aggregate natural-transport and query-residual findings, not raw texts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.natural_transport import correlation
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def render(transport, residual, target):
    """Build every quantitative table from the completed private packets."""
    for p in (transport, residual):
        if read(p / "status.json")["status"] != "COMPLETE":
            raise ValueError("Complete packets required")
    m = pd.read_csv(transport / "metrics.csv")
    r = pd.read_csv(residual / "metrics.csv")
    cross = pd.read_csv(residual / "cross_query.csv")
    energies = pd.read_csv(residual / "signed_energy_allocation.csv")
    tm, rm = read(transport / "manifest.json"), read(residual / "manifest.json")
    oof = pd.read_csv(transport / "private_source_oof.csv")
    base = m[
        (m.edit == "e1")
        & (m.target == "native_delta")
        & ~m.arm.str.startswith("semantic")
    ]
    main = base[
        ["query", "arm", "r", "r_lower", "r_upper", "mse_over_zero", "sign_match"]
    ]
    parts = r[r.kind == "transition"][
        ["query", "arm", "r", "r_lower", "r_upper", "mse_over_zero"]
    ]
    selective = cross[
        (cross.kind == "transition") & (cross.arm == "residual_configuration")
    ][["source_query", "target_query", "r", "r_lower", "r_upper"]]
    source = pd.DataFrame(
        [
            {
                "query": q,
                "estimand": kind,
                "authors": g.author_id.nunique(),
                "OOF_r": correlation(g.actual, g.predicted),
            }
            for (q, kind), g in oof.groupby(["query", "kind"])
        ]
    )
    semantic = m[
        (m.target == "native_delta") & m.arm.eq("semantic::" + m["query"] + "::delta")
    ][["query", "edit", "r", "r_lower", "r_upper"]]
    energy = energies[energies.kind == "transition"][
        ["query", "channel", "signed_energy_allocation", "rms"]
    ]
    text = f"""# GRAIN 逻辑研究：自然材料变化与查询可读余项

2026-09-10。持续研究 goal ACTIVE。完成 NEXT01 的有限自然变化测试、
NEXT02 的首个可计分余项接口，并为 NEXT03/04 给出材料组成与查询特异性实例。
这不是新量表、发布版本升级或新的人格预测 benchmark。

## 1. 本轮找到什么

**信息不缺在固定向量函数里，而主要缺在历史 GRAIN 因子到该函数的接口中。**
对十二位源样本外作者，删除预先选定的一组真实评论后，固定向量函数与原生
模型的判断变化相关：T/F **r=.886 [.742,.958]**、神经质
**r=.681 [.408,.873]**。相对“判断不变”的零变化基准，MSE 分别降低
**78.0%**、**37.0%**。这些是模型响应的重现，不是人格量表准确率。

进一步拆开后，历史 GRAIN 没有表示的**查询位置配置**最值得发展：其变化
与原生变化的相关为 T/F **.775 [.281,.945]**、神经质
**.642 [.156,.901]**，两者方向一致率均为10/12。
这使“余项里还有有用信息”变成了一个明确的、可计分的向量函数，而不再只
是笼统猜想。下面保留全部固定备选，主变化估计量预先指定为 transition。

## 2. 实际做了什么

- 历史冻结 GRAIN：12 因子，whole/native/fixed-128-cross各4；底座是既有
  TF-IDF+SVD24，不是把本轮查询函数重新命名成 GRAIN 因子。
- 源样本276人：T/F来自63个MBTI-only作者，神经质来自213个Big5-only作者。
  每人3个既有视图。只拟合模型固定函数 f，**未读取或拟合真实量表标签**。
- 评估12人，两查询，full/E1/E2共36份材料；与源拟合和历史因子样本不重合，
  但此前研究已用过这12人，不称新盲样本。
- E1删除三查询中语义极性最相关评论的并集；E2删除同数量、近似等长度、
  较低相关的其他评论。{tm["matched_authors"]}/12通过原设长度匹配门槛。
- E1是研究者改变证据可用性，**不是作者性格随时间改变**。源3视图也是不同
  评论抽样，不是相同问题重复回答或连续对话。
- 所有因子、词表、坐标常模冻结。只在源作者拟合固定alpha=100的Ridge；
  没有挑参、没有测试集校准、没有新神经推理或训练、没有API、没有打开lockbox。
- full/E1逐token重建并核对缓存；E2核对同一S1材料/原生缓存。E2没有对应
  完整P/Q缓存，因此不编造它的固定函数或余项结果。

`F`：固定Qwen2.5-1.5B的原生候选条件分数；T/F为p(F)，神经质为0–9候选
归一化后的加权分数。`f`：此前冻结的向量函数，保持原单位、**不截断**。
Pearson区间按完整作者对bootstrap1,999次；12人区间为探索性精度说明。

## 3. 共同算子能否解释不同人的材料变化

源样本按作者5折的描述性诊断（响应是f，不是真实人格）：

{source.to_markdown(index=False, floatfmt=".4f")}

between拟合作者均值，within拟合作者中心化视图，transition拟合按转换位置
中心化的相邻视图差分。源fold诊断的转换中心取训练作者；独立E1/E2评分仅用
冻结斜率乘实际Delta G，没有用评估批次重新居中。

同一函数在源数据相关，并不保证不同材料变换下仍有相同传递关系：

{main.to_markdown(index=False, floatfmt=".4f")}

T/F的GRAIN变化预测反向，特别是transition r=-.412；既有F02单方向控制也
不能解释实际删除。这不撤回之前“沿该向量干预能改变模型”的发现：
**人工走一个方向**与**真实删减文本后实际走了什么路径**是两个不同问题。
共同f算子能追踪实际变化，暂不需要靠“每人不同梯度”解释本轮现象。

水平也不同于变化：between模型在held完整材料的T/F原生/函数相关为.061/.149，
神经质为.286/.619。GRAIN有源内联系，但不宜把该联系直接当自然编辑导数。

## 4. 把差额变成有定义的观测量

固定查询函数有三个已有部分：

    f_q(H) = anchor_q + f_mean + f_configuration + f_final

这里的configuration是**固定问题token位置的中心化隐藏向量排列**：

    f_configuration = <G_centered, P - P_ref - mean_token(P - P_ref)>.

它不是文本跨日情绪流动、不是attention权重，也不自动等于思考过程。
不同文本改变同一问题各位置的表征，这部分位置差异被固定系数读取。

源拟合每部分的GRAIN预测后，定义：

    R_qc(T) = f_qc(H(T,q)) - (b_qc + beta_qc^T G(T)).

得到三个技术代号观测量：R_mean、R_configuration、R_final。
源作者均值作为固定参照，冻结其均值/标准差后可给新作者完整文本打z分。
本轮已保存24个作者×查询的完整位置与各变化分数、来源模型与参考参数。
这些不是三个新心理构念；它们是有数学含义的读出接口候选。

各部分beta求和重现此前总函数beta，最大差
{rm["max_coefficient_sum_error"]:.3g}；截距/anchor闭合误差
{rm["max_intercept_sum_error"]:.3g}。因此不是另换一组有利模型。

    Delta F = beta^T Delta G
            + Delta R_mean + Delta R_configuration + Delta R_final
            + (Delta F - Delta f).

{parts.to_markdown(index=False, floatfmt=".4f")}

between/within备选中R_configuration也保留正方向：T/F r=.741/.790，神经质
r=.713/.645。这里报告主transition结果而不挑最大的那个。

保留抵消项后的有符号能量分配mean(C_j*DeltaF)/mean(DeltaF²)：

{energy.to_markdown(index=False, floatfmt=".4f")}

每查询各项加和为1；T/F的GRAIN项为负、配置余项为+.810，说明存在抵消。
**这些不是心理方差百分比、因果中介比例或测量信度。** 残差归属依赖所选
固定坐标与源映射；恒等式本身不证明归属唯一。

## 5. 共享部分与查询相关部分同时存在

相同12人的配置余项变化，匹配与错配查询：

{selective.to_markdown(index=False, floatfmt=".4f")}

T/F配置对T/F响应较强、也与神经质部分同变；神经质配置对自己的响应较强，
对T/F很弱。当前得到的是**非对称的共享/查询相关结构候选**，不是所有
人格查询已划成互不相交的区域。NEXT04可以在相同材料上扩展，而不是拿不同
来源人群的两张矩阵硬拼。

## 6. 为什么简单语义距离不等于功能作用

此前E5极性坐标s的评论均值有精确材料组成公式：

    Delta mean(s) = removed_fraction * (mean_kept(s) - mean_removed(s)).

本轮最大数值误差{tm["semantic_mixture_max_error"]:.3g}。
它分开了删多少和删掉的材料与剩余材料有多不同，但不表示自由选择偏好。
匹配查询的Delta mean(s)与原生判断变化：

{semantic.to_markdown(index=False, floatfmt=".4f")}

T/F在E1负相关、E2正相关且区间宽。E1本来就按这个语义规则选，因此这不是
独立语义效度验证。低语义相关不等于模型不使用：T/F原生变化RMS在E1为
.0701、E2反而为.1116；神经质分别.0283/.0179。
**不能靠接近人格词或一次材料均值变化替代上下文内的功能读取。**
值得继续发展的不是把E2当无信息噪声，而是比较语义证据、位置配置和选择机制。

## 7. 逻辑图应如何升级

1. 保留“因子与模型相关”“因子方向干预有效”；新增自然材料实际路径层。
2. GRAIN不必独自吸收整个P/Q。引入源固定、查询可读的配置余项作为可计算
   候选对象，之后再研究如何用更轻的文本向量程序得到它。
3. 先分开材料可用性、作者主动选择、条件内表达；删除实验支持第一项，不
   偷渡成第三项的心理变化。
4. 查询余项有实际作用关联，但仍依赖模型隐藏状态。**本轮没有完成无LLM
   raw-text→P/Q**；无生成回答也不等于无模型表示。
5. 人格解释终点仍是独立自评/行为/状态通道。本轮跟踪的是模型怎样判断，
   不用模型自洽替代作者心理效标。旧的人类效标结果继续保留，未被本轮重估。

文献如何指导、而非代替本实验：
- [Mischel & Shoda (1995)](https://pubmed.ncbi.nlm.nih.gov/7740090/)同时关注平均
  水平和情境依赖模式；指导我们不混用between/change，不证明删除是心理情境。
- [Szabo et al. (2015)](https://proceedings.mlr.press/v38/szabo15.html)给出分布
  回归与两层采样的理论先例；有限均值坐标并非完备的人格分布表示。
- [Sundararajan et al. (2017)](https://proceedings.mlr.press/v70/sundararajan17a.html)
  将归因的敏感度与实现不变性公理化；本轮做固定仿射函数的分解，不冒充
  对完整非线性模型的积分梯度或唯一因果分解。

## 8. 复现与下一步

先运行合成数学测试，再运行既有私有缓存。需本地资料与冻结模型tokenizer，
无资料时不能只凭仓库重新生成私人作者数据：

```bash
"../project persona/.venv/bin/python" -m pytest -q tests/test_grain_natural_transport.py
"../project persona/.venv/bin/python" scripts/run_grain_natural_transport.py --output-dir results/grain_vector_mechanisms/logic_natural_transport_replay
"../project persona/.venv/bin/python" scripts/run_grain_query_residual.py --source results/grain_vector_mechanisms/logic_natural_transport_replay --output-dir results/grain_vector_mechanisms/logic_query_residual_replay
"../project persona/.venv/bin/python" scripts/report_grain_natural_transport.py --transport results/grain_vector_mechanisms/logic_natural_transport_replay --residual results/grain_vector_mechanisms/logic_query_residual_replay --report reports/grain_vector_mechanisms/NATURAL_TRANSPORT_LOGIC_REPLAY.md
```

完整私人结果分别保存在`{transport.relative_to(ROOT)}`、
`{residual.relative_to(ROOT)}`。每个新运行目录必须不存在。

下一轮优先：将R_configuration作为明确目标，比较已有非LLM/浅层向量路径
能否提取它；并用同材料更多查询辨认共享与独特部分。材料选择分支转入
“功能贡献为何不等于极性相似度”，不再循环优化这12个历史GRAIN因子的alpha。
真实行动通道、跨人群心理命名、端到端无模型表示是各自的待办，不以本轮
代数闭合宣告它们完成。
"""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    public_tables = target.parent / "natural_transport_tables_v1"
    public_tables.mkdir(exist_ok=True)
    for name, frame in (
        ("held_response", main),
        ("source_diagnostics", source),
        ("residual_parts", parts),
        ("cross_query", selective),
        ("signed_energy", energy),
        ("semantic_control", semantic),
    ):
        frame.to_csv(public_tables / f"{name}.csv", index=False)
    write(
        residual / "report_manifest.json",
        {
            "report": str(target.relative_to(ROOT)),
            "report_sha256": sha(target),
            "source_metrics_sha256": sha(transport / "metrics.csv"),
            "residual_metrics_sha256": sha(residual / "metrics.csv"),
            "cross_query_sha256": sha(residual / "cross_query.csv"),
            "finite_audit_only": True,
        },
    )
    print(str(target))


def main():
    """Generate the declared aggregate report using fixed result paths by default."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transport", type=Path, default=BASE / "logic_natural_transport_v1"
    )
    p.add_argument("--residual", type=Path, default=BASE / "logic_query_residual_v1")
    p.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "reports/grain_vector_mechanisms/NATURAL_TRANSPORT_LOGIC_REPORT_V1.md",
    )
    args = p.parse_args()
    render(args.transport.resolve(), args.residual.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
