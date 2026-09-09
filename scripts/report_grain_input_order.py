#!/usr/bin/env python3
"""Replay reversal evidence and publish aggregate logic findings only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_path_routes import paired_delta
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.natural_transport import paired_summary
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def main():
    """Cross-check raw P/Q, paired source scores and fixed component arithmetic."""
    out = BASE / "logic_input_order_v1/paired_r2"
    manifest, status = read(out / "manifest.json"), read(out / "status.json")
    if (
        status["status"] != "COMPLETE"
        or status["calls"] != 52
        or status["completed"] != 24
    ):
        raise ValueError("Full declared panel required")
    for name, digest in manifest["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Acquisition code no longer matches")
    if sha(manifest["functions"]) != manifest["functions_sha256"]:
        raise ValueError("Frozen function changed")
    with np.load(manifest["functions"], allow_pickle=False) as z:
        fs = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in ("TF_cont", "neuroticism")
        }
    replay_errors, old_errors, repeat_errors = [], [], []
    for path in sorted((out / "private_packets").glob("unit_*.json")):
        p = read(path)
        if sha(path.with_suffix(".npz")) != p["arrays_sha256"]:
            raise ValueError("Packet changed")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            for order in ("original", "reverse"):
                parts = fs[p["query"]].parts(z[order + "_P"], z[order + "_Q"])
                replay_errors.extend(
                    abs(v - p[order]["parts"][k]) for k, v in parts.items()
                )
        old_errors.append(max(p["old_replay_error"], p["old_component_error"]))
        if "same_run_repeat_error" in p:
            repeat_errors.append(p["same_run_repeat_error"])
    n = pd.read_csv(out / "private_observations.csv")
    effects, parts, pairs = [], [], []
    for q, group in n.groupby("query"):
        a = group[group.order == "original"].set_index("author_id").sort_index()
        b = group[group.order == "reverse"].set_index("author_id").loc[a.index]
        cols = ["mean", "configuration", "final", "score", "native"]
        d = b[cols] - a[cols]
        denom = d.score.std(ddof=0) * d.native.std(ddof=0)
        for k in cols:
            v = d[k].to_numpy()
            effects.append(
                {
                    "query": q,
                    "component": k,
                    "n": len(v),
                    "mean_absolute_change": float(np.mean(np.abs(v))),
                    "rms_change": float(np.sqrt(np.mean(v * v))),
                    "nonzero_above_2e_minus6": int(np.sum(np.abs(v) > 2e-6)),
                    "positive": int(np.sum(v > 0)),
                    **{
                        f"native_{key}": value
                        for key, value in paired_summary(
                            d.native, v, calibrated=False
                        ).items()
                    },
                }
            )
            if k in ("mean", "configuration", "final"):
                value = np.mean((v - v.mean()) * (d.native - d.native.mean())) / denom
                parts.append(
                    {
                        "query": q,
                        "component": k,
                        "contribution_to_full_native_r": float(value),
                    }
                )
        pairs.append(
            {
                "scope": "held_change",
                "query": q,
                "target": "native",
                "comparison": "configuration_minus_full_f",
                **paired_delta(
                    d.native.to_numpy(),
                    d.configuration.to_numpy(),
                    d.score.to_numpy(),
                    draws=1999,
                ),
            }
        )
    oof = pd.read_csv(out / "private_source_oof.csv")
    labels = pd.read_csv(
        BASE / "all_factor_source_m29_v1/private_external_links.csv"
    ).set_index(["author_id", "query"])
    for q, g in oof.groupby("query"):
        a = g[g.arm == "mean"].groupby("author_id").mean(numeric_only=True).sort_index()
        b = (
            g[g.arm == "ordinal"]
            .groupby("author_id")
            .mean(numeric_only=True)
            .loc[a.index]
        )
        y = labels.xs(q, level="query").loc[a.index].y_true.to_numpy()
        for t in ("c", "r"):
            for endpoint, truth in (
                ("teacher", a["actual_" + t].to_numpy()),
                ("criterion", y),
            ):
                pairs.append(
                    {
                        "scope": "source_author_mean",
                        "query": q,
                        "target": t,
                        "endpoint": endpoint,
                        "comparison": "ordinal_minus_mean",
                        **paired_delta(
                            truth,
                            b["predicted_" + t].to_numpy(),
                            a["predicted_" + t].to_numpy(),
                            binary=q == "TF_cont" and endpoint == "criterion",
                            draws=1999,
                        ),
                    }
                )
    effect, part, pair = map(pd.DataFrame, (effects, parts, pairs))
    audit_rows = [r["token_audit"] for r in read(out / "private_design.json")]
    audit = {
        "max_pq_component_replay_error": max(replay_errors),
        "max_original_endpoint_error": max(old_errors),
        "repeat_count": len(repeat_errors),
        "max_repeat_error": max(repeat_errors),
        "units": len(audit_rows),
        "neural_calls": status["calls"],
        "seconds": status["seconds"],
        "all_comment_multisets_preserved": all(
            r["comment_bag_unchanged"] for r in audit_rows
        ),
        "max_prompt_length_delta": max(abs(r["length_delta"]) for r in audit_rows),
        "max_token_multiset_l1": max(r["token_bag_l1"] for r in audit_rows),
        "mean_token_multiset_l1": float(
            np.mean([r["token_bag_l1"] for r in audit_rows])
        ),
        "acquisition_sha256": sha(ROOT / "scripts/run_grain_input_order.py"),
        "publisher_sha256": sha(__file__),
        "source_prediction_freeze": read(out / "prediction_freeze.json"),
    }
    if max(replay_errors) > 1e-12 or len(repeat_errors) != 4:
        raise ValueError("Independent component replay failed")
    public = ROOT / "reports/grain_vector_mechanisms/input_order_tables_v1"
    public.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("effects", effect),
        ("component_contributions", part),
        ("paired_comparisons", pair),
    ):
        table.to_csv(public / f"{name}.csv", index=False)
        table.to_csv(out / f"{name}.csv", index=False)
    for name in ("orbit_accounts", "response_metrics", "source_links"):
        table = pd.read_csv(out / f"{name}.csv")
        if name == "orbit_accounts":
            table["prediction_role"] = np.where(
                table.component == "configuration",
                "trained_E5_mean",
                "oracle_pair_mean_for_floor_only",
            )
        if "author_id" in table:
            raise ValueError("Private column in public table")
        table.to_csv(public / f"{name}.csv", index=False)
    write(public / "provenance.json", audit)
    write(out / "report_audit.json", audit)
    floor = (
        pd.read_csv(out / "orbit_accounts.csv")
        .query("component=='configuration'")
        .copy()
    )
    floor["odd_floor_fraction_of_pair_mse"] = floor.odd_floor / floor.pair_mse
    source = pd.read_csv(out / "source_links.csv")
    src = source.pivot(
        index=["query", "target"], columns=["endpoint", "arm"], values="pearson_r"
    )
    src = src["criterion"].reset_index()
    focus = effect[effect.component.isin(["configuration", "score", "native"])]
    report = f"""# GRAIN逻辑研究：真实评论的排列、输入可区分性与目标选择

2026-09-10。NEXT08b有限实验完成；持续goal ACTIVE。

## 1. 新发现

**当前E5评论均值确实把某种LLM读取差异压成了同一个输入。**
12作者相同12条评论原文，仅反转列表：T/F原生判断变化RMS=.1048，N=.0341；
两查询均12/12作者变化超过2e-6。原始端点和同次重复误差都是0。
因而这些变化不是靠给同一均值换一个更强的回归器就能重现的。

但“把顺序加回来”也不是完整解决方案：源拟合位置加权项对新反转的配置
变化r=-.236(T/F)/.050(N)，MSE/零变化约1.22/1.23，没有改善。
它在源作者的T/F目标拟合r从.656升至.685，却把自评关联r从.249降至.203。
**输入区分能力、正确条件运算、作者效标保留，是三个连续但不同的机制。**

## 2. 数据与实际操作

同一旧12位源不重叠研究作者，T/F和N各自original/reverse24对。
评论原文、评论数、同一E5向量多重集保持；显示顺序和编号归属变化。
总提示token长度不变；自然渲染后的边界token有0-4个计数L1差异，均值3。
这是完整评论的呈现顺序干预，不是纯位置编码干预或人的性格变化。

第一次prepare在token多重集检查处停止，未执行任何新神经计算或源拟合。
随后在神经结果出现前记录tokenizer-only修订，保留全部作者和这些差异。
现行包为`logic_input_order_v1/paired_r2/`。旧12作者面板已使用过，不称新盲评。

固定模型Qwen2.5-1.5B，MPS FP32/eager，不生成文本；52次完整前向
（48端点+4重复），耗时{status["seconds"]:.2f}秒。无API、无新E5调用、无LLM训练。
保存了P/Q与native读数并复算固定f的所有部分；全部重放误差为0。

## 3. 新的数学连接：输入等价类而不只是线性拟合

若phi(T)=phi(JT)，任何只看phi的函数g均满足g(phi(T))=g(phi(JT))。
这也适用于非线性g。当前独立E5均值、query加权池、无序分布池都具有这一性质；
不是说E5看不见单条评论内部的词序，而是聚合不保留评论之间的排列。

令t+=(t(T)+t(JT))/2，t-=(t(T)-t(JT))/2，则任意不变预测p满足：

    mean_pair (t-p)^2 = (t+-p)^2 + (t-)^2.

这里的群只有两元素C2。下面是实际配置c与已冻结E5均值程序的分账：

{floor[["query", "pair_mse", "even_error", "odd_floor", "odd_floor_fraction_of_pair_mse"]].to_markdown(index=False, floatfmt=".6f")}

奇项下界分别占该反转配对MSE约18.7%/10.0%。这是该轨道实验的误差下界，
不是原始PANDORA预测误差或+.141未提取效标贡献的解释比例。
其他组件表内为计算下界使用了oracle配对均值，不是另一个已训练预测器。

这一构造结合[Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf)的集合不变性与
[Janossy pooling](https://arxiv.org/abs/1811.01900)的排列平均思想；并非原创通用
数学定理。新的经验连接在于它被应用到当前真实作者材料、同一E5输入和同一
查询函数，补上了旧S6内部block16配置实验不能直接替代的输入接口。

## 4. 哪些计算组件在运动

{focus[["query", "component", "mean_absolute_change", "rms_change", "positive", "native_r"]].to_markdown(index=False, floatfmt=".4f")}

T/F反转使10/12原生读数升高，是同一作者在另一种呈现下被模型读成更偏F，
不是作者变成另一种人格。N配置变化与native变化r=.643，但完整f变化仅.276。
完整f是固定参考下的近似函数，不是原生F，所以其组件组合还可能包含近似误差。

按共同sd(Delta f)sd(Delta native)归一化，可加的相关贡献为：

{part.to_markdown(index=False, floatfmt=".4f")}

N的最终读取部分抵消了另外两项的部分关联；T/F的组合则有另一种作用。
这为NEXT10提供了同一材料、同一干预下的组合记账，不以某一组件的r最大
直接改写完整判断。配对区间和全部组件结果保存在CSV。

## 5. 一个有顺序的输入，为什么仍未闭合读取

增加v(T)=mean_j s_j h_j，s_j从-1到1；v(JT)=-v(T)。源训练只学习原pooled
程序误差的无截距Ridge项，不改base，固定alpha=.01*n_train_views。
五折按作者，63 T/F、213 N作者，三个抽样视图；没有用Y学习或挑选参数。
同一旧828条pooled目标/预测重放误差<1e-16。新模型及held预测先冻结再看端点。

配置反转响应：T/F r=-.236，N=.050；MSE/零变化=1.222/1.228。
当前新增项明确能区分顺序，却未学到该顺序的正确条件作用，故不晋升。
源作者均值的真实效标关联为：

{src.to_markdown(index=False, floatfmt=".4f")}

这些是已使用源样本的探索性连接，不是新人格性能比较。完整teacher指标与
配对差值另存，不能只展示T/F拟合r增加而略去自评关联下降。

## 6. 最重要的逻辑修正与下一步

复制LLM的所有顺序敏感性，未必符合稳定作者测量的目标。如果目标是对一袋
记录给稳定人格位置，任意展示顺序可以作为显式呈现条件，研究不变或平均目标；
如果目标是作者的真实表达过程，时间和互动次序则应该保留。两种用途都需要，
不是删除动态或把每个模型变化都认作人格。

[输入等价类与目标理论](../../docs/GRAIN_INPUT_QUOTIENT_AND_TARGET_THEORY_V1.md)
已将上述三类用途与4W/GRAIN接合。有限反转结果留下两个可操作方向：

1. 条件运算：用独立源作者的真实original/reverse成对端点，比较同一ordinal输入
   和已有早层P/Q输入，以区分缺少配对监督与缺少上下文化表示。固定小型因子设计，
   不在这12人上训练或对Y做精度网格。
2. 目标选择：对同一材料保留呈现不变部分和顺序响应部分。先作为两个测量对象，
   不把两者合成一个“更像人格”的新总分。T/F原源+.141效标剩余仍需独立定位。

## 7. 交付与重放

方案、核心、runner、六个新单元测试已实现；49个相关测试通过。
公开汇总：`input_order_tables_v1/`；私人数值/PQ/模型仅存gitignored结果目录。
没有改发布评分器、lockbox或MEPS用途；没有push。

```bash
"../project persona/.venv/bin/python" scripts/run_grain_input_order.py prepare --output-dir results/grain_vector_mechanisms/logic_input_order_v1/paired_r2
"../project persona/.venv/bin/python" scripts/run_grain_input_order.py run --output-dir results/grain_vector_mechanisms/logic_input_order_v1/paired_r2
"../project persona/.venv/bin/python" scripts/run_grain_input_order.py analyze --output-dir results/grain_vector_mechanisms/logic_input_order_v1/paired_r2
"../project persona/.venv/bin/python" scripts/report_grain_input_order.py
```

已有目录不能重新prepare；完整acquisition再次run会直接返回，不增加调用。
没有公开的原始材料不能从公共仓库独立生成本面板，需要授权本地缓存。
"""
    (
        ROOT / "reports/grain_vector_mechanisms/INPUT_ORDER_LOGIC_REPORT_V1.md"
    ).write_text(report, encoding="utf-8")
    print(pair.to_string(index=False))
    print("Published aggregate report; P/Q replay maximum", max(replay_errors))


if __name__ == "__main__":
    main()
