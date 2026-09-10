#!/usr/bin/env python3
"""Publish aggregate native material-reader evidence with exact replay checks."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.report_grain_logic_review import edge_inventory
from scripts.run_grain_native_material_reader import verify
from scripts.run_grain_real_author_path import BASE, read, sha, write


def publish():
    """Whitelist public aggregate files, never copy identities, tokens or gradients."""
    out = BASE / "logic_native_material_reader_v1_r2"
    repeated = BASE / "logic_native_material_reader_replay_v1"
    manifest, other = verify(out), verify(repeated)
    replay = {}
    for name, digest in manifest["prepared_files"].items():
        replay["construction/" + name] = digest == other["prepared_files"][name]
    source = out / "analysis_r2"
    summary = read(source / "summary.json")
    for name, digest in summary["files"].items():
        replay["analysis/" + name] = (
            digest == sha(out / "analysis_replay" / name) == sha(source / name)
        )
        if digest != sha(out / "analysis_r1" / name):
            raise ValueError("Sidecar repair changed initial numerical tables")
    for folder in ("analysis_r2", "analysis_replay"):
        m = read(out / folder / "manifest.json")
        if (
            sha(ROOT / "scripts/analyze_grain_native_material_reader.py")
            != m["analyzer_sha256"]
            or sha(out / "manifest.json") != m["source_manifest_sha256"]
        ):
            raise ValueError("Analyzer/source manifest changed")
        for name, digest in m["packet_sha256"].items():
            if (
                sha(out / name) != digest
                or sha((out / name).with_suffix(".npz"))
                != read(out / name)["arrays_sha256"]
            ):
                raise ValueError("Native packet changed")
    if len(replay) != 10 or not all(replay.values()):
        raise ValueError("Ten exact construction/analysis file replays required")
    dest = ROOT / "reports/grain_vector_mechanisms/native_material_reader_tables_v1"
    dest.mkdir(exist_ok=True)
    for name in ("reader_summary.csv", "query_pair_summary.csv", "factor_summary.csv"):
        shutil.copyfile(source / name, dest / name)
    construction = read(out / "construction_summary.json")
    write(dest / "summary.json", summary)
    write(dest / "construction_summary.json", construction)
    write(
        dest / "reproduction.json",
        {
            "files": replay,
            "initial_analysis_tables_unchanged": True,
            "summary_only_sidecar_repair": True,
            "construction_manifest_sha256": sha(out / "manifest.json"),
            "source_summary_sha256": sha(source / "summary.json"),
            "new_neural_calls_for_replay": 0,
        },
    )
    readers = pd.read_csv(source / "reader_summary.csv")
    names = {
        "TF_cont": "T/F",
        "neuroticism": "神经质",
        "confirmation_seeking": "寻求确认",
    }
    table = "\n".join(
        f"| {names[r.query]} | {r.total_rms:.6f} | {r.common_rms:.6f} | {r.localized_rms:.6f} | {r.common_error_over_total:.3f} | {r.finite_derivative_relative_l2_error:.2%} |"
        for r in readers.itertuples()
    )
    report = f"""# GRAIN 与原生查询接合：共同材料，不同位置的读法

2026-09-10，第十六轮，NEXT30完成。**找到了当前Cw接口省略的一项：token位置
特异的读取作用。** 本轮是4名旧作者的有限模型机制实验，不是新的人格预测
成绩。前十五轮O/F03、宜人性与自评的关联保留，未重新训练或选取效标。

## 1. 做了什么，直接发现什么

- 旧4作者、3查询（T/F、神经质、寻求确认），8份full/E1材料的24个前缀比较。
- 正文位于问题之前；同一token前缀在因果模型中不受后面问题影响。有限精度
  缓存有4/24比较超过原2e-5最大差门槛，最高{construction["prefix_max"]:.3e}。
  初次失败记录保留；后续不是上调门槛，而是明确用同作者TF前缀固定共同状态。
- 在full材料的{construction["eligible_comments"]}/{construction["input_comments"]}条合格评论、
  {construction["slices"]}个词法切片、{construction["selected_tokens"]}个完整模型token上，
  计算四个冻结GRAIN坐标并构造共同扰动。4条短评论未参与方向，仍保留在原提示。
- 12个完整查询单元、48个方向；{summary["calls"]}次后缀调用、{summary["seconds"]:.2f}秒。
  0新模型拟合、0问卷读取、0API、0新参与者，不改旧lockbox/发布评分器。

把问题放在末尾的意义是：**材料可以共用；不同query如何读取材料必须另算。**
它依赖现有因果提示布局，不外推为所有prompt顺序、所有双向编码器的性质。

## 2. 新公式：平均方向之外，保留位置特异的读者

对同一token的GRAIN分数Gc和隐藏状态Hc，定义C=Gc.T@Hc/n，
D_k[t]=Gc[t,k]*Hc[t]，J_q为原生后缀分数对正文状态的梯度。

    T_qk = <J_q,D_k>
    T_q = C w_q + localized_q,  w_q=n*mean_token(J_q)

这个局部链式法则恒等式最大误差{summary["identity_error_max"]:.2e}；直接从保存的
梯度和方向重新相乘，误差{summary["contraction_replay_max"]:.2e}。线性均值读法的
localized严格为0；原生读法中它实际很大：

| 查询 | 完整作用RMS | 平均方向项RMS | 位置特异项RMS | 仅平均方向的相对L2误差 | 有限差分相对L2误差 |
|---|---:|---:|---:|---:|---:|
{table}

每行由4作者×4方向构成，方向已归一到本材料隐藏状态RMS。不是48个独立被试，
也不把不同候选字母/数字评分的绝对幅度当同一心理单位。约0.97–1.01的比例是
`||T-Cw||/||T||`，**不是解释人格方差的97%–101%**。正负项可以互相抵消。

这些结果指导我们保留token位置的GRAIN×隐藏配置，再让query函数读取；单纯
把每人的所有token压成一个交叉协方差矩阵不足以重建这次本地作用。
此前固定E5仿射读法的Cw恒等式仍然成立，两种读者不可混为一谈。

## 3. 共享与特异的查询读取

同一作者、同一选中正文上，三个查询的完整梯度场平均余弦：

| 查询对 | 平均余弦 | 四作者范围 |
|---|---:|---:|
| 神经质 / 寻求确认 | 0.7038 | 0.6004–0.8223 |
| T/F / 神经质 | 0.0925 | 0.0212–0.1477 |
| T/F / 寻求确认 | 0.0269 | -0.0012–0.0650 |

同一材料确实存在共享与特异的**模型局部读法**，不需要先假定每个GRAIN因子
等于一个人格词。神经质/寻求确认还共享数字输出及部分提示结构，因此这不是
心理量表相关系数或唯一语义原因。保留其作为可研究的路径重叠，而非临床解释。

## 4. 有限干预与再现

所有方向施加±0.001相对RMS，只改变合格正文的块16状态，查询/回答位置固定。
{summary["sign_matches"]}/48个导数符号与有限差分一致；最大对称半差误差
`abs((Fplus-Fminus)/2-epsilon*T)`为{summary["finite_delta_error_max"]:.3e}。
唯一符号差异也保留在私有逐方向表，不删除
小效应。有限差分总相对误差约0.49%–1.67%，说明本次导数能读出实际局部作用。

共同前缀数值固定后的评分重放最大差{summary["score_replay_max"]:.3e}，
三次预定重复的梯度最大差{summary["gradient_replay_max"]:.1f}。这是**直接后缀局部效应**，
不是自然文本修改后经过所有上游层的总效应，也不是作者真实人格被改变。

58项相关测试通过（本轮新增12项），2个已有SWIG弃用警告；未跑全仓库测试。
5个准备文件和5个分析表逐字节重放，合计10个。原分析器最后一步误读macOS
AppleDouble旁文件，导致summary生成失败；新分析器按设计枚举12个JSON修复。
原5张数值表逐字节不变，111次神经调用未重跑。所有私有ID、正文、token、
状态、梯度、切片逐行分数均未公开。

## 5. 理论决议与下一步

原生同材料接口现在是：**共同文本/切片配置 → 共享方向张量 → 查询的局部
读取场 → 分数响应**。旧边缘均值/离散度、Cw及效标关联仍各自保留。
这直接落实用户“同一文字的两个侧面，再按路径加权”的设想。

下一步不是扩展accuracy表：回到NEXT24–26的条件响应函数、能量与完整输入
接口；同时登记一个只用现有梯度缓存的后续问题：位置特异场能否压缩成少量
共同方向×位置权重。按实际作用而非矩阵总能量判断，不从问卷标签挑秩。

解释辅助分支沿用既有证据绑定rubric/prompt；本轮不启动自动精神分析、诊断
或MEPS处理。说明见[同材料解释接口](../../docs/GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)。
数学、文献及接口边界见[原生读者理论](../../docs/GRAIN_NATIVE_MATERIAL_READER_THEORY_V1.md)。

## 6. 执行入口

私有包：`results/grain_vector_mechanisms/logic_native_material_reader_v1_r2/`。
公开聚合：`reports/grain_vector_mechanisms/native_material_reader_tables_v1/`。

```bash
PY="../project persona/.venv/bin/python"
# 只重算分析，不调用模型；每次给新的name。
"$PY" scripts/analyze_grain_native_material_reader.py --name analysis_user_replay
# 若确需完整重现，在已有私有源缓存上另建目录。
OUT=results/grain_vector_mechanisms/logic_native_material_reader_user_replay
"$PY" scripts/run_grain_native_material_reader.py prepare --output-dir "$OUT"
"$PY" scripts/run_grain_native_material_reader.py collect --output-dir "$OUT"
"$PY" scripts/analyze_grain_native_material_reader.py --output-dir "$OUT"
```

不要使用旧runner内的analyze分支；它保留为第一次冻结代码的可追溯记录。
计划a632927、数值扩展/实现43b8128、共同前缀实现e5bd649，均局部提交，未push。
全系统goal持续，不把这个有限机制包称为穷尽全部理论。
"""
    target = (
        ROOT
        / "reports/grain_vector_mechanisms/NATIVE_MATERIAL_READER_LOGIC_REPORT_V1.md"
    )
    target.write_text(report, encoding="utf-8")
    (ROOT / "docs/GRAIN_SYSTEM_LOGIC_EDGES_V1.json").write_text(
        json.dumps(edge_inventory(), indent=2) + "\n", encoding="utf-8"
    )
    print(
        {
            "report": str(target.relative_to(ROOT)),
            "exact_file_replays": len(replay),
            "public_tables": 3,
        }
    )


if __name__ == "__main__":
    publish()
