# 证据状态续研：已完成什么，尚未识别什么

2026-09-07。Codex-only，无 Claude/API 调用，无新神经 forward。

## 结论

本轮把上次研讨的“非证据权重竞争”推进为一个可更新的数值状态候选：
**当前分数 + 证据权重 + 来源依赖记录**。固定表示/核时可进一步写成可查询的一阶矩。
它不是现有神经网络或人类人格的已识别机制，而是下次实验可逐项检验的数学对象。

最重要的修正是：**所有作者统一增加证据权重，不会提高 Pearson/AUC。** 若将来
历史条件确实提高这些指标，必须进一步解释作者特异的证据选择、归属或读出变化；
只说“记忆变多，判断变强”不充分。

## 后台状态：只做了一次有界核实

2026-09-07 14:25–14:27 JST 读取状态/进程，未暂停、重启、改参或追加负载：

| 队列 | 实测状态 | 证据 |
|---|---|---|
| Mac secondary cached readouts | FIT，253/315 完成 | parent 15034，child 28326；进程实际存在且已有 CPU 时间 |
| Windows direct traits | RUNNING，18/128 完成 | parent 48952，child 6324；3 秒 CPU 增量 2.891 秒，状态文件年龄 1.8 秒 |

这是当时快照，不是最终验收。相较先前已记录 231/315 与 12/128，两个队列均有
进展。Windows 状态来自 SSH 读取远端，不以本地旧快照代替。
之后继续理论工作，不启动监控循环。完整 source/paired 分析仍等既定数据包完成。

## 实际运行的有限验算

| 数学构造 | 结果 | 含义 |
|---|---|---|
| 固定 query/KV，重复一组证据 | attention 与权重竞争公式最大差 5.55e-17 | 在声明的单 head 范围内精确；不是最终 LLM 分数定律 |
| 权重整体乘 7 | 曲线最大差 2.22e-16 | 绝对质量不从输出曲线识别 |
| 已知两端点，4 个 k 恢复权重比 | 全部恢复 .7 | 只是准确模型内的参数验算；.7 不是人格相关 |
| 线性分式交比 | 输入 1.285714，输出 1.285714；L2 反例 1.192036 | 有限函数类可以区分；单看饱和不够 |
| 相同分数 .5，不同质量 1/9 | 相同新证据后 .75/.55 | 当前分数不足以保留更新规则 |
| 相同矩，不同来源集合 | 去重更新后 .5 与 .666667 | 质量与矩也不足以识别来源重复 |
| 固定正核，3 查询 × 2 输出 | 压缩矩与逐记录计算最大差 5.55e-17 | 在此核/特征映射下可不生成语言而精确查询 |
| 所有人共同 gate=.4 | r=.894427 前后不变；MAE .4→.76 | 相关不变，绝对误差可以变差；不是任何实测作者结果 |
| 作者特异 gate 协方差展开 | 内容 0、覆盖 .1、交互 .1，总 .2，误差 0 | 外部关联的三项统计账目，不是因果份额 |
| 全体内容相同，仅 gate 与目标关联 | r=1，内容方差 0 | 高相关本身不定位内容推论机制 |

上述十行是同一确定性有限数值包内的见证，不是十项独立真人实验。另有固定种子
731 的 10 个任意向量构造检验协方差恒等式，包含在 9 项新增单元测试中。
19 项测试包括本轮 9 项和前两轮 10 项；不是全 repo 测试或统计显著性。

## 对下一步的影响

1. 归属绑定仍先于权重解释，不改变研讨确定的实证顺序。
2. 新的小面板要比较同源复述/独立机会、当前匹配/更新不同，而非只做更多 k。
3. 小矩状态与来源规则分别验收；不能用拟合一个最终分数宣布更新闭合。
4. 保留 GRAIN 的 choice/Where 通道。覆盖差异可能是有效作者选择，也可能是采样
   不均衡，先区分目标，不再把所有覆盖差异叫噪声。
5. 完成既定 source/direct/history 后再作配对理论分析。没有 hidden/KV 缓存时，
   不能从 logits 自行反演出唯一的证据权重、来源状态或神经机制。

## 产物与重播

- 理论：[GRAIN_EVIDENCE_STATE_AND_WEIGHT_IDENTIFIABILITY_V1](../../docs/GRAIN_EVIDENCE_STATE_AND_WEIGHT_IDENTIFIABILITY_V1.md)。
- 代码：`scripts/check_grain_evidence_state_theory.py`。
- 测试：`tests/test_grain_evidence_state_theory.py`。
- 本地数值：`results/grain_vector_mechanisms/evidence_state_theory_r1/`，含
  `witnesses.json`、`summary.json` 与 `source.py` 快照；遵循 ignored-output 规则。
- code SHA256：`17aa116a7e3cbe61b9a12a8eade22e5cc18bf39821f94103a1afb30006e511d3`。
- witness SHA256：`b8a83a06d50e26e5ffe890946add60ff36ad320c5210d09fbb4cfbfdbcf0b0b3`。

```bash
"/Volumes/mobile3/projects/project persona/.venv/bin/python" \
  scripts/check_grain_evidence_state_theory.py \
  --output-dir /tmp/grain_evidence_state_fresh_replay
"/Volumes/mobile3/projects/project persona/.venv/bin/python" -m pytest -q \
  tests/test_grain_evidence_state_theory.py \
  tests/test_grain_seminar_witnesses.py tests/test_grain_cue_utilization_theory.py
```

重播目录须不存在。代码只使用 NumPy 和标准库，不读模型、文本、作者标签、bridge
或 MEPS 问卷。Ruff/format/diff 检查与 19 项定向测试通过。
本轮不宣布新预测提升、不宣称理论首创、不关闭整个机制研究目标。
