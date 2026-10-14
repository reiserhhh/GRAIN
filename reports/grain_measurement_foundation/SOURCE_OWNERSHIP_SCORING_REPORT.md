# 来源归属接入人格相关计分

2026-09-13，C03/C11/C16有限机制交付。原NLI实验与事后单独冻结的语法修正分开
保存；不覆盖旧结果，不使用真人量表标签选择规则。

## 发现与制作意义

300组相反方向的本人/朋友混合文本，原NLI入口只选中本人证据162组，加入报告
动词及其主语后提高到286组（54.0% -> 95.3%）；可用观察194 -> 294组。
这是来源选择的进步，不是人格预测准确率95.3%。它补上了“先定义证据来源，再
应用固定目标函数”的实际制作步骤。全部引述、未决内容和选择引述这一行为仍在
台账中，只是不自动进入“本人自述”这个观察通道。

## 材料与实测结果

25个既有窄目标；原包装200条，written/explained两类改写各100条；另有PANDORA
既有699评论图中35条第一人称平衡引述，全部适配512-token限制。来源检测器只看
完整原文和指定引述，不看目标ID/设计答案。自然35条没有独立来源金标准。
300混合组=25目标 x 本人高/低 x 自己/朋友在前 x 三种包装，朋友内容方向相反。

| 包装 | n | NLI正确 / 未决 / 错误 | 加语法后正确 / 未决 / 错误 |
|---|---:|---:|---:|
| 原包装 | 200 | 172 / 28 / 0 | 172 / 28 / 0 |
| written | 100 | 50 / 34 / 16 | 90 / 6 / 4 |
| explained | 100 | 0 / 100 / 0 | 95 / 5 / 0 |
| 合计 | 400 | 222 / 162 / 16 | 357 / 39 / 4 |

修正使用既有Stanza1.11 EWT：引述外报告动词的直接主语优先，无明确线索才
回退冻结NLI。局部否定/情态/嵌套来源/被动或不明来源不强判。剩余4条错误全部
来自NLI回退；解析缓存显示报告动词在第0句、引述起点在第1句，未命中首句规则。
本次不再按暴露案例修规则。自然35条从全未决变为other4/未决31，是路由而非准确率。

下表MAE相对同一固定读者的“设计本人单条分数”，不是相对真人量表。
MAE只在可观察子集上算，未观测不补0。

| 包装 | 可观察 NLI / 修正 | 只选本人 NLI / 修正 | 原生MAE NLI / 修正 | 向量MAE NLI / 修正 |
|---|---:|---:|---:|---:|
| 原包装 | 94 / 94 | 94 / 94 | 0 / 0 | 0 / 0 |
| written | 100 / 100 | 68 / 92 | .048073 / .007241 | .033113 / .004807 |
| explained | 0 / 100 | 0 / 100 | 未观测 / 0 | 未观测 / 0 |

未选择来源时，原生MAE依次.157965/.185226/.166337，向量MAE
.118078/.122283/.103093。修正后6组无本人证据仍null。只选本人后的0误差是
输入与参考完全相同的计算恒等关系，不是无误差心理测量。
原生分数是候选数字的条件概率期望，不等于完整生成解析率；向量读者仍需神经
前缀产生P/Q，固定仿射函数不是全LLM独立化。

## 数学接口及继承

对有序片段T，以d(t)属于{self,other,unresolved}构造保序S(T)=[t:d(t)=self]：

\[
M_q(T)=(f_q(S(T)),f_q(T),\mathcal L(T),c(T)),\quad
S(T)=\varnothing\Rightarrow f_q(S(T))=\mathrm{unobserved}.
\]

L保留所有来源/上下文地址，c记录覆盖。若来源正确且本人片段固定，添加不被
选入的他人片段不改变第一通道；这是条件性的来源不变性，不表示引述选择没有
人格信息。也不把报告者等同心理体验者，例如“我说他焦虑”不是“我焦虑”。
本合成包使用第一人称引述，未覆盖所有角色组合。

来源/cue/content分离继承[Pareti (2016), PARC3.0 §3.1](https://aclanthology.org/L16-1619.pdf)；
报告动词/主语与分阶段归属借鉴[Muzny et al. (2017), §§5.1-5.2](https://aclanthology.org/E17-1044.pdf)。
仅适配显式规则，无作者源码复制，不冒充其新闻/小说语料精确复现或完整共指模型。

## 运行、核验与文件

根目录`results/grain_vector_mechanisms/all_plan_closure_v1/ownership_scoring_r1/`；
NLI原结果在根目录，修正在`sieve_v1/`。原文/逐条解析仅本地、gitignored。

- 新500次Qwen原生/P-Q前向168.331秒，复用旧200单条；NLI870对24.582秒。
- 修正新400次Stanza解析，35自然解析复用；首次解析及计分14.361秒，额外原生0。
- `manifest.json`及`sieve_v1/manifest.json`记录不同冻结时点和依赖hash。
- 两目录各有`source_decisions.csv`、`mixed_scores.csv`、`scoring_summary.csv`和`summary.json`。
- 根目录`polarity_responses.csv`、`order_sensitivity.csv`为原路线对照；
  `audited_route_comparison.csv`、`audited_order_comparison.csv`为两路线核验对比。
- `cold_replay_audit.json`：700端点、435x2来源决策、19,108词地址、1,200计分行、
  600完整来源台账通过，最大数值差1.67e-15；同执行者独立算术，不是第三方审计。
- `replay_tables.json`：8表逐字节重现，无新模型调用；63相关测试及Ruff通过。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/run_grain_ownership_scoring.py evaluate
"$PY" scripts/run_grain_quote_source_sieve.py --seconds 1200
"$PY" scripts/audit_grain_ownership_scoring.py
"$PY" -m pytest -q tests/test_grain_quote_source_sieve.py tests/test_grain_source_ownership.py tests/test_grain_query_event.py tests/test_grain_source_event_graph.py tests/test_grain_narrow_evidence.py
```

## 裁决和剩余

保留来源约束计分实验接口，不默认部署通用NLI归属器。语法修正是在已暴露包装
上的开发修正，95.3%不是独立泛化指标。共指/跨句引述/体验者/赞同/事实性及
独立自然注释仍开放，不继续同义包装/阈值网格。下一批回C00/C23历史Big5/H2/MCD
计划与验收证据，避免把全项目缩成来源分类修补。总goal ACTIVE；无MEPS访问、
无训练、无后台遗留、无commit/push/锁箱/私人材料发布。
