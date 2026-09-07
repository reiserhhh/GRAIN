# Persona 双向机制：首轮执行与裁决

日期：2026-09-07。范围：局部计算机制，不是真人人格预测。

## 结论

**执行与完整性通过；“共享人格通路”尚未通过语义特异性检查。** Persona 表达
方向与行为识别方向可以跨任务影响输出，但最后 prompt 位置的方向混有很强的
问法依赖／答案准备成分。仍有较小的问法不变效应值得定位，不能将本轮简单解释
为模型失败，也不能将全部跨任务效应命名为人格信号。

下一步先把观察／干预位置前移到问题之前的证据位置，保留本轮作为末端对照。
**暂不扩大为真人 GRAIN 群体通路聚合**，避免聚合到通用回答偏向。

## 实际完成

| 项目 | 结果 |
| --- | --- |
| 固定模型 | Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 执行 | Mac MPS FP32 eager；不调用 API、不下载、不训练 LLM |
| 开发材料 | 六个场景，24 prompts；用于均值对比方向估计 |
| 评估材料 | 另六个场景，96 prompts；含事实识别和人格归因两种问题 |
| 条件 | 两个层、两种方向来源、三个信号剂量、两条随机方向的正负剂量 |
| 完整结果 | 120 基线，2688 干预记录，3563 forward，716.1 秒 |
| 数值控制 | self-patch／restore 最大差 0 |
| 独立重算 | 方向重建最大差 0；记录 delta 重算误差 3.62e-15 |
| 测试 | 126 项通过；两个既有 SWIG 弃用警告，无新测试失败 |

技术 smoke 另完成 375 forward、110.6 秒，未含后来补齐的 trait 归因问题；其
两个评估场景在完整面板复用，没有据此优化超参数。本轮不是未查看数据的确认试验。

## 什么确实发生了

以下是 `block_16` 的原始 signed slope，即 `(s(+v)-s(-v))/2`。s 是高倾向候选
相对低倾向候选的对数似然差；**这些数字不是 Pearson r、AUC 或人格准确率。**
两个预先固定层的全部结果都在原始报告，不依据本表选择新最优层。

| 来源 → 接收任务 | 干预响应 | 含义 |
| --- | ---: | --- |
| persona → persona 表达 | +1.0540 | 生成端正控制有效。 |
| persona → 无 trait 词的行为表达 | +1.0637 | 不限于包含 extraverted 一词的接收提示。 |
| 行为识别 → persona 表达 | +0.2348 | 反向跨任务影响存在，但还不能解释为完整人格表征。 |
| persona → 行为事实识别 | +1.0089 | 正向提问时出现明显作用。 |
| persona → 反向行为提问 | -0.6751 | 已按同一行为语义重编码，仍翻转，不满足语义不变性。 |
| persona → 外向性归因 | +1.2010 | 确实可以改变模型的人格归因候选分数。 |
| persona → 反向人格归因 | -0.8671 | 不能只报告上一行来证明共享人格机制。 |
| persona → 无关场地事实 | +0.7006 | 同一方向会影响无关的 Yes/No 判断。 |

`block_10` 的作用大多小得多，例如 persona→外向归因 +0.0009。
两个方向在 block_10/block_16 的余弦分别 .0130/.3720，只是该模型坐标内的描述；
G 与 R 的 block_16 范数为 7.5175/2.1909，不能把跨来源效应大小直接当作能力高低。
各来源都使用自己的等范数随机对照；最大单次位移约为接收状态范数的 14.0%。

## 数学上保留下来的候选

对相同材料的正向与反向问题，先按相同语义方向计分，再分解：

\[
a_{\mathrm{even}}=(a_++a_-)/2,\qquad
a_{\mathrm{odd}}=(a_+-a_-)/2.
\]

Persona→人格归因在 block_16 的 even=.1670、odd=1.0341。Even 相对随机对照为
**+.1506，场景 bootstrap 95% CI [.1417,.1600]**；行为识别→人格归因的对应
even 相对随机为 +.0380 [.0355,.0405]。因此不能说全部效应都是一个 Yes 偏置。

但 even 不是心理概念的独立因果估计，odd 也不一定是无用噪声；它们只是对问题
极性变换的两种响应，可能包含任务路由、否定处理、一般肯定倾向和概念内容。
这一分解是既有控制的事后诊断，不是预注册的新中介识别。CI 是六个共享句式
场景、固定方向下的 pointwise 区间，未纳入方向估计误差或多重比较校正。

**推动所有分数不是提高区分能力。** Persona 方向施加到全部高低材料后，正向
人格归因的间距只增加 .0336，CI [-.0021,.0741]；反向问法的间距反而减小 .1134。
这不是一种已经提高人格预测准确率的修复。

主体归属面板也是明确缺口：原生模型只有 1/6 配对呈预期方向，平均高低差
-.2279。此面板有嵌套的第一人称引用，需要同时考虑材料表达与模型归属能力，
不能把它的干预非零直接解释为对“这个作者”的人格推断。

## 下一步执行界限

1. P1B：同材料、同模型、同层、同剂量，单独比较问题前的 evidence-end 位点与本轮
   decision 位点。问题不能回写已编码的前缀；如果全无作用，先检验分布式证据位置，
   不能据此宣布模型没有概念。先做小技术面板，再决定是否扩大。
2. 将人物归属做成明确分隔的配对记录；保留原嵌套引用作为难度条件，不覆盖失败。
3. 进一步记录候选答案总概率质量／实际生成一致性。本轮只有固定候选似然，不声称
   已验证自由生成，也不把 12/12 的合成 trait 符号一致率当作真人准确率。
4. 可重复、带角色和问法控制的路径候选形成后，才进入 GRAIN 的独立文本 A 分组、
   文本 B 干预效应验证。群体均值、协方差、有效人数原语已有单元测试，但尚未实证。

这是一份已完成实验的研究裁决，不代表以上后续实验已经运行。没有开启私人量表、
MEPS 或 bridge 外部标签，也没有改旧长跑队列。本轮顺带发现旧 Mac 缓存读出按
原预算规则暂停在 297/315，未自动重启；该状态与新机制面板的完成结论分开。

## 产物与重放

- 协议：[P1 协议](../../docs/GRAIN_PERSONA_RECIPROCITY_P1_PROTOCOL.md)。
- 采集：`scripts/run_grain_persona_reciprocity.py`。
- 独立验收：`scripts/audit_grain_persona_reciprocity.py`。
- 主产物：`results/grain_vector_mechanisms/persona_reciprocity_full_r1/`。
- 重算：`results/grain_vector_mechanisms/persona_reciprocity_full_audit_r1/`。
- 完整矩阵：主产物的 `reciprocity_summary.csv` 与 `report.md`。
- 问法分解：重算产物的 `query_polarity_summary.csv`。
- 区分度检查：重算产物的 `discrimination_gap_changes.csv`。
- 主运行 identity：`a4003b0088b1f86508c83aa981d7d99a656d6beddb22b554a76c489833093cca`。

```bash
python scripts/run_grain_persona_reciprocity.py --output-dir results/grain_vector_mechanisms/reciprocity_new
python scripts/audit_grain_persona_reciprocity.py --parent-dir results/grain_vector_mechanisms/reciprocity_new --output-dir results/grain_vector_mechanisms/reciprocity_new_audit
```

模型、激活和原始运行包留在 gitignored results，本报告与代码进入版本控制。
