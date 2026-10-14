# Persona 条件生成、盲读回与内部路径定位：执行报告 V1

任务 `CLAUDE-PERSONA-GENERATION-TRACE-20260914-01`。设计：[计划V1](../../docs/GRAIN_PERSONA_GENERATION_TRACE_PLAN_V1.md)、
[三轮研讨](../../docs/GRAIN_PERSONA_GENERATION_TRACE_SEMINAR_20260914.md)、[授权交接](../../docs/GRAIN_PERSONA_GENERATION_TRACE_EXECUTION_HANDOFF_20260914.md)。
产物根：`results/grain_vector_mechanisms/persona_generation_trace_v1/`（`pilot/`、`expanded/`）。
执行者 Claude；Codex 监督。本报告随各阶段落盘更新；数值以 `analysis/*.json` 为准。

## 1. 固定设计与身份

- 写入模型（本地、可hook）：Qwen2.5-1.5B-Instruct，本地观察器目录 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，
  权重 SHA256 `dd924a1…e10a4` 与先前 P1 面板一致；Mac MPS FP32 eager；采样 temperature .8 / top_p .95 / 最多 192 新 token / 种子固定。
- 条件：外向性、神经质两轴 × 高/低 × 两种措辞族（`named` 明确命名特质；`described` 只描述含义）+ 中性 + 两种非人格指令（长度匹配的文体指令）。
  共享后缀禁止自报人格标签。四个任务情境（日记/邮件/评论/建议）；pilot 88 篇 = 64 + 8 + 16。
- 位置对齐：每轴每族的高/低 persona 语句由备选集中挑出 **token 数完全相等** 的一对（E:named 20、E:described 20、N:named 18、N:described 21；
  expanded 另有 anchored 族 E 15 / N 17），使 persona span 在两极 prompt 中占相同绝对位置。
- 划分：discovery = {diary, email}（拟合读者、选点）；development = {opinion}（确认）；evaluation = {advice}（只评估）。
- 盲读者只见文章：TF-IDF+logistic（发现侧拟合）、冻结 GRAIN 四切片坐标（真实作者参照常模；≥48 token）+ 线性探针、
  本地模型文本-only 特质提问读出（0–9 数字期望值；上下文仅文章+固定问题）、DeepSeek Flash 盲评（独立分支）。
- DeepSeek：`.env` 中配置名 `deepseek-v4-flash` **不是**端点提供的标识；经 `/models` 核实为 `deepseek-flash`（文档：DeepSeek-V4.1-Flash）。
  默认开启思考模式并耗尽 max_tokens（首次 2 次调用返回空内容，已记账），随后以 `thinking: disabled`、temperature 0、JSON 模式运行；
  每次调用记录 model、system_fingerprint、token 用量；上限 400 次/40 万 token。只发送本实验编写/生成的合成文本。

## 2. 阶段与产物

| 阶段 | 脚本 | 产物 |
|---|---|---|
| 准备 | `scripts/prepare_grain_persona_generation_trace.py` | `manifest.json`、`materials.json` |
| 生成 | `run_grain_persona_generation_trace.py generate` | `generations.jsonl`（原文、token、persona span、审计）、`model_manifest.json` |
| 盲读 | `… read` | `readout_grain.json`、`readout_local_query.json` |
| 盲评 | `… judge` | `judge_deepseek.jsonl`、`api_accounting.json` |
| 定位 | `… localize` / `components` | `localize_effects.jsonl`、`component_effects.jsonl`、`selected_sites.json` |
| 端点 | `… endpoint` / `endpoint_read` | `endpoint_generations.jsonl`、`endpoint_grain.json`、`endpoint_local_query.json` |
| 分析 | `analyze_grain_persona_generation_trace.py` | `analysis/{audit_by_condition.csv, readers.json, control_positions.json, displacement.json, localization.json, components.json, endpoint.json}` |

（以下各节在相应阶段完成后填写。）

## 3. Pilot（88篇）：自由生成与盲读回

**审计（`analysis/audit_by_condition.csv`）**：88/88 无显式特质标签词（leak 0）、无拒答；平均 152–167 词；
192 token 上限处截断率 62%–100%（要求约150词，截断只影响结尾）；主题命中 2.2–2.6/3；第一人称份额 .01–.03。
没有按服从性筛选任何文本；"stripped"读者结果与 raw 完全相同，说明读回不依赖明确标签词。

**盲读者（只见文章；`analysis/readers.json`；null=配对块内标签翻转 2000 次）**

| 读者 | 外向性 | 神经质 |
|---|---|---|
| 本地文本-only 提问读出（无训练，32篇高/低） | AUC .703，BA .688，null q95 .641，p=.003；评估情境（n=8）AUC .938，p=.070 | AUC .602，null q95 .590，p=.036；评估情境 .750 |
| 同读者的特质选择性（在另一轴的高/低对上） | .477（不区分神经质对） | .434（不区分外向对） |
| DeepSeek Flash 盲评 | AUC .711，p=.006；评估情境 .812 | AUC .705，p=.014；评估情境 .719 |
| 同盲评的交叉轴 | .379 | .320 |
| TF-IDF+logistic（发现侧拟合） | 措辞族留出 .562/.562（n.s.）；情境留出 .812（n=8，p=.14） | .578/.578；情境留出 .250 |
| 冻结 GRAIN 四切片坐标+线性探针 | 措辞族留出 test=described .703（p=.041）、test=named .609；情境留出 .375（n=8） | .500/.578；情境留出 .750（n=8，n.s.） |

读法：模型对同一 persona 提示的自由文本，**在不见提示的条件下可被读回外向性分配**（本地读出与外部盲评一致，且不把神经质对误读为外向差异）；
神经质分配的读回更弱。两个盲评者的交叉轴 AUC 低于 .5：外向高的文本被评为更不神经质、神经质高的文本被评为更不外向——这是评者的特质协变（halo），
不是写入端的选择性证据。词法探针在 88 篇下不能跨措辞族迁移；四维 GRAIN 坐标对外向性可跨措辞族但不跨情境（n=8）。

**对照位置（`analysis/control_positions.json`）**：本地外向读出均值 高 .639 / 中性 .619 / 非人格指令 .566–.592 / 低 .527；
盲评 高 5.44 / 中性 5.13 / 非人格 5.13–5.25 / 低 4.13。低极偏离中性更多；非人格指令不产生系统偏移。

**位移（`analysis/displacement.json`；度量=条件内种子残差的 Ledoit–Wolf 协方差，收缩 .22/.30）**

| 空间 | 轴 | 均值位移 D² | 逐对中位 D² | 同条件种子差中位 D² | 高 vs 中性 | 低 vs 中性 | 非人格 vs 中性 |
|---|---|---:|---:|---:|---:|---:|---:|
| GRAIN 4切片 | 外向性 | 1.32 | 14.05 | 9.35 | .77 | 1.91 | .61 |
| GRAIN 4切片 | 神经质 | 2.31 | 12.89 | 13.30 | 2.55 | 2.74 | .61 |
| TF-IDF SVD8（发现侧基） | 外向性 | 1.70 | 1.95 | 10.79 | — | — | — |
| TF-IDF SVD8 | 神经质 | 1.63 | 3.08 | 12.10 | — | — | — |

读法：单篇层面的 persona 位移与同条件采样离散同量级（逐对 D² ≈ 种子差 D²），效应在均值上出现（16 对均值 D² 1.3–2.3，
相对 ≈14/16≈.9 的噪声基线为 1.5–2.6 倍）；神经质在 GRAIN 坐标上的位移大于外向性，而读回上外向性更强——
读回强度与坐标位移不是同一个量。非人格指令对照给出 .61 的地板。共通/特异通道份额不在此处预设。

## 4. 因果定位（teacher-forced 残差流补丁；`localize_effects.jsonl`、`analysis/localization.json`）

**设计**：seed 11 的高/低 persona 对，发现侧（diary、email × 2 轴 × 2 措辞族 = 8 对）与发展侧（opinion = 4 对）。
两极 persona 语句按 token 数对齐（`choose_aligned`），因此 persona 位置在两个提示里逐位对应。补丁 = 在低极提示的前向中，
把第 k 块输出在 persona 位置替换为高极提示同位置的状态（每 2 层一个位点，14 个位点，无 KV cache 的整段前向）。
被评分的续写是两篇自由生成各自的前 ≤96 个新 token（`X_high` 来自高极提示的生成，`X_low` 来自低极提示的生成），
记录低极提示下的逐 token 平均 log p；恢复率 = (lp_补丁 − lp_低提示)/(lp_高提示 − lp_低提示)。对照：等范数随机方向
（逐位置与 高−低 差向量同范数，加到低极状态上）；自补丁（低极自身状态，block_14）。768 次前向，231 s。

**基线（提示效应，分母）**：高提示 − 低提示 在 `X_high` 上平均 +.037（发现）/ +.023（发展），在 `X_low` 上 −.014 / −.088；
24 个分母中 21 个符号如预期，8 个 |分母| < .01。逐行恢复率在小分母处会爆炸，因此下表同时给出逐行中位数和
**合并斜率**（delta_lp 对提示效应过原点的最小二乘斜率，按对 bootstrap 2000 次的 95% CI）。

| 位点 | persona 补丁 斜率 发现 [CI] | 发展 [CI] | 随机等范数 斜率 发现 / 发展 | persona 中位 发现 / 发展 | 正号份额 发现 / 发展 |
|---|---|---|---|---|---|
| block_00 | 1.000 [.99,1.03] | 1.001 [.99,1.00] | .28 / .16 | 1.00 / 1.00 | 1.00 / 1.00 |
| block_04 | .994 [.98,1.06] | .982 [.98,1.01] | .32 / .41 | 1.01 / 1.01 | 1.00 / 1.00 |
| block_08 | .951 [.88,.96] | .965 [.95,1.05] | .41 / .40 | .95 / .98 | .94 / 1.00 |
| block_10 | .932 [.86,1.03] | .924 [.78,.95] | .32 / .39 | .91 / .85 | .94 / 1.00 |
| block_12 | .772 [.75,.87] | .842 [.66,.85] | .19 / .40 | .80 / .75 | .88 / .88 |
| block_14 | .741 [.71,.79] | .833 [.56,.84] | .18 / .24 | .73 / .75 | .88 / .88 |
| block_16 | .145 [.12,.24] | .321 [.28,.35] | .02 / .10 | .19 / .31 | .75 / .75 |
| block_18 | .133 [.11,.21] | .301 [.25,.34] | −.01 / .08 | .15 / .28 | .75 / .75 |
| block_20 | .034 [.02,.10] | −.005 [−.03,.23] | −.02 / .02 | .03 / .10 | .62 / .62 |
| block_22–26 | ≤ .004 | ≤ .012 | ≈ 0 | ≈ 0 | .38–.62 |
| 自补丁 block_14 | −.000 | −.000 | — | .000 | — |

**读法**：
- **转移边界在 block_14 与 block_16 之间（28 层中）**。到 block_08 为止 persona 位置的状态携带全部提示效应（斜率 1.00），
  block_10–14 逐渐下降到 .74–.84，block_16 骤降到 .15（发现）/ .32（发展），block_20 起为 0。发展侧（留出情境 opinion）
  复现同一形状，边界位置一致；这不是发现侧的选点偶然。
- **随机等范数对照不是零**：block_00–12 上斜率 .2–.4（发展侧 CI 自 block_04 起不含 0）。等范数随机扰动把低极
  persona 位置的信息破坏掉，续写 log p 便向高提示值移动约 .2–.4——这是"移除低极"的份额，两条续写方向一致。
  persona 补丁比随机高出 ≈.6–.7（block_00–12）、≈.55（block_14），这一差值才是"写入高极"的特异份额。
  block_16 起两者同归于 0：此后 persona 位置的状态对续写不再起作用，特质信息已经由注意力搬到后面的位置。
- **轴差**：外向性的传递更深（block_14 中位 .88 / .85，block_16 .17 / .37），神经质在 block_12–14 已降到 .55 / .52，
  block_16 为 .19 / .08。
- 自补丁恒为 0（数值恒等检查）。

**边界**：这是 teacher-forced 的提示位置补丁，回答"续写从 persona 位置读取到哪一层为止"，不回答"特质在书写位置的哪一层
被表示"；被测量的量是模型自身续写的 log p（输出评分空间），不是 GRAIN 坐标（§6 单独处理）。

## 5. 写入侧分解：注意力 / MLP / 单头（`component_effects.jsonl`、`analysis/components.json`、`span_norms.json`）

**设计**：在 §4 选出的 block_12 / block_14（最后两个仍携带 ≥.5 的位点）和 block_16（转移边界后第一个位点），
把该块**注意力子块输出**、**MLP 子块输出**各自在 persona 位置替换为高极提示的值（其余全部保持低极），以及
把该块 12 个注意力头中的单个头在 o_proj 输入处的切片替换为高极值。同 §4 的 12 对、两条续写；1140 次前向，321 s。

| 位点 | 整块残差补丁（§4，发现中位） | attn 输出 斜率 [CI] | MLP 输出 斜率 [CI] | 12 头斜率之和 | 最大单头 |
|---|---|---|---|---|---|
| block_12 | .80 | −.010 [−.02,.01] | .089 [.06,.14] | −.008 | head08 .007 |
| block_14 | .73 | .023 [−.00,.07] | .113 [.10,.14] | .029 | head07 .012 |
| block_16 | .19 | .002 [−.01,.01] | .004 [−.02,.02] | .019 | head07 .019 [.01,.02] |

**persona 位置上高/低状态的几何（`span_norms.json`，12 对逐位置平均）**：‖高−低‖/‖低‖ 从 block_00 的 .84 单调降到
block_14 的 .65、block_26 的 .45；cos(高,低) 从 .51 升到 .73（block_14）、.87（block_26）。两极语句在对齐位置上是
不同的词，因此从嵌入层起状态就大幅不同，随深度逐渐被共同上下文拉近。

**读法**：
- 在 block_12–16 上，**没有任何单个子块或单头**承担 persona 位置的差异：整块残差补丁恢复 .73–.80，而该块
  attn 输出 ≈ 0、MLP 输出 ≈ .09–.11、单头全部 |斜率| < .02。persona 位置的残差差异是嵌入 + 多层增量的累积，
  单层增量只占一小部分；其中 MLP 增量大于注意力增量，说明在 persona 位置上写入差异的是逐 token 的词法/MLP 处理，
  而不是这些位置自身的注意力读取。
- 这一结果也说明 §4 的"边界"要读成**读取侧**的边界：在第 k 块补丁时，第 ≤k 层的 persona 位置 K/V 仍是低极的；
  恢复率从 block_14 的 .74 跌到 block_16 的 .15，意味着下游位置**在第 15–16 层**从 persona 位置读取了大部分特质
  信息（第 >16 层的读取只剩 .15，第 >20 层为 0）。§5b 直接检验这一读法。
- 随机等范数对照在这些位置之所以有 .2–.4 的"移除"份额，与几何一致：差向量范数达状态范数的 .65–.84，
  同范数随机扰动实际上抹掉了该位置的内容。

## 6. GRAIN 端点：干预下重新生成的文本（`endpoint_generations.jsonl`、`endpoint_grain.json`、`analysis/endpoint.json`）

**设计**：在 block_14（§4 选出的最后一个仍携带 ≥.5 的位点；选点只用发现侧）做**仅 prefill** 的 persona 位置补丁，然后让模型
自由生成（T=.8、top-p .95、128 个新 token、seed 11/23/37）。四臂：`control_low`（低极提示原样）、
`intervention_high_into_low`（低极提示 + 高极 persona 位置状态）、`random_equal_norm`（低极 + 逐位置等范数随机方向）、
`reference_high_prompt`（高极提示原样）。对象：发展侧 opinion 与评估侧 advice × 2 轴 × 2 措辞族 = 8 对，96 篇；
1086 s。审计：96/96 无标签词、无拒答，新 token ≥93，平均 108–110 词（满足 GRAIN ≥48 token / 4 切片）。
文本层的 delta g = 冻结四切片坐标之差（臂 − 同 seed 的 control_low），度量 = §3 的 pilot 条件内 Ledoit–Wolf 协方差（收缩 .22）。
这里的 delta g 与 §4 的 delta log p 是两个独立端点，不做投影换算。

| 轴（4 对 × 3 seed） | 臂 | 逐对 D² 中位 | 对均值的 D² | 与参照臂均值方向的余弦（度量内）/ 投影份额 |
|---|---|---:|---:|---|
| 外向性 | 干预 高→低 | 11.24 | .87 | .42 / .39 |
| 外向性 | 随机等范数 | 3.88 | 1.14 | −.40 / −.42 |
| 外向性 | 参照（高极提示） | 5.29 | 1.00 | — |
| 神经质 | 干预 高→低 | 6.57 | 2.19 | .89 / .95 |
| 神经质 | 随机等范数 | 2.80 | 1.19 | .77 / .61 |
| 神经质 | 参照（高极提示） | 8.49 | 1.90 | — |
| 合并 8 对 | 干预 / 随机 / 参照 | — | .67 / .23 / .86 | 干预 .82 / .72；随机 −.11 / −.06 |

逐对"干预 − 随机"的 D² 差：外向性 +6.9 [4.4, 9.1]、神经质 +3.5 [2.4, 4.5]（按对 bootstrap），两轴均 4/4 对为正。

**盲读者在重新生成文本上（每对每 seed 与 control_low 配对之差，n=12；p=符号翻转 4000 次）**

| 轴 | 读者 | 干预 高→低 | 随机等范数 | 参照（高极提示） |
|---|---|---|---|---|
| 外向性 | 本地文本-only 读出 | +.256（11/12 正，p=.001） | +.094（9/12，p=.053） | +.229（12/12，p<.001） |
| 外向性 | TF-IDF 探针（pilot 发现+发展拟合） | +.034（p=.067） | +.021（p=.086） | +.045（p=.057） |
| 神经质 | 本地文本-only 读出 | +.011（p=.58） | −.034（1/12，p=.004） | −.001（p=.95） |
| 神经质 | TF-IDF 探针 | +.027（10/12，p=.025） | +.031（p=.067） | +.047（p=.077） |

**读法**：
- **外向性**：只替换 persona 位置在 block_14 的状态，重新生成的文本在盲读者上与直接换提示同等地读成"高外向"
  （+.256 vs 参照 +.229；随机 +.094）。这是本实验的因果主结果：persona 词的效应在 block_14 的 persona 位置上是
  **可搬运的**，搬过去后下游自由生成随之改变，而不只是 teacher-forced 的 log p 改变。
- **GRAIN 坐标**：两轴上干预都比随机移动更远（逐对 4/4，CI 不含 0），合并方向与提示效应方向的余弦 .82；
  但外向性单轴上干预的方向只与参照有 .42 的余弦、且逐对位移（11.2）大于参照自身（5.3）——补丁移动了坐标，
  移动量超过提示本身，方向只部分重合。4 对 × 4 维下这一方向估计很不稳，不能据此说"补丁沿 GRAIN 的外向方向移动"。
- **神经质**：读者端连参照臂（直接换提示）都读不出差异（本地读出 −.001），所以该轴的端点在这套读者上无信息；
  TF-IDF 探针给出弱正（+.027，p=.025）。GRAIN 坐标上干预与参照方向一致（.89），但随机臂也与参照同向（.77）：
  与 §3 一致，神经质低极离中性更远，"移除低极"本身就朝高极方向走。
- 随机等范数臂在读者上有 +.094 的外向偏移（p=.053），与 §4 teacher-forced 的"移除份额" .2–.4 同向但更小。

### 5b. 读取侧：下游位置在哪一层读取 persona 位置（`readside_effects.jsonl`、`analysis/readside.json`）

**设计（路径补丁，两次前向）**：第一次前向在低极提示中把第 L 层 persona 位置的 K/V 换成高极值，记录该层所有位置的注意力输出
（o_proj 输入）；第二次前向是干净的低极提示，但把第 L 层**非 persona 位置**的注意力输出换成第一次记录的行。
persona 位置自身的残差流全程保持低极，只改变下游位置"在第 L 层从 persona 位置读到的东西"。
同 §4 的 12 对、两条续写；每层 2 次前向，1452 次前向，398 s。
首版（`readside_effects_v0_confounded.jsonl`）直接原位替换 K/V，因为 persona 位置也在层内互相注意，替换会经由 span 内
自传播改写 persona 位置自身的后续状态（layer_00 单层就"恢复" 1.0），已作废并保留为记录。

| 位点 | 合并斜率 [CI] | 发现 / 发展 | 正号份额 发现 / 发展 |
|---|---|---|---|
| 全部 28 层的下游读取同时替换 | 1.000 [1.00,1.00] | 1.000 / 1.000 | 1.00 / 1.00 |
| read_layer_12 | .119 [.05,.16] | .138 / .109 | .50 / .88 |
| read_layer_15 | .063 [.04,.11] | .095 / .048 | .75 / .50 |
| **read_layer_16** | **.402 [.33,.56]** | **.551 / .328** | .81 / .88 |
| read_layer_20 | .191 [.09,.24] | .096 / .239 | .69 / .75 |
| read_layer_11 | .030 [.02,.09] | .062 / .014 | .81 / .88 |
| 其余 23 层 | 每层 ≤ .02 | | |
| 28 个单层斜率之和 | .91 | | |

**读法**：
- 下游位置对 persona 位置的读取集中在**第 16 层**（单层直接效应 .40；发现侧 .55）、其次第 20 层（.19）与第 12 层（.12）；
  单层效应近似可加（和 .91 ≈ 1），全部层同时替换恰为 1.000，说明 persona 效应完全经由"下游位置读 persona 位置"传递，
  persona 位置自身的后续状态不另起作用。
- 与 §4 残差补丁的形状一致：block_14 补丁保留 ≥15 层的读取（.06+.40+.19+… ≈ .7，实测 .74），block_16 补丁只保留 ≥17 层
  （≈ .2，实测 .15），block_20 起为 0。§4 的"边界"就是第 16 层这一读取事件。
- 第 0 层读取 .003：首版的 1.0 完全是 span 内自传播造成的假象。

## 7. 解释与边界（pilot 阶段）

**已建立的链条（外向性，Qwen2.5-1.5B-Instruct）**
1. system prompt 里的 persona 词改变自由生成文本，改变可被不见提示的读者读回（§3）。
2. 该效应在 persona 位置的残差流里从嵌入层起就存在，是逐 token 词法/MLP 处理的累积，不由某一子块或某一头写入（§4、§5）。
3. 下游位置在第 16 层（其次 20、12 层）通过注意力把它读走；读走之后 persona 位置自身的状态不再重要（§4、§5b）。
4. 把 block_14 的 persona 位置状态搬到低极提示里，模型的自由生成随之改变，盲读者读到与直接换提示同等的外向性偏移；
   冻结 GRAIN 坐标的位移也大于随机等范数对照（§6）。

**神经质**：读回在 pilot 上更弱，端点上读者端无信息；GRAIN 坐标位移较大但"移除低极"本身就朝高极方向走。
这一轴的因果端点在本轮不可判，需要更强的读者或更多对。

**随机等范数对照的含义**：在 persona 位置上，高−低差向量的范数达状态范数的 .65–.84，等范数随机扰动等于抹掉该位置内容；
它在 teacher-forced 上给出 .2–.4 的"恢复"，在自由生成读者上给出 +.09——这是**移除低极**，不是**写入高极**。
凡报告"恢复率"，都应同时报告这一移除份额；persona 补丁高于随机的部分才是特异写入。

**对 GRAIN 的含义（按研讨第三轮的修订）**
- delta log p（§4/§5）在输出评分空间，delta g（§6）在冻结 GRAIN 坐标空间，本报告没有把前者投影到后者；两者在
  外向性上方向一致（补丁既改变续写概率，也改变重生成文本的坐标与读回）。
- 几何重叠只说明表征兼容：本实验证明的是"该模型在该提示下的 persona 效应可搬运、可读回"，不是"人类特质迁移有效"；
  共通/特异通道份额本轮未预设、未估计。
- 4 维坐标 + 8 对的方向估计不稳（外向性 cos .42 vs 神经质 .89），扩展 270 篇只增加读回功效，不增加端点的对数；
  端点要更稳需要更多对（不同情境/措辞族）而非更多 seed。

**边界与未做**
- 单模型、单尺寸、两轴、两措辞族、pilot 8 对端点；不追踪 DeepSeek 内部；DeepSeek 只作盲评者。
- 生成上限 192/128 token 造成 62–100% 的截断率，只影响结尾；GRAIN 最低 48 token 全部满足。
- 没有按服从性筛选生成；没有 KV cache 混用（teacher-forced 用整段前向，端点补丁只在 prefill）。
- 未 commit/push/tag，未开锁箱，未发布。

### 5c. 读取侧的头分解（`readside_head_effects.jsonl`、`analysis/readside_heads.json`）

同 §5b 的路径补丁，但第一次前向只替换一个 **KV 头**（发送侧；GQA：2 个 KV 头各服务 6 个查询头），或第二次前向只替换一个
**查询头**在下游位置的输出切片（接收侧）。层 = §5b 的前三层 16、20、12；2016 次前向。

| 层（单层直接效应） | 接收侧最大查询头 [CI] | 其次 | 发送侧 KV 头 | 12 个查询头斜率之和 |
|---|---|---|---|---|
| 16（.40） | **head07 .210 [.13,.38]**（正号 .75） | head11 .060、head02 .053、head00 .036；head10 −.052 | kv1（服务 head06–11）.273 [.16,.48]；kv0 .073 | .32 |
| 20（.19） | **head05 .123 [.08,.14]**（正号 .83） | head09 .018 | kv0（服务 head00–05）.137；kv1 .027 | .15 |
| 12（.12） | head01 .038、head00 .032、head05 .032 | — | kv0 .120；kv1 ≈ 0 | .12 |

**读法**：persona 位置的读取有明确的主承载头——第 16 层 head07 单独搬运约五分之一的提示效应，第 20 层 head05 约八分之一；
两者与各自的 KV 组一致（L16.H7 ∈ kv1，L20.H5 ∈ kv0）。第 12 层的读取分散在多个头上。头级效应之和略小于层级效应
（.32 vs .40），说明层内其余部分（其它头与它们的相互作用）还贡献约 .08。这些是发现+发展 12 对上的合并估计，
未在评估侧情境上复核；头编号只对该 checkpoint 有意义。

## 9. 用量与成本记录

**DeepSeek API（`*/api_accounting.json`；只发送合成文本；密钥不进入提示/日志/报告/git）**

| 运行 | 模型（服务端 id） | 请求 | 重试 / 失败 | prompt tokens（其中缓存命中） | completion tokens | system_fingerprint |
|---|---|---:|---|---:|---:|---|
| pilot 盲评 | deepseek-flash | 90 | 0 / 0 | 30,361（1,280） | 2,981 | aeb56401… |
| expanded 盲评 | deepseek-flash | 270 | 0 / 0 | 89,242（23,808） | 8,370 | aeb56401… |
| 合计 | | 360 | 0 / 0 | 119,603（25,088） | 11,351 | |

pilot 的前 5 次探测请求（thinking 模式默认开启导致空内容）已计入 90 次并在 §1 记录；此后 payload 固定 `thinking: disabled`、
temperature 0、JSON 输出。总量在授权的有限预算内（单次调用上限 300 请求 / 20 万 token）。

**本地计算（Qwen2.5-1.5B-Instruct，MPS FP32）**

| 阶段 | 前向次数 | 秒 |
|---|---:|---:|
| pilot 生成 88 篇（192 token） | — | 1,366 |
| pilot 定位（残差补丁，12 对 × 14 位点 × 3 类） | 768 | 231 |
| pilot 写入侧分解（attn/MLP/12 头 × 3 块） | 1,140 | 321 |
| pilot 读取侧（路径补丁 28 层 + 全层；作废首版 948 次前向 283 s 另计） | 1,452 | 398 |
| pilot 读取侧头分解（3 层 × 14 头） | 2,076 | 561 |
| pilot 端点重生成 96 篇（128 token） | — | 1,086 |
| expanded 生成 270 篇（192 token） | — | 3,823 |
| 读回（GRAIN 冻结坐标 CPU + 本地文本-only 读出） | 2 次/篇 | 见各 `readout_*.json` |

## 8. 扩展 270 篇：措辞族 / 情境留出下的读回功效（`expanded/`）

**设计**：3 措辞族（named / described / anchored）× 6 情境（diary、email、opinion、advice、review、neighbor）× 3 seed；
216 篇特质文本 + 18 中性 + 36 非人格指令。角色划分：发现 = diary/email/opinion，发展 = advice，评估 = review/neighbor。
生成 3,823 s；GRAIN 冻结坐标 270/270；本地读出 1,080 次前向；DeepSeek 盲评 270 请求（§9）。

**审计**：无标签词泄漏（1 篇命中 "emotionally stable" 属低神经质描述用语，stripped 结果不变）；**12 篇含 "As an AI language model" 声明**，
其中 11 篇集中在 review 情境 seed 11（15 篇中的 11 篇，跨全部条件），另 1 篇 advice；同一 seed 在同一任务上抽到相同的开头，
是共享 seed 的副作用。主分析保留全部 270 篇；`analysis_no_refusal/` 为剔除 12 篇后的敏感性版本，主要数字变化 ≤ .02。
最短 48 词；平均 151–163 词；EOS 提前结束率 .11–.37。

**盲读者（`analysis/readers.json`；null = 配对块内翻转 2000 次）**

| 读者 | 外向性（n=108） | 评估情境 review+neighbor（n=36） | 特质选择性（另一轴对） | 神经质（n=108） |
|---|---|---|---|---|
| 本地文本-only 读出 | AUC .590，q95 .564，**p=.007** | .481（n.s.） | .473 | .507（n.s.） |
| DeepSeek Flash 盲评 | AUC .615，q95 .561，**p=.003** | .559（n.s.） | .480 | .540（p=.17） |
| GRAIN4 探针，措辞族留出 test=anchored / described / named | .660（p=.012）/ .580 / .611 | 情境留出 .512 | — | .441 / .488 / .410 |
| TF-IDF 探针，措辞族留出 | .571 / .636 / .556（p .08–.31） | 情境留出 .472 | — | .503 / .528 / .611 |

**读回功效的来源（逐情境 / 逐措辞族 AUC，本地读出 / 盲评；每格 n=18 或 36）**

| 切分 | 外向性 | 神经质 |
|---|---|---|
| opinion | **.840 / .815** | .383 / .549 |
| advice | **.704 / .765** | .407 / .642 |
| diary | .617 / .667 | .605 / .630 |
| review（5 篇声明文本） | .568 / .586 | .370 / .438 |
| email | .457 / .500 | .704 / .568 |
| neighbor | .247 / .525 | .568 / .420 |
| 措辞族 named | **.762 / .741** | .599 / .571 |
| 措辞族 described | .512 / .593 | .509 / .636 |
| 措辞族 anchored | .488 / .495 | .404 / .395 |
| pilot 同款子集（diary/email/opinion/advice × named/described，n=48） | **.682 / .719** | .580 / .680 |
| seed 11 / 23 / 37 | .590 / .630 / .546（本地） | .583 / .438 / .475 |

**读法**：
- pilot 的外向性读回在同款子集上**复现**（.68/.72 vs pilot .70/.71）；整体降到 .59/.62 不是抽样噪声，而是新加的措辞族与情境
  拉低：**anchored 措辞族**（.49/.50）和 **neighbor / email 任务**（.25–.46 / .50–.53）几乎不产生可读回的差异。
- 可读回的效应主要由 **named 措辞**（直接写出特质名与典型行为）和 **允许自我表达的任务**（opinion、advice）承载；
  described（只写行为不点名）在扩展集上只有 .51/.59。这回答了计划的停止规则："操纵是否改变表达、任务是否容许差异"——
  两者都是条件性的，不是模型对任一 persona 措辞在任一任务上都写出可读差异。
- 神经质在 270 篇上整体为 chance（.51/.54），只有 diary/email 与 named 措辞略高；anchored 反向（.40）。
- 评估情境（review、neighbor）上两轴都无区分性；review 的 seed-11 声明簇进一步压低。
- GRAIN4 探针在措辞族留出上外向性 .58–.66（anchored 为测试集时 p=.012）；情境留出 .51。词法探针不能跨族。

**位移与对照位置（`analysis/displacement.json`、`control_positions.json`；度量 = 条件内 Ledoit–Wolf，收缩 .08）**：
GRAIN4 均值位移 D² 外向性 .41、神经质 .26（pilot 1.32 / 2.31，16 对）；逐对 D² 中位 8.7 ≈ 同条件种子差 8.7；
高 vs 中性 .80、低 vs 中性 .69（外向性），非人格 vs 中性 .38。本地外向读出：高 .573 / 中性 .567 / 非人格 .54–.56 / 低 .519；
盲评：高 4.81 / 低 4.24 / 神经质条件 4.37–4.48。低极仍是离中性更远的一极。

**对 §4–§6 的含义**：因果定位与端点都在 named/described × diary/email/opinion/advice 上做，恰是读回成立的子集；
在 anchored 或 neighbor/email 上，写入侧的补丁可能同样"搬运"了 persona 状态而下游文本不变——这是下一步应检验的边界
（端点在无读回情境上应接近零），本轮未做。

## 10. 产物索引与复现命令

根目录 `results/grain_vector_mechanisms/persona_generation_trace_v1/`（约 17 MB，未提交）：

| 路径 | 内容 |
|---|---|
| `pilot/manifest.json`、`materials.json`、`model_manifest.json` | 设计、88 行材料（含 system/user 提示与 block）、本地模型权重哈希 |
| `pilot/generations.jsonl` | 88 篇生成：prompt_ids、new_ids、persona_span、finish_reason、audit |
| `pilot/readout_grain.json`、`readout_local_query.json`、`judge_deepseek.jsonl`、`api_accounting.json` | 冻结 GRAIN 坐标、本地文本-only 读出、DeepSeek 盲评与用量 |
| `pilot/localize_effects.jsonl`、`selected_sites.json` | §4 残差补丁效应（含基线、随机、自补丁）与选点 |
| `pilot/component_effects.jsonl`、`span_norms.json` | §5 attn/MLP/单头写入侧效应、persona 位置几何 |
| `pilot/readside_effects.jsonl`、`readside_head_effects.jsonl`（`*_v0_confounded.*` 为作废首版） | §5b/5c 读取侧路径补丁 |
| `pilot/endpoint_generations.jsonl`、`endpoint_grain.json`、`endpoint_local_query.json` | §6 端点 96 篇重生成与读出 |
| `pilot/analysis/*.json`、`audit_by_condition.csv` | 全部汇总（readers、control_positions、displacement、localization、components、readside、readside_heads、endpoint） |
| `expanded/`（同结构 + `analysis_no_refusal/`） | §8 的 270 篇 |

复现顺序（`PY` = 私仓 `.venv` python）：`prepare_grain_persona_generation_trace.py --design pilot|expanded` →
`run_grain_persona_generation_trace.py {generate, read, judge, localize, components, norms, readside, readside_heads, endpoint, endpoint_read}` →
`analyze_grain_persona_generation_trace.py --run <dir> [--select-sites] [--exclude-refusals]`。所有阶段可断点续跑（按已写行跳过）。
