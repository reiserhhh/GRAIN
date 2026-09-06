# GRAIN 向量推论机制研究日志

更新：2026-09-06。研究 loop：进行中。正式 GRAIN scorer、历史 lockbox 与论文结论未被替换。

## 目标与当前裁决

研究的是：同一作者的文本内容与历史，如何被固定 LLM 编码、组合和读出；哪些可重复的机制能成为 GRAIN 的增量观察器。先检查仪器，再研究自然材料，不把合成事实分类当作人格效度。

本轮完成 VM00/VM01 的最小可重播仪器和合成干预。晚层候选方向有功能作用，但无关任务也发生分数漂移，语义专属性尚未闭环。发现了必须纳入观察器定义的计算形状效应，并用固定 masked padding 控制。详见 [VM01 报告](../reports/grain_vector_mechanisms/VM01_RESEARCH_REPORT.md)。

## 实际实现

- `grain_research/vector_mechanisms/trace.py`：固定 HF checkpoint、残差流 hooks、预回答位置、完整候选序列评分、自身替换与恢复；不更新模型权重。
- `controls.py`、`analysis.py`、`interventions.py`：可控事实材料、场景分组统计、开发侧线性 probe、低秩差分子空间和同范数随机方向对照。
- `inventory.py`、`data_views.py`：显式数据白名单、作者/时间/线程隔离、泄漏线索标记、原评论历史视图。
- `scripts/run_grain_vector_mechanism_{preflight,controls,numerics,interventions,data_views}.py`：对应运行入口；`scripts/summarize_grain_vector_mechanisms.py`：离线复算。
- `configs/grain_vector_mechanisms_v1.json`：JSON 配置，实际替代规划中的 YAML 名称。
- `tests/test_grain_vector_mechanisms.py`：合成可独立运行的单元/仪器测试。

所有原文、激活、prompt token 哈希、逐例预测和完整 source snapshot 位于 gitignored `results/grain_vector_mechanisms/`。公开报告仅含聚合数值，不提交作者 ID、私密文本、模型权重或凭证。

## 轮次与发现

| 轮次 | 问题 | 修正/结果 |
|---|---|---|
| tokenizer smoke | Transformers 5 返回 Mapping，不是裸 token list | 兼容 BatchEncoding，验证真正的 chat-template IDs |
| dtype smoke | 该 GPU/模型的 FP16 第一 block 出现非有限值 | BF16 可运行；未将 FP16 根因伪称已完全解释 |
| 读点修正 | 原文末词可能直接暴露条件标签 | 加共同 pre-question marker；输入层固定 token AUC=.5 |
| 评分修正 | 多喂一个最终候选 token 改变数值前向路径 | teacher forcing 只输入候选序列除最后一个 token 的前缀 |
| 长度审计 | 未来内容似乎改变已读前缀 | 同长度未来值不影响；不同计算长度 BF16 相对 L2 漂移约 .068 |
| 数值控制 | CPU FP32 与固定长度 GPU BF16 对照 | 测试 cases 上固定 masked length=256 后漂移归零 |
| 正式 VM01 | 192 prompts、12 场景，8 discovery/4 held-out | 模型能排序已知事实；词 bigram 也可满分，不能宣称高级推理 |
| 正式 VM01B | 16 配对目标、3,264 前向、2层×2位置×2rank×2强度 | 晚层 decision 方向有作用，但无关任务分数也受影响 |

### 当前 canonical 本地产物

- `results/grain_vector_mechanisms/vm00_local/`：字段/ID inventory。
- `.../vm01_static256_r3/`：192 事实控制、624 forward、48.56 秒，显存峰值约 3.18 GB；不是估算训练时长。
- `.../vm01_patch_static256_r3/`：修正后的干预、3,264 forward、163.67 秒。
- `.../vm01_offline_static256_r3/`：阈值/词法对照和原始输入层负控制复算。
- `.../numerics_{cpu_fp32,cuda_bf16,cuda_bf16_fullreduce,cuda_bf16_static256}/`：数值前缀不变性审计。
- `.../vm02_raw_views_r1/`：32 作者、160 原评论，仅材料准备，尚无真实人格推断结果。

旧运行目录保留为失败分析/修正溯源，尤其 variable-length `vm01_frozen_r2`、`vm01_patch_full_r2` 不用作当前 headline。manifest 绑定材料/代码/model revision；每轮保存实际 source snapshot，不能在旧目录覆盖不同协议。

## 下一队列

以下原队列的第 1 项已经完成首轮 pilot，新增下一步为：**在互不重叠历史上比较 GRAIN/TF-IDF/BGE 与 LLM 中层；修复语义主体归属和输出编码的脆弱性；再做独立证据组合和自然激活干预**。完整 VM02-07 仍有未执行项，当前不需要用户提供新 idea，也不能把理论标为已研究完毕。

1. VM02/03：8 作者技术预检后，对 24 开发作者做 2 当前片段 × 本人/匹配他人/无历史 × 行为/倾向任务。补第二 donor 规则，记录长度、时间间隔、角色与语义差异；不打开人格标签挑层。
2. 内容 token mean 与固定 marker/suffix 分开；本人历史效应不能只靠 suffix 或同一个共享历史重复进入两个 view 来证明作者稳定性。
3. 区分深层语义方向与 Yes/No 读出方向：候选编码/问题极性变换、无关事实对照、自然文本小规模干预。
4. 扩展证据合成、时间顺序、主体归属；通过开发数据决定是否扩大作者数，而不是无限调参。
5. 冻结观察器后才启动 VM05 外部联系及 VM06。MEPS46 仍只允许无标签固定条件检验，不用于训练或问卷效度。

材料限制：已准备作者历史与当前的间隔中位数 17.9 天、最大约 1,272 天。此设计不是稳定人格重测。必须对时间跨度作分层/匹配说明；同一历史导致两个当前表示更接近可能只是共享输入效应。全库扫描 17,640,062 评论，仅从规则筛选的 128 候选作者保留可用 122，再确定 8+24；不是总体代表性抽样。160 评论中 1 条 MBTI 类型词线索，需 paired masking 敏感性。

## 复现入口

研究依赖：Python、numpy、pandas、scipy、scikit-learn、torch、transformers；pytest 为测试依赖，ruff 为开发依赖。模型 pinned 为 `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。首次模型下载须明确执行，默认 local-files-only 不暗中下载。

```bash
python -m pytest -q tests/test_grain_vector_mechanisms.py
python scripts/run_grain_vector_mechanism_controls.py --config configs/grain_vector_mechanisms_v1.json --limit 16 --output-dir results/grain_vector_mechanisms/new_smoke
python scripts/run_grain_vector_mechanism_controls.py --config configs/grain_vector_mechanisms_v1.json --output-dir results/grain_vector_mechanisms/new_full
python scripts/run_grain_vector_mechanism_interventions.py --parent-dir results/grain_vector_mechanisms/new_full --output-dir results/grain_vector_mechanisms/new_patch
python scripts/summarize_grain_vector_mechanisms.py --parent-dir results/grain_vector_mechanisms/new_full --output-dir results/grain_vector_mechanisms/new_offline
```

本次运行主机：Windows `.20` 的隔离工作目录 `D:/projects/GRAIN-vector-research-vm01`，BF16 单模型，未干扰其它项目。Mac 负责原始数据、代码和离线统计。没有启动 LoRA、没有调用付费 API、没有将原始文本发到云端。

## 第二轮：自然历史、共享输入与独立作者对应

完成情况详见 [VM02/03 报告](../reports/grain_vector_mechanisms/VM02_VM03_CONTEXT_REPORT.md)。新增 `context.py`、`run_grain_vector_mechanism_contexts.py`、`run_grain_vector_mechanism_endpoint_audit.py`、`summarize_grain_vector_contexts.py`，可选依赖清单 `requirements-vector-research.txt`。新观察器支持在 GPU 内对指定原文 token span 求 mean/std，不把全序列激活搬回或保存。

| 本地目录（均在 gitignored results/grain_vector_mechanisms/ 下） | 条件数/用途 |
|---|---|
| `vm02_technical_r1` | 7 作者 × 30 = 210；1 技术作者没有 eligible prior donor，未偷借未来文本 |
| `vm02_pilot_r1` / `vm02_pilot_analysis_r1` | 24 × 30 = 720；五历史 arm、两任务、两真人当前片段及一共通 anchor |
| `vm03_shared_anchor_r1` / `vm03_shared_anchor_analysis_r1` | 480；所有作者当前文本相同，验证共享历史足以产生接近满分匹配 |
| `vm03_independent_history_r1` / `vm03_independent_history_analysis_r1` | 384；三份互不重叠单评论历史、两个共通 anchor、两任务、none |
| `vm03_independent_history_no_clues_r1` | 不重跑、不选层；排除类型词线索来源作者后的 23 作者敏感性 |
| `vm03_endpoint_audit_r1` | 48 合成条件；明确正负例、主体归属、两 system、答案代码反转 |

关键数据：block16 decision 的两当前片段匹配 raw_none=.6975、raw_same=.9293、raw_other=.9203；共享中性当前文本 raw_same=1.000。独立历史后，相同 anchor 上 block16 decision=.6446、final_norm decision=.5553；history token mean 在 input=.6063、block10=.7041、final_norm=.6697。它们不是 MBTI 分类 AUC，不能和历史 H2 AUC 混表。全局匿名作者识别、保留输入和人格效度分开。

真实任务默认全 No；明确合成正例能回答 Yes，故没有证据认定采集器只会输出 No。但 context system 会把同事行为错归作者，fact system 在这 6 材料上修正了该错误。反转 Yes/No 代码准确率 0–1/6，输出编码不稳，不将该编码作干预不变性验收。尚未证明自然文本上全 No 是缺证据还是欠推论。

新代数接口 `anchored_decomposition` 保留 reference/current/context/interaction 全部分量并精确重建原响应；它是锚点分解，不是新的正交方差定理，也不把 context 视为要删除的噪声。28 个本地测试通过；范围外完整历史测试套件未在本轮重跑。

```bash
python scripts/run_grain_vector_mechanism_contexts.py --views results/grain_vector_mechanisms/vm02_raw_views_r1/author_views.json --split technical_only --output-dir results/grain_vector_mechanisms/new_technical
python scripts/run_grain_vector_mechanism_contexts.py --views results/grain_vector_mechanisms/vm02_raw_views_r1/author_views.json --split pilot_development --output-dir results/grain_vector_mechanisms/new_context
python scripts/run_grain_vector_mechanism_contexts.py --views results/grain_vector_mechanisms/vm02_raw_views_r1/author_views.json --split pilot_development --anchor-only --output-dir results/grain_vector_mechanisms/new_shared_anchor
python scripts/run_grain_vector_mechanism_contexts.py --views results/grain_vector_mechanisms/vm02_raw_views_r1/author_views.json --split pilot_development --independent-history --output-dir results/grain_vector_mechanisms/new_independent
python scripts/summarize_grain_vector_contexts.py --parent-dir results/grain_vector_mechanisms/new_independent --output-dir results/grain_vector_mechanisms/new_independent_analysis
python scripts/run_grain_vector_mechanism_endpoint_audit.py --output-dir results/grain_vector_mechanisms/new_endpoint_audit
```

这些是执行过的设计的重播命令，不是要求后续 agent 再跑一遍。先利用已保存激活推进未完成机制，避免重复消耗。

## 第三轮：同预算比较、关系组合和计算路径

代码提交：`343def6`。详细证据见 [VM02 同预算报告](../reports/grain_vector_mechanisms/VM02_SAME_MATERIAL_BASELINES.md)与[VM03 关系绑定报告](../reports/grain_vector_mechanisms/VM03_RELATIONAL_BINDING_REPORT.md)。聚合图为 `reports/grain_vector_mechanisms/figures/relational_layer_map.png`，已渲染检查。

| 当前 canonical 目录（同一 results/grain_vector_mechanisms/ 前缀） | 实际完成内容 |
|---|---|
| `vm02_baselines_full_r1` | 24 作者 × 3 独立评论；8 reference 作者；28 条表示条件；本机缓存 BGE/E5，无下载、无截断 |
| `vm02_baselines_sensitivity_r2` | source/length/embedding-neighbor/类型词敏感性；严格 source 匹配仅 2 边，不验收为充分控制 |
| `vm03_binding_full_r1` | 288 合成关系条件，12 场景，289 forward，22.61 秒；token 多重集一致、前缀/重播误差 0 |
| `vm03_binding_analysis_r1` | lexical、独立记录相加、联合编码、全部层 probe、二阶有限差分 |
| `vm03_binding_patch_r3` | 单 marker / rank2 / 随机 /恢复，544 forward，31.05 秒，三个控制误差 0 |
| `vm03_binding_spans_r2` | 身份/行动/全部证据/全 prompt 对照，1,120 forward，60.89 秒，完整 donor/自身/恢复误差 0 |
| `vm03_binding_patch_analysis_r3` | 当前单点、span、collateral、discovery-only 阈值校准汇总 |
| `vm03_relational_kernel_r1` | 加性、乘积、组合核；固定 α=.01，8/4 场景；无 LLM 训练 |
| `vm01_patch_exactcopy_r4` | 新精确 copy 实现复跑原 VM01；16 目标、3,264 forward、167.89 秒；旧有限作用方向保留 |

主发现：

1. 自然作者匹配 E5=.7319、TF-IDF=.7107、Qwen block10=.7041；Qwen 相对 E5/TF-IDF/BGE 的配对 CI 均跨 0。不能把 LLM 中层当成自然人格推理优势。旧 GRAIN 仅做 24 维单位投影；完整 score 最少 12 单位，当前每份 1 单位不适用。
2. 同词汇但主体关系不同的四格材料中，独立编码相加 AUC=.5，联合 E5=.9883、Qwen block16 的 pre-question probe=1。词 bigram 乘积核也可=1，说明必须区分“缺少关系交互”和“模型没有语义信息”；不证明特殊人格能力。
3. 默认回答有顺序/阈值偏差；先身份后行动留出准确率 .8125、校准后 .875；逆序 .5/.6875。问题否定仍脆弱。
4. 句末单点干预很弱；block10 多 token 证据替换的目标朝向作用明显，block16 晚期替换很弱。易读位置与因果使用位置不重合。语义专属性尚未闭环，存在 collateral。
5. 修复 BF16 完整替换用 `x+(y-x)` 的舍入误差，改成直接 copy。旧 span r1 完整 donor error=.375，修正 r2=0；增加 BF16 与多 token 单元测试。Mac 40 tests passed；Windows 最近完整套件 39 passed（多出一个纯 numpy 关系核测试尚未在 Windows 执行），不是声称全仓库测试完成。

失败/修正溯源：sensitivity r1 的 flag 字典真假判断错误，r2 已修；binding point r1 发现双候选重复 forward 成本后中止，r2 改为单次 logit contrast，r3 再应用精确复制；span r1 控制失败，r2 才可使用。源快照现在包含 tests/config allowlist，避免远端仍执行旧 28-test 文件；不包含 .env 或原数据。

旧 VM01 复核的 block16 decision / rank2 / α=1：ownership 配对差 .593750 [.468750,.708333]，voluntary 1.531250 [1.364583,1.697917]；自身/恢复为 0，有限功能作用未被新代码推翻。远端原 VM01 parent 实际在 `results/vm01_static256_r3`，不是新实验所用的 `results/grain_vector_mechanisms/` 子层；首次路径检查失败没有启动模型，修正路径后完成。Mac canonical copy 仍统一放在后者。

### 第三轮命令

以下 Windows 推理脚本应在独立研究 checkout 内运行；本机 encoders 使用 `--device mps`。父目录必须是已验证、相同材料的 canonical 结果，不覆盖旧目录。

```bash
python scripts/run_grain_vector_mechanism_baselines.py --views results/grain_vector_mechanisms/vm02_raw_views_r1/author_views.json --parent-dir results/grain_vector_mechanisms/vm03_independent_history_r1 --output-dir results/grain_vector_mechanisms/new_baselines --device mps
python scripts/summarize_grain_vector_baselines.py --parent-dir results/grain_vector_mechanisms/new_baselines --output-dir results/grain_vector_mechanisms/new_sensitivity
python scripts/run_grain_vector_mechanism_binding.py --output-dir results/grain_vector_mechanisms/new_binding
python scripts/summarize_grain_vector_binding.py --parent-dir results/grain_vector_mechanisms/new_binding --output-dir results/grain_vector_mechanisms/new_binding_analysis --device mps
python scripts/run_grain_vector_mechanism_binding_patch.py --parent-dir results/grain_vector_mechanisms/new_binding --output-dir results/grain_vector_mechanisms/new_point_patch
python scripts/run_grain_vector_mechanism_binding_spans.py --parent-dir results/grain_vector_mechanisms/new_binding --output-dir results/grain_vector_mechanisms/new_span_patch
python scripts/summarize_grain_vector_binding_patch.py --parent-dir results/grain_vector_mechanisms/new_binding --patch-dir results/grain_vector_mechanisms/new_point_patch --span-dir results/grain_vector_mechanisms/new_span_patch --output-dir results/grain_vector_mechanisms/new_patch_analysis
python scripts/run_grain_vector_mechanism_relational_kernel.py --parent-dir results/grain_vector_mechanisms/new_binding --output-dir results/grain_vector_mechanisms/new_kernel --device mps
python -m pytest -q tests/test_grain_vector_mechanisms.py
```

### 尚未完成及下一执行入口

现在不能继续只在已看到的 24 作者/4 合成场景上找高分。下一主任务是**独立自然作者扩样、完整 GRAIN 多评论计分域及 joint-vs-pooled 关系通道**。沿原 VM02 的 144 discovery / 48 calibration / 48 mechanism holdout 目标准备，先做 eligibility/来源覆盖审计；不足则明确实际 N，不改为伪造完整规模。新作者不得与本次 8+24 重叠；PANDORA 历史接触仍记录，不称整个语料是 pristine lockbox。

每份 GRAIN 视图最少 12 个单位，三份互不重叠视图至少 36 条有效评论；同时必须报告 LLM 与 encoder 的 token 上限，不能为了对齐偷偷截断。若此支持不足，先给不足清单，再保留适用的一般 embedding/LLM 观察，不冒充完整 GRAIN。

之后仍需：自然事实摘要/检索/随机历史及反证更新；冻结候选后的自然激活干预；VM05 外部联系；VM06 MEPS 无标签与 Essays/X 冻结复测；VM07 机制图和规范。当前有可执行路径，不需要用户提供新 idea，也没有将整个理论标为完成。

## 第四轮：独立多评论开发与参照支持

代码提交：`1d68c8b`；未 push。报告与协议作为随后独立文档提交。

2026-09-06。上一 goal turn 分类为 progress；本轮亦完成新数据、模型执行、统计
与修正，不是状态复述。无新训练，无付费 API，无 MEPS 问卷读取。

完整结果：[VM02 多评论报告](../reports/grain_vector_mechanisms/VM02_MULTICOMMENT_DISCOVERY_REPORT.md)。
图：`reports/grain_vector_mechanisms/figures/multicomment_discovery.png`，已渲染检查。

- 新 240 人，8,640 条 raw 评论；与原 32 人零重叠。144 development 分为 48
  reference /96 comparison。48 calibration、48 mechanism holdout 仅准备文件，
  未进入模型和效应分析。所有主体标签仍未用于选择。
- `vm02_multicomment_cohort_r1`：54.4 秒流读全库，745/1024 eligible；
  每作者时间跨度中位数约 3 年，不能称短期重测。
- `vm02_multicomment_base_r2`：完整 GRAIN + reference-only 词汇/风格。
  12 条视图全部超出旧半径；没有绕过拒绝规则。r1 交错曾重拟合词表/
  scaler，r2 固定 chronology reference 变换，隔离分组方式。
- `vm02_multicomment_bge_r1`：Windows FP32 6,916 文本、191 秒；
  `vm02_multicomment_e5_r1`：Mac FP32 相同文本、1,186 秒。
- `vm02_multicomment_qwen_r2`：864 条件、865 forward、382 秒；
  原文 max1993 tokens、padding3072、GPU peak4.31GB；全部已拷回 Mac。
  pre-question 与 replay 误差 0。r1 在 preflight 被嵌套引文重复子串挡住，
  r2 改为明确 span 定位，无作者删减。Windows 另一处 GBK 默认解码已修 UTF-8。
- `vm02_multicomment_analysis_r3`：62 条件完整汇总。Qwen10 正文 .9499，
  E5 独立 .9473，配对差 .00256 [-.01103,.01580]；Qwen16 判断 .6507、
  final norm .5951。名义 Qwen–BGE 小幅差不作多重校正后优势。
- `vm02_multicomment_support_r2`：36 条时仅 26/144 在旧域内。标准有限总体
  均值二次型公式与真实向量 Monte Carlo 相符，解释观察量导致的一部分距离
  膨胀；不证明完整 score 的拒绝完全由误差造成，不直接减去所谓噪声。
- Mac 与 Windows 机制专项测试均 **48 passed**；scoped ruff 与 diff check
  通过。完整历史仓库测试未在本轮重跑。

复现入口（新目录，不覆盖 canonical）：

```bash
python scripts/run_grain_vector_mechanism_cohort.py --data-root '/Volumes/mobile3/projects/project persona/data_sets' --exclude-views results/grain_vector_mechanisms/vm02_raw_views_r1/author_views.json --output-dir results/grain_vector_mechanisms/new_cohort
python scripts/run_grain_vector_mechanism_multicomment.py --views results/grain_vector_mechanisms/new_cohort/discovery_views.json --stage base --output-dir results/grain_vector_mechanisms/new_base
python scripts/run_grain_vector_mechanism_multicomment.py --views results/grain_vector_mechanisms/new_cohort/discovery_views.json --stage e5_large --device mps --output-dir results/grain_vector_mechanisms/new_e5
python scripts/run_grain_vector_mechanism_multicomment.py --views results/grain_vector_mechanisms/new_cohort/discovery_views.json --stage bge_large --device cuda --group-plan-dir results/grain_vector_mechanisms/new_e5 --output-dir results/grain_vector_mechanisms/new_bge
python scripts/run_grain_vector_mechanism_multicomment_qwen.py --views results/grain_vector_mechanisms/new_cohort/discovery_views.json --output-dir results/grain_vector_mechanisms/new_qwen
python scripts/summarize_grain_vector_multicomment.py --views results/grain_vector_mechanisms/new_cohort/discovery_views.json --parent-dirs results/grain_vector_mechanisms/new_base results/grain_vector_mechanisms/new_e5 results/grain_vector_mechanisms/new_bge results/grain_vector_mechanisms/new_qwen --output-dir results/grain_vector_mechanisms/new_analysis
python scripts/run_grain_vector_mechanism_support_audit.py --parent-dir results/grain_vector_mechanisms/new_base --output-dir results/grain_vector_mechanisms/new_support
```

下一入口不再是扩同一作者匹配 leaderboard：执行 VM03 最多 12 个 hash 固定
development case 的自然原文/事实摘要/冻结检索/随机历史与主体、否定、矛盾
更新；先审核事实保真。之后 calibration 固定两个候选、自然激活干预，再开
VM05 量表链接与 VM06/07。保留 Qwen10 内容与 Qwen16 判断的既定机制位置，
不因 block5 在 discovery 最高而重新选层。仍有可执行工作，goal 保持 active。

## 第五轮：自然记忆保真和判断读出

2026-09-06。实现提交 e9cc7b6，未push。本轮仍为progress。
报告：[VM03自然记忆](../reports/grain_vector_mechanisms/VM03_NATURAL_MEMORY_REPORT.md)；
设计：[bounded protocol](GRAIN_VM03_MEMORY_AUDIT_PROTOCOL.md)。

从96 comparison作者按新hash选6人，历史12条、后来2条，每个数据选择不读
量表值。先运行6次Qwen事实摘要并冻结源文审核，再运行数值读出；
没有独立新作者确认或心理量表准确率。

| Canonical目录，results/grain_vector_mechanisms/下 | 完成内容 |
|---|---|
| vm03_memory_materials_r1 | 12cases、96人donor池、24条E5检索query、预读出语义审核 |
| vm03_memory_summaries_r1 | 6次greedy生成，106.5s；严格可用1/6、单括号修复后2/6 |
| vm03_memory_trace_r2 | 312自然+384虚构事件，697forward，302.5s |
| vm03_memory_analysis_r3 | 作者配对/位移/半历史交互、引用和语义分开审计 |
| vm03_memory_statement_r1 | 等价陈述式诊断384条件，385forward，167.0s |
| vm03_memory_wording_analysis_r2 | 配对问法效应；3种pre-question表示差均0 |

主要结果：

- 12可解析事实句中7句符合引文规则、4句通过一位源文审核者的语义核验。
  原文引用正确仍可能把行动者、接收者、时间状态写错。未达到schema的4份不伪造
  成“没有证据”；全部作者和2位可用作者的结果分开。
- 自然具体社交行为任务全部No，但历史仍产生明显上下文化位移；未获得自然
  truth，不能说准确率为0或100%。同历史逆序使倾向margin +.8229，
  2/12回答翻转；n=6、多重探索，不宣称普遍人格不稳定。
- 人工事件准确率 .8411、AUC .9428；陈述式 .8750/.9579，
  Δaccuracy +.03385 [−.00260,.07292]。局部改善伴随简单肯定例退化，
  否定后更正为肯定在两种问法均只50%。不继续用同一故事调到满分。
- block10/16问题前状态在两问法严格相等；这定位了任务suffix的计算边界，
  不是额外证明模型之前已经正确理解，更不是证明人类心理因果。

修正溯源：

1. trace_r1在第145条件被负控制停止。raw当前文本未变，但表情符号在单/
   双换行下末token分别549/39524，token数同为108。统一CURRENT_END、
   增加完整token ID preflight，从头运行r2。r1已标FAILED_CONTROL并两端同步。
2. analysis_r1把字符串case索引用于numpy，运行失败；显式feature_row修复。
   r3增加分析自身source snapshot/hash后重播，26个CSV与前版逐字节一致。
3. Mac/Windows专项测试均55 passed，scoped ruff、格式化和diff check通过。
   完整历史测试未重跑。所有模型进程已结束，输出已拷回Mac。

重播入口：使用新输出目录，模型运行用隔离Windows，E5检索用Mac缓存。
每个脚本的 --help 给出必需参数；按顺序执行：

1. scripts/run_grain_vector_mechanism_memory_prepare.py：--views discovery_views.json，--e5-parent 上轮E5目录，--output-dir 新材料目录。
2. scripts/run_grain_vector_mechanism_memory_summaries.py：--sources summarization_sources.json，--output-dir 新生成目录。
3. scripts/run_grain_vector_mechanism_memory_review.py：--sources、--generated、--decisions、--output semantic_review.json。
4. scripts/run_grain_vector_mechanism_memory.py：--materials、--generated、--review、--output-dir；另跑 --controlled-only --question-style statement 到新目录。
5. scripts/summarize_grain_vector_memory.py：--parent-dir trace目录、--review、--output-dir。
6. scripts/summarize_grain_vector_memory_wording.py：--original trace目录、--statement 复测目录、--output-dir。

semantic_decisions.json必须在任何下游读出前，按完整引用源文重新审核；
不能自动照搬本次评级到新生成，也不把阅卷agent的一次判断叫专家金标准。

### 下一轮的可执行队列

先作小范围证据范围审计：区分“文本报告的一次行动”“文本本身正在实施的
言语行为”“一般倾向推论”。固定材料与task定义后，用已知操纵/先验语义
标注校准两种预定读出，不把模型自身答案作truth。还要检查1.5B观察器的
能力限制，不能把它等同于用户所举ChatGPT研究。

随后在48 calibration固定端点（不重选block10/16），推进自然matched
span干预；独立mechanism holdout与VM05外部联系、VM06 MEPS无标签/
Essays/X、VM07 atlas仍未做。本轮没有访问这96人的模型结果或心理标签。
仍有具体可执行问题，goal不完成、不blocked，也不需要用户现在提供新idea。

## 第六轮：证据范围与自然内部状态干预

2026-09-06。实现初次提交28bfae5，BF16强度修正7eef40a；
本轮仍为progress，未push。
文档：[证据范围](../reports/grain_vector_mechanisms/VM03_EVIDENCE_SCOPE_REPORT.md)、
[自然状态干预](../reports/grain_vector_mechanisms/VM04_NATURAL_PATCH_REPORT.md)、
[VM04协议及修正](GRAIN_VM04_NATURAL_PATCH_PROTOCOL.md)。

### 完成产物

以下目录都在results/grain_vector_mechanisms/：

| Canonical目录 | 完成内容 |
|---|---|
| vm03_scope_materials_r1 | 48 calibration当前原文、先验语义编码、288合成范围控制 |
| vm03_scope_trace_r1 | 480条件，481forward，36.016s，重播/prefix误差0 |
| vm03_scope_analysis_r2 | 两问法、冻结阈值、LOO分析、8未决编码全部256种分配敏感性 |
| vm04_natural_patch_r1 | 6目标作者，12当前，24donor配对，2496干预行/2640forward，404.515s |
| vm04_natural_norm_repair_r1 | 384 endpoint随机控制修正，672forward，102.859s |
| vm04_natural_patch_analysis_r2 | 替换修正行后的作者cluster分析，其他原始干预不变 |

正文/标签访问：48 calibration作者的第三视图第一条评论已打开并进入模型。
它们不是未使用样本了。语义编码仅源文证据，不是心理量表值。仍有48
mechanism-holdout作者未打开；没有用MEPS或问卷，本轮不更新心理预测headline。

### 得到的结论

- 当前提供建议与报告过去建议必须分开。自然48条有8条明确当前建议、8未决；
  过去建议没有正例。原始答案偏Yes；合成阈值在40明确样本上BA=.6719，
  但accuracy=.850只略高于多数类.800，召回仅3/8；自然LOO BA=.6563。
- 主问法自然margin AUC=.709 [.478,.920]，不确立稳定自然判别力。
  次级definition AUC=.811但LOO BA=.500，不允许只挑排序优点。
- 源CSV本来已有类型词占位符和少量mojibake。“raw不变”不表示最原始Reddit文本。
- 自然上下文差在rank4仅保留36%至40%能量。完整block16判断状态对倾向margin
  有小的donor定向作用，+.1771 [.0625,.2813]；几何候选多数不强于随机。
  不是共享人格轴、人格准确率，亦不否定所有上下文计算。
- 数值修正：precast等强度不能保证BF16 postcast等强度。判断位置随机重复
  均值范数失配最高24.66%；全部384行按原方向/种子缩放后，逐行误差最多
  0.375%，原基线和192候选输出严格重播0误差。未经修正版endpoint随机对比
  降为provisional，不能继续作为主结论。

### 工程与复现状态

模型任务均完成，产物已拷回Mac。两端61专项测试通过；完整历史测试未跑。
所有模型任务使用既有Qwen缓存，没有训练、下载、收费API或服务重新配置。
私有材料、source coding及96状态缓存不进git。

两个轻微启动问题已修：Windows pytest需要仓库cwd；修正脚本最初命名
不在source archive allowlist内，未启动模型即报文件缺失，已改为
run_grain_vector_mechanism_natural_norm_repair.py并用新zip同步。
这没有改变实验材料或产生可纳入分析的失败模型结果。

### 下一轮具体队列

1. 保留两种候选：任务敏感方向与历史读取路径。用已有开发材料评估
   局部导数/有限差分是否预测低剂量内部效应，与几何方向和随机方向比较。
   不把从同一输出梯度优化出的控制方向当独立心理发现。
2. 检查自然历史的影响是否经“当前正文更新”以外的后续注意力路径进入判断。
   未换历史状态的单位置patch不是完整中介分解，不能把低恢复率直接称失败。
3. 冻结端点、数值控制与选择规则，再开独立mechanism holdout。
   外部量表链接VM05、第二观察器、VM06 MEPS无标签/Essays/X、VM07 atlas仍待做。

GRAIN计分器和既往MCD/VHMCD/SUICA实证没有被替换或推翻。该循环目前仍有
明确可执行问题，不需用户补新idea，不应把goal标为complete或blocked。

## 第七轮：局部敏感性与历史访问路径

2026-09-06。实现提交86df977；没有训练、下载、付费API或push。
见[协议](GRAIN_VM04_SENSITIVITY_PROTOCOL.md)和
[敏感性报告](../reports/grain_vector_mechanisms/VM04_SENSITIVITY_REPORT.md)。

仍为6名discovery作者、12当前评论，不是独立确认。固定Qwen revision、
block10/16、rank2/4、BF16 eager、pad1536；Windows .20完成全部计算。

| Canonical目录（results/grain_vector_mechanisms/下） | 可复现内容 |
|---|---|
| vm04_sensitivity_capture_r1 | 24 own-history/task条件，48状态/梯度缓存；96 forward含48 backward；151.422s |
| vm04_sensitivity_interventions_r1 | 2880行干预，2976 forward，525.812s；恢复、自替换、旧条件重播误差0 |
| vm04_sensitivity_analysis_r1 | 作者cluster汇总、等范数随机对照、局部预测、符号剂量、分区重算 |

总3072 forward，3744512输入tokens，4718592执行tokens，677.234s。
采集GPU峰值4.52GB；冻结参数、仅对中间状态求导，CPU卸载保存张量。
所有384随机配对实际post-BF16范数误差低于1%，最大0.775%。
原区域CSV整体范数float32的微小累积误差由冻结缓存float64重算修正，
不覆盖原始文件，不重跑模型；所有互斥区域比例和通过1e-10检查。

### 新证据与边界

- rank4梯度能量约.739/.929，但与几何子空间重合度仅.00542/.00629。
  这是坐标及标量输出条件下的子空间量，不是心理方差比例或因子数。
- 自然敏感投影仍未稳定超出随机；放大至几何等强度后目标绝对margin变化
  2.672/2.156，非目标也达1.146/1.906，且局部线性预测明显失准。
  不将优化输出梯度的可控性称为人格方向发现。
- block10小预算0.2%的历史正/反向操作：目标+.8125/-.8229，
  一阶预测+.8085/-.8085；MAE约.139/.143。相同目标方向作用于当前
  建议任务为+.0521/-.1354，并非完全无副作用。
  这支持历史位置尚有可访问计算路径，不证明必要性、完整中介或心理因果。
- 后置问题/回答状态的梯度较大不意味着模型忽略正文，因为这些状态已上下文化。

专项Mac63 tests已通过；最终源码通过source_sensitivity_final_c.zip同步到
隔离Windows目录，同一专项63 tests通过（11.35s）。
完整历史测试未重跑。私有梯度缓存、源文和行级作者材料不进git。

### 下一裁决

结束同一6作者上的自由方向探索，先冻结独立作者的有限确认及答案极性控制，
再推进VM05同材料外部联系和第二观察器。48 mechanism holdout尚未打开；
MEPS问卷和用途边界不变。已有代码仍可运行，不需要用户的新idea；goal保持active。

## 第八轮：独立历史访问确认与VM05容量盘点

2026-09-06。先提交协议4d69362，再实现d443c7f、预检修正7e1d2b7；
分析器端到端测试422f352。见
[独立报告](../reports/grain_vector_mechanisms/VM04_INDEPENDENT_ACCESS_REPORT.md)。

48机制保留作者已全部打开指定材料，4人超长整组排除，44完整执行。
canonical vm04_independent_access_r2：1777 forward含176 backward，1056行，
690.703秒；恢复、自替换、梯度前向/历史前缀重播0误差。
总范数随机匹配最大相对误差6.70e-6。220私有缓存已拷回、manifest哈希通过。
vm04_independent_access_analysis_r1是统一作者级分析；不是心理量表结果。

第二当前相对随机定向margin优势+.5909 [.5284,.6556]，Holm p约.00010。
说明源当前定义的历史方向能影响另一当前文本的相同任务，而非在那里再优化。
但普通/交换代码语义控制仅5/8、4/8，反转1/8，未达预定条件。不能把交换
端点反向效应当作已识别纯答案偏置，也不能把机械确认升级为人格特异性。
非目标仍有明确效应。停止同样本prompt/方向搜索，第二观察器能力预检优先。

r1在模型前向前因tokenizer Mapping返回类型差异停止，0调用；保留
FAILED_PREFLIGHT并修复兼容处理，无门槛/条件改变。Mac与最终同步Windows
专项68 tests通过（Windows12.62s），完整历史测试未跑，无训练/API/push。

额外完成vm05_membership_inventory_r1（代码/说明873b192）：仅读官方文件
user_id，240作者有22 Big5-only、207 MBTI-only、10 strict bridge。
不读心理值，不将240错误当成Big5样本量。下一步正确来源扩样、同预算采集，
不能用小bridge承担主H2。Mistral7B本地3权重文件14.50GB存在，未加载；
Mac24GiB/Windows可用6.9GB需先资源预检，不贸然全精度反向。
具体见[GRAIN_VM05_RESOURCE_AND_COVERAGE_NOTE](GRAIN_VM05_RESOURCE_AND_COVERAGE_NOTE.md)。

仍需第二观察器、VM05外部联系、VM06 MEPS无标签/Essays/X、VM07atlas。
本轮没有需要用户补新idea的阻塞，goal保持active。

## 第九轮：第二观察器与完整来源取材

2026-09-06。协议cd99ee7/0405492先于执行；实现d32984b/a972f55，
独立审计实现830395f。无模型训练、下载、付费API、服务重配或push。

1. Llama和Qwen实际生成各216prompt完成，分别101.142s和82.485s。
   Llama普通/交换代码7/8、4/8，反转5/8；Qwen5/8、4/8，反转1/8。
   均未取得代码不变语义资格；不改VM04的机械确认，也不赋予人格解释。
2. Llama严格事实.875的全部24错例都是正确No后附加解释。事后首句诊断1.0
   不能改主门槛，更不保证解释真实。Qwen实际生成.8958与margin .8802
   的差异全部来自三个margin=0；16个旧代码margin重播误差0。
3. 全来源一次扫描17,640,062行、112.091秒。10,090来源作者中7,343
   合格：636 Big5-only、6453 MBTI-only、254 bridge，共264,348评论。
   每人36评论/3时间视图；未读心理值。Mac峰值RSS约5.30GB。
4. 独立核验来源、fold、文本hash、时序和讨论分离通过；源作者无重叠，
   跨作者评论/归一化正文重复0。旧240人中239进入，不是新lockbox。
   人均跨度中位约1170天，不能偷换成同时点trait稳定性。
5. 源群间有2041个共享讨论，其中1096连接bridge与训练来源。人格线索
   覆盖bridge作者55.9%、Big5-only23.0%、MBTI-only17.9%；它们仍是
   自动线索，不是已确认标签报告。后续须做消融及讨论重叠敏感性。

更正：375 bridge比官方交集364多11人，但这11人也不在Big5主文件中。
早先即时更新中“会漏入Big5-only”的推断错误；没有实际训练污染证据。
显式bridge优先排除是防护加强，Big5-only来源1037不变，没有按旧盘点拟合头。

Canonical目录均在results/grain_vector_mechanisms/：

| 目录 | 状态/内容 |
| --- | --- |
| vm03_second_observer_llama_r1 | 完成；原合成回复/服务模型identity |
| vm03_second_observer_qwen_r1 | 完成；原生成回复及first margin |
| vm03_second_observer_analysis_r2 | 主结果、逐family、配对及独立格式诊断；r1保留 |
| vm05_membership_inventory_r2 | 仅读ID的来源盘点；r1保留 |
| vm05_full_source_cohort_r1 | 完成；来源/排除/fold台账与私有原文 |
| vm05_full_source_audit_r1 | 完成；独立hash、来源、时序、线索与共享讨论核验 |

复现入口为run_grain_vector_mechanism_capability.py、
summarize_grain_vector_capability.py、run_grain_vector_mechanism_external_cohort.py、
summarize_grain_vector_external_cohort.py；协议文件列明固定参数。每次新建输出目录。
正式解读见VM03_SECOND_OBSERVER_REPORT与VM05_FULL_SOURCE_MATERIAL_REPORT。
Mac专项74 tests通过；最终source_vm03_vm05_final_c.zip同步Windows后74 tests
通过（11.78s）。scoped ruff与diff检查通过；完整历史测试未重跑。

下一执行按[GRAIN_VM05_SAME_EVIDENCE_DESIGN_V1](GRAIN_VM05_SAME_EVIDENCE_DESIGN_V1.md)：
同预算表示/来源统计头/bridge后验。另保留事实与倾向的代码因子实验、
VM06获准用途内异域复测和VM07机制图谱。代码能力不足不能否定独立统计头
所检测的信息；这两个对象必须分开。不改GRAIN正式计分、不读MEPS问卷。
当前仍有明确可执行问题，goal保持active，不需要用户新增idea。

## 第十轮：VM05技术资格、源头读出与全量向量启动

2026-09-06。04a1dbb/2195fa8实现预检与observer，aa51b06在全量采集前
冻结资源/同材料/无损缓存规则；e9c0109在源头标签转换前实现嵌套统计头。
f6b759c/f90ac25补同一冻结算子的遮蔽与独立配对复算。

- 7,343作者中7,266三版本三模型完整适用：624 Big5-only、6393 MBTI-only、
  249 bridge；没有截断后拼成“同材料”。
- Qwen/BGE/E5各12技术作者通过，跨机288个token签名一致、数值重播0误差。
  E5从Mac已有缓存复制，权重/metadata SHA相同；没有模型下载或训练。
- 原预计11.582GB缓存超预算，r1不合格；无损别名逐值还原通过，r2预计
  6.462GB。24固定分片、三模型顺序采集，单队列最多8h、每片最多30min。
  Windows canonical vm05_full_vectors_r1正在运行，不能报告为已完成。
- 正式冻结GRAIN all36 ready只有1772/7266，12评论视图几乎全部拒绝；
  未放宽门槛。99旧参照/168旧E1作者全部在MBTI-only，需补排除敏感性。
- 18个源头条件及18项配对复算完成，见
  [VM05报告](../reports/grain_vector_mechanisms/VM05_PREFLIGHT_AND_SOURCE_REPORT.md)。
  Big5 sparse raw/cue/placebo pooled r=.1370/.1386/.1368；配对差均未明确。
  旧V7词法投影48列中间特征PCA16 r=.0568，无额外PCA=.0819，差CI跨0；
  两者弱于稀疏参照，不是对完整当前GRAIN理论的否定。
- 90个已保存fold模型独立重播，最大绝对差1.11e-16；没有重拟合或新标签访问。
  canonical vm05_prediction_replay_r1，稀疏pipeline无PCA亦核验。
- 中间特征MBTI AUC raw→cue：EI .5876→.5699，SN .5958→.5845，
  TF .6776→.6790，JP .5706→.5659；placebo几乎不变。TF为值得继续检查
  的线索，不是已识别人格方向。EI/SN BA@.5仍近.5，不能只报AUC。
- 原始data已有TYPE_MENTION，8944命中评论含此标记，不能声称未处理原文。
  词汇遮蔽也不保证全部报告/语义泄漏清除。共同讨论与参照暴露仍需控制。
- 单次重选参数的置换r=-.0380，不足以关闭完整null。常数训练均值的
  pooled OOF也可为负：真实标签-.0850，因此不能把任意负r当“反人格”。

Canonical新增：vm05_preflight_r1、vm05_technical_{qwen,bge,e5}_r1、
vm05_qualification_r2、vm05_source_audit_r2及报告列出的18个源头目录。
全部私有原文/标签/逐人预测/向量与head均gitignored。
Mac专项88 tests通过；Windows冻结采集源码82 tests通过。完整历史测试未跑。

运行纪律：**不要把当前Mac研究源码同步覆盖Windows的冻结采集目录**；其
queue校验代码指纹。运行根为D:/projects/GRAIN-vector-research-vm01，
结果results/grain_vector_mechanisms/vm05_full_vectors_r1。原队列PID28172；
恢复时先核对queue_status及子shard/run_status，PID只是本次记录不是恒定ID。
预算到期后使用原同参数命令加--resume-completed，保留原源代码与三技术父目录；
FAILED_STOP或PARTIAL_RESOURCE_BOUND不能直接重复启动/覆盖，应先核对具体原因。

下一执行：完整向量/源头头→旧参照排除/共享讨论/完整置换→冻结头再bridge。
其后仍有代码语义实验、VM06许可范围跨语料、VM07atlas。本轮不是理论完成，
也没有需要用户新idea的阻塞。goal保持active，没有push。

本轮末全量队列的shard_00三个观察器各303人均已完成，随后进入shard_01。
首批1829文件已备份回Mac的同名vm05_full_vectors_r1目录，三份manifest及
全部909份向量cache SHA验证通过；local_backup_status.json明确whole_queue_complete=false。
Mac目前只是首批副本，不能据此尝试完整源头observer head。Windows原队列继续运行。

## 第十一轮：来源敏感性、完整置换与OOF尺度诊断

2026-09-06。规则8eb26e9/0c5414b，独立审计235cea7/e558d1f，
数学分解c619693。详见VM05_SOURCE_CONTROLS_AND_NULL_REPORT。

- 102条件串行批次运行，MBTI的84条件完成（6294作者、旧参照99人已排除）；
  Big5 raw六控制完成，其余12个cue/placebo控制仍运行。不是102全完成。
- 独立重建85个已完成条件的2040次成员选择，60个随机/purge对照逐24处
  人数匹配；420个MBTI保存fold模型重播最大误差1.11e-16。
- 同人数raw purge比随机平均的AUC：EI -.01128 [-.02041,-.00142]、
  SN +.00184 [-.00963,+.01293]、TF -.00436 [-.00720,-.00116]、
  JP -.00441 [-.01132,+.00187]。cue+purge的TF AUC=.67123，
  但不称纯人格信号；EI/SN BA仍约.5。
- Big5 raw purge r=.07990 vs随机五seed均值.10966，差-.02976
  [-.05255,-.00783]；mean-fold差仅-.00463。原参照r=.13697。
  精确协方差分解查明主要差异不只是均值偏移，还包括各折预测尺度/相关权重。
  没有改test预测、没有用居中后的诊断值取代主成绩。
- 旧中间特征Big5完整199置换：r=.05679，经验p=.010；seed731与之前
  完整重拟合完全相同。12个MBTI retained null共2388次全部重算通过，
  p均触及.005分辨率下限。15条件家族Holm受限于.075，不能偷换为p=0。
  三个Big5 sparse完整null仍在执行，未用弱中间特征null替代。
- Windows冻结aa51b06代码继续正常采集，最新核对Qwen shard_02仍有作者增加；
  Mac备份shard_00/01每模型606人，6份observer shard完整校验。
  shard01归档两机SHA=0795ae6927c45d9fb39fae46bd43db0ced228e94a365128b375f763f76936523。
- Mac专项101 tests通过；完整历史测试未跑。没有语言模型训练/下载/API/push，
  bridge心理值及MEPS问卷未打开，不改变发布GRAIN分数。

复现入口与canonical目录在新报告第6节。正在运行的本地两个批次：
vm05_source_controls_r1（source controls）、vm05_null_family_r1（完整null）；
不得修改计分文件或为查询超时重启。观察器仍使用Windows独立冻结代码，禁止同步覆盖。
输出PARTIAL的审计对应当时已完条件快照，不是当前整批失败。

继续次序：收尾现有批次/采集；完整源头observer与同材料强MBTI sparse参照、
正式ready GRAIN比较；封存来源头再bridge，随后语义/使用机制与VM06/07。
新的可执行问题包括读出尺度稳定性；先诊断，若要校准须只在来源训练侧拟合，
对所有候选同规则另立分析，不能为救测试数值调参。goal保持active。

## 第十二轮：来源控制验收、几何信息边界与强稀疏参照

2026-09-06。新增代码28c99af/0b94409/99df3bf/4c681e8；稀疏队列
1d3e75e/937f914。详见VM05_GEOMETRY_INFORMATION_REPORT及来源控制报告第7节。

- 102来源条件全部完成；独立重建2448次选择，75组等人数配对、72个对比。
  18个Big5保存模型条件/90fold重播8.33e-17以内，连同MBTI合计510fold验收。
- Big5 raw/ cue/ placebo的purge r=.07990/.07971/.06500，相对等人数
  随机平均差-.02976/-.03232/-.04439，但mean-fold差较小。保留原pooled，
  不用测试后重新校准挽救数值。
- Big5 raw完整null199次也已完成：r=.136969、null均值-.014611、p=.005。
  13条件/2587次已独立重算，cue/placebo剩余null仍运行，家族分辨率不改。
- 强word/char MBTI队列r2运行。r1仅为AppleDouble元数据造成的拟合前错误，
  失败记录保留、加测试后新目录启动；不得删除旧结果伪装一次成功。
  EI raw已完成AUC=.672221，旧同材料中间投影=.587588，配对差+.084633
  [.068027,.101767]；但BA=.503630、预测E比例.003285，仍非可靠分类器。
- 冻结profile材料全7266人不读心理值：raw ready144/1566/62，三臂共同
  ready143/1495/61（Big5-only/MBTI-only/bridge），raw与旧preflight完全重播。
- 40个raw、120个共同ready通道条件完成并独立重播800fold；另50解码
  条件完成并独立重播250fold，最大误差2.23e-16。头只用对应独立来源。
  raw MBTI平均AUC：中间48维.5997，规范距离.5381，排序距离.5058，
  规范shape.5306（主PCA）。Big5 ready很小且读出未建立，不回避负结果。
- 冻结16地标在48维中的仿射秩15；距离可恢复投影与垂直范数，不能一般
  恢复垂直33维的方向。全作者恒等式误差<=4.55e-13，合成反例通过。
  这不是发现32个人格因子或证明全部被丢方向具有心理效度。
- 后续探索从距离精确解码，不偷用作者原坐标恢复。TF无PCA完整白化
  AUC=.6281，垂直方向.6247，parallel=.5609；同维随机/来源控制仍待做。
  白化可逆而PCA/正则不对任意坐标变换不变，因此性能变化不等于信息变化。
- Windows14个observer shards完成，当前Qwen shard04继续。Mac验收备份
  shard00/01/02各模型909人，整体未完成，不同步覆盖Windows aa51b06代码。
- 112项tests通过，scoped ruff/diff通过；未跑完整历史suite。无LLM训练/
  下载/付费API/bridge或MEPS心理值读取，未push。发布计分器不变。

下一执行顺序：活跃稀疏/完整null/Windows向量继续；完成源头三观察器与
同ready重拟合比较。数学子问题做坐标等价读出/正则传输重播、等维随机
子空间和条件控制，区分信息损失与读出惩罚，不把新探索写成已验证机制。
来源头/控制封存后才bridge；随后VM06/07和功能使用连接。没有需要用户新idea
才能推进的阻塞，goal继续active。已完成批次不再重跑，未完成批次不重复启动。

本轮收尾：cue sparse null完成并独立复算，raw+cue+12MBTI共14条件/2786次
已验收（vm05_null_family_audit_partial_r3），仅placebo继续。EI raw稀疏
5个模型独立重播最大差9.72e-17，cue正运行。上述更早的13项是过程快照。

## 第十三轮：精确读出搬运与同维随机反证

2026-09-06。机制报告：VM05_READOUT_AND_RANDOM_SUBSPACE_REPORT.md。

- 100折/20全来源头在白化或总半径lift中精确搬运，误差2.89e-15；全材料
  7266人的坐标恒等式通过。TF主PCA原头.6485，重拟合白化头.5859，但
  正确搬运仍.6485。不可用可逆白化后的成绩下降证明信息消失。
- 勘误：parallel+垂直范数平方与地标平方距离不是仿射关系，缺少parallel
  平方范数。换成总半径平方才仿射等价。原代码解码/原OOF成绩不变。
  该修正与同维控制计划先提交292314c，再执行后续200条件。
- 20固定随机正交分解×15/33维×五来源任务全部完成，1000折/200全来源头
  保存。独立重建随机基底并重播1000折，误差0。TF垂直.62473，随机33均值
  .61389、范围.54383至.63955；尚非特殊人格方向。Big5144来源仍很弱。
- 随机批次identity：2a22bac7779f611dad32d188ed7e7f8afa81ca90394e78ace25c4f98656d5ba2。
  完整目录vm05_random_subspaces_r1，独立审计vm05_random_subspace_audit_r1。
  搬运目录vm05_readout_transport_r1；支持门槛、官方fold、发布scorer均不变。
- 发现历史源码归档glob漏收新命名runner/audit。新增只读补档工具，原件保留；
  6项原manifest代码SHA验证通过，110允许文件写到vm05_source_archive_repair_r1。
  这是事后补档，不补造预注册。活跃队列结束后再统一修订共享归档器allowlist。
- 强稀疏EI cue完成AUC=.672686，raw=.672221；差+.000465
  [-.001543,.002563]。placebo已启动，三臂不能提前宣布完成。完整null家族
  仍14项已验收、Big5 placebo执行中，不重启在跑的子进程。
- Windows继续运行，新增03/04备份SHA=f010529498275b5fe60a07adcf43640cc0a2aea9909797f015e806ea81ee1812。
  本地00至04共15份observer shards逐作者cache SHA通过，各模型1515人；
  完整7266队列尚未完成，禁止同步改动冻结Windows代码或提前用部分替代全量。
- 122项专项测试通过，scoped ruff/diff通过。未进行LLM训练/下载/付费API，
  未读bridge/MEPS心理值，未push。新的代码与完整报告分批commit。

本轮具体复现入口（所有路径在独立GRAIN仓库，输出需新目录）：

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
"$PY" scripts/audit_grain_vector_readout_transport.py --profile-dir "$R/vm05_profile_material_r1" --profile-batch "$R/vm05_profile_heads_raw_r1" --geometry-batch "$R/vm05_geometry_information_raw_r1" --views "$R/vm05_qualification_r2/common_views.json" --output-dir "$R/new_transport_audit"
"$PY" scripts/run_grain_vector_random_subspaces.py --profile-dir "$R/vm05_profile_material_r1" --geometry-batch "$R/vm05_geometry_information_raw_r1" --views "$R/vm05_qualification_r2/common_views.json" --output-dir "$R/new_random_controls" --max-seconds 1200
"$PY" scripts/audit_grain_vector_random_subspaces.py --batch-dir "$R/new_random_controls" --geometry-batch "$R/vm05_geometry_information_raw_r1" --profile-dir "$R/vm05_profile_material_r1" --views "$R/vm05_qualification_r2/common_views.json" --output-dir "$R/new_random_audit"
```

上述是复现说明，不意味着应重跑已完成200条件。下一轮收尾最后null/强稀疏
三臂；完成三个观察器同材料采集、源头/ready重拟合及来源控制，再封头开bridge。
几何子线本轮收束；VM06/07及读出语义不变性/实际使用接口仍待做。当前没有
需要用户新idea才能继续的阻塞，goal保持active，不能声明理论闭环。

## 第十四轮：完整null、EI三臂与MEPS材料算子

2026-09-06。没有改动运行中的来源头、Windows冻结aa51b06代码或发布计分器。

- 完整null15/15、2985次inner-refit置换均独立复算。所有p=.005为采样下限，
  Holm15=.075；不能叫家族显著或为更小p追加运行。最后Big5 placebo
  r=.136794、null均值-.014530、区间[-.068690,.035943]。
- EI强稀疏raw/cue/placebo三臂完成AUC=.672221/.672686/.673967。
  3配对差的95%区间均跨0；没有cue特异改进。BA仍.5036至.5049，
  排序与阈值分类分开。15保存fold全重播9.72e-17以内。其余9条件继续。
- 原稀疏进程已退出、状态PAUSED_BUDGET才以同manifest续跑4小时预算，
  每子进程8GiB不变，session96177；当前SN raw，绝不重复启动。
- 修复历史MEPS默认加载mood表的问题；调用方不用值不等于从未读取。
  明确opt-in与默认file-open canary；历史报告追加勘误，未改原结果。
- 新MEPS白名单路线只提取三类CSV允许字段与直接回答段。混合Markdown
  容器扫描/哈希如实披露；无问卷/mood CSV值、邻接registry表访问。
  264视图、1539tokenizer条件，日语英文tokenizer UNK约20%至23%；
  多语E5三问UNK为0。完整自由对话46/46均超512。
- 自由相谈AI字符占88.30%；不能等同AI对向量的贡献百分比。来源归属与
  条件上下文必须分开。138直接回答+1462聊天轮组成1600单元，逐说话者
  1609窗口覆盖366385字符，损失0。没有模型forward或embedding输出。
- ts_iso实际1462条全部clock-only，未推断日期/跨日间隔；5重复内容、
  3字符计数异常均保留。r1至r3来源/token/角色统计不变，独立元数据audit通过。
  windows_r2曾因错误本地路径拟合前退出，canonical使用r3。
- 136项专项tests通过；scoped ruff（ExFAT权限EXE002单列忽略）与diff通过。
  未跑全部历史suite；未下载/训练LLM、调用付费API、打开bridge或push。

完整报告：VM05_SOURCE_CONTROLS_AND_NULL_REPORT第8节、
VM06_MEPS_MATERIAL_PREFLIGHT_REPORT；新增CLI为meps_preflight、meps_windows、
audit_grain_meps_material、summarize_grain_sparse_mbti。
canonical目录：vm06_meps_preflight_r3、vm06_meps_windows_r3、
vm06_meps_material_audit_r1、vm05_sparse_ei_three_arm_comparison_r1、
vm05_null_family_audit_final_r1。MEPSpreflightidentity：
2a1a13248f7bc74984ad1e4eab1f72aa80fd53993f4f26d146530bfa8121bebb。

Windows队列PID28172仍存活，20个observer shards完成，Qwen shard06
最近265/303、GPU峰值4.311GB；Mac已验收备份仍00--04每模型1515人。
不能将新远端完成部分说成本地已备份，不同步覆盖冻结源代码。

下一：完整来源观察器与剩余稀疏/同ready比较；在不改变来源头选择的前提下
准备角色组合与上下文干预的区分设计。冻结来源规则后才bridge和VM06实际
机制验证。仍有可执行研究工作，不需用户新idea，goal保持active。

## 第十五轮：真实回复的上下文利用与内存修复

2026-09-06。上一轮为实质progress；本轮新增机制证据而非只做材料审计。

- 先提交3b4b463冻结RC1。Qwen旧Windows缓存七文件复制到Mac并逐SHA校验，
  未下载新模型，MEPS未传出Mac。r1是BatchEncoding接口拒绝、无forward；
  r2 BF16日语器械未过位置平移容差；改FP32、不放宽门槛，r3误差<=3.58e-6。
  dfb835b提交器械实现后才启动真人forward。
- 46自由相谈中44人满足完整最后回复规则，每人一条16--243 tokens。
  四上下文臂+block10自替换/搬运共270forward，346.8秒、MPS采样6.175GB。
  实际/错排/随机/无前文的当前token逐个相同，输入embedding差0。
- 真实相对长度匹配的平均logp差+.43643[.31067,.57494]，39/44正。
  相对随机+.41975，相对无前文+.82192。区间条件于固定材料/错排，
  不是人口p值/人格效度。完整用户平均向量的高余弦不排除逐位置读出效应。
- block10正文全状态搬运仅+.02431[-.01639,.06162]tail改善；完整真实
  tail差+.41975。不能认定中层用户向量是因果瓶颈，仍要补末层正控制。
  自替换44/44逐token重播0；独立审计核对44成员/528几何行，复算错排、差/CI。
  没有独立重跑270forward，也未保存激活或心理值。
- SN强稀疏旧r2完成5个外折后在全来源阶段触发8GiB保护。原childPID33463
  已退出，父FAILED_STOP；子RUNNING只是旧快照，不可误判仍在跑。
  发现最后inner CSR矩阵仍留在final fit期间，显式释放+GC，不改数学。
  旧14505a7与新函数4个合成fixture的inner分数、参数及预测完全相同，
  weakref测试确认矩阵已释放。4b668db提交；8GiB不放宽。
- 新r3只跑SN/TF/JP九条件，session73299、child35968，4小时上限。
  不重跑EI，旧SN五fold保留但未验收为完整头。真实内存改善仍待任务完成。
- Windows最近24个observer shards完成、BGE shard08继续。新增05/06备份
  648872260字节，两机SHA=32a3d34354195345a5ae3f198af39cb63e626286b0c4cb69ce975b45398364ed。
  Mac00--06共21observer shard逐文件核验、各模型2121人；整个队列未完成。
- 143项专项测试通过。未改Windows aa51b06、发布scorer、锁箱；未训练
  LLM、读取bridge/MEPS量表或push。无关dirty报告原样保留。

核心报告VM06_RESPONSE_CONTEXT_REPORT；canonical vm06_response_context_r1、
vm06_response_context_audit_r1、vm05_sparse_memory_fix_audit_r1。
下一：末层读出正控制、少数固定层的状态/上下文利用路径；完整源头采集、
强稀疏与同ready比较、来源封存后bridge仍需完成。新机制只解释上下文
兼容/利用，不直接命名Who或人格；goal保持active，不需要用户先给新idea。

## 第十六轮：正文状态的层位充分性

2026-09-06。上一轮为progress；本轮完成新干预与真实资源问题闭环。

- 9cd659a先冻结RC2六个站点，0e6b728实现后才运行。复用RC1完全相同的
  44回复/前文/错排；没有层选择、新材料筛选、心理标签或LLM训练。
- 日/英非私人final_norm正控制tail误差均0；正式628forward、832.3秒、
  MPS采样6.175GB。264次自替换和首词不变均精确通过；末block/末norm
  88次tail恢复全0误差。88个实际/错配原始forward重做RC1标量差<=4.44e-16。
- block4/10/16/22搬运的平均tail改善为.011616/.024313/.168065/.343169；
  完整实际前文gap=.419749。block16和22的固定材料描述区间均正，
  但不叫人格效度。末层100%是逐位置读出的预期恒等式，不是语义发现。
- 独立scalar/material audit通过全部264站点行与44成员、SHA、效果/区间。
  未独立重跑628forward。完成预设结果后检查发现仅13/44深度序列单调，
  block22有11人超过原真实前文结果；不能解释为各人恒定比例的信息积累。
  当前证据支持状态可用性依赖层位，持续前文访问/非线性组合/词汇重合仍待分辨。
- 152项专项tests通过。新故障fixture曾在pandas整数列写小数前失败，
  明确float后6个攻击/重播测试通过；真实模型实验未受影响。scoped ruff/diff通过。
- SN强稀疏raw真实内存修复完成：6393人、AUC=.669877，mean-fold=.672291，
  RSS采样6.756GB<不变8GiB。5保存模型独立重播最大差1.11e-16。
  0.5阈值全N、BA=.5，排名与实用决策分开；未在OOF/bridge上调阈值。
  r3在11:39 JST核对为SN cue_mask，child37622仍活跃；其余8条件未验收。
- WindowsPID28172存活，28observer shards完成、E5 shard09推进；8h包尚未停。
  新备份shard07为324287567字节，远端/本地SHA
  70f292ed26e6fa7b6d2c602a7fa169cf72ce759a92443fa6db51506668df22af。
  Mac00--07全部24shard作者/cache/来源/fingerprint核验，各模型2424人。
  远端08完成但未本地备份；不可把本地完整度与远端进度混为一谈。
- 未改发布scorer、Windows aa51b06或活跃来源计算代码；未开bridge/MEPS量表；
  未push，无关dirty报告和legacy锁文件删除保留。

canonical：vm06_response_depth_r1（identity
991a2673c0e65aa8105ed0f5616c86e34c6cf87f9ea6a27879297397122c92b4）、
vm06_response_depth_audit_r1、vm06_response_depth_technical_r1、
vm05_sparse_sn_raw_summary_r1、vm05_sparse_sn_raw_replay_r1。
完整方法/表格：VM06_RESPONSE_DEPTH_REPORT；CLI：run_grain_response_depth、
audit_grain_response_depth，复现命令见报告，不重跑已完成包。

下一轮先复核两个活跃计算，不重复启动。机制侧不再追加层位搜索，固定材料
上设计词袋/词序与持续前文访问对照，并用逐位置读出几何解释高余弦/大作用。
这仍是未完成的新实验，不能写成已有证据。完整来源观察器、强稀疏三臂、
同ready比较、来源封头后bridge和VM07仍需完成。Goal仍active，本轮无阻塞。

## 第十七轮：词袋固定的顺序扰动与持续访问器械

2026-09-06。本轮有新实验/独立审计/资源闭环，不是等待或阻塞。

- RC3协议9372ff1、实现049233f在私人forward前冻结。同RC1的44人，
  446forward、591.14秒、MPS采样6.175GB。全部token词袋精确保留。
  native实际-错配优势.43643；block8双seed平均.30248，token平均.06707。
  差中之差.13395 [.08061,.19581]、.36936 [.26418,.49445]。
  tail敏感性同方向。不能把大降幅叫人格占比：语法/子词破坏是实在的替代解释。
- 独立审计440标量行/352排列记录/全部效果CI，native重播RC1最大差0。
  44个上下文无边界交叉，无重复seed或原样排列；block8/token改位率.9675/.9870。
  没有独立神经重跑，不把hash审计说成重新读取token验证。
  文献对照支持精确子词级打乱和保留顺序推断风险，详见RC3报告。
- SN三强稀疏臂完成：6393人，raw/cue/placebo AUC=.669877/.669909/.668310。
  cue-raw=.000033 [-.001567,.001536]；placebo-raw=-.001567
  [-.003144,-.000073]；cue-placebo=.001599 [-.000151,.003256]。
  条件于已拟合OOF，不多重显著性宣称、不追加调参。全N，BA@.5=.5。
  15fold保存模型分别重播误差<=1.11e-16；三臂RSS采样均<8GiB，生命周期
  修复在SN三臂均已验收。r3进入TF raw，其他六条件仍待完成。
- Mac新增08/09备份640160018bytes，SHA
  3aab6c88cd31998f67a9942d0d2be989db18e2fa2b03b0cbb8d68da43d06dded。
  CRC/安全路径/不覆盖检查通过，全部00--09共30observer shards逐manifest/
  作者/cache/source/fingerprint重核验，各观察器3030人；整队列仍未完成。
  Windows最近32shards完成、Qwen shard10运行，aa51b06未改。
- 新RC4协议659d638，代码72dbc3b；固定after{-1,10,16,22,27}，只阻断
  后续层body-query到全部prefix-key的访问，不删词/换role/动position。
  EN/JA共42非私人forward的layer0独立、patch+cut恒等式、mask/self sham、
  首词不变检查均0误差。189专项tests通过（两条历史SWIG warning）。
  正式834forward包已启动`vm06_response_access_r1`，session76500；**未验收**。
  一项攻击fixture在整数列写小数时提前报错，改为float后测试通过；不是模型失败。
- 未读MEPS心理字段/bridge outcomes，未新训练LLM、付费API或push；不改
  发布scorer、活跃稀疏来源代码/Windows冻结代码，无关dirty报告原样保留。

报告：VM06_RESPONSE_ORDER_REPORT、VM05_SPARSE_SN_COMPARISON_REPORT。
canonical：vm06_response_order_r1、vm06_response_order_audit_r1、
vm05_sparse_sn_three_arm_summary_r1与三个SN replay包。
RC4器械：vm06_response_access_technical_r1；运行/审计CLI为
run_grain_response_access.py、audit_grain_response_access.py。

下一先验收RC4，不重复启动，若失败先查身份/预算/掩码而非删除被试。
最终要用独立audit重算834forward的标量证据，不能以42非私人控制代替。
随后逐位置读出几何与完整源集合比较，不能一直在44短回复上追加自由探索。
完整来源采集、TF/JP稀疏、同ready、冻结来源后bridge及VM07未完成；goal active。

## 第十八轮：持续读取与精确读出闭环

2026-09-06。本轮完成RC4/RC5并独立审计，是progress，不是仅等候。

- RC4正式44人834forward结束，1113.02秒，identity
  7e802bdadf5b0a056ea4b861126927028896f87349e413a731743fd5d6c6651e。
  792私人标量行、36非私人访问条件行完整核验。cut=-1独立、132patch+cut、
  88sham、首词不变端点均0误差；RC1/RC2原值重播<=4.44e-16。
- 阻断after10/16/22时，保留优势D=-.053701/.083854/.315516，原值.419749。
  D10区间跨0，D16/22固定材料区间正。实际臂NLL损伤分别1.980/1.062/.413，
  从第一层就阻断仅.642。中途切断的计算失配非常明显，不能把I当信息百分比。
  这把RC2的“状态充分性”收窄为依赖剩余可访问前文的功能，而非固有人格含量。
- RC5先fdfb342冻结标准softmax/KL问题，167cbae实现。最初非私人英文
  head按8行重算出现4.254924e-5误差，超过不变1e-5，保留technical_r1失败。
  非私人2forward诊断定位为head矩阵形状；改为原始T行、保留tail行号，
  第0行填忽略的0向量。只把CPUfloat64归约分为8行；e450c27在私人前冻结
  修正。technical_r2 EN/JA通过，head误差4.42e-7/2.68e-7，不放宽门槛。
- RC5正式44人98forward+92head调用，151.88秒；1808tail标量位置，identity
  15ed75e0bbe420a9a8a71c61665692ca784a11cf6c2b8b379bf03727c371d16e。
  独立audit最大代数误差1.92624e-14，head逐token误差<=1.21336e-6，
  与RC1人均tail差<=6.97e-7。原式有舍入差而非信息变化，不混写不同精度。
- mean-state cosine=.991758、mean token cosine=.974518，净logp差=.419749。
  精确分解=.756678-.336929=1.072463-.652713=-.198688+.618437。
  净作用为正但actual端局部梯度平均负，说明有限位移不能用一个端点线性化
  代替。KL是模型词分布差，不是心理距离/噪声；数学恒等式是器械，不是首创。
- 209专项tests通过，两条历史SWIG警告。未跑完整历史测试。RC4/RC5均在Mac，
  MPS采样6.175GB；未读MEPS量表/bridge，无LLM训练/付费API/额外下载。
- TF raw完整完成：6393人，AUC=.767212、mean-fold=.767390、BA@.5=.657990、
  macro-F1=.665807；正率预测.215705、真实.352417。5fold重播<=1.11e-16，
  RSS6.015GB，1874.6秒。不同于SN全多数类，但其cue/placebo尚未完成。
  snapshot vm05_sparse_sn_tf_partial_summary_r1；replay vm05_sparse_tf_raw_replay_r1。
- 12:59 JST稀疏r3仍运行TF cue_mask（child43574，100%CPU，510秒）；其余
  五条件未验收。WindowsPID28172最近35shards完成，Qwen shard11推进。
  Mac新增shard10归档310567598bytes，SHA
  31430db1cf3c044d73722452a8fffe9ffbaa545f4af86e83033716e7aaa0d10e；
  CRC/不覆盖安全解包后00--10共33完整shards全部重核验，每观察器3333人。
  远端35不等于本地35；whole_queue_complete=false。

新报告VM06_RESPONSE_ACCESS_REPORT、VM06_RESPONSE_READOUT_REPORT；独立audit
包分别vm06_response_access_audit_r1和vm06_response_readout_audit_r1。
两正式神经进程已exit0，不重跑。保留所有失败technical目录与源snapshot。
未改活跃稀疏来源模块/Windows aa51b06、发布scorer；未push，无关dirty保留。

RC1-RC5短回复机制包现在关闭，不追加自由层搜索。新建
GRAIN_VECTOR_REMAINING_REQUIREMENTS.md逐项对回完整原计划：MEPS264views的
固定/自由神经比较还没做，不能以44份free-chat回复替代；Essays/X当前模型
机制复测、全部源观察器/强稀疏/同ready/封头后bridge和VM07也仍缺。
下一轮先核验活跃任务，随后利用等待时间准备这些缺项的同预算/角色/语言
材料协议，不再仅围绕44回复打转。原goal保持active，当前没有真实阻塞。

## 第十九轮：完整同人固定/自由比较与跨域材料闭环

2026-09-06。本轮是progress：补原计划VM06，而非扩张RC44回复层位实验。

- 966d4ce先冻结multilingual-E5模型/512定长/角色聚合/七项主对比；ea553c1
  在私人提取前增加旧position_ids校验。该唯一unexpected key是[1,514]
  的整数arange非学习buffer，无缺失学习权重。公开EN/JA重复/pad污染误差0，
  compact/fixed最大1.30e-7，原1e-5容差不变。
- 正式vm06_fixed_free_r1覆盖46人264views、1600turns、1609完整窗口，
  366385字符不丢失。1617forward(8公开+1609私人)，269.33秒，MPS采样2.242GB。
  identity ac077428fbd6d8b73c5c39f934c955b7d4b86cda24aa739077bcefccdf64e51b。
  这不是整段对话context forward；未读取心理量表，也不是心理人格打分。
- 三个固定问题与free用户文本都有微弱作者对应；全固定/自由主percentile
  .590821 [.547366,.638377]，cos gap .002465 [.001287,.003730]，Holm p=.0035。
  等turn敏感性.593720。七个one-sided permutation/Holm均<=.025，但Q2/free
  gap双侧bootstrap下限-.000018。两统计不同，不能写七个确认人格机制。
  长度匹配后全固定/free gap降至.001593；长度/主题/AI辅助等仍是替代解释。
- free user/assistant同对话对应percentile=.994203，不能解释为人格。
  固定答案/free assistant=.522464；固定答案/token-mixed=.533575，均CI跨.5。
  assistant占平均87.04%内容tokens，mixed与AI cosine=.999308，与user=.968539，
  部分是加权平均的机械结果。等turn mixed=.578744，不据此选择最优人格头。
  关键分离：跨任务作者对应≠同对话内容回声；AI上下文要保留但不能混成作者证据。
- 独立audit vm06_fixed_free_audit_r1：1394view向量/32矩阵/七项1999置换/
  Holm全重播；pool error8.33e-17、matrix9.99e-16、point3.33e-16、p误差0。
  不重复神经提取或所有bootstrap；节点bootstrap有合成单测。14相关tests通过，
  连同RC5回归选择34tests通过，非全仓库tests。新报告VM06_FIXED_FREE_REPORT。
- 49936b7冻结当前Essays/X inventory脚本：Essays2467人一人一文；X37435篇、
  19954作者，无post ID冲突，但84种原文跨作者共享。X en/ja至少12贴作者
  91/52，尚未做长度/重复/同条件门槛。时间全有zone且可解析，collector group
  不是作者自由choice或实验控制。Essays全CSV被hash，但仅user_id/text列解析。
- vm06_domain_inventory_r1 identity
  d60f6acd95eba9124bf497b5009e6143832710639cdda73747eed7bfb909b12f。
  固定hash抽样2708条、8124tokenizer调用，0神经前向。Essays200条仅mE5
  36条<=512；X日文BGE/E5 UNK~23%，mE5 .001609。低UNK不是跨语言测量等价。
  新报告VM06_DOMAIN_INVENTORY_REPORT；当前神经跨域实验仍未完成。
- 稀疏TF cue也完整：6393人AUC=.765381，meanfold=.765561，1634.85秒，
  RSSpeak6.434GB；5fold独立重播1.11e-16。TF placebo仍运行，child45866，
  最近611秒、RSS5.534GB；JP三臂未完成，不从TF raw/cue部分结果选模型。
- WindowsPID28172仍活跃：38observer shards完成，Qwen12进到84/303，
  374.5秒。冻结aa51b06不改。Mac新增shard11归档313878651bytes，SHA
  f3d63c4654d77e2847c799bcc021c47c53fa8dfa48a9a44e73c31307e24a6964。
  传输SHA/CRC/安全不覆盖解包后，00--11共36shards所有manifest/缓存SHA重核验，
  各观察器3636人。远端38不等于本地38，whole_queue_complete仍false。

未改正在运行的source模块/发布scorer；无训练/新模型下载/paid API/push。
下一轮冻结并运行Essays/X现有机制domain包，同时看源任务边界；只在已终止
且PID退出后恢复剩余冻结条件。完整source vectors/strong heads/同ready/
封头后bridge和VM07 atlas/spec仍未完成。Goal active，无需用户新idea阻塞。

## 第二十轮：跨域功能、独立几何与词句复用机制

2026-09-06。本轮是progress，完成VM06域内机制比较，原goal保持active。
不是人格预测完成，不变更正式GRAIN scorer，也不扩张RC44自由层位搜索。

- 511367a冻结Essays/X材料及六条件Qwen实验。全量源容器hash不变，仅读
  允许的文本/来源列。最终167cases=96 Essays+55 X-en+16 X-ja；X同语言/
  symbol内完整case匹配，去精确跨作者复制/同作者重复，未宣称去完近重复。
  material identity f24273f868d793db6d909324db51f68013bb630e4bc5585634ebcad6b0cd3d5f。
  第二次材料准备同算法重跑得到完全相同hash，不是独立抽样算法。
- 首次public context为15tokens，不足16，在第一forward前拒绝；保留r1
  FAILED_STOP。4e49395只扩展公开英文fixture，未放宽门槛/私有case规则。
  r2完整1008forwards、1339.99秒、MPS6.175GB；EN/JA selfpatch误差0，
  compact/fixed最大4.054e-6。identity 3be6da509f36028c6ba9ea2aa7a698cc48909363276c0f005e54f0ef54023da7。
- 正确/条件匹配其他作者的tail logp增益：Essays .16815 [.14006,.19451]；
  X-en .93911 [.65858,1.25144]；X-ja .27123 [.03704,.62521]。打乱context
  block8后仍有.11964/.77480/.17894；顺序交互.04851/.16430/.09229。
  同一current input embedding完全相同，final pooled cosine仍.990-.994。
  这是teacher-forced原文可预测性，不是心理量表正确率/自由生成质量。
- e4eac66在效果表查看前冻结独立geometry对照：mE5全167，英语BGE/E5各151；
  356全字符覆盖共同窗口、938 pooled向量、469case-observer pairs。
  每模型独立原文encode，统一字符权重，1020forwards/176.83秒，MPS2.248GB。
  tokenizer/weights/旧整数position buffer核验；public pad/repeat误差0，
  compact/fixed<=1.57e-7。identity 7b2b7f7d30efd3ba7ffec4eb566e7033bccfa933fbf207e75a44ca5825d5f283。
- 独立context相似度优势与Qwen功能增益相连：Essays BGE/E5/mE5 r=.350/.447/.411；
  X-en=.721/.899/.815，X-ja mE5=.806。长度差控制后X-en E5=.906。
  此相关不是Big5 r，也不是两个模型内部向量坐标相同/人格机制已定位。
- 334efea明确事后探索：在计算overlap前固定1/4-gram、四个token分区及
  长度/stratum/复用联合条件设计，无新神经前向。排除任一context中精确重复
  4-gram后：Essays .16222 [.13465,.18792]；X-en .34793 [.18367,.54943]；
  X-ja .05947 [-.03679,.14247]。前两有剩余功能，日文小N underresolved。
  不能把不同有效N/权重的分区均值相加当因果成分。
- 联合条件化后X-en E5 r=.49025 [.12792,.74310]，BGE/mE5区间跨零；
  Essays三模型.320/.397/.352。X-ja .600但CI[-.750,.946]、df10，一次bootstrap
  undefined，不能称稳定调整效果。残差不是自动的reasoning/personality。
  overlap identity 376c1819b4a5256b31ba823b48a39af5d93d292d4b2337136f22fe3a7be7e108。
- 三项独立保存数据audit均通过：context36,000 bootstrap、几何7000，
  overlap668逐token masks+668case partitions+19000bootstrap。后两分别用
  独立相关代数、SVD投影替代原OLS；最大差3.06e-15/2.39e-14。
  这些不是独立神经重复提取，也不把条件bootstrap解释为全总体推断。
  新报告VM06_DOMAIN_CONTEXT_GEOMETRY_REPORT；28项域/MEPS/顺序回归tests通过。
- TF三臂完整且15保存fold重播：raw/cue/placebo AUC=.767212/.765381/.767227，
  cue-raw=-.001831 CI[-.002377,-.001294]，placebo-raw CI跨零。强TF对cue
  有小敏感性，不同于弱投影头；不能继承弱头的全部来源控制。新TF报告入库。
  JP raw也完成AUC=.610766，5fold重播误差8.33e-17；cue运行，placebo待执行。
- VM07工作版MECHANISM_ATLAS、11项machine-readable evidence与observer spec
  已整理，schema/link检验通过只代表文件结构，不是科学验收。主结论是
  区分来源匹配、上下文功能、语义/输出代码、外部效度，不硬揉成一个人格轴。
- Windows原PID28172在42observer shards后按8h预算门槛退出，27,431.98秒，
  queue.lock已释放。shard13归档323245028bytes，SHA
  a81b696e38d525eaf5b548e995b19618500546f716e472ddd17b594aae1abff6。
  传输SHA/CRC/安全不覆盖解包通过，00--13全部42shards缓存/manifest再核验，
  每模型4242人。本地保存旧暂停status；whole_queue_complete=false。
  首次获取远端归档metadata时遇Windows进度stderr编码问题，归档已成功；
  只补读SHA与传输，不重复创建/覆盖。随后关进度流并按bytes读取，问题解决。
  Start-Process首个恢复PID62444在SSH会话结束后、更新queue前退出；stdout/
  stderr0字节、无lock/observer子进程，不能称续跑成功。保留该次记录；改
  Win32_Process.Create脱离SSH，launcher7748/queue61296活跃核验旧shards，
  仍原冻结aa51b06、同参数加resume-completed、独立8h上限。以下续跑状态
  以实际queue更新为准，不以返回PID为验收。
  后续实测queue61296已RUNNING bge_large/shard14，旧42全部复核通过，
  seconds107.59，进程持续存在；不是仅launcher返回成功。来源代码未同步覆盖。

未读取MEPS量表/bridge结果，无付费API/新模型下载/LLM训练/push。
下一轮优先完整JP及source representations，再做真正强头的来源控制、
同ready比较与冻结后bridge。不因域内显著关联继续无限搜新切法或新层位；
也不把未完成的source/bridge工作改名成不需要。当前不需用户新idea介入。

## 第二十一轮：四轴强稀疏参考闭环，启动真实强头来源控制

2026-09-06。本轮progress；goal active，不是全研究结束。未改发布scorer/
正在运行的source估计器，也未碰Windows冻结aa51b06代码。

- JP cue完整AUC=.6048405，五fold重播8.33e-17。旧r3在8/9后正常预算
  暂停；核验旧PID退出后以相同协议续跑最后placebo，session34410现exit0。
  placebo AUC=.6094635、meanfold=.6166650，1644.09秒/RSSpeak7.167GB；
  五fold重播8.33e-17。EI/SN/TF/JP共12完整arm、60保存fold已全部重播，
  全局max1.11e-16。旧r2失败SN仍保留，不加入完整汇总。
- a10842a新增显式12父目录汇总入口，拒绝缺失/重复/错cohort；输出
  vm05_sparse_all_axes_summary_r1，identity
  ce9738f884515cff51cabcaf3a992bcc44e009246e86734b0fecd3320f656748。
  raw AUC EI/SN/TF/JP=.672221/.669877/.767212/.610766；BA@.5=
  .503630/.500000/.657990/.556540。两者必须区分，不称67%分类准确率。
  JP cue-raw=-.005926 CI[-.010618,-.001723]，cue-placebo=-.004623
  CI[-.009100,-.000503]。TF有更小cue差异，EI/SN相应区间跨0。
  全12个预设对比/1000 paired bootstrap保留，条件于OOF模型、非全流程
  refit推断/多重比较确认。新报告VM05_COMPLETE_SPARSE_SOURCE_REPORT。
- 493e3d9冻结强来源控制：4轴x3臂x7来源条件=84，剔除99历史reference
  后每条件6294评估作者。retain/purge/5随机count controls内外折重做选择；
  使用word_char真强头，不从旧弱projection的84条件继承安全性结论。
  一Mac CPU child、8GiB RSS、1h/条件、8h/批，剩余<3660秒不新开条件。
  预计数十CPU小时，不缩样本/种子/模型来冒充全量完成。
- 独立来源auditor兼容新父目录并重构完整84设计，拒绝错误头/来源/暴露/
  重复条件。原102条件2448selection records/75配对控制/72差异全部重算，
  vm05_source_control_audit_regression_r1的四CSV+summary与原final逐字节一致。
- 发现null原实现同时缓存全部内外折矩阵的RAM风险。新增无损磁盘缓存，
  每次仅载一个outer折；input值/顺序/dtype、代码、父identity、40矩阵SHA
  绑定，开始/恢复/正常退出均验hash。20/199、完整inner选参、作者数全不变。
  dense Big5/MBTI+sparse MBTI的观察与置换预测/选参在合成数据精确一致，
  mutation/跨fold释放也测试。真实强null尚未跑，不能把合成PASS当经验null。
  老15个完成null保持旧snapshot，不在新代码下恢复其输出。
- 相关115tests通过，两历史SWIG警告；新增汇总后11focused tests通过；
  Ruff/diff检查通过，非全仓库测试。atlas11条schema/link检验通过，仅文档
  合法性，不是科学验收。
- 12父packet完整+最后重播通过后启动vm05_strong_source_controls_r1：
  identity de681be2cd42022737c3cf542d0f211876d2442c81b3d85c03b0dae25422ccf0，
  session89662/child54674，EI_cont_raw_retain_731实测260秒、99.8%CPU、
  RSSpeak6.666GB。不是只有shell启动成功；未完成任何新控制时不报PASS。
- Windows61296继续至BGE15；已完成shard14三observer。新归档309514599
  bytes，SHA c1d168d27543c401278cc8dd3f5da6a3415a15dd3bd3f7f559771bd8bcdd5d4e。
  远端/本地SHA、CRC、安全不覆盖解包及00--14全部45shards的manifest与
  private cache hashes通过；每观察器4545作者，whole_queue_complete=false。

无MEPS量表/bridge outcome访问、付费API、模型下载、新LLM训练或push。
下一步保持单Mac重拟合，审计完成的强来源条件；继续核验Windows完整神经
表示。完整强null、神经源头/同ready/封头后bridge/最终VM07仍必须做。
当前无需用户新idea，没有外部阻塞；不以本轮方法工程修复替代人格机制发现。

## 第二十二轮：同ready身份闭环、比较入口落地与强来源首批审计

2026-09-06 16:15 JST检查点。本轮progress，goal active。正在拟合的
external_heads/links/controls及Windows aa51b06均未修改；只新增独立入口。

- 750c6ac冻结same-ready protocol/registry/独立auditor。真实7,266人材料
  生成registry，identity d125e4a3e6d5e20fabd71419bb9b31da5585517971a4f4857848aa92798dc079。
  raw_ready Big5/MBTI/bridge=144/1566/62；all-arm=143/1495/61；排除旧参照
  后分别144/1539/62、143/1469/61。旧99人中只有27/26落在ready集，不能
  从每个子集机械减99。未读取source/bridge标签值、零forward/fit。
- vm05_same_ready_registry_audit_r1独立重构12数组、840内外训练成员记录、
  12coverage/20training-support行，全部一致。最小训练人数Big5=65、
  MBTI=486；只达到算法最低样本门槛，不等于类别支持/统计功效验收。
  registry相关119tests通过，2历史SWIG warnings；非全仓库测试。
- fae1273新增run_grain_same_ready_comparison.py及ready_comparison模块。
  同36评论/合格作者，支持TF-IDF、BGE/E5、Qwen各冻结body层、[g,q10]/[g,b]。
  g=16 sorted distances，不把完整GRAIN4W理论等同此旧profile。原已完成
  profile父项不重复拟合；真正改变作者集的reference-free分支才允许新g头。
  内外折/全源47target-support检查、内折scaler/PCA及原参数grid、5保存fold
  和全源head、原生概率状态、输出hash/重播。单owned子进程8GiB/默认1h，
  partial不覆盖；所有72神经shards完成/hash通过才允许神经比较。
- 新入口全synthetic Big5/MBTI各5fold+full/replay通过，未touch真人标签；
  12focused tests及46项ready/source/resource/domain regression通过，Ruff/
  diff检查通过。equal-view/任务后decision states未被这个入口悄悄视为完成。
- 实际preflight-only：vm05_ready_big5_sparse_preflight_r1，144人6.37sec，
  identity64e52a0211b830ed26f18c759e6f0a8b84ab90a8844c31e96c0203c09527d3dc；
  vm05_ready_ei_sparse_preflight_r1，1539人9.88sec，identity
  0e2ce2a26e9ba40f12db62f3d075d5c1324717d661f410ebc848bc0583716f7e。
  零标签转换/模型拟合/神经forward，不以preflight替代实际同ready比较。
- 强来源batch已完成EI raw retain/purge/count731共3/84。AUC分别
  .670973/.637082/.644727，meanfold .671157/.646169/.655628。
  vm05_strong_source_audit_first_count_r1独立验3case/72成员记录/1count配对；
  三个独立重播packet共15保存fold max9.71e-17。全5count种子尚未完成，
  auditor正确保留purge-vs-five-count对比为pending，不采用第一个seed结论。
- EI同6294评估作者：排参照重拟合vs原模型同作者限制，delta=-.000462
  CI[-.001932,.001063]。purge-retain=-.033891 CI[-.044132,-.023754]，
  但训练人数从约5,000变约2,400，不能称已证明线程泄漏。first count同样
  下跌只是中途观察；当前count732/58532真实运行310秒，session89662持续。
  新VM05_STRONG_SOURCE_CONTROL_CHECKPOINT明确保留样本量/组成替代解释。
- 复核Brier定义并修正文档：Logistic原生概率可以直接用Brier，不要求先
  calibrator；但Brier混合校准/区分度/不确定性，不能单凭Brier宣称已校准。
  依据sklearn官方calibration说明；仅解释修正，没有改旧代码/阈值/结果。
- Windows61296实测Qwen16已279作者/1109sec，父queue50shards完成，持续
  活跃而非卡死。shard15备份304224683bytes，SHA
  be714f17b0a88a9d57bc031bf70c9097ddeab2502f8238bfd4c08e04f7b2d3f3。
  传输SHA/CRC/安全不覆盖解包及00--15全48缓存/manifest重验完成，每模型
  4848人。50远端完成不等于50本地归档；whole_queue_complete=false。

本轮无bridge outcome/MEPS问卷访问、付费API、模型下载、新LLM训练或push。
继续现有强来源对照、全量向量归档，再真实同ready源头/增量比较；原计划
equal-view/decision/context等缺项仍需补齐。强null和封头后bridge未完成。
当前问题是研究执行与来源机制分离，不需要用户新idea；不将active目标标完结。

## 第二十三轮：任务读出补齐、共同位移诊断与未观测反事实定位

2026-09-06 16:45 JST。本轮progress。开始时验证Mac owned58532在CPU运行，
Windows61296存在且进入BGE17；上一轮不是只做状态复述。本轮未更改任何
active source-estimator/Windows aa51b06文件，未另开Mac真人重拟合。

- 复查原计划发现两问差、正文equal-view与真正history对照必须分开。
  当前全量缓存有相同12评论下behavior/disposition两问，没有同一current
  在own/matched/none历史下的表示。原计划“内部context增量”尚未观测，
  不能拿task delta或三view方差改名完成。直接LLM Big5/MBTI评分对照也
  不能由两条社交主动性Yes/No问题替代，已在remaining map显式补列。
- 92162dd在全量neural标签分析前冻结readout sensitivities；ba3adce实现
  external_readouts、qualify/audit脚本。4sites x3arms x6channels加3个非重复
  logit诊断=75矩阵。total variance按within+between分解，等view不是平均
  std；paired task delta逐view先作disposition-behavior再聚合，非两std相减。
  这是标准矩/差分运算，不称新心理法则，不新命名人格因子。
- 真实技术12人vm05_cached_readout_qualification_r1 COMPLETE，identity
  f4cf9aefbd142c913535a5b64dc7ef71958098ec800d5072c63e8668ce2a7505；
  75矩阵/900作者channel行，0.89秒、零新forward/fit/target values。
  token原pool重播max1.17e-13，总方差max4.55e-13。
  vm05_cached_readout_audit_r1用独立second-moment和逐task代数重建全部75
  矩阵，max1.95e-13，无refit；不是只测测试fixture或信任生成方summary。
- 419a0df新增完整缓存readout loader与same-ready --readout入口，默认
  body_token走原路径；新渠道只限Qwen/GRAIN+Qwen，logit指标只允许一次
  block10标签，不当4个层模型。每作者/完整shard union核验，显式保存
  readout identity。full-source次级编排和实际同ready外部比较仍待做。
  不修改现有full-source body runner去中断84来源控制。
- 对不完整真实全量cache执行preflight-only的decision_task_delta，正确
  在All72 gate拒绝；vm05_readout_incomplete_queue_guard_r1保留FAILED_STOP，
  无source_label_audit/模型文件。此为预期边界验收，不是人格实验失败。
- posthoc技术几何诊断vm05_readout_technical_geometry_r1另立身份
  e149fd04869c6565030017de6129153fb7e8e4098cc832439005a1f1bba71dc3。
  全3arms/4sites保留；raw block04/10/16/final的common task energy fraction
  .7198/.8455/.9616/.8713。标准energy恒等式，不等于prompt纯因果份额、
  作者信号百分比、人格解释方差或噪声率。余量也混有材料/情景/状态。
  技术群均值不进入正式评分/去噪；不根据12人共向量去删“共同噪声”。
  body between-view variance仅.34%/.64%/.78%/1.95%，提醒Qwen token-level
  std与BGE comment-level std测量层次不同，不能仅靠同文量称纯架构比较。
  source内no-PCA敏感性仍预定，不依技术结果加新调参池。
- 新51项readout/ready/source/resource/domain回归tests通过；随后14focused
  tests加energy translation/zero边界通过；两旧SWIG warnings，非全仓库。
  Ruff/whitespace通过。合成完整cache测试会拒绝未选作者缓存被篡改，而
  非只检查这次ready子集；新特征同作者顺序重排及alias均精确可重建。
- Mac来源count732/.645304、count733/.647596完成；当前count734/60520
  运行460秒。vm05_strong_source_audit_three_counts_r1独立5case/120成员
  记录/3nested count配对，purge-vs-five-count仍正确pending。新732/733
  重播10fold max9.71e-17；累计本batch25保存fold通过。原84还未完成。
- Windows61296持续Qwen17，远端53stage-shards完成；新shard16归档
  325480518bytes SHA1163751dfcbeb7d9243927e241ea01ccee19c74840d7924661f662f21d8ae2e3。
  传输SHA/CRC/不覆盖安全解包、00--16全51shard cache及manifest复核通过；
  每模型5151作者。53远端完成不能当53本地归档；全7266尚未齐。

无bridge outcome/MEPS问卷访问、付费API、下载、LLM训练或push。
下一步继续完成live来源控制和向量队列，接入完整source次级readout与真正
same-current历史采集设计；不能以已完成task/mean特征缩小VM05原始范围。
当前仍有明确可执行机制缺口，无需用户新idea，goal继续active。

## 第二十四轮：全源读出入口与EI人数/来源差值的估计对象审查

2026-09-06 17:05 JST快照。本轮progress。开始先确认上一轮5d08cee确已提交，
未重复commit或误称丢失。Mac第五count seed61202实测运行，随后正常完成；
新cue_mask retain/62122接续。Windows61296实测存在且进入shard18，不重启。

- a65bc4e新增run_grain_full_source_readout，补齐全源次级读出工程入口。
  源头624/6393或明确排参照624/6294，非same-ready子集；同fold/统计头，
  retain-thread独立声明。全72缓存sha、原作者union、源材料与parent先核验，
  然后才能转换源标签。1个8GiB/限时owned child，不动正在跑的旧评分文件。
  每条件保存5外折/full source模型、私有预测/支持度、代码/输入/输出sha，
  保存模型重播必须<=1e-10。合成Big5/MBTI端到端与边界 tests通过。
- 真实vm05_full_source_readout_incomplete_guard_r1在All72 gate正确拒绝；
  无source_label_audit/模型输出。该FAILED_STOP是预期验收，不是拟合失败。
  默认正文与次级readout入口虽齐，真实完整神经拟合尚未跑，不能标完成。
  真正same-current历史反事实及直接LLM Big5/MBTI对照仍是未实现的机制缺项。
- EI raw count734=.660838、735=.643254完成，固定五seed均值=.648344；
  retain=.670973，purge=.637082。vm05_strong_source_audit_ei_raw_complete_r1
  独立7cases/168成员记录/5nested count匹配/3eligible contrasts，84总计划
  仍PARTIAL。新734/735的10保存fold max9.71445e-17，累计35fold重播通过。
- 原预定purge-vs-five-count pooled delta=-.011262 CI[-.021184,-.001484]，
  相比purge-retain总-.033891，人数/随机组成损失占相当部分，但不是心理
  信号份额或纯因果中介。五count AUC范围.643254--.660838，不能选最好seed。
- 978b28e在已看全七case后明确posthoc增加rank-AUC独立计算与按折重采样。
  vm05_ei_source_gap_posthoc_r1 identity
  0b74773c9a18e43c1105a17e2e590e69fea5f52043480dd21195a137d3be695e。
  1000次fold内author bootstrap：pooled purge-count -.011262
  [-.021749,-.001395]，meanfold -.006444[-.015413,+.002832]。
  个别fold符号不同（purge较好2/5），不能把pool显著当稳健线程泄漏定论；
  也不把跨0当等效证明。不调test calibration去“修”这个估计对象差异。
- 人数/来源差值的加减恒等式逐fold误差0；标准统计运算不是新人格定理。
  independent sklearn按保存CSV重算36observed标量 max2.26e-16；单列
  vm05_ei_source_gap_rank_audit_r1并写明没有重算所有bootstrap draws。
  支持度E比例.207658，retain预测E仅.003654，BA=.503971；purge无E预测、
  BA=.5。排序信息存在不等于已可用个体分类，亦非LLM人格reasoning证据。
- Windows新shard17 zip305416275bytes，SHA
  f8f733463add7a4652a4e3a6a2141c62441fa12f8d8513557a53d55f21848837。
  SHA/CRC/安全不覆盖解包、00--17全54manifest/作者cache全部再核验，每模型
  5454作者。备份status的旧complete计数与新增verified计数已从实际分片
  统一为54及indices0..17，避免状态页自身矛盾；whole_queue_complete=false。
- 收尾27项full-source/readout/ready/gap/strong-control专项tests通过，
  两条既有SWIG deprecation；Ruff及git diff --check通过，非全仓库测试。
  62122最后实测cue-mask retain运行510秒；active scoring六文件SHA与
  batch冻结承诺仍完全一致。没有因增加新入口改变正在执行的对照方法。

无bridge outcome/MEPS量表/付费API/下载/新LLM训练/push。发布scorer不变。
正式报告新增VM05_EI_COUNT_MATCHED_SOURCE_REPORT；下一步继续其余axis/arm
和全向量队列，执行现已备齐的全源/同ready读出，并冻结真实history反事实
和直接trait打分的仪器方案。未达到需要用户新idea或总计划闭合的终点。

## 第二十五轮：完整历史材料、token边界修复与条件匹配可行性

2026-09-06，至18:05 JST后续归档。本轮为实质progress，未关闭整体goal。
核验本轮起点的运行与git状态，保留无关报告修改。未重启Windows队列，
未修改正在运行的Mac源头估计器，active scoring六文件SHA再次全部一致。

- 82ca8ba/5fb48d8冻结source-safe history设计、原生token/固定当前位置/
  遮罩工具及独立匹配、词四元组近复制审计。原99参照MBTI作者仅作字典，
  不进入本实验预测/测试行；7,167目标为624 Big5-only/6294 MBTI-only/249
  bridge材料。每view保持本人11历史+1当前，三view原36评论，不重采替代。
- vm05_source_history_material_r1 COMPLETE，4551人支持。独立audit_r1
  重建13653匹配view，误差0；12人/38对近复制，旧strict4539。初次大量
  排除来自2569人prefix/suffix token边界不一致，不是原文或人格信息不足。
- 0cd3523：192名哈希失败作者的3456布局中，原1432布局失败；仅将历史后
  原有两个换行归入历史槽，192全恢复、native/current token零改变。加入
  --history-envelope显式版本字段，旧r1保留，启动完整r2不放宽任何等式。
- vm05_source_history_material_r2 COMPLETE，685.386s，7120支持，47人
  无合法先前字典历史；identity
  3b65bb734a57b778a38e1d2cf1651b39fd9885d922bb5f3f51bfbc72c2d8a2ea。
  audit_r2独立21360view/384480布局承诺完整性、匹配误差0，157.102s；
  identity6955200ed63692f9a242af818b2fca19b07f2edcbbde320bb1551f08151c37d5。
  同12人近复制排除，strict620/6239/249，共7108；同GRAIN all-arm ready
  交集143/1459/61。此audit不冒称全量独立tokenizer重播，也未重算47排除。
- d462713：新history instrument入口；6源头作者(2 Big5/4 MBTI)324布局
  preflight重播通过，identity
  1a3c9ee6111d301ea75aba3cf0e8385dfefc3740c20879fa317ce360211c475b。
  real_neural_qualification_passed=false、forwards=0，不能写成神经验收。
  真CUDA资格计划648forwards，需原72shard queue COMPLETE才准加载。
  真实capture仍未运行；发布scorer未改。
- e70fe8e：旧新交集13653view的当前/历史/donor/cost逐项相同；恢复2569人、
  损失0。comparison identity
  e90dae0b213db7ee8815feeed58f7bc86bb18d622bb18429c38a25625cec559e。
  原99字典作者均使用，最大份额1.94%，逆Simpson90.23。不是独立被试N。
- 新问题：三组原donor的subreddit Jaccard中位数均0。先前soft-cost主要
  实现长度/时间优化，不能叫topic控制完成。冻结额外共享subreddit候选
  (保留原最小代价/时间/来源条件，不改原材料)，44.806s完成21324strict
  view可行性；完整三view作者Big5=448、MBTI=5253、bridge材料=209，合5910。
  新J中位数.083/.083/.100；时间误差中位数1.355/.897/.981变差，单位是
  log1p天数差而不是天。不能把共享subreddit等同共享语义topic或situation。
  identityba0fd483e4c0f53a3cfbbaeca3e68e07acdfd9bbb2671f6ce568f68acb16e362。
  备选donor尚无新token/近复制资格；不得直接投入正式采集或替换主结果。
- vm05_history_metadata_independent_audit_r1独立csv分位数/计数54项误差0，
  另108triplets/324保存token布局不变量检查通过。不是第二遍神经采集，
  也不是独立重做新donor的全候选最优化。
- 区分三种数学对象：未上下文化input token相同应为0差异；当前token
  可以受历史影响，但不能受后置问题影响；task-history交互应读decision。
  不把一个设计上必为0的pre-question task差值当推翻人格reasoning。
- 再发现旧source archive窄glob漏收新history脚本。原档不动，按manifest
  12项源码hash从git blob精确补档r1，SHA06ee0b24186a9108edfaec8e8d79dd50f55427b80a8ad1519b96d93c30e15abd；
  audit_r2另2项补档SHA0d58b50b67a46e858ce8df3ae15c99ecb7ddee500a07597c4ccc7d523e31fe6e。
  都明确retrospective，不冒称事前封存。新history入口已用完整allowlist。
- 新cue retain/purge/count731 AUC=.672089/.634684/.660677；三条件15保存
  fold重播max9.71445e-17，累计本84batch50fold已重播。独立source audit
  first3-cue共10cases/240成员记录/6count配对/7对比，整体PARTIAL。
  随后count732完成，最新11/84，未作其重播；66763/cue count733实测
  99.1%CPU，仍在推进。不能从首个cue-count seed宣布稳定优势。
- Windows61296继续shard20。shard18及19三observer均拷回并逐层验收；
  新shard19 zip330105362bytes，SHA
  9999689c037df9bd1478dfe23e693fdedcbfcf317d1c52d5124777374c81679c。
  SHA/CRC、安全不覆盖解包、00--19全60manifest及作者cache逐一再核验；
  每observer6058人，whole_queue_complete=false。不得把远端活跃20当已备份。
- 最终专项122 tests通过、2项既有SWIG弃用警告。Ruff/差异检查通过；这不是
  全历史仓库测试。新增正式报告VM05_HISTORY_MATERIAL_REPORT。

可复现入口（在独立GRAIN根目录；新输出目录，不覆盖本轮证据）：

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
TOKENIZERS_PARALLELISM=false "$PY" scripts/prepare_grain_source_history.py --views "$R/vm05_qualification_r2/common_views.json" --preflight-dir "$R/vm05_preflight_r1" --profile-dir "$R/vm05_profile_material_r1" --qwen-tokenizer "$R/local_model_metadata/qwen_989aa798" --history-envelope body_and_separator --output-dir "$R/new_history_material"
TOKENIZERS_PARALLELISM=false "$PY" scripts/audit_grain_source_history_material.py --views "$R/vm05_qualification_r2/common_views.json" --preflight-dir "$R/vm05_preflight_r1" --qwen-tokenizer "$R/local_model_metadata/qwen_989aa798" --material-dir "$R/new_history_material" --output-dir "$R/new_history_audit"
TOKENIZERS_PARALLELISM=false "$PY" scripts/qualify_grain_source_history.py --views "$R/vm05_qualification_r2/common_views.json" --material-dir "$R/new_history_material" --audit-dir "$R/new_history_audit" --qwen-tokenizer "$R/local_model_metadata/qwen_989aa798" --output-dir "$R/new_history_token_preflight" --preflight-only
```

下一步：继续完成live源头/observer条件；原queue完成后在新Windows历史目录
做真实仪器资格，再进入完整历史向量采集与同支持源头比较。共享subreddit
候选先补新的近复制/布局核验；语义匹配需完整observer资料，不在片段队列
上挑“有效”作者。直接LLM trait研究对照、strong null、神经/同ready读出、
封头后bridge、最终VM07仍需执行。无bridge outcomes/MEPS量表访问，无新
付费API/模型下载/LLM训练/push，整体goal仍active，目前不需用户新idea。

## 第二十六轮：历史采集可运行性闭环与完整EI cue控制（2026-09-06）

范围：不碰运行中的原Windows队列、不改六个active source评分文件；补真实
history采集与验收路径。同步审计新完成的源头条件。仍不读bridge outcomes
或MEPS量表，不开新LLM训练/付费API/下载，不推送远端git。

- cb851d0：全strict registry采集器，3view x3文本arm x2问题 x3历史条件，
  每人54逻辑布局、90组3x3x1536数组。只对完整输入完全相同的布局去重，
  只对完全相同数组无损alias；所有8读出通道保存，不能据效果挑一个丢掉。
  真实648-forward资格必须先通过，再用同6人生产重播比较2268组状态/候选
  logits。修正technical model signature从summary取，不从manifest猜。
- b3a82d8：独立cache验收与串行有界queue。复核整套manifest、作者集合、
  来源角色、54布局、数组形状/有限性、因果/embedding不变量、forward计数
  与byte预算；即使篡改后重新签hash，结构错误仍拒绝。最多8h/次、24GB；
  仅复用完成且重验的分片，partial一律保留停止，不覆盖/冒充complete。
- vm05_history_queue_preflight_r1：5分片280人、15120逻辑布局、8656唯一
  布局，62.64s后按预算在分片边界暂停。显式60s resume重验原5分片，仅
  0.136s即再次预算暂停。都是forwards0，不是模型资格。该preflight保存
  b3a82d8时源码，后续跨平台修复改变fingerprint，不能拿旧queue继续新代码。
- 实际错误父包命令把token-only r1当qualification-replay输入，返回
  ValueError且未建输出目录、未载入模型。Mac r1-r4资格preflight技术作者
  和布局内容相同；r4 identity08c70e7945dfa1c6e900a92f7cdcfdaadc8d973b2225a2f0cc4a8cbdaf061a75。
- 明确当前token读出与decision读出不同；当前不能受到后置问题影响。
  另查技术panel162对问题布局，current位置和decision位置全部相同。三份
  view均值差的std不等于逐token差的std，后者需跨历史协方差，当前没有采集。
- 新独立Windows目录D:/projects/GRAIN-vector-history-vm05，原root不改。
  初次导入失败暴露ready_comparison eager引入未打包v8_bridge依赖。
  d0206a3把几何依赖移到实际调用函数内，并把artifact helper纳入资格/
  collector源码指纹；新增三个入口在纯导出源码目录的隔离启动测试。
- 第二次Windows预检在读取common UTF-8文本时遇GBK默认编码异常。
  adb66fd为history/相关artifact读取显式指定UTF-8，静态回归测试覆盖所有
  相关read_text，不重写原文、不借截断或丢弃失败作者绕过。两次失败均发生
  在模型载入/创建实验输出前；它们是工程可移植性问题，不是神经科学结果。
- 当前Windows token preflight r3 COMPLETE，324布局、8.312s、forwards0；
  identity2ac2531505a3f6b0ead0843140e52f69c5d6a7f64a937d7cccfe69287386189c。
  Mac/Windows代码指纹和semantic layout hash完全一致：
  b3d2368a760d54661f547d4bd9c4958c19308646177c3ccc5ee575033765a45b。
  JSON原始byte hash因平台换行不同，未把byte不同误当token不同。
  archive材料SHAed3c29a202b5a9125c7a2cf1e73fccd41a566e0a1c02240061acf8db69dcdebb；
  当前代码update_r3 SHA11256afddc3bcb51231365ad147496bc98bc548c8a439f15b22c0d6052366437。
  只复制既准许的PANDORA材料/源码/本地tokenizer，不含MEPS/心理值/env。
- 新cue count732-735四条件20fold保存模型复算max9.80119e-17；两组EI raw/
  cue全部14控制现累计70fold重播。独立source审计336条选择记录、10count
  配对、8对比，整体仍14/84 PARTIAL。其他轴/安慰剂不能从这14条推定完成。
- 8c5a050扩展原EI posthoc gap脚本支持预定arm，要求7条全五seed且audited
  source_identity逐一对应，不选择个别seed。cue retained=.672089，五count
  平均=.651767，purge=.634684。pooled purge-count=-.017083
  CI[-.027455,-.007058]；meanfold=-.009176 CI[-.018336,-.000234]，但2/5
  折反向。这是来源敏感性提示，不是已隔离的泄漏/人格因果作用；raw的
  meanfold CI仍跨0，保留多重检验、拟合条件固定与fold异质性边界。
  vm05_ei_cue_count_gap_r1 identity9a4dd4c43e2d9556f3b5cb1a447105caa9e312368cc703167a81b53fa013e966。
  sklearn独立复算42观测量max1.94289e-16；未另重建全部bootstrap draws。
- Windows原queue61296最新到shard22 BGE，远端66/72已完成；本地只到20。
  shard20压缩包309111711bytes，SHA61b9bacedc7959a202a0d5684eb7929975362e628a47978935b3c9e5b9b0b762。
  CRC/安全不覆盖解包/00-20全部63manifest与cache重验通过，每observer6360人。
  修正本地备份状态各计数字段一致。21尚待备份，不能把远端完成当本地齐全。
- 结束前Mac68908执行EI placebo retain，14/84完成；六个active评分文件
  SHA仍全相同。61专项tests通过（含独立导出启动/历史/ready/readout回归），
  两个旧SWIG警告。Ruff/diff检查通过；不是全历史仓库或真实CUDA通过。

正式报告新增VM05_HISTORY_COLLECTION_REPORT.md，更新
VM05_EI_COUNT_MATCHED_SOURCE_REPORT.md。当前代码资格和真实队列仍需实际
运行：待原72shard COMPLETE -> 六作者648forward资格 -> 生产2268组重播 ->
据真实时间/cache开启有界全registry历史采集。然后同支持源头/同ready比较，
直接LLM trait研究对照与强null，封头后bridge，最后综合VM07。没有改变目标
为“多通过几个小测试”，现在仍有明确可执行工作，不需用户新idea。

## 第二十七轮：完整向量、真实历史仪器与直接trait读出（2026-09-06）

原Windows aa51b06队列72/72完成。21/22/23备份经过SHA、CRC、安全不覆盖
解包及全部72manifest/cache重验，每个observer精确7,266作者无遗漏/重复。
local_backup_status=VERIFIED_COMPLETE_BACKUP，remote_queue_status_final已存。
最后23包315446739bytes，SHA
8b4b75457faace6e7f7243dc41d6c4c492de999a3c399e1dbf881476e3661f7d。

- 1873cc3/e2095ae：冻结直接Big5/MBTI提示，完成有界作者采集及独立概率
  重算；保存candidate logits、全词表lognormalizer、条件概率/熵、概率
  质量、首token合规率。标准softmax导数不是人格轴。7266人278397逻辑/
  143045唯一提示全支持，无截断/排除；176.76s。独立角色/网格重数通过。
- 实际预检发现direct与history的tokenizer清单字段不一致，新增direct
  schema校验修复，不改历史材料。Mac r2 / Windows r1相同6人234提示、
  代码与语义记录一致。direct新目录157文件78316630bytes；SHA
  5554b7a5482f9dd6d7d81c30ac9d0401b8ce1f078dde087bd4e0ebe1b27b362c。
  一次解包白名单漏列requirements，在解包前停止，明确修正后完成。
- 原72完成才执行真实history资格：648forward/308.81s/峰值CUDA4.31GB；
  108行五项误差全部0。生产采集器153forward重播324布局，2268组一致；
  另独立别名/索引重建7465608坐标误差0。真实资格、生产重播与两个独立
  审计包均已复制Mac核验，详见VM05_HISTORY_COLLECTION_REPORT。
- 技术panel并非所有输出都为0：raw18view的input embedding对历史变化
  完全不变，block10的current均值18/18发生变化，平均相对L2变化.161826。
  decision task interaction也非零。仅说明中间层确实在使用历史及问题，
  不是人格信号提升16%、也不是6作者之外的泛化。记录在
  vm05_history_technical_nontriviality_r1；不按跨层原始L2大小选赢家。
- 根据6作者76.91s/11694822bytes，估算全history约25.3h/13.85GB，非保证。
  保持128shard/7108strict，每shard2400s/500MB，总24GB，每次8h。首次
  detached SSH子进程未存活，空log保留，未当运行成功。改为保持连接的
  受控父进程：vm05_full_history_vectors_r2，queue65840/首child64464，
  CPU/内存与5/56首分片作者真实推进已确认。完整采集仍在运行。
- history资格与全量之间串行完成direct真实资格：234逻辑提示、86唯一
  输入每次、重复共172forward，83.72s；native重播误差0，独立概率复算
  max3.55e-15。首token合规1.0、candidate mass中位.999441，非语义正确率
  或人格效度，无目标标签读取。direct权重退出后才启动全history。
- 全72 cache真实source读出preflight：Big5-only624/EI reference-free6294
  作者，各3072列equal-view block10；27.53/44.21s，0标签/0拟合。保存于
  vm05_complete_neural_{big5,ei}_readout_preflight_r1，不能称预测结果。
- 新EI placebo前4条件20fold重播max9.71445e-17；累计18条件90fold、432
  选择记录/12count配对/14对比独立审计。live到20/84，最后seed735尚在跑。
  余下3count须验收齐再报告完整安慰剂/跨mask差异，不挑部分seed。
- 10 direct专项tests通过；direct/history/readout组合51tests通过。
  c93d117将后续source/same-ready JSON读取设UTF-8后，影响范围25tests
  再通过（不能相加）；两个旧SWIG警告。六个active评分文件SHA未变化。

更新主计划、剩余要求及历史报告，新增VM05_DIRECT_TRAIT_READOUT_REPORT。
下一轮继续真实全量history、源头神经/同ready比较、direct语义/order与
完整采集、强null、封头后bridge和VM07。不改released scorer，不读bridge
outcomes/MEPS量表，不新LLM训练/下载/付费API/push。goal active，当前仍有
明确可执行工作，不需要用户提出新idea。

本轮最终进程快照：history queue65840/child64464均实测存活，首分片推进
35/56作者；Mac完成21/84，已转入SN_cont_raw_retain_731（75233）。EI三
arm完整家族都已执行，但最新三个placebo seed的独立审计/15fold重播和
整家族gap报告留作下一执行块；当前仍只把18条件/90fold列为已审计。
