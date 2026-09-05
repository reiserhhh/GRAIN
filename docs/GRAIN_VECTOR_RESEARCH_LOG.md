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
