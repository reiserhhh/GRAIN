# GRAIN：LLM 向量表征与人格推论机制实验计划 V1

日期：2026-09-05。状态：`PLANNED_NOT_EXECUTED`。

这是研究设计和实施交接，不是已完成实验、注册封印或新发布版本。背景与历史证据见[项目回顾](GRAIN_PROJECT_RETROSPECTIVE_20260905.md)。

## 1. 目标与研究对象

目标：解释模型如何从同一作者的内容、历史和对话中获得可用于人格判断的信息，并将可重复的机制接回 GRAIN 的个体语言纹理分析。

需要同时保留两个对象：作者产生的语言过程，以及模型在阅读语言时发生的计算过程。我们干预后者，观察它如何利用前者；模型内部因果作用本身不证明人类心理因果。

三个平行结果轴为：

1. **内部机制**：哪里编码了什么、上下文怎样改变编码、哪些计算参与输出。
2. **测量性质**：新对象是否跨片段/任务可重复，是否能在固定参照下比较作者。
3. **外部连接**：哪些对象与 Big5、MBTI 或实际存在的行为/状态指标相联系。

内部机制实验可以在外部指标尚弱时继续；某个方向的外部关联失败，也不等于所有向量机制不存在。

## 2. 理论接口与符号

记作者为 u，文本片段为 i，人的记录时间为 t，网络层为 l，序列位置为 p，模型为 m，判断任务为 q。H_u 为已记录历史；应用层检索/摘要器 R 将历史变成 C_u。W_m 是固定模型权重。

\[
C_u=R(H_u;q,b),\qquad
H^{(l)}_{u,i,q,m}=F_m^{(l)}(W_m;C_u,x_{ui},q).
\]

b 为文本预算。输入 token embedding、上下文化 hidden states、池化后的句段 embedding、应用层记忆、KV cache、模型权重中的知识必须分别记录。l、p 和回答生成步不是人的时间 t；KV cache 也不是独立的长期个人记忆。

模型内部子层以具体 checkpoint 的实现为准。典型残差结构包含上下文依赖的注意力写入与非线性 MLP 写入，但不能只观察 attention 热图就认定某段材料“被用于人格推论”。

### 2.1 固定位置的上下文效应

在同一模型、同一层、同一当前片段、相同 suffix/位置安排下：

\[
\Delta h_{u,l}(x,q)
=h_l(x\mid C_u,q)
-\mathbb E_{v\sim\mathcal P_{\mathrm{ref}}^{\mathrm{matched}}}
h_l(x\mid C_v,q).
\]

匹配只使用可观察条件、长度、语言、时间窗和主题代理；不使用测试人格标签。同人历史必须排除当前片段、近重复和回答泄漏。该差值是实验条件对模型表示的效应，原表示与全部条件同时保留，不作为删除主题的预处理。

对固定 token、固定位置的初始 embedding，未上下文化前应没有历史效应；它是负控制，不应被期待能识别作者。内容 token 的平均输入 embedding 与固定 suffix 的逐层 hidden state 是不同统计对象，必须分列。若后者在输入层已经出现历史差异，先检查位置、模板、mask 和采集站点，而不是宣布发现机制。

再比较“人格判断任务”和“行为观察任务”各自的上下文效应，得到模型向量层面的 difference-in-differences。它只能表示这两个任务的差异响应，不能自动解释为纯人格方向。

### 2.2 内部信息的可读性

\[
\hat y_u=b+\beta^\top z_u,\qquad
z_u=A(H_u^{(l)},\Delta H_u^{(l)}).
\]

A 为固定聚合器，先用 mean、分位数和片段间变化等小集合。线性 Ridge/logistic probe 是信息检测器，不把 probe 的训练称为 LLM 内部学会了人格。选择层、维数、方向与正则仅使用开发作者/inner folds。

原始 hidden size 很大；先报告 rank<=min(16,floor(n_train/10)) 的正则投影敏感性。投影在训练侧拟合；不能以高维 CCA 在 n 很小时获得的近完美相关作为证据。

### 2.3 固定输出统计量

人格判断首先只作实验终点。主输出用完整候选答案序列的条件 log-likelihood，显式记录 tokenization：

\[
s_k(P)=\log p_m(a_k^+\mid P)-\log p_m(a_k^-\mid P).
\]

P 为完整提示，a+、a- 为事先确定的高/低行为描述或两端标签，极性以 codebook 固定。用平衡 wording 和候选顺序复测。量表 1–5 数值可以作为辅助：对五个候选答案归一化后取期望；它是受限答案集中的模型统计，不是已校准的人格量表值。

无证据 prompt 的 s0 单独报告；s-s0 只去掉该设计下的输出基线，不声称移除了所有刻板印象。自由生成 JSON/解释只作输出对照。提取特征发生在答案生成前，不能从已经写出人格答案或解释的 token 中训练 probe 后称为信息发现。

### 2.4 证据组合

\[
I_k(A,B)=s_k(x,A,B)-s_k(x,A)-s_k(x,B)+s_k(x).
\]

A/B 的缺席用等长对照块实现，且交叉顺序；报告 log-odds 和概率两种尺度。I 是该函数/尺度上的非加性交互，不称为信息论 PID synergy。若有收益，随后才估计更复杂的高阶交互。

### 2.5 内部干预

先做模型自然可产生的匹配激活替换，再做子空间强度扫描。对同层、同站点的匹配 donor：

\[
h'=h+\alpha P_S(h_{\mathrm{donor}}-h),
\qquad \alpha\in\{0,.25,.5,1\}.
\]

S 为开发侧候选子空间，P_S 为其投影。主要效应为 s(h')-s(h)，配同维度、同范数随机子空间。完整替换、原激活恢复、自身替换 identity control 都要有。

替换同时可能改变其他概念，需测内容任务/非目标 trait/一般语言损伤。LEACE 可作为线性信息擦除辅助，但其线性保证不等于所有非线性信息均被擦除。梯度/Jacobian 仅作局部敏感性筛选，候选需再通过干预。

### 2.6 接回群体参照

\[
m_{u,k}=A_k(\{\Delta h_{u,l},I_{u,k},\Delta s_{u,k}\}),\quad
\tilde m_{u,k}=\frac{m_{u,k}-\mu_{k,\mathrm{dev}}}{\sigma_{k,\mathrm{dev}}}.
\]

这些是模型条件下的匿名候选测量，不在同批待测作者上重估均值/方差。与 GRAIN 既有 choice、表达分布、动态和残差分别报告共享及增量部分；内部推断稳定性不能直接继承为人的 trait 稳定性。

## 3. 机制问题清单

| 机制问题 | 操作 | 成功时能说明什么 | 不支持时的下一步 |
|---|---|---|---|
| 一般 embedding 是否压缩掉可用线索 | 同预算的 BGE/E5、LLM 早/中/晚层、mean/非 mean 比较 | 在给定读出器下某层保留更多可用信息 | 检查读出和数据支持，不能宣布文本没有信息 |
| 个人上下文是否修改当前证据 | same/other/none 历史 | 本人历史产生可重复的模型效应 | 查长度、主题和材料不足 |
| 模型是否仅检索自我描述 | 掩码直接自我 trait 声明、保留实际行为；互为对照 | 区分自我叙述检索与间接行为推断 | 分别报告信息来源，不全称人格理解 |
| 主体/否定/愿望/引述是否被正确归属 | 最小配对句组与角色交换 | 语义归属计算的功能证据 | 模型能力不适合当前任务，换固定模型或任务 |
| 历史摘要是否只是提示了结论 | 原文/随机片段/检索/不含 trait 名的事实摘要 | 压缩和选择改变证据使用 | 查摘要新增/遗漏事实与量表答案泄漏 |
| 多条线索是否相互解释 | A/B/AB 及顺序交叉 | 非加性作用及其层位置 | 更简单线性读出足够，保留结果 |
| 反证是否更新而非覆盖判断 | 同证据不同顺序、追加否定/修正 | primacy/recency、累积和修正的响应曲线 | 检查模板和证据重复，避免叫人类认知机制 |
| 方向可读是否代表被用 | probe 与 activation intervention 对照 | 指定模型通路参与输出 | 探查未使用信息或其他读出路径 |
| 作者与 assistant 的信息是否混淆 | 用户-only、带角色全对话、匹配 assistant-only 对照 | 互动背景和作者证据可分 | 修正角色归属，不能把 assistant 文风归人 |
| 语义效应还是人物先验/输出标尺 | 无历史、标签极性/描述改写、身份元信息消融 | 局部分离证据与先验/标尺效应 | 先修任务读出，不扩增人格维度 |
| 模型机制是否跨作者/提示存在 | 冻结方向后跨作者、paraphrase、第二模型 | 模型族内或跨模型的可复用机制 | 报告模型/任务特定效应 |
| GRAIN 结构是否遗漏此信息 | GRAIN、内部向量、二者融合使用相同统计头 | 相对既有观察器的增量信息 | 记录冗余，不因可视化新奇宣布新因子 |

## 4. 数据分工与访问边界

| 数据 | 第一用途 | 后续用途 | 特别注意 |
|---|---|---|---|
| 可控合成最小配对材料 | VM01 验证信息与干预仪器 | 多事实、角色、顺序的机制 stress test | 测模型能力；不训练“合成人格”并宣称真人有效 |
| PANDORA 原评论、作者与元数据 | VM02–04 主要机制样本；同人历史/新片段 | Big5/MBTI 外部读出，官方 folds 上分析性复现 | 原样文本角色/时间/引用/重复必须保留；量表报告线索另做 paired ablation |
| MEPS `decrypted_output/` 的 46 人 | 冻结模型与方向的同人任务/自由对话配对 | 模型观察条件对照 | 不拟合人格方向；不读 withheld 问卷、不用于训练或外部效度主结果 |
| Essays | 冻结观察器的异域文本检验 | 已允许标签部分的次级外部联系 | 没有同人长历史；未知时间不能构造真实重测 |
| X 市场文本 | 模型/语言/情境机制迁移 | GRAIN 纹理的域内比较 | 无测得 Big5/MBTI 就不生成“真实标签”；市场关联是别的任务 |

本计划本身不读取新的私密文本或量表。MEPS 邻表登记的是 AI 心智归因、依恋、信任、自我开示等；情绪记录另有 PANAS/POMS。若未来另开其外部联系，须先准确列字段、计分、时点和配对关系，以独立附录改变当前用途，不把“有关心理信息”当成 Big5。

旧研究样本计数不一致是不同路径的事实：Big5 1,401/official-N 1,402/NP 1,386；MBTI 9,042 与非 bridge 8,678；bridge 364/375/377。VM00 必须从实际输入记录每个过滤步骤，不把某个数硬编码成所有实验的母体。

### 4.1 分割与材料构造

1. 使用已经允许研究的开发材料；保留历史 test/lockbox 接触记录。不能把旧 bridge 或已多次看的 PANDORA 随机分一次就称为完全新确认集。
2. 验证器械可先用 8 个不进入效应估计的技术样本。第一真实 pilot 用 24 作者；后续上限 240 作者：144 discovery、48 calibration、48 mechanism holdout，按作者确定并记录选择规则。
3. 240 并非充分统计功效保证；完成 pilot 后报告可检测区间宽度和有效作者数。预算不足时缩小研究问题，不制造“相关低于 .5 就失败”的门槛。
4. 自然条件主实验：每人选择相互不重叠、非近重复、尽量 source-disjoint 的历史与当前材料。时间真实可用则历史必须在前；否则称“上下文集合对照”。
5. 最小 cell 为 2 个不重叠当前片段 × 3 个历史条件（本人/匹配他人/none）× 2 个任务（行为观察/trait），每作者 12 个表示采集条件。24 作者为 288；240 作者为 2,880。此数只计一次主 prompt prefill；多候选答案、多个 trait、重复 donor、改写和干预另计模型调用及 token 预算，不能把它当完整运行成本。
6. 每人只有一个真实历史来源不能冒充独立重测。匹配他人不使用标签；至少 2 种预定 donor 规则作敏感性，后续 donor 重复不能当新被试。
7. 用共同 anchor 片段补测“历史造成的相同刺激响应”。anchor 来自事先固定的合成中性行为材料，不是被试说过的话；结果标为模型反应，不算真人观察。
8. 初期一模型一语言 block，禁止直接混合不同模型的坐标；日语用适合该语种且能力预检合格的模型。

## 5. 顺序、任务包与可验收输出

### TASK-GRAIN-VM00：数据和模型仪器预检

预计 0.5 天。只检查目录/字段/访问范围、模型能力、hooks 和预算，不计算心理结论。

输出 `data_inventory.json`、`label_codebook.json`、`model_manifest.json`、`split_manifest.json`、`preflight_report.md`。固定 model revision、tokenizer、chat template、dtype、量化、设备、最大长度、截断比例、评分答案和极性。明确只能模型内部干预还是仅 API 行为对照。

验收：输入 IDs 连接一致；用户/assistant 角色没有混入；当前片段不在历史；分组 split 零重叠；旧 lockbox 不被加载；模型 logits、无 hooks/有 hooks 原值回放在预定数值容差内一致；重复前向与恢复控制通过。轻微浮点误差按 dtype 校准，不能用宽容差掩盖不同模板。

### TASK-GRAIN-VM01：可控机制材料与内部状态 smoke

预计 1 天。4 类最小对照（主体归属、否定/愿望、情境归因、跨片段组合），各 12 个 base scenario、4 个条件变体，共 192 个 prompt；另有最少 8 个 no-effect 和 8 个 sham 控制。至少一个完整 scenario family 留出，不让同一故事的改写泄漏。

输出 `contrast_manifest.json`、`pair_validation.csv`、`layer_metrics.csv`、`hook_replay_tests.json`、`capability_report.md`。材料不含明确量表分数或类型答案；生成者与检测者任务分离，保存结构化操纵因子。

验收：先能区分被操纵的已知语义事实，再对其内部位置作解释；报告方向一致率、配对效应和 bootstrap 区间。若模型连事实主体都不能区分，当前模型/任务进入 revise，不据此评价 GRAIN 或人的人格。

### TASK-GRAIN-VM02：真实文本逐层可读性图谱

预计 1–2 天。24 作者 pilot，单模型抽 6 个预定层/阶段和固定 suffix 位置。比较同预算 BGE/E5、LLM 输入层/中间层/末层以及既有 GRAIN 特征；保留 raw 与聚合视图。

输出 `layer_observability.csv`、`pooling_comparison.csv`、`author_correspondence.csv`、`representation_similarity.csv`、`layer_map.png`。作者几何使用独立文本 view 的匹配/CKA/距离排序；不在不同作者无对应的矩阵上强算 CKA。

本包先无标签确定采集器可重播和差异可测。外部 probe 留到 VM05，在探索中不偷偷按 Big5挑层。验收是得到可追溯的表征图和操纵效应；所有层都无可读性时先检查任务、长度及表示位置。

凡比较“表示能力”，同时报告内容 token 聚合和固定判断 suffix 两种视图，避免将含内容的 BGE 向量与不含内容的初始 suffix 向量作不公平能力比较。初始 suffix 只用于历史效应负控制。

### TASK-GRAIN-VM03：上下文、记忆与任务的交互

预计 1–2 天。使用同一批作者和冻结采集器；完成本人/他人/none 的主对照，再在最多 12 个开发案例上探索原文、等长随机历史、冻结检索和事实摘要。

输出 `context_effects.csv`、`task_context_interactions.csv`、`memory_ablation.csv`、`evidence_interactions.csv`、`context_displacement.png`。同时记录真实信息条数、摘要覆盖、token 长度和位置。

先用 preselected matched-current 历史设计，保证同一个当前片段不改变。自然条件中严格的等长未必可达：报告匹配误差，再在精确配对合成材料上重复；不能把 padding token 视为完全无语义的替代品。

验收：给出上下文效应是否超出格式/长度对照、是否跨两个当前片段、是否依赖某一种任务。不存在效应也应明确是总表示不变、probe不可读，还是输出读出不变。

### TASK-GRAIN-VM04：小规模激活替换与功能定位

预计 1–2 天，只有 VM01 器械通过后开始。开发侧最多 2 个候选机制、2 个层/位置、2 个子空间 rank（2/4）；在最多 32 个配对案例干预，每条件保留同维随机、sham、全替换和恢复。计算预算见第8节。

输出 `intervention_effects.csv`、`random_subspace_controls.csv`、`collateral_effects.csv`、`restoration_tests.json`、`mechanism_localization_report.md`。在 calibration 上固定候选后，仅对 holdout 做一次主要效果读取。

验收：sham 近数值零；候选作用超过匹配随机作用；恢复使结果返回原状态；非目标任务受损程度被量化。归一化 recovery 只在原始对比明显大于数值误差时报告，否则报告绝对差，避免除以小数产生“巨大恢复率”。

### TASK-GRAIN-VM05：外部人格连接与 GRAIN 增量

预计 1–3 天。使用获准的 PANDORA 标签，各 trait/axis 都报告；TF 可作历史阳性参照，但不只选择它来讲成功。模型和读取方向预先固定后再使用各外部任务 train/inner-fold 来拟合统计头。

核心比较为：GRAIN、传统 TF-IDF/BGE、LLM pooled、内部上下文效应、GRAIN+内部效应，共同材料预算和同类正则头。记录原文直接 LLM 打分为研究对照，不替换正式确定性计分器。

Big5 用 Pearson/Spearman/MAE/RMSE/校准偏差，MBTI 用 AUC/PR-AUC/balanced accuracy/Brier/校准；Brier 只对训练侧校准的概率用，不对 raw Ridge 分数用。按维度报告 Δ 与作者 bootstrap CI，不把标签 0/1 当作“概率真值连续尺度”。

若复用 corrected H2：Big5-only 训练相应头、MBTI-only 训练相应头、bridge 用于结构评估；bridge 不参与方向/权重选择。对于主要个人预测任务，可按对应官方 trait/axis folds；不能把 per-axis folds 强行拼成一个四轴统一 split。所有内部选择与外部评估分开。

输出 `external_link_by_trait.csv`、`paired_increment.csv`、`calibration_report.csv`、`grain_increment_report.md`。报告 .3181 历史官方 package 作为背景；只有同用户、同文本、同辅助信息预算的重跑才能称新方法胜过它。

### TASK-GRAIN-VM06：MEPS 与异域的冻结机制复测

预计 1–2 天。冻结模型、方向和统计规则后，MEPS 46 人只做现有许可的无标签固定条件/自由对话比较。全对话 arm 单独标注为新增观察设计，assistant 只作上下文；对直接 MEPS 答案保留“可能受 AI 辅助”标志。

Essays 检验语义/内部表示的结构，不能凭空产生历史条件。X 市场文本检验语言和领域敏感性。只对有配对对应的表示做跨模型/跨语料对齐，不给无配对不同作者硬配对。

输出 `meps_fixed_free_mechanisms.csv`、`domain_sensitivity.csv`、`model_prompt_stability.csv`。没有支持时区分语言能力、材料短缺、测量器适用域和真实机制差异。

### TASK-GRAIN-VM07：综合与可复用观察器

预计 0.5–1 天。整理机制地图，形成冻结观察器候选，不自动变更 GRAIN 当前发布 scorer。

输出 `MECHANISM_ATLAS.md`、`mechanism_evidence.json`、`GRAIN_VECTOR_OBSERVER_SPEC.md`、复现命令、剩余机制表。只有得到新的功能作用或增量证据才发展新公式；等效旧特征也是可交付结论。

## 6. 统计与判读规则

- **单位**：重采样/置换以作者或合成 scenario family 为单位；层、token、问题、同一作者的多条文本不增加独立 N。成对网络指标需要重采样节点而非把所有边当独立样本。
- **分工**：广泛探索用全效应图和 BH-FDR 辅助；最多 2 个机制进入冻结主要检验，主要 family 用 Holm 或预先确定的 max-statistic。报告所有探索，不把事后挑选写成先验。
- **小效应**：不设置通用 r>.5、AUC>.6 门槛。主问题是方向、区间、随机对照、重播及跨作者重复。区间跨零记录 underresolved，结合实际区间给下一轮所需 N，而不称不存在。
- **主效应和交互**：输出漂移、总表示距离、任务特异性、外部预测增量各有独立估计量；不从其中一个推出其他全部成立。
- **可靠性**：至少两个文本 view、两个任务措辞；评分器扰动与人的状态变化分开。共同条件验证采用现有任务和模拟操纵，不要求真人每天重复同话题。
- **负控制**：作者历史置换、标签置换（完整重跑选择步骤）、无个人证据、格式/长度匹配、sham patch、随机子空间、恢复、答案极性检查。
- **几何**：报告范数、有效秩、长度影响和 CKA/距离结构；大维度、低 N 的最优旋转可能过拟合。跨模型对齐只在开发配对刺激学习，再测留出配对刺激。
- **解释**：模型的解释文本是待检验输出，不是其真实内部思维的逐字记录。必要性/恢复只是指定干预的作用，不是完整生物人格机制证明。
- **结果分支**：有内部作用无外部增益=模型推断机制；有外部可读性无干预作用=表征信息候选；两者都有且可重复=GRAIN 新观察器候选；仅格式/量表提示有效=任务读出偏差；两者皆无且区间窄=收窄该候选后换机制。

## 7. 计划代码与输出接口

以下文件均为**待实现**名称；本次规划不表示这些脚本已可执行。

| 模块/脚本 | 核心职责 |
|---|---|
| `grain_research/vector_mechanisms/data_views.py` | build_author_views、match_context_donors、assert_group_separation |
| `grain_research/vector_mechanisms/trace.py` | encode_with_trace、record_site、replay_and_restore；固定层/位置/模型 |
| `grain_research/vector_mechanisms/readout.py` | score_candidate_sequences、fit_grouped_probe、calibrate_train_only |
| `grain_research/vector_mechanisms/contrasts.py` | context_contrast、task_context_interaction、evidence_interaction |
| `grain_research/vector_mechanisms/interventions.py` | patch_activation、matched_random_subspace、restore_hooks |
| `grain_research/vector_mechanisms/analysis.py` | cluster_bootstrap、group_permutation、geometry_similarity、incremental_cv |
| `scripts/run_grain_vector_mechanism_preflight.py` | VM00 |
| `scripts/run_grain_vector_mechanism_controls.py` | VM01 |
| `scripts/run_grain_vector_mechanism_atlas.py` | VM02–03 |
| `scripts/run_grain_vector_mechanism_interventions.py` | VM04 |
| `scripts/run_grain_vector_mechanism_external_links.py` | VM05 |
| `scripts/run_grain_vector_mechanism_transport.py` | VM06 |
| `scripts/summarize_grain_vector_mechanisms.py` | VM07 |
| `configs/grain_vector_mechanisms_v1.yaml` | splits、模型、层、任务、预算、阶段开关 |
| `tests/test_grain_vector_mechanisms.py` | 分组、配对极性、无作用/恢复、序列评分、synthetic interaction |

`grain_research/` 作为独立研究包，避免当前 `grain_core` 的 legacy alias 搜索路径吞掉新模块；观测器稳定后再设计正式导出。

每次运行保留：

```text
results/grain_vector_mechanisms/<run_id>/   # 私有/ignored
  manifest.json
  data_inventory.json
  split_manifest.json
  model_manifest.json
  contrasts.jsonl
  trace_manifest.json
  private_cache/                          # embeddings/activations；不得进 git
  metrics/
  figures/
  report.md
reports/grain_vector_mechanisms/           # 只放检查过的聚合结果
  MECHANISM_ATLAS.md
  mechanism_evidence.json
```

manifest 包含 git SHA、材料哈希、模型权重 revision、tokenizer/template hash、量化/设备、seed、任务、观察层/位置、访问范围、实际有效 N、失败与跳过原因。私有 token IDs 和 activations 可能携带原文信息，不能因为是数字就公开上传。

## 8. 算力与时间安排

默认沿用用户指定路由：训练/重负载机器为 Windows `192.168.1.20`，总显存约12GB、预期可用小于10GB；评估机器为 Linux `192.168.1.30`，Intel GPU 16GB，已有 llama 服务。此处是约束，不是本日连通性或速度验收。

白盒激活实验需要直接模型运行时的 hooks；普通 llama/Ollama API 不能被假设提供这些接口。因此 `.20` 承接 hooks 采集/干预，`.30` 先作输出端评估与轻量重播，具体 Intel backend 能力由 VM00 核对。Mac 负责目录、统计、图表和文档。

本地已发现 BGE/E5/multilingual-E5/Mistral 缓存目录，先验证完整性。首选能在现有硬件加载、通过日英事实预检的一个开放权重模型；Mistral-7B 是可核查候选，不预设其日语能力和显存足够。若不满足，按模型官方资料选 1–3B 级候选；量化主结果需在小样本 FP16/BF16 对照下检查机制方向是否保持。

初始 batch=1、上下文上限 2,048 tokens，按输入语言分别记录截断。峰值显存超过空闲预算80%就缩小文本/层缓存或换模型；不在发生 OOM 后反复全量重启。第一轮不训练 LoRA，不训练全规模 SAE，不启动全层全头搜索。

只保存选定位置的 residual stream 和必要块输出，不保存全部 attention 矩阵。缓存估算：forward次数 × 层数 × 保存位置数 × hidden size × dtype字节数。例如 2,880 × 6 × 4 × 4,096 × 2 约0.57GB；若保存2,048个位置则约290GB，必须流式提取和删选。默认持久 cache 预算10GB。

| 时段 | 工作 | 明确交付/决策 |
|---|---|---|
| 第1天 | VM00 与 VM01 技术最小批 | 确定可运行模型、可访问材料、真实耗时；决定是否扩展 |
| 第2–3天 | 完成 VM01；VM02 24作者 | 逐层表示图、正负控制、是否值得进入真实上下文实验 |
| 第4–5天 | VM03 + 小批 VM04 | 2个以内候选机制与功能证据，或定位到任务/材料问题 |
| 第2周 | 按 pilot 可行性扩展到最多240作者，VM05–07 | 外部联系、冻结机制复测、GRAIN 观察器规范 |

以上为工程估计；VM00 分别实测 prefill、完整候选序列评分和一次 patch 的耗时/内存后重新预算，记录总调用与总 token，不只计 prompt 数。pilot 墙钟上限6小时（不含下载），后续单包上限8小时；每完成一个作者持久 checkpoint，达到预算输出 partial report 而非丢失结果或无界继续。初次 VM04 干预算例预算<=4,096 实际模型调用。第二模型和多语言都在主模型机制可重复后追加。

职责：本会话负责路线、数学、实现审计和文档；如沿用双 agent，理论规划限一个独立规划角色，其余由当前工程 agent 完成。使用 Codex 协作，MLCS 不启动。当前任务只交付规划，不派发新 agent、不启动远端训练。

## 9. 文献与方法对应

| 来源 | 可借用部分 | 本项目适用边界 |
|---|---|---|
| [Yang & Oshio, 2026](https://onlinelibrary.wiley.com/doi/full/10.1111/ajsp.70102) | 历史互动中的线索与自评对应，互动关系影响可见信息的假说 | 未直接分析聊天日志；不是内部机制证据 |
| [Funder, 1995](https://www.simine.com/407/readings/Funder_1995.pdf) | relevance / availability / detection / utilization | 给实验分工，不将其概念框架当已估计的神经公式 |
| [Peters, Cerf & Matz, 2024, preprint](https://arxiv.org/abs/2405.13052) | 互动 elicitation 设计改变人格判断信息 | 情境产生了不同材料，不能把性能差全部归于评分 reasoning |
| [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | 上下文依赖 attention 与逐层计算 | 具体 hook 站点必须服从实际 checkpoint 架构 |
| [Zou et al., Representation Engineering](https://arxiv.org/abs/2310.01405) | 群体表征、方向与干预 | 概念方向不是已证实的人格轴 |
| [Elazar et al., 2021](https://aclanthology.org/2021.tacl-1.10/) | 区分可读信息与模型使用；amnesic intervention | 去除信息可能有副作用，需要随机和功能控制 |
| [Meng et al., 2022](https://arxiv.org/abs/2202.05262) | 激活扰动与恢复定位 | 原工作研究事实知识，不能把事实定位结论直接推广人格 |
| [Belrose et al., LEACE](https://arxiv.org/abs/2306.03819) | 最小改变下的线性概念擦除 | 线性不可读不等于非线性不存在；本计划只作诊断 |
| [Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a.html) | CKA 和跨表示比较 | 需要相同刺激对应，并处理高维小样本和模型基底差异 |
| [Turpin et al., 2023](https://arxiv.org/abs/2305.04388) | 解释文本的 faithfulness 问题 | 用行为/激活干预检验，不能把解释当内部记录 |

## 10. 第一条执行指令

实现并运行 `TASK-GRAIN-VM00` 和 `TASK-GRAIN-VM01` 的技术最小批。先用可控、无真人标签材料验证层/位置采集、评分极性、配对信息和激活恢复；提交模型及数据 inventory、耗时/内存和器械结果。通过后按本计划进入24作者上下文 pilot。每一步结果都回答一个机制问题，并明确映射到 GRAIN 的 Where/What/Who/When；不要以新增因子数或跑了多少版本作为进度指标。
