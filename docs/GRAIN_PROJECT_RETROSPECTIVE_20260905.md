# Project Persona 到 GRAIN：研究回顾与下一阶段机制地图

日期：2026-09-05。回顾状态：`SOURCE_REVIEW_COMPLETE`。新机制计划随后开始执行；最新状态以[研究日志](GRAIN_VECTOR_RESEARCH_LOG.md)为准。本文以下历史阶段表描述制定计划时的基线，不作为执行仪表盘。

本次回顾结合本会话的连续研究意图、两个本地仓库的路线文档、主要报告和当前代码入口。它不声称重新运行了历史实验，也不声称穷尽所有版本的代码审计。数值属于各自原协议，不能合成一个跨版本排行榜。

配套执行设计：[GRAIN 向量与人格推论机制实验计划 V1](GRAIN_VECTOR_INFERENCE_MECHANISM_PLAN_V1.md)。

## 1. 始终不变的研究目标

Project Persona 的出发点是从自然语言理解个体的人格与心理状态。后来形成的 GRAIN 是支撑这种解释的测量方法论：在明确的参照群体和观察条件下，提取可比较的个体特征，再用行为或既有心理指标确定其意义。项目当前材料不需要被强行制作成一套完成的量表。

历史 SUICA 名称已由 GRAIN 接替；旧文件名、实验编号和冻结报告作为证据定位符保留。理论 V6/V7/V8 与软件版本 v1.1.1 是两套编号。

本次方向修正来自用户的明确问题：LLM 已能从个人对话和历史中读出与心理量表有关的线索，那么这种读取在内部向量计算中怎样发生？GRAIN 已能观察语言纹理，下一步要把“模型如何观察作者”本身纳入可测量对象。

## 2. 历史阶段、已得到什么、没有得到什么

| 阶段 | 已有工作与可保留结论 | 未回答的问题 | 主要定位 |
|---|---|---|---|
| 官方人格预测与 LoRA | 数据准备、用户级聚合、官方 Big5 folds、MBTI 分轴、base control 和生成式评估已建立。早期 Big5 pooled r 约 .1005，MBTI 200-iter balanced probes 接近 chance。 | 这些训练设置的失败不证明 LoRA 永远无效，也没有定位预训练 LLM 的内部人格判断机制。旧生成式 Big5 acceptance 的记录存在滞后，不能把训练完成等同验收完成。 | P01 |
| Representation-first 与 H2 | BGE/E5、TF-IDF、Ridge、PCA、混合和桥接建立了可复用基线。corrected H2 把 Big5、MBTI 训练与 bridge 评估分开。 | 人群关系矩阵的相似，不能转成个体预测或同人一致性。 | P02–P05 |
| 合成语言机制 v1–v4 | 给定人为语言规律，可以恢复人为分数；改变标签噪声可得到恢复曲线。 | 合成成功不能证明相同规律存在于真人文本。早期模板、topic 与强度也可能共同编码标签。 | P06 |
| 去噪/信号提取搜索 | 已试过稳定性选择、残差化、表征空间、主题/情境/情绪、跨文本迁移等。 | 数学操作有效与“删掉了噪声”不同；无收益不能普遍证明某类机制不存在。 | P07 |
| 成果1：flat MCD | 按训练群体学习情境区域，保留作者对情境分布的选择偏离；官方 package 集成记录了明确增益。 | 有用信号主要来自 choice；未证明所有偏离是人格，也未解释 LLM 如何将行为映射到构念。 | P08–P10 |
| H-MCD / V-HMCD / token-slice | 发展递归局部结构、分层表征、跨片段聚合、命名与可移植计分。 | 递归更深不自动带来更强预测；因子轴的可重复性、命名与心理意义是不同问题。 | P11、G01 |
| 旧 V3/V4 条件理论 | choice/base/react 分工，固定与自由条件的两相设计，条件中心化的失败经验。 | “共通背景”“个人选择”“条件下反应”尚不能由一次统计扣除完整分开。 | G01–G03 |
| V6 动态与残差 | 均值之外的路径信息、静态/动态/共享/残差结构得到形式化和数值测试。 | 某些残差有作者关系结构，但没有稳定、可命名的全局动态轴；作者结构不等于心理构念。 | G04 |
| V7 多视图几何 | slice、整文、不同窗口作为平行观察算子；参照群体、共享/私有空间和计分域更加明确。 | 这些是观察者可重复性的基础，不是 LLM reasoning 已被打开的证据。 | G05 |
| V8 / M3 / M4 | 微观事件到条件响应、群间/个体内关系、选择转移、误差分辨率、机制组合和 4W 建立了较完整的研究体系。 | 多数“微观机制”是我们定义的生成模型；它们尚未与真实 Transformer 的具体层和计算通路逐项对应。 | G06–G09 |
| LLM 观察通道 | 可定位事件、受约束编码、重复与跨模型比较、确定性聚合和解释证据已有实现。 | 尚未证明观察事件如何由内部表示产生，或这些表示如何进入人格判断。 | G10 |
| 真人共同条件材料 | MEPS 46 人的任务回答、AI 使用与自由相谈完成目录和向量可行性审计。 | 现有任务不等价，AI 使用不完整且非随机；相邻量表并非统一 Big5 真值。 | G11–G13 |
| 本次新阶段 | 将 LLM 的任务条件、上下文表示更新和内部信息使用接入 GRAIN。 | 隐藏状态仪器、受控干预与外部连接尚待执行。 | 配套计划 |

## 3. 必须一起保存的数值与边界

### 3.1 早期 representation-first/H2

本会话曾报告 PANDORA TF-IDF .184、BGE .148、generative LoRA .100；Essays 内部 ensemble .225、PANDORA→Essays BGE .153。这些用于恢复路线选择，尚未在本次逐个重算原始预测，标记为 `SESSION_REPORTED`，不得替代原协议报告。

本次重读的 corrected H2 报告为：Big5 非 bridge 训练 n=1,037，MBTI 非 bridge 训练 n=8,678，bridge 评估 n=364。直接头 mean balanced accuracy=.5015、AUC=.6087；DDF 路径 .4962/.4966。结构一致性 .5621。原报告“YES beats DDF”只表示点估计略高，没有单独建立优势显著性。[P02]

同人跨量表一致性 AUC=.5263，95% CI [.4986,.5550]，p=.1022；未检出该操作定义下的个体一致性。[P03]

Hybrid 的 AUC=.6156、element_r=.7757，对应 **E5 Big5 头 + BGE/E5 MBTI blend**。保持同一 MBTI blend 而使用 BGE Big5 头时 element_r=.6337。因此 .7757 不能全部归因于 MBTI blend，必须保存两端模型身份。[P04]

后续 H2 总结报告另有 direct BGE AUC=.6340，TF=.723；同文把它称为“linear ceiling”。有限候选比较只能表述为“已测试设置中的最好值”，不能证明线性方法的全局上限。该文对 Big5↔MBTI 链接的若干结论来自预测坐标，不能替代真实双标签结构。[P05]

### 3.2 合成恢复与真人迁移

v1 main r=.9955，但 topic_confound=.9934、feature_removed=.9338，不能解释为已唯一定位所植入的心理机制。v2 的标签噪声曲线从 .9777 降至 .4495，负控制接近零；v3 synthetic→PANDORA=.0263、synthetic→Essays=.0410。[P06]

新阶段应复用“可控世界”的思想，使用严格配对的主体、否定、情境和证据组合，而不再用模板可学性冒充真人机制。

### 3.3 成果1并未被后续测量研究否定

| 历史官方 package 协议 | Mean Big5 Pearson r |
|---|---:|
| 官方预计算 LR-N 参照 | .245975 |
| 官方预计算 LR-NP 参照 | .292628 |
| MCD-only | .231721 |
| N + MCD | .298039 |
| NP + MCD | .318090 |
| NP + MCD + H-MCD | .312492 |

这些是相应条件下的 fold/trait 汇总；N 系列报告 n=1,402，NP 系列 n=1,386，区别于另一准备管线的 n=1,401。原 package 在该报告中记为预计算 unigram N，不应把它改写为本次从原始文本重新复现了完整 word/char 设计。[P08、P09、P11]

NP+MCD 的汇总增益为 +.025461；同用户 pooled OOF bootstrap 的平均 Δ 为 +.023608，95% CI [.013005,.034031]。两个估计量不同。报告的 2,000 次 bootstrap `p=0` 表示有限重采样中未观察到相应尾部，不能写成数学概率零。[P09]

另一个 clean_no_report 协议报告 local baseline=.194258、默认 choice 增强=.248350、validation-selected=.258651。它支持该清洗协议内的增益，不能把 .318090 移植成 clean 协议结果。[P10]

flat MCD 的核心为：

\[
q_{ujk}\propto\exp(-\|c_{uj}-m_k\|^2/\tau),\quad
p_{uk}=\operatorname{mean}_j q_{ujk},\quad
d_{uk}=\log\frac{p_{uk}+\epsilon}{p_{\mathrm{train},k}+\epsilon}.
\]

当前成果1说明“作者相对群体如何选择情境”值得保留；没有推出“所有情境差异都是人格”。

### 3.4 从可重复结构到心理解释，仍有距离

V6 数值几何报告：静态残差 n=1,640，pairwise-distance Spearman=.2597，CKA=.0396；动态残差 n=635，仅宽尺度距离排序 Spearman=.0985 等弱迹象保留。它们来自既定 TF-IDF/SVD 与条件处理，证明的是表征条件下的关系结构。[G04]

旧首次 lockbox 2/7 方向假设通过，整体规则未通过；这保留两个狭窄关联，不能抹掉，也不能升级为整体心理效度。[G02]

后期 What×When 研究主要使用 **expression volume 元数据**，并未读取内容来追踪 Transformer 语义表征。X3 对注册多变量 Big5 几何未检出关系；单维探索候选最高 r=.1413。选择距离的 Mantel r=.04899 是关系矩阵统计，不能与个体 Pearson r 或分类 AUC 相比较。[G02、G07]

V8 语义观察在 24 作者 panel 上得到 coarse-family 作者匹配 AUC=.699 [.554,.798]；跨模型 coarse macro-F1=.771。条件/长度本身强，条件反应仍未识别；这不是人格预测准确率。[G10]

MEPS 三个任务间 raw same-person AUC=.5460/.5651/.6033；长度匹配后 .5166/.5316/.5559。历史规则未晋级。弱匹配与短文本并不妨碍设计新的配对模型实验，但不得写成已确认的共同条件人格稳定性。[G13]

## 4. 过去真正漏掉的机制

| 未完成机制 | 以前覆盖了哪里 | 新阶段需要增加什么 |
|---|---|---|
| 任务条件下的信息读取 | 一般 embedding 和人工定义行为事件 | 同一内容在概括、行为观察、特质判断任务中的逐层表示 |
| 个人历史如何修改当前表示 | 作者聚合、同人 split-half、路径统计 | 固定当前片段，交换本人/匹配他人的历史，测模型内部差异 |
| 原始历史到记忆的压缩与选择 | 有历史文本与若干摘要用途 | 对比随机片段、固定检索、事实摘要；分开信息量、选择和重述效应 |
| 主体、否定、愿望、假设、引述归属 | 语义观察 schema 部分涉及 | 配对事实和内部激活测试：同词汇不同归属如何被区分 |
| 多条证据的相互解释 | M3/M4 合成组合机制 | 真实模型中 A、B、A+B 的非加性交互及其位置 |
| 信息“可读”与“被用” | 线性预测、相关、几何比较 | 候选子空间干预、激活替换、恢复和随机方向对照 |
| 否定和反证如何更新判断 | 人类文本的时间统计 | 历史顺序、相同事实的矛盾/修正与模型更新曲线 |
| 构念先验与个人证据的比例 | 生成式打分和 prompt 修复 | 无个人证据、换人、反转量表方向、任务措辞与人口线索对照 |
| 输出标尺与内部表征的区别 | Parser、schema、阈值、calibration | 判断 log-odds、冻结 probe、自由生成分数各自的响应 |
| 作者过程与观察模型过程分离 | 4W 定义了作者事件和观察算子 | 不把模型层深、回答生成步、KV cache 当成人的心理时间或长期记忆 |
| 跨模型数学可比性 | BGE/E5、不同 slice 的比较 | 同模型内部精确对齐；跨模型使用匹配刺激和几何比较，不能直接减不同坐标 |
| 个体共性与独特性的功能差异 | selection 与 identity 已有证据 | 找到能跨新作者读出的证据子空间，并与身份识别性能分开 |

结论是“内部机制仪器尚未建立”，而不是“以前没有做过向量数学”。现有研究已经包括 PCA/SVD、CKA、距离几何、路径算子和生成模型；本次增加的是可以接触实际 LLM 计算过程的观察与干预。

## 5. 旧完成声明怎样与新研究衔接

2026-08-31 的完成声明在其登记范围内关闭了旧循环，保留了外部连接、W3、R1 等待办。其“没有其他未解决事项”不能被理解为人格推断、模型内部机制和所有未来问题都已解决。[G08]

本次是新研究问题、新计划和新结果目录，不重写旧 lockbox，不把过去检查过的同一 PANDORA 标签重新称为全新盲测。已接触的数据用于开发和分析性复现；之后可对从未用于本次选方向的作者进行留出检验，但仍需注明同语料反复研究历史。

这份回顾没有把“identity 在独特、personality 在共通”当成普遍定理。它是当前选择表示和特定外部几何下的经验模式，也是新内部向量研究应主动比较的候选解释。

## 6. 资料和代码的整理方式

旧 `project persona` 仓库保留原始数据路由、LoRA/embedding/MCD 历史；独立 GRAIN 仓库保存当前理论、可复用计分器、实验报告及本计划。先通过来源索引连接，不搬动受限制数据，不把 private 仓库既有未推送提交混入本次文档发布。

当前可复用接口：

- `suica_core/meps_layout.py`、`suica_core/meps_ai_conv.py`：MEPS 目录、作者/AI 角色与文本 view。
- `suica_core/meps_fixed_profile.py`、`configs/meps_fixed_condition_profile_v1.json`：冻结的 multilingual-e5 对照。
- `scripts/run_suica_op9_embedding_baseline_v3.py`：slice embedding 与作者汇总旧基线。
- `scripts/run_suica_e9_embedding_interpretive_channel_v3.py`：anchor 方向和邻居解释旧对照。
- `docs/V8_LLM_NATIVE_ASSESSMENT_THEORY.md`：语义观察、确定性计分与证据解释接口。
- `grain_core`、`grain_sim`：新名称入口，旧实现命名用于兼容。

在 `grain_core/grain_sim/suica_core/suica_sim/scripts/tests` 中检索 `output_hidden_states`、`register_forward_hook`、`activation.patch`、`causal.trac`、`residual.stream` 未命中。这是本次指定目录的检索结论，不是对所有外部项目的断言。

本机 Hugging Face 缓存存在 BGE、E5、multilingual-E5 和 Mistral-7B-Instruct-v0.3 目录；没有在本次加载模型，目录存在不代表权重完整、显存可用或 hooks 可运行。

## 7. 证据来源表

`P` 来源位于 private `project persona` 仓库，表中给出仓库相对定位符；`G` 来源位于当前 GRAIN 仓库。内容和作者数据不复制进公共文档。

| ID | 来源 | 本次用途 |
|---|---|---|
| P01 | `PROJECT_STATUS.md` 与本会话官方训练回顾 | 识别旧状态段落的日期、训练/验收区分 |
| P02 | `analysis/output/h2_corrected/shared_rep/H2_CORRECTED_REPORT.md` | corrected H2 样本与指标 |
| P03 | `analysis/output/h2_corrected/same_user_consistency/SAME_USER_CONSISTENCY.md` | 个体跨量表一致性 |
| P04 | `analysis/output/h2_corrected/hybrid/HYBRID_REPORT.md` | 两端 head 和 blend 的区别 |
| P05 | `docs/H2_FINAL_SYNTHESIS.md` | 后续已测方法与过强“上限”结论 |
| P06 | `docs/SYNTHETIC_PERSONALITY_LANGUAGE_FINDINGS.md` | v1–v3 合成与迁移 |
| P07 | `reports/representation_signal_observability_report.md` | 旧向量表征/子空间搜索 |
| P08 | `docs/PROJECT_PERSONA_RESULT_1_MCD.md` | 成果1定义 |
| P09 | `reports/pandora_baseline_mcd_integration_report.md` | 官方 package、样本量、OOF/fold 汇总 |
| P10 | `reports/mcd_paper_grade_evidence_report.md` | clean 与 paper-visible 协议区别 |
| P11 | `reports/pandora_baseline_mcd_hmcd_integration_report.md` | flat 与 hierarchical 比较 |
| G01 | [既有路线表](SUICA_THEORY_ROUTE_INDEX.md) | MCD 到 M4 的对象演进 |
| G02 | [论文证据表](paper/GRAIN_THEORY_EVIDENCE_PROVENANCE.md) | 注册外部连接与计量边界 |
| G03 | [条件理论](THEORY.md) | choice/base/react |
| G04 | [V6 数值闭环](../reports/V6_NUMERIC_GEOMETRY_CLOSURE.md) | 静态/动态残差 |
| G05 | [V7 多视图](V7_OPERATOR_INDEXED_MULTIVIEW_V71.md) | 观察算子分工 |
| G06 | [V8 数学路线](V8_MATHEMATICAL_RESEARCH_ROUTE.md)、[M3 基础](SUICA_M3_MICRO_MESO_MACRO_FOUNDATION.md) | 数学对象与实际网络机制的差距 |
| G07 | [4W 实证状态](SUICA_4W_EMPIRICAL_STATE.md) | 内容未读的元数据路线、已更正的跨层结论 |
| G08 | [历史完成声明](SUICA_COMPLETION_DECLARATION.md) | 封闭范围及剩余事项 |
| G09 | [跨层推论契约](SUICA_CROSS_LEVEL_INFERENCE_CONTRACT.md) | 作者、关系、群体统计分开 |
| G10 | [V8 观察试验](../reports/V8_BEHAVIOR_V21_CANDIDATE_REPORT.md) | LLM 编码已有能力和缺口 |
| G11 | [MEPS 目录](../reports/MEPS_AI_CONV_DATA_LAYOUT.md) | 46 人、角色和 withheld 量表路由 |
| G12 | [MEPS 可行性](../reports/MEPS_AI_CONV_FEASIBILITY_AUDIT.md) | 短文本、AI 使用机会、任务区分 |
| G13 | [MEPS 向量对照](../reports/MEPS_V6_FIXED_CONDITION_PROFILE.md) | 已做同人跨任务结果 |

## 8. 下一阶段的中心问题

把原有链条

`人及其条件 -> 自然语言 -> GRAIN 观察/聚合 -> 参照群体中的个体结构`

扩展为可以并行研究的模型链条：

`同一材料 + 任务 + 可用历史 -> LLM 内部表示 -> 信息读取/组合 -> 输出判断`。

两条链通过同作者、同文本、共同参照和外部指标连接。最终目标是找到既参与模型判断、又能跨作者形成可比较测量的机制，而不把模型的内部活动直接命名为人的人格。
