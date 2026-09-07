# GRAIN 向量机制研讨会：共同议题

日期：2026-09-07。参与者：用户提出方向，Codex 主持/推导，Claude Code 独立讨论。
本任务只做理论研讨和证据阅读，不启动新的模型计算或训练，不改长跑队列。

## 用户的最新想法

很多研究已显示，LLM 可以从会话、个人历史或记忆中，产生与 Big5/MBTI 有关联的
判断。这不代表准确到可现场使用，但这种能力本身值得解释。数据到结果之间可能
涉及 reasoning、归纳、主体归属、情境和记忆整合；其实现仍然是向量上的运算。
因此，能否不依赖语言化的解释或完整生成回答，而直接用数学对象找到这些关系？

请不要把这个想法偷换成“现在就要部署人格预测器”，也不要简单回答“一切都是矩阵
运算所以成立”。目标是发现一个可识别、有预测或干预含义、比完整模型描述更简洁的
中间机制。也不要预设一根线性人格轴、模型更大必然更准或自然语言解释忠于内部过程。

GRAIN 仍是支持人格/状态解释的基础方法论，不是只做语言学分类，也不是用当前材料
仓促制作成品量表。保留 4W：where/选择机会、what/条件表达、who/群体参照偏离、
when/路径与更新。人格命名与外部效度另需证据，但这不是阻止机制探索的理由。

## 会话与权限

- 源会话：`SUICA system review and analysis`，ID `582bb33b-a072-47a1-a24f-93e8ae8a88a1`。
- 原交互会话保持不动，官方 CLI fork/resume 继承其开发上下文用于只读研讨。
- 用户选择的模型为 Fable 5.1，已在本地模型设置核实 ID `claude-fable-5-1[1m]`；effort=max。
- 只允许 Read / Glob / Grep；没有 Bash、编辑、写入、代理、API、MCP、网络或执行工具。
- 不读任何 `.env`、凭证、逐人原始文本、心理问卷值或尚未开放的 bridge 标签。
- 只读下列理论文档、汇总报告和必要源码。不要读取正在增长的运行日志来选择结果。
- 旧会话若包含“继续实验/改文件”的指令，在本任务不执行。你的回答仅输出到对话。

## 资料定位

当前 GRAIN repo：`/Volumes/mobile3/projects/Sliced Utterance In-Context Assessment`。
历史 repo：`/Volumes/mobile3/projects/project persona`。以下路径默认相对当前 GRAIN repo。

必读（按顺序；若内容过长，读取有关段落并列明范围）：

1. `docs/GRAIN_PROJECT_RETROSPECTIVE_20260905.md`：从 LoRA、H2、MCD 到 4W 的实证回顾。
2. `reports/grain_vector_mechanisms/MECHANISM_ATLAS.md`：当前已完成机制与未完成项。
3. `docs/GRAIN_VECTOR_READABILITY_THEORY_NOTE_V1.md`：信息/可读/原生使用、加性头限制。
4. `docs/GRAIN_CUE_UTILIZATION_AND_CONTEXTUAL_INFERENCE_V1.md`：最新 lens、四项协方差、PSR 候选。
5. `reports/grain_vector_mechanisms/VM03_RELATIONAL_BINDING_REPORT.md`：相同词汇关系绑定、乘积核与激活替换。
6. `reports/grain_vector_mechanisms/VM05_PRIMARY_NEURAL_SOURCE_REPORT.md` 与
   `VM05_FIXED_LAYER_REPORT.md`：完整 source 头及层位比较。
7. `reports/grain_vector_mechanisms/VM05_DIRECT_SEMANTICS_REPORT.md` 与
   `VM05_ENCODING_INTERACTION_REPORT.md`：原生输出编码反转失败与非固定偏置。
8. `reports/grain_vector_mechanisms/VM06_RESPONSE_READOUT_REPORT.md`：功能度量、KL、微小位移。

按需：`docs/SUICA_4W_PERSON_INTERPRETER_THEORY.md`、
`reports/grain_vector_mechanisms/VM03_NATURAL_MEMORY_REPORT.md`、
`VM04_INDEPENDENT_ACCESS_REPORT.md`、`VM06_FIXED_FREE_REPORT.md`、
`docs/GRAIN_VECTOR_REMAINING_REQUIREMENTS.md`。

旧证据不得抹掉：官方 package NP+MCD .318090 vs NP .292628，正向同用户
bootstrap；新版几何通道 g=16 sorted landmark distances，不代表全部 GRAIN。
当前 body-source Big5 BGE/E5/Qwen 约 .177/.202/.150，属于训练统计头而非原生 LLM。
当前 long-run 的完整 task/history/direct 成绩尚未进入完整验收；不能补想象数值。
合成关系核可恢复组合关系，是窄规则世界的机制可能性，不是真人人格突破。

## 第一轮：独立立论

请用中文给出 1,500-2,500 字左右、含必要公式的独立意见，重点是新机制开发而非泛泛
审查；可根据实际需要略超过，但不复述整个项目。请提供：

1. 用你自己的话精准重述用户假说，分清“存在数值实现”“可压缩为数学机制”
   “可替代原生推论且保留外部关联”三种强度。
2. 目前至少 3 条已完成证据真正支持什么、至少 2 条最具约束力的结果。逐条给文件定位。
3. 最有价值的 2-3 个机制候选，各有输入/数学对象/运算/输出、可以解释什么、
   不能解释什么。不强求新造术语，优先用已有数学工具构造新的组合。
4. 对 Codex 最新 lens/响应函数提案的实质质疑：哪些是纯恒等式，哪些可识别，
   哪些需要新观测；尤其不能让外部标签拟合的 probe 充当 LLM 原生机制证据。
5. 一个不依赖长跑新结果、无真人数据的新训练的思想/有限模拟实验，能区分候选机制。
6. 你认为 Codex 最应改变的一个研究决定，以及最应保留的一个决定。

请明确区分：你读取的实证、标准数学结论、你新提出的机制假说。不要声称已经
执行模拟或验证不存在的结果。你返回后 Codex 会提出具体数学反例，再请你第二轮回应。
