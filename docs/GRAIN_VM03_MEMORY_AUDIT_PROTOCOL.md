# GRAIN VM03：记忆选择、摘要保真与更新

状态：开发侧 bounded mechanism experiment，不是人格预测 benchmark。
日期：2026-09-06。主计划：[VM01–07](GRAIN_VECTOR_INFERENCE_MECHANISM_PLAN_V1.md)。

## 对象与顺序

- 固定新多评论 cohort 的 comparison 作者，按新的 ID hash 取 6 人，
  不按人格标签、过去效果或文本主题挑人。每人第一时间视图 12 评论作历史；
  第三视图前 2 评论作当前文本。12 个案例不等于 12 个独立被试。
- 完整原文保留；角色、时间、讨论去重和同人/他人匹配规则继承 VM02。
  他人从 96 comparison 中匹配，必须在当前文本之前且不共享讨论/重复正文。
  长度以 whitespace words 匹配，不能称 token 精确等长。
- E5-large-v2 固定 revision 的 query embedding 检索本人历史 top3，
  同数 random3 为对照；片段原文不重写，返回时保持源时间序。
- Qwen2.5-1.5B-Instruct 固定 revision 只生成 6 次事实摘要，每份至多
  6 条，包含 source index、原文连续引文、事实句。生成不作数值激活仪器。
- 先审核摘要，再采集下游 logits。严格格式失败、单结尾括号修复、
  超条数、引文不匹配、语义不保真分开登记。不改写生成的事实来提高分数。
  审核仅一位 Codex analyst，不冒称独立专家共识或真人心理真值。

## 条件与输出

自然文本：none、raw_full、raw_reversed、两种 other、retrieval3、random3、
交错 half_a/half_b、summary_unfiltered、summary_cited、summary_quotes、
summary_reviewed。两个固定任务：具体主动社交行为证据、一般主动社交倾向证据。
固定 13 arms ×12 currents ×2 tasks=312 prompts。

有标签部分是另立的人工阅读控制：在自然背景中加入明确标注为实验虚构的
Record Z7，改变主体、假设、否定、引述以及两种方向的更正；8 变体 ×
12 背景 ×2 位置 ×2 问句极性=384 prompts。其正确率只针对该虚构事件，
不是自然作者的人格准确率。仅一个事件模板，不等于 384 独立语义故事。

采集固定 Qwen block10/16 的当前文本池化、证据末端和判断末端；其余层
仅保留可复核 cache，不重新按结果选层。Yes/No 使用同一 prefill 的 logits 差，
同时报告两候选的实际总概率质量；不把条件归一化 Yes 值当作校准人格概率。

## 仪器边界

3072 固定计算长度、BF16 eager、无 KV cache 采集、无静默截断。
正式运行前必须确认每个当前原文在所有条件中的池化 token ID 序列完全一致。
使用统一 CURRENT_END wrapper，不改原文。原因：原 r1 的末尾表情符号
在单/双换行后被切为不同 token；同 token 数仍会产生初始 embedding 差。

回放、相同因果前缀、初始 token pooling 必须逐值误差为 0。
失败的 r1 留档但不可进入结果汇总；修正后从头在新目录 r2 运行。
位置不同不是人格差异；自然 arms 的文量、内容选择和绝对位置仍不相等。

## 估计量

自然材料没有独立行为/量表 truth；报告 margin 差、回答翻转与表征位移，
不报告“自然人格准确率”。按作者 bootstrap，且明确只是 6 人 pilot。
摘要的完整-case 子集只有实际通过 schema 的作者，单列，不用空摘要充当成功。

设 H 为完整历史，Q 为被有效引用的原文 spans，C 为对应的生成事实句，
A 为 C 中语义审核通过的子集。下式只是精确的望远镜求和：

    s(A)-s(H) = [s(Q)-s(H)] + [s(C)-s(Q)] + [s(A)-s(C)]

三项分别记录片段选择、相同来源事实的重述、语义拒绝过滤的模型效应；
每项也可能含文量/位置改变，不能把代数恒等式升级为独立心理机制分解。
引用正确不蕴含命题正确，命题正确不蕴含跨场景人格推论正确。

half_a/half_b 的二阶有限差分在 margin 与条件概率两尺度单列，
只称非加性；不存在等长语义空白，所以不能宣称唯一隔离了证据协同。

## 代码与数据

入口：run_grain_vector_mechanism_memory_prepare.py、
run_grain_vector_mechanism_memory_summaries.py、
run_grain_vector_mechanism_memory_review.py、
run_grain_vector_mechanism_memory.py、
summarize_grain_vector_memory.py，均位于 scripts/。

原文、单作者引文、生成内容、语义审核原记录与全部 hidden states 留在
gitignored results/grain_vector_mechanisms/；公共报告只保存聚合数值、
去识别的错误类型和重播命令。未访问 calibration/holdout 的模型输出或心理标签。

## 先行研究定位

摘要事实保真有既有研究基础，不是本次新发现的普遍现象：
[Maynez et al., 2020](https://aclanthology.org/2020.acl-main.173/) 区分流畅摘要与
忠实于源文的摘要；[FRANK](https://aclanthology.org/2021.naacl-main.383/) 提供多类
事实错误而非单一对错的视角。本实验关注这些错误在作者/人格证据链中的后果。
[Liu et al., 2024](https://aclanthology.org/2024.tacl-1.9/) 提示位置敏感性需要独立测；
本实验短材料的前后对照并不等于复现其长上下文中间位置实验。

## 执行增补：等价陈述诊断

初轮发现极性错误后，追加同一384个已知事件的陈述式验证，不另调参数：
将否定疑问句改为陈述命题并询问是否被最终记录支持；答案仍为Yes/No。
这是post-observation诊断，不称独立确认。不同问法的pre-question表示
还需逐值相等，以排除证据采集差异；最终结果见
[执行报告](../reports/grain_vector_mechanisms/VM03_NATURAL_MEMORY_REPORT.md)。
