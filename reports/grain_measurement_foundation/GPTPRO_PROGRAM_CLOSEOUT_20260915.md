# GPT Pro计划阶段收束

> 2026-09-15全面验收及二次纠正覆盖本页：**NOT_ACCEPTED / NEEDS_RETRY**。GP03的B1本来就是固定来源表，oracle与GP02一致，撤回“个体更新估计器错接”指控；真正缺同reader的B2/M及具体证据卡。GP04未执行角点留出且未验证M4按人检出；GP06把C2模型套到GP04世界并无条件写PASS，冷制作闭环撤回。27测试总数已含PV01的4项；导数失败原因不能直接归精度。局部数学/缓存数值保留，旧r1不改。证据见[Project Persona全面验收](../../../project%20persona/reports/gptpro_seminar_20260915/COMPREHENSIVE_ACCEPTANCE_AUDIT.md)，后续见[全源V2计划](../../../project%20persona/docs/GRAIN_GPTPRO_FULL_COVERAGE_RESEARCH_PLAN_V2.md)。新科研包尚未执行。

本阶段依据四份 GPT Pro报告、汇总稿和三轮 Claude 只读研讨形成。GP00-GP06 已按依赖执行并输出可复核产物；PA01/PV01 也已在主线完成后按有限预算执行，未作为主线的必要前提。

## 交付结果

- GP00：对象、记录、作者级分层和复用资产契约通过。
- GP01：C1 条件反应碰撞、C2 二阶历史碰撞、C3 机会×选择边界及阈值非传递/局部秩反例通过精确证书。
- GP02：在同一有限历史与预算下，B1 标准二阶条件频率是本次 application 最好参照；B2 和 rank-2 压缩没有增益。
- GP03：无状态合成文本经冻结 lexical reader 接回 B1；oracle、text-soft、text-hard 的损失分开保存，文本通道不是 oracle。
- GP04：固定条件竞争族可区分；预先声明的 M4 混合族触发族外诊断。后验最高不被解释成机制真值。
- GP05：PANDORA/Essays 缓存接口和 source/held 对齐通过；自然时间序列、完整机会和随机条件效应因输入缺失而保持 BLOCKED_INPUT。
- GP06：T1 固定条件响应差与 T2 prefix-dependent 下一观察概率完成同一入口的冷构造、坐标运输和回放。
- PA01：在固定竞争族的有限模拟中，ADO 在预算 8/16 的 holdout log loss 略优于固定/随机设计，但预算 24 时随机略优；因此只保留为可选的主动区分参照，不升级为普遍规则。
- PV01：复用冻结 reader，在 source-only 低文本上下文计算 72 个 Jacobian，并以 384 次新前向做 application 干预。action-weighted 对 native donor 的 RMSE 从 0.038105 降至 0.033991（Extraversion）、从 0.008049 降至 0.007107（Neuroticism）；相对完整文本差分仅 Extraversion 改善，Neuroticism 基本持平，且 6 个中心差分检查均为 `PRECISION_LIMITED`。它是有限 reader 机制线索，不是心理效度或新通用算子。

## 研究含义

本轮支持的是一套有限的测量制作纪律：先定义目标和观察权限，再验证信息压缩边界，冻结文本 reader 和 source 参考，最后把支持范围、损失位置和解释层级一并交付。它还没有证明某个匿名分数是人格，也没有证明自然文本完全可以脱离语义模型或替代问卷。

下一阶段只有在出现明确的信息损失反例、文本/reader接口断点或新的可观察条件时继续；不以增加模型、阶数或阈值网格作为默认进展。真人、临床和跨文化效度另需授权的研究设计。

## 运行与修复记录

GP00-GP06、PA01 和 PV01 均已产生独立 `r1` 产物；所有新科学测试共 27 项通过，PV01 另有 4 项合成测试通过。执行中 GP04 首次发现三个 seed 复用了作者/事件 ID，已在 `rival_models.py` 中修正并以独立 ID 重跑 GP04→GP06；最终数字均来自修复后的运行，旧输入未被覆盖。GP06 在收口审计中补齐了公共契约缺失的 `manifest.json`、`metrics.csv` 与 `decision.md`，不改变科学数值。

PV01 的 MPS/FP32 局部梯度有限差分存在精度限制；没有扩大预算或以 application 结果调参。自然材料仍按 GP05 的 `BLOCKED_INPUT` 处理，未启动真人、MEPS 拟合、旧生成队列或付费 API。
