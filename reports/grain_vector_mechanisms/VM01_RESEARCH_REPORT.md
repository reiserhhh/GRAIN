# VM01：GRAIN 向量机制仪器与第一次功能定位

2026-09-05。范围：**合成语义事实上的模型机制**，不是人类人格预测、心理效度或 GRAIN 新量表。

2026-09-06 实现复核：完整替换现已由 BF16 `x+(y-x)` 改为精确赋值。用原材料/方向重跑 `vm01_patch_exactcopy_r4`，16 目标、3,264 forward、167.89 秒，自身/恢复误差均为 0。block16 decision、rank2、α=1 的 candidate-minus-random：ownership 从 .640625 变为 .593750 [.468750,.708333]；voluntary 从 1.567708 变为 1.531250 [1.364583,1.697917]。因此原有限功能作用结论保留，绝对数值按实现版本区分。下面原表保留 r3 历史，不混写两版本。新关系绑定/分布式干预见[后续报告](VM03_RELATIONAL_BINDING_REPORT.md)。

## 结论

可以在固定 LLM 中定位可读的事实信息，并通过低秩激活干预改变回答。但“方向可读”“方向影响答案”“方向语义专属”“方向具有人格效度”是四个不同命题，本轮只支持前两个的有限实例。

最重要的方法发现是：计算精度和总序列执行形状本身会改变观察到的向量。若不控制，容易把计算漂移误认为上下文推理。固定 masked padding 修复了本次已测试的前缀不变性问题，未宣称消除所有硬件数值误差。

## 设计

固定 `Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，Windows RTX 4070 Super，torch 2.10.0+cu130，transformers 5.1.0，BF16、eager attention、eval mode、无 KV cache、无权重训练。它是可运行的白盒观察器，不代表 ChatGPT 记忆研究使用的模型。

192 prompts = 12 场景 × 4 事实任务 × 4 变体。8 场景拟合 probe/子空间，4 场景留出；相同语言模板会跨场景重复，因此不是 held-out-template 泛化。组合任务阳性率 .25，其他任务 .5。

采集共同证据结束 marker 与回答前 suffix 两个位置：初始 embedding、block 00/05/10/16/21/27、final normalization。不采已经生成答案的 token 作为输入特征。所有数值均来自 canonical `vm01_static256_r3` 与 `vm01_patch_static256_r3`，旧 variable-length 结果降为修正记录。

## 事实排序与读出

| 任务 | 留出 prompt N | AUC | 默认 Yes/No BA | discovery 阈值后的 BA | unigram AUC | word 1–2gram AUC |
|---|---:|---:|---:|---:|---:|---:|
| 实际发生 vs 否认/愿望 | 16 | 1.000 | .9375 | 1.000 | 1.000 | 1.000 |
| 两条事实的组合 | 16 | 1.000 | .7917 | 1.000 | 1.000 | 1.000 |
| 叙述者 vs 他人实施行为 | 16 | 1.000 | .9375 | 1.000 | .500 | 1.000 |
| 自愿选择 vs 被迫执行 | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

独立留出单位只有 **4 场景**，不是 64 个独立观察。阈值来自 discovery，是读出诊断而不是额外语义学习。固定 marker/suffix 的初始 embedding AUC 都为 .5；最早观察 block 的证据读点已能解码模板事实。这不足以证明信息只能在深层产生。

一个直接解释是：这组事实已可排序，但默认输出门槛仍有偏置。另一个必要限制是词 bigram 同样满分，不能把这些材料称为只有大模型才能完成的推理。后续必须增加语义等价改写、词汇平衡与组合外推。

## 干预结果

两类任务、两个 decoder block、两个读点、rank 2/4、alpha .5/1，16 个留出配对目标，3,264 次前向。每个候选配 3 个请求位移范数匹配的随机方向，另外有自身替换、全 donor、恢复和无关场景问答。

下表是 **rank=2、alpha=1 的预定切片**；完整网格仍保存在 CSV，不表示只测了最强格。差值以 donor 的原模型输出方向定向，单位为 Yes/No log-likelihood margin；不是准确率或人格相关。区间按场景聚类 bootstrap，仅 4 cluster，探索性且未作多重比较校正。

| 位置 | 任务 | 候选平均位移 | 随机平均位移 | 候选减随机 [95% CI] |
|---|---|---:|---:|---:|
| block16 decision | 主体归属 | .6563 | .0156 | .6406 [.4896, .8021] |
| block16 decision | 自愿选择 | 1.7656 | .1979 | 1.5677 [1.4896, 1.7031] |
| block16 content | 主体归属 | 约 0 | -.0313 | .0313 [-.0208, .0833] |
| block16 content | 自愿选择 | -.0156 | -.0729 | .0573 [-.0417, .1563] |

自身替换与恢复最大偏差 0。无关问题后的 pre-question 激活偏差 0。内容站点的弱作用不能被解释为已找到语义机制。

**专属性未通过充分证明。** block16 decision 干预在无关 setting 问题上也产生平均绝对 margin 位移：主体 .5156（随机 .1797）、自愿 1.1953（随机 .4453）。没有分类翻转不等于没有损伤；原来的大 margin 可以掩盖读出偏移。随机范数是注入前 FP32 请求范数，尚未直接匹配 BF16 舍入后的实际位移，这是后续精度敏感性项。

## 计算层必须进入公式

理想因果 decoder 的前缀状态为：

\[
h_{l,p}=F_{\theta,l}(x_{\le p}).
\]

实际观察量应写成：

\[
\widehat h_{l,p}=F_{\theta,l}(x_{\le p})+
\epsilon_{l,p}(\text{dtype, execution shape, backend}).
\]

因此上下文差分包含真正的模型效应与计算误差差分。以下是 4 个固定前缀 cases 的数值检查，不是 population estimate：

| 执行条件 | 相同长度改变未来 token | 改变总执行长度的最大前缀 relative L2 |
|---|---:|---:|
| CPU FP32 | 0 | 0 |
| GPU BF16，变长 | 0 | 约 .0682 |
| GPU BF16，关闭 reduced-precision reduction | 0 | 约 .0687 |
| GPU BF16，固定 masked 256 | 0 | 0 |

这排除了本例“未来语义反向改变过去状态”的解释，支持 shape-dependent numerical effect；未定位到某一底层 GPU kernel 的唯一根因。不同运算安排的浮点结果可能不同，本来就是 [PyTorch 数值文档](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)指出的限制；这里的幅度与修复是本项目实测。

## 读出方向与证据方向不能混为一谈

对于本模型的一 token Yes/No 候选，在 final normalization 后有精确关系：

\[
s=\log p(Y)-\log p(N)
=(W_Y-W_N)^\top h_{\mathrm{final}}+(b_Y-b_N).
\]

所以晚层可以存在普适答案方向；能沿它移动答案，不自动等价于改变了作者表征。多 token 答案不能直接套用这个单一投影式。观察器另实现完整序列评分，记录实际 candidate IDs。模型结构和层输出语义分别按 [Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)与 [Transformers 输出定义](https://huggingface.co/docs/transformers/main_classes/output)核对。

本轮结果提示将后续链条拆成：**证据编码 → 上下文归属/组合 → 任务条件下的证据使用 → 输出标尺**。候选“人格方向”至少需要在改答案编码、改问题极性、保留事实语义时仍然成立，且有独立外部连接；不能靠一个漂亮的 probe AUC 命名。

## 下一轮

已从 raw PANDORA 准备 8 技术 + 24 开发作者，每人 3 历史评论 + 2 当前评论，跨线程且历史在前。下一轮运行同当前文本的本人/匹配他人/无历史对照，同时比较证据 marker、内容均值与判断 suffix。标签仍不参与选层。

必须处理：共享历史造成的机械相似、不同当前语义造成的反应差异、历史时间跨度很大、长度/位置不完全匹配、1 条类型词线索、样本仅 24 作者。按这些限制设计对照，而不是因而停止机制探索。

代码、数据与运行定位见[研究日志](../../docs/GRAIN_VECTOR_RESEARCH_LOG.md)。本轮未读取 MEPS 问卷、未打开新的外部 lockbox、未修改 GRAIN 正式计分通道。
