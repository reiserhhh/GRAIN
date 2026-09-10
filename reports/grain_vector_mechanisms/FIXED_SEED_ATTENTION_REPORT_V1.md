# NEXT39：上下文条件作用存在，真实注意力已解析接通

2026-09-10，第28轮。**固定相同输出方向后，查询之间的正文作用差异仍然
保留；实际模型中的“值传递+选择权重变化”公式在24/24单元验算通过。**
这是由黑箱局部梯度走向具体可计算环节的一步，不是新增人格预测准确率。

## 实验与主要结果

旧4名来源作者与新8名来源外作者，沿用已有完整正文、GRAIN切片和块16
状态。N/确认两查询统一为固定数字logit线性目标L=c^Tz；最终隐藏方向
W_A^Tc完全相同。新作者只相对这4个源作者独立，材料此前已用于其他研究。

| 查询间作用余弦 | 旧4人 | 新8人 |
|---|---:|---:|
| 原概率目标、块16正文梯度 | .704 | .649 |
| 相同输出方向、块16正文梯度 | .728 | .657 |
| 相同输出方向、最后层value项 | .957 | .930 |
| 相同输出方向、最后层key项 | .707 | .882 |
| 相同输出方向、最后层两项之和 | .847 | .897 |

新8人固定方向的块16余弦范围.519–.761。输出编码统一后差异没有消失，
因而存在查询条件下的不同传输；其中仍包含query状态与布局。不同层不能
通过余弦相减计量心理信息损失，也不把L当成原概率F。

## 哪个计算环节值得继续发展

标准attention反传在真实Qwen2的12头、2KV组和RoPE下精确接合，非最终
query的正文归一化输入作用分为：

    attention_weight * value_direction
    + attention_weight * relative_context_value * key_direction.

| 最后层正文中的量 | 新8人N | 新8人确认 |
|---|---:|---:|
| 平均逐作者key/value场范数比 | 1.826 | 1.887 |
| 丢弃key项的相对平方误差均值 | .731 | .743 |
| value/key场余弦均值 | .003 | .006 |
| head平均选中正文attention质量 | .131 | .140 |

选择权重变化不是小修正。对这层GRAIN方向，新作者64个作用中6个的value/
key符号相反，旧4人为4/32；其余多数同号。两向量场近正交仍可在指定方向
相加或相抵，所以不能只保留正权，或者只看总attention质量就解释人格。
全文上下文汇总o参与相对值(v_j-o)，不能将正文之外的token直接删去。

## 验算与资源

- 最大解析/native正文梯度相对L2误差2.028e-6，绝对误差6.603e-9；
  固定的relative<=1e-3及absolute<=1e-5门槛24/24通过，没有调宽。
- 最后层GRAIN作用加和最大差3.348e-7、相对L2差8.841e-7。
- 原概率评分与前轮canonical缓存重放差为0。
- 预定新作者0的whole:F01块16有限差分：N解析.582633/差分.587611，
  确认.139078/.132610，同号；epsilon=.001对应增量误差4.98e-6/6.47e-6。
  不把这项有限步长结果也写成最后层梯度的1e-9精度；无epsilon重选。
- 28后缀前向、24反向，44.410秒，模型加载3.302秒；无前缀重跑、拟合、
  人类标签读取或API。24个明确packet ID齐全，无后台长队列。
- 构造和主分析代码在`24162d2`冻结后采集；6个数值CSV逐字节重放。
  51项聚焦测试通过，含旋转转置、GQA汇总和q/k/v有限差分。不是全库测试。

后续仅缓存内的audit增补了范数比和丢弃key的误差统计，不选择模型或head。
公共[四张聚合表](fixed_seed_attention_tables_v1/)不含文本、作者或标签。

## 文件与下一步

- 公式：[条件attention理论](../../docs/GRAIN_FIXED_SEED_ATTENTION_THEORY_V1.md)。
- 实现：`grain_research/vector_mechanisms/fixed_seed_attention.py`。
- 执行：`scripts/run_grain_fixed_seed_attention.py prepare|collect|analyze`。
- 复核：`scripts/audit_grain_fixed_seed_attention.py`。
- 私有缓存：`results/grain_vector_mechanisms/logic_fixed_seed_attention_v1/`。
- 下一步：[NEXT40源固定下游方向](../../docs/GRAIN_SOURCE_COTANGENT_READER_PLAN_V1.md)。

公式来源是标准[attention](https://arxiv.org/html/1706.03762v7)、
[GQA](https://arxiv.org/html/2305.13245v3)、[RoPE](https://arxiv.org/html/2104.09864v5)，
不是新发明的神经网络定理。新贡献是把该分解接到同作者/同材料/GRAIN方向
并发现可用作用，随后尝试编译成不依赖当前反向目标的数值程序。Goal ACTIVE。
