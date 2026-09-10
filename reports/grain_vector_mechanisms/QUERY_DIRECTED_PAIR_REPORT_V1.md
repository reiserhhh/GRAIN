# 完整条件评分已经接通，下一缺口是片段作用的方向

2026-09-10。第二十一轮，NEXT33完成，逻辑goal继续。

**这轮保留了查询定向基线的价值，并把它接成了不借当前模型端点的完整
条件评分。** bag/position的评分相关为.610/.593，对照为-.095/.092。
更重要的新定位是：片段作用幅度有相关，方向读取却尚未跟上。

## 1. 实际执行

同4名旧N参考作者、20名旧评估作者，评论对(0,1)/(0,11)，原两种Fourier输入。
数据无关随机投影精确重现；归一化只由4源x3视图构造。每次构造5个数值头：
1个核基线、2个共享单片段、2个mixed；定向基线直接沿用NEXT32。
不把旧18源参数拼入，不据评估分数挑参数，无问卷/MEPS/锁箱/模型调用。

    zhat_pair = bhat(M,R) + dhat_i + dhat_j + mhat_ij
    scorehat = S_original(zhat_pair)

完整数值推理子进程只能打开模型资产、当前输入和写出结果；禁止读取当前
teacher端点、旧预测和本次预测文件。每次3个禁读探针、score/logit重放误差0。
两次完整数值构造均完成，不是后台神经训练。输入仍是块16编码，目标是
固定P/Q的条件pair，不是完整原文F或人的神经质量表。

## 2. 全部比较，不把oracle当预测器

CI为20作者bootstrap2000次，两个pair始终随作者一起抽取。旧作者用于
机制探索，并非新的独立人群。以下为40对观察的pooled分数：

| 输入 | 组装 | 评分r及95%CI | RMSE |
|---|---|---:|---:|
| bag | 定向基线+预测片段作用 | 0.610 [0.250, 0.842] | 0.014470 |
| bag | 同源核基线+同一片段作用 | -0.095 [-0.465, 0.328] | 0.017535 |
| bag | 只用定向基线（诊断） | 0.602 [0.205, 0.836] | 0.015018 |
| bag | 真实基线+预测片段作用（oracle） | 0.891 [0.761, 0.960] | 0.007076 |
| bag | 定向基线+真实片段作用（oracle） | 0.797 [0.608, 0.915] | 0.011650 |
| position | 定向基线+预测片段作用 | 0.593 [0.267, 0.813] | 0.014322 |
| position | 同源核基线+同一片段作用 | 0.092 [-0.083, 0.272] | 0.017101 |
| position | 只用定向基线（诊断） | 0.602 [0.205, 0.836] | 0.015018 |
| position | 真实基线+预测片段作用（oracle） | 0.881 [0.737, 0.958] | 0.007387 |
| position | 定向基线+真实片段作用（oracle） | 0.797 [0.608, 0.915] | 0.011650 |

相对核，RMSE下降约17.5%/16.2%，但配对MSE区间仍跨0；相对只用基线，
pooled改善区间也跨0。不能把点估计提升宣称为已经确证的普遍改进。
基线-only针对同一个pair目标已有r=.602。原生B与原生pair相关.860/.944，
所以这轮完整评分的主要可用信息仍来自基础读取，而非片段细节全部被读懂。
与上一轮B-target的r=.727不是同一个目标，不能当作简单升降比较。

## 3. 更有用的发现：大小与有符号方向分离

相邻配置作用的范数相关.637/.721，作者中心化向量相关却只有-.009/-.065；
在定向基线上的评分增量相关为-.214/-.321。相邻mixed本身仍有局部信号：
MSE/零=.783/.709、范数相关.338/.364；远隔mixed未恢复，结果全部保留。

用已知S的解析导数可将作用写成：

    Delta S = vbar^T d = ||vbar|| ||d|| cos(theta)
    vbar = integral_0^1 grad S(b+t*d) dt

已知链接上的固定16节点积分，复算误差<=2.39e-16。这与
[Integrated Gradients的路径完整性原理](https://proceedings.mlr.press/v70/sundararajan17a.html)
一致；本轮只积分最终数值评分函数，不冒充完整transformer电路归因。

在共同真实基线的诊断中，相邻有符号对齐相关为-.315/-.312，虽然范数相关
为正。最终评分函数的局部线性近似已经与原生变化r=.9981/.9998。
因此下一步不应继续调softmax或能量，而应研究配置向量如何取得查询方向。

误差进一步按固定顺序分解为幅度、方向、评分链接三项，复算误差<=3.35e-16。
相邻bag的RMS为.004086/.005255/.000048，position为.003954/.005763/.000159。
幅度同样有误差；这些项有协方差，不能当作独立人格方差比例。
真实基线只用于这项诊断，不进入候选预测器。

## 4. 决策与交付

保留查询定向基线与相邻mixed的局部价值；完整组装已能数值运行，但不要把
总相关当作局部机制全会了。下一执行[NEXT31位置特异读取](../../docs/GRAIN_POSITION_SPECIFIC_READER_PLAN_V1.md)：
重用12个原生梯度场，分别检查共同隐藏方向、位置系数和GRAIN方向作用。
先诊断压缩与跨作者运输，再决定如何数学化读取系数；0新神经调用。

81项相关测试通过；7构造+7主分析+2方向分析文件精确复现。只公开
聚合表，作者级材料、参数和预测保留在ignored results。所有本轮任务已结束，
无遗留后台队列；全系统goal保持ACTIVE。

- [公式、数学解释及范围](../../docs/GRAIN_COMPLETE_RESPONSE_DIRECTION_THEORY_V1.md)
- [完整结果表](query_pair_tables_v1/)
- 构造：`scripts/run_grain_query_pair.py`
- 主分析：`scripts/analyze_grain_query_pair.py`
- 方向解释：`scripts/analyze_grain_query_pair_direction.py`
- 数值构造先冻结于`95f472b`，主分析`d8bb93f`；方向补充分账`d3b0f7b`，没有改预测。

在独立库使用parent虚拟环境Python；必须选择未存在的输出目录：

```bash
python scripts/run_grain_query_pair.py --name logic_query_pair_new
python scripts/analyze_grain_query_pair.py --source logic_query_pair_new
python scripts/analyze_grain_query_pair_direction.py --source logic_query_pair_new
python -m pytest -q tests/test_grain_query_pair.py tests/test_grain_query_pair_analysis.py tests/test_grain_query_pair_direction.py
```

需要既有私有状态与源Jacobian缓存，但不需要载入神经模型或访问量表数据。
