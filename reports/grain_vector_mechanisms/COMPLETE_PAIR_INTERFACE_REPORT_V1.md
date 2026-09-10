# 完整接口第一次实测：去掉借用端点后，基础上下文读法成为主缺口

2026-09-10，第十九轮，NEXT26有限实验完成。

**本轮把“局部向量作用可读”接到了一个真实可独立执行的完整条件评分程序，
并定位了主要断点：通用上下文相似度没有恢复作者间的基线差异。** 下一步
研究查询定向的上下文作用，不继续优化已经很小的双片段交互误差。

## 1. 做了什么

24旧N源作者、4作者互斥折；每折18训练作者与6测试作者。两个固定评论对
（0,1与0,11），两种继承Fourier输入。0新大模型调用、0问卷读取、0调参网格。

输入来自已缓存块16状态：12评论均值M（18,432维）、136保留模板位置R
（208,896维）、布局L（163维），加各评论的源固定Fourier配置。R可能已经
含正文信息，不能当作“纯噪声”。上游仍依赖原神经编码，不称raw-text LLM-free。

新建4个基线头与8个共享单评论增量头，保留8个旧混合读法；中心化logit公式为：

    z_pair_hat = z0_hat + di_hat + dj_hat + mixed_hat
    score = original_alphabet_softmax_expectation(z_pair_hat)

源拟合后，当前作者不再提供z0/zi/zj答案。四个独立子进程只准读取数字输入和
源模型，禁止全部其它文件；12次禁读探针成功，完整logits及分数逐数重放误差0。
这是运行依赖的闭合，不是新候选测量质量过关。

## 2. 完整评分与借用答案的组装分开看

目标是固定P/Q、只恢复两条评论配置的原模型N评分，不是完整自然文本F或
人的神经质。24人以前已经用于源研究，本轮是探索性OOF，不是新的独立复现。
表中CI按作者整组bootstrap2000次，仅为固定预测的点位区间。

| 输入/评论对 | 借真实additive的RMSE | 全部预测RMSE | 全部预测r及95%CI | 折内r | 源基线均值RMSE |
|---|---:|---:|---:|---:|---:|
| fourier_bag / adjacent | 0.000492 | 0.014702 | -0.034 [-0.406, 0.360] | 0.125 | 0.014736 |
| fourier_bag / distant | 0.000256 | 0.017441 | -0.183 [-0.542, 0.223] | 0.049 | 0.016920 |
| fourier_bag / pooled | 0.000392 | 0.016130 | -0.107 [-0.443, 0.267] | 0.083 | 0.015865 |
| fourier_position / adjacent | 0.000490 | 0.014027 | 0.133 [-0.374, 0.545] | 0.269 | 0.014736 |
| fourier_position / distant | 0.000180 | 0.017405 | -0.146 [-0.518, 0.241] | 0.077 | 0.016920 |
| fourier_position / pooled | 0.000369 | 0.015806 | 0.001 [-0.363, 0.321] | 0.147 | 0.015865 |

相邻position保留一些折内变化（r=.269），但总体差异与源均值基线的差别很小。
两种输入的pooled完整预测r=-.107/.0006，不能将借用真实端点时的约.999相关
当作独立模型精度。**借用组装是诊断对照，不是原结果被改成无效。** 它仍
说明旧混合作用读法可修正一个已知起点，但没有负责求出这个起点。

## 3. 断点在哪里

基线本身：r=-.164，折内r=.010，RMSE=.014795；源均值RMSE=.014197，
原基线作者间SD=.013793。全logit的基线误差也比源均值大8.65%。
全部预测的输出SD只有约.0032-.0038，目标SD约.0141-.0164；这次读法主要
学会共同分布而非作者偏离。

即使把mixed替换成真实值，pooled RMSE仍为bag .016185 / position .015826，
与全部预测 .016130/.015806几乎相同。主要缺口是基线及单评论作用，不在
最后softmax或mixed的精修。所有24个配对比较均公开，不能只保留有利格子。

基线logit误差平方=.006779，是最大单项；但不能直接除总MSE称方差贡献率。
例如position相邻的base×j交叉项=-.002445，误差会抵消。完整有符号分解
逐作者闭合（见summary误差值）。远隔组“真实mixed略差”也是这种补偿，
不是错的交互机制比真的更好。

## 4. 这次如何推进理论

需要区分：输入/资产可独立运行、输入保留所需信息、函数学会正确读取。
三者不是同一个结论。M/R/layout为均值化上下文保留完整声明坐标；Fourier
配置仍然是压缩，不能断言原配对信息全部可逆。18源作者和这一个核也不足以
区分“支持覆盖不足”与“函数族不合适”。

本轮核Ridge采用标准有限源展开，依据
[广义表示定理](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)，不是声称
发明新定理。机制上的下一步是用原模型对条件均值/保留位置的Jacobian定义
查询敏感方向，再把这些方向编译成固定数学读法。

这与GRAIN的接点保持不变：同一材料的群体位置/分布是一侧，特定查询如何
读取它是另一侧。既有同材料关联、原生token局部作用均保留。精神分析/交流
分析类辅助解释沿既有证据绑定协议，不能把这里的模型N分数当临床诊断。

## 5. 验收、文件与下一步

- 计划bf21c94、实现6f09daf先于评估。54相关测试通过，18构造+6分析
  文件重跑一致。新增基线诊断写analysis_r2，原analysis与原计算保持不变。
- source-state/作者ID/模型/逐作者预测全部保留在ignored results；本目录只含
  聚合报告。两次执行都结束，本轮无遗留后台任务；无release或lockbox更改、未push。
- 下一项：[NEXT32查询定向基线读法](../../docs/GRAIN_QUERY_DIRECTED_BASELINE_PLAN_V1.md)。
  四个旧固定源的条件Jacobian，20旧作者检验，源匹配对照，限制调用与时间。
  不改成新精度搜索；成功或失败都要定位具体读取机制。

相关入口：
- `grain_research/vector_mechanisms/complete_pair.py`
- `scripts/run_grain_complete_pair.py`
- `scripts/replay_grain_complete_pair.py`
- `scripts/analyze_grain_complete_pair.py`
- [理论与公式](../../docs/GRAIN_COMPLETE_PAIR_INTERFACE_THEORY_V1.md)
- [定量表和复算记录](complete_pair_tables_v1/)

复算（在独立仓库，用parent虚拟环境Python；新name避免覆盖）：

```bash
python scripts/run_grain_complete_pair.py --name logic_complete_pair_v1
python scripts/analyze_grain_complete_pair.py --name analysis_r2
python -m pytest -q tests/test_grain_complete_pair.py
```
