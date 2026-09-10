# 查询差异也可部分编译：共同方向、条件读法与传入材料分开

2026-09-10，第三十七轮NEXT48。研究对象是相同作者材料、相同GRAIN方向
在两个模型查询下的实际局部响应，不是新增问卷预测实验。

## 1. 本轮提取到了什么

之前固定的是两查询共用的局部读向量。现在只从源4作者分别编译两查询
读向量，新8仍不提供当前preMLP24状态或门控。修正项与当前查询算子差异
r=.886208/.885192；加入后总对比r=.955274→.976535、.954853→.976360。

这支持一个可用的分离：作者材料通过条件化方向进入；局部读法包含共同
部分和小的查询特异部分。两部分都可从源材料构造并供另一批作者使用。
它并不要求每个作者都重新学一条人格轴，也没有把原生上游输入依赖抹去。

## 2. 完整局部公式

对允许的源作者集合S、查询q和固定共同读头w*，定义：

\[
v_q=\frac1{|S|}\sum_{s\in S}J_{24}(x_{s,q})^Tw^*,\quad
v_+=\frac{v_N+v_C}{2},\quad v_-=v_N-v_C.
\]

对当前传入方向\(d_q=\mathcal K_{u,q}D_{u,k}\)，直接得到：

\[
\widehat{\Delta a}_{u,k}
=v_+^T(d_N-d_C)+v_-^T\frac{d_N+d_C}{2}
=v_N^Td_N-v_C^Td_C.
\]

N是neuroticism查询，C在本文专指confirmation-seeking查询，不是Big5
conscientiousness。下一个attention实验的Q/K/V也不能与这两个查询代号混同。

第一项读取“同一读法下，传入路径已有何差别”；第二项读取“平均传入材料
被两种源固定读法如何区别对待”。这来自伴随关系与对称双线性差分恒等式。
新经验内容是源固定近似保留了实际响应，而不是恒等式本身具有原创性。

## 3. 恢复机制和抵消误差分别算

实际局部差分n由当前平均算子作用c和算子差异作用o组成，另有浮点余项。
源估计p、qhat满足：

\[
p+\widehat o-n=(p-c)+(\widehat o-o)+(c+o-n).
\]

新8自身模式查询项误差RMS=.012086，相比完全遗漏o时的.024635下降。
源替换误差=.013454，数值余项约1.84e-7；总误差=.023478，受正协方差影响。
此外\(e_{new}^2-e_{old}^2=2e_{old}\widehat o+\widehat o^2\)逐条核验。
因此本轮既直接测到query项重现，也测到最终误差改善，并非仅凭总r作判断。

保留差异：源4 F02的查询项SSE/zero略高于1，幅度也不是全体精确重现。
这些事实不会取消新8共享机制的用途，但决定我们保留近似形式，而不声称
所有人、所有因子都共用完全相同且无误差的心理机制。

## 4. 接回主线，而不是继续打磨此层

到这里，MLP24局部读法可以直接做固定向量内积；MLP25也已有独立的编译
读法。然而d仍由当前原生上游运输产生，不能把两个高r拼成原文到人格的
完整轻量模型。下一步要问：哪些attention计算把正文GRAIN方向送到读出位置？

已有[Transformer circuits框架](https://transformer-circuits.pub/2021/framework/index.html)
区分选择注意力位置的QK与传递内容的OV；[2025年QK attribution工作](https://transformer-circuits.pub/2025/attention-qk/index.html)
进一步用查询侧与键侧特征的双线性相互作用追踪注意力。它们给本项目提供
拆分机制的方法，而非人格效度。下一轮用实际权重对传入切向量展开Q/K/V
三个通道，不需要先替任何token或因子起心理名称。

尤其要补上一个前提差异：旧末端attention分析固定最终query位置，只扰动
BODY，因而直接query导数为0；传播到attention22的方向可能已改变最后
query位置。旧公式在旧边界内仍成立，新边界必须加入这一项。这是推进逻辑
接口，不是用新结果否定原先限定条件下成立的数学。

按[既定停止规则](GRAIN_SOURCE_QUERY_CORRECTION_PLAN_V1.md)，不再增加
MLP24局部候选。见[结果](../reports/grain_vector_mechanisms/SOURCE_QUERY_CORRECTION_REPORT_V1.md)、
[全链依赖图](GRAIN_UPSTREAM_MECHANISM_INTEGRATION_V1.md)及
[NEXT49](GRAIN_UPSTREAM_ATTENTION_CHANNELS_PLAN_V1.md)。
