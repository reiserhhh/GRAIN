# 同一文本方向如何改变“选什么证据”：上游Q/K/V机制

2026-09-10，第三十八轮NEXT49。主线从局部MLP读法回到正文证据的条件化
运输；原模型、作者、查询、GRAIN方向均不变。

## 1. 新增的机制证据

相同正文GRAIN方向传播到attention22后，不仅改变value，也改变key和最后
query。新8作者中，key通道对本层两查询差异的投影约.904，query约.083，
value约.013。但源4的对应值为.271/.562/.167，不能给所有材料统一删掉Q。
两个查询本身也不同：新8神经质查询Q是负向调整，确认寻求查询Q与K都有
正向作用。完整分解对96个原生向量的相对误差最大约5.27e-6。

这些系数是固定读头下有符号的响应分账，不是人格比例。当前证据定位到
“文本的GRAIN方向能够影响模型如何选取证据”，比仅知道分数相关更具体。

## 2. 三条路径来自同一微分，而非三类心理标签

略去head下标，固定mask和位置，令

\[
a=\operatorname{softmax}(s),\quad s_t=Q^TK_t/\sqrt h,\quad o=\sum_t a_tV_t.
\]

由\(Da=\operatorname{diag}(a)-aa^T\)，得到

\[
\dot o=\underbrace{\sum_t a_t\dot V_t}_{V}
+\underbrace{\sum_ta_t(V_t-o)\frac{\dot Q^TK_t}{\sqrt h}}_{Q}
+\underbrace{\sum_ta_t(V_t-o)\frac{Q^T\dot K_t}{\sqrt h}}_{K}.
\]

所有点表示GRAIN方向的导数，未必是时间变化。RMSNorm的径向项、RoPE和
GQA在实算中保留。因此它既不是直接对词频加权，也不是把attention权重本身
等同于解释。关键还有\(V_t-o\)：受到更多关注的状态，其携带内容若与当前
平均相同，选择变化的这个一阶作用就为0。反向项可能是有用调节，不是垃圾。

对固定输出读向量，把\(g=W_O^Tw^*\)分到各head，K通道可写为

\[
\tau_K=\sum_h\frac{Q_h^T R_{h,k}}{\sqrt{d_h}},\qquad
R_{h,k}=\sum_ta_{h,t}\left[g_h^T(V_{h,t}-o_h)\right]\dot K_{h,t,k}.
\]

这是“查询方向与证据响应向量相接”的一个明确局部数学形式。注意a和o也
依赖Q，所以不能把整个函数误说成对Q全局线性；R不是独立于查询的现成人格
向量，更不是唯一可辨认的心理构念。下一步可以实际替换Q来检验其可复用性。

## 3. 对早先路径论的修订

原末端attention实验固定最终query，只在该层BODY输入施加扰动，故直接
\(\dot Q=0\)。该限定下的K/V分解仍然正确。新实验从H16传播五个block，
最后query位置已经收到正文变化，不能套用旧边界。省略Q会留下new8约.110、
source4约.668的误差/原生作用能量比。问题在接口前提，不在旧定理被推翻。

先前MLP25/24的固定读向量说明部分末端处理可共享；这次说明共享读法之前
的材料选择仍条件化。完整链条是：同材料方向D → 全token运输 → 条件选择
与写入 → 后续调整/读出。局部导数与完整端点、量表效标、人格解释继续分层。

## 4. 文献对应与新实验的问题

[Elhage等的QK/OV框架](https://transformer-circuits.pub/2021/framework/index.html)
区分“注意哪里”与“写入什么”。[Kamath等2025](https://transformer-circuits.pub/2025/attention-qk/index.html)
用query/key特征对的双线性形式解释注意力分数，并指出softmax竞争与位置
编码需要处理。我们的微分不是新发明的定理；本轮把这套标准计算具体接到
真实作者材料的GRAIN方向，保留归一化、位置、竞争和有符号通道，并与原生
更新逐向量核验。这是本项目新增的经验接口。

NEXT50以源作者的固定query向量作“引子”，作用于当前证据K/V及其方向，
检验局部导航函数能否复用。只问这一个问题，不把source4/new8差异叫成人群
心理机制，不从未命名的通道直接推导精神分析或交流分析诊断。

见[修正版报告](../reports/grain_vector_mechanisms/UPSTREAM_ATTENTION_CHANNELS_REPORT_V1.md)
与[下一计划](GRAIN_SOURCE_QUERY_SELECTION_PLAN_V1.md)。
