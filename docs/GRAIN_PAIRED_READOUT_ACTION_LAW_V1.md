# GRAIN 配对输出作用律：向量变化、计分敏感度与曲率

2026-10-08，HM21 已执行并异公式核验。承接 HM10 分类读者微积分、
HM10B1 输出 head 商空间及 HM15–20 干预作用；不是首次提出 softmax、
积分梯度或 Taylor 分解，也不是新的人格效度结果。

## 本轮改变了什么

已有向量近似没有复现完整的跨查询共同作用。HM20 的端点奇对称修正
只减少关键协方差差距的 1.2124%。HM21 定位到两个同时存在的来源：

- 已知非线性输出计分会放大或削弱共同结构，不能简单丢弃。
- 主要差距仍在局部线性作用中，不能仅靠修正输出曲率闭合。

完整属性应用组中 NA/低落查询的输出曲率项解释原 receiver 模型协方差
差距的 15.5590%，奇对称版本的 15.7500%。这些是指定样本、指定计分
单位下的有符号恒等分账，不是心理信息百分比或跨总体效应估计。

## 1. 已知观察律

固定候选数值 \(a=(0,1/9,\ldots,1)\) 和十个实际候选 logits。
只消去 softmax 的公共平移自由度，不中心化文本条件：

\[
 p_j(z)=\frac{e^{z_j}}{\sum_k e^{z_k}},\qquad
 f(z)=a^Tp(z),\qquad
 z\sim z+b\mathbf1.
\]

\[
 g_j(z)=p_j(z)(a_j-f(z)),\qquad
 H_{jk}(z)=\delta_{jk}g_j(z)-p_j(z)p_k(z)(a_j+a_k-2f(z)).
\]

\(H\) 是 **期望计分函数** 的 Hessian，通常不必正定。
它不是 log-partition 的 Hessian/Fisher 信息矩阵。后者为
\(\operatorname{diag}(p)-pp^T\)，半正定；不能把两者混称。

对于原始干预记录的实际参考端点 \(r\)、目标端点 \(t\)，令
\(d=t-r\)。有精确分解

\[
 y=f(t)-f(r)=L+C,\qquad L=g(r)^Td,
\]

\[
 C=\int_0^1(1-u)d^TH(r+ud)d\,du.
\]

同时，\(y=\int_0^1 g(r+ud)^Td\,du\)。主算法用固定32阶积分，
另公式用64阶，仅复核数值精度，不选阶数或模型。

有限分类族中的另一种 Hessian 收缩写法为

\[
 d^TH(z)d=\mathbb E_{p(z)}[(a-f(z))(d-\mathbb E_{p(z)}d)^2].
\]

独立复核通过这一中心矩计算，避免重用主算法的显式 Hessian 矩阵。
推导来自有限 softmax 的直接微分；指数族累积量视角给出了相同结构。

## 2. 共同作用并非单项误差的缩影

在固定 family 概率测度下，两个查询有

\[
 \operatorname{Cov}(y_q,y_s)=
 \operatorname{Cov}(L_q,L_s)+\operatorname{Cov}(L_q,C_s)
 +\operatorname{Cov}(C_q,L_s)+\operatorname{Cov}(C_q,C_s).
\]

因此 \(C\) 的 RMS 很小，不代表其共同结构影响也小。
若两个查询的主要协方差本来接近零，交叉项可以主导符号。
如果旧 compiled 作用为 \(P_q\)，精确差距又有

\[
 \operatorname{Cov}(y_q,y_s)-\operatorname{Cov}(P_q,P_s)
 =\underbrace{\operatorname{Cov}(L_q,L_s)-\operatorname{Cov}(P_q,P_s)}_{\text{局部线性作用差距}}
 +\underbrace{\operatorname{Cov}(L_q,C_s)+\operatorname{Cov}(C_q,L_s)
 +\operatorname{Cov}(C_q,C_s)}_{\text{非线性计分增量}}.
\]

四项保留符号和抵消。这里没有正交分解，不能将绝对值归一化后当方差贡献率。

## 3. 数据确认的性质

全部2172地址/724状态、7种干预、8/4/4来源/验证/应用词汇 family 保留。
312个精确 no-op 为零；没有新神经调用、拟合或真人数据。

完整属性应用组，按 NA/低落/信任三个冻结查询顺序：

| 项目 | NA | 低落 | 信任 |
|---|---:|---:|---:|
| 原生作用 RMS | .00510925 | .00396176 | .00858400 |
| 曲率/原生 RMS | 4.5790% | 1.8430% | 1.5933% |

全部72个角色/设计/查询统计中，该比值为0.0960%–4.6962%。
72个配对查询协方差中71个保持 tangent/native 同号；这是重叠设计的描述，
不是72次独立复制。唯一变号为来源 all-states 的 NA/信任：
\(+1.8814\times10^{-8}\to-9.1191\times10^{-8}\)。

完整属性应用 NA/低落：

\[
 \operatorname{Cov}(y)=8.24505\times10^{-7},\quad
 \operatorname{Cov}(L)=5.46605\times10^{-7},\quad
 \Delta_{\rm link}=2.77900\times10^{-7}.
\]

曲率增量相当于该 native 协方差的33.7050%，但相当于旧 receiver
共同作用 **差距** 的15.5590%。两个分母不能互换。
NA/信任的曲率反而削弱 native 正协方差6.2389%；不能统一解释为增益。

## 4. 下一条明确的机制问题

\(L=g_q(r)^Td_q\) 仍混合参考端点的计分敏感度与 query-conditioned
logit 位移。大部分差距在 \(L\)，不自动证明全部是 transformer 内部的新路径。
下一 HM22 使用 source-only 固定敏感度 \(\bar g_q\)：

\[
 y_q=\underbrace{\bar g_q^Td_q}_{B_q}
 +\underbrace{(g_q(r)-\bar g_q)^Td_q}_{G_q}+C_q.
\]

它将回答共同结构是向量位移本身携带，还是依赖接收参考状态的计分变化。
无需新增 fits/forwards，也不继续搜索 rank/alpha/精度网格。

这是已知观察律上的机制定位；原18协议/100卡/44问题、自然作者、
个人响应算子 \(B_u\) 与心理意义继续保留，不能从本包自动验收。

## 实现、复核与出处

- `grain_research/construction/math_paired_readout_transport.py`
- `scripts/run_grain_math_paired_readout_transport.py`
- `scripts/verify_grain_math_paired_readout_transport.py`
- `results/grain_math_response_v1/paired_readout_transport_r1`
- `results/grain_math_response_v1/paired_readout_transport_verification_r1`

258项关联测试（27文件）通过。2172动作/72统计/72协方差/432差距逐项
另算；Hessian 积分误差约 \(3.64\times10^{-16}\)。同agent异公式复算，
不是第三方盲复建。主代码、协议及结果冻结，不修改历史 HM10–20。

[Sundararajan等2017](https://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf)
§2–4.2给出路径积分/归因完整性的参照。本包仅对已知输出 map 使用微积分，
不能借完整性宣称唯一人格神经路线。
[Wainwright与Jordan2008](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf)
§3.5.1 Prop3.1解释指数族累积量导数和协方差；此处的计分 Hessian
由具体 softmax 直接推导。实际阅读限于相关章节，不声称全读305页。

OpenAI/math固定commit `adc7f1241b42e322a6451854ab7e4b4c146bf78a`
metric/connection稿07-global.tex190–222启发的是参考、观察与单位共同运输；
其椭圆PDE与唯一延拓假设没有被借来证明人格机制。
