# GRAIN：token配置、特征商空间与兼容标量读法

2026-09-11。NEXT59完成，见[实测报告](../reports/grain_vector_mechanisms/POOLED_FEATURE_NULL_RESPONSE_REPORT_V1.md)。
延续GRAIN测量对象、同材料查询响应与外部效标三条线，不将它们合并成一个真值。

## 1. 当前具体缺失的不是另一个回归系数

NEXT57的phi仅含源PCA32正文均值及末token；NEXT58定位源正文/查询补偿。
NEXT59进一步把原生GRAIN响应分成该phi可见/不可见部分。令源正交基U，
P=UU^T。只考虑BODY方向空间，使用固定Frobenius内积：

\[
\mathcal P D=\mathbf1\bar D P,\qquad
D=(\mathbf1\bar D P)+(D-\mathbf1\bar D)
+\mathbf1\bar D(I-P).
\]

calP是正交投影。三项依次是可见均值、零均值token分配、PCA外均值。
三者两两正交、能量相加；后二者在Dphi的核内。

对任何可微f，由链式法则

\[
D(f\circ\phi)_X[D_{null}]
=Df_{\phi(X)}[D\phi_X[D_{null}]]=0.
\]

本轮原生DL[D_null]的96个值都远离数值零，故原生L的这组局部作用不能
通过当前phi表达。它不是有限训练集上模型好坏的判断，而是固定输入契约的
局部反例。它也不否认某f(phi)可预测部分作者分数，更不否认其它embedding。

## 2. 同材料接口其实是一个交叉协方差

令g_t为原生H16 BODY梯度，D_t为GRAIN方向：

\[
\tau_{alloc}=\sum_t\langle g_t-\bar g,D_t-\bar D\rangle
=T\,\operatorname{tr}\operatorname{Cov}_t(g,D).
\]

协方差分母取T；g是协向量，D是向量，上式先固定坐标/度量。更一般的
可逆重参数化下g与D须对偶搬运才能保持收缩；欧氏投影本身并非任意变换不变。

同一作者的GRAIN因子方向在不同token上产生不同配置，查询函数也在不同
token上有不同敏感度。二者的**对齐方式**决定局部作用，不由两者各自均值
决定。新8的分配项投影N=.945894、确认=.943346就是这一对齐的实际量化。

严格区分：零均值配置可以包含内容分布、上下文编码和位置。它不单独识别
纯词序效应；也不能从相同下标推断不同作者的token具有同一语义。

## 3. 从响应到评分，必须保留一个函数

只学一个任意向量场Jhat(X)可得到局部作用，却未必存在标量Lhat使其为梯度。
NEXT34已经指出这一整合缺口；NEXT57恢复了同函数兼容性，却用了过窄的phi。
下一步应同时保留：**token内非线性读取、可比较聚合、标量与导数兼容**。

候选形式不是f(mean X)，而是

\[
\widehat L_q(X)=b_q+v_q^\top z_{last}
+\frac1T\sum_t\psi_q(z_t,p_t),
\quad z_t=\operatorname{diag}(s)^{-1}U^\top(X_t-\mu),
\]

\[
\psi_q(z,p)=a_q^\top z+\sum_m c_{qm}
\exp[-\|(z,p)-\ell_m\|^2/(2\sigma^2)].
\]

查询q决定源函数；p是声明的相对位置特征。核在均值之前作用，梯度随token
变化，因此可以读取旧phi的零均值方向。核中心、基底和全部系数只从源作者
构造。GRAIN字段仍定义方向，不靠把待预测作用塞回输入来实现好结果。

这是一个有限、可微、易复算的候选，不声称它完整表达LLM所有token交互。
保留最后token项也保留NEXT58指出的源支持问题，应实测完整L而不只报tau。

## 4. 先行研究提供的数学工具

[Deep Sets (Zaheer et al., 2017)](https://papers.nips.cc/paper/6931-deep-sets.pdf)
§2–3将逐元素映射与对集合聚合区分，并允许条件输入。我们借用“非线性映射
在聚合之前”这一结构；原文一般性定理有可数域/固定大小等前提，不为当前
64个有限核或变长文本赋予普适表达保证。携带p后的集合是带地址集合，换行
枚举与交换其地址不是同一种操作。

[GDML (Chmiela et al., 2017)](https://arxiv.org/pdf/1611.04678)通过受约束
梯度域学习保证能量/力来自同一势函数，方法使用核的Hessian结构。这里借用
**标量与梯度的兼容性**；不是假设人格服从能量守恒，不直接复现其分子核、
对称性或精度结论。本次拟议实现是有限标量核展开及其解析导数。

[Sobolev Training (Czarnecki et al., 2017)](https://arxiv.org/abs/1706.04859)
提供值与导数联合教学的直接先例。本计划进一步使用源token梯度场约束，而
不是只用每作者四个方向；token行增加函数约束，但不增加独立作者样本数。

## 5. 和历史理论接上，而非重造一套

旧[S6与配置理论](GRAIN_AGGREGATE_CONFIGURATION_THEORY_V3.md)已表明正文
均值保留干预会改变原模型。旧[局部读法](GRAIN_POSITION_READER_THEORY_V1.md)
表明共享方向可存在、纯位置系数不够。NEXT59没有推翻它们，而把这些性质
定位到当前GRAIN四方向与完整原生logit，并量化旧phi的具体失效位置。

NEXT58描述“源样本如何补偿”；NEXT59描述“输入在任何读头前已丢了什么”。
两者可以同时成立。下一轮从原子token标量函数入手，能否保持跨作者完整
评分以及查询差异由实验决定，不能用兼容性定理代替实测。

下一步：[NEXT60](GRAIN_TOKEN_POTENTIAL_EXTRACTION_PLAN_V1.md)，不再开均值
后非线性/增维/alpha网格；保留旧E5、GRAIN测量与局部可用读法。
