# GRAIN：值与方向的补偿、可观察输入与下一表示

2026-09-11。NEXT58计算完成；[报告](../reports/grain_vector_mechanisms/VALUE_TANGENT_COMPENSATION_REPORT_V1.md)。
本页推导的是固定数值函数的性质，保留GRAIN自身的群体测量理论。

## 1. 一个函数同时给值和方向，不代表两者被同样约束

令源固定输入为z=(z_B,z_Q)，函数为

\[
\widehat L(X,q)=\bar L+s_L(a+w_B^\top z_B(X)+w_Q^\top z_Q(X,q)).
\]

GRAIN方向D只改变本实验H16的BODY地址，最后token不动，故

\[
D\widehat L[D]=s_L w_B^\top D z_B[D].
\]

给源数据加入方向约束可以改变w_B，而w_Q在少量源评分点上补偿其影响。
数学兼容性是必要条件，但不是有限源数据唯一确定函数的充分条件。
实际源设计秩20、参数65，未约束维数45；Ridge选定唯一参数来自惩罚规则，
不是多出了45条观察。

NEXT58直接发现源B/Q增量相关近-1，而新8为-.235/-.161。加性函数可以用
跨视图相关完成构造内补偿，无需乘法交互；这不等于原生网络的因果结构。

## 2. 将“互相抵消”转成可复算量

对value/joint两个归一化匹配程序，DeltaL=I+B+Q精确成立。常用两个量：

\[
\kappa=\frac{\operatorname{Var}(B+Q)}{\operatorname{Var}(B)+\operatorname{Var}(Q)}
=1+\frac{2\operatorname{Cov}(B,Q)}{\operatorname{Var}(B)+\operatorname{Var}(Q)},
\]

\[
\Delta\operatorname{MSE}=\mathbb E[2e\Delta L+(\Delta L)^2]
=\sum_{v\in\{I,B,Q\}}\mathbb E[2ev+v\Delta L].
\]

方差取有限组的总体分母n。分母为0则kappa未定义，不填0。kappa可大于1；
MSE分账可有负值。它们既不是因果比例，也不是某人的心理组成百分比。
本次源kappa约1e-6，新8约.85/.87；源留一甚至略大于1。

## 3. 两种不同的“看不见”

**源数据支持不足**：给定特征phi，新输入行r不在源设计A的行空间中。
覆盖率c(r)=||P_row(A) r||²/||r||²。新8评分c约.35/.52，方向约.80–.88。
补源材料或先验可能改善这类问题。

**输入表示本身丢失**：方向D满足Dphi[D]=0，但原函数DL[D]不为0。
这时任何可微f(phi)在D上的响应都为0，是链式法则，不因换非线性头而消失。
补更多同phi训练行也不能在该方向恢复响应，必须改变输入表示或测量目标。

NEXT58只实测第一类和补偿。第二类由下一冻结实验检查，不能提前宣称结果。
反过来，DL在已试方向的零空间上为0，也不保证phi全局充分；有限方向只给
局部信息。这是有决策意义的最小检查，不是要求先穷尽所有向量才继续。

## 4. 将token/slice接点写成明确可计算对象

对源正交基U、BODY均值m，phi=(U^T(m-mu),U^T(X_last-mu))。
把BODY方向D按当前声明的欧氏几何分为：

\[
D_t=D_t^{\mathrm{vis}}+D_t^{\mathrm{alloc}}+D_t^{\mathrm{out}},
\quad D_t^{\mathrm{vis}}=\bar D UU^\top,
\quad D_t^{\mathrm{alloc}}=D_t-\bar D,
\quad D_t^{\mathrm{out}}=\bar D(I-UU^\top).
\]

仅BODY位置使用上述式子，其余位置为0。后两项的Dphi均为0；前者保留
原方向的全部Dphi。三项两两正交，故向量能量相加。原生BODY梯度g给出

\[
\tau=\langle g,D\rangle
=\tau_{\mathrm{vis}}+\tau_{\mathrm{alloc}}+\tau_{\mathrm{out}}.
\]

**向量正交不代表三个标量响应不相关**；作用能量必须保留交叉项。这里
alloc测“同样均值如何分到不同token”，out测“均值里的源PCA盲区”，不把
任一项预先命名为人格、情绪或噪声。它直接将GRAIN方向与同地址模型读法相接。

## 5. 文献如何帮助这一步

- [Czarnecki et al., 2017, Sobolev Training](https://arxiv.org/abs/1706.04859)：
  函数值与导数联合学习是已有方法；NEXT57将其用于源GRAIN方向。文献不是
  四作者仿射模型能够跨作者泛化的保证。
- [Perez et al., 2018, FiLM](https://arxiv.org/abs/1709.07871)：条件输入可以
  调制特征的仿射读法。它提供后续BODY-query交互设计，不说明Qwen或人的
  人格推论就是FiLM；在相同phi上增加交互不能修补phi本身的零空间。
- [Constantine, Dow & Wang, 2014](https://doi.org/10.1137/130916138)：使用输出
  梯度识别敏感子空间，为“按响应而非仅输入方差选表示”提供数学路线。原文
  有[勘误](https://doi.org/10.1137/140983598)，若采用误差界必须使用修订定理
  并核对分布/正则性前提。本轮不套用其全局界，也没有新估计active subspace。

后续若需学习源响应子空间，应使用源梯度与固定参照，先解决变长token的共同
地址；不能将不同作者的第t个token无条件当作同一观测变量。

## 6. 两个用户方向的接合方式

同句话既有GRAIN群体位置，也有查询定向作用；它们可以是两个作者侧面，
也可以通过token分配耦合，后者正是NEXT59要直接检查的输入机制。
心理辅助投映解释仍按[同材料接口](GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)：
数值、原文地址、上下文角色、候选解释与人工复核分开。精神分析/交流分析
的可操作化关系编码提供解释视角，不回写分数，不替代治疗者判断。

下一步：[NEXT59](GRAIN_POOLED_FEATURE_NULL_RESPONSE_PLAN_V1.md)。当前加性池化
家族保持关闭，先选择需要保留的输入，而不是继续在同12人上找更高r。
