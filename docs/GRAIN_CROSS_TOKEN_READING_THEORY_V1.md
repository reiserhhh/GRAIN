# 跨材料读取：从一阶方向到混合响应

2026-09-11，NEXT67。测量对象是固定原模型的人格查询函数，而非人的心理变量。
同一句/一组材料给出GRAIN方向与查询响应，两条路线在同一向量地址上连接。

## 1. 当前已测接口

固定查询q，在H16边界把正文分为A、B，其余状态合记h。模型评分为

\[
L_q(H_A,H_B,h),\quad g_{B,q}=\nabla_{H_B}L_q,\quad
\tau_{B,j,q}=\langle g_{B,q},D_{B,j}\rangle.
\]

D是原始材料上的GRAIN方向。原接口问“沿D移动是否影响查询”；本轮问
“只移动A，是否改变不动的B沿D的作用”。固定D避免把方向重估混入读法变化。

## 2. 有限响应的数学对象

设A扰动为单位方向E_A，步长epsilon。理想精确算术下：

\[
\Delta g_B=\frac{g_B(H_A+\epsilon E_A)-g_B(H_A-\epsilon E_A)}2
=\frac12\int_{-\epsilon}^{\epsilon}
\nabla^2_{B,A}L_q(H_A+tE_A,H_B,h)E_A\,dt.
\]

对B方向投影得到

\[
\Delta\tau_{B,j,q}=\epsilon\,D_{B,j}^{\top}
\overline{\mathcal H}_{B,A,q}E_A,
\quad \overline{\mathcal H}=\frac1{2\epsilon}\int_{-\epsilon}^{\epsilon}\mathcal H(t)dt.
\]

这是沿指定路径的平均混合Hessian作用，不是完整Hessian，也不是点Hessian。
仅在足够光滑的小步长极限下，除以epsilon才趋向点混合导数。实际float32
两端中点有约3.76e-6范数舍入误差；原始状态和实际扰动均保存，没有把
理想路径恒等式冒充浮点逐点精确等式。多项式合成测试验证积分关系。

标量C2函数在同一点有H_BA=H_AB^T；这不代表A扰动整段路径与B扰动另一段
路径的有限结果相等，也不意味着人的对话因果关系可逆。本轮只测前者一种干预。

## 3. 为什么旧公式必为零

\[
\widehat L_q=a_q(h)+\frac1T\sum_t f_q(H_t,p_t,h).
\]

位置、长度、last与源归一化固定时，不同BODY token分别进入不同项，
因此H_BA=0。单token内非线性再强、核中心再多也不能产生此跨token作用。
原生8条件非零与旧模型精确0的对照，是一个函数族限制的直接见证。

H16前缀自身已经是上下文化编码；以上是边界处的偏导数契约，不假设
真实文字变化时last或B编码仍不变。本轮是模型内部机制实验。

## 4. 下一种可兼容函数

保留局部项，令v(H)=mean_t psi(H_t,p_t)-源参照，再引入

\[
\widehat L_q^{joint}=\widehat L_q^{local}+\tfrac12v^\top M_qv,
\qquad M_q=M_q^\top.
\]

对互不重叠的token i,j，局部psi的Jacobian记J_i、J_j，则

\[
\nabla^2_{H_j,H_i}\widehat L_q^{joint}
=T^{-2}J_j^\top M_qJ_i.
\]

这直接补入缺失的依赖。M在评分与导数中共用，参数仍线性可解，保留
保守梯度场及对称混合导数；不是另外训练一个与评分不兼容的响应头。
任意有限维池化仍会限制可表达方向，因此此项是可检验近似，不是原模型全等式。

较一般的非线性rho(mean psi)有J_j^T Hess(rho) J_i/T²。
归一化选择sum softmax(s_t)v_t也能通过共同分母产生跨token依赖；两者均可
解释“别人/别段的证据改变当前证据的读法”，NEXT67没有识别唯一实现来源。

## 5. 既有数学工作的对应与区别

Janizek、Sturmfels与Lee的Integrated Hessians用二阶路径作用分析特征交互，
而不是把单特征重要性当全部解释。我们测的是一个GRAIN组合方向到另一
区域四方向的有限响应，并未计算其双重路径积分及完整性归因。
这种联系指导对象定义，不把既有方法重新命名为原创定理。
[JMLR 2021，§2、式3及交互对称性](https://www.jmlr.org/papers/volume22/20-1223/20-1223.pdf)。

Deep Sets的非线性rho(sum phi)说明汇总后继续计算与单纯求和不同。
本项目把位置作为特征保留，不宣称自然语言顺序无关；其表示定理的
集合/连续性/维数条件也不是当前16维函数的普适保证。
[Zaheer等，2017](https://arxiv.org/abs/1703.06114)。

Set Transformer明确建模输入元素之间的交互，提供联合读取的另一架构。
本轮不照搬大型模型，先检验可解析的二次项是否能读取已测的跨区域作用。
[Lee等，ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)。

## 6. 对全系统逻辑的升级

1. 作者表达位置D、原生查询梯度g，以及材料间如何改变g，成为三个可计算对象。
2. 不能仅凭分数相关选定机制；存在性实验先决定所需函数依赖，再训练读取器。
3. 两查询响应有共同方向也有差异；当前半和/半差是算术分解，不是独立潜变量。
4. 辅助解释可引用具体“在何种其它证据下，此表达作用变化”的证据。
   精神分析/交流分析构念仍需单独编码与经验连接，本轮没有临床命名。

[结果](../reports/grain_vector_mechanisms/CROSS_TOKEN_READING_REPORT_V1.md)；
[下一轮执行计划](GRAIN_COUPLED_POOLING_PLAN_V1.md)。
