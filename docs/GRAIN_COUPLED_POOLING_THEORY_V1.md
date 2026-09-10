# 从全局联动到内容条件化的跨 token 读法

2026-09-11，第57轮，NEXT68。见[实测报告](../reports/grain_vector_mechanisms/COUPLED_POOLING_REPORT_V1.md)。

## 1. 用户接点现在具体到什么程度

同一份材料既有 GRAIN 的局部表达方向 D，也有查询 q 的作用函数 F_q。
NEXT67 实测了改变前段 A 会改变未动后段 B 的读取方式；NEXT68 已把非零
联动放进一个可计算的标量函数。不是把两种作者总分直接相加：

\[
\widehat F_q(H,h)=F_{q,\mathrm{local}}(H,h)+\tfrac12v(H)^T M_qv(H),
\qquad \widehat\tau_{qf}=\nabla_H\widehat F_q:D_f.
\]

分数、token 作用、GRAIN 方向作用来自同一个函数，因而可以在同一原文地址
上逐层解释。当前 H16 仍来自原模型前缀；新程序替代的是后续读取候选，
不是已经删去全部神经编码器。

## 2. 精确限制：全局交互不等于内容特异交互

先用原始源作者固定 U8、中心和尺度得到 z_t。令

\[
b_t=\begin{bmatrix}z_t\\p_tz_t\end{bmatrix},\quad
v=T^{-1}\sum_t b_t-c_\mathrm{source},\quad
C_t=\begin{bmatrix}I\\p_tI\end{bmatrix},\quad p_t=2t/(T-1)-1.
\]

M 对称但不要求正半定，所以保留放大与抵消。对互异 token i,j，

\[
\nabla_{z_j}F_\mathrm{cross}=T^{-1}C_j^TMv,
\qquad \nabla^2_{z_j,z_i}F_\mathrm{cross}=T^{-2}C_j^TMC_i.
\]

当 A 改变但 B、last、布局不动时，定义半差 delta 和密度 k=T grad_z F：

\[
\delta k_j=C_j^TM\delta v=a+p_jb,\qquad j\in B.
\]

这条恒等式对有限端点也成立，不需要把干预当成无穷小。它揭示的问题是：
**后段具体是什么向量，不进入这次响应的局部形状；只进入位置。**
总体变化随 A 改变，但 B 中各处的变化仍只有常数和位置趋势两种模板。
中心 c 在半差中抵消；改变 source centering 或更精确求解不能突破此限制。

## 3. 数据给出的边界与新的开发决策

对每个源作者的 B 目标密度 Y_B，用 X_B=[1,p_B] 正交投影：

\[
P_B=X_B(X_B^TX_B)^{-1}X_B^T,\qquad
\|Y_B\|_F^2=\|P_BY_B\|_F^2+\|(I-P_B)Y_B\|_F^2.
\]

这里每人使用不同自由系数，是放松共享 M 后的乐观上限；不能把投影当成
源外预测。先按每人的 token/坐标数取平均能量，再按作者等权汇总，与主损失一致。
可表达份额为神经质2.939%、确认寻求3.034%；实测误差减少2.668%/2.568%，
已到这个乐观改进幅度的90.78%/84.64%。因此停止该线性汇总二次族的调参。
它证明一个小交互可被数值学出，也指出下一处功能限制，不抹掉 NEXT67 的发现。

## 4. 下一函数需要什么性质

更一般的同函数结构是

\[
v=T^{-1}\sum_t\phi(z_t,p_t)-c,\quad
F=F_\mathrm{local}+\rho_q(v),\quad
\nabla^2_{z_j,z_i}F=T^{-2}J_\phi(z_j,p_j)^T
\nabla^2\rho_q(v)J_\phi(z_i,p_i),\ i\ne j.
\]

旧版 phi 线性，所以 J_phi 不读内容。让 phi 在每个 token 上先非线性，
即可使相同全局变化在不同局部内容上产生不同响应。这里暂保留二次 rho，
一次只发展这个缺项；并不要求生成解释文本或重新训练完整 LLM。
这也区别于 NEXT59 的问题：不能先压成简单均值，再期待任意强的后端找回
被均值合并的配置。需要比较**非线性放在汇总前还是只有汇总后**。

原始研究可作为数学定位，而不是我们的准确率保证：

- [Deep Sets](https://arxiv.org/abs/1703.06114)提出逐元素映射后汇总、再读取的函数结构。
  我们保留带位置的元素，不能据此称文字本身顺序无关。
- [Wagstaff et al., 2019](https://proceedings.mlr.press/v97/wagstaff19a.html)
  给出连续 sum-decomposition 的通用表示维数限制。它提醒固定小汇总不是万能，
  但不直接证明本实验一定需要更高维，更不推出特定人格函数的最小维数。
- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)明确建模元素间交互。
  它提供之后考虑条件选择的路线；本轮没有实现该神经架构。

## 5. 心理辅助解释的位置

用户所说“同一句话的两个侧面”可保留为 GRAIN 表达配置与查询条件作用；
跨材料项则描述其它内容如何改变这里的作用。精神分析/交流分析的关系事件、
表达模式可作为另一个有定义的观察编码，不需要把二次数学系数命名成心理构念。
沿用[同材料解释接口](GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)：
原文地址、角色、固定数字、候选解释、人工复核分开。未做新的临床命名或调用。

下一步 [NEXT69](GRAIN_NONLINEAR_POOLING_PLAN_V1.md)已冻结为两个数值拟合的
同大小功能比较。整体 goal ACTIVE；不把完成一个函数族写成理论全局完成。
