# GRAIN：token组织与隐方向共同形成查询功能

2026-09-08。继[载体理论](GRAIN_QUERY_CONTEXT_CARRIER_THEORY_V1.md)，
由[本轮真实模型实验](../reports/grain_vector_mechanisms/QUERY_REGION_FUNCTIONAL_REPORT.md)推进。

## 从位置发现到构造方法

前包定位“问题文字不变但问题状态吸收作者证据”；本包得到跨作者可用的
矩阵读出。将token对齐后的查询区表示为Z_q∈R^(L_q×d)，不是只用h_last。
在声明的E/Q条件和编辑路径中，积分梯度给出精确的有限作用表示：

    Phi_P,u,q = <G_uq,D_uq>_F, D_uq=Z1_uq-Z0_uq.

这是微积分基本定理沿两条P路径的对称平均，不是新数学定理。本轮Simpson
近似是数值实现。新的经验结果是：以参照作者的平均G_ref,q替代G_uq，仍能
在两位留出作者读取三查询的同向作用，并明显优于简单token均值。

这里的“共享”不是让TF、N、确认用同一矩阵，而是**每查询有一套可跨人复用
的构造**。不同查询的矩阵可有部分共享的隐方向、不同token组织和尺度。

## 可解释的矩阵分解

    G_ref,q = U_q Sigma_q V_q^T
    Phi_hat = sum_k sigma_qk u_qk^T D_uq v_qk.

u不是人格因子，而是固定查询的token组织权重；v不是天然人格向量，而是该
观察器/条件下的功能隐方向；D携带具体输入变化。单个作者可以在相同函数下
产生不同响应，Who不要求只存在于G_u-G_ref里面。

若按token均值读取，相当于把矩阵投影到常数token方向：

    G_mean = 1_L (1_L^T G_ref / L).

本轮该投影明显损失作用。因而“句向量与人格词越近，就越属于该人格”的
单一距离图像过于简化。更合适的是：查询给出功能坐标和组织方式，作者证据
经固定前缀进入这些坐标，多个正负项共同形成判断。

保留前4个奇异项也不总能保留作用，因为所需目标是
E[<G-G_k,D>_F²]，并非只有||G-G_k||_F²。大能量压缩和功能压缩仍需分开。
这与此前GRAIN的功能投影思想一致，不因新token载体而推翻旧结果。

## 语义与输出组织不是二选一

完成数值分析后回看固定token，特质词附近存在较大贡献，但评分说明、标点
也参与。其向量已吸收上下文，不能仅凭字面是否“有心理意义”判断是否保留。
N的绝对路径贡献中评分指令约53%，TF查询内容约72%；这是路径坐标分账，
不是把评分格式称为人的人格成分，也不是旧prompt格式错误的再诊断。

因此先保留完整查询功能矩阵，再考虑语义/输出组织的机制分解。不要直接裁剪
成少数人格关键词，也不要反过来宣称全部效果来自格式。

## 主线连接与精确边界

    author texts + query + declared context
        -> fixed contextual representation Z
        -> group-reference functional matrix G_ref,q
        -> query-specific response configuration
        -> complete-judgment / GRAIN author-reference interfaces.

本轮已实现到响应配置的有限正例。Phi_P仍是Q0条件下的P作用，不能自动
替代完整自然Delta中的对称Phi_R，更不能直接当作者人格分数。该边界决定
下一步应连接哪些目标，而不是否定现在的函数。

下一包冻结矩阵在12位未参与构造的作者上只前向运输；若继续有用，停止该
层的局部精度迭代，转到完整推论和独立Who参照。persona写读对偶、更小特质
和跨模型是扩展，不阻挡当前可运行构造的形成。
