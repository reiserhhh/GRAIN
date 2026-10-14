# HM23：完整输出向量运输与已知计分接口

2026-10-08。HM22之后的下一有限包；尚未执行。
主命名空间 `results/grain_math_response_v1/output_vector_transport_r1`，
另有 `output_vector_transport_verification_r1`。不改HM10–22冻结代码/结果。

## 科学问题

HM22指出：部分主要共同作用已经存在于固定计分敏感度下，但有些符号
依赖参考敏感度。现在不再无限细分同一缓存。实际改变模型所预测的对象：
**同一来源输入和同一核，把直接标量作用替换为完整输出logit位移，随后
用已知分类计分函数解释作用。** 检查丢失共同结构来自标量拟合方式，
还是当前输入/核本身不足以运输输出向量。

## 固定边界

- 2172原地址/724状态/7臂；source1080、validation528、application564；
  8/4/4词汇family、归属、顺序、长度、312 no-op与所有失败保留。
- 复用HM16 source origin、context/delta RMS、360个唯一来源状态的权重。
- 只用原 receiver_tensor 核、原平均对角单位、alpha=1、零动作截距。
  不增加kernel/degree/rank/alpha/layer/prompt/anchor网格。
- 0新神经调用/训练/API/私文/Claude/自动化/发布/旧队列。
  允许一个CPU多右端来源拟合；不是新神经训练。
- 已知候选alphabet和logit公共平移商沿用，不命名成新人格因子。
- held实际reference logits是既有观察器缓存，**不是**LLM-free原始文本
  推断输入。这一包专门定位输出作用运输，独立参考函数仍另待开发。

## 1. 准备与来源矩阵

核验HM22主/verification hash及HM16输入/模型、原native地址。冻结本包
配置、代码、计划后，先只加载source reference/target候选logits。
每状态对三查询保存

\[
D_{i,q}=\Pi(z_{i,q,t}-z_{i,q,r}),\quad
\Pi=I-\mathbf1\mathbf1^T/10.
\]

30列完整输出，实际每查询为9维零和商；不选低秩。保存列顺序和地址，
360×30来源矩阵不可混入held端点或沿query改变来源状态。
各列沿用自然logit单位，不做应用数据标准化或反向调幅。

\[
x_i=\Delta h_i/s_x,\quad c_i=(h_{i,r}-o)/s_c,\quad
K_{ij}=\langle x_i,x_j\rangle(1+\langle c_i,c_j\rangle).
\]

这里 \(h\) 是冻结HM16的有序多span观察，不是单一平均embedding。
kernel unit \(u\) 必须与旧receiver模型逐值一致。

## 2. 一个多右端来源拟合

\[
S=W^{1/2}(K/u)W^{1/2}+\alpha I,\quad
A=W^{1/2}S^{-1}W^{1/2}D.
\]

用一个共同SPD系统和30右端求解，保存dual、source targets、系数单位、
残差、完整source文件hash及未读held标志。它等于30个同核Ridge输出；
不能因为共同求解就宣称学得心理共享路径或耦合multi-task机制。

\[
\widehat D_{*,q}=(K_{*,S}/u)A_q,\quad
\widehat y_{*,q}=f(z_{*,q,r}+\widehat D_{*,q})-f(z_{*,q,r}).
\]

no-op输入对应零输出。reference不替换为来源均值。公共logit平移与
候选坐标同步重排不得改变结果。

## 3. 评价改变了哪种机制

来源模型冻结后才读held目标端点，评价全部原地址，不挑选NA/低落赢家。
记录完整向量位移误差、每查询作用RMS/MAE/RMSE/Pearson、family对照，
以及所有角色/7臂/三查询对的共同协方差与相对旧scalar receiver的差距。
主要对象是共同作用运输；不把一个更低MSE冒充机制完成。

使用HM22同一anchor对新预测位移分账 \(\widehat B,\widehat G,\widehat C\)。
核验三项和等于新非线性作用，不改anchor或旧native分账。
保留逐地址的vector remainder与其跨查询关系，不把它删为噪声。

关键判别：若固定B结构改善而G/C仍失配，保留参考作用的下一层；若
原生joint符号/幅度改善，确认“运输完整向量后再计分”的机制价值；若
共同结构仍未保留，输入/源支持/核的缺口不能再归给scalar输出压缩。
后者触发最小局部decoder Jacobian干预计划，不续核或秩网格。

## 4. 必要代数见证与独立算术

- 单独的synthetic constant-covector线性readout：多输出拟合再收缩
  必须等于先收缩目标再拟合。防止把单纯列增多当共享机制。
- 变化reference且非线性readout反例：上述交换关系一般不成立。
- 显式feature map \([x,x\otimes c]\) 的小规模primal/dual对照，
  以及真实360×360共同system用另分解复算，不能调用同一个fit helper。
- 零和gauge、alphabet同步重排、no-op、来源角色泄漏、invalid data，
  小分母协方差/非零抵消；记录误差不改容差来通过。
- 逐地址重算全部预测、统计、分账与协方差；family区间是固定已知模型
  的材料不确定性，不是新心理作者总体CI。

## 5. 可验收交付与终止

新helper/runner/verifier/tests/config，source-only model manifest，完整
outputs/actions/metrics/covariance/remainders/hash和中英文数学定义。
关联focused tests实际运行一次，报告真实范围而非相加测试数。
报告明确给出“完整向量运输是否改变共同作用”的一项决策和下一最小信息。
不能用代数恒等式、拟合成功或文件存在宣布主线理论完成。

本包成功包括发现有效机制，也包括定位同核运输的确切边界；不调整门槛
救援、不隐藏负结果。原18协议/100卡/44问题、自然作者、\(B_u\)、人格与
行为解释保持全范围。本包结束不完成/reset主goal，保留可恢复checkpoint。
