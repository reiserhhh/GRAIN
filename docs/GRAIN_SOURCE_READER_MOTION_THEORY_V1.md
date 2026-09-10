# 条件读取不是只选择旧方向，也可以学习方向怎样移动

2026-09-11。NEXT80 / 第69轮有限研究。承接
[NEXT79输出范围](GRAIN_CONTEXT_DIFFERENTIAL_READER_THEORY_V1.md)，对应
[实际结果](../reports/grain_vector_mechanisms/SOURCE_READER_MOTION_REPORT_V1.md)。
这里的 N/C 是 neuroticism / confirmation-seeking 两个固定模型查询；C不是尽责性。

## 1. 从位置样本到局部运动样本

令源材料的两查询末位平均为 c_i，查询差异读向量为 d_i=r_iN-r_iC。
仅用归一化权重时，dhat(c)=sum_i pi_i(c)d_i 永远处于源 d_i 的仿射包内。
NEXT79因此发现的是有限输出族的限制，不是条件信息没有用。

本轮在每个源参照 S 内定义：

```text
v_iS = (mean_S(c)-c_i)/||mean_S(c)-c_i||
epsilon_S = .01 h_S
j_iS = [d_i(c_i+epsilon_S v_iS)-d_i(c_i-epsilon_S v_iS)]/(2 epsilon_S)
J_iS = j_iS v_iS^T
dhat(c) = sum_i pi_i(c)[d_i + J_iS(c-c_i)]
```

采集 d_i 时，两查询BODY与其它位置不变，只有末位同时移动。实际执行
suffix17..end，在block22 post-attention最后位置求原评分 L 的梯度。
因此 j 是该**指定隐藏状态干预**下读向量的方向secant；不是完整Hessian，
也没有假设真实读向量只依赖 c。不同正文上的同一个 c 仍可能产生不同读法。

应用函数保持：

```text
rhat_N/C(c) = m_S +/- dhat(c)/2
Fhat_q(H) = rhat_q(c)^T phat_q(H_q)
```

pi、h、m与四种 phat 都未重调。源方向构造也遵循留一，应用不读取当前目标。

## 2. 可直接证明的性质

**输出扩张。** 令 U 是由 d_i-dbar 张成的旧子空间，新空间
Uplus=span(U,{j_iS})。任意本轮预测都属于 dbar+Uplus。
对任意目标 d 与候选 dhat，在正交投影 Pplus 下有：

```text
||d-dhat||^2
  = ||(I-Pplus)(d-dbar)||^2 + ||Pplus(d-dhat)||^2.
```

U包含于Uplus，所以最小可达向量误差不会因扩张而增加；**实际预测误差**
仍可能增加，因为列的组合和外推不一定正确。投影到某个证据 p 后，方向
会发生抵消，因此不能把该向量下界搬成标量作用下界。

**同一函数的导数。** 对任意输入方向 D：

```text
D dhat = sum_i [(D pi_i)(d_i+J_iS(c-c_i)) + pi_i J_iS D c]
D Fhat_q = rhat_q^T D phat_q + (D rhat_q)^T phat_q.
```

当前四个GRAIN正文方向具有 D c=0，于是 D pi=0，新增读头不额外产生正文
导数项。这使本轮评分与方向响应可以由同一个数值函数计算。若下一轮证实
读头还直接依赖BODY，则这一零项不再成立，需要重新声明函数并保留乘积法则。
尤其 r 本来是原生 L 的端点梯度，不能把其变化再机械加回 L 的一阶归因，
否则可能把曲率信息重复计算。

这些是有限维微分和正交分解，不是新发现的普适心理学定理。

## 3. 数据对理论增加了什么

- 源all4读头运动68.88%的能量在旧输出范围外；rank3扩为7。读法不只是从
  已有源方向中择一，它有可测的局部运动方向。
- 新8向量误差下界降低15.51%，实际向量误差只降低1.42%。表示能力、参数
  外推与目标可观测量是三个不同问题，不宜再用一个总体r代替它们。
- 在原生证据上，差异读头K+V误差较条件核减少5.30%，但仍略差于旧常数读头。
  接回数值组件也有小幅收益；反向调制没有恢复，源4分数排序仍负，均已保留。
- 典型新8方向位移约为59个教学步长。下一步应区分远于教学区间的非线性
  与未进入条件输入的正文/布局依赖，不继续磨核带宽。

这把用户的“同一句文字的两个侧面”具体化为：GRAIN给出文本变化方向 D，
查询路径给出如何选择证据以及从何方向读取证据；现在又有了**读取方向如何
随条件变化**的可测对象。三者可连接，但不是三个分数相加。

## 4. 文献位置和下一步

[Solak等，NIPS2002](https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html)
将函数与导数观测共同用于动态系统模型。本轮借鉴的是同时教学“位置”和
“局部变化”的思路；没有实现其GP后验，不借用其不确定性或全局兼容性保证。
本轮兼容性来自明确写出 Fhat 并对它微分。

[NEXT81](GRAIN_READER_MATERIAL_DEPENDENCE_PLAN_V1.md)固定两查询各自完整末位，
先看非末位材料是否仍使读头不同，再沿同一批GRAIN方向扰动正文。若存在，
下一算法输入应为 d(B,c,layout)，而不仅是 d(c)。这个发现可支持同材料的关系
观察解释接口；精神分析/交流分析构念仍通过独立观察编码建立语义，不让LLM
自己生成标签再验证自己。未新增临床材料使用或诊断输出。整体goal ACTIVE。
