# 查询先改变怎样选择，再改变沿什么方向读取

2026-09-11，NEXT73 / 第62轮。[实测报告](../reports/grain_vector_mechanisms/QUERY_SELECTION_ORIENTATION_REPORT_V1.md)。

## 1. 原来绑在同一Q上的两种作用

固定材料K/V/dK和读向量g，定义

    a(Q)=softmax(K Q/sqrt(h)+mask),
    v_i=g^T V_i,
    d_i(Q)=Q^T delta K_i/sqrt(h).
    R(Q_a,Q_d)=Cov_(a(Q_a))(v,d(Q_d)).

同一个Q既决定注意力的概率测度，也决定GRAIN方向在logit空间如何投影。
把Q全换掉会同时改变两者。NEXT73执行2x2交叉，不凭整体角度来猜哪项重要。

新8查询对比：只换方向仍r=+.905，只换选择反转为-.780，全换-.890，原来+.939。
对称有限分账给选择.863、方向.137；源4则.617/.383。这里是固定模型程序的
有限交换作用，而不是人类人格形成的贡献率。方向非零，不能删除。

## 2. 可直接使用的条件协方差operator

因为方向Q_d是线性的，可先编译

    C_f(Q_a)=Cov_(a(Q_a))(v,delta K_f)/sqrt(h),
    R(Q_a,Q_d)=sum_heads Q_d^T C_f(Q_a).

这给出可分离的接口：材料与选择先产生C，再由查询方向读取。C依赖当前材料，
不是一条跨所有作者固定不变的“人格向量”；head符号、token地址与GRAIN方向
保留。固定C时线性成立，但C随Q_a变化，因此整体仍包含非线性。
代码现已输出C和全部交叉作用，并通过独立的协方差公式和标量差分重建。

## 3. 选择变化与高阶联合矩的关系

对v/d固定的路径l_i(t)=l_i+t*b_i，直接对有限softmax求导：

    d/dt Cov_(a(t))(v,d)
      = E_a[(v-Ev)(d-Ed)(b-Eb)].

若Q自身沿u变化，则b=K u/sqrt(h)，同时d还显式变化，于是

    D_Q R[u] = Cov_a(v,delta K*u/sqrt(h))
             + E_a[(v-Ev)(d-Ed)(b-Eb)].

第一项是方向变化，第二项是选择测度变化的三阶联合中心矩。这是有限求和的
解析结果，不是本轮另测了真实数据的三阶数值，也不意味着要立即再开高阶矩
搜索。新增一个运行后synthetic差分测试核对了这两个导数项的和，未改主实验。
它说明下一selector为什么不能只拟合两个边际分布，而应保持同函数的
值与方向导数。标准配分函数导数/累积量依据见
[Wainwright & Jordan §3.5.1，Proposition2及后续证明](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf)。

## 4. 下一开发应直接针对选证据程序

此前状态重建是在逼近整个1536维中间空间，泛核实验则缺少查询选择结构。
现在可以改为源查询定向的逐token logit势函数ellhat_qh(H16)，再用softmax
产生权重，并让dellhat来自同一个函数的导数。目标从无差别状态相似转为
实际使用的选择关系；仍保留GRAIN定义的正文方向，不按人的量表标签挑结果。

NEXT74先只替代key选择输入，保留native V以隔离这一环节，不以局部成功
冒充全模型脱离LLM。零点偏移在softmax logits里是精确的gauge自由度；对其
处理不等于从作者文本删掉topic/choice。保留真正改变相对选择的部分。

这次有限分解已经给出下一开发处方，不需要再反复拆头、换归因定义、刷相关。
整体主线仍是作者文本->GRAIN可测差异->数值查询读法->外部人格/行为联系；
辅助精神分析/TA解释要另接原文与关系事件，不能由C的坐标直接命名。
