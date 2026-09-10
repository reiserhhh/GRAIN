# 内容与选择须共同生成评分：从分账到共同势函数

2026-09-11，NEXT71 / 第60轮。对应[实测报告](../reports/grain_vector_mechanisms/EVIDENCE_COMPONENT_REPORT_V1.md)。

## 1. 这一轮排除了什么解释，又保留了什么

同一材料上的GRAIN方向与查询读法继续用 `tau_qf=<grad F_q,D_f>` 接合。
NEXT70源外作用的3.60%/5.41%改善是真实的冻结程序比较；NEXT71发现它不是
一个新“人格选择器”的独占收益，而是局部读头改变、内容与选择项抵消，再加
少量log质量项。改了函数族后旧系数也会变化，不能只把全差额归给新增特征。

raw源跨段响应完全来自logZ；相对读法的跨段响应则允许C/S一起变化。确认
查询中的C虽然与目标同向，S/M会反向。这是同一个函数中多条计算作用的
具体例子，不是它们分别代表几种心理机制或可以独立切掉的噪声。

## 2. 带符号的误差账本

令e0为匹配局部基线误差，d_j为相对该基线的各项变化，d=sum_j d_j。则

    E1-E0 = 2<e0,d> + <d,d>
          = sum_j 2<e0,d_j> + sum_jk <d_j,d_k>.
    credit_j = -[2<e0,d_j> + <d_j,d>]/<e0,e0>.
    sum_j credit_j = 1-E1/E0.

这是对二次交叉项对称分配的记账约定，保留负值和超过100%的抵消。它不是
唯一的因果归因。源外raw/N的分账是+8.867-15.247+9.346+.637=+3.602个百分点；
若只报最后的3.602，便看不到前面约15个百分点的反向作用。

## 3. 为什么不能直接删除看似有害的导数项

对固定v、可微权重a_t(X)，相对内容评分是F=sum_t a_t v^T z_t。其微分分为

    dF = omega_C + omega_S
    omega_C = sum_t a_t v^T dz_t
    omega_S = sum_t (v^T z_t) da_t.

由乘积法则，两个一阶形式的外微分满足

    d omega_C = sum_t da_t wedge (v^T dz_t)
    d omega_S = -d omega_C.

它们通常各自有非零旋度，但总和为零。raw的a=w/T也适用。对常数lambda：

    d(omega_C + lambda*omega_S) = (1-lambda)*d omega_C.

所以任意给C/S不同权重，通常得不到同一个单值评分函数的梯度；局部路径积分
将依走法而变。M=d(sum b logZ)与完整local各自可积，C/S则一般需联合设计。
本轮新增两组synthetic方向导数测试，验证C/S的混合导数不对称互相抵消；
它检验的是数学构造，不假称每个真实拟合状态都具有同样旋度数值。

开发上的具体处方是：**调整生成权重和内容的标量程序，再求全部导数**，
而不是在看过效果后挑导数项组合成新“模型”。这保留发现空间，也保持计分与
作用同源；既有有限关联无需因此被否定。

## 4. 质量与关系读取可以来自同一对数配分函数

在固定source查询Qbar及当前证据K,V上，令某head的

    l_t = Qbar^T K_t/sqrt(h) + mask_t
    v_t = g^T V_t
    d_f,t = Qbar^T dK_f,t/sqrt(h)
    Psi(s,t) = log[(1/N_allowed) sum_allowed exp(l_i+s*v_i+t*d_f,i)].

在s=t=0处，标准指数族恒等式给出

    Psi_s = E_a[v],  Psi_t = E_a[d_f]
    Psi_st = E_a[v*d_f]-E_a[v]*E_a[d_f] = Cov_a(v,d_f)
    Psi_ss = Var_a(v),  Psi_tt = Var_a(d_f).

NEXT50的key作用正是各head `Psi_st` 之和。log质量的一阶响应与相对内容
读取的混合二阶响应因而可以统一记录，但它们不是同一个量：一阶均值大不
保证协方差大，协方差有符号且可在head间抵消。若方差非零，还可写为

    Psi_st = rho_a(v,d_f) * sd_a(v) * sd_a(d_f).

这将下一机制问题具体化为“缺少选择敏感度、缺少被读内容差异，还是二者
没有对齐”，而不是再次调一个不知用途的核。此处是有限向量势函数的推导；
NEXT72才对真实缓存编译并分账，不能现在就宣布其运输表现。

## 5. 理论参照与适用层次

- [Wainwright & Jordan, §3.5.1、Proposition2、式3.41](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf)
  给出对数配分函数导数与均值/协方差的关系。我们将它接到查询、GRAIN方向与
  同材料证据；不把这一标准定理称为原创人格定律。所读公开稿的命题编号为2。
- [Jain & Wallace (2019)](https://aclanthology.org/N19-1357/)展示注意力权重与
  模型重要性可不一致；[Wiegreffe & Pinter (2019)](https://aclanthology.org/D19-1002/)
  强调解释定义及整个模型、冻结权重等条件。这支持检验实际作用而非仅画权重。
- [Chrysostomou & Aletras (2021), §§3-4](https://aclanthology.org/2021.acl-long.40.pdf)
  将任务特异、非上下文化的词型权重接入注意力，在其分类实验中改进解释忠实性。
  这提示“上下文选择”与“被选信息对任务的意义”应区分，不要求把GRAIN换成词典。

当前H16/当前K,V仍是神经表示，source数据教的是固定模型推论而非真人量表。
共同势函数若可编译，推进的是可计算机制；人格意义仍接独立的量表/行为关联。
精神分析或交流分析只在同原文证据层提出可核对的关系/表达模式候选，不从
这个协方差直接推断患者的无意识动机。GRAIN的群体参照与原文地址仍保留。
