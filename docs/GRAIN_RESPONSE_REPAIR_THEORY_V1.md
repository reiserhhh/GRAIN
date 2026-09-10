# 响应教学如何同时改变状态读法与运动传输

2026-09-10。NEXT54：冻结四格机制分析，不是新的人格因子或新预测器。
[实测报告](../reports/grain_vector_mechanisms/RESPONSE_REPAIR_REPORT_V1.md)。

## 1. 从“教变化”到“改变读取位置”

前一轮保留同一函数给出的状态和方向：

    Xhat = Psi_B(X16) = X16 + Z(X16)B
    dhat = D Psi_B(X16)[D_f] = D_f + DZ(X16)[D_f]B.

GRAIN规定四个同材料方向D_f，其协方差Sigma_G规定源训练时优先保留哪些
变化。加入方向损失会改变共同参数B，因此同时改变Xhat和dhat。不能把
“训练使用导数”直接等同于“推论改善只通过导数支路发生”。

在固定source Qbar/rbar、权重和布局后，NEXT54的K组件对d是线性的：

    A_q(X,d) = ell_q(X)^T vec(d).

ell_q(X)不是新拟合的系数。它隐含实际RMSNorm微分、softmax竞争、中心化
value、位置变换和固定端点读头；随状态X变化。GRAIN方向先改变证据，
查询条件再决定怎样把证据变化读成一个作用值。

## 2. 两条途径与交互的精确分解

将归一化匹配值对照记V、联合教学记J，定义

    delta_ell = ell(XJ)-ell(XV), ell_bar=(ell(XJ)+ell(XV))/2
    delta_d = dJ-dV, d_bar=(dJ+dV)/2.

四格交叉给出精确恒等式，无需小扰动近似：

    delta_A = delta_ell^T d_bar + ell_bar^T delta_d
            = s + t
    interaction = delta_ell^T delta_d.

s是改变状态后对平均方向的读取变化；t是平均读法下方向的变化。交互分别
分一半到s/t，所以不能再把interaction加到s+t里。这个对称分配不是唯一
因果中介定义。交叉XJ,dV等仅是诊断：只有VV和JJ保持各自同函数导数的来源。

新8人两query两映射中，s贡献总端点误差降低的80.4–83.4%，t贡献16.6–19.6%。
交互RMS约为总改变RMS的1%。源4查询间对比反而变差，完整保留。结论是
**方向教学主要经由状态依赖读法产生了这一局部改善**，而不是发现某个
人格因素“八成静态、两成动态”。也未把ell的变化进一步专归于norm或某一头。

## 3. 为什么状态距离略变差、功能却更好

原生state误差用全维欧氏距离衡量，而端点只读其中一个状态依赖方向。
两者不是同一度量。源响应教学允许略微增加无关方向的重建误差，同时改善
对当前作用函数重要的配置。NEXT53全状态误差约增0.26%、全方向误差约减1.1%，
端点误差减15–19%；NEXT54现在定位出其中主要经由改变状态发生。

对e=VV-y和delta=s+t，恒等式为

    MSE_J-MSE_V = 2 E[e s] + 2 E[e t]
                + E[s^2] + E[t^2] + 2 E[s t].

新8的两个线性项均为负，胜过正的改变量能量项。修正与目标相关不是成功
定义，是否朝原误差反方向移动才决定误差降低。给两支路各分一半交叉项，
可精确记账，不能把“修正幅度百分比”和“误差减少百分比”混写。

## 4. 接完整推论函数时必须保留两条链

令P(X16)为原生blocks17–21，S为从pre22到最后固定logit L的完整后缀。
替代后的整个函数是

    Lhat_q(X16) = S_q(Psi(X16))
    D Lhat_q[D_f] = D S_q(Psi(X16))[D Psi(X16)[D_f]].

与原生动作之差可精确写成

    error_f = (D S(Xhat)-D S(Xnative))[d_native]
              + D S(Xhat)[dhat-d_native].

前项是状态处读取算子的变化，后项是运动运输误差；不能只证明一个K组件
近似得好就跳到整个L已经近似得好。NEXT55直接执行这个复合函数与其导数，
使本轮分解回到主线，而不是继续局部调alpha、basis或query。

## 5. 先行理论与本轮具体贡献

[Czarnecki et al., 2017, Sobolev Training](https://arxiv.org/abs/1706.04859)
将函数值与导数共同用于函数近似；
[Srinivas & Fleuret, 2018, Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html)
分析Jacobian匹配与输入噪声下蒸馏的联系。两篇原始页面本轮已核对。
这些工作支持“响应信息可以教学函数”，不直接证明本系统的人格效度。

我们本轮的具体做法是：GRAIN同材料方向定义响应度量、固定source map同参
生成状态与方向，再用真实模型组件的四格交换解释其功能改善。这是可复现
的组合与局部发现，不声称发明链式法则、对称交互分配或导数教学本身。

用户提出的两系统接点因而已经从“分数相加”发展为：

    同作者材料 -> GRAIN定义可比较变化 -> 学习保留变化的函数
              -> 查询条件下的作用读出 -> 另行连接量表/行为与辅助解释。

精神分析/交流分析继续沿用[同材料证据绑定接口](GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)：
表达与关系事件的候选解释可研究，但本轮数值分账没有新增诊断或心理构念。
