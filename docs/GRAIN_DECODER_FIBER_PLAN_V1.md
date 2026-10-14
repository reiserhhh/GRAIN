# HM24：观察纤维与 decoder 局部作用

2026-10-09。HM23完整输出运输失败后的最小新信息包。不是重新训练，
不是rank/alpha搜索；原冻结结果、原比值检查失败全部保留。

## 前置发现与问题

224文的属性span为1–4token，因此0–3阶矩在这些span内是单射。
HM23不能靠“完整属性被摘要丢掉”来解释。角色与背景span却只有均值：
这些均值相同的方向是否仍改变评分？旧拟合失败是输入观察遗漏、还是
观察已经保留但没有学到对应的局部作用律？这两种机制必须区分。

## 微分与几何判据

固定整篇query-context的block16状态为H，观察为OH，评分为F_q(H)。
若F_q在某邻域能够写成phi_q(OH)，则必有

\[
\ker O\subseteq\ker dF_q(H),\qquad
\nabla F_q=O^Tb_q.
\]

这只是必要条件；在邻域各处成立且相应纤维连通才可据此作局部函数分解。
单点梯度为零不证明全局独立或“不含人格”。分解

\[
g_q=P_{\mathrm{row}(O)}g_q+P_{\ker O}g_q=g_q^v+g_q^n.
\]

真实新信息是null局部作用，不是SVD恒等式。属性单射对应g^n近零；
neutral/actor仅均值时须实际检查。O只观察六个span，不等于完整序列。
未观察的system/question/其它token仍存在且不自动称噪声。

## 冻结设计

每个原family按pair_id取第一对，共16对×3查询=48梯度单位，
source8、validation4、application4保持原角色。选择不读输出。
这不是2172全地址的独立复制，也不是自然文本心理效度。
本地原Qwen、原整篇token、原block16、原alphabet均不变。

每单位执行一次无权重梯度的前向/反向：模型参数全部冻结，block16
output换成同值leaf，计算条件评分对整个状态的导数；核对原六span
与question状态和原candidate logits。仅保存六span梯度，不读私文。
根据visible/null梯度及独立seed的random-null方向构造三个方向。
总Frobenius半径为六span原状态范数的1%，不是每token1%。
每方向±各一次真实patch，共7forward/单位，336forward上限。
局部null方向是结果定义的最大梯度见证，不当预选语义因子或独立预测。
random-null作同单位幅度对照；零null空间严格返回零，不补造方向。

每个实际float32 patch记录实现后的delta，Odelta、visible/null导数
以及有限中心差分。数学null与float32实现误差分开，不能把近似同观测
冒称字节相等。只在null导数远大于可见漏入作用且有限差分确认时，
记为该观察族的局部不足见证。大patch非线性不由一阶导数代替。

另用原donor-reference完整属性delta计算g·delta，与既有原生full patch
对比，判断作用律的信息价值；不为了让它通过追加integral/Hessian网格。
每span同时核验b=(O^+)^Tg的pullback恒等式。

## 文献与借鉴边界

[Zahm等的向量函数梯度降维](https://arxiv.org/abs/1801.07922)提示使用
函数敏感度而不只使用输入方差；已读§1与§2至Prop2.6开头。
这里不继承其Gaussian/Poincare总体误差界，未证明文本满足该分布。
[AtP*](https://arxiv.org/abs/2403.00745)的§2–3.1提示梯度会有饱和与
抵消失败，因此同时做真实有限patch，而不把梯度当唯一因果证据。
OpenAI/math固定commit的07-global.tex:240–272借鉴观察/坐标/作用共同
运输的思路，不把其PDE唯一性证明套给语言模型。

## 交付与停止

逐单位checkpoint、原缓存hash、模型hash、梯度npz、actual patch json、
全48单位与144方向统计、属性作用对比、另公式pullback/中心差分复核、
最小单测与报告。每阶段不超过1800秒；保存再继续，不开无界loop。
若null有效，下一步才研究补充局部观察；若属性一阶很好但运输差，
优先局部covector来源运输；若一阶也弱，下一步需有限decoder曲率。
三种结果均改变科学决策，不能只用测试数或文件存在宣布理论完成。
