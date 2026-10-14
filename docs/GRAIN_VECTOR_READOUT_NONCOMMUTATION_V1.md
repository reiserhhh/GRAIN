# HM23：完整输出运输与参考计分的非交换关系

2026-10-09。算法对象是冻结模型的干预作用，不是真人心理分数。
数值源 `output_vector_transport_r1`；原point-ratio复核失败保留，
`output_vector_units_verification_r1`确认全部物理单位并量化条件数问题。

## Mathematical Definition

Let D be the source matrix of zero-sum candidate-logit displacements, and
let H denote the fixed weighted kernel smoothing operator. For each query,
the actual reference-dependent score action is

\[
T_r(d)=a^T\{\operatorname{softmax}(r+d)-\operatorname{softmax}(r)\}.
\]

The vector-first estimator is T_{r_*}(H_*D), whereas the scalar estimator is
H_*[T_{r_i}(D_i)]_{i\in S}. Their difference is a nonlinear,
reference-dependent commutator. If every T is the same linear covector,
T(d)=b^Td, the two are identical. In general they are not. Thus changing
the predicted object is a genuine change of approximation law, not proof
of improved information recovery. A common SPD system with30right-hand
sides is30independent output fits under an identity output metric.

The input observation retains six typed span means and the first three
position moments of the two property spans. It has18432coordinates but
is not the full token-state tensor. For a T-token property with T<=4,
the0–3moments have column rankT, so the property token states are
recoverable. Other spans retain only their means. A failure on the
property-only interventions does not by itself demonstrate lost property
information; it can arise from the learned action law or reference context.

## 实际结果与决策

固定同一kernel/unit/alpha1后，应用完整属性的向量运动相对RMSE为
.99177/.99290/.99234；三个评分RMS只有原生的约1.05–1.19%。
NA/低落/信任的评分RMSE全部比旧标量模型高，四个应用family均无收益。
关键NA—低落共同covariance原生+8.24505e-7，新模型-1.10432e-9：
接近零使绝对缺口变小，但没有恢复其正关系。多数共同作用留在残差中。

来源原生标量与旧target最大差2.33e-15，没有发现训练单位或地址错配。
完整来源D的范数2.18255，而预测来源范数仅.0371762；失败不是只出在
held材料。保留旧标量与新完整向量模型，不升级新模型，不扫alpha/rank。

## 数值边界

原alternate verifier严格point检查失败，不覆盖成PASS。
新伴随复核全部2172作用/354family/72指标/72joint/72分账/72矩阵地址；
原值差最大约1e-15，288比值中两项仍不能通过原point容差。
两者均在由实际端点差和中心矩运算推得的误差传播界内；这确认的是
极小分母的病态比值，而不是它们成为精确心理效应占比。

对于R=N/D，若|D|>e_D，误差界为

\[
|\Delta R|\le\frac{e_N+|N/D|e_D}{|D|-e_D}.
\]

不放宽原阈值，也不靠roundoff分账推断新机制。

## 下一科学问题

不是继续堆输出维度，而是检查局部decoder作用是否通过当前观察O
因子化：若F=phi\circ O，则kerO必须被dF消掉。HM24将用固定family
的小规模实际gradient/patch来区分可见属性作用律与隐藏context纤维。
此处的梯度是观察器局部响应，不是作者心理响应B_u，也不是已经脱离LLM。
