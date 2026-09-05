# VM05：冻结几何通道的同证据比较

2026-09-06。本补充落实原计划中“正式profile仅在ready子集比较”，并把
当前实现的不同信息保留规则分开。不改发布scorer、不重开bridge标签。

## 对照范围

当前source主表的grain_intermediate是旧V7词法SVD24维的mean/std，共48列。
它不是GRAIN的全部4W理论，也不是正式地标分数。现有V8 external脚本还使用
规范地标距离的行中心化/行标准化。以下通道分列，不笼统称“GRAIN成绩”：

1. intermediate：固定旧词法作者向量，48维。
2. whitened：同一向量的冻结参照白化；保存作几何诊断，不另加择优参照。
3. canonical_distances：16个由参照自身距离指纹定序的地标距离，保留地标身份。
4. sorted_distances：正式V7 profile，逐作者对距离排序，未保留同列地标身份。
5. canonical_shape：既有V8 canonical scale-residual，去除每人的距离均值和尺度。
6. reference_radius：冻结参照径向距离，只作支持/信息保留诊断，不命名人格因子。

raw/cue_mask/placebo_mask均用同一冻结算子，无重新拟合参照或查看心理值。
raw_ready用于raw主比较；三臂共同ready用于配对掩码敏感性。不能让各模型在
各自最有利作者集上比较，也不能让masked后的拒绝分数冒充正式分数。
所有支持外数值只保存为诊断，保留原拒绝理由。共同支持选择本身可能关联
语言/人格分布，因此子集结果不能外推全部PANDORA。

## 可直接验算的信息边界

设冻结白化作者向量q在R^d，m个地标l_j，d_j=||q-l_j||。令
A_j=l_j-l_0，则：

\[
 A_j^\top(q-l_0)=\frac{d_0^2-d_j^2+\|A_j\|^2}{2}.
\]

全部带身份的地标距离可恢复q-l_0在span(A)上的投影，以及垂直部分的范数，
但一般不能恢复垂直方向。若span(A)秩为k<d，则固定投影/非零垂直范数后，
仍有d-k-1个角向自由度；垂直空间为一维时还有离散符号歧义。
最多m=16地标在d=48空间中k<=15，所以不能从距离维数直接假定原表示完整保留。
这是环境空间的数学边界；真实作者可能落在更低维流形上，尚不能据此说
实际丢失人格信息。需用同作者、同折、同头的外部读出进一步检验。

逐人排序另丢地标对应，行标准化另丢距离均值/尺度；这些是为不变性付出的
信息选择，不是“所有差异都应删除”。不从模型压缩定理推出心理解释。
代码以投影重构和不同合成点的同距离见证检查恒等式，不拟合心理机制。

## 实施顺序

- prepare_grain_vector_profile_comparators.py生成三臂私有特征、支持名单、
  raw旧结果精确重播、冻结地标的秩/不可辨识性诊断，所有心理值未读。
- 随后仅在来源作者上按原五折/inner规则比较：intermediate、canonical
  distances、sorted distances、canonical shape。全列，不挑外折最好通道。
- 正式同ready子集的TF-IDF/BGE/E5/Qwen读出也必须重新在该来源子集拟合；
  全来源已拟合OOF仅restrict是不同估计量，另列而不冒充同协议。
- 掩码在共同ready、同作者集配对；旧参照排除敏感性沿用既定标记。
- 内部向量增量先按相同来源拟合和复杂度规则评估，不把加入更多维数本身
  解释为LLM reasoning。不在bridge上选通道/参数。
- 除已冻结的正式profile比较外，规范距离与信息边界分析标记为机制分解，
  不是事后修改主榜或宣称完整GRAIN理论失败。

输出：results/grain_vector_mechanisms/vm05_profile_material_r1/，私有向量和
作者ID不得提交。来源控制/null/CUDA活跃计分代码不变。
