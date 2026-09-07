# Codex 独立立场：优先改变降维的目标函数

2026-09-08。在读取Claude第一轮正式答复之前写下；基于当前代码、D1–D3B及本轮
一次来源检索。这里是提案，不是新的真人实验结果。

## 第一优先：功能敏感方向，不是位移最大方差

我们实际拟合的是位移SVD，再拟合分数。这优先保留大的d，不保证保留会改变
分数的d。Active subspaces 以 E[grad f grad f^T] 选方向；控制领域CoBRAS把
状态/位移的分布与输出梯度同时使用。它非常贴近当前缺口。

设 C_d=E[dd^T]，C_g=E[gg^T]，g=grad_h s。CoBRAS式启发是平衡两者，不是只
最大化 tr(U^T C_d U)。一般坐标变换 d'=T d、g'=T^{-T}g 下，g'^T d'=g^T d，
C_d'C_g'=T(C_d C_g)T^{-1}。乘积的谱比未声明度量的欧氏cosine更具不变性。
实际使用应明确中心化/二阶矩、度量和协方差正则化；这不是声称现有cosine算错。

更重要：自然证据下g与d成对且依赖同一作者。直接把两个边际矩阵相乘会丢掉
部分联合关系。最小见证：d均匀取e1/e2，g=d时g^Td恒1；g取相反的基向量时恒0；
两个世界的C_d、C_g相同。应同时报告配对损失
L(P)=mean_i[g_i^T(d_i-Pd_i)]^2，而不是把边际谱当作已解释了全部响应。

新增机制对象：**可发生的位移与局部敏感度的配对分布**。该对象有条件结构，
但不一定能用现有GRAIN48的线性gain表示。D3B未获gain提升，不等于这里没有结构。

## 第二优先：同人多查询中的共享响应，而不是不同群体子空间的直接交集

AJIVE把同一批受试者上多个数据块拆成joint/individual，正好适合
M[作者,证据变换,查询]。先让每个查询对应一个作者x响应数据块，再比较共享
score subspace。2025 AJIVE理论指出信噪和子空间错位影响多块整合收益，提示
少量有信息查询优于不断增加弱查询。本项目只移植该设计，不把其噪声假设直接
当作LLM事实。跨任务不能要求共享方向同号；共享空间与有符号功能相关分开。

## 第三优先：让向量学生学习函数及响应，不只复制一个最终数字

Sobolev训练同时学习函数值和导数；我们可先采用有限证据变换的响应蒸馏：

L = mean(student(x,q)-teacher(x,q))^2
  + eta*mean((student(E_j x,q)-student(x,q))-Delta_teacher(i,j,q))^2.

这样不要求teacher hidden和student embedding在同一坐标系。值拟合与响应拟合
分别报告。查询使用分支/主干或向量值核的共享参数，再加低容量特质专属项。
DeepONet给出按输入函数和求值点分离的范式；这里仅借架构与逼近思想，有限查询
集不直接享有无限维算子逼近定理，且小样本优先ridge/kernel而非再训大型网络。

## 后继候选，不抢占第一包

1. **Mori-Zwanzig闭合**：投影后的遗漏量可表现为历史依赖，不必叫无用噪声。
   LLM层号/证据前缀不是默认平稳自治时间，需把层/输入当条件，先测有限历史项。
2. **路径signature**：相同端点但证据顺序不同可用二阶有向面积区别；它不是
   “固定标量梯度闭路积分非零”。2025粗糙路径全局逼近结果有具体可积性、路径
   空间和时间增广条件，不把语言默认为Brownian motion。
3. **Kakeya近期证明的策略**：从最难/最集中的情况出发，做多尺度分解与重标定，
   看粗细两层是否都逼近同一极端界，从而逼出隐藏结构。对GRAIN的转译是先找
   全局公式误差大但跨扰动一致的局部簇，再比较局部响应场，而非直接更多参数。
   文本高维向量不是R3中遍历所有方向的单位线段；借证明策略，不套结论。

## 初步排序与有限采集

同一24位source作者、两个既有查询、一个窄特质行为查询；各两视图。
先8人做资源测量，再完成剩余固定作者。统一由文本位置固定的证据变换，不用
每个查询自己挑不同的donor来伪造共享。每个case记录原分数/向量、自然变换、
局部梯度和配对有限patch。预算按case先算，避免“几百梯度”隐形翻倍。
不打开bridge/MEPS问卷；不以人格标签拟合这些路径。

## 已核对的一次文献

- Constantine, Dow, Wang (2013/2014), Active subspaces:
  https://arxiv.org/abs/1304.2070
- Otto, Padovan, Rowley (2022/2023), CoBRAS:
  https://arxiv.org/html/2207.14387v4 (Sections1–2, covariance transformation)
- Sensitivity-balanced extreme-event mechanisms (2026):
  https://arxiv.org/html/2606.05618v1
- Feng et al. (2017/2018), AJIVE: https://arxiv.org/abs/1704.02060
- Yang, Ma (2025), shared-subspace estimation: https://arxiv.org/abs/2501.09336
- Czarnecki et al. (2017), Sobolev Training: https://arxiv.org/abs/1706.04859
- Lu et al. (2019/2021), DeepONet: https://arxiv.org/abs/1910.03193
- Mori-Zwanzig deep learning (2022): https://arxiv.org/abs/2209.05544
- Ceylan, Proemel (2025), rough-path approximation: https://arxiv.org/abs/2512.16396
- Wang, Zahl (2025), Kakeya: https://arxiv.org/abs/2502.17655
- Guth (2026), proof exposition: https://arxiv.org/html/2604.03416v1 (Sections6,10–14)

无需把这个清单全量实现。Claude应反驳排序或公式，找一个更有判别力的下一实验。
