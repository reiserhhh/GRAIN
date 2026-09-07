# GRAIN数学文献地图 V2

检索日期：2026-09-08。服务两次研讨会的理论整合，不是穷尽性文献综述。
第一次研讨会的CoBRAS/AJIVE/Sobolev/Kakeya/Marton等来源保留在
`reports/grain_vector_mechanisms/MATH_SEMINAR_20260908.md`。
以下是本次新增或重点重读的一手来源；预印本和正式发表分开，不把研究类比
写成可直接套用的定理。访问源主要为arXiv、作者原文、PMLR、JMLR和出版社。

| ID | 来源与核对范围 | 数学用途 | 移植前提/边界 |
|---|---|---|---|
| S01 | Park, Choe, Veitch, ICML2024：[Linear Representation Hypothesis](https://arxiv.org/abs/2311.03658)，概念定义、causal inner product相关定理 | 测量协向量与操控向量的对偶；几何须声明内积 | 可分概念、线性表示与基等假设不自动适用于相关的人格构念 |
| S02 | Park, Nief, Choe, Veitch, ICML2026：[Information Geometry of Softmax](https://arxiv.org/abs/2602.15293)，本次详读v1的Section2、Theorem3、Discussion，另核对v2摘要与会议记录 | KL/Bregman几何、自然/对偶坐标、dual steering | 最终softmax表示；probe超平面上概念概率恒定是强假设 |
| S03 | Wang et al., 2026-08修订预印本：[FishBack](https://arxiv.org/html/2605.17231v2)，Theorems1–2、AppendixE | Fisher通过Jacobian拉回中间层，约束下最小二次代价 | 局部近似；奇异度量需range条件/正则；全算法每case很贵 |
| S04 | Otto, Zolman, Kutz, Brunton, JMLR2025：[Enforce, Discover, Promote Symmetry](https://jmlr.org/papers/v26/24-1315.html)，框架与摘要定理范围 | 用Lie导数统一发现/施加/放宽对称性 | 需明确群动作/向量场；删文本不自动可逆，语义不变性要另测 |
| S05 | Schmidt, Schneider, Bethge, NeurIPS2025：[Equivariance by Contrast](https://arxiv.org/abs/2510.21706)，摘要与实验范围 | 由有限群动作配对学习等变表征及表示 | 已声明的群动作，不是对自然改写的无条件可识别性保证 |
| S06 | Geiger et al., JMLR2025：[Causal Abstraction](https://jmlr.org/papers/v26/23-0058.html)，摘要/方法统一范围 | 给patching、steering、高低层模型一个干预兼容框架 | 需要定义变量、干预对应与近似标准；不是由静态相关自动得因果解释 |
| S07 | Geiger, Harding, Icard, 2025预印本：[How Causal Abstraction Underpins Computational Explanation](https://arxiv.org/abs/2508.11214)，摘要 | 将哲学上的计算解释连接泛化与干预 | 借解释框架，不赋予内部模型人类心理构念含义 |
| S08 | Khemakhem et al., AISTATS2020：[iVAE/Nonlinear ICA](https://proceedings.mlr.press/v108/khemakhem20a/khemakhem20a.pdf)，Section4 Theorems1–3 | 条件变化帮助潜变量识别；识别到何种等价类 | 条件独立指数族、injective mixing、参数变化满秩；nk+1是充分条件而非通用因子数上限 |
| S09 | Wang, Wu, Fan, COLT2024：[Nonlinear spiked covariance matrices](https://proceedings.mlr.press/v247/wang24b.html)，摘要 | 高维谱中的信号传播与目标对齐 | 指定渐近非线性随机特征模型，不是实测文本噪声检验公式 |
| S10 | Dandi et al., AISTATS2025：[Spectrum of Learned Features](https://proceedings.mlr.press/v258/dandi25a.html)，摘要/模型范围 | 训练引起的谱尾部与泛化变化 | 两层网络、大batch特定梯度步的严格结果，非所有LLM结论 |
| S11 | Benaych-Georges, Nadakuditi, 2011：[Finite low-rank perturbations](https://arxiv.org/abs/0910.2120)，摘要 | 有限秩扰动的谱/特征向量相变 | 阈值依赖基础随机矩阵和扰动模型；不能直接用p/n给相关评论贴噪声标签 |
| S12 | Williams, Beer, 2010：[Nonnegative Decomposition of Multivariate Information](https://arxiv.org/abs/1004.2515)，摘要 | 冗余、独有、协同信息及格结构 | PID有不同冗余定义；不把编辑交互直接叫MI协同 |
| S13 | Varley, npj Complexity2024，2025勘误更新：[Synergy-first backbone decomposition](https://www.nature.com/articles/s44260-024-00011-1)，方法说明 | 分层追踪高阶组合信息 | 概率估计和具体协同定义重要；24作者不适合直接高维MI估计 |
| S14 | Huang, Annals2019：[Sensitivity Conjecture proof](https://arxiv.org/abs/1907.00847)，摘要/命题领域 | Boolean超立方体的谱/组合思考，提示检查高阶交互 | Boolean最大敏感度，不是平均梯度，也不是任意实值响应低阶定理 |
| S15 | Candes, Tao, 2005：[Decoding by Linear Programming](https://arxiv.org/abs/math/0502327)，摘要/条件 | 稀疏结构与有限测量设计 | 特定矩阵条件；不能承诺任意32个half-subsets恢复66个交互 |
| S16 | Sriperumbudur et al., JMLR2010：[Hilbert Space Embeddings and Metrics on Probability Measures](https://jmlr.org/papers/v11/sriperumbudur10a.html)，摘要与characteristic核条件 | 作者分布而非单均值；条件与choice保留 | 特征核的总体单射不等于有限数据充分，更不等于心理构念识别 |
| S17 | Ramsauer et al., ICLR2021：[Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)，摘要 | 查询驱动关联记忆/检索的数学候选 | 相应更新与attention有联系，不表示整台LLM只是Hopfield检索 |
| S18 | von Oswald et al., ICML2023：[Transformers Learn In-Context by Gradient Descent](https://proceedings.mlr.press/v202/von-oswald23a.html)，构造与摘要范围 | forward里可实现特定回归更新，候选内部适应机制 | 特定线性attention/回归设置，不是自然人格推断已执行GD的证明 |
| S19 | Bai et al., NeurIPS2023：[Transformers as Statisticians](https://arxiv.org/abs/2306.04637)，摘要 | 上下文算法选择的存在性/统计计算 | 容量构造不等于本模型实际机制；梯度rank不能单独证明算法选择 |
| S20 | D'Acunto et al., 2026-08预印本：[Sheaf-Based Federated Representation Learning](https://arxiv.org/abs/2608.10016)，摘要 | 通过共享pilot样本连接异质局部空间 | 联邦表示学习设计；不强制全局空间，亦不由局部拟合误差宣称拓扑障碍 |
| S21 | Memoli, FoCM2011：[Gromov-Wasserstein Distances](https://doi.org/10.1007/s10208-011-9093-5)，论文定位/原文摘要 | 比较度量测度空间，容许坐标不同 | 度量同构不识别同一作者/构念；已有同人配对时不能让无约束匹配重排身份 |
| S22 | Etingof et al.：[Introduction to Representation Theory](https://math.mit.edu/~etingof/reprbook.pdf)，Schur引理定位 | 不可约表示、重复分量和交换子代数 | 复域不可约自同态的标量结论不能直接推广到实域/重复分量 |

## 不扩大成新长跑的处理

S02/S03指导观察几何，S04/S05指导允许的变换，S06/S07指导解释标准，
S12/S13/S14指导组合证据，S16连接GRAIN作者分布，S17–S19提供竞争计算模型。
S08–S11解释哪些额外条件有利于识别，不作为主线继续研究的入场门槛。
S15/S20/S21作为相应困难出现后的算法储备。全部不要求重新训练大模型。

上一研讨会的Kakeya与Marton只提供证明策略；本次的群/信息几何公式可直接
定义局部测量对象，但其经验适用性要由小面板确认。定理成立与模型满足条件
分别记录，这是让数学帮助发现而不是制造人为上限的基本规则。
