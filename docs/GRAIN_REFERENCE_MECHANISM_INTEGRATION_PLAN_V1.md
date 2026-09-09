# GRAIN 与人格推论机制的连接研究计划

日期：2026-09-09。用户授权：补全文献缺口、调查独立 GRAIN 与向量路径的连接，并立即执行首个有信息价值的实验。本文是有限研究计划，不是新版本发布或完整心理理论的完成声明。

## 1. 裁决与已有证据

两条线不必合并成同一种算法。GRAIN 是条件化、群体参照的测量方法；当前导航程序是查询相关的计算通道。真正接口是：**把路径程序当作版本化观察算子，再研究其读出的作者条件分布、相对位置与响应结构。** 这不是将旧 SUICA-19 分数拼到 embedding 后面，也不是把学生回归模型改名为 GRAIN。

实际查阅的基础：[GRAIN 理论稿，式 1–12](paper/GRAIN_THEORY_MANUSCRIPT_EN_V1.md)、[条件导航公式](GRAIN_CONDITIONAL_NAVIGATION_THEORY_V1.md)、[最新交付](../reports/grain_vector_mechanisms/CONDITIONAL_NAVIGATION_DELIVERY_REPORT.md)、[文献审计](GRAIN_LITERATURE_METHOD_ALIGNMENT_AUDIT_20260909.md)。

- CON 已显示：T/F 否认响应相关 .343→.585；状态时间响应 -.184→.789，但状态水平与固定函数的一致性 .433→.211。差分监督不能独自约束作者位置。
- [NAV round4](../reports/grain_vector_mechanisms/NAVIGATION_ROUND4_REPORT.md) 已做固定参照与路径分组；旧功能分组对原生响应的相对收益 3.4%，作者置换 p=.2784。不能直接假定“同类作者共享一条路径”。这里的分组是功能分数，不是旧 SUICA-19 因子。
- [VM02](../reports/grain_vector_mechanisms/VM02_MULTICOMMENT_DISCOVERY_REPORT.md) 中旧冻结 GRAIN 支持域拒绝新多评论视图。沿用理论与 API 不等于沿用旧参照系数；本轮建立明确命名的研究参照，不修改旧支持半径。
- B0 T/F AUC=.705、B4 相谈状态 r=.351 仍保留作已交付计算路线。首轮不重开外部标签比较，不因本轮拟合指标替换它们。

## 2. 两条理论如何接上

设作者材料为 \(\mathcal T_u=\{(T_{ut},c_{ut},h_{ut},o_{ut})\}\)。

\[
z_{ut}=R_v S_\lambda(T_{ut}),\qquad
e_{utq}=g_{q,v}(z_{ut},c_{ut},h_{ut}).
\]

\(R_v,S_\lambda\) 是 GRAIN 的观察与切片算子；\(g_q\) 是向量机制研究得到的查询读取函数。完整 LLM、固定函数 f、B0、B4、CON 是不同的通道版本，不能混作同一真值。

| GRAIN 的对象 | 路径研究带来的对象 | 接口与检验 |
|---|---|---|
| where：条件、机会、选择 | 归属、否认、时间条件改变证据使用 | 保存条件分布与条件响应，不把条件整块删除 |
| what：表达及局部配置 | 查询相关证据向量、P/Q 投影、中介作用 | 测量通道由固定函数定义；干预确定哪些成分影响原生输出 |
| who：相对参照群体的位置 | 原始材料水平、作者响应配置 | 固定参照；水平与响应分别报告；作者 ID 仅用于划分/配对 |
| when：历史与状态路径 | 次序、近期性、上下文更新 | 先识别程序的时间处理，再用真实时间序列研究人的变化 |

### 2.1 分布与函数的接口

\[
\mu_{u,c}=\mathbb E_{T\mid u,c}[\phi(R_vS_\lambda T)],\quad
m_{uq}(c)=\langle f_q,\mu_{u,c}\rangle_{\mathcal H}.
\]

当 \(f_q\in\mathcal H\) 且期望存在时，内积等于条件平均读数。它把“作者文本群”与“人格查询函数”接在一起；不是所有神经网络都自动属于给定 RKHS。有限版本可直接把多个查询值组成显式特征。若使用线性 embedding 均值，仅保留一阶信息；高阶核/联合分布可扩充，但有限样本不能宣称获得全部信息。

这里有两层抽样误差：作者数，以及每位作者的文本数。增加同一人的评论不等于增加独立作者。顺序无关分布只覆盖纹理的一部分；when 应另外保存转移/路径特征，不强塞入均值。

### 2.2 条件选择与条件响应的精确连接

令 \(\pi_u(c)\) 为观测条件频率，\(\pi_0\) 为固定参照机会分布，\(m_0(c)\) 为参照作者条件平均。则：

\[
\mathbb E_{\pi_u}m_u-\mathbb E_{\pi_0}m_0
=\mathbb E_{\pi_0}(m_u-m_0)
+\mathbb E_{\pi_u-\pi_0}m_0
+\mathbb E_{\pi_u-\pi_0}(m_u-m_0).
\]

三项是同条件响应偏移、选择分布差异、选择×响应交互。这是恒等分解，不是三个独立的心理原因。它保留了过去“去主题”可能误删的成分。没有机会集合时，\(\pi_u\) 只能称观测选择/曝光分布；不能将平台供给误称个人自由偏好。

首轮使用人工改写条件，频率由设计决定，所以**只研究条件响应，不估计自然选择行为**。八种改写也不是完整因子交叉设计，不把差异称为独立主效应或完整人类响应算子。

### 2.3 群体参照与个体响应

在固定的八条件研究设计 \(Q_0\) 上：

\[
b_q(c)=\mathbb E_{u\in P_0}s_{uq}(c),\quad
a_{uq}=\mathbb E_{Q_0}[s_{uq}(c)-b_q(c)],\quad
r_{uq}(c)=s_{uq}(c)-b_q(c)-a_{uq}.
\]

\(\mathbb E_{Q_0}r=0\)，故水平与响应形状可分开记录，原始值仍可完全重建。固定 \(P_0,Q_0\) 后，换评价批次不改变同一作者得分。保留原始文本分数作为另一项，不将八个改写均值当成自然人格水平。参照标准化本身不会提高 Pearson r，这一步解决解释与比较单位，不承诺精度奇迹。

## 3. 文献连接与边界

| 先行方法 | 实际借用 | 本项目额外要做的工作 |
|---|---|---|
| Mischel & Shoda (1995), [CAPS](https://doi.org/10.1037/0033-295X.102.2.246)；Fleeson (2001), [state distributions](https://pubmed.ncbi.nlm.nih.gov/11414368/) | 平均水平与条件变化可共同描述人 | 人工文本改写是测量函数的响应，不是同一个人的真实行为改变 |
| Song et al. (2009), [conditional RKHS embeddings](https://icml.cc/2009/papers/538.pdf), §2–4 | 条件分布与读取函数可通过算子连接 | 有限核、条件支持、可估计性；未在首轮实现完整条件嵌入估计器 |
| Szabó et al. (2015), [distribution regression](https://proceedings.mlr.press/v38/szabo15.pdf) | 作者是文本样本集；两阶段抽样；核均值+回归 | 原文一致性条件不能直接套到依赖的社交媒体序列；当前只取可计算接口 |
| Soni et al. (2022), [Human Language Modeling](https://aclanthology.org/2022.findings-acl.52/) | 作者历史状态可作为语言模型上下文，而不只是 bag mean | 与轻量、可重放读取器比较；不启动大规模作者 LM 训练 |
| Ilse et al. (2018), [attention MIL](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf) | 文本集合的加权证据汇总 | 权重不是人的选择概率，也不是因果路径；必须与均值与固定查询权重比较 |
| Rasmussen & Williams (2006), [GPML §2.2](https://gaussianprocess.org/gpml/chapters/RW2.pdf) | 核的 Schur 补可构造在指定锚点为零的修正空间 | 本轮采用确定性受限 RKHS 拟合，不将 GP 方差冒充心理置信区间 |

CAPS/Fleeson 是已有 GRAIN 心理学基础，不宣称本轮首次发现。条件分布、分布回归、受限核是标准数学；可能的贡献是**可干预的查询证据通道与条件/参照/历史测量链的可执行连接**，不是独创上述定理。完整人格机制仍需在人类关联、原生 LLM 使用和可替代计算三个层面分别累积证据。

## 4. 有限工作包与优先级

| ID | 动作、文献缺口对应 | 可验收输出 | 决策与预算 |
|---|---|---|---|
| INT-01 / P0 | 首轮连接：固定 GRAIN 参照分解；原始作者锚点约束条件残差；对应 LIT-ALIGN-03 | 数学模块、单元测试、source-author-held/自然材料比较、重建与批次不变性结果 | 本次执行；只读旧激活；最多 3 查询×3 方法，2 CPU 线程，不新增 LLM 调用 |
| INT-02 / P0 | 同材料 native/f/B0/B4/CON/受限函数比较；对应 LIT-ALIGN-01 | `target_location_comparison.csv`、水平与条件响应并列、错误分解 | INT-01 后；1 个工作块，先缓存。旧层位扫描不重跑 |
| INT-03 / P1 | 真正作者多文本分布：用现有 VM02 多评论缓存连接 choice、响应与交互；比较均值/条件分布，不直接借旧聚类 | 条件支持表、同作者跨视图 profile、选择分解表、与查询效应的距离关联 | 约半天 CPU；先检查同作者同材料交集，缺桥接数据只补固定小面板，不全量重编码 |
| INT-04 / P1 | 两个已见效机制的中介验证：T/F 归属/否认，状态时间归属；对应 LIT-ALIGN-02 | 两张计算图、带符号 donor patch 原生效应、选择性对照 | 1 个限定神经计算块；先测实际已有层位，结果能改变路线才扩展 |
| INT-05 / P2 | 全 Big5/MBTI 查询及小特质共享/独特算子；对应 LIT-ALIGN-04 | 基于相同材料/干预的查询效应矩阵、一次共享核与条件检索对照 | 接口成立后复用 M28/M31；不把跨查询不对齐的 token 直接拼轴 |
| INT-06 / P2 | 机制压缩与 GRAIN 心理连接；对应 LIT-ALIGN-05 | 冻结 raw-text 程序、零/四 block 区别、外部关联与材料支持说明 | 最后；PANDORA 正确来源、MEPS 仅外部，不按标签选权重；旧程序始终列出 |

先前审计的额外缺口进入明确卡片：目标 native/f 分离(INT-02/04)、读写非逆与条件选择(INT-04/05)、非线性探针可能自解任务(INT-02/04)、共享路径≠相关矩阵(INT-05)、压缩与心理保留分离(INT-06)。Fisher/SAE/全层扫描只在某张卡片无法辨认机制时采用，不同时堆叠。

## 5. INT-01 的执行预案

**材料锁定。** 复用 CON01 每查询 156 行：48 个控制世界、12 个源作者平均激活锚点、96 个条件端点。另 12 名作者×8 改写条件是已暴露自然材料探索面板，不称新 lockbox。使用现有 T/F、extraversion、dejected-state 三查询。检查作者集合不相交。人类量表和 MEPS 不读入。

**固定三个比较。** 同一线性核、同一源数据：原 value、原 secant、value + 在锚点为零的受限 secant 修正。不使用二次核、不扫权重。不从评价结果选择“最佳查询”。

设 \(A\) 取出训练内作者锚点，中心化核为 \(K\)：

\[
K_\perp=K-KA^\top(AKA^\top)^+AK.
\]

它是对锚点评价子空间正交投影后的核；\(AK_\perp=0\)（精确算术）。拟合：

\[
\hat s(x)=s_{\rm value}(x)+g(x),\qquad
\min_{g\in\mathcal H_\perp}\|R-g(X)\|_M^2+\|g\|_{\mathcal H_\perp}^2.
\]

其中 \(R\) 是 value 模型的残差，\(M=I+\lambda n/m D^\top D\) 沿用 CON 的源核 trace 权重，避免本轮因换权重而解释不清。修正不加自由截距，否则破坏锚定。训练来源按作者留一，去掉该作者全部端点及均值锚点。留出分数不进入核中心或伪逆。

保证的是**训练参照点不漂移**，不是陌生作者分数不漂移；也不是原锚点等于真实心理值。源锚点目前是平均激活而非所有原始文本，因此原始评价材料另列。

**读数。** 同时报告三查询的原始材料、八条件总水平、作者平均水平、四类响应；目标同时列 fixed f 与 native。报告 MSE、相关、响应幅度，作者级配对 bootstrap。GRAIN profile 报精确重建误差、源参照与批次不变性、作者水平/条件交互的能量，但不把它们命名成人格方差成分。

**停止与推进。** 完成三公式后即停止本轮，不以弱外部 r 触发长跑。若约束保水平且传递响应，进入多文本应用；若仅锚点成立而新作者权衡仍在，报告有限参照支持与差分近零谱方向，并推进 INT-02 的目标/中介诊断，不自动调核参数。无新 GPU 队列。

## 6. 产物与执行状态

- 计划及研究来源：本文件、上一轮方法审计与 42 篇来源账本。
- 代码：`grain_research/vector_mechanisms/reference_integration.py`。
- 执行：`scripts/run_grain_reference_integration.py`。
- 测试：`tests/test_grain_reference_integration.py`。
- 私有结果：`results/grain_vector_mechanisms/reference_integration_v1/INT-01/`，禁止上传作者/向量/标签。
- 公开汇总：`reports/grain_vector_mechanisms/REFERENCE_MECHANISM_INTEGRATION_REPORT.md`。
- 预案时状态：INT-01 准备执行；INT-02–06 尚未执行。完成后在报告中记录实际结果，不改写本预案以迎合输出。

### 执行后附记

INT-01 与 INT-02 的同材料缓存对照已执行，见[实际报告](../reports/grain_vector_mechanisms/REFERENCE_MECHANISM_INTEGRATION_REPORT.md)。INT-02 没有新增层位/非线性模型；对照已足以将下一步指向有效计算的保留与中介定位。INT-03–06 仍待执行。首轮没有覆盖 GRAIN 全部历史/动态机制，也没有改动旧发布的分数或参照系数。
