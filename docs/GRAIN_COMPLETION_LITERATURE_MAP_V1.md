# GRAIN 终局规划：跨领域文献、引用链与方法对齐

2026-09-11；用于[全系统终局计划](GRAIN_COMPLETION_PLAN_V1.md)。
这是有目标的跨领域证据回顾，不是 PRISMA 系统综述，也不是“查遍所有论文”的声明。
本轮核验了下列原始来源/机构记录，重点精读可访问论文的相关定义、方法及引用段；
没有声称全文精读每一本书或每一篇文章。标记：F=全文相关段已读，A=摘要/作者研究页，
M=出版或机构元数据/目录。访问受限的内容不承担关键证明。

## 1. 阅读带来的六项设计修正

**第一，先固定解释对象，再造公式。** Theory Construction Methodology 将现象、
机制假说、形式模型及其再现能力分开。我们应要求一个机制同时解释若干已登记现象，
而不是把新恒等式本身当作理论完成。这直接对应终局计划 WP01–03。[R01]

**第二，相关结构不唯一指定潜在原因。** 相似协方差既可能来自共同潜因子，
也可能来自相互作用。把“聚成团”直接命名为心理因子会跳过识别问题。
因此 GRAIN 可以以配置、分布与条件响应为对象，唯一因子命名不是强制终点。[R02, R16, R17]

**第三，特质和状态不是在同一统计量上争夺名称。** 状态分布、条件响应和个人平均
需要不同的观测设计；群体间关系也不能直接代替同人过程。
这支持保留 where 选择与 when 变化，而不是把它们残差化掉。[R06, R07]

**第四，可读信息取决于读者，模型使用又是另一层。** 条件 probing 应比较
GRAIN 之外是否还有可用信息，以及路径之外 GRAIN 是否还有作用；不是单看总方差或 PCA。
原生使用需要干预关系，不能只由 probe 的相关系数推出。[R18, R21–25]

**第五，压缩和独立实现必须有信息及计算契约。** 先均值再拟合非线性，
不等于先逐 token 提取非线性再聚合。有限矩有表示容量边界；
值、导数、有限变化必须对应同一个程序。[R19, R20, R26, R29]

**第六，测量解释和用途是证据链，不是“像人格”就能成立。**
外部量表、观察者、行为任务和模型回答属于不同方法通道。
当前可完成基础方法与有限心理连接，不要求同时做完临床工具开发。[R03–05, R28]

## 2. 原始来源与具体用途

### A. 理论建构、心理测量、人与情境

| ID | 原始来源与访问范围 | 对本项目的使用及边界 |
|---|---|---|
| R01 | Borsboom et al. (2021), *Theory Construction Methodology: A Practical Framework for Building Theories in Psychology*. [作者机构全文](https://pure.uva.nl/ws/files/68321294/1745691620969647.pdf), F，五步骤及 mutualism 示例 | 用现象清单与竞争模型组织计划；不把论文方法视为保证发现真理论的算法 |
| R02 | van der Maas et al. (2006), *A Dynamical Model of General Intelligence: The Positive Manifold of Intelligence by Mutualism*. [机构记录/摘要](https://research.vu.nl/en/publications/a-dynamical-model-of-general-intelligence-the-positive-manifold-o/), A | 相关结构可由相互作用产生；借用生成式反例思路，不把智力模型移植成人格真相 |
| R03 | Borsboom, Mellenbergh & van Heerden (2004), *The Concept of Validity*. [机构记录](https://dare.uva.nl/id/45c23710-9842-4ec9-a7ac-eee24bce2a20), M；DOI 10.1037/0033-295X.111.4.1061 | 提醒“属性存在、测量为何随之改变”是另一类主张；这是效度哲学的一种立场，不代表全部共识 |
| R04 | AERA, APA & NCME (2014), *Standards for Educational and Psychological Testing*. [官方开放入口](https://www.testingstandards.net/open-access-files.html), F，开放版入口与效度框架 | 解释、参照群体、测量精度、用途分别有证据；不将临床部署门槛提前变成本阶段发现门槛 |
| R05 | Brennan (2001), *Generalizability Theory*. [出版社目录](https://link.springer.com/book/10.1007/978-1-4757-3456-0), M | G-study/D-study 组织作者、场景、机会、评分器的方差；本轮未读完整书，不直接搬用不可识别的全随机效应式 |
| R06 | Fleeson & Jayawickreme (2015), *Whole Trait Theory*. [论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4472377/), A/已访问摘要，部分全文访问遇验证码；并回查 Fleeson (2001) [原始记录](https://pubmed.ncbi.nlm.nih.gov/11414368/) | 区分状态分布的描述与背后的解释机制；稳定均值不是特质理论的唯一对象 |
| R07 | Molenaar (2004), *A Manifesto on Psychology as Idiographic Science: Bringing the Person Back Into Scientific Psychology, This Time Forever*. [机构记录](https://dare.uva.nl/id/9ad493a2-825b-4348-ac79-a7aede156153), M | 群体关系不能自动外推个体内机制；此处是方法约束，不要求同人每天重复相同话题 |
| R08 | Funder (1999), *Personality Judgment: A Realistic Approach to Person Perception*. [作者著作出版页](https://www.educate.elsevier.com/book/details/9780122699306), M，RAM 目录；1995 原文抓取失败 | 将线索相关性、可获得性、识别与利用作为分层问题；不能从本轮目录访问声称完成 RAM 全文审计 |

原来我们把“噪声是什么”视为唯一问题，以上文献提示更好的问题是：
哪些变异属于待测对象，哪些来自观测条件，哪些在当前设计中根本不可分？
这要求保留条件选择，而非假定公共部分全是噪声、个人偏离全是人格。

### B. 与真实文本、LLM 人格/心理推论直接相邻的工作

| ID | 原始来源与访问范围 | 对本项目的使用及边界 |
|---|---|---|
| R09 | Soni et al. (2022), *Human Language Modeling*. [ACL](https://aclanthology.org/2022.findings-acl.52/), A | HaRT 将作者历史纳入语言模型；比较“历史/记忆提供什么”，不是只比较一段均值 embedding |
| R10 | Gjurković et al. (2021), *PANDORA Talks: Personality and Demographics on Reddit*. [ACL](https://aclanthology.org/2021.socialnlp-1.12/), A；项目已有 Table 8 协议审查 | 保留标准预测与辅助标签路径；官方预计算特征协议不等于生成式 JSON 或全新文本复现 |
| R11 | Peters & Matz (2024), *Large Language Models Can Infer Psychological Dispositions of Social Media Users*. [PNAS Nexus](https://academic.oup.com/pnasnexus/article/3/6/pgae231/7692212), F，数据与推论方法 | 同用户大量状态文本、特定量表和汇总方式是条件；文章提供能力现象，不给出神经路径因果解释 |
| R12 | Yang & Oshio (2026), *Attachment towards generative AI influences self-AI agreement in personality reports*. [机构摘要](https://waseda.elsevierpure.com/en/publications/attachment-towards-generative-ai-influences-self-ai-agreement-in-/), A；DOI 10.1111/ajsp.70102 | 738 名日本 ChatGPT 用户中的 self–AI agreement 与关系因素；观测关联不能证明记忆读取机制或普遍测量准确性 |
| R13 | Tak et al. (2025), *Mechanistic Interpretability of Emotion Inference in Large Language Models*. [ACL](https://aclanthology.org/2025.findings-acl.679/), A | 以 appraisal 概念定位并干预情绪推论；是机制研究的直接先例，不是把文本状态都当稳定人格 |
| R14 | Anthropic (2025), *Persona Vectors: Monitoring and Controlling Character Traits in Language Models*. [作者研究页](https://www.anthropic.com/research/persona-vectors), A | 模型行为可受特征方向干预；赋予 AI persona 与推论人类特质可能相接，但不是互逆定理 |
| R15 | Hossain et al. (2025), *AutoPsyC: Automatic Recognition of Psychodynamic Conflicts from Semi-structured Interviews with Large Language Models*. [ACL](https://aclanthology.org/2025.clpsych-1.2/), A | 141 份 OPD 访谈、长对话处理与特定冲突识别提供辅助编码先例；不同于 TEG/TA，也不同于五分钟自由相谈 |

由这些先例提出的项目推论是：我们应使用同一材料、明确定义的查询、固定教师及
独立效标去比较；“LLM 能推测”只能作为需要解释的现象，不能作为所有输出的真值。
目前可主张的组合贡献候选是**群体参照的 GRAIN 配置与查询作用在同一材料地址上相接，
并将作用计算移植到明确依赖的小程序**。本轮不作首创或无先行研究的断言。

### C. 可识别性、集合表示、降维与可用信息

| ID | 原始来源与访问范围 | 对本项目的使用及边界 |
|---|---|---|
| R16 | Hyvärinen, Sasaki & Turner (2019), *Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning*. [PMLR 全文](https://proceedings.mlr.press/v89/hyvarinen19a/hyvarinen19a.pdf), F，§§3–5、Theorems 1–3 | 条件信息可帮助分离，但要求可逆混合、条件独立和足够变化；从同一文本算出的 topic 不自动满足辅助变量假设 |
| R17 | Locatello et al. (2019), *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*. [PMLR](https://proceedings.mlr.press/v97/locatello19a.html), A/理论结论核对 | 没有额外归纳偏置不能保证唯一解耦；支持报告等价类，不是说无监督结构没有用 |
| R18 | Fukumizu, Bach & Jordan (2009), *Kernel Dimension Reduction in Regression*. [论文](https://arxiv.org/abs/0908.1854), A | 目标是保留给定目标的条件信息，而不只是最大输入方差；当前局部核空间检验不是总体条件独立检验 |
| R19 | Zaheer et al. (2017), *Deep Sets*. [NeurIPS](https://papers.nips.cc/paper/6931-deep-sets), A | 区分逐元素变换与汇总，保留必要的集合结构；时间顺序不是自动置换不变 |
| R20 | Wagstaff et al. (2022), *Universal Approximation of Functions on Sets*. [JMLR](https://jmlr.org/papers/v23/21-0730.html), A | 聚合表示的容量有条件；最坏情形通用逼近界不等于当前 GRAIN 实际误差界 |
| R21 | Xu et al. (2020), *A Theory of Usable Information Under Computational Constraints*. [原始预印本](https://arxiv.org/abs/2002.10689), A | 固定读取函数族后才谈可用信息；简单读者读不到，不等于原始信息不存在 |
| R22 | Hewitt et al. (2021), *Conditional Probing: Measuring Usable Information Beyond a Baseline*. [ACL 全文](https://aclanthology.org/2021.emnlp-main.122.pdf), F，定义与附录D | 比较 G 与 G+路径、路径与路径+G 的增量；损失差在给定函数族下解释，不随意叫互信息 |
| R23 | Hewitt & Liang (2019), *Designing and Interpreting Probes with Control Tasks*. [ACL](https://aclanthology.org/D19-1275/), A | probe 自己可能学习任务；设置选择性对照，作者分组和置乱也需作用在正确单位 |

具体到现行式子，`ker Dφ` 的非零原生响应是**已给定压缩的局部不足见证**。
它不要求新发明一个人格因子；可先比较线性矩、非线性矩与同分数内部配置。
若增加表示后作用仍不可预测，就应记录本输入族的限制，而非继续无界加 K。

### D. 机制、函数抽取、层级解释与外部连接

| ID | 原始来源与访问范围 | 对本项目的使用及边界 |
|---|---|---|
| R24 | Geiger et al. (2025), *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*. [JMLR 全文](https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf), F，§2、Definition 25、复合关系 | 用干预及输出的对应定义机制抽象；允许近似，不把隐藏层坐标命名当因果解释 |
| R25 | Rubenstein et al. (2017), *Causal Consistency of Structural Equation Models*. [作者预印本](https://arxiv.org/abs/1707.00819), A | 宏观与微观模型应对相应干预给出一致预测；宏观聚合不是随意相加 |
| R26 | Czarnecki et al. (2017), *Sobolev Training for Neural Networks*. [NeurIPS](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html), A | 数值与导数共同约束函数；不能独立拼接不相容的分数头和梯度头 |
| R27 | Bareinboim & Pearl (2016), *Causal Inference and the Data-fusion Problem*. [PNAS/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4941504/), A，全文本轮遇验证码 | 跨来源连接需要明确差异与识别条件；不同人的标签总体不能替代同人匹配数据 |
| R28 | Campbell & Fiske (1959), *Convergent and Discriminant Validation by the Multitrait-Multimethod Matrix*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/13634291/), M | 组织多构念、多方法对照；不把同一 LLM 的自解释当独立外部效度 |
| R29 | Sundararajan, Taly & Yan (2017), *Axiomatic Attribution for Deep Networks*. [PMLR](https://proceedings.mlr.press/v70/sundararajan17a.html), A | 完整性与路径归因有标准先例；路径、基线与干预合法性仍需要定义，不由总和相等得到唯一原因 |

## 3. 实际追溯的引用链，而非只有平行书目

| 本轮沿着哪篇回查 | 已核验的引用关系 | 为计划补的拼图 |
|---|---|---|
| R01 → R02 | TCM 正文用 van der Maas 2006 mutualism 解释理论建构五步；已读示例段 | 不只拟合相关矩阵，要求机制生成现象，并比较不同生成机制 |
| R24 → R25 | JMLR §2 与 Definition 25 明确承接 Rubenstein 2017 exact transformations；已读原文对应段 | 微观/介观/宏观连接必须连同干预映射一起定义 |
| R22 → R21 | Conditional Probing 附录D讨论其定义与 Xu 2020 的关系；已读对应段 | “额外信息”须依赖声明的读者族，不能以一个模型失败宣称绝无信息 |
| R06 → Fleeson 2001 | Whole Trait Theory 的描述性状态分布线索回查原始密集状态研究记录 | 保留状态变化与个人分布；不要求人的表达恒定不变 |

另检索了 CAPS（Mischel & Shoda, 1995）与 RAM 的原始全文入口，抓取受限。
这两条作为后续理论阅读项保留，不伪报精读，也不作为本计划某条关键公式成立的唯一来源。

## 4. 公式层面应如何对齐，而非堆砌高级术语

| 项目现有对象 | 标准数学连接 | 必须保留的假设/差异 |
|---|---|---|
| `m_u=Ψ(A_u;reference,protocol)` | 测量模型、概化理论、条件分布 | 作者×场景×时间未交叉时不能拟合出不存在的可识别方差分量 |
| `φ(B)=(mean,cross-moments)` | 集合表示、目标相关降维 | 低维矩不保证对所有查询充分；不以输入解释方差替代目标作用 |
| `J=PJ+(I-P)J` | 正交投影、微分核空间 | 固定 token/GRAIN/模型位置的局部定义；移动 G 后要补 `Dφ` 项 |
| `s=e(H)^T r(H)` | 乘积微分与函数近似 | 证据与读取都可依赖 H；拟合值/导数需共享一个函数 |
| `ΔF=∫DF[γ']dt` | 路径积分、归因、因果抽象 | 数值闭合是必要核对，不自动证明路径自然、因果唯一或人类机制 |
| G+路径预测优于 G | 条件 probing / 受限预测风险差 | 不同函数族、样本泄漏或标签选模会改变解释 |
| 群内误差极限 | 条件期望、全方差分解 | `Var(估计均值)` 可以降，`E Var(Y|X)` 不随样本量自动消失 |
| trait/state/方法差异 | 层级条件模型、MTMM | 无时间配对不识别状态效度；自评/他评/行为需各自命名 |

这些连接表明项目不是“完全无前人理论”，也不是照抄某一个现有方法。
应将创新候选写在**接口、组合与经验机制**上，将已知数学定理准确引用。
没有理由为了显得新颖而把正交投影、因子旋转或链式法则重新命名。

## 5. 文献没有替我们补好的地方

文献能告诉我们如何提出可识别问题，不能保证当前4个 GRAIN 因子足够、
同一条人格词路径跨模型成立、或完全去掉编码器后仍能保留真人关联。
这些是终局计划的有限实验决策，不是可用引用直接宣判的结论。

本轮最重要的规划变化是：把“全新高级数学能否解救模型”改为
**用正确的数学对象，确定哪些计算必须保留、哪些可以抽取，以及抽取后还保留什么心理联系**。
未来若必须增加新方法，先写出它将区分的两个机制与可观察差异；
没有这两项，就不再以新增文献或新公式为由开启下一轮研究。
