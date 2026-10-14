# GRAIN 全体系文献登记簿：广域补充 W01-W68

日期：2026-09-12。母目录：[持续登记簿](GRAIN_LITERATURE_REGISTER.md)。
配套：[综述](GRAIN_WHOLE_SYSTEM_LITERATURE_REVIEW_20260912.md)、
[机制矩阵](GRAIN_MECHANISM_PREDECESSOR_MATRIX_20260912.md)、
[研讨记录](GRAIN_WIDE_LITERATURE_SEMINAR_SYNTHESIS_20260912.md)。

## 登记规则

W编号是新增的稳定来源ID，不替换原R/P/M/N/T编号。首批为65条，执行时增补W66-W68；
不是所有来源都已全文精读，也不都是此前所有项目文件从未提到过的论文。
若旧计划已提过但母登记簿遗漏，此处补入并记录实际阅读。

F=读过原文指定相关节；I=原文索引节选；A=原始摘要；M=元数据；
“Claude”=该agent报告的实际阅读，未暗示Codex再次全文复核。
作者简写用于检索；投稿应从DOI导出完整书目。标注“短题名”的条目不能直接当完整APA题名。
年份区分预印本、online first与期刊卷期；访问日期统一为本轮日期。
出版商403、PMC验证码、无法解析PDF均保留为阅读深度限制，不以模型记忆补全文。

## A. 最接近的文本测量、构造与生成研究

| ID | 原始来源 | 阅读证据 | 对应环节 / 不应混同的对象 |
|---|---|---|---|
| W01 | Kjell, Kjell, Garcia & Sikström (2019; online 2018). **Semantic measures: Using natural language processing to measure, differentiate, and describe psychological constructs**. [DOI](https://doi.org/10.1037/met0000191); [PubMed](https://pubmed.ncbi.nlm.nih.gov/29963879/) | Codex A：原始摘要 | 开放语言成为心理测量指标；不是GRAIN首次提出 |
| W02 | Kjell, Giorgi & Schwartz (2023). **The Text-Package: An R-Package for Analyzing and Visualizing Human Language Using Natural Language Processing and Transformers**. [DOI](https://doi.org/10.1037/met0000542); [作者机构记录](https://researchconnect.stonybrook.edu/en/publications/the-text-package-an-r-package-for-analyzing-and-visualizing-human/) | Codex A/M；执行时补核作者教程及textTrain回归源码，未运行原R包 | embedding到测量/预测的实用基线；本地为Python适配 |
| W03 | Kjell, Sikström, Kjell & Schwartz (2022). **Natural language analyzed with AI-based transformers predict traditional subjective well-being measures approaching the theoretical upper limits in accuracy**. [原文](https://www.nature.com/articles/s41598-022-07520-w) | Claude F：PMC方法与上限讨论；Codex Crossref核题名 | 受提示的短词汇回答；上限论证有信度和任务条件 |
| W04 | Mesquiti, Cosme, Nook et al. (2026). **Language-based assessments can predict psychological and subjective well-being**. [原文](https://www.nature.com/articles/s44271-026-00400-3) | Claude A：出版社摘要 | 语义测量的后续检验；不能直接复制早期高相关为通用预期 |
| W05 | Wright et al. (2026). **Assessing personality using zero-shot generative AI scoring of brief open-ended text**. *Nature Human Behaviour*, 10, 541-555. [DOI](https://doi.org/10.1038/s41562-025-02389-x); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12974486/) | Codex I：原文Trait scoring、方法、讨论；出版社访问受限；Claude交叉核对效标范围 | 已有开放叙事、Big5、日常行为与心理健康关联；并非只有自评相关 |
| W06 | Hussain, Ma, Khandelwal, Oltmanns & Gupta (2026). **Language-based personality assessment from life narratives: a focus on model interpretability and efficiency**. [原文](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1760246/full) | Codex F：§§3.2-3.3、4.1-4.2 | 窗口编码、序列聚合、trait-specific attention；与N04相关但不是同一论文；注意力不是心理因果证明 |
| W07 | Gjurković (2025). **A Computational Framework for Interpretable Text-Based Personality Assessment from Social Media**. 博士论文预印本，SIMPA。 [原文](https://arxiv.org/abs/2510.02811) | Claude F：§§8-9.3；执行时Codex补读§9.1.3-9.3.2与Table9.4；原生代码及具体encoder revision未核实 | statement-to-item匹配、模型阈值差异及专家反馈；本地E5/IPIP适配不等于原生复现 |
| W08 | Russell-Lasalandra, Christensen & Golino (2026). **Generative psychometrics via AI-GENIE: Automatic item generation and validation with network-integrated evaluation**. *Behavior Research Methods*, 58, 217. [原文](https://link.springer.com/article/10.3758/s13428-026-03082-1) | Codex F：Methods步骤1-5、结构验证与限制 | 生成条目、embedding网络、真人结构检验；题目语义空间不是作者得分空间 |
| W09 | Hommel, Wollang, Kotova, Zacher & Schmukle (2022). **Transformer-Based Deep Neural Language Modeling for Construct-Specific Automatic Item Generation**. [DOI](https://doi.org/10.1007/s11336-021-09823-9) | Claude F：PMC方法 | 构念到条目生成已有先例 |
| W10 | Götz, Maertens, Loomba & van der Linden (2024; online 2023). **Let the algorithm speak: How to use neural networks for automatic item generation in psychological scale development**. [DOI](https://doi.org/10.1037/met0000540); [APA补充材料](https://supp.apa.org/psycarticles/supplemental/met0000540/met0000540_supp.html) | Claude A；Codex M：作者、卷期及online/issue区分 | 不将2023与2024视作两项独立研究 |
| W11 | Tan et al. (2026). **Large-Language-Models-as-a-Judge in Theory-Agnostic Adaptive Metric-Alignment for Prototypical Networks in Personality Recognition**. JAM，arXiv v1；页面列出期刊DOI。 [原文](https://arxiv.org/html/2607.08374v1); [arXiv](https://arxiv.org/abs/2607.08374) | Codex F：§III-A、III-B及式1-7；Claude F：协调/评价节；期刊最终版未另核 | 监督式跨量表表征与pseudo-facets；非无标签人格构念发现 |
| W12 | Ha (2026). **An interviewer-evaluator-judger LLM framework for text-based personality inference**. [原文](https://www.nature.com/articles/s41598-026-62333-5) | Codex A/M；受限全文 | 多阶段推论的近邻；不以摘要认定机制或效度已闭环 |
| W13 | Mislevy, Almond & Lukas (2003). **A Brief Introduction to Evidence-Centered Design**. ETS RR-03-16. [DOI](https://doi.org/10.1002/j.2333-8504.2003.tb01908.x); [ETS](https://www.ets.org/research/policy_research_reports/publications/report/2003/hsgs.html) | Codex A/M：官方记录 | 构念主张、证据模型与任务设计的已有框架 |
| W14 | Mislevy & Baxter (2004). **The Case for an Integrated Design Framework for Assessing Science Inquiry**. CSE Report 638. [原文](https://files.eric.ed.gov/fulltext/ED483405.pdf) | Codex F：pp.1-5，设计问题与模板 | ECD实际设计前身；此PDF不是“Layers, Structures, and Terminology”，已核封面纠正检索误配 |
| W15 | Kane (2013). **Validating the Interpretations and Uses of Test Scores**. [DOI](https://doi.org/10.1111/jedm.12000); [ETS](https://www.ets.org/research/policy_research_reports/publications/article/2013/jpdc.html) | 双方A：原始摘要/机构记录 | 分数解释与使用的推论链；不是因子相关即完成测量 |

## B. 情境、选择、状态和测量对象

| ID | 原始来源 | 阅读证据 | 对应环节 / 边界 |
|---|---|---|---|
| W16 | Tett & Burnett (2003). **A personality trait-based interactionist model of job performance**. [DOI](https://doi.org/10.1037/0021-9010.88.3.500) | Claude A：原始摘要 | trait activation；表达机会不是项目新造问题 |
| W17 | Meyer, Dalal & Hermida (2010). **A Review and Synthesis of Situational Strength in the Organizational Sciences**. [DOI](https://doi.org/10.1177/0149206309349309) | Claude F：定义、四种strength成分 | 综述兼概念框架；非任意语料的机会估计器 |
| W18 | Rauthmann, Sherman & Funder (2015). **Principles of Situation Research: Towards a Better Understanding of Psychological Situations**. [DOI](https://doi.org/10.1002/per.1994) | Claude A/M | 区分客观线索、解释和感知，避免循环定义 |
| W19 | Sherman, Rauthmann, Brown, Serfass & Jones (2015). **The Independent Effects of Personality and Situations on Real-Time Expressions of Behavior and Emotion**. [DOI](https://doi.org/10.1037/pspp0000036) | Claude A | 加法模型是有根据的比较项；不是所有文本交互必须为0的定律 |
| W20 | Bauditz et al. (2026). **Construal and Contact in Person-Situation Relations: Evidence from Controlled and Naturalistic Designs**. [DOI](https://doi.org/10.1177/08902070261449352); [PubMed](https://pubmed.ncbi.nlm.nih.gov/42615000/) | Codex A/I；Claude F：两研究方法与讨论 | 接触与解释分开；摘要N189，Claude核对实验分析N186；自然研究未严格完全分离二者 |
| W21 | Lee & Beck (2026). **Idiographic Momentary Profiles of Personality Facets**. [DOI](https://doi.org/10.1037/pspp0000568) | Claude F：PMC方法/结果；Codex Crossref为2026-01，未确认Claude回报的online 2025 | 个体内多种状态配置先例；不要求都组织为单个稳定轴 |
| W22 | Shaffer & Postlethwaite (2012). **A Matter of Context: A Meta-Analytic Investigation of the Relative Validity of Contextualized and Noncontextualized Personality Measures**. [原文记录](https://onlinelibrary.wiley.com/doi/10.1111/j.1744-6570.2012.01250.x) | 双方A；Codex发现页面有2013 corrigendum | 情境化提高效度的前例；不把其数值作为MCD验收线，精确合并数值须连勘误读 |
| W23 | Lievens, De Corte & Schollaert (2008). **A Closer Look at the Frame-of-Reference Effect in Personality Scale Scores and Validity**. [DOI](https://doi.org/10.1037/0021-9010.93.2.268) | Claude A | 框架与效标是否匹配；统计残差化不等于改写回答条件 |
| W24 | Schulze et al. (2021 issue; online 2020). **Hidden framings and hidden asymmetries in the measurement of personality: A combined lens-model and frame-of-reference perspective**. [DOI](https://doi.org/10.1111/jopy.12586) | Claude A；Codex Crossref核完整题名及日期 | 评分者与被评者的参照框架可能不同 |
| W25 | Horstmann & Ziegler (2020). **Assessing Personality States: What to Consider when Constructing Personality State Measures**. [原文](https://d-nb.info/1245363158/34); [DOI](https://doi.org/10.1002/per.2266) | Claude A；Codex I：定义、构造、采样段 | 状态也可测量；不能用trait重测目标淘汰有意义的状态变化 |
| W26 | De Boeck (2008). **Random Item IRT Models**. [DOI](https://doi.org/10.1007/s11336-008-9092-x) | Claude A | 从固定题转向条目总体；不直接涵盖自选择且未知暴露的自由文本 |
| W27 | Cho, Gilbert & Goodwin (2013). **Explanatory Multidimensional Multilevel Random Item Response Model: An Application to Simultaneous Investigation of Word and Person Contributions to Multidimensional Lexical Representations**. [DOI](https://doi.org/10.1007/s11336-013-9333-5) | Claude A；Codex Crossref核题名/2013卷期，平台后续online日期不当首次出版 | 人和词/条目贡献一起建模的前身 |
| W28 | Cole, Martin & Steiger (2005). **Empirical and Conceptual Problems With Longitudinal Trait-State Models: Introducing a Trait-State-Occasion Model**. [DOI](https://doi.org/10.1037/1082-989X.10.1.3) | Claude A | TSO；状态、特性与测量机会不由一项ICC决定 |
| W29 | Castro-Alvarez et al. (2022 issue; online 2021). **Mixed-Effects Trait-State-Occasion Model: Studying the Psychometric Properties and the Person-Situation Interactions of Psychological Dynamics**. [DOI](https://doi.org/10.1080/10705511.2021.1961587); [作者代码](https://github.com/secastroal/ME-TSO) | Claude A；Codex Crossref核完整题名/日期；代码未执行 | 随机效应状态模型；不默认它是连续时间模型 |
| W30 | Asparouhov, Hamaker & Muthén (2018). **Dynamic Structural Equation Models**. [原文](https://www.statmodel.com/download/DSEM.pdf); [DOI](https://doi.org/10.1080/10705511.2017.1406803) | Codex F：within/between、交叉分类、式6-11、附录平稳关系 | 个人动态参数/测量误差已有正式模型；token顺序不自动是心理时间 |
| W31 | Oravecz, Tuerlinckx & Vandekerckhove (2009). **A Hierarchical Ornstein-Uhlenbeck Model for Continuous Repeated Measurement Data**. [DOI](https://doi.org/10.1007/s11336-008-9106-8); [作者原文](https://cidlab.com/prints/oravecz2009hierarchical.pdf) | Codex A | 回归基点、波动与恢复的层级随机过程 |
| W32 | Oravecz, Tuerlinckx & Vandekerckhove (2011). **A Hierarchical Latent Stochastic Differential Equation Model for Affective Dynamics**. [DOI](https://doi.org/10.1037/a0024375); [作者原文](https://sites.socsci.uci.edu/~zoravecz/bayes/data/Articles/Oravecz2011AHLSD.pdf) | Codex A/I：原文简介 | 将潜在动态与观测误差分离；不是由文本自动发现情绪真值 |
| W33 | Carlier et al. (2022). **In Search of State and Trait Emotion Markers in Mobile-Sensed Language: Field Study**. [DOI](https://doi.org/10.2196/31724) | Claude F：PMC方法/结果 | 状态语言标记与trait标记可表现不同 |
| W34 | Ganesan et al. (2026). **From Word Sequences to Behavioral Sequences: Adapting Modeling and Evaluation Paradigms for Longitudinal NLP**. [ACL](https://aclanthology.org/2026.acl-long.2182/); [原文](https://aclanthology.org/2026.acl-long.2182.pdf) | Codex A/I；Claude F：评价拆分、within/between指标 | 按人/时间/两者切分的直接模板；“前瞻分割”不能把旧结果变成预注册 |
| W35 | Fisher et al. (2025 preprint). **Using Natural Language Processing to Track Negative Emotions in the Daily Lives of Adolescents**. [预印本DOI](https://doi.org/10.21203/rs.3.rs-6414400/v1); [原始预印本记录](https://pmc.ncbi.nlm.nih.gov/articles/PMC12047991/); [后续记录](https://pmc.ncbi.nlm.nih.gov/articles/PMC13487627/) | Codex A/I：预印本；Claude F：后续PMC方法；后续卷期未核 | 个体化与总体模型；不把两版本当独立复制 |
| W36 | Soni et al. (2026). **Who We Are, Where We Are: Mental Health at the Intersection of Person, Situation, and Large Language Models**. [原文](https://arxiv.org/abs/2603.05953) | Codex A/M；Claude F | 人、语言推断的情境、心理结果联合研究；小样本不是普遍定律 |
| W37 | Kern et al. (2016). **Gaining insights from social media language: Methodologies and challenges for psychology**. [DOI](https://doi.org/10.1037/met0000091) | Claude F：数据准备、分析方式、样本非代表性 | 实用方法综述；前往其原始实证而非只转引 |
| W38 | Eichstaedt et al. (2021). **Closed- and open-vocabulary approaches to text analysis: A review, quantitative comparison, and recommendations**. [DOI](https://doi.org/10.1037/met0000349) | Claude A；全文PDF提取失败 | 已有开放/闭合词汇直接量化比较 |
| W39 | Marengo, Montag & Settanni (2025). **Inferring Personality From Social Media Activity Using Large Language Models: Cross-Model Agreement, Temporal Stability, and Convergent Validity With Self-Reports**. [DOI](https://doi.org/10.1111/jopy.70019) | Claude F：PMC方法/结果；Codex Crossref核完整题名 | 读者之间一致性与对真人效标的一致性分开 |
| W40 | Fan et al. (2023). **How well can an AI chatbot infer personality? Examining psychometric properties of machine-inferred personality scores**. [DOI](https://doi.org/10.1037/apl0001082) | Claude A | 聊天测量信度与区分/效标效度可能分离 |
| W41 | Hickman et al. (2022). **Automated video interview personality assessments: Reliability, validity, and generalizability investigations**. [DOI](https://doi.org/10.1037/apl0000695) | Claude A；Codex Crossref核完整题名 | 训练目标是自评或访谈者评价，会改变效度对象 |
| W42 | McAdams (1995). **What Do We Know When We Know a Person?**. [DOI](https://doi.org/10.1111/j.1467-6494.1995.tb00500.x) | Codex A | 特性、个人关切与生活叙事不必压成一个分数 |

## C. 数学、表征、解释与跨层次方法

| ID | 原始来源 | 阅读证据 | 对应环节 / 边界 |
|---|---|---|---|
| W43 | Lock, Hoadley, Marron & Nobel (2013). **Joint and Individual Variation Explained (JIVE) for Integrated Analysis of Multiple Data Types**. [原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3671601/); [DOI](https://doi.org/10.1214/12-AOAS597) | Codex A/intro | individual是view-specific，不是个体人格 |
| W44 | Palzer et al. (2022). **sJIVE: Supervised Joint and Individual Variation Explained**. [原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9481062/); [DOI](https://doi.org/10.1016/j.csda.2022.107547) | Codex F：§§2.1-2.6、式1、定理1 | 共享/特有空间与目标预测联合估计；不按定义赋予心理所有权 |
| W45 | Roberts et al. (2014). **Structural Topic Models for Open-Ended Survey Responses**. [DOI](https://doi.org/10.1111/ajps.12103); [作者原文](https://bstewart.scholar.princeton.edu/document/38) | Codex F：pp.3-5，prevalence/content、协变量选择 | 说什么与怎样说的条件模型前身；不自动是人格模型 |
| W46 | Blei & Lafferty (2007). **A Correlated Topic Model of Science**. [原文](https://arxiv.org/abs/0708.3601) | Codex A | 主题比例可相关；非独立人格维度声明 |
| W47 | Blei, Griffiths & Jordan (2010). **The Nested Chinese Restaurant Process and Bayesian Nonparametric Inference of Topic Hierarchies**. [原文](https://arxiv.org/abs/0710.0845) | Codex A/M | 层级主题与路径结构；不与2003早期会议版混同 |
| W48 | Hearst (1997). **TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages**. [ACL](https://aclanthology.org/J97-1003/) | Codex A/I | 共同切分的前身之一；切分不自动给出心理条目 |
| W49 | Golino & Epskamp (2017). **Exploratory graph analysis: A new approach for estimating the number of dimensions in psychological research**. [原文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0174035) | Codex A | 网络维数估计；数据矩阵行列含义仍须明确 |
| W50 | Ghorbani, Wexler, Zou & Kim (2019). **Towards Automatic Concept-based Explanations**. ACE. [原文](https://proceedings.neurips.cc/paper/2019/file/77d2afcb31f6493e350fca61764efb9a-Paper.pdf) | Codex F：§§2-4 | 多尺度切分、聚类、重要性已有数学近邻；原应用是图像 |
| W51 | Koh et al. (2020). **Concept Bottleneck Models**. [PMLR](https://proceedings.mlr.press/v119/koh20a.html) | Codex I：原文引言与方法框架 | 可干预中间概念；并非人格效度自动成立 |
| W52 | Mahinpei, Clark, Lage, Doshi-Velez & Pan (2021). **Promises and Pitfalls of Black-Box Concept Learning Models**. ICML workshop预印本. [原文](https://arxiv.org/abs/2106.13314) | Codex I：原文方法/讨论节选 | 命名概念数值可夹带额外信息；不可只看下游准确率 |
| W53 | Sun, Oikarinen, Ustun & Weng (2025). **Concept Bottleneck Large Language Models**. ICLR. [原文](https://proceedings.iclr.cc/paper_files/paper/2025/file/de4ce91dfe56b919ee1c228d6a78f866-Paper-Conference.pdf) | Codex A/I：引言、作者项目；方法细读未完成 | 分类、生成、概念控制；2024早期三作者版不是此四作者正式版 |
| W54 | Xie, Raghunathan, Liang & Ma (2022). **An Explanation of In-context Learning as Implicit Bayesian Inference**. ICLR. [原文](https://arxiv.org/abs/2111.02080) | Codex A | 特定生成条件的ICL解释，不是所有人格推论的证明 |
| W55 | von Oswald et al. (2023). **Transformers Learn In-Context by Gradient Descent**. [PMLR](https://proceedings.mlr.press/v202/von-oswald23a.html) | Codex A/I：线性attention构造简介 | 特定回归/架构的数学对应；不能泛化成所有LLM均显式执行该算法 |
| W56 | Shen, Mishra & Khashabi (2024). **Position: Do Pretrained Transformers Learn In-Context by Gradient Descent?**. [PMLR](https://proceedings.mlr.press/v235/shen24d.html) | Codex A/M | 对实际预训练模型外推的讨论；不与W55的限定定理混为反驳 |
| W57 | Chorin, Hald & Kupferman (2002). **Optimal Prediction with Memory**. [DOI](https://doi.org/10.1016/S0167-2789(02)00446-3); [作者原文](https://math.berkeley.edu/~chorin/CHK02.pdf) | Codex F：§2、投影与广义Langevin分解 | 未解析变量可进入记忆核与残余力；统计恒等式非心理动力学验证 |
| W58 | Shalizi & Crutchfield (2001). **Computational Mechanics: Pattern and Prediction, Structure and Simplicity**. [作者页](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.htm); [原文](https://arxiv.org/abs/cond-mat/9907176) | Codex A | 用未来分布定义预测状态；causal state不等于心理因果构念 |
| W59 | Singer & Wu (2012). **Vector Diffusion Maps and the Connection Laplacian**. [DOI](https://doi.org/10.1002/cpa.21395); [原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3886882/) | Codex I：§3局部坐标/正交运输 | 局部运动坐标可以对齐；不是任意表征都同一几何 |
| W60 | Chaloner & Verdinelli (1995). **Bayesian Experimental Design: A Review**. [原文](https://people.eecs.berkeley.edu/~jordan/courses/260-spring09/readings/chaloner-verdinelli.pdf) | Codex A/M | 效用驱动提问/设计已有理论；综述不能充当新发现 |
| W61 | Ireland, Slatcher, Eastwick, Scissors, Finkel & Pennebaker (2011). **Language Style Matching Predicts Relationship Initiation and Stability**. [DOI](https://doi.org/10.1177/0956797610392928) | Codex A | 互动/关系层已有先例；不能由距离相近直接推出人际相性 |
| W62 | Giorgi et al. (2018). **The Remarkable Benefit of User-Level Aggregation for Lexical-based Population-Level Predictions**. [原文](https://arxiv.org/abs/1808.09600) | Codex A | 作者到群体的聚合次序已有实证；群体预测非个体效度 |
| W63 | Caliskan, Bryson & Narayanan (2017). **Semantics derived automatically from language corpora contain human-like biases**. [DOI](https://doi.org/10.1126/science.aal4230); [原文](https://arxiv.org/abs/1608.07187) | Codex A | 向量化不自动消除文化/群体偏差 |
| W64 | Zhang et al. (2026). **Automated item generation for personality assessment: development and validation of large-language-model-derived HEXACO situational judgment tests**. [DOI](https://doi.org/10.1016/j.jrp.2025.104680); [出版社](https://www.sciencedirect.com/science/article/abs/pii/S0092656625001126) | Codex A/intro；Crossref出版2026-02，未因DOI含2025就确定online年 | 情境化条目生成及真人验证的直接前例 |
| W65 | Debelak, Koch, Aßenmacher & Stachl (2025). **From Embeddings to Explainability: A Tutorial on Large-Language-Model-Based Text Analysis for Behavioral Scientists**. [原文](https://journals.sagepub.com/doi/pdf/10.1177/25152459251351285) | Codex F：pp.1-3背景；后续教程仅定位 | 行为科学embedding/解释实践入口；原文载明2025-10 OSF链接勘误 |

## D. 可复用实现，不等于本地已运行

执行更新见[方法卡](GRAIN_PREDECESSOR_METHOD_CARDS_V1.md)。sJIVE已运行原作者固定版本；
2026-09-13原生text 1.9和MEM作者函数也已运行，SIMPA仍为明确标示差异的本地适配。
续批实测见[已知步骤报告](../reports/grain_vector_mechanisms/KNOWN_GAP_COMPLETION_REPORT_V1.md)。
其余行仍保留初次核查深度。

| 来源 | 原始实现 | 本轮检查程度 | 采用时还需确认 |
|---|---|---|---|
| W02 / 旧T01 | [OscarKjell/text](https://github.com/OscarKjell/text) | 已隔离安装text 1.9；原生作者40人示例与PANDORA单fold均执行，DESCRIPTION/sessionInfo存档 | 真实比较使用本地E5/624人材料，不是原论文数据；PCA/惩罚网格/选择准则与Python不同 |
| W44 | [lockEF/r.jive](https://github.com/lockEF/r.jive) | 执行时固定commit 850c72cb18aaf2b6cad1e911d002541413973a48，GPL-3，原源码未修改；已知模型及真实两视图均运行 | 固定rank/eta的本地适配，不是原论文数据复现 |
| W50 | [amiratag/ACE](https://github.com/amiratag/ACE) | README/文件目录，MIT标识；未执行 | 图像接口须改造，不应把rare-author证据机械删掉 |
| W53 | [CB-LLMs](https://github.com/Trustworthy-ML-Lab/CB-LLMs) | 作者仓库及项目页 | 当前license、权重和语料权限未核；无需立刻训练 |
| W29 | [ME-TSO](https://github.com/secastroal/ME-TSO) | Claude定位；本轮未源码审计 | 观测间隔与模型假设，不当作ctsem同义词 |
| W30 / 旧M7 | [ctsem论文与软件](https://www.jstatsoft.org/article/view/v077i05) | 旧登记复用 | 连续时间、测量模型、抽样机制须各自声明 |
| W08/W49 | [EGAnet](https://r-ega.net/) | 原论文给出包和算法；本轮未安装 | semantic matrix与person response matrix用法不同 |
| W07 | [SIMPA博士论文](https://arxiv.org/abs/2510.02811) | 相关章节方法可读 | 未找到可核实的代码版本，不能承诺直接clone即跑 |
| W34 | [ACL正式条目](https://aclanthology.org/2026.acl-long.2182/) | 协议可按文实现 | 本轮未定位确定的作者代码；无权复用其私有日记 |
| W11 | [JAM作者项目](https://research.jingjietan.com/JAM) | 论文给出入口，本轮未核实现 | 数据/标签链接、license、最终版本后续再审 |

## E. 执行时新增的方法来源

| ID | 原始来源 | 阅读及执行范围 | 用途 |
|---|---|---|---|
| W66 | Markowitz, D. M. (2021). **The Meaning Extraction Method: An Approach to Evaluate Content Patterns From Large-Scale Language Data**. [原文](https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2021.588823/full) | F：Method、Preprocessing、PCA、Discussion/Validity；非全文全部引文追溯 | MEM实现细节、分段、局限；不是Boyd/Schwartz署名论文 |
| W67 | Bates, Mächler, Bolker & Walker (2015). **Fitting Linear Mixed-Effects Models Using lme4**. [JSS](https://www.jstatsoft.org/article/view/v067i01) | A及本地lme4实际调用；非全文逐页阅读 | 标准随机截距/斜率参照，版本1.1-37 |
| W68 | Wang et al. **E5-large-v2 official model card**. [作者模型卡](https://huggingface.co/intfloat/e5-large-v2)；基础论文[2212.03533](https://arxiv.org/abs/2212.03533) | 模型卡Usage及FAQ；此条不是新论文计数 | query前缀、pooling、0.7-1.0 cosine尺度；阈值不能跨encoder照搬 |
| W69 | Gjurkovic, Vukojevic & Snajder (2022). **SIMPA: Statement-to-Item Matching Personality Assessment from text**. [DOI](https://doi.org/10.1016/j.future.2021.12.014)；[作者实验室](https://takelab.fer.hr/publications/simpa/) | 作者摘要、出版社元数据；具体检测/计分方法此次由同作者W07第9章核对，不冒称期刊全文逐页读完 | FGCS 130:114-127，online 2021-12；不能把2025博士论文当作SIMPA首次提出年份 |
| W70 | Revelle. **psych: fa / factor.scores**. [fa](https://personality-project.org/r/psych/help/fa.html)；[计分](https://personality-project.org/r/psych/help/factor.scores.html) | 原作者帮助页及本地2.5.6函数；MINRES/promax/回归计分已执行 | 真正共同因子、特有方差、相关因子及固定样本外计分；标准软件，不计新研究论文 |
| W71 | Sentence Transformers. **paraphrase-distilroberta-base-v1**. [模型卡](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1) | 卡片、pooling/max-length配置、实际权重调用；revision `48bffbbd27bf028ecdd0cd55abb51236ec12ef1b`，Apache-2.0 | SIMPA所述RoBERTa释义家族的参照；不能确认它就是原作者checkpoint |
| W72 | Wen et al. (2024). **Affective-NLI: Towards Accurate and Interpretable Personality Recognition in Conversation**. [方法/实验](https://arxiv.org/html/2404.02589v1)；[作者代码](https://github.com/preke/Affective-NLI) | F：III公式1-7、IV-A数据；仓库README/目录，未运行原训练代码；GitHub API未给出license | 对话及情绪信息与正/反人格描述配对，再监督微调；电视剧角色数据不等于真人自报人格效度；不是零训练通用NLI |
| W73 | Sentence Transformers. **nli-deberta-v3-base**. [模型卡](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | 卡片、config的label方向、实际40句对调用；revision `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`，Apache-2.0 | SNLI/MNLI训练的三类蕴含判断。继承通用工具处理极性/归属，不称GRAIN原创或Affective-NLI复现 |
| W74 | Stan Development Team. **Simplex distributions: Dirichlet**. [官方参考](https://mc-stan.org/docs/functions-reference/simplex_distributions.html) | Dirichlet密度、simplex支持域与参数要求；作为标准工具来源，不计新研究论文 | 三类同/反/无证据误差的本地敏感性模型；Dirichlet(1+n)不是GRAIN新定理，也不验证模型标注等于真人金标准 |
| W75 | Boyd. **Meaning Extraction Helper v2 / MEM PCA template**. [源码](https://github.com/ryanboyd/MEHv2)；[作者R模板](https://www.ryanboyd.io/software/meh/Supplemental/PCA-Template-for-MEM.R) | Claude定位、Codex源代码核验；MEHv2 revision `3b306a74490b5f99c171ac683e6cb8fe8f3ea3dc`，12个选定源文件/资源hash存档；R原函数实际运行，C#应用未运行 | 同输入PCA/旋转子空间误差约1e-14。英语可选LemmaSharp路径、词表和筛选与本地仍不同；仓库MIT不自动覆盖附带语言资源 |
| W76 | **Language-Based Assessment Models (L-BAM) library**, text软件资源. [教程](https://r-text.org/articles/LBAM.html)；[源库](https://github.com/OscarKjell/text)；[隐性动机模型](https://github.com/AugustNilsson/Implicit-motive-models)；[open_models](https://github.com/theharmonylab/open_models) | Claude定位仓库文件，Codex从已安装text 1.9的`extdata/The_L-BAM_Library.csv`解析61条非空记录；元数据及hash存档，模型未执行 | 不需要仅依赖在线表格；逐模型核encoder/layer/pooling和许可。目录的open字段不是精确license，RoBERTa头不能直接接E5向量；此条不是新增论文计数 |

BLT公开资源补查：[OSF bm9gn](https://osf.io/bm9gn/)的API和README已核对。
README说明发布的是混淆后的因子分数、预测模型及验证notebook，不是原始用户词矩阵
或完整因子发现材料。本轮未加载其用户级分数、未运行其notebook。
因此用本地标准FA做适配，而不以找不到原始数据为由停工，也不冒称原文精确复现。

2026-09-13构造规则续批软件来源：**W77，SciPy beta-binomial**
([官方定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html))。
已核PMF及参数条件，实际用于来源作者的检测次数MLE与留出覆盖预测，不作为新论文计数。
目标设计理论沿用W60；本轮复核原文书目和可检索模型未知讨论，未升级为全文已读。
公式/实际结果见[构造规则报告](../reports/grain_measurement_foundation/DESIGN_RULE_REPORT.md)。

2026-09-13事件证据续批：**W78，McCoy, Pavlick & Linzen (2019),
Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference**。
[ACL原文](https://aclanthology.org/P19-1334/)，[作者代码/数据](https://github.com/tommccoy1/hans)。
本轮读官方摘要和PDF抽取的heuristic定义/实验说明，未完整逐页复读全部附录。
采用受控改变语法而观察方向/归属错误的方法，不复用其句库、不称HANS精确复现。
本批实际结果见[EVENT报告](../reports/grain_measurement_foundation/EVENT_EVIDENCE_REPORT.md)。
W13 ECD官方概述也重新核对；提取、评价与累积的分工沿用现有理论。

2026-09-13 NEXT91数学来源：**W79，Pitman，STAT205 Lecture 10: Conditional
Expectation**，[Berkeley原讲义](https://www.stat.berkeley.edu/users/pitman/s205f02/lecture15.pdf)。
已读§10.1离散定义及§10.2.3 Hilbert投影论证，并核§10.3投影/塔性质。
同批参考[MIT正交投影课程](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/projections-onto-subspaces/)。
用于精确GRAIN分组下的经验条件期望与嵌套正交分解，是标准理论来源，不计新论文或新定理。
本地新证据见[NEXT91报告](../reports/grain_vector_mechanisms/GRAIN_CELL_RESIDUAL_REPORT_V1.md)。

2026-09-13 C14语义函数学习：**W80，Hinton, Vinyals & Dean (2015),
Distilling the Knowledge in a Neural Network**，[原文](https://arxiv.org/abs/1503.02531)。
本轮实际读PDF的§1、§2和§2.1，包括soft-target转移、温度softmax、居中logit高温极限。
本地采用教师连续函数的平方误差回归思想，不使用原文温度损失，不声称精确复现或首创。
同轮复核W69的[SIMPA作者页](https://takelab.fer.hr/publications/simpa/)，
以trait-constrained证据和通用语义相似度的区分限定C14。
结果与差异见[SEMANTIC_LINK报告](../reports/grain_measurement_foundation/SEMANTIC_LINK_REPORT.md)。

2026-09-13观察变化分解来源：**W81，Kitagawa (1955), Components of a Difference
Between Two Rates**，[DOI](https://doi.org/10.1080/01621459.1955.10501299)。
本轮核出版社题名、年份、JASA50(272):1168-1194，未读付费全文。
**W82，Das Gupta (1993), Standardization and Decomposition of Rates: A User's Manual**，
[美国Census原报告](https://www2.census.gov/library/publications/1993/demographics/p23-186.pdf)。
实际读第1章因果边界及§2.2公式2.1-2.9与示例；采用对称乘积分解，保持全部交叉项的
统计解释。GRAIN实现把不共同的文本条件保留为单独支持项，不宣称新定理。
见[RESPONSE_ACCOUNTING报告](../reports/grain_measurement_foundation/RESPONSE_ACCOUNTING_REPORT.md)。

2026-09-13功能读出来源：**W83，Czarnecki, Osindero, Jaderberg, Swirszcz & Pascanu
(2017), Sobolev Training for Neural Networks**，NeurIPS30。
[会议原文](https://proceedings.neurips.cc/paper_files/paper/2017/file/758a06618c69880a6cee5314ee42d52f-Paper.pdf)。
实际读Introduction、§2式1-2及§3动机。其导数监督与本批有限评论线性差分不同；
本批AT从同一T导出，没有额外独立导数观测，不搬用原文样本复杂度收益结论。
**W84，GNU GSL，Linear Least-Squares Fitting**，
[官方文档](https://www.gnu.org/software/gsl/doc/html/lls.html)。
核搜索索引中的weighted/Tikhonov目标及normal equations，整页open失败；
不声称完整研读。结果与标准代数推导见
[FUNCTIONAL_READER报告](../reports/grain_measurement_foundation/FUNCTIONAL_READER_REPORT.md)。

2026-09-13 C06重放：W43 JIVE补读§1.2/§2.1/§2.2，区分共同、view-specific、
残差与缩放。冻结V7是借鉴而非原JIVE算法；`R=X-J`不具独立成分/正交保证。
**W85，DeYoung, Jain, Rajani, Lehman, Xiong, Socher & Wallace (2020),
ERASER: A Benchmark to Evaluate Rationalized NLP Models**，ACL，4443–4458。
[原文](https://aclanthology.org/2020.acl-main.408.pdf)；
[出版页/软件附件](https://aclanthology.org/2020.acl-main.408/)；DOI10.18653/v1/2020.acl-main.408。
Codex实际读§4.2式1–3及§6讨论。删除/仅保留rationale的预测变化不同于原文地址
覆盖率，随机/长度控制也不同于人格外部效度。本批只建立精确可追溯接口，未运行
ERASER基准或宣称rationale faithfulness。详见
[多视图地址报告](../reports/grain_measurement_foundation/MULTIVIEW_ADDRESS_REPORT.md)。

2026-09-13 C03/C16实际接入的新增参照（非只列线索）：

- **W86，Qi et al.(2020), Stanza: A Python Natural Language Processing Toolkit
  for Many Human Languages**。[ACL论文](https://aclanthology.org/2020.acl-demos.14/)、
  [官方pipeline](https://stanfordnlp.github.io/stanza/pipeline.html)、
  [官方代码](https://github.com/stanfordnlp/stanza)。Codex读摘要/processor依赖与离线调用
  文档，实际部署1.11.0 EWT/nocharlm五processor；699原评论及完整冷重放完成。
- **W87，Universal Dependencies relation reference**，
  [ccomp](https://universaldependencies.org/u/dep/ccomp.html)、
  [xcomp](https://universaldependencies.org/u/dep/xcomp.html)、
  [advcl](https://universaldependencies.org/u/dep/advcl.html)。实际阅读定义/例句，用作
  图适配语义：报告来源、控制及状语不同；语法边不自动确定共指/施事/事件真实发生。
- **W88，Sauri & Pustejovsky(2012), Are You Sure That This Happened? Assessing
  the Factuality Degree of Events in Text**。[原文](https://aclanthology.org/J12-2002.pdf)。
  Codex实际读§2、§3.4的SIP/NSIP、§4.1/4.3算法及词汇资源描述；非全39页审读。
  继承来源视角/嵌套/极性/确定程度分工；未复现完整De Facto、原646谓词词典或FactBank。
  修正本repo被动报告的主语/来源误归，保留查询相关性为单独层。

产物与边界见[SOURCE_EVENT_GRAPH报告](../reports/grain_measurement_foundation/SOURCE_EVENT_GRAPH_REPORT.md)。

### 查询到事件的粒度与范围续批（2026-09-13）

- **W89，Chen et al.(2024), Dense X Retrieval: What Retrieval Granularity Should
  We Use?** [论文](https://aclanthology.org/2024.emnlp-main.845/)、
  [作者仓库](https://github.com/chentong0/factoid-wiki)。实际读§2、附录C质量分析和G错误案例；
  原子性/必要上下文启发本批局部检索与范围解释分工。已定位作者公开Flan-T5-large
  propositionizer模型接口，未下载/运行；本批依存图片段不是其独立命题复现。
- **W90，Choi et al.(2021), Decontextualization: Making Sentences Stand-Alone**。
  [原文入口](https://aclanthology.org/2021.tacl-1.27/)，DOI10.1162/tacl_a_00377。
  实读摘要/任务定义，PDF访问失败，不记全文精读或模型复现。用于约束“截取片段”
  与“意义保持的脱语境化”两者的区别。

本批[QUERY_EVENT报告](../reports/grain_measurement_foundation/QUERY_EVENT_GROUNDING_REPORT.md)：
80自然材料的局部/范围对比，既有模型判断不是人工金标准。

### 同地址双线性构造续批（2026-09-13）

- **W91，Lin, RoyChowdhury & Maji (2015), Bilinear CNN Models for Fine-Grained
  Visual Recognition**。[CVF原文](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf)、
  [作者实现](https://github.com/cvl-umass/bilinear-cnn)。实读摘要和§2的局部外积/池化
  定义；没有下载或执行图像CNN。继承“同一位置两类特征外积再聚合”的数学形式，
  适配为同一文本单位的GRAIN24与语义30。另以均值外积、作者内随机配对作对照，
  不宣称双线性特征或协方差恒等式是GRAIN原创。执行入口
  `GRAIN_LOCAL_BILINEAR_CONSTRUCTION_PLAN_V1.md`。

## F. 保留但尚不承担新颖性裁决的线索

### 固定条件参照续批（2026-09-13）

- **W92，Gelman & Little (1997), Poststratification Into Many Categories Using
  Hierarchical Logistic Regression**。[作者全文](https://sites.stat.columbia.edu/gelman/research/published/poststrat3.pdf)、
  [Statistics Canada记录](https://www150.statcan.gc.ca/n1/en/catalogue/12-001-X19970023616)。
  实读摘要与§2.1-2.3：条件均值、已知目标权重及稀疏单元的模型借力。
  本地继承直接条件标准化，不声称实现该文logistic多层模型；自然社群不是问卷抽样层，
  也不默许条件缺失随机。执行见`GRAIN_CONDITIONAL_REFERENCE_PLAN_V1.md`。
- **W93，Manski (2003), Partial Identification of Probability Distributions**。
  [Cemmap目录/导论节选](https://www.cemmap.ac.uk/wp-content/legacy/forms/manskipaper.pdf)。
  实读导论pp1-5与目录，未读完整章节。继承“数据+明确完成类→集合”及识别/抽样
  不确定性分离；GRAIN本地使用单位球与线性映射自行推导共享向量范围。
  没有将节选当作Manski1989原文或假称复现全书估计器。

本批`conditional_reference_r2`对两语料构造固定目标及计分对象；未知条件范围不是
心理属性CI。先前文献已覆盖的标准化/凸几何不作原创声明。

### 条件响应借力续读（2026-09-13）

- **W94，Rasmussen & Williams (2006), Gaussian Processes for Machine Learning**。
  [作者第2章全文](https://gaussianprocess.org/gpml/chapters/RW2.pdf)。实读§2.2的
  条件均值/协方差式2.23-2.24、§2.7式2.38/2.40的固定均值及随机系数合并。
  这是此前`GRAIN_REFERENCE_MECHANISM_INTEGRATION_PLAN_V1.md`已引用的方法的
  再使用与统一编号，**不是新发现的先行研究**。本次以作者内条件为随机过程索引，
  来源GRAIN均值定义核，30个语义响应共用核权重，整体隐藏条件做来源外验证。
  不等同于旧文本核、心理因果响应或完整GP超参数贝叶斯推断；来源均值/核的估计
  不确定性未纳入模型SD。见`GRAIN_CONDITIONAL_RESPONSE_PLAN_V1.md`及对应报告。
  正式方法本身属于继承，阴性适配结果不能反驳GP定理。

W68整段/分片续读（2026-09-13）：重新读取E5官方卡Usage/FAQ/Limitations及
[基础论文摘要与版本记录](https://arxiv.org/abs/2212.03533)。官方卡明确query前缀、
masked mean/L2和512-token上限，因此本轮使用同一前500个content token做whole/piece
对照，另列完整材料覆盖。记录为既有资源的进一步使用，不增算一篇新论文。
词法位置重置/重复模板/上下文限制共同改变，不把该比较叫注意力的独立因果检验。
执行计划：`GRAIN_MATCHED_CONTEXT_CONSTRUCTION_PLAN_V1.md`。

- Claude S11：Bunker & Kwan (2021)，offline/social-media Big Five，DOI待补。
- Claude S22/S23：Mehl et al. (2006)，[10.1037/0022-3514.90.5.862](https://doi.org/10.1037/0022-3514.90.5.862)；
  Tackman et al. (2020)，[10.1002/per.2283](https://doi.org/10.1002/per.2283)。仅Claude摘要层，完整题名待导出。
- Claude S31/S38/S39：Dukanovic & Krpan (2025)，[10.3389/fpsyg.2025.1564979](https://doi.org/10.3389/fpsyg.2025.1564979)；
  LLM评分者G-theory [arXiv 2507.19980](https://arxiv.org/abs/2507.19980)；
  adaptive interviewer [10.1038/s41598-026-46517-7](https://doi.org/10.1038/s41598-026-46517-7)。
  不以访谈者表现替代人格效度。
- Claude S44：Tay et al. (2022)，**A Conceptual Framework for Investigating and Mitigating Machine-Learning Measurement Bias (MLMB) in Psychological Assessment**，
  [10.1177/25152459211061337](https://doi.org/10.1177/25152459211061337)。
  Claude摘要，Codex出版社访问失败后通过Crossref核准题名；全文方法仍待读。
- Garrido & Russell-Lasalandra的embedding维数论文：从W08引用链定位，原作版本待核。
  不单独采用W08转述的PCA/EGA成功率作为普遍定理。
- [ATLAS](https://www.tandfonline.com/doi/full/10.1080/17439760.2026.2701096)、
  [response-free semantic latent structures](https://arxiv.org/abs/2602.12575)、
  [symptom bottleneck dissertation](https://research-portal.uu.nl/ws/files/260821025/gizemdissertation%20-%2067e841bedad3b.pdf)：
  已定位但未充分验证其具体方法、出版状态或心理效标。不冒充本轮独立实证。

## G. 引用链与检索停止点

本轮实际有用的追踪链：MEM/BLT -> 相关/结构/层级topic模型；
Semantic Measures -> transformers测量 -> 后续复制；
ECD -> construct-specific AIG -> AI-GENIE/HEXACO SJT；
JIVE -> sJIVE；concept bottleneck -> leakage -> CB-LLMs；
PANDORA -> 同作者SIMPA；HaRT/DLATK -> longitudinal NLP；
CAPS/DIAMONDS -> contextualization -> contact/construal。

这是一份机制导向的范围综述登记，不是PRISMA系统综述。
未系统导出所有数据库结果、重复项、排除理由和引文网络，因此不报“穷尽率”。
停止扩张是因为已覆盖机制矩阵各层并找到可改变下一步的直接对手，
不是因为已证明全领域不存在更接近的方法。下一轮优先精读最接近对手，
而不是继续增加没有决策用途的泛相关文献。
