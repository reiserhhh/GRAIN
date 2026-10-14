# Project Persona / GRAIN 文献与来源登记簿

更新：2026-09-12。用途：找得到出处、分清实际读到哪里、知道哪条主张依赖哪篇文献。
这是持续维护的定向目录，不是“这些文献都已全文精读”的声明，也不是系统综述。

执行补录：先行方法已进入实际比较，见[方法规格与读取范围](GRAIN_PREDECESSOR_METHOD_CARDS_V1.md)
及[第一批结果](../reports/grain_vector_mechanisms/PREDECESSOR_INTEGRATION_REPORT_V1.md)。
广域登记增补W66-W68（MEM教程、lme4、E5模型卡），W02/W07/W44更新了实际读取/运行深度。
N10本轮补读Factor generation及Stability；没有把MEM适配器当作BLT原生FA实现。

## 1. 使用规则与版本

- **F**：读过全文中明确列出的相关段，不代表逐页读完整篇。
- **I**：检索索引展示的原文方法/讨论节选；原站全文仍受限。
- **A**：摘要/作者研究页；只能据此判断直接相关性，不能裁决其完整方法有bug。
- **M**：书目、目录或来源入口；尚不承担证明。
- **C**：源码检查；实现存在不代表已运行或有效。
- **继承**：从本项目9月11/12日阅读记录导入，本轮没有重新逐篇访问。
- **本轮**：此次双向审查重新查看了指定内容。新/旧阅读证据不混为同一次精读。

保留原有R/P/M编号以免历史引用失效。`P3=R06`、`M1=R16`不重复登记。
论文、书籍、预印本、报告、演示文稿和代码分开标记。没有核实的完整题名不自行补造。
正式投稿时从DOI/出版社导出完整作者与书目格式，不能直接将本表的作者简写当APA参考文献。

关联文件：

- [全体系综述](GRAIN_WHOLE_SYSTEM_LITERATURE_REVIEW_20260912.md)、[27项机制矩阵](GRAIN_MECHANISM_PREDECESSOR_MATRIX_20260912.md)。
- [广域补充W01-W65](GRAIN_LITERATURE_REGISTER_WIDE_20260912.md)：65项新增登记，保留逐项阅读深度与版本；与此处原有65项学术/理论来源合计130项编号来源，不是130篇全文精读。
- [三轮独立检索/交换记录](GRAIN_WIDE_LITERATURE_SEMINAR_SYNTHESIS_20260912.md)：SIMPA、semantic measurement、ECD、sJIVE、DSEM、AI-GENIE/JAM已是直接比较对象。
- [双向理论审查与RQ1-RQ5](GRAIN_BIDIRECTIONAL_THEORY_AUDIT_20260912.md)。
- [9月11日文献地图](GRAIN_COMPLETION_LITERATURE_MAP_V1.md)：R编号的首次用途及引用链。
- [9月12日研讨文献档案](GRAIN_BREAKTHROUGH_SOURCE_DOSSIER_20260912.md)：P/M编号的方法对齐。
- [系统回顾](GRAIN_SYSTEM_RETROSPECTIVE_AND_ARCHITECTURE_20260911.md)：项目历史与实证来源。
- [V5研究规划](GRAIN_SCIENTIFIC_TARGET_AND_RESEARCH_PROGRAM_V5.md)：当前规划入口。

## 2. 原始核心书目：心理理论与测量

| ID | 作者、年份、题名及原始出处 | 阅读证据 | 在项目中的用途 |
|---|---|---|---|
| R01 | Borsboom et al. (2021). **Theory Construction Methodology: A Practical Framework for Building Theories in Psychology**. [DOI](https://doi.org/10.1177/1745691620969647)；[作者全文](https://pure.uva.nl/ws/files/68321294/1745691620969647.pdf) | 继承F：五步骤、mutualism示例；本轮核对书目 | 从现象到竞争生成模型；RQ1-5 |
| R02 | van der Maas et al. (2006). **A Dynamical Model of General Intelligence: The Positive Manifold of Intelligence by Mutualism**. [作者机构记录](https://research.vu.nl/en/publications/a-dynamical-model-of-general-intelligence-the-positive-manifold-o/) | 继承A | 相关结构可由交互产生，不唯一指定潜因子；RQ5 |
| R03 | Borsboom, Mellenbergh & van Heerden (2004). **The Concept of Validity**. [DOI](https://doi.org/10.1037/0033-295X.111.4.1061) | 继承M：机构记录 | 测量属性与分数关系的理论立场；不能等同所有效度学派；RQ1/4 |
| R04 | AERA, APA & NCME (2014). **Standards for Educational and Psychological Testing**. [官方开放版](https://www.testingstandards.net/open-access-files.html) | 继承F：开放入口、效度框架 | 解释、用途、参照和精度；不是本阶段临床验收清单；RQ4 |
| R05 | Brennan (2001). **Generalizability Theory**. [出版社/DOI](https://doi.org/10.1007/978-1-4757-3456-0) | 继承M：目录，未读全书 | 作者、场景、评分器和机会的误差设计；RQ2 |
| R06 | Fleeson & Jayawickreme (2015). **Whole Trait Theory**. [作者全文入口](https://pmc.ncbi.nlm.nih.gov/articles/PMC4472377/) | 继承A/I：状态分布与解释段；直接访问曾遇验证码 | 描述性分布与解释机制；别名P3；RQ2/5 |
| R07 | Molenaar (2004). **A Manifesto on Psychology as Idiographic Science: Bringing the Person Back Into Scientific Psychology, This Time Forever**. [机构记录](https://dare.uva.nl/id/9ad493a2-825b-4348-ac79-a7aede156153) | 继承M | 个体内过程和群体协方差分开；RQ2/5 |
| R08 | Funder (1999). **Personality Judgment: A Realistic Approach to Person Perception**. [出版页](https://www.educate.elsevier.com/book/details/9780122699306) | 继承M：目录；不是1995论文全文 | 线索相关性、可获得性、识别和利用；RQ3 |
| P4 | Rauthmann et al. (2014). **The Situational Eight DIAMONDS: A Taxonomy of Major Dimensions of Situation Characteristics**. [原始记录](https://pubmed.ncbi.nlm.nih.gov/25133715/)；[作者PDF](https://rap.ucr.edu/pub3.pdf) | 继承A | 情境特征不只是topic；RQ1/2 |
| P5 | Karelaia & Hogarth (2008). **Determinants of Linear Judgment: A Meta-analysis of Lens Model Studies**. [DOI](https://doi.org/10.1037/0033-2909.134.3.404) | 继承A；完整公式比较待补 | 线索效度与利用分开；RQ3 |
| P6 | Rasch (1977). **Specific objectivity原始文献档案**（完整书目信息待原件核对）. [档案](https://www.rasch.org/rasch.htm)；[数学说明](https://www.rasch.org/rmt/rmt11a.htm) | 继承M/数学说明，未读原始memoir | 非同题比较并非新问题；RQ2 |
| P8 | Liu & Wang (2017). **Non-ignorable Missingness Item Response Theory Models for Choice Effects in Examinee-selected Items**. [DOI](https://doi.org/10.1111/bmsp.12097)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/28390145/) | 本轮A：双潜变量、MNAR及模型比较摘要 | 选择与作答联合建模的直接对手；RQ2 |
| P9 | Liu, Qiu & Wang (2019; online 2018). **Item Response Theory Modeling for Examinee-selected Items with Rater Effect**. [DOI](https://doi.org/10.1177/0146621618798667) | 本轮A；继承I：模型方法节选，全文待补 | 选择与评分器已有联合处理；RQ2/3 |
| P11 | Jeon et al. (2021). **Mapping Unobserved Item-Respondent Interactions: A Latent Space Item Response Model with Interaction Map**. [DOI](https://doi.org/10.1007/s11336-021-09762-5)；[原文](https://arxiv.org/abs/2007.08719) | 继承F：§2.2；本轮核对题名 | 交互几何不自动等于能力空间；RQ2 |
| R28 | Campbell & Fiske (1959). **Convergent and Discriminant Validation by the Multitrait-Multimethod Matrix**. [原始记录](https://pubmed.ncbi.nlm.nih.gov/13634291/) | 继承M | 多方法效标不能由同一LLM的多个回答替代；RQ4 |

## 3. 文本、人格、叙事和LLM机制

| ID | 作者、年份、题名及原始出处 | 阅读证据 | 在项目中的用途 |
|---|---|---|---|
| R09 | Soni et al. (2022). **Human Language Modeling**. [ACL](https://aclanthology.org/2022.findings-acl.52/) | 继承A | 作者历史表示，HaRT；RQ3 |
| R10 | Gjurković et al. (2021). **PANDORA Talks: Personality and Demographics on Reddit**. [ACL/DOI](https://aclanthology.org/2021.socialnlp-1.12/) | 继承A及项目Table8协议审查；本轮未重跑复现 | N/NP、桥接与原始预测历史；RQ1/4 |
| R11 | Peters & Matz (2024). **Large Language Models Can Infer Psychological Dispositions of Social Media Users**. [PNAS Nexus](https://academic.oup.com/pnasnexus/article/3/6/pgae231/7692212) | 继承F：数据与推论方法 | 能力现象与神经机制分开；RQ3 |
| R12 | Yang & Oshio (2026). **Attachment towards Generative AI Influences Self-AI Agreement in Personality Reports**. [DOI/全文](https://onlinelibrary.wiley.com/doi/10.1111/ajsp.70102?af=R) | 本轮F：§§2.1-2.3、结果、§4限制；补充prompt未读取 | 从仅摘要升级；关联与披露/记忆机制不同；RQ3 |
| R13 | Tak et al. (2025). **Mechanistic Interpretability of Emotion Inference in Large Language Models**. [ACL](https://aclanthology.org/2025.findings-acl.679/)；[PDF](https://aclanthology.org/2025.findings-acl.679.pdf) | 本轮F：pp.4-5 patching、p.9 §11局限 | 情绪概念定位与干预的直接先例；RQ3/5 |
| R14 | Anthropic (2025). **Persona Vectors: Monitoring and Controlling Character Traits in Language Models**. [作者研究页](https://www.anthropic.com/research/persona-vectors) | 继承A；研究报告，不以网页代替全部方法 | persona控制不保证人类特质推论互逆；RQ3 |
| R15 | Hossain et al. (2025). **AutoPsyC: Automatic Recognition of Psychodynamic Conflicts from Semi-structured Interviews with Large Language Models**. [ACL](https://aclanthology.org/2025.clpsych-1.2/) | 继承A | 心理动力访谈编码，不等于TEG/TA或临床诊断；RQ4 |
| P1 | Brickman, Gupta & Oltmanns (2025). **Large Language Models for Psychological Assessment: A Comprehensive Overview**. [DOI](https://doi.org/10.1177/25152459251343582)；[PDF](https://journals.sagepub.com/doi/pdf/10.1177/25152459251343582) | 继承F：pp.1-6；本轮核对题名与引用链 | 流程本身已有，追查N04；RQ1/4 |
| P2 | Fang, Nguyen & Oberski (2022). **Evaluating the Construct Validity of Text Embeddings with Application to Survey Questions**. [DOI/全文](https://link.springer.com/article/10.1140/epjds/s13688-022-00353-7) | 继承A/intro；本轮A与机构书目核对 | 问题语义表示不等于人的测量；RQ1/4 |
| P7 | Egami et al. (2022). **How to Make Causal Inferences Using Texts**. [作者全文](https://naokiegami.com/paper/CausalText.pdf) | 继承F：摘要与工作流程 | 发现、表示选择和推断的关系；RQ1/4 |
| P10 | Zoanetti, Zhou & Tan (2020). **Integrating Natural Language Processing Features within Explanatory Item Response Models**. [ACER记录](https://people.acer.org/en/publications/integrating-natural-language-processing-features-within-explanato/) | 继承A：演示文稿摘要，不冒充正式方法论文 | NLP性质进入解释性IRT已有先例；RQ2 |
| P12 | Speer, Delacruz, Chawota, Perrotta & Rudolph (2026). **Unpacking the Validity of Open-Ended Personality Assessments Using Fine-Tuned Large Language Models**. [DOI](https://doi.org/10.1177/10944281251413746) | 本轮A：出版社摘要；全文受限 | 提示、叙事证据和不同效标；RQ1/3/4 |
| P13 | De Duro, Franchino & Stella (2026). **Natural Language Processing Psychometrics**. [预印本](https://arxiv.org/abs/2608.07316) | 本轮A与题名核对；预印本，完整协议待读 | 合成到真人与语言证据近邻；RQ1/4 |
| P14 | Cheng & Liu (2019). **Enhancing Item Response Theory for Cognitive Diagnosis**. [arXiv](https://arxiv.org/abs/1905.10957)；[会议DOI](https://doi.org/10.1145/3357384.3358070) | 本轮A；模型简称DIRT，不与其他Deep-IRT项目混同 | 问题语义与IRT参数联合学习；RQ2/4 |
| P15 | Lee, Son & Jia (2025 online; 2026 issue). **AI-powered Automatic Item Generation for Psychological Tests: A Conceptual Framework for an LLM-based Multi-Agent AIG System**. [出版社/DOI](https://link.springer.com/article/10.1007/s10869-025-10067-y) | 继承A/intro；本轮Crossref核对完整题名 | 自动生成构念相关题目已有先例；RQ4 |
| N01 | Jain & Wallace (2019). **Attention is not Explanation**. [ACL/DOI](https://aclanthology.org/N19-1357/)；[PDF](https://aclanthology.org/N19-1357.pdf) | 本轮F：§§1-4，模型假设与替代注意力 | 热图与模型作用不能直接画等号；RQ3 |
| N02 | Wiegreffe & Pinter (2019). **Attention is not not Explanation**. [ACL/DOI](https://aclanthology.org/D19-1002/)；[PDF](https://aclanthology.org/anthology-files/pdf/D/D19/D19-1002.pdf) | 本轮F：§§2-3.4、§4引入 | 模型一致的替代路径与简单读者诊断；RQ3 |
| N03 | Piastra & Catellani (2025). **On the Emergent Capabilities of ChatGPT 4 to Estimate Personality Traits**. [DOI](https://doi.org/10.3389/frai.2025.1484260)；[作者PDF](https://www.patriziacatellani.com/images/2025/On_the_emergent_capabilities_of_ChatGPT_to_estimate_personality_traits_Froe_2025.pdf) | 本轮A：从早期会话恢复出处；未重读全部实验 | Big5数字估计，不是MBTI轴分类；RQ3 |
| N04 | Oltmanns et al. (2026 issue; 2025 online). **Language-based AI Modeling of Personality Traits and Pathology from Life Narrative Interviews**. [DOI](https://doi.org/10.1037/abn0001047)；[PubMed](https://pubmed.ncbi.nlm.nih.gov/41021515/)；[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12616480/) | 本轮A/I：方法与讨论索引节选；直接PMC访问验证码 | 生活叙事、监督编码器、多方法效标的强近邻；RQ1/4 |
| N05 | Mischel & Shoda (1995). **A Cognitive-Affective System Theory of Personality: Reconceptualizing Situations, Dispositions, Dynamics, and Invariance in Personality Structure**. [PubMed](https://pubmed.ncbi.nlm.nih.gov/7740090/)；[作者机构PDF](https://psychology.columbia.edu/sites/default/files/2016-11/246.pdf) | 本轮M/A：书目与摘要；PDF新定位，未精读 | CAPS和条件反应的先例；RQ2 |
| N06 | Pennebaker & King (1999). **Linguistic Styles: Language Use as an Individual Difference**. [PubMed/DOI](https://pubmed.ncbi.nlm.nih.gov/10626371/) | 本轮M/A；不是当前ESSAYS预处理文件的版本证据 | Essays、语言风格历史来源；RQ1/4 |
| N07 | McCrae & Costa (1989). **Reinterpreting the Myers-Briggs Type Indicator from the Perspective of the Five-Factor Model of Personality**. [DOI](https://doi.org/10.1111/j.1467-6494.1989.tb00759.x)；[作者机构记录](https://scholars.duke.edu/publication/1458465) | 本轮A | Big5/MBTI桥接历史；跨量表关联不是个体精确换算；RQ4 |
| N08 | Schwartz et al. (2013). **Personality, Gender, and Age in the Language of Social Media: The Open-Vocabulary Approach**. [原始全文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0073791) | 本轮A/I及引用链定位 | 开放词汇与人口/心理关联；沿其引文找到N09；RQ1 |
| N09 | Chung & Pennebaker (2008; online 2007). **Revealing Dimensions of Thinking in Open-Ended Self-Descriptions: An Automated Meaning Extraction Method for Natural Language**. [DOI](https://doi.org/10.1016/j.jrp.2007.04.006)；[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0092656607000451)；[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2542899/) | 本轮F/I：出版社开放intro与方法说明、索引结果节；非全文逐页精读 | MEM是因子发现直接近邻；关注领域不等于正负极性；RQ1 |
| N10 | Kulkarni et al. (2018; preprint 2017). **Latent Human Traits in the Language of Social Media: An Open-Vocabulary Approach**. [PLOS/DOI](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201703)；[预印本](https://arxiv.org/abs/1705.08038) | 本轮F：Materials and methods → Factor generation、评价目标、背景；全部测试细节待复查 | BLT直接近邻；比例相关性过强表述见审查L1；RQ1/2/5 |
| N11 | Funder (1995). **On the Accuracy of Personality Judgment: A Realistic Approach**. [DOI](https://doi.org/10.1037/0033-295X.102.4.652) | 本轮M：由R12参考文献核对；未精读 | RAM原始出处，不拿1999书目录代替；RQ3 |

## 4. 数学工具与具体职责

| ID | 作者、年份、题名及原始出处 | 阅读证据 | 在项目中的用途 |
|---|---|---|---|
| R16 | Hyvärinen, Sasaki & Turner (2019). **Nonlinear ICA Using Auxiliary Variables and Generalized Contrastive Learning**. [PMLR/PDF](https://proceedings.mlr.press/v89/hyvarinen19a/hyvarinen19a.pdf) | 继承F：§§3-5、Theorems1-3 | 别名M1；识别需条件，而非任意topic；RQ2/5 |
| R17 | Locatello et al. (2019). **Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations**. [PMLR](https://proceedings.mlr.press/v97/locatello19a.html) | 继承A/理论结论 | 无偏置的唯一解耦不保证；RQ5 |
| R18 | Fukumizu, Bach & Jordan (2009). **Kernel Dimension Reduction in Regression**. [原始预印本](https://arxiv.org/abs/0908.1854) | 继承A | 目标相关降维，不是最大总方差；RQ1/4 |
| R19 | Zaheer et al. (2017). **Deep Sets**. [NeurIPS](https://papers.nips.cc/paper/6931-deep-sets) | 继承A | 逐元素变换后聚合，不等于均值后非线性；RQ1 |
| R20 | Wagstaff et al. (2022). **Universal Approximation of Functions on Sets**. [JMLR](https://jmlr.org/papers/v23/21-0730.html) | 继承A | 有限聚合的容量边界；RQ1/4 |
| R21 | Xu et al. (2020). **A Theory of Usable Information under Computational Constraints**. [原始预印本](https://arxiv.org/abs/2002.10689) | 继承A | 读者依赖的可用信息；RQ1/5 |
| R22 | Hewitt et al. (2021). **Conditional Probing: Measuring Usable Information Beyond a Baseline**. [ACL/PDF](https://aclanthology.org/2021.emnlp-main.122.pdf) | 继承F：定义、附录D | GRAIN与路径增量比较；RQ1/4/5 |
| R23 | Hewitt & Liang (2019). **Designing and Interpreting Probes with Control Tasks**. [ACL](https://aclanthology.org/D19-1275/) | 继承A | probe能力与表示信息分开；RQ3/5 |
| R24 | Geiger et al. (2025). **Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability**. [JMLR/PDF](https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf) | 继承F：§2、Definition25、复合关系 | 干预映射，不等于心理因果；RQ3/4 |
| R25 | Rubenstein et al. (2017). **Causal Consistency of Structural Equation Models**. [原始预印本](https://arxiv.org/abs/1707.00819) | 继承A | 微观/宏观干预对应；RQ4 |
| R26 | Czarnecki et al. (2017). **Sobolev Training for Neural Networks**. [NeurIPS](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html) | 继承A | 一个程序的值与导数约束；RQ3/4 |
| R27 | Bareinboim & Pearl (2016). **Causal Inference and the Data-fusion Problem**. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4941504/) | 继承A；直接全文访问受限 | 跨来源拼接需要识别条件；RQ2/4 |
| R29 | Sundararajan, Taly & Yan (2017). **Axiomatic Attribution for Deep Networks**. [PMLR](https://proceedings.mlr.press/v70/sundararajan17a.html) | 继承A；2026-09-13 Codex补读原文§3式1/Proposition1及§5基线段 | 积分归因与完整性，不保证唯一原因；RQ3。M31参照锚定续批复用，不算新找到的论文 |
| R30 | Ribeiro, Wu, Guestrin & Singh (2020). **Beyond Accuracy: Behavioral Testing of NLP Models with CheckList**. [ACL原文](https://aclanthology.org/2020.acl-main.442.pdf)；[作者代码](https://github.com/marcotcr/checklist) | 2026-09-13 Codex读§§2.1-2.3：能力矩阵、minimum functionality/invariance/directional expectation、模板覆盖。代码仅定位、未安装或复制 | 窄目标作用实验的前辈方法；用无标签文本的预测关系测试，不把作者设定的行为方向当真人量表真值。新增匹配原生/固定向量函数响应与分量分解；不是论文精确复现或因果人格效度 |
| M2 | Gresele et al. (2020; UAI2019). **The Incomplete Rosetta Stone Problem: Identifiability Results for Multi-View Nonlinear ICA**. [PMLR/PDF](https://proceedings.mlr.press/v115/gresele20a/gresele20a.pdf) | 继承F：§§1-3 | 同文多个读者不自动满足独立噪声视图；RQ3/5 |
| M3 | Park & Muandet (2020). **A Measure-Theoretic Approach to Kernel Conditional Mean Embeddings**. [NeurIPS/PDF](https://papers.neurips.cc/paper_files/paper/2020/file/f340f1b1f65b6df5b5e3f94d95b11daf-Paper.pdf) | 继承F：§4、Theorems4.2-4.4 | 条件响应分布估计；RQ2 |
| M4 | Szabo, Gretton, Poczos & Sriperumbudur (2015). **Two-stage Sampled Learning Theory on Distributions**. [PMLR](https://proceedings.mlr.press/v38/szabo15.html) | 继承F：setup/risk；本轮核对正式题名 | 作者数与每作者文本数是两层；RQ2/4 |
| M5 | Park, Shalit, Schölkopf & Muandet (2021). **Conditional Distributional Treatment Effect with Kernel Conditional Mean Embeddings and U-Statistic Regression**. [PMLR](https://proceedings.mlr.press/v139/park21c.html) | 继承F：§5.1；本轮核对书目 | witness定位差异；作者比较不自动是处理效应；RQ2/5 |
| M6 | Blackwell (1951/1953). **Comparison of Experiments / Equivalent Comparisons of Experiments**. [原始文献档案](https://celebratio.org/Blackwell_DH/article/255/)；Khan, Yu & Zhang，**A Simple Proof of Blackwell's Theorem on the Comparison of Experiments for a General State Space**，[DOI](https://doi.org/10.1016/j.econlet.2024.112146) | 继承A/M；本轮Crossref核对后续证明为2025 issue，不因DOI含2024便定出版年；原始PDF未读 | 保留哪些决策而非所有向量信息；原作与后续证明不混为一篇；RQ4 |
| M7 | Driver, Oud & Voelkle (2017). **Continuous Time Structural Equation Modeling with R Package ctsem**. [JSS](https://www.jstatsoft.org/article/view/v077i05) | 继承A/作者代码 | 潜在状态与观测误差，token序号非心理时间；RQ2/5 |
| M8a | Salvi et al. (2021). **The Signature Kernel is the Solution of a Goursat PDE**. [原始预印本](https://arxiv.org/abs/2006.14794) | 继承A；原M8主文 | 有序跨因子交互表示；RQ5 |
| M8b | Cass, Piatti & Pei (2025). **Numerical Schemes for Signature Kernels**. [原始全文](https://arxiv.org/html/2502.08470v1) | 继承F：intro/algorithm；本轮核对题名，预印本 | 原M8后续算法文；数值工具非心理理论；RQ5 |
| M9 | Di et al. (2009). **Multilevel Functional Principal Component Analysis**. [作者预印本](https://arxiv.org/abs/0906.1457) | 继承A | 人际和个体内函数变化；RQ2/5 |
| M10 | Király & Tomioka (2012). **A Combinatorial Algebraic Approach for the Identifiability of Low-Rank Matrix Completion**. [ICML/PDF](https://icml.cc/2012/papers/510.pdf) | 继承F：Prop2.13、Example2.16；本轮核对题名 | 连接图不足以恢复任意响应；RQ2 |
| M11 | Wang & Singh (2018). **Provably Correct Algorithms for Matrix Column Subset Selection with Selectively Sampled Data**. [JMLR](https://www.jmlr.org/papers/v18/15-233.html) | 继承F：intro/objective/assumptions；本轮核对题名 | 主动选择观察不是新发明；RQ4 |

### 2026-09-13 来源归属续批

| ID | 作者/版本/原文 | 实际阅读 | 对接、继承与范围 |
|---|---|---|---|
| R31 | Silvia Pareti (2016). **PARC 3.0: A Corpus of Attribution Relations**. LREC, 3914-3920. [原文](https://aclanthology.org/L16-1619.pdf) | Codex读§3.1及来源/cue/content定义；ACL元数据核对 | C03/C16，区分报告来源与内容。没有下载其受限底层新闻语料或复现标注实验；GRAIN本次合成/本地原文适配，不证明作者人格 |
| R32 | Grace Muzny, Michael Fang, Angel Chang & Dan Jurafsky (2017). **A Two-stage Sieve Approach for Quote Attribution**. EACL, 460-470. [原文](https://aclanthology.org/E17-1044.pdf) | Codex读§§5.1-5.2，quote-to-mention / mention-to-entity和报告动词依存线索 | C03/C16显式来源修正。文学小说任务；本次仅继承分阶段/明确来源原则，未复制作者代码、无通用共指或完整原实验复现。实现/结果见SOURCE_OWNERSHIP_SCORING_REPORT.md |

## 5. 工具登记，不等同论文结论

| ID | 原始仓库 | 已查看 / 未完成 | 用途 |
|---|---|---|---|
| T01 | [OscarKjell/text](https://github.com/OscarKjell/text) | 继承README及[作者手册](https://oscarkjell.r-universe.dev/text/doc/manual.html)；未安装 | embedding、语义计分、token/layer接口；RQ1/4 |
| T02 | [koheiw/LSX](https://github.com/koheiw/LSX) | 继承README与论文链接；未执行 | 种子引导文档标度；RQ1 |
| T03 | [maifeng/SPAR_measure](https://github.com/maifeng/SPAR_measure/tree/3924fc4dd2207191b39049ea4d66d81b8d29aad1) | 本轮README、core.py、zca.py；固定commit；未安装执行；GPL-3.0-or-later源码 | 白化与批次依赖审查L2/L3；RQ1/2/5 |
| T04 | [devon7y/semanticfa](https://github.com/devon7y/semanticfa) | 继承README | 条目语义结构不等于人的反应结构 |
| T05 | [rprimi/embeddcv](https://github.com/rprimi/embeddcv) | 继承README | embedding内容效度候选 |
| T06 | [cdriveraus/ctsem](https://github.com/cdriveraus/ctsem) | 继承README与M7摘要 | 状态/测量联合模型 |
| T07 | [ilkhem/iVAE](https://github.com/ilkhem/iVAE) | 继承README/config inventory | 可识别潜变量候选，须先看假设 |
| T08 | [crispitagorico/sigkernel](https://github.com/crispitagorico/sigkernel) | 继承README | 路径核CPU/GPU候选 |
| T09 | [stanfordnlp/pyvene](https://github.com/stanfordnlp/pyvene) | 继承README/API及源码节选 | 序列化激活干预；不直接干预人 |
| T10 | [philchalmers/mirt](https://github.com/philchalmers/mirt) | 继承入口定位，未执行 | 选择/反应IRT比较 |
| T11 | [cran/lsirm12pl](https://github.com/cran/lsirm12pl) | 继承入口与[2025包论文](https://journal.r-project.org/articles/RJ-2025-021/)摘要 | 潜空间交互IRT比较 |
| T12 | [AI-for-Health-Data/OCEANprediction](https://github.com/AI-for-Health-Data/OCEANprediction) | 本轮由N04原文定位；尚未检查源码 | 生活叙事强监督基线候选 |

除T03外，本表未锁定仓库commit。复用前必须锁版本、核对license、数据权限和计算假设。
T03关联的2024论文是Yan, Mai, Wu, Chen & Li，
[A Computational Framework for Understanding Firm Communication During Disasters](https://doi.org/10.1287/isre.2022.0128)。
本轮只核对仓库给出的书目，未审计该论文；不能把README问题归罪于其论文结果。

## 6. 已追溯的引用链与优先补读

| 引用链 / 入口 | 本轮实际得到什么 | 下一次该读哪里 |
|---|---|---|
| R01 → R02 | 继承TCM正文中的mutualism示例 | 需要生成机制竞争时读原始数学，不重复泛搜 |
| R24 → R25；R22 → R21 | 继承因果抽象及条件probing的直接理论来源 | 推导具体抽取目标时回到相应定义 |
| R12 → N11 | 从论文参考表补齐RAM原始DOI | RAM原文与镜头模型方程 |
| N08 → N09；N10 → N09 | 查到MEM与BLT的直接方法继承关系；N10背景明确讨论N09 | 两方法的计分、旋转、文本与作者切分 |
| P1 → 作者出版目录 → N04 | 发现生活叙事与多方法心理效标的直接实验 | 完整训练/切分/聚合协议与T12 |
| N01 ↔ N02 | 读到真实的方法争论，而非只记住两句相反标题 | 只在有具体LLM路径问题时移植适用测试 |
| R06 → Fleeson (2001) | [原始密集状态研究记录](https://pubmed.ncbi.nlm.nih.gov/11414368/) | 继承A/M；状态分布与采样设计 |

优先补读队列已由广域综述更新：**W07/W01（SIMPA/语义测量）+ P8/P9/W26（选择与随机条目）+ W13（ECD）**先做共同模型对照；W44/W30是共享与动态方法入口。P12/N04、N09/N10仍为效标与因子比较核心，不另起无限文献队列。
未核实完整方法的条目不进入“已推翻”列表。文献受限时保留访问说明，不用二手总结填补原文。

仍保留但不承担本轮论证的线索：

- [Zayas, Lee & Shoda的if-then assessment章稿](https://bpb-us-e1.wpmucdn.com/blogs.cornell.edu/dist/5/11179/files/2024/09/2021Zayas_Lee_Shoda_modelingmind_draft_17August2020.pdf)：继承M。
- [2026 interviewer/evaluator/judger文章](https://www.nature.com/articles/s41598-026-62333-5)：已补录W12，仍为摘要层。
- ECD已补录W13/W14；Wilson构造模型、LST-R及TEG/TA的具体版本仍待原始版本逐条对齐，
  不用一个笼统学派名替代量表手册/方法论文，也不复用受版权保护条目。

## 7. 新增记录的最小模板

每条记录须包含：稳定ID；作者与年份；准确题名或明确标注短题名；DOI/原始URL；
出版类型/版本；阅读日期与具体节页；关联主张/RQ；假设和适用边界；原文发现、我们的推论、
实验反例各属哪一层；代码版本/license；访问或材料限制。

若结论改变，添加日期与修订理由，不抹掉旧阅读历史。此处保存元数据和我们的短评，
不集中复制全文、受限PDF、私人文本或密钥。

## 8. 2026-09-13 M06表示交换方法补录

本节为M06实际使用的方法来源，不扩写为一次全领域综述；旧R24等理论登记保持原编号。

| ID | 来源/版本与链接 | 本次读取范围 | GRAIN对应、继承及边界 |
|---|---|---|---|
| M06-R1 | Geiger, Lu, Icard & Potts (2021), *Causal Abstractions of Neural Networks*, NeurIPS论文；[原文入口](https://arxiv.org/abs/2106.02997) | F：2026-09-13，表示交换与高层反事实对应的方法部分；未声称逐页复核所有实验 | 继承表示交换来测试组件作用。M06在同位置替换C/P/Q并跑剩余模型；没有证明完整高层因果抽象、唯一网络路径或人格效度。未复制论文代码，无新增第三方代码license依赖。 |
| M06-R2 | Goldowsky-Dill, MacLeod, Sato & Arora (2023), *Localizing Model Behavior With Path Patching*, arXiv预印本；[原文](https://arxiv.org/abs/2304.05969) | F：2026-09-13，HTML第2.1-2.3节，节点/路径定位及所选材料上的作用范围 | 用于区分节点组交换与逐边路径干预。M06是前者，未执行后者；两种编写材料不代表人群普遍性。有限立方差分及顺序平均分账使用标准数学，不归为本项目新定理。未使用其软件代码。 |

对应结果：[共享/特有查询作用报告](../reports/grain_measurement_foundation/SHARED_QUERY_ACTION_REPORT.md)。
原文机制工具与本地结论分开：文献提供干预思路；P区主要分账、归属串入和跨查询Q作用
是本次固定模型/边界/材料上的实测，不能归为两篇原文的发现。代码和方案hash已在模型调用前冻结。

## 9. 2026-09-13 M07独立读者补核

| ID | 来源与版本 | 阅读深度 | 本地对应与边界 |
|---|---|---|---|
| M07-R1 | Belrose等，*Eliciting Latent Predictions from Transformers with the Tuned Lens*；2023首稿，当前arXiv v6修订2025-11-11；[原文入口](https://arxiv.org/abs/2303.08112) | A：2026-09-13核读摘要/书目，未声称本轮完整方法复现 | 中间状态可经仿射probe读出预测，提示区分表示坐标与后续读出。M07用既有来源算子，不训练该论文的词表分布probe；编译恒等与心理效度仍分开。未复制其代码。 |
| M07-R2 | Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*；NIPS2014 Deep Learning Workshop论文的arXiv v1；[原文](https://arxiv.org/abs/1503.02531) | A：2026-09-13核读摘要/版本 | 继承教师知识与简化运行程序的区分。本轮无新蒸馏训练，不声称本地Jacobian对齐等于其训练方法；原点值、变化作用、外部联系分别验收。 |

对应[M07实测与公式](../reports/grain_measurement_foundation/QUERY_PROGRAM_ACTION_REPORT.md)。
仿射复合恒等、相同表示下的两点误差下界为标准数学推导；查询作用在不同计算层的
分账和当前绑定/顺序反例是本地结果，不挪用为两篇论文的发现，也不标为新的普遍定理。

## 10. 2026-09-13 M10条件设计补核

| ID | 来源/版本 | 本次读取与本地对应 |
|---|---|---|
| M10-R1 | Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press，第2章；[公开章节](https://gaussianprocess.org/gpml/chapters/RW2.pdf) | F：核读第2章权重空间与条件预测公式，未声称全书/原实验复现。继承线性/Gaussian条件矩、核与目标损失；本包二元文本/Markov状态为二阶矩working模型。没有把其正态后验声明搬成心理真值。未复制第三方代码。 |
| M10-R2 | Chaloner & Verdinelli (1995), *Bayesian Experimental Design: A Review*, Statistical Science；[作者技术报告入口](https://www.stat.cmu.edu/tr/tr599/tr599.html) | A：核读题名/作者/来源入口，继承已有登记的目标期望效用设计，不声称本轮完整复现综述。Schur增益为标准数学，本地连接的是来源文本、条件支持、相关事件/副本与错设修复规则。 |

对应[M10报告](../reports/grain_measurement_foundation/NEW_CONDITION_DESIGN_REPORT.md)。
`X_source d=0`的观察不可识别及独立残差方差加法是标准线性代数/概率事实；
实际新条件选择与额外来源修复结果来自本包，不归为先行论文的心理实证。

## 11. 2026-09-13 M04/M05共同观察与条件函数补核

| ID | 来源/版本 | 实际读取、继承与本地边界 |
|---|---|---|
| M0405-R1 | Álvarez, Rosasco & Lawrence (2012), *Kernels for Vector-Valued Functions: a Review*；arXiv v2，2012-04-16；[原始入口](https://arxiv.org/abs/1106.6251)、[HTML](https://arxiv.org/html/1106.6251v2) | F：2026-09-13核读3.3及4节，输入条件与多输出依赖、分离核/协区域化。继承多输出对象而非强行相加的思路；本包使用有限线性基底与来源标定，不复现GP实验，亦不由其独立噪声示例推出同文本通道独立。未复制第三方代码。 |
| M0405-R2 | JCGM (2011), *Evaluation of measurement data — Supplement 2 to the GUM: Extension to any number of output quantities*, JCGM 102:2011；[DOI](https://doi.org/10.59161/JCGM102-2011)、[BIPM原PDF](https://www.bipm.org/documents/20126/2071204/JCGM_102_2011_E.pdf) | F：2026-09-13核读6.2.1及式(3)，多输出函数/敏感度矩阵与协方差传播。固定线性映射下使用精确矩阵传播，不把一般非线性近似称精确；未以该标准证明心理构念效度，也未复制受限内容或第三方代码。 |

对应[M04/M05实测](../reports/grain_measurement_foundation/JOINT_CONDITIONAL_OBSERVATION_REPORT.md)。
矩阵白化、四角点联合矩、事件乘积及条件漂移混合是继承的标准数学；查询K的有限收益、
弱反演失效及事件配对恢复是本地材料/冻结观察器上的结果。结果后乘积检验单独登记，
没有把它写成原r1事前假设；未新增人类外部效度或精确论文复现主张。

## 12. 2026-09-13 M02证据历史与更新补核

| ID | 来源/版本 | 实际读取、继承与本地边界 |
|---|---|---|
| M02-R1 | Green, Karvounarakis & Tannen (2007), *Provenance Semirings*, PODS；[作者PDF](https://web.cs.ucdavis.edu/~green/papers/pods07.pdf)、[DOI](https://doi.org/10.1145/1265530.1265535) | F：2026-09-13摘要/引言、来源注记及第11节的扩展限制，未声称全篇算法复现。继承来源依赖，而非把正半环当可相减概率；本地显式版本/撤回另外实现。未复制第三方代码。 |
| M02-R2 | Katsuno & Mendelzon (1992), *On the difference between updating a knowledge base and revising it*, 收录于*Belief Revision*的书章；[出版社/DOI](https://doi.org/10.1017/CBO9780511526664.007) | A：2026-09-13出版社引言摘录/书目，原KR论文为1991；本次所读为1992章节，2009在线日期不是原创年份。继承旧世界认识修订/世界变化区分；不声称实现全部逻辑公理。 |
| M02-R3 | Morf & Kailath (1975), *Square-root algorithms for least-squares estimation*, IEEE TAC 20(4),487–497；[DOI](https://doi.org/10.1109/TAC.1975.1100994)、[作者上传全文](https://www.researchgate.net/publication/3027429_Square-Root_Algorithms_for_Least-Squares_Estimation) | F：2026-09-13核读作者上传摘要/引言及IV节几何正交化部分；纸面日期1975年8月，平台标9月，采用正式卷期。不复现完整递归状态空间算法；本地QR创新是对平方根/投影原理的有限实现。修复原增量协方差相消，主批量结果不变。 |

M02条件矩继续继承M10-R1。结果见[事件历史报告](../reports/grain_measurement_foundation/EVIDENCE_HISTORY_REPORT.md)：
版本账、来源学习相关事件、个人条件函数与摘要未来更新的可执行连接是本地研究内容；
Schur公式、非负矩匹配、正交投影和两点误差下界不是本项目原创数学。
Kaminski/Bryson/Schmidt 1971平方根综述书目已找到但DOI正文不可访问；未列作已精读依据。
Särkkä/Svensson 2023书PDF访问失败，不据二手摘要补写其方法细节。

## 13. 2026-09-13 M04响应共享补核

| ID | 来源/版本 | 读取与本地对应 |
|---|---|---|
| M04-R1 | Yao, Müller & Wang (2005), *Functional Data Analysis for Sparse Longitudinal Data*, JASA100(470),577–590；[作者PDF入口](https://utstat.utoronto.ca/fyao/2005-jasa.pdf)、[DOI](https://doi.org/10.1198/016214504000001745) | A：2026-09-13作者PDF的搜索展示首面/摘要；直接PDF请求两域均超时，DOI403，未标全文精读。继承总体函数矩、观察误差、个体条件预测的思路。M04仅五离散条件、独立机会半组交叉矩及标准谱分解，没有复现随机稀疏时间平滑FPCA、渐近定理或原置信带。未复制代码。 |
| M04-R2 | Mischel & Shoda (1995), *A Cognitive–Affective System Theory of Personality: Reconceptualizing Situations, Dispositions, Dynamics, and Invariance in Personality Structure*, Psychological Review102(2),246–268；[DOI](https://doi.org/10.1037/0033-295X.102.2.246)、[出版社预览转载](https://www.researchgate.net/publication/15459280_A_Cognitive-Affective_System_Theory_of_Personality_Reconceptualizing_Situations_Dispositions_Dynamics_and_Invariance_in_Personality_Structure) | A：2026-09-13预览首页/摘要，平均水平与跨情境模式的对象区分；不把本地有限函数核当整套CAPS机制，也不以其理论存在作为本包心理效度。 |
| M04-R3 | Álvarez等(2012)，沿用M0405-R1；[作者页面](https://inverseprobability.com/publications/alvarez-vector12.html) | 本轮A复核作者书目/摘要，方法沿用已登记章节，不重复冒称新全文审查。共享的响应核可以带符号依赖并有个人系数；五条件PSD矩阵不代表任意新语义条件核。 |

对应[响应共享报告](../reports/grain_measurement_foundation/RESPONSE_SHARING_REPORT.md)。
正面与阴性均保留：响应交叉世界有用，平滑世界随机斜率已相当，条件特有世界不共享。
均值中心化产生负协方差是标准投影恒等，不是新心理学发现；新内容在来源文本单位、
同人条件支持、共享模式/个人系数及目标相关匹配的制作步骤。没有把任意PSD核或BLUP署名为原创。

## 14. 2026-09-13 M01/M09证据身份与话轮依赖

| ID | 来源 | 本轮读取/继承与非复现范围 |
|---|---|---|
| M0109-R1 | Sauri & Pustejovsky (2012), *Are You Sure That This Happened? Assessing the Factuality Degree of Events in Text*, CL38(2),261-299；[原文](https://aclanthology.org/J12-2002.pdf)，DOI10.1162/COLI_a_00096 | 既有W88续读：PDF来源视角、嵌套上下文及来源角色的段落/定义，非本轮全39页精读。事实性不能离开来源/模态；本地保留这些字段但未复制646谓词词典或完整De Facto算法。 |
| M0109-R2 | Bunt, Petukhova, Gilmartin, Pelachaud, Fang, Keizer & Prevot (2020), *The ISO Standard for Dialogue Act Annotation, Second Edition*, LREC,549-558；[原文](https://aclanthology.org/2020.lrec-1.69.pdf) | F，选读sections1-3，尤其2.1功能/反馈依赖、2.2修正的reference segments。继承角色/前文关系与多功能，不把整话轮候选地址称精确依赖解析；本地十查询不是完整ISO实现。 |

对应[实测报告](../reports/grain_measurement_foundation/EVIDENCE_IDENTITY_REPORT.md)。
该两项理论提供接口原则，不保证固定NLI能正确解析：本轮具体事实正常而抽象身份
读者失败，原样保存。行空间识别、机会标准化与约束最小二乘属标准数学，贡献不归于
发明这些公式，而在测量对象分账、具体证据到个人目标的集成及实测压缩/改写反例。

## 15. 2026-09-13 具体命题/条件与有结构缺失

| ID | 来源 | 读取范围与继承边界 |
|---|---|---|
| M0109-R3 | Universal Dependencies v2，[xcomp](https://universaldependencies.org/u/dep/xcomp.html) | 读完整HTML主语控制定义、继承主语准则与secondary predicate例。继承候选外部论元控制；不是所有xcomp都继承上一主语，也不是完整语义共指。 |
| M0109-R4 | Universal Dependencies v2，[advcl](https://universaldependencies.org/u/dep/advcl.html) | 读完整HTML定义及时间/条件/目的例；保留原文条件从句，不将该边直接定义成一种心理条件。 |
| M0109-R5 | Manski & Tabord-Meehan (2017), *Evaluating the Maximum MSE of Mean Estimators with Missing Data*, The Stata Journal17(3),723-735，[出版社](https://journals.sagepub.com/doi/10.1177/1536867X1701700311) | A：读取摘要/书目，缺失造成非点识别且无普遍最优点估计。未重现wald_mse命令；本地有限材料上下界由有界和式直接推导，非本研究新定理。检索到Manski部分识别综述的Elsevier页面403，不记全文已读。 |

M0109-R1本轮仅核ACL书目；R2沿用前轮对话功能依赖，不重复冒称全文精读。
对应[命题回应报告](../reports/grain_measurement_foundation/PROPOSITION_RESPONSE_REPORT.md)：
实际解析选择性漏读、个人计分分母偏移及原文角色分槽是本地连接，标准数学继承。

## 16. 2026-09-13 对话双账、省略与版本

| ID | 来源 | 读取与实际继承 |
|---|---|---|
| M0109-R2续 | Bunt等2020，[ISO对话标注](https://aclanthology.org/2020.lrec-1.69/) | 本轮复核书目/摘要并沿用2.1功能依赖、2.2修复reference segment。语义内容和对话功能分层可避免把一轮压成一个类型；本地规则不是完整ISO分类器。 |
| M0109-R6 | Dalrymple, Shieber & Pereira (1991), *Ellipsis and Higher-Order Unification*, Linguistics and Philosophy14,399–452，DOI10.1007/BF00630923；[作者预印本](https://arxiv.org/abs/cmp-lg/9503008) | A：本轮核摘要/书目；1995为预印本提交，不改原创年份。继承省略需受语义前件约束的动机；本地只有明确前问/previous answer及辅助词极性替换，没有高阶合一、量词作用域或完整省略理论的实现。 |

M02-R1/R2已登记的来源/修订思路继续使用。[双账实测](../reports/grain_measurement_foundation/DIALOGUE_CLAIM_LEDGER_REPORT.md)
发现原评价混用了通用请求和任务求助范围；保留原结果及结果后目标分账，不将范围改正
冒称新模型泛化提升。话语多功能是继承原则，本地贡献在其与证据更新/个人计分的连接。
