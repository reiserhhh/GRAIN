# GRAIN 文献审计：来源与阅读台账

日期：2026-09-09。对应[综合审计与数学比较](GRAIN_LITERATURE_METHOD_ALIGNMENT_AUDIT_20260909.md)。

## 1. 范围、方法与阅读标记

目标不是计算引用数量，而是核对四个接口：真实作者证据→心理指标；文本→模型中间表征；中间表征→实际判断；大模型判断→可执行小程序。

检索方向包括 `personality inference activation probing`、`emotion inference appraisal mechanistic`、`persona steering situation`、`user representation PANDORA synthetic`、`causal abstraction nonlinear alignment`、`Sobolev Jacobian distillation`、`vector-valued kernels`、`Hodge ranking`、`within between person`。来源优先正式会议/期刊、作者稿与作者研究页面；搜索引擎摘要只用于定位或明确标注的摘要级条目。近期检索包含 2026 年夏季作品，不将预印本当成同行评审共识。

阅读标记：

- **M**：已打开全文，阅读本任务相关的方法、公式或结果段；不表示逐页阅读全部附录或独立复现证明。
- **A**：已核对作者/出版方/索引提供的摘要与元数据；不据此推演未读公式。
- **R**：作者发布的研究说明，非独立同行评议。
- **L**：访问受限，只保留检索可见的原始来源片段，不能充当核心定量依据。

以下 42 项不包含随机搜索结果或重复版本。原始研究优先；标作方法综述的条目仅用于其方法体系，不当独立实证复现。文献结果与本地结果的数据、标签、模型和指标不同，不建立跨论文准确率排行榜。

## 2. 真实文本、作者与人格推断

| ID | 文献与一手入口 | 状态 / 阅读 | 在本次比较中的用途 |
| --- | --- | --- | --- |
| S01 | Schwartz et al. (2013), [Personality, Gender, and Age in the Language of Social Media: The Open-Vocabulary Approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC3783449/) | PLOS ONE；A | 开放词汇/主题与作者差异；不是内部 LLM 机制研究。 |
| S02 | Gjurković et al. (2021), [PANDORA Talks: Personality and Demographics on Reddit](https://aclanthology.org/2021.socialnlp-1.12/) | SocialNLP；A，本轮不重新复核全部表格 | 数据与预测/辅助标签/结构分析的任务区分；不拿旧基准替代本轮目标。 |
| S03 | Soni et al. (2022), [Human Language Modeling](https://aclanthology.org/2022.findings-acl.52/)；[全文](https://aclanthology.org/2022.findings-acl.52.pdf) | Findings ACL；M，任务定义与历史状态 | 同作者多文档历史的条件化，先于 GRAIN 存在的具体方案。 |
| S04 | Peters & Matz (2024), [Large language models can infer psychological dispositions of social media users](https://academic.oup.com/pnasnexus/article/doi/10.1093/pnasnexus/pgae231/7692212) | PNAS Nexus；M，材料/提示/聚合方法 | 真实文本关联的背景，不据其平均 r 推断内部因果机制。 |
| S05 | Yang & Oshio (2026), [Attachment towards generative AI influences self-AI agreement in personality reports](https://doi.org/10.1111/ajsp.70102)；[作者机构摘要](https://waseda.elsevierpure.com/en/publications/attachment-towards-generative-ai-influences-self-ai-agreement-in-/) | Asian Journal of Social Psychology 29(3)；A | 用户所提论文；自我开示/关系与评分一致性的观察性问题。 |
| S06 | Zhu, Jin & Coifman (2025), [Investigating LLMs in Inferring Personality Traits from User Conversations](https://arxiv.org/abs/2501.07532) | 预印本；A | 条目先行再聚合的候选对照；不把摘要中的改善视作本项目复现。 |
| S07 | Zhu, Jin & Coifman (2025), [Can LLMs Infer Personality from Real World Conversations?](https://arxiv.org/abs/2507.14355) | 预印本；A | 真实访谈、稳定性与效度的区别；与 S06 同研究团队，不预设数据独立。 |
| S08 | Härtel et al. (2026), [In the Blink of an AI: Exploring LLMs' Capability to Infer Traits From LinkedIn](https://link.springer.com/article/10.1007/s10869-025-10087-8) | Journal of Business and Psychology；A，出版方内容 | 线索效度与使用、范围收缩；仓库原有 lens model 线索，不是此次首次提出。 |
| S09 | Bi et al. (2025), [Leveraging the Dual Capabilities of LLM: LLM-Enhanced Text Mapping Model for Personality Detection](https://ojs.aaai.org/index.php/AAAI/article/view/34517)；[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/34517/36672) | AAAI 39(22)；M，跨注意力与对比损失 | 文本到人格标签语义的直接监督先例；ETM 不是无标签机制蒸馏。 |

## 3. 心理概念、用户表征与人格控制

| ID | 文献与一手入口 | 状态 / 阅读 | 在本次比较中的用途 |
| --- | --- | --- | --- |
| S10 | Tak et al. (2025), [Mechanistic Interpretability of Emotion Inference in Large Language Models](https://aclanthology.org/2025.findings-acl.679/)；[PDF](https://aclanthology.org/2025.findings-acl.679.pdf) | Findings ACL；M，probing、appraisal、干预与相关附录 | 最接近“证据→心理中介→判断”；区分可读性与实际使用，注意正确样本筛选。 |
| S11 | Shu, Singh & ElSherief (2026), [Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders](https://arxiv.org/html/2604.25866v2) | 预印本，2026-06 修订；M，方法与效应分析 | 稀疏/分布式情绪特征与共享结构；旧题名版本不重复计数。 |
| S12 | Frising & Balcells (2025/2026), [Linear Personality Probing and Steering in LLMs: A Big Five Study](https://arxiv.org/html/2512.17639v2) | 预印本，2026-01 修订；M，数据生成/读出/干预 | 虚构角色与模型问卷的方向，不能冒充真实作者预测；按标签聚合也改变了单位。 |
| S13 | Chen et al. (2024), [Designing a Dashboard for Transparency and Control of Conversational AI](https://arxiv.org/html/2406.07882v3) | TalkTuner 作者稿；M，方法及 Appendix C | 合成用户属性探针→真实 PANDORA 人口学测试；read/control 位置不相同。 |
| S14 | Chen et al. (2025), [Persona Vectors: Monitoring and controlling character traits in language models](https://www.anthropic.com/research/persona-vectors) | 作者研究说明；R | 模型行为写入方向，不是人类 Big5 的直接读取器。 |
| S15 | Sofroniew et al. (2026), [Emotion Concepts and their Function in a Large Language Model](https://arxiv.org/abs/2604.07729) | 预印本；A | 模型中的情绪概念与行为作用；需明确它指模型角色、文本角色还是作者。 |
| S16 | Wei et al. (2026), [Beyond Static Personas: Situational Personality Steering for Large Language Models](https://aclanthology.org/2026.findings-acl.958/)；[方法稿](https://arxiv.org/html/2604.13846v1) | Findings ACL；M，§3 的识别/检索/控制公式 | IRIS 提供条件特异方向库；主题、阈值和读写方向不能不经比较照搬。 |
| S17 | Hoppe et al. (2026), [Controllable and explainable personality sliders for LLMs at inference time](https://arxiv.org/html/2603.03326v1) | 预印本；M，§3–§6 | 干预后分布上的后续 probe；有效减干扰与严格正交/独立不是一回事。 |
| S18 | [From Traits to Circuits: Toward Mechanistic Interpretability of Personality in LLMs](https://openreview.net/pdf?id=XMGzodoTwV)；相关期刊入口 [A mechanistic interpretability perspective on personality in large language models](https://doi.org/10.1016/j.ipm.2026.104962) | OpenReview 稿全文请求受阻；期刊摘要可检索；L/A | 记录人格回路这一直接邻近方向，不引用未核对的性能或证明。两入口的版本关系待全文比对，不计两次。 |
| S19 | Courtis & Hu (2026), [Mechanistic Personality Analysis of LLMs Steering Personality via Latent Feature Interventions](https://arxiv.org/abs/2606.28770) | 预印本；A；页面注明写于2024、2026上传 | SAE/对比方向的人格控制先例。上传新不等于方法在2026首次产生。 |

## 4. 机制定位、可解释性争论与计算读出

| ID | 文献与一手入口 | 状态 / 阅读 | 在本次比较中的用途 |
| --- | --- | --- | --- |
| S20 | Marks et al. (2024/2025), [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/html/2403.19647v3) | ICLR 2025；M，方法说明 | 稀疏特征回路、跨样本回路结构；“群体归纳回路”不是本项目独有概念。 |
| S21 | Mueller et al. (2025), [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/html/2504.13151v1) | ICML 2025；M，两个任务轨与比较 | 因果变量与回路定位分开，不能假设 SAE 始终优于基础组件。 |
| S22 | Arad et al. (2025), [Findings of the BlackboxNLP 2025 Shared Task: Localizing Circuits and Causal Variables in Language Models](https://aclanthology.org/2025.blackboxnlp-1.32/)；[全文](https://arxiv.org/html/2511.18409v1) | BlackboxNLP；M，方法/§4.4/讨论 | 小型非线性与集成的进展，以及解释器自己学会变量的风险。 |
| S23 | Geiger et al. (2025), [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://www.jmlr.org/papers/v26/23-0058.html) | JMLR；A，本轮未逐条重验全文定理 | 对应干预下的高低层计算关系；仓库已有基础，不将小程序拟合直接等同因果抽象。 |
| S24 | Makelov et al. (2024), [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://openreview.net/pdf?id=Ebt7JgMHv1) | ICLR；A/论文定位 | 干预有效与自然使用的区别；不作“所有 patch 无效”的推论。 |
| S25 | Wu et al. (2024), [A Reply to Makelov et al. (2023)'s "Interpretability Illusion" Arguments](https://arxiv.org/html/2401.12631v1) | 预印本回应；M，争论与实验解释 | 与 S24 一起读取；不同定义与训练/测试设置会改变判定。 |
| S26 | Sutter et al. (2025), [The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dbb98528c9870377f3f0d133aae6050b-Abstract-Conference.html)；[作者稿](https://arxiv.org/html/2507.08802v1) | NeurIPS；M，定义/结论/假设说明，未独立重证附录 | 高容量映射可以替代任务计算；保留复杂度与简单模型对照，不否定所有非线性。 |
| S27 | [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) (2026) | 作者研究报告；M，Jacobian lens 方法 | 平均 Jacobian 读出可语言化内容；非当前例子的完整推理轨迹，仓库已有引用。 |

## 5. 数学和统计方法

| ID | 文献与一手入口 | 状态 / 阅读 | 在本次比较中的用途 |
| --- | --- | --- | --- |
| S28 | Park, Choe & Veitch (2024), [The Linear Representation Hypothesis and the Geometry of Large Language Models](https://proceedings.mlr.press/v235/park24c.html) | ICML；A，本轮以官方理论摘要核对 | 读写空间与内积的选择。报告的坐标变换等式为独立基础推导，不冒充全文复证。 |
| S29 | Park et al. (2026), [The Information Geometry of Softmax: Probing and Steering](https://arxiv.org/html/2602.15293v2) | 作者稿，arXiv注明ICML 2026；M，指数族与对偶几何部分 | 输出相关度量；部署前仍需明确近似、秩与成本。 |
| S30 | Czarnecki et al. (2017), [Sobolev Training for Neural Networks](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html) | NeurIPS；A；仓库已引用 | 值+导数监督的原始路线；不是 GRAIN 首创。 |
| S31 | Srinivas & Fleuret (2018), [Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html)；[PDF](https://proceedings.mlr.press/v80/srinivas18a/srinivas18a.pdf) | ICML；M，目标函数与 Taylor 论证 | 有限编辑、局部导数和噪声增强之间的适用条件；曲率项在本报告独立展开。 |
| S32 | Jiang, Lim, Yao & Ye (2011), [Statistical ranking and combinatorial Hodge theory](https://arxiv.org/abs/0811.1067)；[全文](https://arxiv.org/pdf/0811.1067) | Mathematical Programming；M，梯度/环流定义与分解 | 差分图的锚定、循环和可识别方向；不对由同一标量导出的差分做循环“发现”。 |
| S33 | Micchelli & Pontil (2005), [On Learning Vector-Valued Functions](https://doi.org/10.1162/0899766052530802)；[作者早期稿](https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/micchelli.pontil.pdf) | Neural Computation；A，摘要与算子值核范围 | 共享+特异任务核的标准来源；不是 JMLR 的另一篇 kernel selection 论文。 |
| S34 | Constantine, Dow & Wang (2013/2014), [Active subspace methods in theory and practice: applications to kriging surfaces](https://arxiv.org/abs/1304.2070) | 预印本及 SIAM J. Scientific Computing 期刊版；A | 响应梯度空间而非总方差 PCA；仓库已有路线。 |
| S35 | Khemakhem et al. (2020), [Variational Autoencoders and Nonlinear ICA: A Unifying Framework](https://proceedings.mlr.press/v108/khemakhem20a.html) | AISTATS；A | 附加条件可支持潜变量识别，但有明确的生成模型假设。 |
| S36 | Belrose et al. (2023), [LEACE: Perfect linear concept erasure in closed form](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html) | NeurIPS；A，线性保证范围 | 可作诊断，不作为删除全部 topic/situation 的生产处理；线性不可读不等于无信息。 |
| S37 | Curran & Bauer (2011), [The Disaggregation of Within-Person and Between-Person Effects in Longitudinal Models of Change](https://doi.org/10.1146/annurev.psych.093008.100356)；[作者PDF](https://curran.web.unc.edu/wp-content/uploads/sites/6785/2015/03/CurranBauer2011.pdf) | Annual Review of Psychology；M，§中分解与时间趋势公式 | 多层均值与个体内变化分开；不能将编辑反应当作人的时间变化。 |

## 6. 新方法、讨论与观察清单

| ID | 文献与一手入口 | 状态 / 阅读 | 在本次比较中的用途 |
| --- | --- | --- | --- |
| S38 | Sun, Wang & Hu (2026), [The Price of Amortized Inference in Sparse Autoencoders](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76358e88c3ebd38ba58af9d0d84b71e4-Abstract-Conference.html) | ICLR；A | 稀疏度/重建/特征吸收等取舍；不把 SAEs 视为现成真因子分析器。 |
| S39 | Dalili & Mahdavi (2026), [Subspace-Aware Sparse Autoencoders for Effective Mechanistic Interpretability](https://arxiv.org/html/2606.06333v1) | 2026-06 预印本；A，方法概述 | 多维块候选；其理论条件未独立复核，不作为“所有特征一定多维”的证据。 |
| S40 | Chen, Plaat & van Stein (2026), [How Does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding](https://ojs.aaai.org/index.php/AAAI/article/view/40281) | AAAI；A | 推理 token 的计算作用，算术任务不直接转移为人格推理机制。 |
| S41 | Lin & Liu (2026), [Position: Mechanistic Interpretability Must Disclose Identification Assumptions for Causal Claims](https://arxiv.org/abs/2605.08012) | 2026-05 立场预印本；A | 记录当前领域争论，不作已达成共识或实证定理。 |
| S42 | Ha (2026), [An interviewer–evaluator–judger LLM framework for text-based personality inference](https://www.nature.com/articles/s41598-026-62333-5) | Scientific Reports，出版方标记2026-07-23在线接受稿；A/L，全文抓取失败 | 最近的访谈→评价分工先例；不引用未读实验数值，不据此推断内部机制。 |

## 7. 原公式对照索引

这不是把别人的整篇方法搬来，而是明确其数学操作与本地接口。

| 文献族 | 论文操作的统一记号 | 本地对应 / 差距 |
| --- | --- | --- |
| ETM / S09 | \(\operatorname{softmax}(QK^\top/\sqrt d)V\)；正确标签语义的对比目标 | 标签/条目语义对齐是可运行对照；真实监督与源构造必须另列。 |
| 情绪 appraisal / S10 | \(\widehat a_j=v_j^\top h+b_j\)，再干预表征观察判断 | 当前有文本编辑，尚需把编辑链接到实际中介。 |
| IRIS / S16 | 情境内正负激活概率差；\(e_t=\operatorname{softmax}_t(\cos(v_t,\Delta v))\)；按 \(e_t\) 加权干预 | 局部条件算子的具体先例；不是无LLM读取，也不是人格测量方程。 |
| 顺序控制 / S17 | 在 \(h+\alpha v_1\) 的分布上再学后续方向 | 干预会改变其他方向作用；不把训练后的近正交写成独立性定理。 |
| Jacobian lens / S27 | 平均下游 Jacobian 变换后再词表读出 | 与预测上层状态的回归不同；本项目可先用标量 VJP。 |
| Sobolev / S30–31 | 值误差 + 导数误差 | CON 是指定有限编辑方向的割线版本，不是完整 Jacobian。 |
| Hodge / S32 | \(\min_s\|Ds-y_{edge}\|^2\)，残差再分循环部分 | 当前图不含环；标量导出的边本来就无循环。 |
| 向量值核 / S33 | \(K(x,x')\) 为输出任务间的正半定算子 | 当前同一查询三分量不等于全查询共享/特异模型。 |
| 主动子空间 / S34 | \(C=E[\nabla f\nabla f^\top]\) | 高方差未必高作用；理论已有，不能把它当已完成的干预路径。 |
| 多层统计 / S37 | \(\beta_B\bar z_u+\beta_W(z_{ut}-\bar z_u)\) | 人际位置与条件变化分开后再连接，不能用差分单独恢复作者次序。 |

## 8. 阅读后保留的空白

没有本轮确认的论文完整替我们交付“真实作者、多量表共享/特异路径、保持群体参照和条件响应、无需完整 LLM 的新文本运行”全部接口。但这一判断来自定向检索，不构成无先例证明。

最需要转化成代码对照的是 S09/S10/S16；最需要约束公式解释的是 S26/S31/S32/S37。S38–S41 保留为工具选择和讨论背景，不将它们扩成新的无界研究支线。

下一轮引用前仍须完成：S18/S42 的全文获取、预印本最新版本核对、外部仓库许可与接口兼容性。它们不阻碍当前三个核心接口的开发。
