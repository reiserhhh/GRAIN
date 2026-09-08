# GRAIN 文本到特质计算路径：文献与理论拼图

现阶段最有价值的方向，是在受控文本世界中分离计算机制，以已经冻结的目标
函数为终点，建立局部、条件化的路径图，再持续回到真人文本检验。完整中间
状态重建与有效投影重建是两条并存路线；受控世界为两条路线提供共同实验场。
它不是用模拟人格标签训练一个新黑箱，也不是以二维可视化代替计算机制。

## 1. 范围与证据起点

本综述对应2026-09-09的待研究清单。文献整理和计划已经完成，模型实验仍为
PENDING。具体工作包、资源和恢复规则见[执行计划](GRAIN_PENDING_PUZZLE_EXECUTION_PLAN_V2.md)。
研究对象是基于自然语言的心理推论方法，不是发布新的临床量表。

本地已有三类证据，分别回答不同问题：

| 已有证据 | 实际含义 | 本轮设计后果 |
|---|---|---|
| M28九函数在12名独立作者上的水平一致性r=.776–.991，证据响应r=.550–.976 | 冻结模型的部分作用可由参照矩阵运输 | 新实验可把这些固定函数当成明确的计算终点 |
| M30B独立PANDORA参照函数在46人相谈中保留状态关联r=.422/.564/.555 | 构造材料与外部测量人群可以分离 | 值得检验更可控的合成参照，不必先收真人训练标签 |
| P1H/I人物绑定、P1R/S/T顺序与行为对比结果 | 人物归属、语义、条件和答案编码均可能改变运算 | 世界生成器必须操纵关系，而非只替换人格形容词 |

证据入口：[M28](../reports/grain_vector_mechanisms/ALL_FACTOR_FUNCTIONS_REPORT.md)、
[M30B](../reports/grain_vector_mechanisms/NARROW_VECTOR_FUNCTION_REPORT.md)、
[绑定](../reports/grain_vector_mechanisms/QUERY_ROLE_AND_BINDING_REPORT.md)、
[顺序与线索](../reports/grain_vector_mechanisms/ORDER_GAIN_AND_CUE_REPORT.md)。
报告中的历史“待跑”文字由最新[主线状态](GRAIN_MAINLINE_DISCOVERY_PLAN_V3.md)覆盖。

PANDORA参照仍是真实文本；上述结果不是已经完成的“纯合成构造→真人验证”。
模型前段仍提供上下文化表示，因此也不是LLM-free端到端成绩。这两个缺口
正是新实验要接上的部分，不抹去已有函数与真实关联。

下一轮需分开两种操作：合成文本测试旧真人参照公式，以及只用合成参照构造
新公式后返回真人文本。后者设置`R_synth`，与旧`R_real`同协议比较；两者
都保留冻结观察器的预训练知识。不能把第一种结果当作第二种已经成功。

## 2. 受控世界与因果抽象

Geiger等的因果抽象框架提供了一个适合本项目的比较单位：不要求神经元与
心理概念一一对应，而是提出高层变量或程序，检验替换这些变量后，神经系统
与高层程序是否产生对应变化。其2021年工作在具有已知组合结构的自然语言
推理任务中应用了这种方法；2025年框架进一步统一了多类机制干预。[^1][^2]

Singh等在受控合成任务中分离出归纳头形成所需的子运算及其交互。可借鉴的是
“把纠缠变量拆开，再观察组合”，不是推断人格一定由归纳头实现。其研究涉及
训练过程，我们下一轮只研究冻结观察器，不照搬训练规模。[^3]

建议的世界变量为行为者、行为对象、可选择机会、自愿/被迫、否定/引用、
事件时间、信息可靠性和证据组合。它们有作者设定的事件真值。由这些事件
推断出的外向性或依恋倾向则是待解释的模型输出，不能被生成器直接宣布为
真人心理真值。高层模型先定义语义关系，不预先规定所有人格推论规则。

一张地图上的边应记录“改变该关系，哪些状态和目标分数怎样变化”。节点
应尽可能对应可重复的计算作用；同义词接近、二维点云重叠、同名注意力头
都只能帮助定位，不能单独定义路径。

## 3. 路线A：从读出反向寻找必要运算

注意力的QK子计算决定信息路由，OV子计算决定被读取信息如何写入后续表示。
前馈网络、残差、归一化和位置机制也属于可能的依赖。我们的末位置状态Q
不是注意力QK术语中的查询矩阵。[^4]

EAP以梯度近似干预贡献，适合作为便宜的候选排序；EAP-IG以路径积分缓解
局部梯度近似问题。Hanna等明确说明，找到相似的边集合不等于复现同样的
功能，所以本项目应先筛候选，再针对少量候选做实际替换和组合验证。[^5][^6]
不应先对所有头、所有层、所有目标开全量穷举。

拟用的局部边分数是候选而非精确因果效应：

```
score(edge e) ≈ <activation_e(donor)-activation_e(recipient), gradient_e(f_q)>
```

梯度终点优先使用冻结f_q，不必每次都跑生成后缀。发现有用边后补原生判断
对照，以区分“驱动我们的近似函数”与“驱动原生模型”的作用。

若MLP成为主要瓶颈，transcoder文献提供局部替代的思路：拟合稀疏的输入到
输出变换，分离输入依赖与固定权重部分。它仍然是学习的模型，不能因为稀疏
就称为显式心理定律；默认不新训整套transcoder。[^7]

重要的代码缺口：现有`prefix.capture_prefix`使用`torch.no_grad()`并导出
detached NumPy缓存，且仍载入整套模型权重。它不能直接提供上游梯度，也不是
独立可部署的裁剪模型。必须新增有作用域的可微前缀采集器；旧实验接口不改。

## 4. 路线B：重建有效投影，而非全部中间向量

现有计分的代数结构为：

```
z_q1 = <G_q, P_q-P_Rq>
z_q2 = g_q' (Q_q-Q_Rq)
f_q = b_q + z_q1 + z_q2
```

所以独立程序只要能计算这两个量，就足以复现这个固定函数。这个结论是
当前线性读出的代数性质，不是“人格只需要两维”，更不代表原始文本经两次
加权平均即可生成这两个量。

IG提供了连接变化与导数的依据；active subspaces则按输出敏感度而非输入
方差寻找方向，符合“保留真正被读出的部分”的目标。[^8][^9][^10] 可以在有
共同坐标的较早层表示v上研究：

```
J_q(v) = derivative of (z_q1,z_q2) with respect to v
C_q = E_pi[J_q(v)' J_q(v)]
```

pi是明确的场景/干预分布，不能偷偷等同于全人群。不同长度文本不能未经对齐
直接堆token梯度。优先按语义字段和角色对齐，或在同一局部布局内计算；谱尾
大就保留较高秩或分条件算子，不把分布式信号强压成一条方向。

不能在末端固定线性读出上计算梯度秩，再把必然低秩当成新发现；有研究价值的
是其**上游拉回**对证据、条件和位置如何依赖。文献中的误差界需要相应输入
分布和光滑性条件，这里不直接赋予离散自然语言一个自动保证。

符号回归适合在先识别出低复杂度模块之后尝试。Cranmer等先利用结构归纳偏置
分离模型模块，再提取符号式，研究场景主要是物理系统；不能据此保证任意LLM
前缀都能被短公式替代。[^11] 第一候选应是可解释的门控、加权和、成对关系
与少量条件交互；小神经网络只能作为有成本记录的对照，不冒称机制已解释。

对替代误差保留两部分，避免把源头问题都算在新程序身上：

```
s_native - s_independent
  = (s_native - f_q(P,Q)) + (f_q(P,Q) - s_independent)
```

两项可抵消。分别检查符号、协方差和自然编辑响应，不只看最后总误差。

## 5. 查询身份、编码与概念几何

M28的E与EI证据响应相关仅.047，提示语义接近不等于计算等价。需要把构念
措辞、判断对象、时间窗、答案编码分开操纵。改变问题token后，原G矩阵不再
具有相同坐标含义；新问法需要独立参照构造或明确的对齐检验。

Park等区分输入上下文与输出词的线性表示，并说明几何依赖于所选内积。
这使“离外向性这个词更近”不能直接充当导航终点。[^12] Garg等2026年COLT
工作进一步区分线性表示与线性可读性；在其稀疏特征模型下，可线性解码是
比可线性存储更强的要求。这里可以借用这个区分，不能把其容量界当成GRAIN
实际需要的维数或样本数。[^13]

Function Vectors在上下文学习任务中展示了任务函数可以由激活向量携带，并
通过干预参与新输入上的运算。它启发我们寻找查询可复用成分，但不保证不同
人格查询共享一个向量，也不保证脱离完整模型后仍能独立运行。[^14]

## 6. 局部图册、共享路径与群体结构

早期M20/M21已有教训：某个有限函数能保留作者关联，却不一定跟随真实内部
作用。因而新地图应把计算效应和人类指标分列，而非择其一作为万能正确性。
参照增多也不保证平均更好；可能需要按条件区分局部图。

候选的局部拼接是：

```
f_q(x,c) ≈ sum_j pi_j(x,c) f_qj(x,c), pi_j >= 0, sum_j pi_j = 1
```

这是拟研究的门控模型，不是已经证明的全局势函数。门控只能用文本/条件和
独立构造数据；不能用真人标签选择图。评价图间重叠区域与自然编辑切换时的
连续性，比较分区和简单平均，不先假设模型一定存在统一光滑导航场。

不同查询的P token行不一致，故共享路径优先在共同干预集合上比较：

```
K(q,r | c) = Cov_eta[Delta f_q(eta), Delta f_r(eta)]
```

现有M31可作基础，但相关或相同头集合只是候选证据。下一步应验证同一内部
替换是否同时改变两个目标，并保留正负与交互项。若按GRAIN作者群体分层，
先用一份文本定义群体，再用另一份文本检验响应结构；不按量表标签分群后
反过来宣称找到了独立人格路径。

CKA等表示相似性工具可以处理某些坐标变换问题，但研究的是表征相似，不是
两个系统必然计算同一程序。[^15] 功能交换和组合响应仍是更接近本任务的桥梁。

## 7. 状态、条件与固定参照

Funder的RAM把判断过程中的线索相关性、可获得性、检测和利用区分开；这与
本项目“材料→表示→函数→外部关联”的分层一致。[^16] CAPS强调个人响应
可以依情境改变；Fleeson用状态分布连接个体差异与时变表现。[^17][^18]
这些理论支持保留条件与时点，而不是要求同一作者在所有材料下同分。

Peters等的对话实验显示不同互动条件的人格推断关联有明显差别。它提供材料
机会层的动机，不证明本项目的具体注意力机制，也不是我们Qwen系统的预期
精度。[^19] 我们当前相谈状态关联强于课题回答的结果，还涉及不同终点时点，
应按相同criterion的交叉比较和已有pre/mid/post记录进一步分账。

固定参照必须区分两件事：用于构造G/g的参考文本，与用于描述作者相对位置
的常模总体。二者不是同一个样本数要求。新样本到来不应改变已报告分数；
但换人群/条件时，参照位置的解释可以改变。MTMM提供完整交叉关系的组织
方式，不要求这46人足以拟合复杂的潜变量模型。[^20]

## 8. 共享前缀与运行独立性

因果Transformer在完全相同的token前缀和相同运行条件下，可以复用该前缀的
计算。SGLang的RadixAttention展示了系统层面的KV复用。[^21] 可借鉴其前缀树
思想，但不必为此迁移整个运行时；先验证同一作者文本、不同查询的缓存边界。
查询之前若已有trait-specific指令，就不能把整个文本段当成共同前缀。

省去重复推理是工程收益，不等于发现心理机制，也不等于LLM-free。运行依赖
应分层记录：完整前缀、提取的局部神经程序、独立低维算子；另记录是否仍需
原tokenizer、词表embedding、复制权重或外部解析器。构造时借助LLM与运行时
不需要LLM是不同判据。

## 9. 暂缓扩展仍有文献接口

人格提示写入与作者人格读取的互逆问题保留。Persona Vectors显示一些模型
行为特征可通过激活方向监测和控制，适合作为“写入端”的材料；它研究的是
模型行为，不是测出真人人格，也不能直接推出生成和推断互为逆映射。[^22]
恢复此分支前先得到至少一条可工作的文本读取局部程序，再检查双向干预。

跨模型、跨语言、长期真人重测和临床部署也保留为扩展，不作为当前局部机制
发现的入口门槛。现有英文参照→日语文本结果足以支持继续尝试，但表示相近
不自动消除语言、文化和测量条件差异。历史LoRA、旧lockbox和大网格不自动重启。

## 10. 本轮文献改变了哪些决策

| 决策 | 采用的工作方式 | 不采用的捷径 |
|---|---|---|
| 如何造世界 | 已知事件关系，未知人格归因；场景族分离 | 生成“外向80分”的文本，再用同一提示标签证明外向 |
| 如何找路径 | 目标梯度筛候选、少量真实替换、局部组合 | 注意力权重大或图像漂亮即视为机制 |
| 如何压缩 | 比较P/Q重建与有效投影重建 | 因输出是两项投影就假设原文处理只需两维 |
| 如何归纳大地图 | 条件化局部图与有符号功能共享 | 把所有方向平均成一条人格轴 |
| 如何返回心理学 | 同人外部指标、时点、材料机会逐层连接 | 模型一致性高就直接命名心理因子 |
| 如何控制投入 | 一块有用局部地图后就做现实哨点检查 | 等所有34目标、所有场景都画完才第一次看现实 |

理论上的新增组织是一个可证伪的组合：**受控关系世界 → 目标条件化作用图 →
独立状态/投影程序 → 作者参照与外部连接**。因果抽象、梯度归因、子空间方法
本身都有先行研究；原创性若成立，应体现在这个组合对人格推论的具体实现、
发现的关系和可重复的跨材料计算，而不是重新命名既有定理。

## 文献

以下为原始论文或作者研究材料。正文机制对应只使用可访问的摘要、方法段落和
明确结论；摘要级来源不用于推导其全文未核实的定理细节。公开文献不包含本地
参与者材料，不能替代本地已核对的结果账本。

[^1]: Geiger, A., et al. (2025). [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://www.jmlr.org/papers/v26/23-0058.html). JMLR, 26(83), 1–64. 正式版理论范围；不以旧预印本年份代替正式发表年份。
[^2]: Geiger, A., et al. (2021). [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997). NeurIPS 2021. 受控NLI与interchange intervention实例。
[^3]: Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C. Y., & Saxe, A. M. (2024). [What needs to go right for an induction head?](https://proceedings.mlr.press/v235/singh24c.html). ICML, PMLR 235, 45637–45662. 受控合成任务及子回路交互。
[^4]: Elhage, N., et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). 作者技术研究。QK/OV分解；不是人格实验。
[^5]: Syed, A., Rager, C., & Conmy, A. (2024). [Attribution Patching Outperforms Automated Circuit Discovery](https://aclanthology.org/2024.blackboxnlp-1.25/). BlackboxNLP, 407–416. 2023预印本的会议版本。
[^6]: Hanna, M., Pezzelle, S., & Belinkov, Y. (2024). [Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms](https://arxiv.org/abs/2403.17806). EAP-IG与功能忠实性；参考其方法正文。
[^7]: Dunefsky, J., Chlenski, P., & Nanda, N. (2024). [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944). NeurIPS 2024. MLP替代及输入依赖/固定项分解；参考方法正文。
[^8]: Sundararajan, M., Taly, A., & Yan, Q. (2017). [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html). ICML, PMLR 70, 3319–3328. Integrated Gradients。
[^9]: Constantine, P. G., Dow, E., & Wang, Q. (2013). [Active subspace methods in theory and practice: applications to kriging surfaces](https://arxiv.org/abs/1304.2070). 作者预印本；2014发表于SIAM Journal on Scientific Computing。输出敏感度子空间及条件化误差分析。
[^10]: Constantine, P., & Gleich, D. (2014). [Computing active subspaces with Monte Carlo](https://arxiv.org/abs/1408.0545). 梯度协方差的抽样估计与bootstrap。
[^11]: Cranmer, M., et al. (2020). [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287). 作者预印本；NeurIPS 2020版本题名为Discovering Symbolic Models from Deep Learning。模块化后符号提取。
[^12]: Park, K., Choe, Y. J., & Veitch, V. (2024). [The Linear Representation Hypothesis and the Geometry of Large Language Models](https://proceedings.mlr.press/v235/park24c.html). ICML, PMLR 235, 39643–39666. 输入/输出表示与因果内积。
[^13]: Garg, N., Kleinberg, J., & Peng, K. (2026). [How Many Features Can a Language Model Store Under the Linear Representation Hypothesis?](https://proceedings.mlr.press/v336/garg26a.html). COLT, PMLR 336, 5358–5376. 线性表示与线性可读性的不同容量要求；仅在其模型假设内成立。
[^14]: Todd, E., et al. (2024). [Function Vectors in Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html). ICLR 2024. 函数的激活表示及干预。
[^15]: Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html). ICML, PMLR 97, 3519–3529. 表示比较与CKA。
[^16]: Funder, D. C. (1995). [On the accuracy of personality judgment: A realistic approach](https://pubmed.ncbi.nlm.nih.gov/7480467/). Psychological Review, 102(4), 652–670. [作者全文](https://rap.ucr.edu/FunderPR1995.pdf)。
[^17]: Mischel, W., & Shoda, Y. (1995). [A cognitive-affective system theory of personality](https://pubmed.ncbi.nlm.nih.gov/7740090/). Psychological Review, 102(2), 246–268. 情境与个人响应；本综述采用摘要所述理论范围。
[^18]: Fleeson, W. (2001). [Toward a structure- and process-integrated view of personality: Traits as density distributions of states](https://pubmed.ncbi.nlm.nih.gov/11414368/). Journal of Personality and Social Psychology, 80(6), 1011–1027. 状态分布与个体差异。
[^19]: Peters, H., Cerf, M., & Matz, S. C. (2024). [Large Language Models Can Infer Personality from Free-Form User Interactions](https://arxiv.org/abs/2405.13052). 预印本，对话条件比较。
[^20]: Campbell, D. T., & Fiske, D. W. (1959). [Convergent and discriminant validation by the multitrait-multimethod matrix](https://pubmed.ncbi.nlm.nih.gov/13634291/). Psychological Bulletin, 56(2), 81–105. 交叉效度组织。
[^21]: Zheng, L., et al. (2024). [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). NeurIPS 2024，初稿2023。共享前缀与RadixAttention。
[^22]: Chen, R., et al. (2025). [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509). [作者研究说明](https://www.anthropic.com/research/persona-vectors)。模型行为控制与监测，不是真人人格效度。
