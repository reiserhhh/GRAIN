# GRAIN 全系统逻辑图谱 V1

2026-09-10。用途：整个 Project Persona / GRAIN 的理论连接与开发总图，不是
新量表、发布评分器或新的心理效度结论。此图谱将历史 V8 与九月向量路径研究
接起来，不以新路线抹掉旧成果。具体数值以所链接的版本化报告为准。

## 1. 最终问题与三个需要同时存在的系统

最终问题是：如何从作者的自然语言材料形成可计算、可解释、相对明确群体有
意义的人格/状态推论？不是只预测一个量表，也不是只识别作者，更不是只拟合
LLM 的答案。

1. **表达与测量系统**：作者在机会条件下选择、表达、互动；GRAIN 将记录转成
   可比较的表达配置、分布、路径与响应对象，并放回参照总体。
2. **模型推论系统**：给定相同材料与一个明确查询，定位模型如何读取、归纳、
   组合信息，再尽可能将其有用计算抽取成固定数学程序。
3. **解释与效标系统**：将技术对象、模型读出与自评、他评、状态或行为通道
   分别连接；LLM 可渲染解释，但不能悄悄改写固定计分。

这三者通过相同作者、相同材料、相同查询和明确时间范围连接，并不互相替代。

```mermaid
flowchart TD
  U[作者历史/当前状态/可用机会] --> C[条件与选择 C1: where]
  U --> T[实际表达与互动: what/when]
  C --> T
  T --> O[记录/角色归属/采样/切片/表示]
  O --> G[GRAIN: 分布/水平/路径/响应/残差]
  G --> N[固定群体参照: who]
  O --> H[查询条件化模型状态 P/Q]
  Q[目标查询/构念定义] --> H
  H --> F[固定功能读出 f]
  H --> M[完整模型原生判断 F]
  G -. 同材料关联/源拟合/干预 .-> H
  G --> J[同材料交叉协方差 C_GH]
  O --> J
  J --> W[固定查询方向读取 Cw: 幅度/配对/效标联系]
  W --> I
  G --> D[共同token方向张量: GRAIN x 隐藏配置]
  O --> D
  D --> S[查询梯度读取: Cw + 位置特异作用]
  H --> S
  S --> I
  H --> R[直接投影/早层/无 Transformer 运行程序]
  R --> F
  N --> I[人格/状态解释候选]
  F --> I
  Y[独立自评/他评/行为/场景结果] --> I
  I --> A[证据约束的解释与未来应用]
```

图中实线也不全是已证明的因果关系。下表明确每条边的逻辑类型。

## 2. 每个结果必须带的对象地址

定义对象地址

\[
\omega=(\text{cohort, author unit, material, clock, operator, reference,
query, statistic, endpoint, model/version}).
\]

例如“F02 与 T/F 有联系”不完整。F02 的作者均值、文内离散度、跨抽样期差分
是三个不同统计对象；T/F 自报、固定函数 f 与原生模型 p(F) 是三个不同终点。
对象地址只是报告/实验协议，不是新心理学术语。

| 输出 | 数学对象 | 回答的问题 |
|---|---|---|
| 个体分数 | \(m_u=\Psi(A_u;P_0,Q_0,S,R,\mathcal C)\) | 此人在指定参照中处于哪里？ |
| 个体内变化 | \(m_{uv}-\bar m_u\) 或相邻视图差分 | 此人的不同记录如何变化？ |
| 人际关系 | \(d(A_u,A_v)\) | 两人的哪些表达特征相近？ |
| 群体关系 | \(\mathrm{Cov}_u(A_u,B_u)\) | 群体中两类位置如何共同变化？ |
| 模型响应 | \(F(H+\delta)-F(H)\) | 此向量操作怎样改变模型输出？ |
| 心理连接 | \(\mathrm{Assoc}(m_u,Y_u)\) | 技术读数与某一实际效标怎样相关？ |

## 3. 全链路逻辑登记

类型：D=定义；M=数学命题；E=已有有界经验结果；I=模型内干预；
H=仍待选择的机制假说；U=需要额外观测才能识别。状态不是理论完成百分比。

| ID | 前提 -> 推论 | 类型/现状 | 替代机制与下一连接 | 证据入口 |
|---|---|---|---|---|
| L01 | 心理过程在机会中产生语言 -> 可观察表达 | H/U | 沉默、选择、引用、AI 代写使记录不等于完整行为 | unified V8 §§2-4 |
| L02 | where 是可选择的 -> 条件选择也是作者信息 | E | 可用机会不等于选择偏好；先保留二者再比较 | 4W / rind V3 |
| L03 | 固定条件 -> 减少机会差异后的表达比较 | E | 固定条件也限制 C1；不推成任意总体内都提高信度 | rind / retrospective |
| L04 | 记录 -> 切片/向量/因子分数 | D/E | 带事件元数据的展示规范化可以保留真实路径；去掉时钟则展示反转与实际次序不可分。评论内序不等于评论间序 | V7 / clock and target contract §§2-3 |
| L05 | 向量近邻/聚类 -> 局部共同表达结构 | E | 主题、格式、模型几何也能产生簇；簇数不自动等于心理因子数 | V7 / route index |
| L06 | 多个局部观察 -> 作者分布与参数 | D/E | 采样量、时间窗、条件组成决定汇总对象 | unified §§4-5 |
| L07 | 群体因子坐标 -> 新作者固定参照分数 | D/E | 批内标准化与固定常模不同；新材料算子需显式版本 | V7 / NAV reference |
| L08 | 同人检索/折半几何 -> 可重复作者信息 | E | 身份/写作习惯可重复也不等于人格标签 | retrospective |
| L09 | 残差仍有作者结构 -> 保留残差作为候选信息 | E | 新增源固定查询配置余项：可计分且跟随模型变化；并非全部残差均为心理信息 | V6 / unified §8 / natural transport §4 |
| L10 | 动态不是均值函数 -> 路径提供额外观察对象 | M/E | 不推出额外人格维度；应区分表达顺序、抽样历时、模型层深 | V6 / dynamic report |
| L11 | 微观过程 -> 分布/响应/转移/互动/高阶路径 | M/E | 玩具世界证明机制可实现，不证明人类只由这些机制组成 | M3 atlas |
| L12 | 局部机制组合 -> 介观与群体关系 | M/E | 组合可有反馈/抵消/支持集漂移；单边成功不保证复合成功 | M4 / cross-level contract |
| L13 | 样本趋大 -> 估计误差减少 | M | 个体事件随机性、遗漏结构与估计方差不是同一误差 | unified §8 / cross-level §8 |
| L14 | 文本 + 查询 -> P/Q 条件化表征 | E/I | NEXT39固定完全相同输出方向，旧4/新8正文梯度余弦仍.728/.657，显示条件传输差异；query状态与布局共同包含。原概率.704/.649与新logit目标分开，不当量表相关 | fixed seed attention theory §1; output seed theory §§1-3 |
| L15 | 表征含信息 -> 读出可利用信息 | M/E | 固定P/Q的已测输入族中，f及三分量不变而原生锚定值变化；任何只读这些量的函数不能精确重建该干预族。自然分布另论 | scalar/conditional report §4 / input quotient |
| L16 | 精确锚定/路径积分 -> 分数变化分解 | M/E | 参考R的概率余项RMS=.013657，sigmoid曲率仅.000081；内部logit梯度也变化。不是单纯输出映射或未分账噪声，Hessian公式不等于已测完整Hessian | retained logit report §1; action geometry §3 |
| L17 | 源拟合固定读出 -> 重现完整模型部分判断 | E/I | 同4源完整N pair定向读法r=.610/.593，对照-.095/.092；但基线-only已有.602。相邻配置幅度r=.637/.721，有符号方向未恢复；不以完整分数替代局部机制 | query-directed pair report §§1-3; query-directed baseline report |
| L18 | 无完整 LLM 生成 -> 可数学单独运行 | E | E5均值接口扩展九查询；E/A/O保留效标r=.189/.148/.206。E5仍有编码器，token查表依赖不同 | NAV build / reading level report §4 |
| L19 | GRAIN 与查询关联 -> 可建立共同接口 | E/M | NEXT72共同配分函数的Cov(value,GRAIN方向logit)精确重建旧key作用3.747e-16；新8引子交换value SD约变2%，query对比却+.939到-.890。关键是带符号联合关系，不是边际强度；NEXT73隔离a(Q)与d(Q)，不把局部模型相关写成人格效度 | query partition geometry theory §§1-5; QUERY_PARTITION_GEOMETRY_REPORT_V1; query selection orientation plan |
| L20 | GRAIN 差分 -> P/Q 方向 -> 原生输出改变 | I/E | 新4作者3查询48个正文方向，有限差分与梯度47/48同号、相对L2误差.49%-1.67%；为固定P/Q的块16直接作用，不是自然文本总路径。实际E1旧结果保留 | native material reader report §4; natural transport §3 |
| L21 | 同一因子的离散度/变化/均值 -> 同一机制？ | M/E | pooled损失精确分为作者间/视图内；两种读取斜率确实不同但不自动改善。within不识别作者截距 | LOG01 / reading level report §§2,5 |
| L22 | 路径可读性高 -> 人类效标相关高？ | E | TF实际R/校准R自评r=.199/-.047，作用改善尚未恢复关联。N相邻实际幅度/正交范数自评r=.427/.469；保留不同几何统计，不据Y挑方向 | action geometry report §5 |
| L23 | 因子轴旋转 -> 共同对象仍可追踪 | M/E | C'=CT与w'=T^-1w保持Cw；多查询作用距离可连接旧作用加权几何。任意G坐标重表达还须传递输出度量 | material operator theory §4; LOG02 |
| L24 | 共享路径 -> 量表相关？ | H/E | 五Big5/四MBTI组内共32条分层关系；N余项作者水平较特异，其他部分共享。下一步接实际效标 | all-factor atlas / configuration extraction §6 |
| L25 | 联合 GRAIN+路径 -> 更多人格预测信息 | E | O/F03配对方向/自评r=-.272，错位.044，幅度乘积.009；A查询波动另有.224。S/Y贡献主要为对齐项-.249。源探索和无拟合折内账，不发布新预测器 | material geometry report §§2-4; same-material report §3 |
| L26 | 自评 Big5/MBTI -> 行动倾向/实际行动 | H/U | 自评是独立通道而非实测行动；T,Y 不识别未观测 B | dynamic report §6 / MTMM |
| L27 | 稳定 LLM 解释 -> 可重复且有据的叙述 | E/H | 新增原文区间/数字绑定校验、候选与替代解释协议；解释事件不等于token切口。新prompt尚未做LLM评价；独立盲编码与看数字后的渲染分开 | same-material interpretation theory §§4-5; V8 LLM assessment |
| L28 | 查询性“人格意义” -> GRAIN 因子构念 | H/E | 已找到T/F、E/A/O配置余项的外部关联；技术组件、特定效标与完整心理构念仍分层连接 | 4W / configuration criterion report |
| L29 | 多语/跨域可运行 -> 可比较心理量 | H/E | 运行兼容、几何稳定、测量等价是三条边；无需为探索先完成全部 | retrospective / unified §12 |
| L30 | 合成导航成功 -> 真人材料路径可计算 | E | 原顺序/成对源训练已分开；源OOF与旧12人运输响应仍可分歧，支持覆盖不是自动运输保证 | conditional / paired support reports |
| L31 | 当前局部接口 -> 人格/状态解释器 | H/E | NEXT72在同函数下保留边际与联合矩；导数里的中心化可增强信号，不是删除作者的条件选择。绝对对齐与head抵消是计算量，不是人格比例；有效token数含非BODY。评分/解释仍绑定同原文与群体参照，心理意义另接行为，K/V/dK神经依赖仍存 | query partition geometry theory §§3-5; QUERY_PARTITION_GEOMETRY_REPORT_V1; same-material interpretation theory |

## 4. 历史成果如何留在新系统中

| 开发段 | 当前仍保留的作用 | 不应混用的结果 |
|---|---|---|
| LoRA/表示优先/H2 | 预测与结构恢复分开；真实对照重要 | 结构矩阵相关不是个体预测 r |
| MCD | 条件相对的位置、选择和局部信息可补 baseline | 官方预计算协议 NP+MCD .318 与 clean 协议不是同一 benchmark |
| HMCD/V3 | 自由选择相 + 条件固定相；分布与条件响应 | 情动词的状态作用不等于稳定 trait 子量表 |
| V6/V7 | 静态、动态、共享、私有残差与多观察平面 | 0 个确认轴不否定分布式作者结构 |
| V8 M3/M4 | 微观-介观-宏观的有类型机制关系 | 人类机制玩具世界不等于 Transformer 电路 |
| 4W | where/what/who/when 联合解释 | where 选择包含 who，不是四个独立因子 |
| 向量查询/导航 | 固定功能、自然端点变化、便携计算 | 拟合 f、改变 F、相关 Y 是不同成功 |
| GRAIN-路径接合 | 相同材料上的分数/路径/原生干预连接 | 旧因子库只代表已实现的切口，不代表 GRAIN 全能力 |

历史来源：[`GRAIN_PROJECT_RETROSPECTIVE_20260905.md`](GRAIN_PROJECT_RETROSPECTIVE_20260905.md)、
[`SUICA_THEORY_ROUTE_INDEX.md`](SUICA_THEORY_ROUTE_INDEX.md)。
原始 parent MCD 报告仍在 ../project persona/reports/，不是在独立仓库重新全量复现。

## 5. 从“公式有余项”到“机制有解释”

\(Y=f(X)+R\) 总可以成立，但没有确定 R 的来源；同样
\(\Delta H=D\Delta G+R_H\) 并未让 D 变成唯一心理路径。我们需要逐边问：

1. 谁生成这个对象？作者、情境、记录器、表示模型、拟合器还是解释器？
2. 观察到了什么？值、差分、分布、子空间、模型响应还是效标？
3. 什么替代机制仍能产生相同现象？哪个最小改变能区分它们？
4. 结果能发展出什么新计算，而不是仅多出一条“不得宣称”？

接合的一般形式保留三个不同算子：

\[
G_u=\mathcal G(T_u;\omega),\quad
H_{uq}=\mathcal H_M(T_u,q),\quad
\widehat Y_{uq}=\mathcal R_q(G_u,H_{uq};\text{source}).
\]

研究 \(G\to H\to F\) 的干预作用可以发现“模型如何判断”；研究
\((G,H)\to Y\) 的作者级关系可以发现“什么信息与效标相关”。两条都成功才
能发展一个对该效标有意义的模型解释接口；它们仍非人脑机制的同一性证明。

### 5.1 第二轮：接口不能只靠同名因子相接

同一12人真实材料删减后，冻结向量函数f追踪原生F的变化：T/F r=.886、
神经质r=.681；源拟合GRAIN变化映射弱得多，T/F可反向。因此加入两条明确边：

```mermaid
flowchart LR
  D[实际材料变化 Delta T] --> G[历史 GRAIN 变化 Delta G]
  D --> H[查询条件化状态变化 Delta H]
  G --> B[源固定 beta Delta G]
  H --> F[固定函数变化 Delta f]
  B --> R[查询可读余项 Delta R]
  F --> R
  R --> C[均值/问题位置配置/最终读取]
  C --> N[固定参照的新观测候选]
```

这里Delta R=Delta f-beta Delta G，不能因恒等式成立就说机制唯一。
实际可发展的部分是其中**问题位置配置余项**：与原生变化r=.775/.642；
三个部分已逐个保存源拟合系数、固定参照与新作者读数。它需要P/Q状态，
不是无模型的文本编码器，也不是新命名的人格构念。
材料删减还给出精确组成式Delta mean(s)=(k/n)(mean_kept-mean_removed)，
但极性相似度不等于模型功能作用。机会、材料可用性、作者选择仍分开处理。

详见[自然变化/查询余项报告](../reports/grain_vector_mechanisms/NATURAL_TRANSPORT_LOGIC_REPORT_V1.md)。

### 5.2 第三轮：共同输入表示与查询功能连接

冻结的E5均值线性程序可以提取N配置余项：E1变化忠实度r=.841，和原生模型
变化r=.733（12作者）。幅度约真实余项变化的12.9%，但MSE相对零变化减少18.4%。
T/F未获得同样的自然变化重现。纯token查表的N完整水平相关.744，却未跟随
实际变化。由此分开**位置函数**、**证据响应函数**与**统计单位/材料支持**。

每评论的已学功能分数psi_q(h)=w_q^T h，使该轻量程序的删除变化精确等于
(k/n)(mean_kept(psi)-mean_removed(psi))。N权重与人格词极性方向cosine=.083，
表明它不是直接拿人格词相似度充当功能读数；这不是独立心理维度的证明。

同一固定输入与Ridge算子S_X下有新接口关系：

    R_direct = S_X(c-m_G),   R_derived = S_X(c)-m_G
    R_direct-R_derived = m_G-S_X(m_G).

因此“直接预测余项”减去的是输入可重现的GRAIN部分，而非完整GRAIN。两种
加减法不能随意互换或称更纯人格。该关系可检验、可用于设计共同表示接口。
五个Big5查询在213作者上、四个MBTI查询在63作者上分别分析，N余项在作者
水平相对特异，A/E等保留共享；未混合人群或将模型组件关系当真实量表关系。

详见[轻量提取/作用算子报告](../reports/grain_vector_mechanisms/CONFIGURATION_EXTRACTION_LOGIC_REPORT_V1.md)。
下一连接：固定技术读数与已许可外部效标的信息分配，而非继续追求模型自洽。

### 5.3 第四轮：从模型读取走到作者效标

同一276源作者的已有真实量表，已与全部九查询组件及两个轻量提取目标联接。
T/F余项/自报r=.305；外向性、宜人性、开放性为.227/.186/.176。
E5均值OOF读数保留T/F r=.249，但N仅.061。N的局部模型响应成功仍保留，
用途不同，不被效标差额抹去。这里3视图作者均值也不同于12人单次删除响应。

两条可计算连接分别是：

    Cov(c,Y) = Cov(m_G,Y) + Cov(R,Y)
    Cov(R,Y) = Cov(R_hat,Y) + Cov(R-R_hat,Y).

第二式T/F按同一sd(R)sd(Y)标准化后为.304=.157+.147，定位了读出误差中的
效标联系，不是一个“剩余人格比例”。完整f的T/F效标.201则由配置+.240和
均值/最终读取约-.039共同构成，说明组合算子会保留或抵消特定效标关联。
因此理论应同时记录**机制重现**、**效标保留**与**组合后的作用**，不让一个
相关数充当整套理论的优劣裁决。

详见[效标连接与作用记账](../reports/grain_vector_mechanisms/CONFIGURATION_CRITERION_LOGIC_REPORT_V1.md)。
Funder(1995)的线索相关性/可获得性/检测/使用框架，为这些不同连接提供先例，
不是当前神经机制已被既有心理学理论证明的意思。

### 5.4 第五轮：从误差存在走到误差属于哪个读取环节

固定目标t与预测p可以描述成t=a+bp+e，因此：

    Cov(t-p,Y)=(b-1)Cov(p,Y)+Cov(e,Y).

T/F余项+.1475的剩余关联贡献中，幅度+.0068、无法用p仿射重标定消去的部分
+.1406。这是新的定位，不是新的分数。它把“只是缩得太小”的解释降级，也不
把非仿射剩余直接叫作非线性机制。另一方面，pooled平方损失精确拆成作者均值
和视图内两项；同一输入分别学习的斜率确有差异，但作者间专用训练未提高T/F。

同一固定接口已实际扩展全部九查询：E5余项E/A/O的效标r=.189/.148/.206。
N视图内读取的自然变化r=.788，但组件忠实度低于原pooled，不做自动升级。
两种成功分别归档：**保留人际效标信息**与**读取材料变化**。

这把系统的“放大人格信号”主张细化为：先指定目标信息、统计层级与输入
可用性，再找到相应的计算算子；不是统一删噪、统一放大或统一换一个更高r的头。
[完整报告](../reports/grain_vector_mechanisms/READING_LEVEL_LOGIC_REPORT_V1.md)保留全臂结果。
下一边对接历史S6内部配置机制与当前E5输入；不把两个空间中的“均值核”当成同一物。

### 5.5 第六轮：输入相同、模型不同，应该补输入还是换目标？

同一12作者的12条评论保持原文，只反转列表，E5均值不变，但原生T/F/N判断
变化RMS=.105/.034。两查询均12/12人发生变化，旧端点及重复误差为0。
这是当前真实输入等价类中的可观察失配，不再只是低r或另一个空间的内部干预。

对反转J，定义t+=(t+tJ)/2、t-=(t-tJ)/2。任何不变预测p在这个等权二元轨道上：

    mean_pair (t-p)^2 = (t_plus-p)^2 + t_minus^2.

这个误差下界属于该人为呈现实验，不是原始自然作者效标中的人格比例。
源拟合ordinal向量新增了顺序区分能力，却没有跟上实际配置变化；源T/F拟合
略升、自评联系反降。由此将系统分成三个需要分别发展的接口：**输入可区分**、
**条件运算可迁移**、**作者效标信息保留**。

下一层逻辑也随之明确：对任意呈现排序稳定的作者位置，和真实表达顺序中的
动态响应，是不同目标。若只追求复制LLM，可能把呈现偏差一起复制；若一概删
顺序，又会丢掉真实when。GRAIN保留两个观察对象，再决定各自的解释与效标。
详见[实际输入实验](../reports/grain_vector_mechanisms/INPUT_ORDER_LOGIC_REPORT_V1.md)与
[输入等价类/目标理论](GRAIN_INPUT_QUOTIENT_AND_TARGET_THEORY_V1.md)。

### 5.6 第九轮：从固定分数推进到有依据的条件函数

本轮同时回答“能否只换读法”和“输入有没有丢区别”。7臂、276作者的固定
仿射/三次/分量读法未清楚补回原生联系；旧M11固定P/Q后，当前分量相同而
原生值仍变。后者给出实测输入等价类上的所有标量读法风险下界，不靠低r猜测。

旧M11/M28的两查询函数完全相同。对旧24人/query按当前代理投影后，条件
起点仍留有共同根效标贡献TF+.0946、N+.2123。下一步因此是构造已有干预
定义的b(E)与条件敏感度K(E,d)，而不是把所有误差压成一个非线性总分。
换参考会改变分项而不改变总函数，故参照与依赖输入支持也属于公式的定义。

[条件函数逻辑](GRAIN_CONDITIONAL_FUNCTION_LOGIC_V1.md)把该接口接回GRAIN的
条件组成、作者位置和路径测量，并逐项保留M12-M19已经完成的构造与反例。
[实测报告](../reports/grain_vector_mechanisms/SCALAR_CONDITIONAL_LOGIC_REPORT_V1.md)
提供完整七臂、精确聚合式、条件投影及输入族结果；没有新增神经调用。

### 5.7 第十轮：同一条件缺口不要求同一修复函数

当前代理下剩余TF条件联系主要在B=F(C(E),x0)，N则在保持评论均值的配置A。
两条操作顺序都保留N的局部配置贡献，但会按交互I搬移分账。这把“条件项”
细化成不同的模型计算对象，没有将其直接命名为主题或人格。

新配置候选用均值作相位条件，读出非线性分布差，保持零位移及反射奇偶。
四臂实测表明新增表示能力没有普遍改善原生作用重建；一些自评联系仍可保留。
下一步不是调频数，而是检验跨评论共同作用，并继续追踪TF均值化后保留的
条件载体。[完整结果](../reports/grain_vector_mechanisms/CONDITIONAL_FACTORIAL_KERNEL_LOGIC_REPORT_V1.md)。

### 5.8 第十一轮：误差分层成为可定位作用

410次后缀调用补上独立评论四格与TF条件载体四格。内部混合差与softmax
曲率明确分账，前者确实存在；当前按token相加的核读出因此需要局部联合
函数，而不是仅增加单token非线性。N相邻交互强度的源关联是一个可发展线索。

TF均值化后保留的136位置非正文状态有明确作用。正文改成均值不等于清除
上游已经送入模板位置的信息；固定字面不能替代固定状态。固定M0/R0时
布局仍造成读数差异，这又与GRAIN前置条件/参照设计相接。
[实测报告](../reports/grain_vector_mechanisms/COMMENT_CARRIER_LOGIC_REPORT_V1.md)
及[交互函数理论](GRAIN_CONDITIONAL_INTERACTION_OPERATOR_V1.md)给出全部接口。
没有把24人点位关联升级成新量表，也没有因关联区间宽而删除实际计算作用。

### 5.9 第十二轮：共同矩阵不等于共同作用最优

两个具体读法已执行：N按作者四折的对称张量核，TF四参照作者的R路径积分
矩阵；32次后缀前向/反向对，37.61秒，无新前缀或问卷拟合。
N局部交互可读，但对类型均值已经解释大量共有作用；远隔个体变化仍不跟随。
TF矩阵有91.18%共同欧氏能量，平均矩阵却只读出部分实际个体作用。

因此新增逻辑不是“再多几个因子”，而是**参数相似性必须经过实际位移的作用
度量才能连接到作者差异**。正规方程与参照支持内的校准候选已经写出；不把
少量参照上可积分的公式误当跨作者最优预测器。完整条件B的布局/正文输入
仍明确标为原模型端点，不用组装相关跳过输入链。
见[本轮报告](../reports/grain_vector_mechanisms/JOINT_CARRIER_READOUT_LOGIC_REPORT_V1.md)
和[作用度量推导](GRAIN_CONDITIONAL_INTERACTION_OPERATOR_V1.md#6-共享函数的几何不能只按矩阵范数定义)。

### 5.10 第十三轮：找到条件位移曲率，补齐完整输入账本

一次四参照内校准使其余20人T/F作用误差相对下降4.67%，但区间跨零。
更有决策价值的是：软参照支持仅约0.9%；矩阵参数中的布局差异很大，却不
主导被遗漏的实际作用，R位移路径变化才是最大的分项。这让下一对象从
固定平均矩阵变成明确的条件响应函数，而非重复精度/维度搜索。

N相邻共同方向有折内幅度信息可读；远隔pooled负相关不能直接解释成人格
方向反转。最后softmax的局部余项很小，下一步应看方向与正交能量的分工。
固定Ridge再训投影目标与投影原输出相同，但条件能量不等于条件均值的能量。

[第十三轮报告](../reports/grain_vector_mechanisms/ACTION_GEOMETRY_LOGIC_REPORT_V1.md)
保留全部作用/效标/负条件；[输入依赖图](GRAIN_CONDITIONAL_INPUT_DEPENDENCIES_V1.md)
精确区分F、b(E)、B，避免把部分公式的闭合扩大为完整系统独立可运行。
83项相关测试通过、15份数值/聚合产物字节重放；0新神经调用或参与者。

### 5.11 第十四至十六轮：同材料接口进入原生读法

同评论GRAIN/编译查询的共同变化，已分解为边缘波动和配对方向，并归纳成
可复用作者C矩阵；O/F03的配对方向与自评r=-.272，宜人性查询波动r=.224。
第十六轮进一步发现：当前query-last原生布局可固定同一正文，但不同query
在不同token位置的读取不可简单平均。带token位置的GRAIN×隐藏配置张量，
与原生分数梯度相乘，可重放实测局部作用；旧Cw是其均匀读者特例。

原生4作者、3查询、48方向，111后缀调用/72.59秒，58项相关测试、10文件
精确重放。共同前缀数值差异与旁文件读取问题均保留并明确修复；未重跑神经
收集，未新增问卷读取。见[原生报告](../reports/grain_vector_mechanisms/NATIVE_MATERIAL_READER_LOGIC_REPORT_V1.md)
和[原生接口理论](GRAIN_NATIVE_MATERIAL_READER_THEORY_V1.md)。接下来回到
NEXT24–26条件作用/能量/完整输入，不以一个局部导数代替完整人格推论。

### 5.12 第十七轮：曲率发生在内部，不只在输出换算

4参照作者×2正文条件的旧R路径，转到log-odds后仍有明显曲率。总概率
非线性余项RMS=.013657，sigmoid-only=.000081；精确拆分误差0，梯度场
端点变化约起点范数的.443。这个发现保留了状态相关读法的发展方向。

一次有界径向/切向二次函数测试已完成：20旧评估作者的响应r从.295到.323，
作用MSE/零却从.913到1.122。真实起点也不救回，所以不升级该候选。数学
反例解释为什么块内伸缩可能漏掉方向耦合，但真实耦合位置仍需所选接口
来指定，不能由失败直接命名机制。41相关测试、10文件精确重放，0神经调用。

见[曲率报告](../reports/grain_vector_mechanisms/RETAINED_LOGIT_CURVATURE_REPORT_V1.md)
和[内部函数理论](GRAIN_RETAINED_LOGIT_CURVATURE_THEORY_V1.md)。NEXT24有界
比较结束，继续条件能量和完整输入接口，不把逐次失败变成无边界调参。

### 5.13 第十八轮：均值之外的强度，以及条件方差的来源

同24作者/48文本对，正核混合分布可统一算均值、二阶强度与协方差，矩恒等式
误差9.89e-17。原预测强度的缺口含正负交叉项，不可全部命名为残差方差。
现有相邻读法强度r=.486/.476，折内仍.448/.438；新增第二矩主要恢复群体
量级，未优于源平均强度。分布评分相对均值评分只改变原生作用RMS的约
.165%/.034%，不是当前丢失主作用的解释。

更关键的理论接点：完整固定输入使模型输出确定，聚合输入可产生条件方差。
额外条件降低理想平均风险，但不保证每个群或每次拟合都改善。它把GRAIN的
条件化思路接到具体函数输入，见[条件矩理论](GRAIN_CONDITIONAL_MOMENT_THEORY_V1.md)。

[实证报告](../reports/grain_vector_mechanisms/CONDITIONAL_ENERGY_REPORT_V1.md)：
46相关测试、9文件重放、0神经调用；NEXT25结束。下一[完整条件文本对接口](GRAIN_COMPLETE_PAIR_INTERFACE_PLAN_V1.md)
拟合当时借用的z_0/单评论端点，再接既有mixed读法；执行结果见下段。

### 5.14 第十九轮：执行闭合后，读取函数的缺口显现

完整条件N pair评分已用数值输入/源模型独立运行，禁读所有其它文件的四个
进程精确复算。4个基线头+8个单评论头，无新神经调用或Y读取。目标仍是
固定P/Q、两评论恢复的模型分数，不是完整F或人的人格。

pooled r=-.107/.0006，RMSE=.01613/.01581；真实additive+旧mixed有约.9997
相关，差异说明借用起点对总分的重要性，不否认旧局部mixed作用。基线本身
r=-.164、折内.010，误差大于源均值；给真mixed也无法救回。因而下一步应
发展查询定向上下文读法，而非继续交互或最终概率精修。

这是“输入可见不等于读法学会”的实测实例：完整均值化上下文坐标仍在，
但有限源下通用相似度核主要拟合共同分布；压缩评论配置另有信息限制。
见[报告](../reports/grain_vector_mechanisms/COMPLETE_PAIR_INTERFACE_REPORT_V1.md)、
[理论](GRAIN_COMPLETE_PAIR_INTERFACE_THEORY_V1.md)和
[NEXT32查询定向方案](GRAIN_QUERY_DIRECTED_BASELINE_PLAN_V1.md)。

## 6. 本轮文献与执行入口

### 第二十五轮补充：共享与专用输入，离开四人面板

只换输入方向，不换接收查询的目标、输出基底或拟合规则。N/确认子空间
重叠.662，TF与二者约.260。N借确认作用r=.508、自身.473，但误差近乎
相同；确认借N=.551、自身.576。TF自身.863明显高于借N的.623与借确认.575。
这支持部分共享/部分专用的局部读法，不把子空间重叠解释成量表相关。

几何旋转不变不等于标准化Ridge程序不变；新测试验证惩罚必须一起变换。
同8列也不等于严格同有效自由度。全矩阵与例外保留，不选择新最佳donor。
24新头、0神经调用、40数值文件重放、55测试。NEXT37已固定8名旧许可源外
作者（两来源各4），没有采集新目标；用旧读法等权集成，native上限600秒。
[报告](../reports/grain_vector_mechanisms/QUERY_DIRECTION_TRANSFER_REPORT_V1.md)、
[理论](GRAIN_QUERY_DIRECTION_TRANSFER_THEORY_V1.md)、
[源外作者实验](GRAIN_READER_NEW_AUTHOR_TRANSPORT_PLAN_V1.md)。

### 第二十四轮补充：低能量输入与查询读取相接

新增8个源查询残差方向，覆盖残差能量2.985–3.192%，而等宽PCA覆盖7.071%；
前者T/F、N、寻求确认的GRAIN作用r=.863/.473/.576，后者.684/.114/.149。
作用SSE/零也同步改善到.657/.874/.712，源系数误差下降，作者中心化结果
保留。不是单纯调alpha或报告少数正例。完整梯度误差变大却不抹掉可测方向
更准确的结果：Frobenius误差和D构成的测量半范数是不同对象。

4旧作者/3查询，24源拟合，0新神经调用，当前梯度/旧结果禁读。43文件精确
重放、48测试通过。历史同名入口已恢复，新runner独立r2与初次未审阅数值
一致。GRAIN本体、外部效标和解释协议仍独立。
[报告](../reports/grain_vector_mechanisms/QUERY_RESIDUAL_INPUT_REPORT_V1.md)、
[三类方向理论](GRAIN_QUERY_RESIDUAL_INPUT_THEORY_V1.md)、
[下一步跨查询借用](GRAIN_QUERY_DIRECTION_TRANSFER_PLAN_V1.md)。

### 第二十三轮补充：内容权重与源特征空间的明确边界

同4作者，内容向量预测T/F局部作用r=.638（纯位置.498），去因子均值后
作者r=.614，幅度仅12%。当前GRAIN交互源拟合改善但留出更差，不替换GRAIN。
源空间投影证明最低误差仍.866-.959，正则化增加仅.000012-.000069；PCA32
保留39.8%输入方差。下一步应找查询相关的压缩残余，而非继续调整同一读法。

基底/系数在梯度空间正交，在测量作用空间有交叉项；两层恒等式均复算。
34测试、48数值文件精确重放、0神经调用。[报告](../reports/grain_vector_mechanisms/CONTENT_COEFFICIENT_READER_REPORT_V1.md)、
[理论](GRAIN_CONTENT_COEFFICIENT_THEORY_V1.md)、
[NEXT35残余输入计划](GRAIN_QUERY_RESIDUAL_INPUT_PLAN_V1.md)。

### 第二十二轮补充：方向词汇可共享，token权重需要内容条件

同4旧作者/3查询，原生48个GRAIN切线作用；自身rank8压缩r=.952/.998/.990，
其他3作者rank8方向+当前梯度系数oracle为.862/.987/.977，源三次位置读法
仅.498/.047/.033。于是共享隐藏方向V和局部系数a成为两个可分别测试的对象，
不是用高r跳过系数预测。局部对象仍是原生模型导数，不是新心理量表分数。

18次nested-mode增量中4次能量误差改善但作用变差；标准有符号误差恒等式
重放至2.22e-16。测量半范数说明为何“能量更像”与“作用更像”不同，也不应
把对当前四个方向不可见的部分叫人格噪声。23测试、18数值文件精确重放、
0新神经调用。[报告](../reports/grain_vector_mechanisms/POSITION_SPECIFIC_READER_REPORT_V1.md)、
[理论](GRAIN_POSITION_READER_THEORY_V1.md)、
[NEXT34内容/GRAIN系数计划](GRAIN_CONTENT_COEFFICIENT_READER_PLAN_V1.md)。

### 第二十一轮补充：完整组装与有符号方向

同4源的完整条件pair已数值接通，r=.610/.593，对照-.095/.092；基线-only
已有.602。相邻配置幅度r=.637/.721，但共同原生评分方向的对齐r=-.315/-.312。
最终评分函数的切线近似与原生变化r=.9981/.9998，缺口主要在链接之前。
将DeltaS写成长度、读取敏感度、夹角的乘积，并作固定顺序的误差分账，
80个配对诊断条目复算误差<=3.35e-16。原生基线只作诊断，不混入预测器。

[报告](../reports/grain_vector_mechanisms/QUERY_DIRECTED_PAIR_REPORT_V1.md)、
[方向理论](GRAIN_COMPLETE_RESPONSE_DIRECTION_THEORY_V1.md)、
[NEXT31位置读取](GRAIN_POSITION_SPECIFIC_READER_PLAN_V1.md)。81相关测试、16
数值文件精确重放，0新神经调用；发现有效组装，也定位了应开发的局部机制。

### 第二十轮补充：查询定向的基础读法

四个旧N参考作者提供条件候选Jacobian，另外20作者测量数值读法；同源核
r=-.050，查询仿射r=.727 [.443,.885]，评分RMSE降25.3%。这证明了本地
“指定查询所读方向”的可用性，不是只增加相似度特征。原测量20前向/
40反向目标/38.41秒；8个向量差分的80候选分量同号。

作者中心化logit误差/变化=.524 versus kernel1.061，整体向量误差却受共同
偏移影响。读法平均还有来源明确的配对项：zhat=zbar+Jbar(c-cbar)-C_Jc，
C_Jc的源RMS=.052493，复算误差2.45e-15。源位置与源读取不能任意拆开平均。
它把已有同材料共变思想接到了一个可独立算的条件函数。

[报告](../reports/grain_vector_mechanisms/QUERY_DIRECTED_BASELINE_REPORT_V1.md)、
[作用几何理论](GRAIN_QUERY_DIRECTED_BASELINE_THEORY_V1.md)、
[NEXT33同源完整组装](GRAIN_QUERY_DIRECTED_PAIR_ASSEMBLY_PLAN_V1.md)。
本轮用固定块16输入与源导数；临床解释和人的外部效标没有由这次模型r替代。

### 第二十七轮补充：回答编码、传输与上下文相对作用

新8人N/确认的输出方向余弦.954，而正文梯度.649；源输出方向空间仍覆盖
99%以上能量。有限候选字母本身约束秩，不把它命名成共通人格空间。T/F
没有概率饱和，按源gain换算后新作者作用r仍.011，故继续开发条件读法。

标准注意力反传提供值项a_j W_V^T g与选择项
a_j*((v_j-o)^T g)*W_K^T q/sqrt(d)。玩具例子中其它token改变o，使目标
key作用+.25变-.50，而目标token/query/注意力权重全不变。这个解析机制
具体说明缺失的全局上下文变量；真实模型head的量级将在NEXT39验证。
同一各向异性传输改变夹角的反例已测试，因此下一步固定相同seed而非仅比角度。

[报告](../reports/grain_vector_mechanisms/READER_OUTPUT_SEED_REPORT_V1.md)、
[理论](GRAIN_READER_OUTPUT_SEED_THEORY_V1.md)、
[NEXT39](GRAIN_FIXED_SEED_ATTENTION_PLAN_V1.md)。77聚焦测试、9数值文件精确
重放、0神经调用；两套系统的同token接合与独立辅助解释层均保留。

### 第二十八轮补充：实际attention中可分离的条件读取

同一输出s下正文J仍有query差异（旧4/新8余弦.728/.657），因此条件传输
确实不同，不再仅是不同输出seed经过共同矩阵的歧义。query布局与状态仍
共同包含。两层分别定义局部干预，不能用余弦变化计算心理信息损失。

实际最后block12头/2KV/RoPE的值和key作用24/24复算，相对误差2.03e-6。
新8人key/value逐作者范数比平均1.826/1.887，丢弃key误差.731/.743。
GRAIN投影中仍有反号相抵，保留有符号加和而非只取正attention权重。

[报告](../reports/grain_vector_mechanisms/FIXED_SEED_ATTENTION_REPORT_V1.md)、
[理论](GRAIN_FIXED_SEED_ATTENTION_THEORY_V1.md)、
[NEXT40源固定g程序](GRAIN_SOURCE_COTANGENT_READER_PLAN_V1.md)。28前向/24反向，
44.410秒，51聚焦测试，6CSV精确重放。当前q/k/v/g仍有原模型依赖；下一步
替换当前g，不冒称全部LLM推论已独立。整体goal继续ACTIVE。

### 第二十九轮补充：共同读出与上下文响应可以分开实现

最后层GRAIN局部作用现在可写为tauhat=(A D)^T gbar，A保留当前材料attention，
gbar仅取旧源。新8人自身查询源r=.809/.955，pool=.919/.869；全式误差明显
低于value-only，当前g/梯度禁读。此处并非整个人格预测，输入q/k/v仍来自模型。

实际误差恒等式delta_tau=Lambda delta_g闭合至6.11e-16，提示应该在所测
响应的几何下判断读出误差。N的g欧氏误差增加而GRAIN作用误差减小就是
一例，不需把整个场拟合到完美才发展可用接口。源pool和自身两法全部保留。

[报告](../reports/grain_vector_mechanisms/SOURCE_COTANGENT_READER_REPORT_V1.md)、
[理论](GRAIN_SOURCE_COTANGENT_READER_THEORY_V1.md)、
[NEXT41](GRAIN_COMPLETE_TRANSPORT_JVP_PLAN_V1.md)。24数值包/6CSV精确复算，
54聚焦测试，0神经调用。下一步将块16方向传至post-attention残差并用源t读，
保留最终query与直接残差，不把最后BODY公式误当完整上游传输。goal ACTIVE。

### 第三十轮补充：完整作用的生成位置与最终读取分开

完整H16方向经各层attention/MLP与query状态传播，再由固定源t读，
新8人N/确认r=.99814/.99610；进入最后层前单独已有.99688/.98816。
因此最后attention并非全部推论机制，它主要调整已形成的响应。direct
项包含此前10层的attention/MLP，不是“attention不重要”或普通直通文本。

链式乘积与残差望远镜展开给出下一步：在固定深度20/23/26读取，并检查
有符号增量如何相加，最终实现可提前停止的程序。保持层/目标地址与旧
NEXT40区别，不用同一个r符号抹掉不同实验对象。心理解释另行连接。

[报告](../reports/grain_vector_mechanisms/COMPLETE_TRANSPORT_JVP_REPORT_V1.md)、
[理论](GRAIN_COMPLETE_TRANSPORT_THEORY_V1.md)、[NEXT42](GRAIN_LAYER_ACCUMULATION_PLAN_V1.md)。
源捕获浮点试跑已修复且留档，全部源评分重放0；96JVP/4有限前向、总
10源前向/反向，172.89秒。57测试、3CSV精确复算，无当前目标反向/标签。

### 第三十一轮补充：把条件变换定位到可拆解的门控更新

层20的N作用在旧4人r=.876，新8人只有.092；到层23，新8为.952，确认
查询为.823但幅度仍约半。层25MLP随后给出两查询最大的正投影更新，
累计SSE/zero从.275/.314降至.022/.028。这个位置已被实际计算定位，不是
按最终注意力图猜测。两查询共享有用更新位置，但份额/个体路径不同。

下一步是SwiGLU乘积法则与RMS导数：把输入变化、门控变化显式分开，
并分别固定源门控/源norm检验所需的条件系数。原样共享矩阵可工作与
必须读当前系数是两条有用分支。两种结论都发展提取机制，而不是给层起人格名。

[报告](../reports/grain_vector_mechanisms/LAYER_ACCUMULATION_REPORT_V1.md)、
[公式](GRAIN_LAYER_ACCUMULATION_THEORY_V1.md)、[NEXT43](GRAIN_GATED_UPDATE_PLAN_V1.md)。
100JVP/167.352秒，4真实截短检查精确；24完整参考精确，5CSV重放，65测试。
无当前评分/反向/标签，整体goal ACTIVE，未改解释通道/旧锁箱/评分规范。

### 第三十二轮补充：机制有用不等于每个局部参数都要因人调整

源固定层25完整门控/RMS算子在新8人保留r=.996936/.998451。与此同时，
单独丢掉输入或门控变化项会使SSE/zero升至.201-.385。因此应保留乘法
机制，而不必在这个已测试位置为每个作者重新拟合细小系数。源程序可精确
编译为单个wbar，作用=wbar^T d；当前d仍承载上游的实际条件化过程。

这不是“模型本来就是线性的”，也不是“0.997人格准确率”。它给出一个
局部可替换的计算对象：读数几何允许完整向量有.213/.194相对误差，但
有用的查询方向作用仍被保留。下一步借用query读头，区分差异来自w还是d。

[报告](../reports/grain_vector_mechanisms/GATED_UPDATE_REPORT_V1.md)、
[理论](GRAIN_GATED_UPDATE_THEORY_V1.md)、[NEXT44](GRAIN_COMPILED_QUERY_TRANSFER_PLAN_V1.md)。
96JVP/149.086秒、数值3.059秒，源排除/6CSV复算、75测试；单向量编译差
7.772e-16。预算暂停后原包续跑，未增加额度或重复采样。整体goal ACTIVE。

### 第三十三轮补充：相同正文扰动，经不同查询运输后可由共享读头读取

12位作者两查询的BODY状态/位置/GRAIN完全一致；固定recipient目标后互借
读头，新8作用r仍.990935/.994290。等权共享读头是可保留的局部计算对象，
不需要强行让每个人格查询都有一套完全不共通的末端坐标。

精确对称差异C=R+T；自身模式新8的T投影.914272，R .085728，pool则
R为负、T超过1，说明必须保留抵消。它定位的是计算差异的入口，不是人的
心理份额或唯一因果中介。当前d仍来自上游查询上下文；下一步找到它的形成过程。

[报告](../reports/grain_vector_mechanisms/COMPILED_QUERY_TRANSFER_REPORT_V1.md)、
[理论](GRAIN_COMPILED_QUERY_TRANSFER_THEORY_V1.md)、[NEXT45](GRAIN_QUERY_TRANSPORT_FORMATION_PLAN_V1.md)。
384数值动作、0神经调用、8CSV重放、83测试；全系统goal继续ACTIVE。

### 第六十一轮补充：边际相似，联合方向仍可反转

NEXT72以同源查询/K/V/dK缓存编译共同log配分函数的一阶/混合二阶矩，重建
NEXT50的key作用最大差3.747e-16。新8交换引子value SD只变约2%，而N/确认
对比相关从+.939到-.890，共享引子.477。旧局部效应被解释为联合几何，不是
新增问卷预测结果。相同边际分布、相反协方差的synthetic反例给出信息缺口。

自身引子新8绝对对齐.087826/.070253、head符号聚合保留.389114/.319030。
小总作用可以来自低对齐与抵消，不能把这些计算比分别叫人格信号比例。
U/B目标投影+.205872/-.068884，R=U-B=.274756，说明这里中心化是评分导数
中的必要项，不推翻旧SUICA关于保留条件选择信息的结论。

0拟合/神经/Y；R2总CPU1.917672秒，12包/3CSV精确重放，34相关测试通过。
首轮清单混入OS sidecar后已修正，真实三表数值未变、首轮留存。NEXT73只做
一次选择测度a(Q)与方向d(Q)的交叉标量接口，指定下一源operator保留的关系。
见[报告](../reports/grain_vector_mechanisms/QUERY_PARTITION_GEOMETRY_REPORT_V1.md)、
[理论](GRAIN_QUERY_PARTITION_GEOMETRY_THEORY_V1.md)、[NEXT73](GRAIN_QUERY_SELECTION_ORIENTATION_PLAN_V1.md)。

### 第六十轮补充：内容与选择共同生成评分，误差可发生抵消

NEXT71的四冻结函数、48包重建到1.665e-15，6张表完全重放，0新拟合/神经/Y。
raw源跨后段作用仅logmass；局部、内容、选择项的B变化严格为0。相对读法则让
C/S同时跨段变化：源N GRAIN投影C/S/M=.020849/.004811/.013304，确认查询为
.023730/-.012795/-.028489。带符号投影不是相关或独立信息份额。

新8raw作用改善N3.602个百分点，由local+8.867、C-15.247、S+9.346、M+.637
共同构成；确认5.413个百分点亦有大幅抵消。M在8个重叠的query/arm/投影单元
正贡献，不能因此当作8次独立验证或唯一有效成分。两个synthetic测试进一步
验证C/S分别的旋度会互相抵消，因此下一开发必须保持共同标量生成。

共同log配分函数可把logmass的一阶响应与相对value的混合二阶协方差接起来。
NEXT72将从冻结source-query缓存编译其几何，区别值对比、方向敏感度与对齐；
不再新增泛核拟合或attention排名。见[报告](../reports/grain_vector_mechanisms/EVIDENCE_COMPONENT_REPORT_V1.md)、
[理论](GRAIN_EVIDENCE_COMPONENT_THEORY_V1.md)、[NEXT72](GRAIN_QUERY_PARTITION_GEOMETRY_PLAN_V1.md)。

### 第五十九轮补充：保留证据质量，但区分选择和读出

NEXT70同源四作者/四视图，两查询共四拟合；相同1313参数、固定16核，比较
(m,logZ)与(m/Z,logZ)。源N相对选择把坐标SSEzero从.995625降到.979114，
GRAIN B作用从.969572降到.948326；确认查询不复制收益。新8则raw作用误差
较local低3.60%/5.41%，全查询读数r=.151881/.349486，未优于旧完整读法。

标准商法则显式给出直接内容、相对竞争和log质量项。raw也因logZ产生非局部
响应；归一化不创造信息，但线性读出与Ridge不在非线性重表达下保持等价。
质量仅指平均核响应，不是所有主题/情境选择。下一对冻结程序分账，识别
哪个计算作用带来了已有局部收益，再决定怎样改选择机制，而非再搜中心。

68测试/Ruff；48预测包/11CSV重放，384方向差分最大1.96e-10；阶段CPU32.817521秒。
见[报告](../reports/grain_vector_mechanisms/NORMALIZED_EVIDENCE_REPORT_V1.md)、
[理论](GRAIN_NORMALIZED_EVIDENCE_THEORY_V1.md)、[NEXT71](GRAIN_EVIDENCE_COMPONENT_PLAN_V1.md)。

### 第五十八轮补充：增加内容依赖，仍须正确选择证据

NEXT69以相同1305参数、四源四视图把16Gaussian特征放在汇总前，实际预测B
响应57.81%/53.26%能量离开旧位置子空间。但源坐标SSEzero .993427/.995628
比NEXT68更差；GRAIN作用为.962056/1.058208，确认查询方向相反。新8整读数
r=.143683/.356251，几乎与匹配local一样。按冻结规则关闭二次池化族。

下一连接不是继续增加维度，而是分开核总量Z与相对选择m/Z。保留logZ可从
(m/Z,logZ)恢复(m,logZ)，因此不删除choice。商法则把梯度分为直接内容与
相对所选均值的竞争项，这与NEXT50的原生选择机制接上；实际新函数待NEXT70。
不是声称任意归一化都会增强人格预测，也不把16个核当完整分布的唯一表示。

2拟合/23.445285秒CPU，61相关测试通过；48包/12CSV精确重放，192方向差分
误差1.12e-10。见[报告](../reports/grain_vector_mechanisms/NONLINEAR_POOLING_REPORT_V1.md)、
[理论](GRAIN_NONLINEAR_POOLING_THEORY_V1.md)、[下一计划](GRAIN_NORMALIZED_EVIDENCE_READER_PLAN_V1.md)。

### 第五十七轮补充：已有交互，尚缺局部内容条件化

NEXT68已把源跨材料效应学进同一个评分/导数标量；B坐标响应SSEzero为
.973321/.974318，GRAIN投影作用为.960824/.961599。旧局部对照严格为零。
新8确认总分r=.356->.387但方向误差变大，神经质无提升，不替换完整模型。

从实际公式推出delta_k_B=a+p_B*b；在每个作者独立放松系数的源投影中，
该族也只能表达2.939%/3.034%目标能量。当前已经取得宽松误差减少上限的
90.78%/84.64%，所以继续优化求解或alpha不能补齐范围以外的响应。
这不是人格信息少，而是**未改区域的不同内容必须有不同的响应模板**。

一般式Dphi_B^T Hessian(rho) Dphi_A给出下一机制：把非线性放在逐token
汇总前，使Jacobian依赖局部向量。NEXT69同样16维/136交互系数，只用冻结
源Gaussian特征替换线性矩；两求解，不启动神经训练或新8标签搜索。

55相关测试，48预测包/11CSV精确重放，384方向差分误差1.37e-10；29.464秒CPU。
见[报告](../reports/grain_vector_mechanisms/COUPLED_POOLING_REPORT_V1.md)、
[理论](GRAIN_COUPLED_POOLING_THEORY_V1.md)、[下一计划](GRAIN_NONLINEAR_POOLING_PLAN_V1.md)。

### 第五十六轮补充：同一表达的作用会随其它材料变化

NEXT67在源4作者/两查询中，仅将前半BODY沿冻结GRAIN等权组合方向移动1%，
后半、last、非BODY行逐位相同。原生后半梯度半差/原梯度均值.261%/.314%，
8/8条件非零；32个后半固定方向作用全部改变。相同输入下NEXT66给精确0。
未改原生重放误差0，17调用20.025807秒，无新作者/问卷/模型拟合。

函数关系升级为delta_tau_B=epsilon D_B^T averaged_Hessian_BA D_A。
这连接表达方向、条件、另一表达的读取方式；不是两个最终分数的加权相关。
实际float32端点与有限路径均保存，未把有限差分称为精确点Hessian。

运行后源缓存证实同BODY两查询的B响应余弦均值.708，半差能量15.99%；
在GRAIN四方向上半差占20.65%，余弦范围-.734至.939。提示共享与查询特异
读法在不同投影下会不同，不把这些算术比例当成人格因子的共有方差。
作用小，不能替代完整评分恢复，也没有解释先前全部拟合误差。

51相关测试、5CSV精确重放、收缩恒等式误差1.85e-15。原始论文衔接包括
Integrated Hessians的交互对象、Deep Sets的非线性汇总和Set Transformer的
元素交互。下一NEXT68用对称二次池化项补入跨token混合导数，四数值求解、
180秒CPU、零新增原生调用，不再追加存在性剂量/区域搜索。

见[理论](GRAIN_CROSS_TOKEN_READING_THEORY_V1.md)、
[报告](../reports/grain_vector_mechanisms/CROSS_TOKEN_READING_REPORT_V1.md)、
[NEXT68](GRAIN_COUPLED_POOLING_PLAN_V1.md)。整体goal ACTIVE。

### 第五十五轮补充：低能量不是全部原因，还缺跨材料联合读法

NEXT66同1169系数函数增加源归一化配对变化目标，2次新求解，无新神经/Y。
源变化SSEzero从1.0203/1.0226到1.0031/1.0101，完整新8评分r=.046/.296、
作用SSEzero1.123/1.107，未替换旧模型。原始目标略升、新目标按设计下降，
没有用一项局部改善掩盖另一项退步。

一次源谱分解表明，条件576方向在当前数值分辨率下只覆盖变化目标的
11.34%/8.66%；达到该投影极限的Ridge代价10209/7419。继续加权不能新增
字典外方向。另一方面，当前a(h)+mean f(H_t,p_t,h)对不同BODY行的混合
导数严格零；这与“拥有BODY×last交互”并不矛盾。合成数值见证已核验。

Deep Sets的非线性外层rho与attention的共同分母提示一种具体替代：其它
材料改变当前片段的读取作用。下一NEXT67只改前半BODY，保持后半与last
字节不变，量原生后缀在后半的梯度变化；源4、2查询、17次调用/120秒上限。
原生效应大小尚未测，不能把当前字典外残差全归给这一新假说。

44相关测试，32预测/16CSV精确重放，128FD误差1.62e-10，主CPU11.06秒。
见[报告](../reports/grain_vector_mechanisms/PAIRED_RESPONSE_OBJECTIVE_REPORT_V1.md)、
[理论与一手文献](GRAIN_PAIRED_RESPONSE_OBJECTIVE_THEORY_V1.md)、
[NEXT67](GRAIN_CROSS_TOKEN_READING_PLAN_V1.md)。整体goal ACTIVE。

### 第五十四轮补充：修复几何后，微小响应需要显式的学习优先级

NEXT65保留原始完整last距离，同BODY/源目标/1169参数，只新增2次求解。
源N配置变化r从白化模型.378升至.530、SSEzero.944降至.906，保留局部进展。
新8完整评分r=-.009/.285，完整作用仍不如旧原始模型；不替换旧读取器。

几何可精确分解成源子空间内距离与空间外径向衰减，新8因子.6755/.7393。
恢复距离没有解决全部学习问题。运行后零拟合账本进一步给出严格恒等式：
两端密度误差平均=配对均值误差+配对半差误差。半差仅占源目标能量
.1905%/.4836%，其相对误差仍差于零；平均场拟合本身也不佳。
因此变化小不该被删，而要实验检验是否需要独立的相对精度教学通道。

下一NEXT66保留同一可微scalar/全部输入/原目标，增加一项源单位能量半差
监督；2次求解、120秒CPU，不做调权搜索。GRAIN方向与查询读法仍在同文本
地址相接；心理辅助解释沿用证据绑定，不把数学导数直接视作患者事实。
37focused tests，32包/17CSV精确重放，128FD误差1.79e-10，7.40秒主CPU。

见[报告](../reports/grain_vector_mechanisms/FULL_CONTEXT_GEOMETRY_REPORT_V1.md)、
[理论](GRAIN_FULL_CONTEXT_GEOMETRY_THEORY_V1.md)、
[NEXT66](GRAIN_PAIRED_RESPONSE_OBJECTIVE_PLAN_V1.md)。整体goal ACTIVE。

### 第五十三轮补充：条件函数已实现，源上下文几何需要保留

NEXT64同source4原始/干预教加性与条件scalar，4求解，既有new8应用。
源交互预测RMS只达原生7.54%/8.20%，相关.143/-.371；新8完整GRAIN方向
RMSE比匹配加性低.757%/1.092%，但整评分r=.108/.250不如对照.228/.381，
也不如旧.503/.538。配置query差r=.499/SSEzero.789保留，未全门槛晋级。

结果后代数检查表明：n个源样本满秩中心化PCA白化满足CC'=nI-11'。
n=4时全部源间距离sqrt8，原CV.0774/.0478被消掉；不是新数学定理，
是本次输入处理的具体后果。源3维context只覆盖新8约9.40%/11.03%位移
能量，不能把补空间自动当噪声，也不能把这点归为所有评分退步的已证原因。

13.888663秒CPU、0神经/Y；44测试，32预测与17CSV精确重放，384方向差
<=6.74e-10。下一NEXT65仅改context为完整last距离核，保持源目标、BODY
与系数数量，2新求解，原对照冻结，不做alpha/带宽网格。
[报告](../reports/grain_vector_mechanisms/CONTEXT_MODULATED_POTENTIAL_REPORT_V1.md)、
[理论](GRAIN_CONTEXT_MODULATED_POTENTIAL_THEORY_V1.md)、
[NEXT65](GRAIN_FULL_CONTEXT_GEOMETRY_PLAN_V1.md)。整体goal ACTIVE。

### 第五十二轮补充：原生同正文改读法，混合导数的有限效应

NEXT63仅源4/2query，把last换成同query其它3源last均值，BODY、其余状态、
布局和GRAIN方向不变。last变动约5.35%/5.74%，BODY梯度变化10.45%/16.08%；
32个GRAIN作用全部变动。未改原生重放的评分/梯度/作用差为0；原Hermite
加性函数的BODY/作用变化也精确0，因此特定的结构缺项已被实测，而非只猜。

这是积分混合Hessian沿固定delta_last的作用，不识别全Hessian或人类心理
因果。旧源U8保留约21.2%/25.2%变化能量，动作有符号投影却1.387/.898，
补空间-.387/.102，说明高维能量与任务投影不同，不能删除看似多余的部分。
9后缀反传/11.338838秒，加载3.681338秒；0拟合/新8/Y，37测试、5CSV重放。

下一NEXT64只做一条context乘积核函数与同信息加性对照。源原始+干预目标
均等提供，all4训练/新8应用；源mean干预依赖全部source4，不伪装留一。
4拟合/120秒CPU、0新神经；不继续dose/donor或alpha网格。
[报告](../reports/grain_vector_mechanisms/CONTEXT_READING_INTERACTION_REPORT_V1.md)、
[理论](GRAIN_CONTEXT_READING_INTERACTION_THEORY_V1.md)、
[NEXT64](GRAIN_CONTEXT_MODULATED_POTENTIAL_PLAN_V1.md)。整体goal ACTIVE。

### 第五十一轮补充：局部方向容量兑现一部分，下一缺项是上下文改读法

NEXT62固定原source4/new8、两query与4个GRAIN方向，新增center导数核项。
源字段可表达能量由.039838/.028048增到.260542/.217058，参数593vs81。
新8完整方向误差减少6.29%/7.53%；配置query差SSE/zero .924365->.823856。
确认F03配置作用r=.881370保留，同时F01为负；完整BODY梯度误差也增加，
不能把特定GRAIN投影改进说成整个向量场恢复。

整体L新8r=.503312/.538314略低于旧.533976/.546934，完整作用SSE/zero
1.080006/1.092412，全升级未过。10系统总目标都降低、源字段损失略增；
SVD逆解范数3万量级，容量与稳定可估计性不同。没有alpha或中心搜索。
10拟合/0新神经/Y，11.033820秒CPU、38测试、13CSV/24预测包精确重放。

新旧加性模型都使dK_body/dH_last恒零；H16已有上下文，但其后的读法
无法让末位状态再改变BODY敏感度。NEXT63只在源4用同query其它源last
均值替换当前last，BODY/其它状态固定，实测这个交互后开发条件读取器。
这为用户“同句话的两个侧面”提供同地址、同函数作用接口，不另造临床命名。
[报告](../reports/grain_vector_mechanisms/HERMITE_TOKEN_POTENTIAL_REPORT_V1.md)、
[理论](GRAIN_HERMITE_TOKEN_POTENTIAL_THEORY_V1.md)、
[NEXT63](GRAIN_CONTEXT_READING_INTERACTION_PLAN_V1.md)。整体goal ACTIVE。

### 第五十轮补充：梯度字典表达缺口，不是标量兼容性失败

NEXT61固定20源系统，独立重建加权值/密度设计，col(D)投影与ker(V)约束
分解完成。全源核前置可表达密度能量.039838/.028048，对照.005668/.003973，
约7倍改善保留；但96.0%/97.2%仍在当前有限字典外，占实际字段误差89.4%。
同源评分保持可达同样极限，最大变化1.37e-13；截距/末token锚定rank4/3。
源点可解不等于新作者泛化，字典外目标也不是可删除的人格噪声。

达到极限需要候选系数增量62.64/19.77倍、增加ridge成本590/438，只换来
字段误差改善.113/.115。一阶最优条件误差1.75e-13，优化器已正确求解。
80主诊断+80重放最小二乘，0预测拟合/神经/Y/新8打分；2CSV/20数组精确
重放、49测试、6.367727秒CPU。下一NEXT62新增可积的局部方向核字典，
不再调同一alpha或反复分析旧系数。H16依赖仍在，整体goal ACTIVE。
[报告](../reports/grain_vector_mechanisms/TOKEN_OBJECTIVE_COMPATIBILITY_REPORT_V1.md)、
[理论](GRAIN_TOKEN_OBJECTIVE_COMPATIBILITY_THEORY_V1.md)、
[NEXT62](GRAIN_HERMITE_TOKEN_POTENTIAL_PLAN_V1.md)。

### 第四十九轮补充：token非线性兼容评分，配置作用小幅接回

NEXT60完成20源拟合，同rank8描述/同核中心/同目标，只比两聚合顺序。
新8候选完整L r=.533976/.546934，SSE/源常数=.782701/.744320；对照关系
相近，因此完整评分收益不全归于非线性前置。候选零均值token配置r=.486409/
.086785，query对比.571306；对照该响应为0。完整tau误差下降6.66%/2.13%，
但SSE/zero仍1.152543/1.181391，整体替代不晋级，局部机制继续发展。

源rank8原生投影评价r=.851/.808，固定子空间仍有用；所有源梯度密度loss
已>1。下一步NEXT61是源字典列空间与评分保持零空间分解，不是参数搜索。
其投影仅作源解释，不评估新8以免变成另一轮救分。完整查询差值未过zero
比较也完整保留。7.782211秒CPU、13CSV精确重放、42相关测试、0神经/Y。
真实192方向有限差分误差3.07e-10。H16仍神经，当前teacher无回填。
[报告](../reports/grain_vector_mechanisms/TOKEN_POTENTIAL_EXTRACTION_REPORT_V1.md)、
[理论](GRAIN_TOKEN_POTENTIAL_EXTRACTION_THEORY_V1.md)、
[NEXT61](GRAIN_TOKEN_OBJECTIVE_COMPATIBILITY_PLAN_V1.md)。整体goal ACTIVE。

### 第四十八轮补充：旧输入零空间承载主要原生GRAIN响应

NEXT59同12作者/24上下文/96作用，分D为源PCA均值、零均值token分配和
PCA外均值，使用原生完整H16梯度收缩。新8分配项对响应投影.945894/.943346，
可见项.058362/.029880；不可见合计平方比.893104/.949316。所有96方向的
不可见作用非零，minabs.005527，收缩相加误差1.33e-15。四因子都保留，
F03/N均值作用更大；不能把平均分配投影说成每个因子的人格比例。

向量能量分配约99%受方向构造影响，实际梯度对齐才是新证据。零均值配置
也不必等于纯词序。本轮把旧S6与NEXT31性质定位到NEXT57失败的输入，不
将早期版本视作失效或重复宣称首创。任何f(phi)的核方向导数必为0，故应
改变聚合前输入，而不是继续在旧phi上换非线性头。

5CSV精确重放、49测试，10.699698秒CPU/0拟合/神经/Y。NEXT60拟将源梯度
教学与非线性token标量核结合，匹配聚合前/后的两种同函数读法。计划尚未执行。
[报告](../reports/grain_vector_mechanisms/POOLED_FEATURE_NULL_RESPONSE_REPORT_V1.md)、
[理论](GRAIN_POOLED_FEATURE_NULL_RESPONSE_THEORY_V1.md)、
[NEXT60](GRAIN_TOKEN_POTENTIAL_EXTRACTION_PLAN_V1.md)。整体goal ACTIVE，无模型任务。

### 第四十七轮补充：源补偿已定位，下一步检验输入抹去的作用

NEXT58对冻结value/joint的DeltaL=I+B+Q精确计算：源4构造内B/Q相关近-1，
kappa为1.027e-6/3.660e-7；新8为-.235/-.161、kappa=.849/.866，留一略放大。
这解释该数值程序为何能维持源值同时学斜率，却换人后失配，不是直接宣布
原生模型或心理系统采用同样补偿。新8评分行覆盖.346/.521，方向.796–.877，
源支持对象确实不同；F04覆盖最低但原始相关最高，覆盖不等于品质。

去除作者共同四作用水平后N F04从.639保留.629，确认寻求.610降至.220。
两者幅度误差仍大；中心化是另一个统计对象，不是删掉人格噪声。
11CSV精确重放、42测试，5.249秒CPU/0拟合/神经/Y。新增7公开聚合表。

下一步区分数据支持与输入零空间：把D分为源PCA均值、零均值token分配、
PCA外均值，用已存完整H16梯度量化原生作用。后两项对池化phi导数为0，
若原生作用非零就应改变输入，而非在相同phi更换非线性头。尚未执行该检验。
[报告](../reports/grain_vector_mechanisms/VALUE_TANGENT_COMPENSATION_REPORT_V1.md)、
[理论](GRAIN_VALUE_TANGENT_COMPENSATION_THEORY_V1.md)、
[NEXT59](GRAIN_POOLED_FEATURE_NULL_RESPONSE_PLAN_V1.md)。整体goal ACTIVE，无模型任务。

### 第四十六轮补充：同函数响应可学习，作者评分关系需另行保存

NEXT57用源原生L及四个GRAIN导数教同一个标量函数，应用没有suffix或当前
答案回填。20次小求解，正文均值/末位PCA32特征固定。value新8分数r=.300/
.336，joint=-.782/-.473；joint F04响应r=.639/.610，全部因子保留报告。
源导数近拟合没有自动保存新作者间评分，这个差别是新的机制设计线索。

源数据设计rank20/65，LOO15；Ridge的唯一解不是源观察唯一确定的机制。
BODY方向的末位特征导数为0，下一步分解正文斜率与查询项是否在源样本中
互相补偿。这里只提出可检验机制，不把可能的补偿当已发现因果链。
17CSV精确重放，192导数核验，36测试；0新神经/Y。文件枚举/审计精度错误
已记录并修正，源拟合与预测不变。关闭池化加性族，不再做同12人的参数网格。

[报告](../reports/grain_vector_mechanisms/COMPLETE_FUNCTION_EXTRACTION_REPORT_V1.md)、
[理论](GRAIN_COMPLETE_FUNCTION_EXTRACTION_THEORY_V1.md)、
[NEXT58](GRAIN_VALUE_TANGENT_COMPENSATION_PLAN_V1.md)。整体goal ACTIVE，无模型任务。

### 第四十五轮补充：实际源截距控制了完整函数的读取敏感度

NEXT56固定四兼容函数I=X、O=X+b、V=X+V、F=X+b+V。新8 O梯度与F余弦
.9998，V与I为.9993；源截距/总校正能量约99.91%。这将上一轮状态位置诊断
进一步定位到实际源系数。V查询对比r=.811、SSE/zero=.351，优于I的.774/.404，
四因子对比误差均改善但符号17/32未变。源4对比仍负，完整分数门槛仍未过。

共同b无当前作者间方差，却能经grad S(X+b)改变差异的读法；这是计算机制，
不是把b命名为topic或心理噪声。隐藏状态替代家族关闭，NEXT57改学完整L和
同函数GRAIN导数，尝试用源数值程序移除当前后缀，而不继续调偏移/基底。

48新反传/39.67秒，26CSV精确重放，41测试；0新拟合/API/Y。见
[报告](../reports/grain_vector_mechanisms/SOURCE_OFFSET_MECHANISM_REPORT_V1.md)、
[理论](GRAIN_SOURCE_OFFSET_MECHANISM_THEORY_V1.md)、
[NEXT57](GRAIN_COMPLETE_FUNCTION_EXTRACTION_PLAN_V1.md)。整体goal ACTIVE，无模型任务。

### 第四十四轮补充：共同位移改变了个体差异的读取位置

NEXT55将三个固定Psi及同函数导数接真实suffix22–end。新8联合分数相关
.548/.420，SD却只有.056/.057对比原生.904/1.039；完整GRAIN响应r=.049/
.022。三候选均未过整函数门槛；identity的查询间响应对比仍r=.774，保留其
作用范围。源4查询对比为负，同样报告。这不是问卷预测r，旧NEXT53组件有效。

结果后冻结的I/J缓存交叉显示，I状态配J方向保持对比r=.775，J状态配I方向
则.057。方向仅改1.82%，梯度范数却降到约20%、方向余弦约.20–.25，故损失
主要来自状态移动后的读法。精确分账delta_tau=delta_g:dbar+gbar:delta_d。
token恒定位移能量99.95%不是源截距份额；NEXT56才分离实际源截距和其它项。

72实际反传/54.34秒，16主CSV重放；交叉24数组14CSV精确重放，0额外神经调用；
37测试。见[报告](../reports/grain_vector_mechanisms/WHOLE_PATH_ASSEMBLY_REPORT_V1.md)、
[理论](GRAIN_WHOLE_PATH_ASSEMBLY_THEORY_V1.md)、
[NEXT56](GRAIN_SOURCE_OFFSET_MECHANISM_PLAN_V1.md)。整体goal ACTIVE，无运行中模型。

### 第四十三轮补充：教变化也改变了状态依赖的读法

NEXT54固定VV/VJ/JV/JJ，0拟合/神经/Y，解释而非再次提高NEXT53的结果。
A(X,d)=ell(X)^T d，差异精确等于delta_ell^T dbar+ellbar^T delta_d。
既有新8误差改善中80.4–83.4%由状态路线分得、16.6–19.6%由方向分得，交互
RMS约为总改变量1%。状态平均距离略变差不妨碍功能更准，因为度量不同。
所有新8单query/contrast因子改善；源4对比却变差0.90–1.44%，完整保留。

这是对称计算记账，不是人格静动份额；交叉组合不是新函数导数。12.595秒
CPU，24包12CSV精确重放，34测试通过。当前固定基底家族结束。
[NEXT55](GRAIN_WHOLE_PATH_ASSEMBLY_PLAN_V1.md)计划冻结pointwise联合/值对照/
identity后接实际后缀22–end，对整个固定L和四方向响应测一次，不调局部参数。
见[报告](../reports/grain_vector_mechanisms/RESPONSE_REPAIR_REPORT_V1.md)、
[理论](GRAIN_RESPONSE_REPAIR_THEORY_V1.md)。全系统goal ACTIVE，暂无模型任务。

### 第四十二轮补充：源响应教学产生可保留的数值组件

NEXT53联合点式/因果图的新8N/C组件r约.900/.924，原定两query替代门槛通过。
运行前发现损失归一化改变正则化相对强度，补同归一化值对照后发现两种效应：
仅归一化对照本身已可用；GRAIN方向教学再降新8端点误差14.8–19.1%，全方向
误差降1.09–1.12%，状态误差略增约0.26%。源4端点依然弱，没有混成总体成功。

原source基底/尺度/引子/读头不变；同B生成状态与方向。截距维持，非恒定
系数大幅缩小。GRAIN可以规定源函数应保留的扰动分布Sigma_G；这是有实际
计算结果的接口，不是两个问卷分数相加。组件输入已无需当前blocks17–21，
H16仍原生，输出不是完整人格分数。20solves/13.150秒、0神经/Y；48包25CSV
精确复算，49测试通过。NEXT54做一次固定状态/方向交叉解释修复，然后回整链。
见[报告](../reports/grain_vector_mechanisms/RESPONSE_AWARE_STATE_REPORT_V1.md)、
[理论](GRAIN_RESPONSE_AWARE_STATE_THEORY_V1.md)、
[NEXT54](GRAIN_RESPONSE_REPAIR_MECHANISM_PLAN_V1.md)。整体goal ACTIVE。

### 第四十一轮补充：相似位置与相似运动不是同一个学习目标

NEXT52只拟合源作者H16->pre22状态，同一函数解析生成方向。新8点式/因果
状态误差降47%/42%，方向误差却增39%/50%；identity保留N/确认组件r=.878/
.893但幅度约4倍。三个替代均未通过原定门槛，旧NEXT51原生输入组件保留。
方向分解发现源PCA32覆盖31.94%的GRAIN扰动能量；基底盲区的方向必原样通过。
这不是心理信息比例，而是当前函数家族的可学习范围。

由此连接GRAIN与路径的一种新方式是Sigma_G=sum D_f D_f^T/4，以
tr((Jhat-J)Sigma_G(Jhat-J)^T)度量需保留的方向响应。NEXT53固定同基底和参数，
只改为源值+方向联合目标；不是新8标签调参，也不是两个总分简单相加。
7.185秒CPU、10solves、0神经/Y；24包/9CSV精确复算，44测试通过。
见[报告](../reports/grain_vector_mechanisms/SOURCE_STATE_TRANSPORT_REPORT_V1.md)、
[理论](GRAIN_SOURCE_STATE_TRANSPORT_THEORY_V1.md)、
[NEXT53](GRAIN_RESPONSE_AWARE_STATE_PLAN_V1.md)。全系统goal ACTIVE。

### 第四十轮补充：选择的作用抵达端点，意义依赖后续配对

NEXT51固定源query引子和后续读头，在新8的N/确认端点key组件r=.973/.978，
对比r=.963、RMS59.2%，两项原定门槛均过。源4对比=.921，完整因子保留。
局部w*下交换引子的-.890，经实际端点读头变为+.868、源读头+.447：
局部结果有效但不能推及所有读法，query作用不是只在一个引子内。

选择/读头/交叉MSE=.007729/.000777/-.004120，负交叉项说明固定读头
可部分补偿选择近似。原生其它项加回后的r=.995仅为替换审计，不作独立预测。
24reverse+2JVP+4FD共20.365秒；完整96个H16作用与原模型差≤4.777e-6；
24向量/12预测/7CSV重放，67测试。NEXT52转向Psi(X16)及其同函数导数，
从当前K/V/dK移除blocks17–21，不继续局部层/头/读头精度搜索。见
[理论](GRAIN_SELECTION_ENDPOINT_THEORY_V1.md)、
[报告](../reports/grain_vector_mechanisms/SELECTION_ENDPOINT_REPORT_V1.md)、
[NEXT52](GRAIN_SOURCE_STATE_TRANSPORT_PLAN_V1.md)。

### 第三十九轮补充：可共享查询引子，但查询差别需要相应选择

NEXT50仅从源4编译Q引子，预测不读当前Q。同/另/共享query引子的新8对比
r=.939345/-.890468/.477162；同query31/32同号，RMS为原生29.1%，共享
为8.4%。三者各query的原定SSE/zero门槛均过，不能把对比差别偷偷改成
原来的通过门槛。保留完整因子/source/pooled表，不再局部调参。

该式把同一材料的GRAIN方向接到token级选择作用，查询引子决定该变化如何
参与证据竞争；当前K/V/dK仍依赖query-conditioned原生状态。它为辅助解释
提供数值/文本共同地址，不自动命名精神分析或交流分析构念。

576动作/288对比、12作者，0模型/拟合/Y，.373秒及.430秒重放；12数组和
6CSV完全重放，62相关测试通过。下一步r^T u的端点贡献与源rbar替代：
不能因局部w*^T u相近就跳过后续运输。见[理论](GRAIN_SOURCE_QUERY_SELECTION_THEORY_V1.md)、
[报告](../reports/grain_vector_mechanisms/SOURCE_QUERY_SELECTION_REPORT_V1.md)、
[NEXT51](GRAIN_SELECTION_ENDPOINT_PLAN_V1.md)。

### 第三十八轮补充：同材料方向不仅搬运内容，还改变证据选择

NEXT49以完整preAttention22状态/方向展开Q/K/V。新8自身局部对比投影
=.083093/.903954/.012952；source4=.561950/.271327/.166722。神经质
查询新8的Q=-.125849、K=1.125420；确认寻求则Q=.444047、K=.452388。
因此查询可调整已有选择，不能把旧BODY-only限制硬套在传播后的方向上。

96前缀JVP用77.972秒，96原生增量向量相对差≤5.272e-6；0评分/拟合/Y。
最初报表的key/query身份列冲突已在r2明确修复，预测与数值对比未改、未重跑。
12数组/6CSV/1JSON重放，116测试通过。NEXT50只用源固定query引子检验
当前证据K/V/dK的选择读法，随后回整链，而非继续层/头网格。见
[理论](GRAIN_UPSTREAM_ATTENTION_CHANNELS_THEORY_V1.md)、
[报告](../reports/grain_vector_mechanisms/UPSTREAM_ATTENTION_CHANNELS_REPORT_V1.md)、
[NEXT50](GRAIN_SOURCE_QUERY_SELECTION_PLAN_V1.md)。

### 第三十七轮补充：查询特异修正也可编译，回到完整上游运输

NEXT48仅从源4编译分查询读向量，新8的修正项与当前算子差异r=.886208，
总对比r=.976535、SSE/zero=.060734。公共项与NEXT47完全一致，8新作者
使用相同分query/mode向量；pooled也通过。源4 F02的修正SSE略高于零
基线，完整保留；没有用人格标签挑选或重新拟合。

新8源误差RMS=.013454、查询误差=.012086，总误差=.023478。误差及
平方误差变化恒等式均核验，不能把相关误差直接相加。4.078秒数值、0模型
调用，12数组/8CSV重放、105测试通过。MLP24局部细调按预定规则结束。

下一步是真正仍需当前材料的全token传输。旧末端BODY-only扰动固定最终
query，而attention22的传入方向可能已改变它；NEXT49完整展开Q/K/V，
保留旧公式边界，检验新的上游选择/写入机制。详见
[理论](GRAIN_SOURCE_QUERY_CORRECTION_THEORY_V1.md)、
[报告](../reports/grain_vector_mechanisms/SOURCE_QUERY_CORRECTION_REPORT_V1.md)、
[全链接合图](GRAIN_UPSTREAM_MECHANISM_INTEGRATION_V1.md)。

### 第三十六轮补充：局部计算可共享，作者材料仍以条件化方向进入

NEXT47仅用旧4作者编译源共同读向量，8个新作者使用完全相同的v，仍保留
MLP24回调：自身模式对比r=.955274、SSE/zero=.091333，最终T投影-.304183。
没有用当前x或人格标签。这把当前门控依赖从此处局部读法中移除，但没有
假装上游d已不需模型。源4留一与pooled、所有因子结果均保留。

剩余误差明确分成源替换、遗漏的查询项和数值余项。新8前两者RMS为
.013454/.024635，后者1.84e-7。下一步只问较大的查询项是否也能从源编译，
之后结束此层的细调，回到完整文本输入链。

4.069秒，0原生调用/拟合/Y；101聚焦测试、12数组与6CSV重放通过。见
[理论](GRAIN_SOURCE_COMMON_COUNTERACTION_THEORY_V1.md)、
[报告](../reports/grain_vector_mechanisms/SOURCE_COMMON_COUNTERACTION_REPORT_V1.md)。

### 第三十五轮补充：回调主要作用于已有差异，算子变化并非同一个方向

NEXT46真实block24的局部2x2交换完成，新8自身模式传入/算子项RMS为
.090579/.024635，局部投影.918570/.081430。相对最终T则为
-.313540/+.067912，合计-.245629。这把“负向更新”细化成可计算的两个来源，
而不是把整个模块当作噪声；源4和pooled也保留总体方向，但因子细节不同。

两个查询的平均J仍按当前作者x计算，并非所有人共享一个固定J。下一步
NEXT47以源4构造共同读向量，检验能否去掉这一处当前状态的依赖。该检查
不需要新神经调用，不改变GRAIN计分或心理解释规则。

28前缀执行14.103秒、加载3.492秒；完整96对角向量通过，3CSV及12数组
重放一致，96聚焦测试通过。详见[新理论](GRAIN_QUERY_COUNTERACTION_THEORY_V1.md)
与[报告](../reports/grain_vector_mechanisms/QUERY_COUNTERACTION_REPORT_V1.md)。

### 第三十四轮补充：路径差异会被增强，也会被后续更新修订

NEXT45将NEXT44的运输差异T精确接到17个前置原生更新。新8人在层23
累积RMS=.208，最终preMLP25为.163；MLP24有负投影-.246，四因子均为负。
源4也有MLP24反向作用，但attention22的作用不同，不能推广固定层排名。

发展出的精确递推为delta_d_next=Abar delta_d+delta_A dbar，区分已有
差异的传输与当前算子对比。矩阵顺序保留，不要求可逆/可交换；两项测试
验证代数，尚不等于从末位更新识别了全部完整Jacobian。NEXT46只测试MLP24。

[报告](../reports/grain_vector_mechanisms/QUERY_TRANSPORT_FORMATION_REPORT_V1.md)、
[理论](GRAIN_QUERY_TRANSPORT_FORMATION_THEORY_V1.md)、[NEXT46](GRAIN_QUERY_COUNTERACTION_PLAN_V1.md)。
1728有符号记录、.245秒、0新神经调用、12CSV重放、93测试；整体goal ACTIVE。

表内历史证据简称的可定位入口：
- unified V8：[总体系](SUICA_UNIFIED_THEORY_SYSTEM_V8.md)；4W：[解释链](SUICA_4W_PERSON_INTERPRETER_THEORY.md)。
- rind：[独立库实证](../reports/suica_rind_regime_test_v3.md)；原理论在 parent
  `../project persona/docs/SUICA_RIND_THEORY_BASE_V3.md`，属于历史资料位置，不是当前新实验。
- M3：[机制类型](SUICA_M3_MECHANISM_ATLAS_THEORY.md)；M4：[机制组合](SUICA_M4_MECHANISM_COMPOSITION_THEORY.md)。
- cross-level：[跨层连接](SUICA_CROSS_LEVEL_INFERENCE_CONTRACT.md)。
- NAV：[真实/合成读出参照](../reports/grain_vector_mechanisms/NAVIGATION_REFERENCE_ROUTE_REPORT.md)。
- all-factor：[查询函数总表](GRAIN_ALL_FACTOR_MATHEMATICAL_ATLAS_M31.md)。
- factor/path：[四路线实证](../reports/grain_vector_mechanisms/FACTOR_PATH_ALL_ROUTES_REPORT.md)。
- LLM assessment：[观察-表示-计分-解释](V8_LLM_NATIVE_ASSESSMENT_THEORY.md)。

- [逻辑研究计划与活跃问题](GRAIN_LOGIC_RESEARCH_PLAN_V1.md)
- [参考理论](GRAIN_REFERENCE_FUNCTION_FRAMEWORK_V1.md)
- [条件计算及实际依赖](GRAIN_CONDITIONAL_NAVIGATION_THEORY_V1.md)
- [全系统输入依赖与嵌套端点](GRAIN_CONDITIONAL_INPUT_DEPENDENCIES_V1.md)
- [V8 类型体系](SUICA_UNIFIED_THEORY_SYSTEM_V8.md)
- [4W](SUICA_4W_PERSON_INTERPRETER_THEORY.md)
- [动态与效标通道](../reports/grain_vector_mechanisms/DYNAMIC_PATH_CHANNEL_REPORT.md)

本文是当前逻辑总入口；报告正文不会自动“证明”登记的每条边。新的实验将更新
连接状态与实例，不把有限清单的完成写成穷尽全部可能性。
