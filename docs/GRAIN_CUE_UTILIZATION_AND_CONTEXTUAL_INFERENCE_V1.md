# GRAIN：从线索可读性到情境化作者推论

日期：2026-09-07。状态：理论推导与合成反例完成；真实作者机制尚待原队列结果。

本轮按用户要求暂停人工盯盘，不中断、不扩充、不重新设定正在运行的长任务。
本文补充 [向量可读性](GRAIN_VECTOR_READABILITY_THEORY_NOTE_V1.md) 与
[4W 理论](SUICA_4W_PERSON_INTERPRETER_THEORY.md)，不是替换它们，也不是建立一张成品量表。

## 1. 本轮要补上的机制

“文本里有没有可用于人格推论的线索”和“LLM 是否按恰当方式使用它”不是一个问题。
至少需要区分：

| 对象 | 本轮数学化方式 | 不能直接等同于 |
|---|---|---|
| 线索与外部目标的关系 | 线索条件下的目标预测函数 | 完整人格真值 |
| LLM 对线索的使用 | 同一线索条件下的原生输出函数 | 训练后统计头的读出 |
| 两者是否方向一致 | 两个函数的协方差几何 | embedding 总信息量 |
| 历史如何贡献证据 | 依赖历史的条件似然比更新 | 记忆长度或重复次数 |
| 未命名作者配置是什么 | 对一组条件的预测响应函数 | 已发现的心理特质 |

Funder 的 Realistic Accuracy Model 区分线索的相关性、可获得性、检测与利用；
它提供的是判断过程框架，不允许把几个相关系数相乘就宣布识别了各阶段成功率。
这与 GRAIN 当前“可读但未必被原生使用”的问题相接，而非证明 LLM 已复制人的判断过程。
[Funder, 1995](https://rap.ucr.edu/FunderPR1995.pdf)

## 2. 线索利用的协方差分解

### 2.1 先固定比较对象

在参照框架 \(F\) 下，令 \(Z\) 为声明过的线索表示，\(Y\) 为外部量表或行为指标，
\(S\) 为经过明确答案编码的 LLM 原生分数。三者来自同一组配对作者。
\(Y\) 是有误差、有限适用范围的观测指标，不是无误差的潜在人格。

假设二阶矩存在且 \(Y,S\) 的方差非零，定义总体条件期望：

\[
m_Y(Z)=E[Y\mid Z,F],\qquad m_S(Z)=E[S\mid Z,F],
\qquad e_Y=Y-m_Y,\quad e_S=S-m_S.
\]

由 \(E[e_Y\mid Z,F]=E[e_S\mid Z,F]=0\)，得到标准协方差恒等式：

\[
\operatorname{Cov}(Y,S)=\operatorname{Cov}(m_Y,m_S)
                         +\operatorname{Cov}(e_Y,e_S).
\]

定义 \(R_Y=\sigma_{m_Y}/\sigma_Y\)、\(R_S=\sigma_{m_S}/\sigma_S\)，
以及 \(G=\operatorname{Corr}(m_Y,m_S)\)、\(C=\operatorname{Corr}(e_Y,e_S)\)，则

\[
r_{YS}=G R_Y R_S+C\sqrt{1-R_Y^2}\sqrt{1-R_S^2}.
\]

这是 lens-model 类分解，不是 GRAIN 的新定理。经典线性版本也可由**同一样本、
相同列空间、含截距的正交 OLS 投影**得到；此时 \(R\) 只描述该线性空间的解释力。
零方差分量对应的相关未定义，应报告其协方差贡献为零，不能任意赋一个相关值。
原始 Tucker 文献此轮只核实书目信息；现代方法文献及这里的直接代数推导用于公式核对。
[Tucker, 1964](https://pubmed.ncbi.nlm.nih.gov/14216901/)，
[Hamm & Yang, 2017; online 2016](https://doi.org/10.1002/bdm.1969)

关键解释：

- \(R_Y\)：所选线索对目标的可预测性。受表示、函数类、样本影响，不是“人格预测天花板”。
- \(R_S\)：所选线索对 LLM 输出的可解释性。不是跨日重测信度，也不是答案编码不变性。
- \(G\)：两种线索使用函数的对齐程度；可读性高并不迫使 \(G\) 高。
- \(C\)：未被当前线索模型解释的共同变动；不能自动命名为隐性人格或 reasoning。

### 2.2 “方向”应在协方差度量下比较

若 \(m_Y=\beta_Y^T Z\)、\(m_S=\beta_S^T Z\)，则

\[
G=\frac{\beta_Y^T\Sigma_Z\beta_S}
        {\sqrt{\beta_Y^T\Sigma_Z\beta_Y}\sqrt{\beta_S^T\Sigma_Z\beta_S}}.
\]

它不是裸系数的余弦。对可逆坐标变换及相应系数运输，两个预测函数和这个 \(G\)
不变；更换坐标轴名称不会创造作者信息。PCA 截断不是可逆变换，且重新施加 Ridge
惩罚可能改变函数，所以这种不变性不能替实际模型的压缩比较作结论。

### 2.3 当前 Ridge / cross-fit 管线必须保留四项

对任意已拟合预测 \(\widehat m_Y,\widehat m_S\)，定义剩余项为实际值减预测值，则总有：

\[
\begin{aligned}
\operatorname{Cov}(Y,S)={}&\operatorname{Cov}(\widehat m_Y,\widehat m_S)
 +\operatorname{Cov}(e_Y,e_S)\\
 &+\operatorname{Cov}(\widehat m_Y,e_S)
 +\operatorname{Cov}(e_Y,\widehat m_S).
\end{aligned}
\]

四项均除以 \(\sigma_Y\sigma_S\) 即严格闭合为实际 Pearson 相关。
收缩估计、非线性拟合、不同线索空间以及折外预测，通常都不满足两项交叉协方差为零。
同一训练折的 OLS 正交性也不自动转移到测试作者。Hamm 与 Yang 对二值判断的讨论
同样说明，离开普通线性投影后需要保留交叉项；这里是这一原则的 GRAIN 应用。

一个构造反例：\(Y=S=X\)，但两个预测函数均为 \(0.5X\)。四项各为 \(0.25\)，
真实 \(r=1\)；只保留前两项会错误地得到 \(0.5\)。这不是数据泄漏或模型新发现，
而是“把收缩剩余误差当正交残差”的公式错误。

四项是**有符号的统计账目**，可以相互抵消，不是互斥的因果贡献百分比。
应逐折报告并区别 pooled 与 fold-average；由总体条件期望导出的 \(R\leq1\) 和
平方根公式不能直接套到任意折外预测的标准差比上。

## 3. MBTI 的概率质量不能由一个相关代替

对二值指标 \(Y\) 和概率输出 \(S\in[0,1]\)，令 \(q(S)=P(Y=1\mid S,F)\)。标准
Brier 分解为：

\[
E[(Y-S)^2]=\underbrace{\operatorname{Var}(Y)}_{\text{uncertainty}}
-\underbrace{\operatorname{Var}(q(S))}_{\text{resolution}}
+\underbrace{E[(q(S)-S)^2]}_{\text{calibration error}}.
\]

分辨率、校准与阈值分类是不同对象。文献中的 reliability 项在这里指预测校准误差，
不是心理测量的重测信度。有限数据分箱并不等于知道总体 \(q\)，重新拟合校准器须在
source-inner 内进行；本轮不拟合校准器，也不改变现有采集的答案解码。
[Murphy, 1973](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2)

合成的两个等频群组：真实概率为 \((.2,.8)\)，预测概率为 \((.1,.4)\)。
AUC 为 \(.8\)，但 \(.5\) 阈值下 balanced accuracy 为 \(.5\)；Brier 为 \(.245\)，
略好于总体先验的 \(.25\)，却仍有 \(.085\) 校准误差。这解释**可能出现什么现象**，
不把当前 MBTI 实验归因为这个特定构造。Lens 的 Pearson 恒等式也不是 AUC 分解。

## 4. 历史与情境：保留选择证据，防止重复证据虚增

沿用 4W，令 \(c_i\) 为作者面对/选择的条件，\(O_i\) 为机会集合，\(H_i\) 为此前历史，
\(x_i\) 为表达，\(\theta\) 暂指待解释的作者配置。固定框架与外部机会序列，链式分解为：

\[
p(D\mid\theta,F)=\prod_i
 p(c_i\mid O_i,H_i,\theta,F)
 p(x_i\mid c_i,O_i,H_i,\theta,F).
\]

完整条件历史下这是概率链式规则，不需要把句子假定为相互独立。若机会集合本身也
由作者选择，则还需对它建模，不能在条件化时偷偷丢掉那部分选择证据。
对两个候选配置 \(\theta_a,\theta_b\)，后验 log odds 是先验 log odds 加上每次
**选择似然比**与**条件表达似然比**的对数。

这给出原 4W 的推论含义：自选 topic/situation 可能提供证据，不能一律残差化去掉；
真正外部固定或随机分派、且条件历史下不依赖配置的选择机制，其似然比才可相消。
AI 根据历史自适应提问与作者自选话题应分开记录，不能默认同一机会机制。

一条证据 LR=4、先验各半，后验为 \(.8\)。完全复制这条证据，条件 LR=1，后验仍为
\(.8\)；误当两个独立证据会得到 \(16/17\)。因此重复记忆或助手复述不能按独立作者
开示累计。自然文本中的依赖强度需要估计，不能把所有相近内容都当完全重复。

“LLM 在上下文中进行这种近似更新”是**候选机制**。Xie 等在特定预训练生成假设下
给出隐式 Bayesian inference 的解释，不能据此宣布现实 LLM 是正确的 Bayesian
人格推论器。行为似然、内部实现、外部目标测量模型仍是三层不同对象。
[Xie et al., 2022](https://arxiv.org/abs/2111.02080)

## 5. 不可命名的作者变量：先定义为条件响应函数

比“找一根人格轴”更宽的候选对象是：给定同一机会/查询族 \(\mathcal Q_F\)，作者
历史所支持的未来响应分布。先固定有限的行为事件或数值函数 \(\tau\)，可写成

\[
\Psi_F(D)(q)=E[\tau(X_{\text{future}})\mid D,q,F].
\]

在规定的 \(q,\tau\) 家族下，如果两份历史诱导相同响应分布，则它们在该观察制度下
预测等价；有限个均值相同还不能推出完整分布相同。这个定义不要求各个坐标立刻有
心理学名称，也不要求不同 embedding 的轴方向一致。

一个可估计的比较量为

\[
d_F^2(u,v)=\int_{\mathcal Q_F}
 \|\Psi_F(D_u)(q)-\Psi_F(D_v)(q)\|^2\,d\omega_F(q).
\]

\(\omega_F\) 是事先声明的共同机会权重；\(\tau\) 的尺度也必须固定。这通常只是作者
历史间的伪距离：不同的人可以在有限观察下得零距离。响应函数可能同时包含稳定作者
差异、当前状态和历史适应；只有分离这些来源后，才谈 trait 或 state 的心理解释。

Predictive State Representations 用未来 action-observation tests 的预测概率表征
动态状态；computational mechanics 按未来预测分布组织历史等价类。这为“先定义
运动对象的可观测作用，再寻找坐标”提供先例，而非证明人类人格就是一个有限状态系统。
原 PSR 的有限 POMDP 表征结果不能直接移植到无限自然语言与开放社会条件。
[Littman, Sutton, & Singh, 2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)，
[Shalizi & Crutchfield, 2001](https://csc.ucdavis.edu/~cmg/compmech/pubs/cmppss.htm)

这里的实验落点不是要求同一个人每天复述同一话题，而是区分：**历史能否改善共同
条件下新表达的预测，改善的是总体语言延续还是作者特有响应？** 自然历史中的条件
分布不等于随机干预；在没有共同支持的场景上，只能给出模型外推，不能声称识别了它。

LLM 的 task vector / function vector 文献还提示另一种对象：表示“如何完成任务”的
操作与表示“这个作者是谁”的配置可能不同。当前工作应分离作者证据、任务条件和答案
编码，而非把 prompt 造成的激活差直接叫作者人格向量。
[Hendel et al., 2023](https://aclanthology.org/2023.findings-emnlp.624/)，
[Todd et al., 2024](https://arxiv.org/abs/2310.15213)

## 6. 已运行的九个数学见证

全部来自确定性构造/固定种子的正交向量；没有读取任何作者、量表或模型缓存。

| 构造 | 结果 | 所排除的错误推断 |
|---|---|---|
| 相同线索可读性，对齐 | \(R_Y=R_S=.8,\ G=1,\ r=.64\) | 这是一个可能世界，不是实际效果 |
| 同样可读性，正交 | \(G=0,\ r\approx0\) | 可读性高必然原生预测高 |
| 同样可读性，反向 | \(G=-1,\ r=-.64\) | 只需增加向量信号强度 |
| 可逆线索坐标变换 | 预测函数与 \(G\) 不变 | 裸坐标方向就是心理机制 |
| 共同线索未建模 | 总 \(r=1\)，残差项贡献 \(.96\) | 大残差相关必然是新能力 |
| 加入该共同线索 | 总 \(r=1\)，残差项约零 | 残差拥有固定不变的心理含义 |
| 两个预测同时收缩一半 | 四项各 \(.25\)，少两项误差 \(.5\) | Ridge 剩余误差天然正交 |
| 两群概率低于阈值 | AUC \(.8\)，bal acc \(.5\)，Brier \(.245\) | 单一阈值概括全部概率信息 |
| 同一证据完全重复 | 正确后验 \(.8\)，错算独立时 \(.941176\) | 更多重复记忆就是更多独立证据 |

最大协方差恒等式数值闭合误差：\(2.220446049250313\times10^{-16}\)。
这验证代数与代码一致性，**不是新定理的发现，也不是已识别真人心理机制**。

重播入口：

```bash
"/Volumes/mobile3/projects/project persona/.venv/bin/python" \
  scripts/check_grain_cue_utilization_theory.py \
  --output-dir results/grain_vector_mechanisms/cue_utilization_theory_replay
"/Volumes/mobile3/projects/project persona/.venv/bin/python" \
  -m pytest -q tests/test_grain_cue_utilization_theory.py
```

输出目录必须尚不存在。原始 packet 为
`results/grain_vector_mechanisms/cue_utilization_theory_r1/`，包含 JSON 数值、代码与输出 SHA256。
数据目录仍遵循 repo 的 ignored-output 规则；本表和重播代码进入版本控制。

## 7. 对当前研究路线的具体改变

**已知但尚不充分：** 已完成的向量 source probes 衡量可读性，原生语义 panel 衡量
输出函数对行为线索与答案编码的反应；两者还不是同一配对作者上的完整 lens 分解。
当前 direct cache 保存候选 logits，不是九个 trait prompt 的完整隐藏状态；secondary
缓存的两个 social-initiation prompts 也不能替代那九个任务。不能跨 prompt/pooling
直接相减并宣称发现 trait circuit。

**长跑完成后的最小分析顺序：**

1. 先完成既定 direct/source 与 history/task 比较，不以部分结果改网格或解码。
2. 在共同 source 作者、相同条件与参考框架下，预先声明线索 \(Z\)。对目标与 native
   score 分别建立 source-inner 选择的折外预测；逐折输出四项，不先假定交叉项为零。
   当前头只有目标预测时，不能伪装成已经有 native-score 的预测函数。
3. 同时比较概率排序、proper loss 与答案编码敏感性。它们共同定位“信息未读出”
   与“读出后映射不稳”，但还不构成神经因果证明。
4. 若历史增益成立，下一机制问题是相关作者证据、语言重复还是条件预测更新；优先
   使用既定 own/matched/no-history 控制。只有这些不足时，再单独提议一个响应函数
   预测实验，不在本轮新增全量采集、模型训练或未授权量表连接。

本轮新增贡献是把**线索利用对齐、条件证据更新、预测响应函数**接入同一理论路径，
并明确哪些量可以用已有输出继续估计。剩余经验缺口不会由合成恒等式自动消失。

## 8. 与 Claude 的后续研讨

同日[三轮研讨](../reports/grain_vector_mechanisms/CODEX_CLAUDE_VECTOR_SEMINAR_20260907.md)
进一步区分描述性线索对齐与原生因果使用；不能用拟合出来的 G 单独证明模型使用。
新增反例说明残差相关可在去掉重要共同线索后上升，应看固定分母的贡献而非 ΔC。

后续 [证据状态与权重可识别性](GRAIN_EVIDENCE_STATE_AND_WEIGHT_IDENTIFIABILITY_V1.md)
将该候选接到加权矩更新与来源状态：只保留当前分数不能保证下一步响应相同，
但共同正向权重缩放也不可能独自提高 Pearson/AUC。新增协方差分解区分内容、
覆盖和交互；这是条件性代数与有限反例，不是实际作者的已识别贡献。
原“纯内容平均与正确处理重复的条件 LR 都可不变”反例保留；加入非证据槽后是另一
嵌套模型，可产生证据/非证据权重竞争。作者侧响应函数与模型侧预测状态继续分开。
研讨给出有限响应闭合与绑定优先的小实验提案，没有改原长跑或宣布新的心理效度。
