# GRAIN 先行研究统合：第三批接口与阶段交付

2026-09-12。继[V1](PREDECESSOR_INTEGRATION_REPORT_V1.md)、
[V2](PREDECESSOR_INTEGRATION_REPORT_V2.md)后，补做自然证据盲评、冻结作者计分、
检测误差传播和真实条件模型。以下是本地适配结果，不是原论文精确复现。
本阶段交付范围与下一起点见[统合决议](../../docs/GRAIN_PREDECESSOR_STAGE_DELIVERY_V1.md)。

## 1. 这一批真正改变了什么

**GRAIN的下一步不应是简单增加一个人格回归头，而应把“证据说了什么、何时能观察到、
在哪个参照下可比较”变成可共同估计的接口。** 三条具体结果支持这个起点：

- 独立模型盲评的80对真实语句中，只有2对是同概括程度、同方向的自我证据；
  纳入具体事件后有18对。检索到具体行为和检索到惯常倾向是不同任务。
- 实际冻结GRAIN几何评分器对624名新于旧参照的作者只允许124人计分。
  按原观察算子重放没有消除这一范围差异；不能把旧参照直接当成新材料的通用规范。
- 标准交叉随机效应模型在18,144条真实评论上正常收敛，检测概率同时有作者和所选
  subreddit差异。证据可用性不只是总文量，也不能直接等同于人格强度。

全部只用Big5-only开发材料。没有MEPS/bridge训练、旧队列、新神经训练、开锁箱或提交推送。
唯一生成式辅助是独立Codex模型对80个盲化句对的语义标注；没有把它作为专家金标准。

## 2. 自然句对：继承SIMPA分类与通用NLI，不以模型验证自己

使用已有source-only分层包：cosine四档各20对，来自71名作者；外层fold0不进入标注包。
独立`gpt-5.6-sol / xhigh`评定者只读取候选句、条目及标注规则，没有查看相似度、
NLI输出、外部人格分数或本研究结果。80条结果的hash在运行前绑定，后续审计不修改它。
这是单个模型评定者，尚无人工或跨评定者一致性检验。

分类沿用[SIMPA七类判定，W07 §9.2](https://arxiv.org/html/2510.02811v1)，
另保留作者归属、事件/一般范围及字面蕴含。原研究有心理学专家及扩展条目；本地没有这些。

| 80对中的对象 | 同方向、自我归属的对数 |
|---|---:|
| 一般化证据，类别1 | 2 |
| 一般化或具体事件证据，类别1/3 | 18 |

在cosine>=.6的20对简单随机样本内，一般化同向1/20；纳入事件则10/20。
同向比例的二项近似95%区间分别为[.0013,.2487]、[.2720,.7280]，只针对这位模型的判定。
并非“自然文本只有5%能测人格”，也不是全人群检测precision：抽样框只包含带I的候选句
与其top-1条目，条目库覆盖之外的证据不在分母内。

通用[NLI工具W73](https://huggingface.co/cross-encoder/nli-deberta-v3-base)在54个
字面关系可判定的句对上与评定者一致38/54=.704；26个含糊对保留，不强制赋真值。
其中2个支持和2个反对均判对；50个中性对中34个判对、13个误判反对、3个误判支持。
全猜中性的表面准确率为50/54=.926，所以不能把70.4%作为工具有效性的主证明。
可用之处是方向区分；主要风险是把未知过度判为矛盾。

| 固定规则 | 样本内接受数 | 一般化同向数 | 包含事件的同向数 |
|---|---:|---:|---:|
| cosine>=.6 | 20 | 1 | 10 |
| NLI argmax支持 | 9 | 1 | 5 |
| 两者都通过 | 4 | 0 | 3 |

分层抽样的总体比例按`N_h/n_h`加权，保存在`weighted_rule_comparison.csv`。
例如NLI一般化加权precision=.433仅由1个样本中的正确支持贡献，不能据此宣称已优于
另一模型。联合规则3/4的事件同向率也不能变成稳定的75%精度结论。本批不调阈值。

## 3. 检测错误怎样进入分数，而不是直接删掉所有可疑语句

原计分是已检测正/反条目数之比。现在加入三个可能结果：同键、反键、无足够证据。
类别1/2用于一般化支路，类别1/3与2/4用于事件包含支路；不是自我归属的材料不计作者证据。

\[
q\mid n\sim\operatorname{Dirichlet}(1+n_{same},1+n_{reverse},1+n_{none}),
\quad Z_s\mid q\sim\operatorname{Categorical}(q).
\]

这是标准Dirichlet-多项式误差敏感性模型，不是GRAIN新定理。
先给同一条原句抽取一次方向状态，再对整条评论重采样；同一评论被抽到两次时不重新判方向。
每轮同一个q作用于所有作者，保留校准参数的不确定性。无证据的分母仍返回缺失。

接受层20条盲评中，`同/反/无`分别为一般化`1/2/17`、事件包含`10/5/5`。
对504名source作者的428个原始匹配，400次敏感性模拟得到：

| 支路 | 保留匹配数中位数 [2.5%,97.5%] | 有任何证据作者数中位数 [2.5%,97.5%] |
|---|---:|---:|
| 一般化 | 91 [27,177.1] | 51 [17.0,89] |
| 事件包含 | 317.5 [234.0,399] | 135 [109,157] |

这些是**模型标注假设下的敏感性分布，不是真实人格置信区间**。误差概率在作者/条目间
可交换、评论可重采样是工作假设；模型评定者自身的错误、真实时间依赖和漏检尚未估计。
不能把结果解释为应该清除其余文本。更具体的行为证据可以留在事件支路，再研究重复和情境。

## 4. 完整原生计分支路与48维输入不再混称

核对了实际接口：`RepresentationRuntime`提供词汇SVD单位表示；
`author_features_from_embeddings`生成均值/标准差48维；
`score_geometry_bundle`再应用冻结Mahalanobis度量、16个匿名地标及支持范围，输出
**排序的地标距离剖面**。这才是此次可复用的完整native几何计分路径。

原参照用141名发现作者，观察协议最多32评论、4096个项目token；当前624人均不在
旧240名发现/校准/确认作者中。它仍是已多次研究的开发资料，不称新锁箱。

| 重放 | 可计分 | 范围外 | 说明 |
|---|---:|---:|---|
| 旧VM输入：36条正文直接聚合 | 144 | 480 | 用于历史分支对照；非完整观察协议 |
| 按冻结native算子重放 | 124 | 500 | 每人32评论，平均2321个项目token；范围阈值未变 |

两者差异不是修一次文本格式就能消除。拒绝只说明落在旧径向范围外，并不证明那些作者
没有心理信号或来自不同文化；径向范围也不是完整密度/领域判别器。

在124个共同支持作者上重新编码完全相同的3968条native文本，E5无截断；三种读出
采用相同官方五折、内层三折及4个Ridge alpha。支持条件在读取本轮Y前确定。

| 同材料、同124作者对照 | mean Big5 r |
|---|---:|
| E5评论均值/PCA64/Ridge | .1123 |
| 原生GRAIN坐标48/Ridge | .0487 |
| 冻结GRAIN距离剖面16/Ridge | .0290 |

几何相对raw48的Δ=-.0197，配对OOF CI[-.1126,.0805]；相对E5的Δ=-.0833，
CI[-.1814,.0221]。没有此配置的增量证据；也不能据此否定整个GRAIN理论。
与624人E5=.1957不是同样本/训练文量比较。未在500个拒绝作者上强制报告测量分数。

**完整GRAIN不是一个当前可无条件调用的统一人格头。** 已核接口还包括V7多视图、
V8语义事件聚合、时间/选择/互动和向量路径分支。它们分别需要对应观察算子、冻结事件
读者、时序/机会结构或模型隐状态。本批没有这些共同缓存与匹配观测，因此列为接口能力，
不把native几何或48维输入代表整套系统，也不为填表恢复旧长跑。

## 5. 真实条件层：标准模型可以直接继承

在504名source作者、18,144条评论、2506个已选subreddit上拟合：

\[
\operatorname{logit}P(D_{uc}=1)
=\beta_0+\beta_L z(\log words_{uc})+a_u+b_c.
\]

D是该评论是否有至少一次释义检测，不是人格真值。用lme4二项GLMM、Laplace近似、
bobyqa，单一预设模型，不搜索。399条评论被检出，比例.02199。
`beta_L=.1233 (SE=.0532)`；作者随机截距方差.7904，subreddit方差1.1310。
优化码0、无收敛警告、非奇异解。这些是logit检测尺度的方差，不是“人格方差占比”。

它把“已经取到的文本中何处较容易出现候选证据”变成可计算对象。
当前只有实际评论和字数，未访问社区的可选机会集合仍为unknown；因此不估计未选择概率、
不把缺席当回避，也不作作者/社区的因果解释。AIC只作拟合描述，不作新作者预测成绩。

## 6. 产物、复现及开放项

私有产物根目录：`results/grain_vector_mechanisms/predecessor_integration_v1/`。
本批最终：`native_grain_r1`、`evidence_audit_r3`、`observed_conditions_r2`；
早期r1/r2保留。证据r2补中性基线和区间，r3补审查要求的冻结/方向检查，未重新挑规则。
原80条包及模型盲评在`evidence_sampling_r1`，原句、作者ID、模型及个体输出均gitignored。

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
# 从独立GRAIN仓库运行；不要覆盖已有目录。
"$PY" scripts/run_grain_native_reference.py --output results/grain_vector_mechanisms/predecessor_integration_v1/native_grain_r2
"$PY" scripts/run_grain_evidence_calibration.py --output results/grain_vector_mechanisms/predecessor_integration_v1/evidence_audit_r4
"$PY" scripts/run_grain_evidence_conditions.py --output results/grain_vector_mechanisms/predecessor_integration_v1/observed_conditions_r3
"$PY" -m pytest tests/test_grain_evidence_contracts.py tests/test_grain_native_reference.py tests/test_grain_evidence_calibration.py tests/test_grain_common_factors.py tests/test_grain_paraphrase_reference.py tests/test_grain_predecessors.py tests/test_grain_reference_integration.py tests/test_grain_vector_mechanisms.py -q
```

最终测试141 passed，19.01秒；含3个独立审查回归测试。
2条既有SWIG警告。新增Pythonruff通过；真实R拟合完成。
本地固定checkpoint，无新增依赖。盲评可重放已冻结标签，但重新让模型标注不保证逐字相同。

独立模型审查提出三项实现问题并已修复：缓存接受标记未严格核对；JSON混淆矩阵轴向
不明确；标注文件hash及枚举约束不足。`evidence_contracts.py`现在从原缓存向量重算
29,075个top-1匹配，最大cosine差8.33e-17，逐行检查出处、条目键及接受阈值。
标注hash、归属/范围/关系枚举在读取前固定；JSON显式区分expected行和predicted列。
修复后的NLI与GLMM小任务已重跑，数值不变。没有重跑编码器或旧训练。

外部依赖仍明确保留：专家标注/eTRS与SIMPA原checkpoint、BLT原始词矩阵、受控机会集合、
真实重复测量的目标定义及独立心理/行为效标。这些不会因本地适配完成而自动成立。
