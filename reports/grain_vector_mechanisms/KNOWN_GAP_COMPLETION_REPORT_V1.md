# GRAIN 已知步骤补全报告 V1

2026-09-13。对应用户授权：先补可执行的先行方法步骤，Claude只读寻找未定位原资源。
这是先行方法接入的后续交付，不是新的大规模训练或完整论文复现声明。

## 1. 本轮补到了哪里

| 工作 | 已执行结果 | 对下一步的意义 |
|---|---|---|
| 原作者MEM计算 | 504人建参照、120人留出；相同二值词矩阵下，留出坐标匹配绝对r中位数0.99987，Procrustes相对误差1.81e-14 | 本地PCA/旋转核心与原函数接近一致；剩余差异应查词汇加工，不必重写分解公式 |
| 原生R `text` | 隔离安装text 1.9；作者示例及真实作者单fold均跑通 | 已从Python同类适配推进到原包实际执行 |
| 原生text真实单fold | 120名留出作者mean r=0.30420；同批作者旧Python适配=0.30541 | 两种实现结果接近；不是新预测提升或完整5-fold结果 |
| 自然语言锚点 | 新增60条本地锚点，覆盖10 facets；匹配428→455，覆盖作者217→226/504 | 原始量表条目之外的表达可增加检索覆盖，但需判定归属/方向/范围 |
| 新增匹配筛查 | 固定本地NLI对27条新增匹配输出：支持7、中性14、矛盾6 | 扩大词库不能自动解决语义利用；这是模型判定分布，不是准确率 |
| 原始资源检索 | Claude两轮只读检索；定位MEHv2源码，导出text附带的61条非空L-BAM模型记录 | 已找到后续可直接复用的源代码/模型入口；未下载全部权重或运行它们 |

**结论：有些缺口是尚未执行已公开实现，本轮已补；有些是原材料、专家判断或观察设计缺失，不能靠安装包替代。**

## 2. 原生方法与本地适配的差异

### MEM

执行[作者PCA模板](https://www.ryanboyd.io/software/meh/Supplemental/PCA-Template-for-MEM.R)
中的四个函数定义，保留下载字节及SHA256，不执行模板内硬编码的Windows文件操作。
使用本地训练侧固定的300词二值矩阵、12维；原函数体未修改。

| 数据 | 匹配绝对r最小值 | 匹配绝对r中位数 | Procrustes相对误差 |
|---|---:|---:|---:|
| 训练504人 | 0.99763 | 0.99986 | 1.03e-14 |
| 留出120人 | 0.99791 | 0.99987 | 1.81e-14 |

旋转坐标略有差异，整体子空间数值一致。此处没有使用人格标签。
新找到的[MEHv2](https://github.com/ryanboyd/MEHv2)源代码确认词法路径还有
大小写/转换表、分词、可选词形还原、分段、停用词和文档频率选择。
固定revision `3b306a74490b5f99c171ac683e6cb8fe8f3ea3dc`，assembly版本2.3.02.0。
英语可选词形还原由LemmaSharp/LemmaGen路径执行，不是必然开启的默认步骤。
本地sklearn停用词和95%出现率上限不能冒充这套完整词法管线。
Windows C#程序本轮未运行；仓库MIT不自动覆盖所有附带语言资源的许可。

### text

[原生textTrain](https://r-text.org/reference/textTrain.html)以训练侧标准化、PCA和
回归选择构成评分器。作者附带40人、10维嵌入示例，使用本地限定3×2 CV、PCA5、
两档penalty后，r=0.29888；包内单侧p=0.03050，另算双侧p=0.06101，
双侧95%CI=[-0.01392,0.55840]。这是软件示例，不是PANDORA或论文原成绩复现。

真实作者检查复用既有624人/36评论的E5均值材料。官方fold0的120人只作本轮最终评分；
504人内3×2嵌套CV、各trait按RMSE选择penalty，先标准化再PCA64。
penalty预设为0.001/0.01/0.1/1，最终五维均选1；保留边界警告，未看留出结果后扩网格。

| Trait | 原生text留出r | 旧Python同批留出r |
|---|---:|---:|
| Extraversion | 0.60302 | 0.61364 |
| Neuroticism | 0.26949 | 0.28175 |
| Agreeableness | 0.24348 | 0.22702 |
| Conscientiousness | -0.00192 | 0.00109 |
| Openness | 0.40695 | 0.40357 |
| Mean | 0.30420 | 0.30541 |

旧Python的PCA/标准化次序、alpha尺度与选择准则不同，故这是匹配输入/作者的原包
能力参照，而非逐参数算法等价证明。作者ID、顺序、fold和标签均与旧OOF核验一致。
这批材料有历史开发曝光，不是新lockbox；不能把本表0.304与旧624人pooled 0.196直接比较。

另核验textSimilarity在`center=FALSE, scale=FALSE`下与NumPy原始cosine最大误差
4.44e-16。其默认中心化是向量内部坐标中心化，不能直接当作锚点检索的原始cosine。

## 3. 自然锚点与反馈

`configs/grain_natural_anchors_v1.json`固定60条Codex撰写的公开句子：
40条惯常表达、20条具体事件，五个Big5域各两个facet，均有正反向。
它们不是原SIMPA专家eTRS，也不覆盖IPIP的全部30 facets。
先固定文字，再在504名来源作者的23,560条候选句上运行；未读取人格效标。

| 检索库 | 锚点数 | 匹配句数 | 有匹配作者/504 |
|---|---:|---:|---:|
| IPIP300 | 300 | 428 | 217 |
| 本地惯常表达 | 40 | 41 | 32 |
| 本地事件表达 | 20 | 5 | 5 |
| IPIP+惯常表达 | 340 | 450 | 225 |
| IPIP+全部本地表达 | 360 | 455 | 226 |

新增27条，旧428条均仍被检出；其中7条旧匹配改了最近锚点，3条改变域或正反键。
更大的候选库增加最大相似度过线机会，覆盖增加不等于精度提升。
`anchor_scope`只描述锚点的概括程度，不把源句自动标为trait或state证据。

27条新增匹配全部进入隐藏相似度/人格分数的私有审查包；本地固定
`cross-encoder/nli-deberta-v3-base`输出支持7、中性14、矛盾6。
没有人工金标准，不报告precision；不把矛盾自动改为反向人格，也未推广任何锚点。
两次锚点输出的审查包SHA相同，NLI结果复用而不重复计算。

已实现反馈接口：记录候选/词库hash、来源分区、评定者类型、正反键和范围；
至少两位评定者，反对票阻止推广；重复、留出来源、效标选优均拒绝。
默认要求人工评定，显式允许的模型模式只产出provisional；即使人工通过，也不自动
声称专家资格或量表效度。真正独立复核仍是下一步，而非伪造已完成反馈闭环。

## 4. Claude检索交付与资源边界

两轮使用现有只读Claude会话，公开网页/仓库检索；未给它私人文本、量表分数或密钥。
原始回复保留在`claude_retrieval/response.json`、`followup.json`。
首轮将“未找到”概括成“不公开”的说法已在第二轮纠正，本报告不沿用。

| 对象 | 本轮结论 | 下一步实际缺口 |
|---|---|---|
| SIMPA | [作者页面](https://takelab.fer.hr/publications/simpa/)及[博士论文第9章](https://arxiv.org/html/2510.02811v1)可读；精确checkpoint、453条专家eTRS及反馈标注在有界检索中未定位 | 原专家材料/权重来源；如继续索取需另行联系作者，尚未发送邮件 |
| BLT | [OSF](https://osf.io/bm9gn/) README已说明混淆分数、模型和验证notebook | 原始词矩阵及完整发现规格仍未定位；未下载用户级档案 |
| MEM | 原作者R核心已跑，MEHv2源码和关键资源已存档 | 原完整词法路径对照与资源许可，不是PCA公式缺失 |
| L-BAM | [text源库](https://github.com/OscarKjell/text)安装包自带目录，成功解析61个非空模型记录 | 逐模型核encoder/layer/pooling、许可和输入契约后才运行 |

L-BAM目录有valence、power/achievement/affiliation等相关入口，如
[Implicit-motive-models](https://github.com/AugustNilsson/Implicit-motive-models)和
[open_models](https://github.com/theharmonylab/open_models)。它们是可复用线索，
不等于当前已有Big5/MBTI计分器。E5向量不能直接喂给要求RoBERTa第23层的回归权重。
目录来自text 1.9随包快照，不冒充实时完整目录；14个选定源码/元数据文件已存档并hash。

同作者博士论文包含后续LLM扩展，不能全部回填为2022期刊版的方法。
ctsem测量层需要定义观察误差与时间对象；选择IRT需要机会集合。
这些接口不会因找到GitHub就自动变成当前数据已满足的实验。

## 5. 代码、产物与复跑

本轮新增脚本：`run_grain_known_native.py/.R`、`run_grain_text_native.py`、
`fit_grain_text_native.R`、`run_grain_natural_anchor_reference.py`、
`audit_grain_natural_anchor_matches.py`、`fetch_grain_known_gap_resources.py`。
可复用逻辑在`grain_research/natural_anchors.py`；测试在`tests/test_grain_known_gaps.py`。

所有生成物位于gitignored的`results/grain_vector_mechanisms/known_gap_completion_v1/`：

| 目录 | 权威文件 |
|---|---|
| `native_r4/` | summary.json、contract.json、sessionInfo.txt、原函数和RDS |
| `text_pandora_r1/` | comparison.csv、summary.json、alignment_audit.json、五维RDS及私有预测 |
| `anchors_r2/` | coverage.csv、summary.json、private_matches.csv、private_blind_review_packet.json |
| `anchor_nli_r1/` | summary.json、私有逐对NLI输出 |
| `upstream_r2/` | manifest.json、lbam_nonempty_registry.csv、源文件及许可 |
| `claude_retrieval/` | 两轮原始检索结果 |
| `Rlib/` | 隔离安装的text 1.9及依赖；不写系统R库 |

在独立GRAIN repo根目录复跑，必须使用新输出名。依赖既有、受许可的本地材料/缓存，
这些命令不是无数据的从零下载流程。Rscript当前按本机`/usr/local/bin/Rscript`调用。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
BASE='results/grain_vector_mechanisms/known_gap_completion_v1'
"$PY" scripts/run_grain_known_native.py --rlib "$BASE/Rlib" --output "$BASE/native_repeat"
"$PY" scripts/run_grain_text_native.py --rlib "$BASE/Rlib" --output "$BASE/text_repeat"
"$PY" scripts/run_grain_natural_anchor_reference.py --output "$BASE/anchors_repeat"
"$PY" scripts/audit_grain_natural_anchor_matches.py --packet "$BASE/anchors_repeat/private_blind_review_packet.json" --output "$BASE/nli_repeat"
"$PY" scripts/fetch_grain_known_gap_resources.py --rlib "$BASE/Rlib" --output "$BASE/upstream_repeat"
```

原生接口调通时修复了data.frame/matrix约定、htest结果序列化和Dim列名；
资源目录解析修复了CSV多行引号前言。失败尝试保留在native_r1-r3、upstream_r1，
不把这些接口错误当作论文方法失败。新增代码ruff通过；七个相关测试模块38 tests通过。
没有启动新神经训练、MEPS/bridge训练或旧队列；没有开锁箱、commit或push。

## 6. 下一个最小决策

先复核27条新增证据及3条改键旧证据的归属/方向/范围，再决定哪些自然锚点值得保留；
模型辅助可用于开发，不替代专家依赖。原包数值对照既已接近，不继续在该处追逐小数点。
原SIMPA资料继续作为明确资源缺口；可先用公开小型语义对照和有记录的本地反馈开发。
若选择接入已训练L-BAM，先选一个与现有测量对象匹配的目标并冻结编码器契约，
不为了填齐61行启动权重下载与全量评价。
