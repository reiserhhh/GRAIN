# 先行研究统合：第一批执行报告

后续更新见[第二批报告V2](PREDECESSOR_INTEGRATION_REPORT_V2.md)。本文件保留第一批历史；
GRAIN48维的准确身份是TF-IDF→SVD24坐标的mean/std，而非24个命名行为指标。

日期：2026-09-12。阶段：I0-I4执行中；第一批有限参照已经运行，不是全体先行能力追平完成。
目标：先得到可以实际使用、明确来源和适配差异的参照，再决定GRAIN需要新增什么。

## 1. 本轮获得了什么

1. 三条可运行参照：无标签的MEM结构发现、语义表示回归、句子到量表条目的证据匹配。
2. 原作者sJIVE代码、标准随机效应模型和不等间隔OU模型的已知机制测试；
   sJIVE另完成真实来源作者的共同协议比较。
3. 一个具体适配问题：E5的cosine尺度使0.6阈值完全失去筛选作用。先修检测契约，
   不能把此适配器的弱结果解释成“前人的SIMPA无效”。
4. 结构稳定、心理关联和语句证据正确性已变成三个分别记录的输出，避免只比较一个r。

完整来源、公式及27机制路由见[方法卡](../../docs/GRAIN_PREDECESSOR_METHOD_CARDS_V1.md)。
本轮没有修改发布版计分器、没有新神经网络训练、没有使用MEPS或bridge训练、没有开锁箱。
旧NEXT91/P1长跑没有恢复，也没有提交或推送。

## 2. 真实材料与比较协议

使用现有VM05开发来源缓存中共有完整表示的624位Big5-only作者，每人36条评论：
22,464条评论、29,075个包含第一人称I的候选句；另有公开IPIP NEO300条目。
这是受既有缓存覆盖和文本预算限制的子集，不是完整PANDORA或新招募测试群体。
旧资料已被研究过，本轮结果为回顾性/探索性比较，不冒充新的前瞻验证。

- 外层使用这些作者的官方repeat=0五折：120/126/130/119/129人；一人只在一个fold。
- 内层3折选Ridge α∈{1,10,100,1000}；词表、PCA、scaler均在相应训练分区fit。
- 每个模型有config/hash、逐折模型、OOF预测、逐trait指标、来源地址及异常记录。
- 成对CI为1000次作者bootstrap，条件于OOF拟合结果；不是整个训练过程的重复采样CI。
- E5评论向量复用既有cache；新增句子/条目用同revision本地MPS编码，用时约11分钟。
  batch/device与旧cache不同，均有独立契约。没有下载新模型或调用付费LLM。
- 原始材料仍有既有扫描标记：MBTI类型词74条、Big5词134条、report clue32条；类别可能重叠。
  本轮没有做新clean/no-report ablation，因此不拿这些结果宣称无泄漏人格效度或降噪成功。

## 3. 共同作者上的外部关联

以下为pooled OOF五维Pearson的算术平均，不是平均折分，也不是原论文重现分数。

| 本地方法 | mean Pearson r |
|---|---:|
| Word TF-IDF + Ridge | 0.1450 |
| MEM适配器 + Ridge | 0.0525 |
| E5评论均值 + PCA64 + Ridge | 0.1957 |
| GRAIN冻结48维词汇行为投影 + Ridge | 0.0864 |
| IPIP匹配计数适配器 + Ridge | 0.0632 |
| 语义表示 + 条目计数 | 0.1836 |
| GRAIN词汇投影 + 条目计数 | 0.0983 |
| GRAIN词汇投影 + 语义表示 + 条目计数 | 0.1662 |
| 语义分支：单次标签打乱sanity check | 0.000002 |

GRAIN这一行是VM05固定词汇SVD24坐标的均值/标准差，不是最新GRAIN全部功能；
不能据此宣布“GRAIN整体输给E5”。MEM也不是为了最大化Big5预测而发现的因子。

E5的逐维相关：E=0.3370，N=0.2309，A=0.1821，C=-0.0256，O=0.2540。
均值隐藏了明显的维度差异，不应称五维同样有效。本轮0.1957与以前约0.196接近，
但作者、文本预算、管线及复用历史不同，不能据此说已触及人格预测上限。

主要配对差值：

| 比较 | Δ mean r | 条件OOF bootstrap 95% CI |
|---|---:|---|
| 语义相对GRAIN词汇投影 | +0.1093 | [0.0675, 0.1509] |
| 语义+条目相对语义 | -0.0121 | [-0.0238, -0.00006] |
| GRAIN词汇+条目相对GRAIN词汇 | +0.0119 | [-0.0055, 0.0295] |
| 三视图拼接相对语义 | -0.0295 | [-0.0515, -0.0099] |

接近零的CI边界不宜作为强结论；本轮多项探索比较未作整体多重检验校正。
单次shuffle接近零是一个实现检查，不是正式置换检验p值，也不排除原文标签泄漏。

## 4. 从结构到证据：发现的差距

**无监督结构确实可以成为可评分对象。** MEM在504位作者上发现12个组件，
在另外120位作者上计分。8次移去20%训练作者后，训练载荷对齐的一致系数中位数
为0.7129，held-out坐标相关中位数0.7364；同人交替评论两半的组件相关中位数0.3954。
对齐只看训练词载荷，不用held-out人格标签或选择最相关的心理轴。
12是有限参照的工程上限，不是发现了12种人格，PCA组件也不冒称共同因子。

**条目检索可以追溯，但当前检测门槛无效。** 29,075个候选句的top-1匹配全部≥0.6，
每句随机选择一个IPIP条目也全部≥0.6。top-1 cosine中位数0.7992，随机配对0.7383，
top-1与top-2差值中位数仅0.0051。随机条目不是真正标注的负例，因此这里只证明阈值
没有拒绝作用，不证明所有最高匹配都错。该现象与[E5官方FAQ](https://huggingface.co/intfloat/e5-large-v2)
描述的高绝对cosine尺度相符；不能从别的编码器搬一个阈值就认为已继承前人的检测能力。

计数回归的预测两半相关仅0.029-0.168，语义回归为0.621-0.707。
高覆盖并不保证高质量；未监督条目百分位与外部量表相关为0.027-0.136，
计分覆盖97.6%-99.5%。这些是待校准匹配器的描述，不是可部署量表。

已产出60条不含人格标签的本地盲评队列。需要分别判定“作者本人”“正反向”“特质相关”，
而不是让同一个相似度同时承担所有含义；本轮没有把模型判断冒称专家标注。

## 5. 条件、动态与共享结构

| 已知机制例 | 结果 | 说明 |
|---|---|---|
| lme4随机截距/斜率 | 恢复相关0.9951 / 0.9879 | 100人×8条件×4场合；生成机制已知 |
| 观测机会量的Poisson例 | raw count相关0.4364，rate相关0.8838 | 证明接口能利用已知暴露量，不是新IRT理论 |
| 不等间隔OU | 衰减率0.6292，真值0.7；扩散0.4811，真值0.5 | 简单可观测状态特例，不是完整ctsem |
| 原作者sJIVE：已知两视图 | held-out目标r=0.9875 | 共享/特有成分均能恢复；不属于真人效度 |

对真实624位作者再用同一组训练侧PCA16×两视图比较：简单拼接Ridge为0.1671，
固定rankJ=2、rankA=(2,2)、η=.5的sJIVE为0.0036。首轮250迭代全部触顶，
复查使用原默认.001收敛阈值及1000迭代上限后，25/25个拟合均收敛；仍无预测增益。
不再为了追分继续rank/η搜索。结果只说明该有限配置不宜替换当前读出，
不说明sJIVE整体无效，也不说明共享结构不存在。

条件资料核对：评论都有已选subreddit、thread和时间戳，但没有可选机会集合。
不能因已实现随机效应模型，就声称真实语料已分离机会与偏好；也没有把随机评论视图
直接送进OU冒充真实时间序列。需要的观测、估计对象和下一步触发器已写入方法卡。

## 6. 当前决议与下一步

**保留并采用**：作者级共同协议、固定参照、标准MEM结构参照、语义读出、条目来源接口、
随机效应/OU/sJIVE的标准实现参照。它们减少我们重复发明一般数学的必要。

**修正而不推广**：E5绝对阈值式心理证据判定；未经归属与极性判断的语义相似；
“加一个共享模型或拼更多特征就一定有益”的假设。旧GRAIN有效结果仍保留。

**本阶段仍开放**：SIMPA原生编码/专家扩展的可比实现；BLT的FA/斜交估计器；
相似度到可归属心理证据的校准；真实条件层及评分误差传播。当前只有一个GRAIN
词汇投影参与共同协议，没有把全体系的预测能力比较完。

下一步仍属于先行能力追平，不跳到新理论世界：

1. 以60条标签盲队列确定证据判断规范，核实原生paraphrase模型/可复用TRS；
   用独立的句对正确性开发集校准，不用外层Big5相关选择阈值。
2. 补真正的FA/特有方差及斜交参照，与MEM比较结构、计分稳定性和解释，而非只比Big5 r。
3. 只在检测能力建立后，比较计数、校准证据、条件响应及GRAIN视图是否提供互补信息。
4. 真实条件/时间模块先明确机会与观察契约；缺失的反事实机会保留未知，不以模拟通过结案。
5. 用上述最强适用参照重定RQ：开放表达中“可观测机会→有效证据→人的属性”的损失
   怎样估计与改进，哪些构造规则能预测在新情境中的测量收益。

I0/I1/I2已有实作交付，I3完成有限共同协议，I4得到更窄的下一步起点；
**全部I0-I4尚未宣称最终完成，当前goal保持active。**

## 7. 文件与复现

代码：`grain_research/predecessors.py`及下列脚本。Python复用父项目`.venv`；
R用本机Rscript和已安装lme4，没有新装包。上游sJIVE保持GPL-3来源记录，未复制进公共源码。

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
"$PY" scripts/prepare_grain_predecessor_reference.py --encode --max-seconds 1800
"$PY" scripts/fetch_grain_predecessor_sources.py
"$PY" scripts/run_grain_predecessor_reference.py --output results/grain_vector_mechanisms/predecessor_integration_v1/comparison_r2
# 以下不覆盖旧输出；重新运行时将r2改成新的目录名。
Rscript scripts/run_grain_predecessor_process_reference.R results/grain_vector_mechanisms/predecessor_integration_v1/process_r2 results/grain_vector_mechanisms/predecessor_integration_v1/upstream
"$PY" scripts/run_grain_predecessor_integration.py --output results/grain_vector_mechanisms/predecessor_integration_v1/integration_r2
"$PY" scripts/audit_grain_predecessor_structure.py --output results/grain_vector_mechanisms/predecessor_integration_v1/structure_r2
"$PY" -m pytest tests/test_grain_predecessors.py tests/test_grain_reference_integration.py tests/test_grain_vector_mechanisms.py -q
```

输出根：`results/grain_vector_mechanisms/predecessor_integration_v1/`。

| 子目录 | 内容 |
|---|---|
| material | 原始材料契约、IPIP来源、句子编码逐batch缓存 |
| comparison_r2 | 8方法OOF、逐维指标/CI、配对比较、两半相关、条目来源及覆盖 |
| integration_r2 | 收敛后的sJIVE与matched concat、分量贡献、来源及60条盲评队列 |
| process_r2 | 标准已知模型结果、R环境、模型文件 |
| structure_r2 | 标签盲结构复采样、原始条件/泄漏标记审查 |
| upstream | 固定commit的原作者代码、DESCRIPTION/许可与checksum |

`comparison_r1`数值与r2一致，但r2补了严格缺失汇总和bootstrap配置。
`integration_r1`是迭代不足的诊断记录，不作正式参照；保留当时执行源码快照。
`process_r1`有一条代数说明混在数字表，已从r2数字表移至文字说明，避免冒充实验。
所有含作者/原文/个体分数的文件留在被gitignore的results内，不发布私人盲评队列。

最终检查：上述三个测试文件合计127 passed（29.12秒），另有既有SWIG依赖的
deprecation warnings；新Python文件ruff通过，`git diff --check`通过。
独立从OOF重算8方法均值与结果表一致；材料、OOF、sJIVE三套作者顺序和Y完全相同，
sJIVE预测有限且25个fit未触迭代上限。私人材料及盲评队列的gitignore已确认。
