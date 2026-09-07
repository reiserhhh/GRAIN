# GRAIN S0–S4：功能路径研究闭环报告

2026-09-08。状态：本轮全部计算完成，没有遗留后台任务。
理论修订见 [GRAIN_FUNCTIONAL_MECHANISM_THEORY_V2](../../docs/GRAIN_FUNCTIONAL_MECHANISM_THEORY_V2.md)。

## 结论先行

**找到两个可用机制，并定位一个明确的跨特质差异。**

1. 功能敏感方向比位移大方差方向更适合描述 LLM 的特质判断变化。旧作者上
   学到的算子在新作者上仍有效；真实梯度说明这种局部作用高度可压缩。
2. 只用 LLM 评分监督、从未用问卷标签训练的轻量 T/F 学生，作者外 AUC=.7173，
   高于同文本教师 .6591；配对差值 95% CI=[.0398,.0774]。
3. 神经质没有获得相同收益：其标签协方差的大部分仍留在教师减学生的残差中。
   因此下一步应定位这部分剩余作用，而非再给所有特质共用更多聚合参数。

这里的高内部响应 r、学生复现教师的 r、与心理指标的 r/AUC 是三种不同指标。
以下明确区分，不把 .99 内部响应精度宣传为人格预测准确率。

## 1. 执行范围与成本

| 阶段 | 数据与计算 | 完成状态 |
|---|---|---|
| S0 功能坐标 | 96 既有开发作者；TF60/N36；嵌套作者 CV；零神经调用 | 完成 |
| S1 梯度与公共编辑 | 24 新 source-only 作者，2视图×3查询，12评论/视图 | 750F+144B，1046.9秒 |
| S1B 查询内容辨别 | 前8作者视图0；相同0–9格式，不同查询内容 | 42F+8B，58.4秒 |
| S2 分数/响应学生 | 24作者；4特征家族×值/响应损失；内层只用E1 | 5.5秒，零新神经调用 |
| S2 冻结学生源连接 | 2328 MBTI-only +213 Big5-only | 5.5秒，零新拟合/神经调用 |
| S3 评论敏感度 | 24作者×2视图×12评论×3查询；3特征家族 | 1.2秒，零神经调用 |
| S4 教师样本量阶梯 | TF2328/N213；24/96/384/全部训练作者 | 6.6秒，零神经调用 |

F/B 指固定 Qwen2.5-1.5B observer 的前向/反向。合计792F+152B，采集约18.4分钟，
不含模型装载与准备时的少量 E5 查询锚点编码。模型 FP32/MPS；block16 后缀图
适配24GB Mac，没有全网络参数梯度、API费用或新 LoRA 训练。

旧长队列未恢复；bridge、MEPS心理量表、历史lockbox未打开。原文、作者ID、逐例
评分和模型数组保留在 gitignored results。Git只收代码、测试和聚合解释。

## 2. S0：功能监督确实找到了方差排序漏掉的信息

同一位移、同一响应，5外层/3内层作者CV，rank2/4/8及alpha1/10/100。
数据变换在训练折拟合。比较的是完整估计器，非纯坐标变换因果实验。

| 查询 | SVD r | full Ridge r | PLS r | PLS对SVD的MSE差及95%CI |
|---|---:|---:|---:|---|
| TF（60作者） | .4586 | .8029 | .8021 | -.000241 [-.000312,-.000172] |
| N（36作者） | .3883 | .7663 | .7882 | -.000045 [-.000066,-.000026] |

S0模型在S1评分前冻结。24新作者、公共E1/E2编辑上的迁移：

| 查询 | SVD r | full Ridge r | PLS r | PLS MSE/零响应MSE |
|---|---:|---:|---:|---:|
| TF | .5039 | .8589 | .8420 | .3015 |
| N | .4753 | .8561 | .8715 | .2150 |

这是**新作者上的向量干预响应预测**。输入仍来自观察器内部状态，不是无LLM推理。

## 3. S1：局部线性成立，完整证据作用并不局限于该测点

| 查询 | 实际patch对gᵀd的r | MSE/零响应MSE | 自然编辑对单位置patch的r |
|---|---:|---:|---:|
| TF | .9936 | .0185 | .2360 |
| N | .9956 | .0126 | .3363 |
| 寻求确认 | .9942 | .0108 | .3934 |

梯度/状态配对的功能投影同样有效：rank8 CoBRAS候选r=.9817/.9846/.9824，
位移方差投影仅.6102/.6326/.5881。该分析使用测试案例的实测梯度，单独报告。

N 的被删评论 block16 梯度×状态之和，与真实删文响应r=.6683；加上查询位置
一阶项，MSE/零响应由.9312降至.7015。它是零状态参照的局部代理，不是精确删文
导数，却明确指出：正文中的分布式作用值得继续追踪。

### 共享与专属

N/确认的同评论每token敏感度剖面平均r=.6090；去除编辑/视图共同均值后，
内部响应r=.7033 [作者bootstrap .5962,.7975]，自然响应r=.2567 [.0704,.4493]。
T/F与这两者的自然响应接近零相关。低秩共同作者空间没有统一的强结论，不能
把三个查询强压成同一个潜因子。

S1B在同8作者/视图/编辑上比较：N/确认patch r=.6303，N/数量内容r=-.0280，
确认/数量内容r=-.3231；三者都使用0–9候选格式。结果支持进一步研究查询内容
选择的路径，而非将共享全部归于答案数字。小面板结果不要求无关内容严格零相关。

### 组合

8作者的联合删文交互RMS/联合响应RMS：TF=.884，N=.560，确认=.832。
梯度投影的状态混合差分与分数交互r=.840/.817/.420。非加性已存在于状态组织，
但比例不是方差解释率，机制可能涉及组合、位置、长度、归一化和读出。

## 4. S2/S3：哪些简化不足，为什么继续S4

S2用原分数和E1差值训练；E2和联合编辑既不拟合也不选参。四家族保留前一层
输入：mean → query pool → std/landmark → rank4二次项。5外层/3内层作者CV。

- T/F分数复现最好观察到r=.6205（query/value-only）；response-match的E2差值
  r约.24–.36，有部分复现，但E1在新作者上仍负相关，没有完整复制作用规则。
- N的分数复现弱；E2无一致增益。联合编辑小面板可以出现高相关，但不能掩盖
 其它条件的弱表现。完整96行指标保存在本地metrics.csv。
- 结构全部冻结、仅用内层教师损失选出的单一学生为quadratic/value-only。
  在源连接中，TF AUC=.6286；N r=.0362。不能事后换成外部标签成绩更好的变体。
- S3内容/上下文对每评论敏感度的解码，没有稳定超过长度/位置对照。
  N对照r=.320，内容r=.288、上下文r=.239；TF相应.166/.138/.145。
  该简单归因蒸馏不作为完整编辑效应模型继续扩张。

因此提出并执行S4：不是无限加功能，而是利用已有教师缓存区分样本量限制。

## 5. S4：教师评分可被轻量系统吸收，且特质间有明显差异

只以native cue-mask分数为目标。每个作者三视图等权汇总，mean/query两类特征；
训练内标准化/PCA最多64维/Ridge，内层教师MSE选alpha。相同5外层作者折比较
各训练样本量，所有OOF分数落盘后才在此脚本中接外部标签。
这些D1心理标签在既往研究已被看过，故仍是开发性外部连接，不冒称新锁箱。

### T/F 均值学生学习曲线

| 每外折训练作者数 | 教师复现r | 与MBTI标签的r | AUC |
|---|---:|---:|---:|
| 24 | .4796 | .2617 | .6564 |
| 96 | .5837 | .2998 | .6827 |
| 384 | .6418 | .3366 | .7033 |
| 全部（1862–1863） | .6668 | .3632 | .7173 |

query pool全部样本为.6692/.3639/.7173，几乎相同；优先保留更简单mean。

### 最终配对比较（同一cue-mask文本）

| 方法 | 拟合监督 | TF AUC（2328作者） | N Pearson r（213作者） |
|---|---|---:|---:|
| Qwen直接查询 | 无本地训练 | .6591 | .1765 |
| E5统计对照 | 实际source量表标签、既有OOF协议 | .7537 | .0404 |
| S4均值学生 | 仅LLM已有评分、嵌套作者CV | .7173 | .0673 |

学生TF AUC 95%CI=[.6956,.7389]；比教师+.0582，配对CI=[.0398,.0774]；
比标签训练E5低.0364，CI=[-.0526,-.0194]。它不是最强标签监督预测器，而是
**不使用人格标签拟合、也不需要LLM生成推理的可行实例**。既有编码器仍是神经模型。

N均值学生从24训练作者r=-.070，96人r=.013，到全部170–171人r=.067；教师
复现从.329升到.461，但没有保留同等心理关联。学生N的相关CI约[-.074,.192]。
所有区间按作者配对重抽样、条件于冻结OOF预测，不包含重复训练的不确定性。

### 为什么学生可以更好，也可以失败

精确分解Cov(Y,S)=Cov(Y,f)+Cov(Y,S-f)：TF学生保留94.1%的标签协方差、
评分标准差由.1187降为.0817；N只保留26.0%，残差与标签r=.143。
这为“哪些信息被压缩、哪些必须继续提取”提供了可计算对象。上述比例不是
心理方差解释率，也不是已唯一识别出的因果机制。

## 6. 代码、异常与复现

10个新研究入口：
`analyze_grain_functional_basis.py`、`freeze_grain_s0_operators.py`、
`run_grain_multiquery_sensitivity.py`、`analyze_grain_multiquery_sensitivity.py`、
`fit_grain_response_student.py`、`run_grain_numeric_query_probe.py`、
`connect_grain_response_student.py`、`fit_grain_evidence_sensitivity.py`、
`run_grain_student_learning_curve.py`、`analyze_grain_teacher_student_bridge.py`，均在scripts。

主要本地产物，前缀均为`results/grain_vector_mechanisms/`：

- `functional_basis_s0_r1/`及`frozen_operators/`：内外CV与冻结SVD/Ridge/PLS模型。
- `multiquery_sensitivity_s1_r1/`：材料、全部私有激活、梯度、manifest与成本账。
- `multiquery_sensitivity_s1_analysis_r1/`：作用、投影、共享空间、评论归因、联合项。
- `response_student_s2_r1/`：全部OOF分数、96指标、已冻结joblib学生。
- `numeric_query_probe_s1b_r1/`：同答案格式的新内容查询。
- `response_student_source_connection_r2/`：已冻结学生的源标签连接。
- `evidence_sensitivity_s3_r1/`：评论敏感度与删文代理。
- `student_learning_curve_s4_r1/`：样本曲线、选参、接标签前冻结的OOF分数。
- `student_bridge_analysis_s4_r1/`：配对CI、协方差分账。

异常记录：一次S1准备后的lint变更导致重新冻结材料，旧preflight-only目录保留、
没有旧神经采集；源连接r1因Neuroticism大小写差异在匹配处明确报错，修正并新增
测试后使用r2，未改标签、预测或成员。48视图中1条未达到20%删文长度匹配容差，
保留并标记。无NaN、OOM或自动缩小文本预算。

从已有缓存重播关键路径（需要本地研究venv与受限数据，输出须用新目录）：

```bash
python scripts/analyze_grain_functional_basis.py --output-dir results/grain_vector_mechanisms/s0_replay
python scripts/analyze_grain_multiquery_sensitivity.py --panel results/grain_vector_mechanisms/multiquery_sensitivity_s1_r1 --output-dir results/grain_vector_mechanisms/s1_replay
python scripts/fit_grain_response_student.py --panel results/grain_vector_mechanisms/multiquery_sensitivity_s1_r1 --output-dir results/grain_vector_mechanisms/s2_replay --include-quadratic
python scripts/run_grain_student_learning_curve.py --connection results/grain_vector_mechanisms/response_student_source_connection_r2 --panel results/grain_vector_mechanisms/multiquery_sensitivity_s1_r1 --output-dir results/grain_vector_mechanisms/s4_replay
```

新神经采集仅在新目录先`run_grain_multiquery_sensitivity.py --phase prepare`冻结，
再`--phase run --max-authors 8`试跑、同目录`--max-authors 24`续跑。不要覆盖旧结果。
`freeze_grain_s0_operators.py`直接运行需设置`PYTHONPATH=.`。
本轮关键测试与历史相关回归共169项通过；Ruff检查/格式化通过。未重跑全仓数千项
测试与全部历史实验。仅有已知第三方Swig弃用警告。

## 7. 下一步主线决策

本轮关闭S0–S4，不继续当前参数网格。保留成功的T/F轻量通路，并将N的教师剩余
响应作为下一机制对象：多位置状态、查询条件化证据关系与必要交互。H5读写对偶
仍是独立未执行问题，不混入本轮闭环。下一包先给明确的剩余作用目标与有限采样
设计，而不是重启长跑或要求用户提供新的宏大假说。
