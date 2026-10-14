# GRAIN 先行研究统合：第二批结果与研究起点

2026-09-12。I0-I4继续执行；本批补齐共同因子与释义模型的适配差异，
加入通用NLI参照和证据稀疏度/抽样误差接口。不是完整原论文复现或全体系验收。
前批历史结果见[V1](PREDECESSOR_INTEGRATION_REPORT_V1.md)，公式/来源见
[方法卡](../../docs/GRAIN_PREDECESSOR_METHOD_CARDS_V1.md)。

## 1. 最重要的推进

**现在可以分别运行“形成稳定坐标”“找到有关语句”“判断语句支持什么”三个环节。**
它们不是同一个任务，也不应再由一个cosine或Big5相关同时代表。

- 标准共同因子可以形成稳定的作者文本坐标：held-out坐标复采样相关中位数0.9138，
  分半0.6565；但5因子读出Big5的mean r为0.0296。说明共同结构形成与目标链接需要分别设计。
- 换成释义家族后，0.6阈值的接受率从100%变成1.857%；只剩540条匹配，
  277/624作者有任何匹配。36评论预算下的可用证据量成为显性变量。
- 不必自行发明极性推理工具。现成NLI在40个设计句对中31个符合预设三类标签；
  正向释义9/10判为支持，反向和描述他人各10个均未判为支持。单次事件仍有3/10被泛化。
  这是工具层语义能力，不是77.5%的人格预测准确率。

本批没有新神经网络训练。下载了两个有Apache-2.0许可的固定公开checkpoint，
全部文本编码/推断在本机完成；没有调用付费LLM、MEPS、bridge训练或锁箱。

## 2. 共同因子不再冒充PCA

真实材料仍为624名Big5-only开发作者、每人36评论，官方作者五折及内层三折。
两个新方法固定200词、5维，同样的词频输入和Ridge选择预算；不是用Big5挑维数。

| 参照 | mean Big5 r | 载荷复采样一致性中位数 | held坐标相关中位数 | 分半相关中位数 |
|---|---:|---:|---:|---:|
| 词频 → MINRES共同因子 → promax → Ridge | 0.0296 | 0.9076 | 0.9138 | 0.6565 |
| 同词频 → PCA5 → varimax → Ridge | 0.0337 | 0.8679 | 0.8965 | 0.6209 |

预测差Δr=-0.0041，配对OOF bootstrap 95% CI [-0.0136, 0.0053]。
结构稳定性差值是描述性比较，未检验其显著性；不以此宣称共同因子普遍胜出。
20个CV拟合、8个发现集dropout拟合均无负特有方差。固定发现解中最大因子相关绝对值
0.2575，最小特有方差0.2065，off-diagonal RMS为0.0448。

标准模型为 `R ≈ L Phi L' + diag(psi)`。新样本使用训练均值、标准差及固定回归权重计分。
已用已知相关三因子、非等方差残差的合成数据检查恢复；单独评分与换批评分一致。
BLT原文的完整估计器/旋转顺序和原始词矩阵没有公开复得，故称BLT式适配。
OSF README已核：发布的是混淆后的因子分数和验证notebook，不是原始词矩阵。

**修正旧描述**：本轮GRAIN对照artifact是词汇TF-IDF(1-2gram, max5000)→SVD24，
再取24坐标的均值和标准差。它不是24种命名行为指标，也不是全部GRAIN评分器。
此前称“24个词汇行为指标”过于具体，现改正。其r=0.0864的计算没有改变。

## 3. SIMPA家族参照与错误类型

采用W71释义模型的原生mean pooling与128token上限；100/29,375段被截断。
与原生SentenceTransformer对10条公开条目的实现校验最大绝对差为0。
与E5比较时，编码器和原生长度上限同时不同，记录为家族适配比较。
SIMPA正式期刊论文发表于2022年；2025论文是后续博士总结，不是最早来源。

| 同源作者上的方法 | E5匹配计数 | 释义模型匹配计数 |
|---|---:|---:|
| 计数 → Ridge | 0.0632 | 0.0095 |
| E5语义表示 + 计数 | 0.1836 | 0.1891 |
| GRAIN词汇SVD分支 + 计数 | 0.0983 | 0.0777 |

参照E5语义单分支仍为0.1957。释义计数叠加后的差为-0.0066，CI [-0.0226, 0.0076]；
没有观察到这一配置的增量。不能把稀疏、未校准、未专家扩展的适配器当作SIMPA全部能力。

**阈值作用与句对正确性必须分开。** 释义模型对随机条目配对的接受率为0.0103%，
但随机配对不是真实负标签，不能把它报成false-positive rate。
10组每组4种设计句对的接受比例为：

| 句对类型 | E5 cosine≥0.6 | 释义 cosine≥0.6 | 通用NLI判为entailment |
|---|---:|---:|---:|
| 同一人的同向释义 | 10/10 | 6/10 | 9/10 |
| 同一人的反向陈述 | 10/10 | 7/10 | 0/10 |
| 描述别人但句中仍有I | 10/10 | 8/10 | 0/10 |
| 单次事件而非明确习惯 | 10/10 | 2/10 | 3/10 |

设计句对公开可检查，属于小型语义机制测试，不是自然文本precision/recall校准。
NLI有4个“描述别人”被判成contradiction而非neutral，另有泛化/程度错误，全部保留。
因此不会把所有非支持简单翻成反向人格分数。

新读的[Affective-NLI W72](https://arxiv.org/html/2404.02589v1)已经使用对话、情绪及
正反人格描述进行监督学习。其两套数据是电视剧对话及角色标注，不是我们的真人自报
连续标签。本批采用的是独立的[通用NLI W73](https://huggingface.co/cross-encoder/nli-deberta-v3-base)，
既没有复现它的训练，也不把它的分类机制当GRAIN首创。

## 4. 条目分数的可用性与误差

每域至少一条匹配的作者比例依E/N/A/C/O为18.43%、17.31%、8.49%、11.38%、12.98%。
作者匹配数中位数为0；不能把无证据当中性，也不能只报告有分数者而隐藏覆盖。

加入整评论重采样，保存正向比例在固定检测结果下的抽样变动。
3,120个作者×域单元中，3,112个未达到预设的4条含证据评论显示门槛；仅6个非退化区间、
2个退化区间。该门槛是保守显示规则，不是心理测量样本量定理。
区间不包含检测错误、参照拟合或真实时间依赖，不冒充个人“真实人格”的CI。

另外产出80条source-only分层盲评包，每个cosine区间20条；不含Big5标签或相似度显示。
保留原来的随机80条包，新增分层包解决其几乎只抽到拒绝项的问题。
标注规范需分别记录归属、同/反向、概括程度及相关构念，不能让“与条目很像”承担全部判定。
这些包尚未标注；专家eTRS、原始反馈标签和原checkpoint仍未取得。

## 5. 当前阶段决议

| 事项 | 当前决定 |
|---|---|
| MEM与共同因子的一般数学 | 继承标准实现，保留两条参照，不重复发明旋转/特有方差模型 |
| 无生成式LLM的语义读出 | 保留E5+Ridge可运行参照；不是脱离所有预训练语言模型 |
| 绝对cosine直接当心理证据 | 不作为统一判定；检索、归属、方向、时间概括分别建模 |
| 极性与语义关系工具 | 先采用现成NLI参照，不从零开发同类分类器 |
| 共享/特有结构 | 保留已跑sJIVE作为竞争参照，不因拼接就承诺收益 |
| 条件/时间/误差接口 | 保留标准随机效应、OU与分层抽样；真实机会集合继续unknown |
| 全GRAIN vs前人能力 | 未作整体胜负裁决；当前只比较冻结词汇分支 |

I1/I2的两处重要适配缺口已补，I3有更多同协议结果。阶段仍active：句对独立校准、
完整GRAIN分支的可比接口、真实条件层与检测误差传播尚需作明确的下一批处理决定。
不得把这些余项静默改写成已完成。下一批任务与新的RQ候选见
[继续执行计划](../../docs/GRAIN_PREDECESSOR_NEXT_TRANCHE_V1.md)。

## 6. 复现与审计

所有私有结果位于`results/grain_vector_mechanisms/predecessor_integration_v1/`，gitignored。
新子目录：`common_factor_r1`、`paraphrase_r1`、`entailment_r1`、`evidence_sampling_r1`。
原作者文本/个体预测/词载荷/标注包不进入公共文档或提交。

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
# 已完成的输出不可覆盖，复跑时给 --output 新目录。
"$PY" scripts/run_grain_common_factor_reference.py --output results/grain_vector_mechanisms/predecessor_integration_v1/common_factor_r2
"$PY" scripts/run_grain_paraphrase_reference.py --allow-download --output results/grain_vector_mechanisms/predecessor_integration_v1/paraphrase_r2
"$PY" scripts/run_grain_entailment_reference.py --allow-download --output results/grain_vector_mechanisms/predecessor_integration_v1/entailment_r2
"$PY" scripts/audit_grain_evidence_sampling.py --output results/grain_vector_mechanisms/predecessor_integration_v1/evidence_sampling_r2
"$PY" -m pytest tests/test_grain_common_factors.py tests/test_grain_paraphrase_reference.py tests/test_grain_predecessors.py tests/test_grain_reference_integration.py tests/test_grain_vector_mechanisms.py -q
```

无新依赖安装；复用R psych/GPArotation和Python环境。公开模型为固定revision的safetensors，
未允许remote code。编码器`position_ids`兼容性提示已观察，无缺失训练权重提示；
释义encoder与原生SentenceTransformer的独立输出核对一致。

上述五个测试文件合计133 passed，21.01秒；2条既有SWIG弃用警告。
新增Python文件ruff通过；R脚本经过已知模型和28个真实/复采样拟合执行。
`git diff --check`通过，私有计分/原句/盲评文件均确认被gitignore。
