# VM05：同证据外部联系实施设计 V1

2026-09-06。下一执行包，尚未采集全量表示、读取心理结果或拟合新头。
承接完整来源取材，而不是重复240作者匹配或把代码控制不合格当作全线停止。

## 要回答的问题

在相同作者、相同36条原评论上，LLM的预回答内部表示是否比一般embedding
保留更多可外部读取的心理信息？信息是否超出显式人格线索、共同讨论和
GRAIN已有观察特征？这检验“可读信息”，不直接证明LLM本身使用了这些信息。

## 先做的无标签预检

1. 输入固定为vm05_full_source_cohort_r1，材料hash与独立审计必须一致。
   来源636 Big5-only、6453 MBTI-only、254 bridge；各官方fold分别保留。
2. 逐模型核对现有权重revision、tokenizer和完整输入长度。三个12评论视图
   都不截断；以完整作者为共同适用单位，排除及适用率须逐来源报告。
   Qwen优先沿用已经验证的3072固定padding；不能拿1536裁掉后半正文。
3. GRAIN冻结算子对36评论全体及三个12评论视图分别运行适用域检查。
   正式拒绝保留；单单位投影/mean/std等只能标中间特征，不能冒称正式分数。
   核对旧投影/参照总体的训练来源；若与本批作者重合，报告既有观察器
   重用而非严格新作者拟合。不放宽半径，不在bridge重估参照。
4. 原文/人格线索遮蔽及共享讨论敏感性的具体规则先进行无标签覆盖盘点。
   线索遮蔽与等量随机词遮蔽区分；共享讨论对照必须匹配减少的文本量。
   不把删除较难作者、删除条件后文量下降解释为人格信号消失。
5. 先固定12位技术作者（各来源按既有hash取4位）测耗时/显存/数值重播，
   不看标签调模型。单批仍上限8小时，缓存上限10GB，.20不挤占其他服务。
   若预算不足，拆来源/阶段保存；不以先完成的小样本代表完整比较。

预检必须输出common_material_manifest、model_manifest、support/coverage和
control_feasibility表。完成后将具体消融材料规则提交封存，再读取源头标签。
此步骤是真正的执行前置，不以该设计文档代替已经完成的实验。

## 固定表示与读出

主表统一使用每人全部36条；LLM可以按三个12条组编码再聚合，但不因此少看
其他24条。不同tokenizer的token数不强行相等，完整相同原文和单位ID才是
同证据。片段可靠性另作三个视图分析，不将它们当三个独立被试。

| 表示 | 角色 | 注意 |
| --- | --- | --- |
| word/char TF-IDF | 局部稀疏参照 | 每次训练折内fit，不用全人群词表 |
| pinned BGE/E5 comment vectors | 一般embedding参照 | 全36评论mean/std；revision与旧机制实验一致 |
| Qwen block10正文pool | 主要内部表示候选 | 预回答、非生成的人格解释；原comment token区间 |
| Qwen block04/block16/final_norm | 固定层敏感性 | 不按bridge或test最佳层挑headline |
| 任务末位、任务差、跨view变化 | 次级内部特征 | 问法/代码影响单列，不称已解释的人格方向 |
| GRAIN完整profile与中间特征 | 既有观察器参照 | 适用域、上游拟合来源、独立新作者边界分列 |
| GRAIN+内部特征 | 增量候选 | 等证据、同头，同样的训练折投影规则 |

第一轮不开全层全维搜索，不训练LLM。密集特征scaler/PCA在每次inner/outer
训练折重新拟合，PCA上限16且不超过floor(n_train/10)。原维Ridge作为
固定敏感性，不按外层成绩追加方法。问法变化前的正文状态需通过因果前缀
不变性；激活改变和外部可读性不同，不能把二者任一单独叫人格机制完成。

Big5：Ridge，alpha固定网格[.1,1,10,100,1000]，以inner CV mean Pearson选择。
MBTI：独立轴LogisticRegression，C=[.01,.1,1,10]，inner AUC选择；不对
ground-truth 0/1宣称连续人格概率。固定0.5阈值的BA与AUC/PR-AUC、Brier分列。
要额外校准或调阈值时，只能用训练侧OOF，不能从bridge选择阈值。

Big5和每个MBTI轴分别沿用官方5个outer folds；inner为作者分组3-fold，
seed731。所有单位随作者移动；缺标签/非二元编码须明确拒绝并计数。
Big5字段为五个英文trait列；MBTI任务源positive_probability须验证0/1
和既有type极性，不能因列名把它当模型预测。Bridge用其五维/四轴标签，
但只有全部源头模型和表示规则冻结后才读取结果。

## 分析与决定

- 个体预测：每trait Pearson/Spearman/MAE/RMSE；每轴AUC/PR-AUC/BA/Brier。
  多视图先形成作者级预测，主要CI按作者配对bootstrap；报告条件于拟合模型
  的区间与模型重拟合的不确定性不是同一种区间。
- 结构后验：用独立源头头对相同bridge文本预测Big5与MBTI，再比较5x4矩阵。
  真实Big5输入的DDF只作结构参照，不进入文本预测能力榜。
- 主要增量只比较预定block10与BGE，以及GRAIN+block10与可用GRAIN；
  相关/准确率无需>.5。看配对差、区间、效应量、线索/讨论控制是否保持。
  未显示增量仍可表明冗余或读出受限，不宣布文本没有人格信息。
- 标签置换必须重跑选择步骤；历史暴露/来源选择和多层敏感性显式标探索。
  主family多重比较与全部负结果保留。不要用本结果直接取代历史.3181。

执行入口目标为run_grain_vector_mechanism_external_preflight.py、
run_grain_vector_mechanism_external_features.py、
run_grain_vector_mechanism_external_links.py（本页为待实施名称）。
产出外部联系表、paired_increment、校准表、bridge结构表及支持/消融报告。
MEPS问卷不进入本包，GRAIN发布计分器不改变。
