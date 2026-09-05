# VM02：同文本预算的表示对照

日期：2026-09-06。状态：`PILOT_COMPLETE__NO_LLM_SUPERIORITY_ESTABLISHED`。

## 结论

互不重叠的短评论上，E5、词 TF-IDF、BGE 与 Qwen 中层都保留作者对应。**当前没有证据证明 LLM 中层胜过这些基线，也没有把这种对应解释成人格效度。** 下一步应研究上下文中的关系绑定，而不是仅寻找更高的相似度。

这补齐了 24 作者 pilot 的同预算比较，不等于完成 VM02 的大样本、完整 GRAIN 作者计分和跨域要求。

## 数据与估计量

- 使用 `vm03_independent_history_r1` 完全相同的 24 作者 × 3 条历史评论；每个作者三条原文互不重复。另有 8 个技术作者 × 3 条评论用于 TF-IDF/scaler 拟合，不参与主分数。
- 每条 60–180 词，保留原文；BGE/E5 均无截断、未下载新权重。没有使用 Big5、MBTI 或 MEPS 问卷标签。
- 对每个 view 对 a<b，计算本人第二条评论的 cosine 是否高于陌生作者第二条评论；再平均三对 view。这里的 `author_rank_auc` 是条件作者匹配胜率，不是 MBTI AUC 或人格准确率。
- 95% CI 来自 2,000 次作者节点 bootstrap；同一作者的重复抽样副本不能成为彼此的陌生作者。24 作者而非 1,656 条比较边是抽样单位。区间以当前参考拟合、筛选样本和冻结模型为条件。
- 表中 Qwen decision 只用第一个共同 anchor，与此前报告的两个 anchor 均值不是同一个估计量。history mean 在两个 anchor 前缀下相同。

## 结果

| 表示 | 作者匹配胜率 | 作者节点 95% CI |
|---|---:|---:|
| E5-large，整条评论联合编码 | .7319 | [.6577, .8050] |
| word 1–2 gram TF-IDF，reference-only 拟合 | .7107 | [.6439, .7786] |
| Qwen block10，history token mean | .7041 | [.6404, .7805] |
| BGE-large，整条评论联合编码 | .7023 | [.6211, .7759] |
| char 2–5 gram TF-IDF，reference-only 拟合 | .6854 | [.6229, .7459] |
| GRAIN 冻结单评论投影 | .6606 | [.5843, .7459] |
| Qwen block16，decision suffix | .6588 | [.6048, .7166] |
| 旧 23 个 anchor 特征 | .6069 | [.5477, .6654] |
| Qwen input embedding，history mean | .6063 | [.5556, .6589] |
| Qwen final norm，decision suffix | .5713 | [.4936, .6511] |
| format10 | .5646 | [.4962, .6386] |

全部 28 个表示条件、transductive TF-IDF 与 hashing 对照均保存在 CSV；没有只保留赢家。block10 是在同一 pilot 中注意到的，不是独立确认结果。

Qwen block10 相对 BGE 的配对差为 +.0018，CI [-.0587, .0692]；相对 E5 为 -.0278，CI [-.1008, .0521]；相对 word TF-IDF 为 -.0066，CI [-.0878, .0783]。这些宽区间既不支持 LLM 优越，也不证明等效。

### 不能冒充完整 GRAIN 对照

所用旧 GRAIN artifact 是冻结 TF-IDF/SVD 单位投影，24 维，存在历史 PANDORA 接触。**它不是完整 GRAIN 作者分数。** 当前 geometry bundle 要求至少 12 个单位，每份历史仅 1 条，不满足完整计分域；不能把其 std 填零后称完整 GRAIN。后续应另构造多评论、互不重叠的作者视图。

### 条件与来源敏感性

| 条件 | word TF-IDF | BGE | E5 | Qwen block10 |
|---|---:|---:|---:|---:|
| 全部比较 | .7107 | .7023 | .7319 | .7041 |
| 本人正对来自不同 subreddit | .6934 | .6522 | .7030 | .6713 |
| 陌生评论长度在目标评论 1.25 倍内 | .7133 | .6945 | .7205 | .6902 |
| 排除出现类型词的作者，n=23 | .7154 | .7003 | .7372 | .7022 |
| 陌生评论取 BGE 中最接近目标的 5 个 | .7278 | .5944 | .6833 | .6361 |
| 陌生评论取 E5 中最接近目标的 5 个 | .7139 | .6278 | .6528 | .6722 |

语义邻居条件会同时匹配内容、风格与潜在个体信息，不是干净的 topic 因果控制；尤其不能依据这种下降把向量成分删除。它说明结果依赖比较集的难度。

**严格 same-target-subreddit 只有 2 个作者、2 条有效边，不能验收来源控制。** 未为这个条件给 CI；不存在“所有 nuisance 已排除”的结论。跨 subreddit 正对也不排除同一主题、惯用语或个人背景。

第一版敏感性脚本把非空 flag 字典当作阳性，导致 clean 子集为空；已修复为检查字典值并加入测试。`vm02_baselines_sensitivity_r1` 作废，使用 r2。

## 观察器与来源

BGE 使用 CLS + L2、不加检索 query 指令；E5 使用 `query: ` + masked mean + L2，符合其对对称相似度/特征用途的说明。两者为本地 FP32、512 token padding，batch 上限 4；不声称已对这些 encoder 做完整跨设备数值等效证明。[BGE 官方说明](https://huggingface.co/BAAI/bge-large-en-v1.5)、[E5 官方说明](https://huggingface.co/intfloat/e5-large-v2)。

原文相同不代表模型可用信息绝对相同：Qwen 还读取固定 system、任务和中性 anchor；BGE/E5 用各自模型规定的模板。所有模型 revision、pooling、文本哈希与来源均在 manifest/provenance 中。

## 产物与复现

私有产物位于 `results/grain_vector_mechanisms/`：

- `vm02_baselines_full_r1/`：`comparison.csv`、`paired_differences.csv`、`comparison_vectors.npz`、两个 encoder cache、输入 ledger、模型与投影 provenance；96 条文本，28 个比较条件，约 38 秒。
- `vm02_baselines_sensitivity_r2/`：`conditional_comparison.csv`、`coverage.csv`、`paired_differences.csv`。
- `vm02_baselines_smoke_r1/`：跳过两个 encoder 的技术 smoke，不作为最终比较。

```bash
python scripts/run_grain_vector_mechanism_baselines.py --views results/grain_vector_mechanisms/vm02_raw_views_r1/author_views.json --parent-dir results/grain_vector_mechanisms/vm03_independent_history_r1 --output-dir results/grain_vector_mechanisms/new_baselines --device mps
python scripts/summarize_grain_vector_baselines.py --parent-dir results/grain_vector_mechanisms/new_baselines --output-dir results/grain_vector_mechanisms/new_sensitivity
```

## 下一步

1. 不按本表微调人格方向；转向“同词汇、不同关系”的组合控制，测联合编码相对独立编码相加的增量。
2. 扩充独立作者与多评论视图，使用不看标签的来源覆盖规则；重新评估条件匹配和完整 GRAIN 分数。
3. 冻结采集器后才进行 PANDORA 外部连接。当前没有证明人格推断提高。
