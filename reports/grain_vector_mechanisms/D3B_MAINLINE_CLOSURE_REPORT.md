# D3B：新作者响应检验与主线阶段闭环

2026-09-08。状态：COMPLETE。32 位全新作者，64 个文本视图，580 次 forward，
649.63 秒。模型和 ridge 均在查看这些响应之前冻结；本轮没有训练 LLM。

## 裁决

**保留同一查询下跨作者共用的紧凑响应公式，不添加当前 GRAIN 群体增益项。**
在新作者上，共用公式能预测向量干预产生的判断变化。加入群体调节对神经质的
排序略有帮助，但两个任务的 MSE 都变大，额外复杂度没有换来更好的主指标。

本轮有限 goal 形成了可复现链：真实作者人格关联 -> 查询引导的证据组织 ->
实际向量干预 -> 新作者响应预测；群体关系以响应特定的邻域线索保留，而非
强加给所有响应的一套增益。精确恒等式与实验近似分别归档。

## 1. 设计与冻结

- 开发集：D2 全部64人。D2 原始留出结果已被阅读，因此本次不再把它当新材料。
- 新作者：20 位 TF、12 位神经质；排除 D1、D2 及冻结 GRAIN 参照成员。
- 每人两视图，每视图两种证据 donor、两种 patch 剂量，共8次响应。
- 两个模型均使用训练位移的 rank8 子空间；投影按训练 RMS 缩放，零位移对应零响应。
- 群体调节模型另用训练作者 GRAIN profile 的4维 PCA 坐标形成双线性交互。
- ridge 候选1/10/100，按开发集3折作者分组 CV 的 MSE 选择。
  两任务都选 shared=10、conditioned=100；预处理全部在每个训练折内拟合。
- repeat/self-patch 最大分数误差均为0。未读取心理标签、bridge 或 MEPS 问卷。

此处 shared 指**同一查询下跨作者共用**，不表示 TF 与神经质共用相同系数。
跨特质的共享是另一个研究对象，见本报告第4节。

## 2. 完整主比较

| 查询 | 模型 | 新作者 / 响应 | r | RMSE | MSE / 零响应 MSE |
|---|---|---:|---:|---:|---:|
| T/F |共用算子|20 /160|.391710|.022540|.869855|
| T/F |GRAIN调节|20 /160|.364596|.022848|.893827|
| 神经质 |共用算子|12 /96|.369447|.009691|.498677|
| 神经质 |GRAIN调节|12 /96|.399893|.010392|.573477|

共用算子相对“没有变化”的 MSE 下降13.01%/50.13%。这些 r 是干预响应预测，
不是本轮新增的问卷预测精度，也不是160/96名独立作者的相关。

按作者重采样，conditioned-minus-shared 的配对 MSE 差：

- T/F：+.00001400，95%区间[-.00004940, .00008099]。
- 神经质：+.00001409，95%区间[-.0000003584, .00003067]。

这支持当前保留简单模型的工程选择，不把两模型差异写成已确定的总体优劣。
开发集 CV 也偏向共用模型：TF MSE/zero=.7166 vs .7955；N=.5386 vs .6709。

## 3. 数据细节给出的下一步方向

分开看 donor 与剂量，响应并非均匀分布：

| 新作者共用公式的响应 r | 半剂量 | 全剂量 |
|---|---:|---:|
| TF：移除 query-selected 评论所得 donor |.4689|.4619|
| TF：移除对照评论所得 donor |.3263|.3203|
| N：移除 query-selected 评论所得 donor |.3745|.3763|
| N：移除对照评论所得 donor |-.0005|-.0132|

神经质对照条件仍有较低 MSE，但作者/视图间的响应排序接近零。因此不能把总
MSE 改善解释为所有条件下同样强的区分力。更有价值的方向是查询相关的证据
坐标：两个任务在选中证据条件下都保留正向的响应预测。

公式可复用不要求每种文本变化都同样有信息。群体调节增添了少量响应排序信息，
但当前尚不值得替代较简单的公式；保留完整预测和负条件，不继续追调这些作者。

## 4. 用户新增理论的缓存实验

对两个开发集共用算子做主角与功能向量分析：

- rank8 位移子空间重叠系数：.272375。
- 两个功能读出方向的余弦：.089451。
- T/F 方向在 N 空间内的平方投影比例：.262506；反向为.204244。

**部分空间相近与不同的读取方向同时存在。** 这为“共同数学程序 + 特质专属
读出”提供一个具体实例。它是几何结果，不是 TF 与 N 量表相关，也不表示已经
找到了精确共用的独立人格成分。窄特质可以在同一程序内被研究，不必只限于
Big Five / MBTI。

详见 `docs/GRAIN_SHARED_AND_TRAIT_SPECIFIC_PATHS_V1.md`。下一轮最有价值的是
同一作者、同一证据变换、多种查询的功能响应矩阵，再研究共享/专属分解；
不是扩大旧矩阵，也不是先要求所有量表都高相关。

## 5. 可复现交付

```bash
# 运行位置：GRAIN 仓库；Python 环境：project persona/.venv
python scripts/run_grain_conditional_response.py --phase fit \
  --development results/grain_vector_mechanisms/d2_real_author_path_r1 \
  --output-dir results/grain_vector_mechanisms/d3b_conditional_operator_REPLAY

# 复用已完成的新作者，不重跑模型
python scripts/run_grain_conditional_response.py --phase evaluate \
  --holdout results/grain_vector_mechanisms/d3b_fresh_author_path_r1 \
  --output-dir results/grain_vector_mechanisms/d3b_conditional_operator_REPLAY

python scripts/analyze_grain_cross_query_geometry.py \
  --operator-dir results/grain_vector_mechanisms/d3b_conditional_operator_REPLAY \
  --output results/grain_vector_mechanisms/d3b_conditional_operator_REPLAY/cross_query_geometry.json
```

既有结果目录是 `results/grain_vector_mechanisms/d3b_conditional_operator_r1/`，包含
`development_cv.csv`、`frozen_selection.json`、4个 NPZ 公式、`held_author_metrics.csv`、
`paired_model_comparison.csv`、`private_held_predictions.csv`、`summary.json` 和
`cross_query_geometry.json`。私人材料、向量和个体预测都留在 gitignored results。

新作者原始观测目录：`results/grain_vector_mechanisms/d3b_fresh_author_path_r1/`。
上一包：`d2_real_author_path_r1/` 和 `d234_real_author_analysis_r2/`。
合计新增96位作者、192视图、1736次 forward，模型计算34.53分钟；旧队列未重启。

验证：26项相关测试通过，Ruff通过。无新依赖、API调用、问卷开放或LLM训练。
主体理论档案：`docs/GRAIN_MAINLINE_MECHANISM_THEORY_V1.md`。

## 6. 结束与下一阶段

本轮主线机制目标完成，所有已启动包结束。当前保留的公式是查询特定、作者
共享的向量响应近似；GRAIN 群体层负责组织可比较的响应与材料，而不是强迫所有
差异都进入个人增益参数。新提出的细粒度特质/共享路径研究作为下一阶段，
不会以“还有任意残差”作为理由无限延长本轮。
