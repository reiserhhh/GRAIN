# 条件响应模型构造报告

2026-09-13。C01/C12/C15/C17 续批；主 goal 继续 ACTIVE。

## 本轮完成

用624名作者的22,464条原评论检验整组条件隐藏；不读取人格量表标签。
来源训练/验证/最终来源/开发留出=378/126/504/120。
固定来源条件2506个；30维是IPIP-keyed语义响应坐标，不是直接问卷得分。

| 方法 | 作者数 | 隐藏条件数 | NMSE | MSE | MAE |
|---|---:|---:|---:|---:|---:|
| source_prior | 120 | 1239 | 0.910515 | 0.00019253 | 0.010281 |
| intercept | 120 | 1239 | 0.910894 | 0.00019233 | 0.010316 |
| grain_gp | 120 | 1239 | 0.914958 | 0.00019315 | 0.010362 |
| shuffled_geometry | 120 | 1239 | 0.912746 | 0.00019282 | 0.010348 |

NMSE以来源作者条件内单位残差方差逐维归一化，先作者内再作者间平均。
不是Pearson r/AUC；数值越低越好。整组隐藏后目标评论不进入该作者的条件输入。

| 配对差（负数更好） | delta NMSE | 作者bootstrap 95%区间 |
|---|---:|---:|
| intercept_minus_source_prior | 0.000379 | [-0.021264, 0.020893] |
| grain_gp_minus_intercept | 0.004064 | [0.001268, 0.006631] |
| grain_gp_minus_source_prior | 0.004442 | [-0.017770, 0.025558] |
| grain_gp_minus_shuffled_geometry | 0.002211 | [-0.002528, 0.006785] |

## 公式与计分出口

标准高斯过程条件分布提供：mu_T + K_TS (K_SS + diag(1/n))^-1 (m_S-mu_S)。
参考 [GPML第2章，式2.23/2.24/2.38/2.40](https://gaussianprocess.org/gpml/chapters/RW2.pdf)。
条件描述符使用来源GRAIN24条件均值，RBF中位距离规则固定；噪声模型是假设而非人格误差定理。

固定nu输出严格拆分为：已观测贡献 + 缺测条件的来源参照 + 缺测条件的个人支持修正。
模型补全不改变原先缺测经验point=None，也不改变覆盖率。每项保留单人向量/条件/贡献。
model_latent_sd只含给定来源矩及核的潜在缺测函数不确定性；不含来源估计或真实问卷误差。

| 模型 | 目标 | 覆盖中位 | 完整经验目标 | 平均个人补全幅度 | 范围越界数 |
|---|---|---:|---:|---:|---:|
| source_prior | full | 0.0460 | 0 | 0.000000 | 0 |
| source_prior | common5 | 0.3355 | 0 | 0.000000 | 0 |
| intercept | full | 0.0460 | 0 | 0.002615 | 0 |
| intercept | common5 | 0.3355 | 0 | 0.001939 | 0 |
| grain_gp | full | 0.0460 | 0 | 0.002803 | 0 |
| grain_gp | common5 | 0.3355 | 0 | 0.002132 | 0 |
| shuffled_geometry | full | 0.0460 | 0 | 0.002748 | 0 |
| shuffled_geometry | common5 | 0.3355 | 0 | 0.001985 | 0 |

## 制作规则与下一步

当前固定门槛是否支持条件几何升级：False。
只依据本轮预先声明的GP相对截距及置乱几何比较，不继续搜核/带宽/融合权重。
来源未出现条件保留原观测与unknown标志，不伪造来源条件或默认平均人。
自然社群条件是作者选择与他人构成的观测条件，不能当作随机刺激的因果响应。
Essays只有一个已记录条件，不足以分离截距和跨条件函数；不拆段冒充新条件。

模型给出可复算的借力方式；未来环境可复用制作规则并重新冻结条件参照。
完整Study B、真实机会集合、路径/心理解释及独立复制仍由总台账继续推进。

## 可复跑产物

私有目录：`results/grain_vector_mechanisms/all_plan_closure_v1/conditional_response_r2`；原文/身份/向量均不公开。
主入口：`scripts/run_grain_conditional_response.py --output <new-private-directory>`。
`contract.json`记录输入与代码hash；`selected.json`和`source_selection.csv`记录来源选参。
`held_metrics.csv`、`paired_author_intervals.csv`、`completion_summary.csv`为汇总。
`held/private_reference.json`为冻结模型，`private_*`文件保留所有重算地址和值。
主运行 3.18 秒，零神经调用；算术审计另见audit.json。

## 结果解释与验收补记

本轮填的是“缺测条件如何借力”的具体制作环节，并非继续追逐人格预测分数。
随机截距与来源条件参照几乎持平；条件核相对截距的NMSE增加.004064，配对区间
[.001268,.006631]。因此三种可用公式都保留为参照，但不将条件核或未观察条件的
个人补全设为默认测量通道。GRAIN24的条件均值相似，不等于作者对条件的反应相似。
这与此前跨文本作者结构、向量路径或人格外部相关的结果不是同一个检验。

额外算术解释使用同一批固定预测，不重fit、不选择新模型。对任一新增修正d，
设旧预测残差e=y-pred_old，则精确地有：

    risk(new)-risk(old) = E[d^2/sigma^2] - 2 E[d e/sigma^2].

这里的E按本轮的“先坐标/作者内条件，再作者”权重计算。相对截距，条件核新增
修正能量为.001962，而两倍有效对齐为-.002102，两者相减正好为+.004064。
这是为什么“加入条件距离”没有改善当前响应重建的定量解释；不是证明所有条件
函数无用，也不是凭经验推翻高斯过程定理。见`risk_decomposition.csv`。

高斯模型可给出完整数值，但**经验完整度仍为0/120**。来源prior臂的模型潜在SD=0
只表示该臂把来源均值当作固定常数，并不表示它没有预测误差。不同模型SD不能直接
当作信度排名；仍需保留经验覆盖、实际误差和来源估计不确定性的区别。

验收：12,192个隐藏单元预测以精度矩阵LOCO公式全量复算、960个模型完成对象、
480条风险分解、6个原向量冷评分均通过，最大算术差2.22e-16。109项相关测试通过。
单独CLI已实际执行，输出coverage=.0692791、empirical_complete=false；无后台任务。
这仍是当前执行者的独立公式核验，不冒充另一个执行者的研究复制。

最终产物为r2；r1因新增模块/测试缺shebang触发挂载卷上的Ruff EXE002，r2只修正
文件头并全量冷跑，样本、配置、选择和数值均未改变。第一次pytest从父repo启动
因import路径失败，切回独立repo后全部通过；无依赖安装。r1保留，不覆盖。

冷评分命令（在独立repo根目录，输出路径必须尚不存在）：

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/score_grain_conditional_response.py \
  --model results/grain_vector_mechanisms/all_plan_closure_v1/conditional_response_r2 \
  --arm grain_gp \
  --directions results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/private_directions.npz \
  --input results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/pandora/private_single_input.npz \
  --target full --output results/grain_vector_mechanisms/all_plan_closure_v1/conditional_response_r2/private_new_score.json
```

输入必须沿用同一E5 revision、query前缀、原评论单位和向量协议；维度相同不代表
可任意更换encoder。方向包和来源模型须保持配套。下一步转回C00/C11的历史窄目标
证据核清与C03/C16有针对性的证据对象建设，不追加核/带宽救援网格。
