# GRAIN 参照与推论机制连接实验

2026-09-09。INT-01 完成；INT-02 同材料缓存对照完成。后续条件分布、神经中介与跨查询工作仍在计划中。本轮无新 LLM 调用，无训练/评价外部量表。

## 1. 发现与决策

**可以把两条路线连接成同一测量链，而不抹掉它们的区别。** 查询函数提供证据读数，GRAIN 提供条件分布、群体参照和作者位置/响应的区分。已实现可计算的接口，而不是强制把旧 SUICA-19 因子接进预测器。

本轮得到了一个有用的局部构造：响应修正在所有源作者均值锚点上严格为零，却仍能保留 T/F 否认与状态时间响应。与此同时，新作者的水平偏差没有随之消失。说明**固定源参照点是可实现约束，跨作者迁移是另一个需学习的对象**，不能用归一化替代。

最有价值的下一步不是调一轮新的混合权重：同材料下，较后层固定函数 f 已很好保留 native 排序，早层压缩更弱。应定位早层到有效证据通道的转换，再在多文本参照中比较这些通道。

## 2. 实际设计

- 查询：extraversion、T/F、adapted_dejected5。最后一个是当前低落状态查询，不把它改称人格因子。
- 每查询源数据：48 控制世界 + 12 源作者平均激活锚点 + 96 源作者改写端点，共 156 行。
- 源内留一作者：去掉该作者全部端点和锚点；共 12 折。另作一次最终源拟合。
- 自然材料评价：另一组 12 作者 × 8 条件，12×8 不是 96 个独立作者。这是之前已暴露的探索面板。
- 三个固定公式：linear_value、linear_secant、anchor_secant。核与 CON trace 权重不扫参。
- 数据用途：native/f 是计算目标，不是人类人格标签；没有读入 PANDORA 人格标签或 MEPS。源作者与评价作者交集为 0。
- 条件是改写同一真实材料的归属、否认、时间和移除，不是被试重复行为实验，不估计自然 topic 选择概率。
- 计算：run_r1 28.16 秒；格式修正及加入同材料对照后的 run_r2 29.13 秒。两次核心结果表完全相同。2 CPU 线程，零新增神经调用，无后台队列。

## 3. 与 GRAIN 的数学连接

完整链为：

\[
\text{作者多文本与条件}
\to R_vS_\lambda(T)
\to g_q(\text{证据、条件、历史})
\to \{m_{uq}(c),\pi_u(c),\text{响应/路径}\}
\to \Psi(\cdot;P_0,Q_0,v)
\to \text{外部心理/行为关联}.
\]

g 是可替换且版本化的观察通道，不是 GRAIN 全体。当前四 block 程序仍有神经前端；本轮的 NumPy 计算从既有激活运行，不能宣称已经完成全新 raw-text 无 LLM 系统。

参照分解实现为 \(s_{uq}(c)=b_q(c)+a_{uq}+r_{uq}(c)\)，其中 \(b\) 只从 12 源作者估计，\(a\) 是相对条件参照的平均偏移，\(r\) 在固定条件分布下均值为零。重建误差最大 \(5.55\times10^{-17}\)，同一作者更换评价批次后的 profile 变化为 0。预测器自身批次计算的浮点差异小于 \(10^{-15}\)。这些是接口的代数性质，不是新的人类重测信度。

不同程序使用不同的明确版本参照，不能把它们的 z 分数当作等价心理单位。原始文本分数、八条件平均和响应形状分别保存；没有把“八种人工改写的平均”称作自然人格基线。

更一般的选择/响应分解亦已实现并通过恒等式测试：

\[
\mathbb E_{\pi_u}m_u-\mathbb E_{\pi_0}m_0
=\underbrace{\mathbb E_{\pi_0}(m_u-m_0)}_{\text{同条件响应}}
+\underbrace{\mathbb E_{\pi_u-\pi_0}m_0}_{\text{选择/曝光}}
+\underbrace{\mathbb E_{\pi_u-\pi_0}(m_u-m_0)}_{\text{交互}}.
\]

这使原 GRAIN 的 where/choice 与新查询函数有了精确接口。本轮只测其代数，真实 choice 分解等待 INT-03，未用人工设计频数冒充人格选择。

理论来源与迁移范围在[连接计划](../../docs/GRAIN_REFERENCE_MECHANISM_INTEGRATION_PLAN_V1.md)中逐项列出。最直接的接口是[条件核嵌入](https://icml.cc/2009/papers/538.pdf)与[分布回归](https://proceedings.mlr.press/v38/szabo15.pdf)；心理含义接续 GRAIN 已有的[状态分布研究](https://pubmed.ncbi.nlm.nih.gov/11414368/)。使用这些理论不意味着它们保证了本数据集的心理效度。

## 4. 受限核的结果

\[
K_\perp=K-K_{:A}K_{AA}^{+}K_{A:},\qquad
\hat s=s_{\rm value}+g_\perp.
\]

这是标准核子空间投影，与[GP 条件协方差的 Schur 补](https://gaussianprocess.org/gpml/chapters/RW2.pdf)有同样代数结构。本轮用作确定性约束，不采用高斯心理分布假定。修正在源锚点为零，截距不额外自由拟合。

| 查询 | 原 secant 源锚点最大漂移 | 受限修正最大漂移 | 修正核保留的源 trace 比例 |
|---|---:|---:|---:|
| E | .011955 | 2.78e-16 | 15.33% |
| T/F | .034285 | 8.88e-16 | 17.28% |
| 状态 | .016779 | 5.83e-16 | 18.15% |

剩余 15–18% 是中心化源输入核的 trace 比例，不是“人格信息比例”。投影仅限制新增修正，不从原 baseline 删除其余输入。**保住参考点和保留部分响应确实可以并存。**

自然材料、相对于 native 变化的两个预先选定效应：

| 效应 | value r / 幅度比 | secant r / 幅度比 | 锚定 secant r / 幅度比 |
|---|---:|---:|---:|
| T/F 否认 | .343 / .229 | .585 / .917 | **.598 / .917** |
| 状态时间 | -.184 / .110 | .789 / .350 | **.765 / .312** |

这不是人格预测相关，也不代表 .598 显著超过 .585。相对 value 的响应形状恢复具有实质变化；源作者留一中的对应 r 是 T/F .349→.279、状态 .088→.267，前者相关未提高但幅度从 .142→.630。必须把自然面板与源内结果并列，不只看较大的数字。

所有四类响应相关如下，避免只展示优势单元：

| 查询 / 操作 | value | secant | 锚定 secant |
|---|---:|---:|---:|
| T/F / 归属 | .356 | .420 | .415 |
| T/F / 否认 | .343 | .585 | .598 |
| T/F / 时间 | .229 | .386 | .381 |
| T/F / 移除 | .021 | -.089 | -.080 |
| 状态 / 归属 | .072 | .159 | .146 |
| 状态 / 否认 | -.715 | .134 | .125 |
| 状态 / 时间 | -.184 | .789 | .765 |
| 状态 / 移除 | .038 | -.037 | -.007 |
| E / 归属 | .262 | .217 | .174 |
| E / 否认 | -.040 | .185 | .278 |
| E / 时间 | -.172 | -.105 | -.104 |
| E / 移除 | -.386 | -.161 | -.147 |

**水平权衡没有关闭。** 状态全部端点对 f 的相关：value .433、secant .211、锚定 .220。只取原始材料：.440、.211、.215。源锚点约束不是新作者区域的约束。所有四操作合并后的 paired-author native 响应 ΔMSE，锚定相对 value：T/F +.000319、状态 +.0000438、E -.000000981；三个 95% bootstrap CI 均跨零。故不把此程序晋升为总体最优预测器。

## 5. 同材料目标对照：INT-02

8 个通道、3 查询、原始/全部端点/作者平均/4 类响应、native 与 f 两目标，共 336 行。全部使用同一批 12 作者，未增加层位搜索。native 与自身 r=1 只是计算校验。

原始材料对 native 的相关：

| 查询 | 固定 f | B0 | B4 | value | secant | 锚定 secant |
|---|---:|---:|---:|---:|---:|---:|
| E | .939 | .506 | -.153 | .167 | .172 | .112 |
| T/F | .844 | -.351 | -.251 | -.123 | .165 | .085 |
| 状态 | .952 | -.456 | .355 | .417 | .215 | .216 |

固定 f 的否认/时间响应对 native 相关分别 .912/.990。f 可读到大量目标相关运算；早层压缩只恢复其中一部分。f 自身也有尺度偏差，例如 T/F 否认幅度比 1.605，所以要同时看 MSE、斜率与相关，不能把 f 宣称为原生模型精确复制。

这里的 B0/B4 数值是**12 作者对 native 的一致性**，不是之前 64 人 T/F AUC 或 46 人状态问卷相关的反转。两个评价目标不同。此结果支持下一步研究压缩链的证据转换，而不是再次用外部标签找一个更高 r。

## 6. 下一步已经具体化

1. INT-03 先核查 VM02/GRAIN 条件视图与导航作者材料的交集，建立同材料的多文本分布接口。观察作者“位置”和“响应形状”是否跨文本对应；先不硬聚类，不按旧支持半径强行打分。
2. INT-04 做 T/F 否认和状态时间归属的中介定位：沿已知层位，把原生分数所需的条件解析/证据汇总变成可测函数。若需新调用，只补一个固定小面板。
3. 在这两个接口上扩展 INT-05 的共享/特异查询函数，再执行 INT-06 冻结外部连接。保持 Big5/MBTI 小特质都可接，但不启动全因子性能搜索。

本轮没有启动上述后续神经计算，也没有宣布整个计划完成。已经完成的是机制通道到 GRAIN 参照逻辑的首个可执行连接及其边界定位。

## 7. 文件与复现

代码：[reference_integration.py](../../grain_research/vector_mechanisms/reference_integration.py)；运行器：[run_grain_reference_integration.py](../../scripts/run_grain_reference_integration.py)；测试：[test_grain_reference_integration.py](../../tests/test_grain_reference_integration.py)。

最终私有包：`results/grain_vector_mechanisms/reference_integration_v1/INT-01/run_r2/`。run_r1 保留为首次运行记录，不覆盖。

- `construction_freeze.json`：来源、源拟合、代码、计划哈希。
- `prediction_freeze.json`：自然预测保存后、目标拼接前的哈希。
- `anchor_diagnostics.csv`、`level_metrics.csv`、`response_metrics.csv`。
- `reference_profile_checks.csv`、`batch_invariance.csv`、`paired_author_intervals.csv`。
- `target_location_comparison.csv`、`comparison_provenance.json`。
- `program_*.npz`、`reference_*.npz`、`private_*`：本地私有工件，不上传。
- `summary.json` 与 `run_status.json`：完成状态、用途、耗时。

在独立 GRAIN 仓库运行：

```bash
"../project persona/.venv/bin/python" scripts/run_grain_reference_integration.py --output-dir results/grain_vector_mechanisms/reference_integration_v1/INT-01/run_r3
"../project persona/.venv/bin/python" -m pytest -q tests/test_grain_reference_integration.py tests/test_grain_conditional_models.py tests/test_grain_reference_system.py tests/test_grain_reference_framework.py tests/test_grain_reference_groups.py tests/test_grain_conditional_navigation.py
```

需要本地已冻结的 CON/NAV 缓存，运行器不会联网下载或重新生成缺失作者资料。33 项相关测试通过；ruff 检查与格式化通过。未重跑整个仓库测试集。测试框架还报既有 SWIG 类型弃用警告，不影响本轮测试结果。
