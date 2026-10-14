# NEXT91：GRAIN因子交互与同得分组内的向量作用同时存在

2026-09-13。**有限实验完成，独立数值复算PASS；总研究goal仍ACTIVE。**
本批执行2026-09-11已写明但未启动的NEXT91，不恢复后续旧训练队列。

## 1. 主发现与下一步决定

旧GRAIN输入保留了有用信号，新增完整二次交互能进一步对齐LLM的本地查询作用。
但相同GRAIN四维分数下的token仍受到不同读取，这一部分不能只靠增加g的函数解决。
因此下一阶段保留两类输入机制：**跨因子的联合表征，以及同一配置内部的查询-向量对齐**。
不需要把它们误写成两种互相否定的理论，也不是继续搜更多匿名因子数。

| 复用8人，四方向合计 | 神经质查询 | 确认寻求查询 | 两查询差 |
|---|---:|---:|---:|
| 原线性GRAIN矩 | .450520 | .327163 | .356620 |
| 加完整二次交互后的累积作用投影 | .852435 | .624885 | .729802 |
| 任意精确g组权重的饱和累积作用投影 | 1.035920 | .827124 | .850779 |
| 同g组内余项 | -.035920 | .172876 | .149221 |

数值是`sum(tau_part * tau_native) / sum(tau_native^2)`，不是Pearson r、
预测正确率或“人格信息百分比”。饱和投影大于1，是组内反向作用抵消的结果。
所有有符号交叉项均保留，不把负项截断成0。

来源4人的对应结果不同：线性N/C=.008442/.050829，二次累积=.140791/.161307，
饱和累积=.538083/.529047，组内=.461917/.470953。两个cohort均报告；本次不拟合，
“new8”是历史命名的复用材料，不是本轮全新独立样本。

## 2. 数据、算子和数值范围

复用NEXT90完全相同的12位作者、24个原生查询上下文、H16状态、BODY地址、四个g
及四个原始扰动方向。先重新核验旧551项输入绑定和通过的审计，再冻结本批代码/计划。
没有新增作者、查询、原生模型调用、拟合、问卷标签、MEPS或锁箱访问。

三个嵌套空间：A1=[1,中心化g1..g4]；A2在A1上加入完整10个gi*gj；
Pcell按四维g**精确相等**分组，计算组内均值。没有四舍五入或按结果选分箱宽度。
非零列先归一长度再用机器精度SVD秩规则，保留原空间而不改变度量。

| 几何 | 来源4人 | 复用8人 |
|---|---:|---:|
| token数量范围 | 794–1004 | 692–1127 |
| 线性/二次秩，所有人 | 5 / 15 | 5 / 15 |
| 精确组数范围 | 63–76 | 68–82 |
| 单token组占全部token | 2.49%–3.90% | 2.93%–4.05% |
| 组内自由度中位数T-cell_count | 790 | 815.5 |
| 二次设计矩阵条件数范围 | 7.15–11.67 | 5.65–10.98 |

饱和上界并未因为所有token都是单例而退化为恒等映射；每组通常有多个token。
它允许在本材料每个不同g上指定任意权重，是宽松的描述上界，不是来源拟合的可部署函数。

## 3. 现在可以写清的公式

固定欧氏/Frobenius度量、g和非BODY位置。J为同一原生标量查询输出对BODY的梯度，
D为一个原始GRAIN调制方向；定义三个正交投影P1、P2、Pc：

\[
J=P_1J+(P_2-P_1)J+(P_c-P_2)J+(I-P_c)J,
\qquad \tau=\langle J,D\rangle=\sum_{k=1}^{4}\langle J_k,D\rangle.
\]

前式的四个向量正交，故梯度能量相加；后式的四个标量不正交：

\[
\tau^2=\sum_k\tau_k^2+2\sum_{k<l}\tau_k\tau_l.
\]

饱和空间是有限均匀token分布上以g为条件的期望：PcJ=E_emp[J|g]。
剩余作用具有具体可解释的数学对象，而非泛称“未知噪声”：

\[
\tau_{within}=\sum_c n_c\operatorname{tr}\operatorname{Cov}_c(J,D),
\]

Cov分母用n_c；组内方向与查询梯度如何共同偏离各自均值，决定了这项作用。
24上下文×4方向及查询均值/差中的该等式已数值核验，最大误差1.44e-15。
这是标准条件投影/协方差分解在当前路径接口的应用，不是GRAIN新发明的数学定理。
[Pitman讲义§10.2.3](https://www.stat.berkeley.edu/users/pitman/s205f02/lecture15.pdf)
给出条件期望的Hilbert空间投影视角；[MIT投影课程](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/projections-onto-subspaces/)
为有限维算子参照。

二次扩展所增加的是**新的输入统计量**`mean(gi*gj*h)`。
这不同于对旧`mean(h), mean(gi*h)`换一个非线性预测头：若输入已把某方向压到核空间，
下游再复杂也不能恢复该方向的一阶响应。当前结果支持扩展输入中的因子交互，而不是
把任何“非线性模型”都视为等价的解决方法。

## 4. 向量能量、具体方向和查询共享性

复用8人组内余项保留N/C梯度能量86.77%/89.01%；这不是86%–89%人格信息丢失。
它在四个被测方向上的有符号作用可以小、为负或被少数方向主导。
例如F03组内项：N=-.144267，确认寻求=1.042700；对应平方作用比=.475634/1.285740。
同一配置余项对不同查询可以反向，这反对统一正权重或一刀切删除余项。

组内两查询的共同能量比例为复用8人.775876、来源4人.859548；平均作者余弦
.658483/.726829。共享的向量读取和查询差别两者都保留。共同部分也可含一般模型处理，
不直接命名为共享心理特质。

## 5. 验收与产物

12个作者单元、5张私人明细表和5张cohort表均完成，analysis/replay文件逐字节相同。
公开表行数为2/32/160/240/8；审计独立重建全部汇总，另以最小二乘和显式分组复算投影。

- 原NEXT90线性作用对齐误差：4.44e-15；旧结果原样保留。
- 作用分解误差：4.47e-15；梯度能量误差：5.26e-16。
- 独立二次投影误差：7.91e-17；独立cell投影误差：0。
- 准备/分析/重放/审计的规范数值CPU合计8.44秒，低于冻结60秒预算；未按结果扩预算。
- 93项相关测试通过，2条既有SWIG弃用警告；本批新增8项测试。

私有可重跑目录：`results/grain_vector_mechanisms/logic_grain_cell_residual_v1/`。
代码：`grain_research/vector_mechanisms/grain_cell_residual.py`、
`scripts/run_grain_grain_cell_residual.py`、`scripts/report_grain_grain_cell_residual.py`。
公开数字与hash：[publication.json](grain_cell_residual_v1/publication.json)、
[作用表](grain_cell_residual_v1/responses.csv)、[能量表](grain_cell_residual_v1/gradient_energy.csv)。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
OUT=results/grain_vector_mechanisms/logic_grain_cell_residual_replay
"$PY" scripts/run_grain_grain_cell_residual.py prepare --output "$OUT"
"$PY" scripts/run_grain_grain_cell_residual.py analyze --output "$OUT"
"$PY" scripts/run_grain_grain_cell_residual.py replay --output "$OUT"
"$PY" scripts/run_grain_grain_cell_residual.py audit --output "$OUT"
"$PY" -m pytest -q tests/test_grain_cell_residual.py
```

这些命令需要未改动的本地历史缓存，公开仓库不含私人文本/神经状态。代码阶段分别
启动进程、验证hash并限制数据文件读取，完成输出不覆盖；不要求重新启动旧模型。

## 6. 接回总计划

C09/NEXT91的规定问题、比较、重放和验收已完成；该结论为测量输入选择提供新证据。
C14继续研究真实同地址的GRAIN配置与语义证据，C07/C08继续负责可独立计算的读者。
开发候选为来源固定的完整二次矩与查询条件下的局部向量对齐，两者按作用分别保留。
当前梯度是机制诊断目标，不可偷偷作为新作者预测器的输入。
不自动延伸多项式阶数、扩大作者抓取或重开精度竞赛；先整合现有路径/证据接口。
