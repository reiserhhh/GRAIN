# M05：从组内查询作用到冷计算观察量

2026-09-13。运行：`query_moment_r1`。本报告是机制TODO的一个子包，
不是14项机制或整个理论已完成。执行前规格：
[计划](../../docs/GRAIN_QUERY_MOMENT_OPERATOR_PLAN_V1.md)。

## 1. 这次增加了什么

**NEXT91的诊断余项，现在有了一个新文本可计算、有标量原函数的观察量。**
当原GRAIN均值及交互矩全部不变时，这个量仍能变化；来源材料上学出的查询子空间，
在另8名历史作者上也能找到原生读者敏感的组内方向。

在两条预定查询的组内方向上，相对二次矩参照的原生局部作用RMSE分别下降
24.8%和22.3%。完整梯度场和作者总体读数没有相应全面改善。
因此保留新观察通道，不替换现有读者，不追加rank/alpha/多项式阶数网格。

这是函数作用与新观察信息的发现；本批原生参照是固定LLM的局部导数，
不是人格预测百分比、真实文本编辑的因果效应或临床分数。

## 2. 构造与公式

同一原文单位得到冻结GRAIN坐标 \(g_t\in R^4\) 与中间状态
\(h_t\in R^{1536}\)。\(P_g\) 为精确同分组的均值投影。
旧统计保留全部，不删除文本、选择信息或原状态：

\[
M_d=T^{-1}p_d(g)^\top H,\quad d=1,2,\qquad R=(I-P_g)H.
\]

来源4名作者的查询梯度组内部分，经固定rank4 SVD得到 \(V_q\)。
仅来源材料参与拟合；当前作者梯度不进入应用接口。新增量为：

\[
K_q=T^{-1}(RV_q)^\top(RV_q),\qquad
F_q(H,g)=\langle B_{q,2},M_2\rangle+\operatorname{tr}(A_qK_q)+c_q.
\]

\(A_q\) 是允许正负的对称4×4矩阵，\(c_q\) 由来源标量锚定。
导数匹配继承 [Czarnecki等，2017，Sobolev Training](https://proceedings.neurips.cc/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html)
使用函数导数来学习函数的思想；此处采用显式可积矩函数和正规方程，
不是对其神经训练系统的复现，也不将标准协方差/投影代数称作新定理。

固定 \(g\) 时，新增项具有精确导数：

\[
\nabla_H F_{\mathrm{within}}=2RV_qA_qV_q^\top/T.
\]

这解决的是“任意梯度近似未必有一个对应分数”的接口问题。
对 \(D_R=(I-P_g)D\)，其有限变化也有精确表达：

\[
\Delta F_{\mathrm{within}}
=2\epsilon\,\operatorname{tr}[A_q(RV_q)^\top D_RV_q]/T
+\epsilon^2\operatorname{tr}[A_q(D_RV_q)^\top D_RV_q]/T.
\]

原生模型侧本批仅评价 \(\langle J_q,D\rangle\)，未新跑有限suffix扰动。
改变真实文字还会改变token数、GRAIN坐标与组成员，不能直接套固定 \(g\) 导数。

### 数学边界

- 若 \(P_gD=0\)，所有仅依赖精确 \(g\) 的一阶状态矩都对D不敏感；
  不只是当前二次多项式。因此继续提高 \(p(g)\) 阶数不能解决这类遗漏。
- \(V_q\to V_qQ,\ A_q\to Q^\top A_qQ\) 对正交Q不改变分数或作用场。
  子空间坐标轴本身不是可直接命名的人格维度。
- \(H=P_gH+R\) 与 \(H'=P_gH-R\) 产生相同旧矩及K。
  新函数不识别这种组内翻转；协方差仍是压缩量，不是完整状态表示。
- 精确g全部唯一时R=0，新量退化；当前重复g来自同slice映射。
  更换切片/连续分组规则需重审此对象，不能默默替换为近邻聚类。

## 3. 材料与运行边界

4名来源作者构造、8名来源不重叠作者评价，每人12条历史评论；
均是已经暴露的研究材料，不是新锁箱。查询固定为neuroticism和
confirmation_seeking。每个作者/查询比较线性矩、完整二次矩、二次矩加K三臂；
4个旧GRAIN方向、2个来源查询子空间方向、1个随机rank4方向全部保留。

rank4、alpha=1e-6×Gram均迹、seed91351只设一次；作者总拟合权重相同。
模型先冻结，再应用，随后打开评价梯度。504行预测、逐作者位置和全部失败参照均保存。
数值应用不反序列化教师目标；输入完整性检查会读取目标文件字节计算hash，
不将其误称为文件级物理隔离。原文冷运行另有独立进程访问约束。

## 4. 结果

以下“原生作用”是隐藏状态局部导数；within2为两条来源查询子空间方向，
每查询16个方向读数、按8名作者聚类。

| 查询 | 二次矩within2 RMSE | 加K RMSE | 相对下降 | signed projection |
|---|---:|---:|---:|---:|
| neuroticism | 8.966217 | 6.741409 | 24.8% | .816161 |
| confirmation_seeking | 8.467365 | 6.577834 | 22.3% | .392461 |

signed projection为 \(\sum\widehat d\,d/\sum d^2\)，不是Pearson r，
不是“恢复81.6%人格”。旧矩在这些方向预测为0是代数结果。
两个查询各5/8作者改善；999次作者配对bootstrap的RMSE降幅区间为
[-2.413,6.178]与[-.735,3.808]。点估计可支持继续使用该候选，
不据此宣布跨作者普遍改善。

失败也定位了它的作用范围：

| 查询 | 线性总体分数r | 二次矩总体分数r | 加K总体分数r | 加K全场NMSE |
|---|---:|---:|---:|---:|
| neuroticism | .801468 | .279991 | .285127 | 1.001956 |
| confirmation_seeking | .641876 | .310622 | .318558 | .996029 |

这些r都以原生模型分数为参照，不是人类量表。完整场NMSE约1；
随机组内方向RMSE略变差，旧4方向也无普遍收益。
因此“特定方向新增可见作用”和“完整读者重建”是两个不同的结果。

### 冷运行与外部联系

原文重新分词、四片词法计分、token地址分配、提取17块中间状态，随后只运行矩算子。
一组既有作者原文加一组3条公开新写文本，共2次前缀调用；
执行2.141秒、加载5.285秒。与历史原文的H/g差异均为0。
2次故意读取教师缓存被拒绝，无suffix/logit/生成/反向调用。
仍依赖冻结Qwen前17块及下一QKV提取，归为D1；不是D0或不含神经权重。

外部连接最初因原始ID/匿名ID命名空间不同得到0匹配，
现已沿用既有stable_key规则修正，不改任何模型/分数。
最终来源2人、评价4人有Big5，共6/12；全5trait×2query×3arm结果全部保留。
来源n=2不报r，评价n=4仅接口描述。神经质查询对应Neuroticism的r为
线性−.300、二次−.533、加K−.461；不据其它极端小样本相关改名或翻方向。
confirmation_seeking专属外部量表不可用。较大历史同材料结果另见
[旧报告](../grain_vector_mechanisms/SAME_MATERIAL_INTERPRETATION_REPORT_V1.md)，不混称本次复验。

## 5. 产物与验收

运行根目录：`results/grain_vector_mechanisms/mechanism_todo_v1/query_moment_r1/`。

- `contract.json`、`models/*`：来源/公式冻结及两查询矩阵。
- `application/private_observations.csv`、`private_predictions.csv`：当前输入观察与盲应用。
- `evaluation/summary.csv`、`field_summary.csv`、`paired_comparisons.csv`：全部比较。
- `evaluation/private_author_positions.csv`：来源均值/标准差参照的位置，来源n=4。
- `evaluation/external_links.csv`、`external_join.json`：不参与选择的全外部联系。
- `cold/*`：原文独立进程、坐标、分数和访问记录；私有材料保持本地ignored。
- `audit.json`：独立算术最大误差3.32e-13、原文复算H/g一致、6名标签连接。

旧矩零变化最大误差2.02e-14；候选解析导数/有限差分误差1.36e-11。
原模型、应用结果和冷脚本hash保持不变。ID修正只改变外部审查层。
62项相关测试通过，包括坐标旋转不变性与组内翻转不可识别反例；
本批6个Python文件Ruff通过，仓库git diff --check通过。未重跑全部历史测试。

运行入口（从GRAIN repo；新的prepare需要空输出目录）：

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
OUT=results/grain_vector_mechanisms/mechanism_todo_v1/query_moment_r1
"$PY" scripts/run_grain_query_moment_operator.py prepare --output "$OUT"
"$PY" scripts/run_grain_query_moment_operator.py train --output "$OUT"
"$PY" scripts/run_grain_query_moment_operator.py apply --output "$OUT"
"$PY" scripts/run_grain_query_moment_operator.py evaluate --output "$OUT"
"$PY" scripts/score_grain_query_moments.py prepare --output "$OUT"
"$PY" scripts/score_grain_query_moments.py run --output "$OUT" --device mps
"$PY" scripts/audit_grain_query_moments.py --output "$OUT"
```

## 6. 对TODO的实际推进

M05“诊断转可计算观察量”子包完成，M07新增明确D1冷输入实例。
M05完整的有限原生作用及M04新条件函数整合仍保留，不能只靠梯度比较勾掉父项。
下一包转向M04/M08/M09：把可观察证据接入个人条件响应及状态/互动时间函数。
M11/M12并行承接来源协方差失配；不继续调这个12人面板以获得更漂亮的相关。
总体goal ACTIVE，未发布、未commit/push、未用MEPS拟合、无本包后台任务。
