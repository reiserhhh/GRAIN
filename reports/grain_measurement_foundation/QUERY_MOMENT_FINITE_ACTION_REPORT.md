# M05：查询矩的有限作用与统计不变反例

2026-09-13。执行包 `query_moment_finite_r1`，
[运行前计划](../../docs/GRAIN_QUERY_MOMENT_FINITE_ACTION_PLAN_V1.md)。

## 1. 实际发现与决议

**来源编译的组内查询协方差，不仅能近似局部导数，也能预测一部分有限输出变化。**
在小幅 .02 干预下，neuroticism / confirmation_seeking 两查询的完整有限变化
RMSE相对二次GRAIN矩下降33.3% / 25.1%。在预定 .10 幅度下分别下降46.7% / 22.2%。
收益主要来自预定的两个查询组内方向，不是所有方向都有改进。

**同时，组内集合与协方差并未完整保存原模型的作用信息。**
将同GRAIN单元内的状态反序，全部均值矩与K不变，但16/16作者×查询的原生输出改变。
这将“新增统计量有用”与“统计量足够重建读者”两个命题直接分开。

决议：保留K为有作用依据的辅助观察量；不替换完整读者，不继续扫rank/alpha或
多项式阶数。M05有限原生作用子项完成，条件/自然材料整合由M04/M14承接。
下一步制作两个可冷运行的局部实例，不继续优化这个8人干预面板。

此处输出单位是冻结Qwen的**数字token logit对比**，不是概率、人格分数或人类效度。
8人均为历史研究已暴露材料；source4模型固定、应用8人不参与本轮拟合。
这是隐藏状态干预，不是自然文字改写。更大幅度的优劣不升级成真实作者变化规律。

## 2. 输入、公式与干预

继承[前一包](QUERY_MOMENT_OPERATOR_REPORT.md)已冻结的两查询模型。
每名作者12条历史评论，得到block16后的BODY状态H和4维GRAIN坐标g。
P为精确同g单元的均值投影，C=I-P，R=CH；不删除原文、选择信息或完整H。

\[
M_d=\frac{p_d(g)^TH}{n},\quad
K_q=\frac{(RV_q)^T(RV_q)}n,\quad
F_q=c_q+\langle B_{q,2},M_2\rangle+\operatorname{tr}(A_qK_q).
\]

Vq及对称Aq来自原source4，rank4与正则化原样保留。三个比较臂为线性g矩、
二次g矩、二次g矩加K。注意：**“二次g矩”仍对H线性**；K才引入H的二次项。

7方向：F01–F04、两来源查询子空间、一个固定随机rank4子空间。
全部方向仅使用当前H/g和来源矩阵；干预D归一至BODY RMS。
每方向ε=.02/.10、正负各一次。每上下文加原值和两种非局部压力，共31次。
8作者×2查询，共496次无梯度suffix17 forward，295.524秒；加载3.369秒。
预算480秒未耗尽，没有为完成矩阵延长预算，也无重训或新反向传播。

两条路径计分完全相同的实际float32补丁。非BODY行、g、查询、位置及旋转/掩码保持。
预测先封存，再运行原生值；应用阶段32次故意读教师文件的尝试均被拒绝。
哈希准备阶段只读教师文件字节，不将其称作完全未接触文件。

对任意D，固定g时新增项的精确有限变化为

\[
\Delta F_K=\frac{2\epsilon}{n}\operatorname{tr}[A(RV)^T(CDV)]
 +\frac{\epsilon^2}{n}\operatorname{tr}[A(CDV)^T(CDV)].
\]

分别报告奇分量(f+−f−)/2及偶分量(f++f−)/2−f0。
实际float32舍入保留在计算中，不把理想双精度输入拿来与另一补丁比较。

## 3. 有限变化结果

下表是所有7方向、正负两侧完整Δ的RMSE。每人总权重相同。
区间为8作者配对bootstrap 2000次的**绝对RMSE下降**，不是相对百分比区间。
这些是固定小面板上的描述性配对区间，不能把448次干预当448名独立被试。

| 查询 | ε | 二次矩RMSE | 加K RMSE | 相对下降 | 下降95%区间 | 改善作者 |
|---|---:|---:|---:|---:|---|---:|
| neuroticism | .02 | .143468 | .095655 | 33.3% | [.009322, .076557] | 5/8 |
| confirmation_seeking | .02 | .126755 | .094992 | 25.1% | [.010439, .045807] | 6/8 |
| neuroticism | .10 | 1.655186 | .882131 | 46.7% | [.729462, .817176] | 8/8 |
| confirmation_seeking | .10 | 1.971717 | 1.533036 | 22.2% | [.410974, .466192] | 8/8 |

完整Δ的MSE等于奇、偶分量各自MSE之和，正负成对时为代数恒等式。
奇/偶分量及F4/within2/random分组的128行全部写入 `evaluation/metrics.csv`。

| 查询 | ε | 奇分量：二次矩→加K RMSE | 偶分量：二次矩→加K RMSE |
|---|---:|---|---|
| neuroticism | .02 | .123351 → .078323 | .073263 → .054913 |
| confirmation_seeking | .02 | .109987 → .080029 | .063005 → .051173 |
| neuroticism | .10 | 1.175136 → .686402 | 1.165631 → .554083 |
| confirmation_seeking | .10 | 1.387962 → 1.118345 | 1.400440 → 1.048571 |

收益定位而非平均数掩盖：within2方向几乎承载全部改进；原F4奇分量改善很小，
部分偶分量变差；随机子空间多项误差也变差。相对RMS相同不等于每个方向有相同
端点敏感度；all7绝对误差由作用更大的within2方向主导。这不是全向量通用重建。

### 切线与有限作用并不等价

原生旧梯度仅在评估阶段作为切线参照。.02时原生切线的奇分量RMSE为
.030225 / .023921，优于加K；.10时为.868064 / 1.140332，点估计反而略差于加K。
不能据后者宣布源模型比真实导数更正确：导数只给局部切线，而本模型奇项同样
对ε线性，大幅时可能因来源系数偏差抵消更高阶误差。原生非线性已显著出现。
偶分量结果说明K具有非零曲率通道，但没有证明其曲率完全等同原生Hessian。
不选一个看起来最漂亮的幅度作为新运行参数。

## 4. 不变统计量的直接反例

反序：对每个精确g单元，反转其H行与物理位置的对应，其他内容不变。
反射：H'=PH−R。二者理论上都保留Md和K；前者更保留完整单元多重集合。
它们是非局部压力对照，相对位移平均为1.002 / 1.370，不能称为微扰。

| 查询 | 压力 | 原生Δ RMS | 原生绝对Δ范围 | 超过重放容差 |
|---|---|---:|---|---:|
| neuroticism | 同单元反序 | .324552 | [.008827, .873056] | 8/8 |
| confirmation_seeking | 同单元反序 | .222605 | [.018459, .608822] | 8/8 |
| neuroticism | 组内残差反射 | .179775 | [.024927, .260114] | 8/8 |
| confirmation_seeking | 组内残差反射 | .082304 | [.027778, .129442] | 8/8 |

反序观察量相对误差最大4.52e−16、显式分数变化最大4.44e−16。
反射因float32舍入，观察量相对误差最大6.75e−9、显式变化最大3.12e−9。
全部16原值重放误差为0；运行前重放容差固定1e−4。

### 对制作方法的数学含义

同单元置换Π满足Πg=g。观察映射O(H)=(Md,Kq)满足O(ΠH)=O(H)。
这是标准的置换不变性，不是新发明的数学定理。
任何只取O作输入的计分器s都满足s(O(ΠH))=s(O(H))。
若原生函数f的两个值相差Δ，由三角不等式必有

\[
\max\{|s(O(H))-f(H)|,\ |s(O(H))-f(\Pi H)|\}\ge |\Delta|/2.
\]

本批最强反序对给出单对最大误差下界.436528 / .304411，单位仍是logit对比。
这是对这对固定输入的下界，不是对自然作者分布的总体泛化误差下界。
反射的舍入版本不拿来宣称任意函数的严格相同输入下界。

**因此不能仅靠换回归头或增加p(g)阶数补回被当前观察映射合并的信息。**
需要保留行与位置/历史的绑定，或定义一个确实允许忽略该差异的测量目标。
本干预不区分位置编码、上下文历史和语义顺序分别贡献多少；不直接命名为人格。
这也不否定GRAIN已有动态/有序表示：反例只针对这里不使用位置的矩算子。

与前一包“保留全部边际却解除主体-命题绑定会丢掉目标”的结果相接：
共同制作原则是**先保留目标所需关系，再选择允许的不变性和压缩方式**。
这为M14的证据接口提供规则，而不是再增加一个无人能解释的因子名称。

## 5. 验收与文件

496次实际原生调用完成；1488行显式分数、1344行有限变化比较、896行奇偶/切线
比较、96行压力比较、16原值重放、128指标行及4压力汇总均保存。
独立公式重新构造496份补丁字节及1488个标量，最大误差1.60e−14。
审计复算全部896奇偶行和128指标行，通过；没有额外重复神经调用。
63项相关测试、4个新Python文件Ruff、两仓库git diff --check通过。

运行根：`results/grain_vector_mechanisms/mechanism_todo_v1/query_moment_finite_r1/`。

- `contract.json`：计划、源码、来源模型、输入、496动作与预算冻结。
- `application/private_predictions.csv`、`private_observations.csv`、`freeze.json`：预测及补丁/矩哈希。
- `native/*.json`、`status.json`：每上下文的实际原生值、累计时间、逐次文件记录。
- `evaluation/private_actions.csv`、`private_components.csv`、`private_stresses.csv`：全部成对比较。
- `evaluation/metrics.csv`、`stress_summary.csv`、`replays.csv`、`summary.json`：汇总及重放。
- `audit.json`：独立数式/字节/汇总复算，非独立研究者或第二次神经重复实验。

关键实现：`grain_research/query_moment_actions.py`；执行与审计为
`scripts/run_grain_query_moment_finite_actions.py`、
`scripts/audit_grain_query_moment_finite_actions.py`。
所有原文/状态继续本地ignored，未发送外部服务，未碰MEPS、问卷拟合、commit或发布。

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
# 新实验需空目录；现有冻结结果不要重写。
OUT=results/grain_vector_mechanisms/mechanism_todo_v1/query_moment_finite_replication
"$PY" scripts/run_grain_query_moment_finite_actions.py prepare --output "$OUT"
"$PY" scripts/run_grain_query_moment_finite_actions.py apply --output "$OUT"
"$PY" scripts/run_grain_query_moment_finite_actions.py native --output "$OUT"
"$PY" scripts/run_grain_query_moment_finite_actions.py analyze --output "$OUT"
"$PY" scripts/audit_grain_query_moment_finite_actions.py --output "$OUT"
```

不为填矩阵自动重跑；命令仅为明确复现入口。
总体goal仍ACTIVE。M14下一包要交付共同B1–B7制作规格下的个人条件响应与当次
状态/过程两个实例，保留证据地址、条件绑定、时钟、来源pilot和误差分账。
K属于可选辅助输入，不强迫两个实例都以它取代已经可用的证据读者。
自然语义未解析项、独立行为效度与独立复建仍分别保留在M01/M13/D01/D02。
