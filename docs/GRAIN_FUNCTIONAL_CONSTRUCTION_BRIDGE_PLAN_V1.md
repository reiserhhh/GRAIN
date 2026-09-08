# GRAIN 功能构造连接实验 V1

2026-09-08。状态：**IMPLEMENTED_NOT_RUN**。本次研讨选定的唯一下一研究包。
替代S27的默认续跑位置，不撤销S0-S26成果，不启动旧长队列。
研讨记录：[主线重定向](../reports/grain_vector_mechanisms/MAINLINE_REORIENTATION_SEMINAR_20260908.md)。

## 1. 研究问题与新增价值

同一作者的同一公共文本编辑，为什么使不同特质查询一起改变、反向改变，或
只改变其中一个？已有局部功能方向，能否预测这种**完整自然响应的联合结构**？

这不是再问“g^T d的r能否更高”。S1已经比较单查询局部与自然效应，S21已经
比较跨查询读出。本包新增的是冻结参照参数后，对留出作者的跨查询联合变化
给出事前可计算预测，并将不匹配定位到具体环节。

不要求把全部Transformer化约成一个公式。允许固定编码器作为观测器；每个
特质可估计自己的参数。程序可复用与所有查询共用相同参数是两种主张。

## 2. 固定资产与边界

主包只用`multiquery_sensitivity_s1_r1/`，24作者、2视图、原生T/F、N、确认
查询和公共E1/E2；8作者的联合编辑为次级组合面板。根位于gitignored的
`results/grain_vector_mechanisms/`。不重新选择评论、不读取量表来选择作者。

复用：
- `scripts/analyze_grain_multiquery_sensitivity.py::load_panel/response_rows`。
- `scripts/run_grain_multiquery_sensitivity.py`的已冻结读出和字段定义。
- `scripts/run_grain_cross_query_functional.py`的S21包，只作独立平行分析。

S1是block16末查询状态、自然删文位移、端点梯度；S21是layer17 head pairing、
不同查询与积分梯度。不能因同为1536维而拼成一张表，也不混合两个观察站点。
S1没有存储联合编辑patch时，该项标缺失，不虚构M_joint。

S9已完成旧CUDA/BF16与新MPS/FP32教师的分账，保留其报告，不重新推理。
S4的T/F学生AUC不标成新功能协向量的AUC；数字Feeling也不标成原生T/F。

## 3. 精确对象

对作者u、视图v、查询q、公共编辑j：

    d = h(T_j x,q)-h(x,q)
    Delta = S_q(T_j x)-S_q(x)          完整自然编辑响应
    M = S_xq(h+d)-S_xq(h)             既有单位置patch响应
    l = g_uq^T d                     既有本人端点梯度近似，诊断用途
    a = g_ref,q^T d                  参照作者估计后冻结的可运输近似

保留精确分账：

    Delta = a + (l-a) + (M-l) + (Delta-M).

四项依次为冻结近似、参照运输误差、局部一阶余项、未由该站点patch覆盖的作用。
最后一项不是人格噪声；它可包含其它位置、上下文与长度变化等作用。
S1只有端点梯度，不将其称作积分梯度。

对同一作者/编辑的两个查询，令b_rs=g_ref,r^T d_s：

    phi_read = [(b_qq-b_rq)+(b_qr-b_rr)]/2
    phi_evidence = [(b_qq-b_qr)+(b_rq-b_rr)]/2
    phi_read + phi_evidence = b_qq-b_rr.

四格交互I=b_qq-b_qr-b_rq+b_rr单报，不能再次加进上式。
交叉读出是同一声明坐标下的功能模型反事实，并非新增实际跨查询patch。
原始查询评分单位各自保留；跨查询量级比较附参照集冻结的单位标准化结果。

## 4. 联合响应关系与可检验限制

定义epsilon_q=Delta_q-a_q，精确协方差是：

    Cov(Delta_q,Delta_r)
      = g_ref,q^T Cov(d_q,d_r) g_ref,r
      + Cov(a_q,epsilon_r) + Cov(epsilon_q,a_r)
      + Cov(epsilon_q,epsilon_r).

共同Sigma_D只是受限模型；不同q的位移一般有不同交叉协方差。
不逐例计算M/Delta，不用未知“覆盖率”乘K，也不假定三个余项为零。
向量分账、跨作者协方差和人的量表相关分别报告。

两种预测强度分开：

1. **条件于留出输入**：冻结g后，读取留出d产生a；不读留出Delta或梯度参与
   拟合。由a预测自然响应及其联合结构。这仍需教师prefix产生d，不是LLM-free。
2. **总体结构运输**：连参照集交叉协方差一起冻结，预测留出作者的响应协方差。
   人群输入分布变化与功能运输误差分开，不把第一种称作第二种已成功。

上述四项在看见留出Delta后可作描述性分账，但这种事后恒等式本身不是预测成功。

## 5. 有限执行步骤

### A. 资产绑定与冻结

新入口建议` scripts/analyze_grain_functional_construction_bridge.py `，不改旧runner。
读取manifest/数组哈希、query/edit/site、完整作者单位及已有读出。固定按作者
hash划分12参照/12评估；所有视图和编辑跟随作者，盐与名单在计算响应统计前保存。
这些作者都参与过开发；这是包内留作者重用分析，不是新招募独立确认。
主分析E1/E2。联合面板保持原8人，样本不足时只作描述，不挑有利样本补齐。

实现冻结：SHA-256盐为`grain-functional-construction-bridge-v1-20260908`，
按hash排序取12位参照作者；固定1000次整作者bootstrap，随机种子20260908。
区间条件于一次冻结参照拟合，不包括重新估计参照参数的不确定性。
标准化单位为参照作者全部view/edit自然响应的总体标准差；同时保留原始单位。

参照g按作者等权平均，避免多视图作者权重更多。主模型无新参数搜索。
原始分数和原始梯度全部保留；任何单位缩放只由参照集估计。

### B. 竞争模型与分账

对相同留出输入比较：零响应、参照每query/edit均值、共同位移+专属读出、
专属位移+共同读出、专属位移+专属读出。共同量用预先声明的等权平均定义，
原始与参照标准化坐标分别报告，不声称这是唯一可识别分解。
不把对测试作者实测g的拟合视作冻结运输模型。

逐query报告a对M、a对Delta，以及M对Delta；再报告完整响应协方差矩阵的
条件输入预测与冻结总体预测、两阶分配、四项协方差分账。按作者bootstrap，
先在相同view/edit条件内看跨作者结构，合并时保留条件均值产生的额外关系。
不能把E1平均更强误认成所有人拥有同一种响应配置。

### C. Who的连接

Who首先定义为作者的多查询响应配置，不等同于g_u-g_ref。
仅在已有GRAIN参照几何可按匿名ID绑定、且来自与响应文本不重叠的材料时，
附一次固定邻域对配置的预测；对照参照全局均值和可用的条件邻域。
没有独立几何则列为明确的下一连接，不新读原文、不生成新embedding，
不按当前响应重聚类后宣称发现identity共享。失败不删除choice通道。

## 6. 预算、停止与分支

**本包0次神经调用、0次新梯度、0次量表连接**；本地CPU、最多2线程。
先用标量投影计算交叉协方差，避免无意义构造1536平方矩阵；预估十分钟内，
20分钟硬上限。遇缺字段保存可用分析与缺口，不自动补神经采集。
代码测试覆盖作者隔离、实际query方向、两阶分配、协方差恒等式和近零分母。

不设r>.8或AUC>.66生死线，不要求本包提高问卷成绩。以效应、误差、简单对照
及区间共同选择一个后继；差别不清楚就保留两个局部解释，不无限加网格。

| 主要结果模式 | 允许的下一研究问题 | 不允许的自动续跑 |
|---|---|---|
| 专属读出解释自然响应的查询差异 | 新特质同程序运输，或有限persona写读关系，两者选一 | 多层神经质精度搜索 |
| 查询条件化位移带来明确新增作用 | 将该关系坐标作为轻量学生目标，而非只拟合总分 | 重复E5均值/任意PCA网格 |
| a已解释M，但Delta-M有系统结构 | 定义多位置/完整证据机制的小采集包 | 不经目标绑定就恢复S27的C追踪 |
| Who配置被独立GRAIN几何组织 | 发展群体参照下的共享/专属函数 | 直接命名人格或用量表分组回证 |

这些模式可并存，不拿RMS当独立方差比例。新采集的作者角色、观察器、目标与
资源上限需另写协议；本计划不预授权440F或更大队列。

## 7. 交付与理论位置

输出目录`results/grain_vector_mechanisms/functional_construction_bridge_v1/`：
`manifest.json`、`split_manifest.json`、`response_accounting.csv`、
`cross_query_predictions.csv`、`covariance_accounting.csv`、`mechanism_decision.json`。
逐人内容全部gitignored；公开报告
`reports/grain_vector_mechanisms/FUNCTIONAL_CONSTRUCTION_BRIDGE_REPORT.md`。

本包应交付“哪些查询差异由什么运算解释、哪些需要其它状态”的机制地图，
不是新的accuracy leaderboard。它完成时只关闭这条有限问题，不宣布GRAIN
所有理论闭环。理论v1整合须有对象/假设/构造/误差、同程序多查询实例、
真实自然响应连接、群体参照与心理解释接口、成本和开放域的清楚账本。

标准数学依据：两玩家对称分配、双线性形式与协方差分解；机制简化的解释
框架见[Geiger et al., JMLR2025](https://jmlr.org/papers/v26/23-0058.html)。
本包不因预测相关或分账恒等式成立就宣称完整因果抽象成立。
