# GRAIN Measurement Discovery Theory V1

状态（2026-09-15验收及二次纠正）：**部分有限结果保留，构造闭环未验收，完整报告计划未完成**。GP03的B1同对象oracle/soft/hard比较有效，原审计对B1的错接指控撤回；B2/M多臂对照仍缺。GP04模型外诊断、GP06冷制作/运输、PV01非平凡作用保留的完成说法不能沿用。证据和修复见[全面验收](../../project%20persona/reports/gptpro_seminar_20260915/COMPREHENSIVE_ACCEPTANCE_AUDIT.md)，新主线为[全源理论补齐V2](../../project%20persona/docs/GRAIN_GPTPRO_FULL_COVERAGE_RESEARCH_PLAN_V2.md)。新科研任务尚未执行。

## 核心结论

本阶段把开放文本测量拆成四层：观察设计、文本/读者通道、目标函数、解释范围。GP01 的精确世界证明：单点或相邻边际相同不能推出完整历史等价；机会与选择只观察到乘积时不能唯一分解。GP02 的有限比较显示，标准二阶条件频率在本世界优于二次回归和 rank-2 表压缩，因而“标准方法已足够”是有效结论。GP03 将数字状态经过无状态固定语法与 source-only TF-IDF/Ridge reader，明确计算了 oracle、text-soft 和 text-hard 的通道损失。

GP04 的固定三轴条件设计能够区分预先规定的竞争响应族；未知混合族只能得到族外偏离提示，后验最高不是机制真值。GP05 对现有 PANDORA/Essays 冻结缓存完成接口审计：作者、单位和 source/held 可复用，但文档顺序不是可信时间，条件也不是完整机会或随机干预。GP06 用相同 bundle/scoring 入口冷制作两个不同对象：T1 是两条件响应差，T2 是给定观察前缀的下一观察概率；二者不互换，也不把它们命名为人格。

PA01 的主动区分参照只在固定模拟族和有限预算内有小幅、预算依赖的 holdout 优势。PV01 的局部 Jacobian 加权作用对 native donor 有有限 RMSE 改善，但中心差分受 MPS/FP32 精度限制，故保留为 reader 机制线索，不把它升级为 GRAIN 通用算子。

## 形式化对象

生成过程和测量过程分开：

\[
O_t\to S_t\to X_t\to R_t,\qquad
R_{1:t}\xrightarrow{\mathcal M}Y_t.
\]

\(O\) 是表达机会，\(S\) 是选择，\(X\) 是文本，\(R\) 是有限 reader 的观察量，\(Y\) 是明确目标的测量值。缺失机会不能补成零；作者ID用于分组而非输入特征；source/validation/application必须按作者隔离。

信息保留检验是：

\[
\Phi(H_1)=\Phi(H_2)\Longrightarrow T(H_1,a)=T(H_2,a).
\]

若反例成立，结论是该表示对目标不充分，而不是所有文本信息都无效。有限预测对象写为：

\[
\mathcal P_{L,\mathcal A}(H)=\{P(Y_{1:L}\mid H,do(a_{1:L})):a_{1:L}\in\mathcal A\}.
\]

只有在模拟中明确声明 do；自然缓存只使用记录条件概率。

## 两个冷制作实例

T1：

\[
\theta_{response}=\frac{\bar p(R\mid z_1=+1)-\bar p(R\mid z_1=-1)}{2}.
\]

它是固定两条件下的响应差，单位为 reader probability contrast。支持范围要求两个条件格都被观察；缺一格返回 null。

T2：

\[
P(Y_{next}=1\mid Y_{t-2},Y_{t-1})=p_{Y_{t-2},Y_{t-1}}.
\]

它是 prefix-dependent 的下一观察目标，不是终身稳定特质。source table 固定后，application 只用已揭示前缀更新；未来值不能参与早期预测。

## 方法边界

本阶段发现的贡献候选不是新的 PSR、ADO、因果识别定理或心理潜变量。可复用的组合是：target contract + observation ledger + source-frozen reader + explicit support/bounds + cold replay + loss-location ledger。任何人格或临床命名都必须再通过独立心理/行为效标与同人情境数据。

现有自然接口的 `BLOCKED_INPUT` 不削弱 GP01-GP04 的有限机制结果；它只限制自然动态与干预效应的解释。下一阶段若要推进自然材料，最低新增信息是可验证的时间/事件边界、条件机会记录和不依赖待评作者真值的重复观察设计。
