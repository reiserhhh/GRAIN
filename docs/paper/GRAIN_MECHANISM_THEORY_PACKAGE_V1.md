# GRAIN 机制轮理论包 V1：继承定理、本地发现、反例与适用范围

2026-09-14 补项：[Codex 接手报告](../../reports/grain_measurement_foundation/CODEX_EXECUTABLE_GAP_CLOSEOUT_20260914.md) 增加条件分配二阶矩 `p(1-p)[(dm)^2+dSd^T]`、来源误分混合失败反例、元数据条件接口与自身/第三人计分区分。自然 generic binding 不等于自身计分；已回复上下文不等于真实曝光机会。新结果清单独立保存，旧冻结清单不覆写。

2026-09-13。任务 `CLAUDE-TODO-CLOSURE-20260913-01` 的D03产物。
本包整理2026-09-13机制轮（M01–M14）的理论内容，接在三语理论稿V1
（`GRAIN_THEORY_MANUSCRIPT_{EN,JA,ZH}_V1.md`，2026-09-03）之后，不重写正文。
三语增补稿：`GRAIN_MECHANISM_SUPPLEMENT_EN_V1.md`、`_JA_V1.md`、`_ZH_V1.md`（段落一一对应）。
数值出处：`GRAIN_MECHANISM_EVIDENCE_PROVENANCE.md`；每个数字都来自本次独立审计核过的产物文件。
冻结仅指可复核的内容清单（`claude_todo_closure_20260913/d03_theory_package/manifest.json`）；不commit/tag/push。

## 0. 地位

理论稿V1给出情境条件化多层模型、开发流程与八条可证伪命题。本机制轮回答的是
V1第4/8节留下的制作问题：证据单位如何从原文形成并累积、条件响应如何与个人水平同函数
计分、观察如何按目标预先选择、分数如何取得参照与误差、解释链如何可追溯、同一方法如何
制作两个局部工具。结果分四类记录，四类不得互换：**继承定理**（标准方法及其本地对应）、
**本地发现**（在声明的世界或公开编写材料上实测的规律）、**明确反例**（保留不救援的失败）、
**适用范围**（哪些对象不能由此推出）。

## 1. 继承定理与本地对应

| 继承方法 | 出处 | 本地变量/假设对应 | 用于 |
|---|---|---|---|
| 高斯条件化/线性最小均方读出 `ŝ=t'[(I-GH)m+Gy]` | Rasmussen & Williams 2006, ch.2 | θ=个人参数与状态；H=事件/机会设计；R=读者噪声；m,Σ来自来源 | M04/M08/M14 目标核 |
| REML 与混合模型方差分量 | Bates et al. 2015 (lme4) | 试点作者共同机会表下的m/V/p/λ/R；不用评价真值 | M11 独立试点 |
| 连续时间OU/精确转移 `Φ=e^{AΔ}` | Oravecz et al. 2009; Driver et al. 2017 | 当次状态s(t)、不规则时间、窗口梯形权 | M08 |
| 工具变量/控制函数 | Wooldridge 2015 | r=记录的伙伴输入分配；u=实际输入；v=共同状态线索 | M09 |
| 信念修订与世界更新的区分 | Katsuno & Mendelzon 1992 | revise=更正旧主张；new event=世界变化 | M02 对话双账 |
| 来源半环注记 | Green, Karvounarakis & Tannen 2007 | 版本/撤回/引用摘要的依赖图；不作可减概率 | M02 |
| 平方根/正交化估计 | Morf & Kailath 1975 | QR创新形式修复近重复报告的数值相消 | M02 修复伴随 |
| 有界缺失下的均值识别区间 | Manski 类部分识别 | `[(S-m)/(N+m),(S+m)/(N+m)]`，未读值∈[-1,1] | M01 命题回应 |
| Schur补信息增益 | 标准 | 固定目标下读取哪份报告；副本增益0 | M10 |
| 导数匹配学习函数 | Czarnecki et al. 2017 | 组内查询协方差K的可积标量原函数 | M05 |
| 依存句法/事实性/对话依赖 | UD; Sauri & Pustejovsky 2012; Bunt et al. 2020 | frames通道、显式引用规则、功能依赖候选 | M01/M09 |
| 测量等价≠预测等价 | Millsap 1995 | 观察单位一致与相关不错不同 | M12 |

这些方法不署名为GRAIN新定理；本地贡献是把它们接到文本证据、来源学习、目标核与解释规则上，
并用有限见证检验其失效条件。

## 2. 本地发现（LF）

| 编号 | 发现 | 实测 | 产物 |
|---|---|---|---|
| LF1 | 具体命题绑定比内容相近保留更多测量对象信息，但漏读改变分母 | 留出措辞 MSE 内容.045369 / 绑定.018183 / 完整事件.011838；r .403/.895/.896；383条件轮廓平均偏移.035705 | `proposition_response_r1/completion/summary.json`, `audit/selection_bias.csv` |
| LF2 | 同一句话可同时更新旧主张与实施新的言语行为，二者必须分账 | 14对话72操作/72分数正确；忽略版本/末极性/版本账MAE .958333/.875/0 | `dialogue_claim_ledger_r1/summary.json` |
| LF3 | 新事件、复述、副本、更正、撤回、摘要需要不同更新；相同当前分数不决定更新 | 水平MSE .026027/.035029/.044062，覆盖.9688/.8750/.7383；四记录见证下界.0762939 | `evidence_history_r1/evaluate/comparison.csv`, `update_witnesses.json` |
| LF4 | 尾部选择带条件增量；高相关可掩盖参数错误 | 未校正r .917/.930 vs 校正后.771/.796；参数RMSE 1.47/1.53 vs .633/.640 | `choice_design_r1/choice_metrics.csv` |
| LF5 | 可共享的是来源学习的响应协方差，不是答案；平滑世界随机斜率已够 | 交叉世界隐藏差分r .8414/MSE .044659 vs .156086；平滑世界.014255 vs .014327 | `response_sharing_r1/evaluate/comparison.csv` |
| LF6 | 组内查询协方差是可冷计算的新观察量并预测有限变化 | .02干预RMSE −33.3%/−25.1%；.10下−46.7%/−22.2% | `query_moment_finite_r1/contract.json`, `evaluation/` |
| LF7 | 同事件关系（乘积）而非更多分数拼接恢复联合条件读数 | 配对r .3578，错配.0408，完美事件参照.3594 | `joint_conditional_observation_r1/evaluate/metrics_by_seed.csv` |
| LF8 | 查询条件化的证据运算主要落在问题区状态 | P分账73.5%–80.5%；他人引用保留94.7% | `shared_query_actions_r1/component_actions.csv` |
| LF9 | 查询作用可编进文本区投影系数；状态重建与编译投影等价 | A/B差1.55e-15；4层程序C分账98.6%但幅度.004379 vs .288880 | `query_program_actions_r1/independent_audit.json` |
| LF10 | 同一证据按目标核给出不同读数；窗口/当次分开 | RMSE .30905→.14319；窗口覆盖.69271→.96875 | `temporal_measurement_r1/metrics.csv` |
| LF11 | 来源学习设计优于覆盖优先；补采修复隐藏方向 | −25.9%/−30.0%；Var(d)=.034133；补采后.036596→.010940 | `new_condition_design_r1/evaluate/` |
| LF12 | 独立试点可学人口矩/恢复率/噪声；条件化计分器混合a/b | 覆盖.333/.557→.969/.964；表观恢复率.6748→.2960 | `pilot_transport_r2/gaussian/`, `text/` |
| LF13 | 计算解释链可完成；时限编辑比请求极更改变查询 | +.076500（12/12）vs +.013705（8/12） | `explanation_chain_r1/summary.json` |
| LF14 | 同一制作方法产出两个可冷运行工具；当前表达误差与排序同时改善 | 当前表达RMSE .2446→.1189、r .5575→.6145；3840独立复算最大差3.77e-11 | `local_instrument_r1/evaluation/metrics.csv`, `audit.json` |
| LF15 | 许可语料含大量自然前置话轮；极性回答对上有限版本账可计分约44% | 433,561跨作者对；400对极性回复中175可计分 [.389,.486]；求助目标0/400 | `all_plan_logic_v1/natural_antecedent_scan_r1/`, `natural_antecedent_coverage_r3_polar400/` |
| LF16 | 条件可由来源学习从文本读取：水平不损，条件半差r损失约1/4 | oracle .2052 → LDA软分配 .153（hard .142）；level不变 | `all_plan_logic_v1/open_condition_assignment_r2_lda/` |
| LF17 | 被提问的机会份额是作者分半稳定的可观测坐标 | 7,957作者：分半Pearson .756 [.727,.780]，去发帖量后.731 | `all_plan_logic_v1/natural_opportunity_ledger_r1/` |

## 3. 明确反例（CE）——保留，不救援

| 编号 | 反例 | 实测 | 产物 |
|---|---|---|---|
| CE1 | 内容分数相同而证据身份目标相反 | .993314相同，目标±.165707 | `evidence_identity_r1/content_collision.json` |
| CE2 | 抽象身份元描述NLI失效，具体事实正常 | 抽象蕴含.000406 vs 具体.998065；留出六通道r .166 vs TF-IDF .619 | `evidence_identity_r1/observer_sanity.csv`, `composition_metrics.csv` |
| CE3 | 拒绝命题提问的固定观察器高置信错误 | 3/24 top匹配 | `interaction_response_r1/nli/public_nli.csv` |
| CE4 | 本地Qwen公开探针未达基本识别 | EN 4/6、JA 2/6；24名MEPS输出仅上下文依赖证据 | `interaction_response_r1/natural/` |
| CE5 | 弱分配下IV失真、排除限制违反仍偏差 | c RMSE 5.70895；+.296 | `interaction_response_r1/math/coefficient_predictions.csv` |
| CE6 | 同单元反序保持全部矩与K却改变原生输出 | 16/16 | `query_moment_finite_r1/evaluation/` |
| CE7 | 相同当前分数/协方差的两历史在同一更正后不同 | (.390625,−.390625) vs (0,0) | `evidence_history_r1/evaluate/update_witnesses.json` |
| CE8 | 只在新条件出现的响应方向对旧数据完全隐藏 | 差分误差恰增.034133 | `new_condition_design_r1/evaluate/identifiability_identity.json` |
| CE9 | 收缩降低误差却损失排序与覆盖 | 社交半差r .4898→.2052，覆盖.7656 | `local_instrument_r1/evaluation/metrics.csv` |
| CE10 | 第三人称过去报告漏读使本人分母偏移 | 4/16留出未读；最大偏移.152778 | `proposition_response_r1/audit/selection_bias.csv` |
| CE11 | 自然材料上声明目标不可计算 | 12自然句0/48目标可计算；273事件0合格 | `claude_todo_closure_20260913/m01_m09_dialogue_input_entry/natural_coverage/coverage.json`, `local_instrument_r1/natural/summary.json` |
| CE12 | 原增量更新函数数值相消 | 31/282失败，最大7.88e-7；修复后5.86e-12 | `evidence_history_r1/evaluate/square_root_repair_freeze.json` |
| CE13 | 高维各向同性后验欠置信，使软分配退化为跨单元平均 | |π−.5|≈.054；条件r −.043 | `all_plan_logic_v1/open_condition_assignment_r1/` |
| CE14 | 高置信的从句级系统误分不能由后验膨胀表示（事件独立或从句相关均无效） | 1/8留出从句π=.185；r3=r2 | `all_plan_logic_v1/open_condition_assignment_r3_lda_clause/` |
| CE15 | 自然问句几乎不带条件从句；声明求助目标自然出现率为0 | 条件从句57/400；目标0/400 | `natural_antecedent_coverage_r3_polar400/coverage.json` |

## 4. 适用范围（S）

- S1 模拟参数组是声明世界的参数恢复，不是真人人格相关或行为效度。
- S2 公开编写句库（128句、14对话、72文本等）是机制材料；同一句库多次分析是分析性再现。
- S3 读者依赖：固定 `cross-encoder/nli-deberta-v3-base@6c749ce…` 与固定Qwen前缀；换读者须重校准。
- S4 对话规则只在显式引用短语与声明的两方角色内工作；开放省略/情态/共指未解决（M01/M09未满足清单）。
- S5 工作区间是来源构造+工作模型的方差，不是心理95%CI；测量不变性未被共同单位证明。
- S6 MEPS/相谈只验证，不拟合、不选参、不外发；24名输出禁作行为解释。
- S7 同agent的数值复核与跨agent复建都不是盲第三方复建（D02边界）。
- S8 心理/行为含义需要独立编码与效标（D01），任何高分不自动命名。
- S9 自然前置对只覆盖父评论同为语料作者的线程；极性绑定是显式规则扩展，省略含否定者另记，不是通用语法。

## 5. 两个示范与第三次构造

M14两实例（社交条件轮廓、任务当次过程）经手册§4命令在新目录重建，与冻结r1逐值一致
（bundle/指标/冷分数差0，审计PASS）。只依据手册与库函数的构造尝试复现了社交实例的
来源矩（差0），并为“任务完成短/长”制作了第三个两条件对象，冷入口22次实际NLI对
与缓存边际计分差1.44e-12。构造日志记录15处需查实现、需询问或自补规则之处，
已回填手册§15。这是带开发上下文的跨agent复建，不替代Study D的独立执行者。

## 6. 与理论稿V1的接合

- §3/§4：LF3/LF5/LF10/LF12把"测量对象先于模型"落实为来源学习的目标核；增补稿S2–S3。
- §5表1：不新增真人效标行；增补稿以独立表列机制轮结果，并标明S1–S2。
- §6命题：增补三条可证伪命题（P9证据身份不可由内容相似度恢复；P10当前分数与协方差不决定更新；
  P11观察设计不能替代目标响应的可识别性）。
- §8限制：第六点“方法论闭环≠心理完备”保持；增加D01/D02依赖的明确表述。
- 必须在三语中保持相同的主张：V1九条不变；新增（i）模拟参数≠人类患病率/效度，
  （ii）显式规则≠通用语法，（iii）跨agent复建≠盲复建。

## 7. 冻结清单

清单由 `scripts/build_grain_theory_package_manifest.py` 生成，包含本包、三语增补、出处表、
手册、TODO、审计与闭环报告、21份机制报告的sha256。未commit、未tag、未push、未开锁箱。
