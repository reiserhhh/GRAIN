# Claude独立验收：M01–M14与D01–D03的完成情况

2026-09-13。任务 `CLAUDE-TODO-CLOSURE-20260913-01`，第一阶段产物。
本文件在任何算法/实验补齐之前写成；第二阶段的实际补齐另见
[CLAUDE_TODO_CLOSURE_REPORT.md](CLAUDE_TODO_CLOSURE_REPORT.md)。
审查者：Claude（Fable 5.1，独立CLI后台任务）。Codex所称“12/14有限验收”按待审声明处理。

## 0. 审查方法与实际执行

| 项目 | 本次实际做了什么 | 产物 |
|---|---|---|
| 要求来源 | 逐项核对 `GRAIN_MECHANISM_TODO_20260913.md` 机制卡的“状态/最小实验/验收输出”，并回读其出处：V3 §5 B1–B7、Study B/C/D、§8完成规格；V5 §1.2/§6/§7；V1 WP10/WP11；V2 F01–F08表 | 本文各节“原要求”行 |
| 报告 | 通读对话双账、D01盘点、命题回应、证据身份、事件历史、互动响应、解释链、局部制作及执行修正8份；其余12份读结论/材料/边界节 | — |
| 测试 | 用 `project persona/.venv` Python 3.14.3 实际运行27个机制相关测试文件 | **250通过，0失败**（`audit_verification/tests/`） |
| 冻结 | 新脚本 `scripts/verify_grain_mechanism_freezes.py` 只读复算23个结果目录中全部freeze/manifest/contract记录的hash | 研究文件全部一致，仅 `pilot_transport_r1`（被r2取代的预跑）2项旧代码hash不同（`audit_verification/freeze_verification_r1/`） |
| D01入口 | 逐一重算 `interpretation_inventory_r1/source_inventory.json` 的9个文件hash；不读取私人数值 | 9/9一致（`audit_verification/d01_inventory_hash_check.json`） |
| 数值 | 对各报告标题数字在结果目录中定位文件（grep/读JSON） | 全部找到对应文件；差异仅为报告四舍五入 |
| 未做 | 没有重跑任何神经阶段、没有读取MEPS/问卷/原始相谈、没有改动r1目录 | — |

测试按文件计数（通过/总数）：dialogue_claim_ledger 14、dialogue_speech_scope 3、
proposition_response 13、source_event_graph 13、evidence_history 13、evidence_accumulation 8、
evidence_identity 12、relational_evidence 17、explanation_chain 7、interpretation_evidence 10、
query_moment_operator 9、query_moment_audit 2、query_moment_actions 17、contextual_instrument 8、
local_instrument_cold 2、interaction_response 8、joint_conditional_observation 13、choice_design 16、
response_sharing 10、new_condition_design 15、shared_query_actions 8、query_program_actions 5、
temporal_measurement 9、pilot_transport 9、interpretation_inventory 1、relational_meter 2、
contextual_measurement 6。各报告所称测试数（56/42/28/34/44/26等）与这些文件的组合一致。

判定词汇沿用交接文件第一阶段第3条：**机制存在 / 有限适用规律 / 明确失败反例 /
自然解释 / 第三方复建**。“登记验收产物齐全”指TODO卡片“验收输出”一行的每个对象在
产物中存在；“父项开放”指卡片“剩余问题”所指的开放语义/自然解释仍未解决。二者分开记。

## 1. 总裁决

| 项 | Codex声明 | 本次裁决 | 类别 | 真实缺口（摘要） |
|---|---|---|---|---|
| M01 | 部分完成 | 登记验收产物齐全；父项开放语义未解 | 机制存在+明确失败反例 | 无新文本统一入口；自然覆盖未成表；开放语义5项未满足 |
| M02 | 有限验收完成 | 成立 | 机制存在+有限规律 | 自然273地址无修订链（归M01） |
| M03 | 有限验收完成 | 成立 | 有限规律+反例 | 未知机会时只报轮廓；无自然尾部发现器（已声明） |
| M04 | 有限验收完成 | 成立 | 有限规律+反例 | 新语义条件仍靠M01；平滑世界随机斜率等价已如实记 |
| M05 | 有限验收完成 | 成立 | 机制存在+反例 | 同单元反序反例保留；K与NLI单位未合并（已声明） |
| M06 | 有限验收完成 | 成立 | 机制存在 | 非全34×34、非神经路径唯一（已声明） |
| M07 | 有限验收完成 | 成立 | 有限规律+反例 | 全查询无神经重建未实现（已声明） |
| M08 | 有限验收完成 | 成立 | 有限规律 | 动力学为声明参数（由M11/M12承接，已闭环） |
| M09 | 数值机制完成/自然接口未通过 | 登记验收产物齐全；自然言语行为观察未通过 | 机制存在+明确失败反例 | 自然互动中“对方供给机会 vs 本人回应”需记录分配或独立编码 |
| M10 | 有限验收完成 | 成立 | 有限规律+反例 | 前瞻施测调度未做（已声明非目标） |
| M11 | 有限验收完成 | 成立 | 有限规律 | bootstrap 3达上限/1触界保留；工作区间≠心理CI |
| M12 | 有限验收完成 | 成立 | 有限规律+反例 | 共同冲击结构修正归M09 |
| M13 | 有限验收完成 | 成立 | 自然解释的计算层 | 独立心理/行为编码留D01 |
| M14 | 有限验收完成 | 成立 | 有限规律+反例 | 社交半差r .490→.205/覆盖.766保留；自然273事件0合格；非D02 |
| D01 | 盘点9/9 | 盘点成立；解释依赖仍开放 | 自然解释 | 新对象（对话/修复/请求/条件响应）无对应独立编码与效标 |
| D02 | 未派发 | 未做 | 第三方复建 | 本轮可做跨agent复建并记录规则缺口，不等于盲复建 |
| D03 | 未做 | 未做 | — | 三语理论稿V1（2026-09-03）早于机制轮；CLAIMS_LEDGER无机制行；无冻结清单 |

结论：Codex所称12项有限验收经本次独立核查**成立**，没有发现靠改宽标准或只勾选而结案的项。
M01/M09的“登记验收输出”已经齐全，但父项开放语义按原卡片仍未解决；本审查不把它们改记为完成。
D01–D03按原定义均未完成。第二阶段只补真正缺失的可运行接口/制作交付，不重刷模板精度。

## 2. 逐项审查

### M01 从开放文本学出目标相关的证据与条件（P0）

- 原要求（卡片验收输出）：一个可从新文本计算的证据接口；保留原文地址的条件表示；
  至少一组能区分内容亲近与目标证据的作用对照；记录条件/证据估计损失。最小实验：
  固定工具，比较“给定正确条件/由文本估计条件/仅内容相似度”三层，再检查许可的自然原文。
  出处V5 G1–G3、C03/C14/C16。
- 证据路径：`grain_research/{event_evidence,source_event_graph,evidence_identity,proposition_response,dialogue_claim_ledger}.py`；
  `proposition_response_r1/completion/summary.json`（留出措辞：内容NLI r=.403131/MSE=.045369，
  依存绑定 r=.894985/MSE=.018183 且15646/18432机会，完整事件 r=.895985/MSE=.011838）；
  `evidence_identity_r1/content_collision.json`（内容分数.993314相同而行动−意愿目标±.165707）；
  `proposition_response_r1/audit/{selection_bias.csv,finite_material_bounds.csv}`（383偏移等式、768范围）；
  条件地址 `predicate_condition_addresses=202`；自然12句 `natural_status_counts={"unresolved":12}`。
- 已完成范围：三层对照、地址化条件、内容碰撞反例、漏读分母公式与有界范围均实际交付；
  测试13+13+12+17+14通过；冻结文件hash一致。
- 真实缺口：(1) 五个接口各自只有冻结公开材料的runner，**没有一个接受新对话文本并输出
  “哪些目标可计算、哪些缺前件”的统一入口**（D01盘点报告交付顺序第1条已点名，未做）；
  (2) 自然材料上的目标覆盖只有“12句0匹配”一句话，没有按前件类型分解的覆盖表；
  (3) 父项开放语义未满足清单：a. 跨记录事件同一性（自然273地址无核实链接）；
  b. 显式引用短语之外的省略/情态；c. 来源两条件之外的未命名条件；d. 报告行动的事实性/实施
  状态（`factuality=not_observed`）；e. 自然材料中声明目标的可计算率为0/12。
- 最小补齐动作（本轮执行）：新增新文本入口 `scripts/score_grain_dialogue_ledger_input.py`
  （原文JSON→固定Stanza→双账→按目标的可计算/缺前件账+有界范围），在14公开对话上
  必须与r1逐格一致；对已缓存解析的12自然句运行同一入口，输出只含计数的覆盖表。
- 验收条件：公开重放0差异；自然覆盖表每行有前件缺失原因；未满足清单a–e原文保留在TODO卡。

### M02 多次证据累积（P0）

- 原要求：来源学习相关事件、版本更正/撤回、副本零信息、摘要反例、个人函数/目标区间。
- 证据路径：`evidence_history_r1/evaluate/comparison.csv`（水平MSE .026027/.035029/.044062，
  覆盖.9688/.8750/.7383）；`update_witnesses.json`（同分数同协方差、同一更正、不同结果）；
  `square_root_repair_freeze.json`（旧增量31/282失败保留，修复最大差5.86e-12）；13+8+13测试。
- 已完成范围：全部。冻结14文件一致。
- 真实缺口：自然台账修订链0、跨记录同事件0——属于M01开放项，不是本项未完成。
- 最小补齐动作：无。验收条件：已满足。

### M03 机会、选择与条件响应（P0）

- 原要求：两层选择树、机会offset、自由/固定/两相设计、状态选择反例、尾部信息。
- 证据路径：`choice_design_r1/{choice_metrics.csv,design_metrics.csv,exact_witnesses.json,audit.json}`；
  报告：未校正r .917/.930高于校正后.771/.796但参数RMSE 1.47/1.53 vs .633/.640；16测试。
- 已完成范围：全部；hash 5/5。
- 真实缺口：未知机会供给下偏好/供给不可分（已声明为轮廓输出）；无自然尾部发现器（卡片明示不要求）。
- 最小补齐动作：无。

### M04 一个函数同时保存水平与条件响应（P0）

- 原要求：来源配对响应核、个人水平/响应分账、隐藏条件/有限变化、目标相关相似人、简单方法适用范围、反例。
- 证据路径：`response_sharing_r1/evaluate/comparison.csv`（交叉世界隐藏差分r .8414/MSE .044659，
  几何/随机斜率.156086；平滑世界随机斜率.014255 vs 全核.014327）；`joint_conditional_observation_r1`
  （联合r .3578/.0408/.3594）；`temporal_measurement_r1`；10+13测试。
- 已完成范围：全部；hash一致。
- 真实缺口：新语义条件仍由M01；本项按原卡片不要求解决。
- 最小补齐动作：无。

### M05 GRAIN配置与查询作用转成可用输入（P0）

- 证据路径：`query_moment_r1`（组内方向RMSE −24.8%/−22.3%，两次D1冷运行）；
  `query_moment_finite_r1/contract.json`等（.02下−33.3%/−25.1%，.10下−46.7%/−22.2%；
  同单元反序16/16改变原生端点而矩不变）；`joint_conditional_observation_r1`；9+2+17+13测试。
- 已完成范围：新增可冷计算观察量+有限作用+压缩盲点反例+条件应用；全部成立。
- 真实缺口：K与NLI单位未融合（卡片已声明不要求）。
- 最小补齐动作：无。

### M06 共享/特有查询路径（P1）

- 证据路径：`shared_query_actions_r1/{component_actions.csv,ownership_interactions.csv,cross_query_Q.csv}`；
  P分账73.5%–80.5%；他人引用保留94.7%；560调用；8测试；hash一致。
- 已完成范围：全部。真实缺口：无未交付项；非唯一注意力路径已声明。

### M07 两条独立路线的剩余依赖（P1）

- 证据路径：`query_program_actions_r1/{summary.json,independent_audit.json,binding_order_components.csv}`；
  A重建/编译差1.55e-15；B早期C分账98.6%但幅度.004379 vs 原生.288880；同词袋18/18归属差分为0；5测试。
- 已完成范围：全部。真实缺口：全查询无神经重建未实现——卡片明示不以此阻塞。

### M08 trait/state动态对象与时间窗口（P1）

- 证据路径：`temporal_measurement_r1/{metrics.csv,kernels.csv,natural_clock_comparison.csv}`；
  RMSE .30905→.14319；窗口覆盖.69271/.96875；276作者日期应用；623 MEPS元数据对；9测试。
- 已完成范围：全部（已知动力学）。真实缺口：动力学参数由M11/M12来源pilot学习——已闭环。

### M09 对话中的共同生成与互动响应（P1）

- 原要求（卡片验收输出）：作者响应、对方输入和相互作用的分账及可识别范围；保留单次会话
  测量目标；无连续轮次语料不补造延迟，可先完成只依赖轮序的版本。
- 证据路径：`interaction_response_r1/math/`（共同原因c RMSE .11343→.05396，反馈.13298→.05853，
  弱分配IV 5.70895、排除限制违反+.296、恒定伙伴`conditional_sum_only`）；`nli/public_nli.csv`
  （普通前文66/72，拒绝命题提问3/24）；`natural/observer_qualification.json`（Qwen EN4/6、JA2/6失败，
  24名MEPS输出禁作行为解释）；对话双账（只依赖轮序的版本）；8+14+3测试。
- 已完成范围：登记验收输出全部存在；失败反例如实保留。
- 真实缺口：自然对话中的言语行为观察器未通过（3/24、2/6）；“对方提供的表达机会”与
  “本人回应”在自然记录中无分配记录，不可由文本自动识别（互动报告§2）。这两项按交接文件
  属于“通用语义未解”，不再调参。剩余可运行部分与M01共用新文本入口。
- 最小补齐动作：与M01同一入口；把M09未满足项写成D01编码规范的对象（对谁说/回应哪个命题/
  态度/当前实施还是报道），不再做NLI/Qwen提示网格。
- 验收条件：未满足清单进入TODO卡与D01规范；父项状态保持开放。

### M10 未知真参数下的观察选择（P0）

- 证据路径：`new_condition_design_r1/evaluate/{final_comparison.csv,identifiability_identity.json,next_observation_loss_surface.csv}`；
  −25.9%/−30.0%；隐藏方向恰增Var(d)=.034133；补采64来源后.036596→.010940（−70.1%）；15测试。
- 已完成范围：全部。真实缺口：无。

### M11 分数含义、参照与误差（P1）

- 证据路径：`pilot_transport_r2/gaussian/{fits.json,bootstrap_fits.json,metrics.csv}`；33主拟合收敛、
  360 bootstrap中3达上限/1触界保留；9测试；hash 123/123。
- 已完成范围：全部。真实缺口：工作区间≠心理CI（已声明）。

### M12 跨情境保留/重估（P1）

- 证据路径：`pilot_transport_r2/text/`（覆盖.333/.557→.969/.964）；条件化计分器展开显示a/b混合与zu项；
  “保留/重估/重造”决策表在报告中。
- 已完成范围：全部。真实缺口：共同冲击结构修正归M09（其数值部分已完成）。

### M13 可计算对象到可解释意义（P1）

- 原要求：原文证据→数学操作→个人相对位置/变化→候选含义；分别记录模型功能、自评关系、实际行为证据。
- 证据路径：`explanation_chain_r1/{summary.json,contrasts.csv,cards/,interpretation_audit/}`；
  请求极差+.013705（8/12）、时限差+.076500（12/12）；FA/查询独立展开3.55e-15/4.44e-16；
  历史S协方差r=−.263890不转给U；7+10+9+2测试；106产物+16冻结一致。
- 已完成范围：计算解释层全部；三层证据表如实分开。
- 真实缺口：独立心理/行为编码——属D01。最小补齐动作：无（D01承接）。

### M14 同一制作方法的两个局部示范（P1）

- 原要求（V3 Study B验收产物）：每个工具有构念/对象卡、证据单位、计分器、参照、材料要求、
  未参与制作作者的结果及可读解释；B1–B7共同规格；两情境参数差异记录。
- 证据路径：`local_instrument_r1/{instruments/,application/,evaluation/metrics.csv,cold/,audit.json}`；
  手册 `docs/GRAIN_CONTEXTUAL_CONSTRUCTION_HANDBOOK_V1.md` §1–§14；当前表达RMSE .2446→.1189、
  r .5575→.6145；社交半差r .4898→.2052、覆盖.7656保留；3840独立复算最大差3.77e-11；86冷NLI对；8+2测试。
- 已完成范围：计算制作、冻结、冷运行、独立复算、解释卡、手册；hash 89/89一致。
- 真实缺口：(1) 真人两情境效度=D01；(2) 独立制作=D02；(3) 手册§5按裸函数名引用
  `calibration/restore_units/learn_cell_moments/fit_pilot` 等，未给模块路径（实际分散在
  `pilot_transport/choice_design/joint_conditional_observation/response_sharing/new_condition_design/evidence_history`），
  第三方需查实现——这是D02应记录的规则缺口，在本审查先记下。
- 最小补齐动作：在D02复建中记录并回填手册的函数定位表；不改冻结bundle。

### D01 心理与行为解释依赖

- 原要求：独立盲编码、实际行为效标或适配时间的外部量表；缺少时列最小新增观察。
- 证据路径：`interpretation_inventory_r1/{source_inventory.json,summary.json}`（本次9/9 hash一致）；
  `NARROW_TARGET_RECONCILIATION_REPORT.md`（改编低落/负性/合并 r .564458/.421701/.554777为46人旧结果重放；
  关系/会话族最小q .121572）。
- 已完成范围：既有入口盘点与用途边界；对象对应表（报告§“为什么D01没有整体勾完”）。
- 真实缺口：新对象无对应独立编码与效标：对话命题/修复账、当次求助/修复言语行为、条件响应参数、
  匿名因素含义。旧46人关联只能支持旧对象。当前情绪查询可复用改编评分（同读者同时点）。
- 最小补齐动作（本轮执行）：(1) 机器可读的对象→可复用证据/所需新编码映射表；(2) 独立编码
  规范V1（单位、字段、角色/命题/态度/实施时态/引用/前件、盲于模型输出、一致性计划、隐私）；
  (3) 在14公开对话上生成预填“模型编码”的编码表模板，明确标注为模型编码而非人类盲编码。
- 验收条件：映射表不把任何旧r转给新指标；规范可被未参与者执行；依赖保持开放。

### D02 独立制作复建

- 原要求（V3 Study D）：只接收制作手册与新情境材料说明，不给旧结果/最优参数；记录必须
  临时询问/补规则之处；由独立角色检查。
- 证据路径：无产物。同agent冷启动测试（`cold/`）存在但不是D02。
- 真实缺口：全部。
- 最小补齐动作（本轮执行，按交接文件允许的跨agent复建）：按手册§4命令在新目录重建两实例并
  与r1数值比对；再只依据手册§2/§3/§5与库函数签名，为一个新的两条件对象独立写制作脚本，
  逐条记录“需查实现/需询问/需补规则”的决策日志。
- 验收条件：复建日志逐条有编号；数值比对结果如实；边界写明“跨agent、有开发上下文，非盲第三方”。

### D03 理论论文与冻结包

- 原要求：定义、继承定理、机制发现、反例、两个示范、作用范围、许可与中日英稿；冻结/发布另按用户指示。
- 证据路径：`docs/paper/GRAIN_THEORY_MANUSCRIPT_{EN,JA,ZH}_V1.md`（2026-09-03，含crosswalk与provenance）；
  `docs/CLAIMS_LEDGER.md` 无 mechanism_todo_v1 行；无本轮内容清单。
- 真实缺口：机制轮（M01–M14）的理论包未整理；三语稿未覆盖本轮；无可复核清单。
- 最小补齐动作（本轮执行）：写机制理论包索引（继承定理/本地发现/反例/适用范围分栏，每行有产物路径）；
  按既有crosswalk术语写三语一致的增补稿（作为V1稿§5/§6/§8的补编，不重写全文）；
  生成本地manifest（文件hash），不commit/tag/push、不开锁箱。
- 验收条件：三语增补每段一一对应；数字只引用本审查核过的产物；清单hash可重算。

## 3. 共通发现

1. **hash与测试**：所有r1研究文件与其冻结记录一致；`pilot_transport_r1`只有config/contract，
   其记录的旧代码hash与当前不同，属于被r2取代的预跑，报告已以r2为准。
2. **AGENTS/CLAUDE最新覆盖**说“12/14”“D02不派发”“goal PAUSED”，与产物一致；本任务的
   用户指令覆盖“不派发D02”一条（交接文件首段）。
3. **手册可执行性**：§4复建命令依赖 `condition_accumulation_r1/public_scored_bank.csv`、
   `public_nli.csv` 与私人 `relational_evidence_r1/natural/private_relations.jsonl`（natural阶段可略），
   不是零资产发行包；§5函数名无模块定位。
4. **CLAIMS_LEDGER**未登记任何机制轮主张；D03须补但只在本地。
5. **私人数据**：本次只读私人文件的hash与行数（D01）、解析缓存的结构，不输出个人文本/数值。

## 4. 第二阶段执行清单（按交接文件顺序）

| 序 | 动作 | 改变的主线判断 | 预算 | 输出 |
|---|---|---|---|---|
| 1 | M01/M09新文本入口+公开重放+自然覆盖表 | “接口可用”从声明变为可运行；父项未满足清单定稿 | 本地Stanza≤100句、0 NLI | `claude_todo_closure_20260913/m01_m09_dialogue_input_entry/` |
| 2 | D01对象映射+编码规范+模型预填编码表 | 明确哪些旧效标可复用、哪些必须新编码 | 0神经调用 | `d01_evidence_object_map/`、`docs/GRAIN_INDEPENDENT_CODING_SPEC_V1.md` |
| 3 | D02手册复建+独立构造日志 | 手册是否足以复建；规则缺口回填手册 | 数值CPU≤240秒、冷NLI≤160对 | `d02_handbook_rebuild/`、`d02_independent_construction/` |
| 4 | D03理论包索引+三语增补+本地manifest | 本轮结果可交接、可复核 | 文档 | `docs/paper/GRAIN_MECHANISM_THEORY_PACKAGE_V1.md` 等、`d03_theory_package/manifest.json` |
| 5 | 更新TODO/手册/AGENTS/CLAUDE/闭环报告 | 状态与产物一致 | 文档 | 见闭环报告 |

不做：模板精度网格、新神经训练、MEPS拟合/读取、远端主机、commit/push/tag、开锁箱。
