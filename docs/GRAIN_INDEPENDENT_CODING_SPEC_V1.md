# GRAIN 独立编码规范 V1（D01最小材料规范）

2026-09-13。任务 `CLAUDE-TODO-CLOSURE-20260913-01` 第二阶段产物。
用途：为对话双账、命题回应、互动响应三个新观察对象提供**独立于模型输出的人类编码**，
使D01“心理与行为解释依赖”有可执行的最小材料规范。本规范不是量表，不产生人格分数。
模型（含Claude/Codex/固定规则）的编码只能作为“模型编码”标注，不得冒充本规范要求的独立编码。

## 1. 编码对象与对应的模型输出

| 新观察对象 | 模型侧输出（冻结规则） | 本规范要求的独立编码 | 现有效标能否复用 |
|---|---|---|---|
| 命题版本账（M02接口） | `operation ∈ {new, repeat, alias, revise, retract, no_claim_update, …}`、活跃主张有限均值 | 每话轮的引用类型与被引用对象 | 否：旧46人关联属于其他对象 |
| 当次言语行为（M09接口） | `direct_request_candidate`、`task_assistance_candidate`、`discourse_permission_candidate` | 发话者、受话者、被请求实施行为的主体、是否当次实施 | 否：依恋/关系自评不是“请求已实施”的金标准 |
| 命题绑定（M01接口） | `frames` 通道（意愿/过去报告/偏好/能力/当前体验/当前请求/前瞻）、`actor` | 命题主体、模态、时态/实施状态、条件从句 | 否 |
| 条件响应参数（M09数值） | IV/控制函数系数 c,d | 记录的伙伴输入分配（或随机分配） | 否：自然记录无分配时不得从文本补造工具变量 |

机器可读对应表：`results/grain_vector_mechanisms/mechanism_todo_v1/claude_todo_closure_20260913/d01_evidence_object_map/object_evidence_map.json`。
可直接复用的既有联系仅限对象/时点一致者（当前情绪查询的改编评分、旧关系/会话函数、34特质参照）。

## 2. 编码单位与字段

单位=一个话轮（turn）。每单位填写以下字段；不确定填 `unresolved`，不猜。

| 字段 | 取值 | 说明 |
|---|---|---|
| `speech_source` | respondent / partner / quoted_other / unresolved | 谁发出这句话；引语内的话记为quoted_other |
| `addressee` | partner / respondent / third_party / none / unresolved | 对谁说 |
| `proposition_actor` | respondent / partner / other / none / unresolved | 句内被陈述或被请求的行为主体（“请帮我”→partner） |
| `modality` | intention / past_action_report / preference / ability / current_experience / current_request / prospective / other | 可多选，用`;`分隔；多功能并存不强制单类 |
| `enactment` | enacted_now / reported_past / planned / hypothetical / unresolved | 该行为是当次实施、过去报告、计划还是假设 |
| `reference_type` | new / alias / revise / retract / none / unresolved | 与此前哪一主张的关系；alias=复述同一主张，revise=替换，retract=撤回 |
| `reference_target` | 话轮ID或none | 被引用的话轮 |
| `condition_mentioned` | 原文条件从句的字符区间或none | 不改写、不归类为心理情境 |
| `quoted_or_reported_speech` | yes / no | 是否含引语/转述 |
| `polarity_if_claim` | +1 / −1 / none | 仅当为对某命题的明确肯定/否定时 |
| `notes` | 自由文本 | 记录歧义与理由 |

字段与冻结规则输出一一对应，但编码者**不得看到**模型列。预填的
`model_coded_pilot_sheet_public_dialogues.csv` 中 `MODEL_*` 列在发放给编码者前必须删除。

## 3. 材料、样本与流程

1. 材料：只使用已获许可的材料。公开编写对话（14条/81轮）用于培训与一致性练习；
   自然材料须经授权，编码在本地进行，原文不上传外部API，编码表不含原文时才可提交仓库。
2. 编码者：至少2名未参与规则设计的人；每人独立完成后再合并。任何agent的编码只作为
   第三列“模型编码”参与比较，不计入人类一致性。
3. 规模（最小）：训练集14公开对话；正式集≥40个自然话轮，且至少覆盖
   `reference_type∈{alias,revise,retract}` 各≥5例，否则版本账字段的一致性不可估计。
4. 一致性：对每个字段报告Cohen's κ或Krippendorff's α与原始一致率；不报告总平均掩盖单字段失败。
   分歧记录保留，不以讨论后的“共识”覆盖原始独立编码。
5. 盲序：先完成人类编码，再解冻模型列；比较时按字段分别报告模型-人类一致，不合并成单一准确率。
6. 结果用途：仅用于（a）验证冻结规则在自然材料上的覆盖与错误类型，（b）为M01/M09
   父项的未满足清单提供自然证据。不用于拟合规则、选择阈值或训练模型。

## 4. 明确不做的事

- 不把依恋、信任、情绪自评当作“请求/修复已实施”的效标。
- 不把当次请求计入行动报告目标；不把过去报告当已核实行为。
- 不从单句`you`推断真实对话伙伴；无前文时`addressee=unresolved`。
- 不对MEPS原始问卷或相谈原文做新的读取、拟合或外发；MEPS仅验证角色。
- 不因编码结果调整冻结规则；如需新规则，另起版本并保留旧结果。

## 5. 交付清单

- 本规范；`object_evidence_map.json`；预填模型编码表（MODEL列删除后发放）；
- 编码完成后：去文本的编码表、一致性报告、模型-人类逐字段比较、分歧清单。
以上完成前，D01保持开放；本规范只是最小材料规范，不是D01完成。
