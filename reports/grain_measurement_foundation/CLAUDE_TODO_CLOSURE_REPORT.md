# Claude TODO闭环报告：实际补齐、结果、测试与逐项裁决

2026-09-13。任务 `CLAUDE-TODO-CLOSURE-20260913-01` 第二阶段产物；第一阶段见
[CLAUDE_TODO_CLOSURE_AUDIT.md](CLAUDE_TODO_CLOSURE_AUDIT.md)。
本报告只记录实际运行过并留有产物的工作；没有新神经训练、MEPS拟合/读取、远端主机、
commit/push/tag或开锁箱。结果根目录：
`results/grain_vector_mechanisms/mechanism_todo_v1/claude_todo_closure_20260913/`（下文以`closure/`代指）。

## 1. 一页结论

| 项 | 裁决 | 本轮实际动作 | 状态 |
|---|---|---|---|
| M02–M08, M10–M14 | 12项有限验收**确认成立** | 250测试通过、23目录冻结hash一致、标题数字定位到产物文件 | 完成（有限） |
| M01 | 登记验收产物齐全；父项开放语义未解 | 新文本入口+公开重放486格0差异+自然覆盖表 | 有限结案+未满足清单 |
| M09 | 登记验收产物齐全；自然言语行为观察未通过 | 同一入口；未满足项写入D01编码规范 | 有限结案+明确失败反例 |
| D01 | 盘点成立；解释依赖开放 | 对象→证据映射、独立编码规范V1、模型预填编码表 | 外部依赖（人类编码/效标） |
| D02 | 跨agent复建完成；盲第三方未做 | 手册§4重建逐值一致；手册-only构造+第三对象；10查实现/1询问/4补规则；手册§15回填 | 外部依赖（独立执行者） |
| D03 | 理论包与三语增补完成；冻结待用户 | 机制理论包索引、EN/JA/ZH增补、出处表、crosswalk增补、本地manifest | 完成（本地），发布待用户 |

“12/14”经独立审查成立。本报告不把M01/M09改记为完成；它们的登记验收产物齐全这一事实与
父项未满足清单同时保留。整个TODO的剩余支付都在人类侧：独立编码、盲复建、冻结/发布决定。

## 2. 逐项实际补齐

### 2.1 M01/M09：新文本入口与自然覆盖

改变的主线判断：从“接口存在于冻结runner内”变为“可对新对话文本运行并说明可计算性”。

- 新增 `scripts/score_grain_dialogue_ledger_input.py`：原文JSON→缓存/本地Stanza解析→冻结
  `read_dialogue`/`scope_speech`/`frames`→`turn_accounts.csv`（无文本）、`target_computability.csv`
  （每对话×4目标：可计算/分数/活跃主张/未解决同目标极性轮/未分配极性轮/已知与未知条件范围/缺前件原因）、
  `coverage.json`、`(private_)ledgers.json`、`manifest.json`；`--replay-check`与冻结r1逐格比较。
- 新增 `scripts/build_grain_natural_dialogue_input.py`：从冻结命题包的缓存自然句生成私有输入（不读新语料）。
- 测试 `tests/test_grain_dialogue_ledger_input.py`（4项，无神经调用）。
- 公开重放 `closure/m01_m09_dialogue_input_entry/public_replay/`：14对话/81轮/33缓存解析，
  0新解析、0 NLI；与r1比较486格（操作+4目标分数+话语计数）**0不一致**；56目标中9可计算
  （每主对话恰1个声明目标）。
- 自然覆盖 `closure/m01_m09_dialogue_input_entry/natural_coverage/coverage.json`：12句/12轮，
  12缓存解析、0新解析；**0/48目标可计算**，原因全部为`no_claim_update`（无前置问题）；
  框架候选：过去报告11、意愿4、当前请求1、前瞻1、未特定29（=46，与命题报告一致）；
  行为者：本人11、他人15、未解析6、伙伴4、来源域内10；目标相关框架0；条件从句10。
  这就是“自然输入中哪些目标可计算、哪些缺前件”的实测答案：缺的是**前置命题与显式引用**，不是解析。
- 有界范围：公开集未分配极性轮2（挑战集的裸`Yes.`与无前文修复），对应未知条件包络[-1,1]。

未满足清单（保留在TODO卡与审计§2）：a 跨记录事件同一性；b 显式引用之外的省略/情态；
c 来源两条件外的未命名条件；d 报告行动的事实性/实施；e 自然材料声明目标可计算率0/12；
M09另加 f 自然言语行为观察器未通过（3/24、2/6），g 自然记录无伙伴输入分配。

### 2.2 D01：证据对象映射与独立编码规范

改变的主线判断：明确哪些旧效标可按原对象复用、哪些新对象必须新编码，不再把旧r转给新指标。

- `closure/d01_evidence_object_map/object_evidence_map.json`：7个对象。可按原对象复用：当前情绪查询
  （改编评分，r .564458/.421701/.554777为46人重放）、关系/会话函数（q≥.121572，探索）、34特质参照。
  开放依赖：命题版本账、当次言语行为、条件响应参数、匿名因素含义。
- `docs/GRAIN_INDEPENDENT_CODING_SPEC_V1.md`：单位=话轮；字段 speech_source / addressee /
  proposition_actor / modality / enactment / reference_type / reference_target / condition_mentioned /
  quoted_or_reported_speech / polarity_if_claim；≥2名未参与者、最小样本、逐字段κ/α、盲序、隐私与禁止事项。
- `closure/d01_evidence_object_map/model_coded_pilot_sheet_public_dialogues.csv`：81行公开对话，
  `MODEL_*`列为冻结规则输出（模型编码），`HUMAN_*`列空；发放前须删MODEL列。
- 9个既有入口hash复核9/9（`closure/audit_verification/d01_inventory_hash_check.json`），未读私人数值。
D01按原定义仍开放：需要人类独立编码与目标对应的效标。

### 2.3 D02：手册重建与手册-only构造

改变的主线判断：手册是否足以让另一执行者复建；缺口回填手册。

- 手册§4命令重建（`closure/d02_handbook_rebuild/local_instrument_rebuild/`，26秒）：5阶段+2冷入口
  （22/64实际NLI对）+独立审计PASS；与r1比较6 bundle/60指标行/10冷分数**全部差0**
  （`rebuild_vs_r1_numeric_comparison.json`）。
- 手册-only构造（`scripts/construct_grain_instrument_from_handbook.py`，不导入runner；
  `closure/d02_independent_construction/`）：社交实例来源矩与r1差0；第三对象
  `task_condition_profile_v1`（任务完成短/长两条件轮廓）从构造者声明的模拟pilot建成，
  新作者以冻结冷入口22次实际NLI前向计分，与缓存边际差1.44e-12。
- 决策日志：LOOKUP 10、ASK 1、RULE 4（去重10处）；bootstrap种子规则复制自runner这一点已如实记录，
  不作为“手册固定了规则”的证据。手册新增§15函数定位表、bundle必需键、未写明规则。
- 边界：构造者带开发上下文；V3 Study D的盲第三方复建仍开放；真实新对象的来源pilot来源是方法缺口。

### 2.4 D03：理论包、三语增补、本地清单

- `docs/paper/GRAIN_MECHANISM_THEORY_PACKAGE_V1.md`：12继承方法→本地对应、14本地发现、12反例、8范围、
  两示范+第三次构造、与V1稿接合、冻结清单说明。
- `docs/paper/GRAIN_MECHANISM_SUPPLEMENT_{EN,JA,ZH}_V1.md`：S1–S7段落一一对应，复用crosswalk术语，
  新增命题9–11与三条须保持一致的主张。`GRAIN_THEORY_MANUSCRIPT_CROSSWALK.md`增补§7。
- `docs/paper/GRAIN_MECHANISM_EVIDENCE_PROVENANCE.md`：每个数字的产物路径。
- `scripts/build_grain_theory_package_manifest.py` → `closure/d03_theory_package/manifest.json` + `MANIFEST.md`
  （文件sha256）。未commit/tag/push；CLAIMS_LEDGER未改（其规则要求数字由脚本从产物计算，留待正式登记）。

## 3. 测试与检查

| 集合 | 结果 | 记录 |
|---|---|---|
| 27个机制相关测试文件（第一阶段） | 250通过 | `closure/audit_verification/tests/tests_batch1_*.xml`, `tests_batch2_*.xml` |
| 新入口+相关集合（第二阶段） | 71通过（含新增4项） | `closure/audit_verification/tests/tests_batch3_phase2_new_entry.xml` |
| 冻结hash | 23目录，研究文件全部一致；pilot_transport_r1预跑2项旧hash（被r2取代） | `closure/audit_verification/freeze_verification_r1/` |
| ruff | 5个新脚本+1新测试全部通过 | — |
| `git diff --check` | 通过 | — |
| 新神经调用 | 冷入口实际NLI：重建22+64对、第三对象22对；解析0新调用 | 各manifest |

复跑命令（输出目录须不存在）：

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/verify_grain_mechanism_freezes.py --output <新目录>
"$PY" scripts/score_grain_dialogue_ledger_input.py --input results/grain_vector_mechanisms/mechanism_todo_v1/dialogue_claim_ledger_r1/public_bank.json --parses results/grain_vector_mechanisms/mechanism_todo_v1/dialogue_claim_ledger_r1/parses.json --output <新目录> --replay-check results/grain_vector_mechanisms/mechanism_todo_v1/dialogue_claim_ledger_r1
"$PY" scripts/construct_grain_instrument_from_handbook.py --output <新目录> --pilot-raw <重建目录>/raw/social_condition_profile_v1_91401_pilot_source.jsonl --reference-bundle results/grain_vector_mechanisms/mechanism_todo_v1/local_instrument_r1/instruments/social_condition_profile_v1_91401.json
"$PY" -m pytest -q tests/test_grain_dialogue_ledger_input.py tests/test_grain_dialogue_claim_ledger.py tests/test_grain_contextual_instrument.py
```

## 4. 最小下一动作（均在人类侧）

1. D01：按编码规范招募≥2名未参与者，对许可自然对话（≥40轮，含alias/revise/retract各≥5）独立编码；
   完成后用新入口比较模型-人类逐字段一致。
2. D02：向未参与设计的执行者交付手册（含§15）与一份新情境材料说明，不给r1结果；用决策日志模板记录。
3. D03：由用户决定是否冻结/登记（CLAIMS_LEDGER）与发布；本地manifest已可复核。
4. M01/M09父项：只有在D01自然编码返回后，才决定是否值得扩展显式规则；不做模板精度网格。

## 5. 新增/修改文件清单

新增脚本：`scripts/verify_grain_mechanism_freezes.py`、`scripts/score_grain_dialogue_ledger_input.py`、
`scripts/build_grain_natural_dialogue_input.py`、`scripts/construct_grain_instrument_from_handbook.py`、
`scripts/build_grain_theory_package_manifest.py`；新增测试 `tests/test_grain_dialogue_ledger_input.py`。
新增文档：本报告、审计报告、`docs/GRAIN_INDEPENDENT_CODING_SPEC_V1.md`、
`docs/paper/GRAIN_MECHANISM_THEORY_PACKAGE_V1.md`、三语增补、`GRAIN_MECHANISM_EVIDENCE_PROVENANCE.md`。
修改文档：`docs/GRAIN_CONTEXTUAL_CONSTRUCTION_HANDBOOK_V1.md`（§15）、
`docs/paper/GRAIN_THEORY_MANUSCRIPT_CROSSWALK.md`（§7）、`docs/GRAIN_MECHANISM_TODO_20260913.md`、
`AGENTS.md`、`CLAUDE.md`（两仓库）、Persona `PROJECT_STATUS.md`。冻结r1目录未改动。
