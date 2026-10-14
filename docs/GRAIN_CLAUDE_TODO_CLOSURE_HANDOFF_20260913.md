# Claude独立验收与补齐授权

任务ID：CLAUDE-TODO-CLOSURE-20260913-01。
用户当前指令：todo的闭环先让Claude检查完成情况，然后让Claude补齐。
这覆盖旧AGENTS/CLAUDE里的“尚不派发D02”安排限制；并非授权自动发布或使用私人资料训练。

## 目标和协作

从现有Project Persona的SUICA上下文派生Claude后台任务，Fable 5.1 / max。
Codex让出本轮修改权。先独立检查，再实际补齐，不只提出另一份计划或重述Codex结论。
目标是用于制作情境化文本心理测量的基础理论与制作过程，不是仓促做临床成品量表，
也不是单纯提高PANDORA预测精度。继承适用的成熟方法；不强求精确复现每篇前身论文。

工作仓库：`/Volumes/mobile3/projects/Sliced Utterance In-Context Assessment`。
历史仓库：`/Volumes/mobile3/projects/project persona`。
Python：`/Volumes/mobile3/projects/project persona/.venv/bin/python`。
两个工作区都有大量既有改动和未跟踪文件，不清理、不重置、不覆盖他人工作。

## 第一阶段：先验收，不先改算法

1. 读当前AGENTS/CLAUDE最新覆盖、`docs/GRAIN_MECHANISM_TODO_20260913.md`、
   `docs/GRAIN_ALL_PLAN_RESEARCH_LOOP_V1.md`。按其索引复核历版终极计划的要求，
   不机械重读所有历史文件，也不把旧待办状态当现在状态。
2. 检查每项M01-M14及D01-D03：原要求、最新报告、实际代码/数据产物、测试、
   冻结文件、可运行性和反例是否匹配。Codex所称12/14是待审声明，不是既定裁决。
3. 区分机制存在、有限适用规律、明确失败反例、自然解释和第三方复建。
   允许有用的有限结论结案；不得靠改宽验收标准、缩窄未交付要求或只勾checkbox结案。
   同时也不要把任何自然语言例外都升级成无限分类准确率课题。
4. 重点读最新两个报告：
   `reports/grain_measurement_foundation/DIALOGUE_CLAIM_LEDGER_REPORT.md`、
   `reports/grain_measurement_foundation/INTERPRETATION_EVIDENCE_INVENTORY_REPORT.md`。
   以及其引用的PROPOSITION_RESPONSE、EVIDENCE_IDENTITY、EVIDENCE_HISTORY、
   INTERACTION_RESPONSE、EXPLANATION_CHAIN、LOCAL_INSTRUMENT_CONSTRUCTION报告。
5. **必须先写独立审查结论，再开始补代码/实验。** 文件：
   `reports/grain_measurement_foundation/CLAUDE_TODO_CLOSURE_AUDIT.md`。
   每项写证据路径、已完成范围、真实缺口、最小补齐动作及验收条件。

## 第二阶段：直接补齐

完成第一阶段后无需再等待Codex批准，按审查结果执行有用且有界的缺项。
优先真正缺失的机制连接/自然对象接口/制作交付，不在同一模板或精度指标上循环。
每项先声明要改变哪个主线判断、预算和输出，再运行、检查、更新状态。

- M01/M09：检查现有来源/模态/条件/话轮/版本接口能否满足原有限机制要求。
  剩下若是具体可运行接口就补齐；若是通用语义未解，明确哪些原要求仍未满足，
  不把公开模板正确率当自然语言能力，也不为追求通用完美无限调参。
- D01：9个已有证据入口已盘点；先复用合法、目标/对象/时点对应的既有外部联系。
  不把旧25查询的人类相关转给新的请求/修复指标。Claude编码只是模型编码，
  不能冒充独立人类盲编码；没有实际行为真值时保留该依赖并提供最小材料规范。
- D02：本次允许Claude执行独立制作尝试。先从M14手册/指定公开材料起建，
  记录每次必须查实现、询问或补规则的地方。你已有开发上下文，所以属于
  跨agent复建而非完全不知情的第三方盲复建；如原要求更强，保留准确边界。
- D03：整理本轮理论包及中日英稿。先搜已有三语论文，复用其学术表达和出处，
  避免三份互不一致的新稿。继承定理、本地发现、反例和适用范围分开。
  如冻结仅指可复核的内容清单，可生成本地manifest；不自行commit/tag/push或开锁箱。

## 权限与成本边界

- 可编辑相关代码、测试、计划、报告；手动代码改动优先apply_patch。
- 保留冻结r1及失败结果，修正用新版本/伴随文件。审查原产物再运行，缓存优先。
- 不启动旧长队列、新神经训练或无界参数网格；超过短CPU/现有模型有界测试时，
  先记录预计资源与剩余决策价值，长训练/新付费服务需要用户另行确认。
- 默认重训练主机192.168.1.20 Windows、可用VRAM小于10GB；评估主机192.168.1.30
  Linux Intel GPU 16GB。此任务不要求连接主机或启动训练。
- MEPS及相谈资料仅本地验证，不拟合/选参/重选查询、不读取未授权原始问卷，
  不上传个人原文、心理值、向量或身份。先用已有匿名汇总/许可数值缓存。
- 不读取或打印.env和凭证、不改系统权限配置、不删除数据、不打开锁箱。
- 不自动commit/push/release、不另开多agent树；独立完成本任务，遇真实外部依赖明确记录。
- 这是独立CLI后台任务；不要恢复Codex的paused goal、MLCS或旧自动化。

## 交付

1. `reports/grain_measurement_foundation/CLAUDE_TODO_CLOSURE_AUDIT.md`：先写。
2. `reports/grain_measurement_foundation/CLAUDE_TODO_CLOSURE_REPORT.md`：实际修正、结果、测试、
   完成/有限反例结案/外部依赖的逐项裁决及最小下一动作。
3. 本地生成物：`results/grain_vector_mechanisms/mechanism_todo_v1/claude_todo_closure_20260913/`，
   每实验独立子目录，含参数、输出、来源和错误记录，不覆盖旧结果。
4. 更新TODO、共同制作手册、AGENTS/CLAUDE最新状态及必要论文；与真实产物一致。
5. 最终中文短总结给用户，列实际文件、测试、哪些项真的补齐，哪些需要人类或新资料。

若额度或上下文耗尽，保存准确checkpoint和剩余动作；不要把“任务已派发”写成“研究完成”。
