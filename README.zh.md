# GRAIN：个体语言纹理分析法

**Group-Referenced Analysis of Individual Natural-Language Expression**<br>
**基于群体参照的个体自然语言表达分析法**

[English](README.md) | [日本語](README.ja.md) | **简体中文**

GRAIN 是面向文本测量研究的基础分析法。它把一个人的自然语言事件看成在
多种粒度上被观察的局部“颗粒”，并在明确参照群体下分析这些颗粒可复现的
分布、方向、条件响应与路径结构；它们组织起来的多尺度模式，就是个体的
**语言纹理**。GRAIN 本身不是量表，也不要求现有语料成为量表。未来的文本
测评只有在补齐自身的构念、人群、效度与用途证据后，才能建立在这一基础上。

早期的西瓜与瓜皮模型仍作为方法发展史保留：话题和情境不是应被直接剥除的
包装，因为条件选择与条件内反应也可能承载个人信息。GRAIN 将这一发现扩展
为不依赖单一切片隐喻的通用分析法。参见[名称迁移与兼容规则](docs/GRAIN_NAMING_MIGRATION.md)。
公开迁移范围记录于 [v1.1.0 发布说明](docs/RELEASE_NOTES_V110.md)。
标签门禁修正记录于 [v1.1.1](docs/RELEASE_NOTES_V111.md)。

## 理论论文稿（2026-09-01）

已将完成的理论体系整理为不预设投稿期刊的理论与方法论文，并分别提供
[英文](docs/paper/GRAIN_THEORY_MANUSCRIPT_EN_V1.md)、
[日文](docs/paper/GRAIN_THEORY_MANUSCRIPT_JA_V1.md)和
[中文](docs/paper/GRAIN_THEORY_MANUSCRIPT_ZH_V1.md)版本。
[三语对照表](docs/paper/GRAIN_THEORY_MANUSCRIPT_CROSSWALK.md)固定了章节、
公式、专业术语和主张边界。三版均不声称 GRAIN 已经是完成效度验证的量表或
临床工具。[证据来源表](docs/paper/GRAIN_THEORY_EVIDENCE_PROVENANCE.md)
逐项记录主要数值与仓库内原始分析报告的对应关系。

## 旧名称时期的封存发布 (v0.2.1)

v0.2.1 及其中的标识符是在旧名称下封存的历史证据，因此保持不变；其科学
含义统一由当前 GRAIN 理论解释。

- **V7 技术核心** — 由观测算子与参照群体索引的相对文本几何、冻结几何
  bundle、拒绝规则、多视图校准、不确定性协议与未来数据闸门。最高受支持
  声明是 `OPERATOR_INDEXED_RELATIVE_TEXT_GEOMETRY_WITHIN_DECLARED_DOMAIN`,
  不是人格、信度、临床、跨语言普适或市场预测声明。
- **方法** — `docs/THEORY.md`(瓜皮模型;三通道: choice / style / react)、
  `docs/RULEBOOK.md`(具约束力的实验设计规则,每条均可追溯到实测失败)、
  `docs/VALIDATION_PLAN.md`(P0–P5 证伪框架)。
- **完整算例** — `docs/WORKED_EXAMPLE_MANUAL.md`:在 PANDORA(Reddit)上
  经审计构建的 19 构念电池 + 12 选择轴(开发层锚定性能示例:MBTI
  thinking-feeling ridge CV r = 0.346;Essays 大五迁移平均 r ~ 0.144),
  以及分数解读规则。
- **审计台账** — `docs/CLAIMS_LEDGER.md`:每条主张的状态、七轮对抗性
  审计、包含撤回。台账是权威记录;行文不得超出台账评级。
- **AI 操作标准** — `docs/AI_ANALYST_GUIDE.md`:角色分离(scorer /
  builder / coder / auditor / interpreter / human)、固定提示词、
  护栏 G1-G11(每条均可追溯到一次被实际捕获的失败)。
- **封存的预注册** — `docs/PREREGISTRATION.md`:封印即本仓库初始提交哈希。
  **开箱 #1 已于 2026-07-07 执行**:预注册成功规则**未通过**(BH-FDR
  q<.05 下 2/7,规则要求 ≥4/7)——全部结果原样记录于
  `reports/suica_lockbox_opening_1.md`。H2(第一人称 → 神经质,r=+0.111,
  q=.002)与 H6(政治/新闻选择 → 开放性,r=+0.096,q=.006)在锁箱层级
  获得确认。剩余 1 次开箱。Essays 确认半集的**标签**仍未开封(但其
  **正文**已在 2026-07-29..30 的 V8 真实文本实验中被完整读取,不再是
  "正文未接触"的保留集 — 见 `docs/V8_PUSH_REMEDIATION_20260803.md`
  第 2 项)。
- **代码** — 新代码使用 `grain_core/` 与 `grain_sim/`；`suica_core/`、
  `suica_sim/` 和 `run_suica_*` 仅作为冻结实验的兼容标识保留。无需受限
  语料即可验证 v0.2.1 锁箱。

## 封存发布之后 main 的现状 (2026-08-03)

> **更新 (2026-08-10) — 同一性时代。** main 现已搭载同一性理论(IDT,
> T1–T10 及附录 A–V:`docs/SUICA_IDENTITY_THEORY_V1.md`)、两条实证线
> (M4-K 十二腿 / M4-L 三腿)、完整走完的防御阶段(前瞻封印→对抗验证
> 零驳倒→可携带工件锁箱→真实文本治理 R-G1..G8→规则重放→**先测量后开封,
> 命中 3/5**),以及修复后的水平定律。η 地板定律在从未见过的维度 3/3 命中、
> taxometer 以 0.0042 误差读取新单元(两者晋升为预测级)。F2/F4/F5 的归因
> 已用带日期的注记更正。整合地图见 `docs/SUICA_V8_IDT_INTEGRATION.md`。
> 常设规则 1–24;缺陷登记簿 #1–#42。一切均为 EXPLORATORY、合成、
> 仪器-世界定律——绝无关于个人的断言。

v0.2.1 封存不变(CI 在每次 push 时于独立 worktree 中校验标签树),
main 在三个层面向前推进:

- **V8 统一理论体系** — `docs/SUICA_UNIFIED_THEORY_SYSTEM_V8.md`:
  类型化测量链 (H,F)→B→X→Z→𝔄→𝔐→Θ→D,三种输出类型 V(个体向量)/
  R(关系)/ P(群体场)之间禁止隐式转换;M3 微观(事件机制)–中观
  (复制关系几何)–宏观(群体关系场)层;M4 条件流形、机会生态与组合
  语法层。导航图见 `docs/SUICA_THEORY_ROUTE_INDEX.md`;理论论文草稿见
  `docs/V8_THEORY_PAPER.md`;实践手册见 `docs/V8_MANUAL.md`;开发史
  (日文)见 `docs/V8_DEVELOPMENT_REPORT_JA.md`。
- **M4-D/E/F 实验线(已收束)** — 16 个预注册实验把复合环路迁移之墙
  分解为四部分:路径误认(两阶段构造修复,翻转 196→73,合并几何
  .6519→.7605)、估计器自伤(岭罚偏差;λ~1/n 恢复教科书式 n^(−1/2)
  标度)、成对指标的共模性质,以及发现目标函数中分散的、随世界而异的
  公共偏移(正式登记的未解决问题)。类型化 R→V 桥带有 200/200 设计
  空值拒绝的完整记录。D3 面板设计定律:仅靠规模的挽救需要约 10^14 倍
  事件预算,不可行 — 必须改变面板的**构成**。台账行 M4-D.1..M4-F1。
- **治理** — V8 落地在公开前经过对抗性评审,全部发现及其处置记录于
  `docs/V8_PUSH_REMEDIATION_20260803.md`。特别地:PANDORA 跨量表桥接的
  头条数值(element r=.498)为 **POST_HOC_OPERATOR_SELECTED_EXPLORATORY**
  — 引用时必须同时给出同日先行运行的空结果(r=−.103)与选择顺序。
  v0.3.0 将为 V8 数值加装锁箱。

## 快速开始(无需数据)

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_suica.py          # 39 个通过
python -m pytest -q -p no:cacheprovider          # 970 个通过 (2026-08-03; v0.2.1 标签树为 318)
python scripts/verify_grain_v021_lockbox.py       # 便携式 v0.2.1 封存校验
python scripts/run_grain_synthetic_ground_truth_v2.py   # P0:估计器
python scripts/run_grain_p0b_thin_cell_regime_v3.py     # P0-B:薄单元
```

这些合成装置在不使用任何真实数据的情况下,以植入真值验证整个估计层。
复现完整算例请按 `docs/DATA_ACCESS.md` 获取数据(本仓库不含任何用户
文本或用户 ID)。`results/` 下的运行工件有意不入库;依赖这些工件的锁定
测试在干净 checkout 中会带理由跳过(对入库文件的哈希检查始终执行)。

## 证据状态的诚实总结

已在留出级(T3)确认:选择轴稳定性(5/5 轴,收缩 0.027)、15/15 个被发现
构念在未见用户上的确认、陌生人零假设下 2 个构念的 react 签名。已证伪并
退役:条件均值中心化(三种独立方式)、情感词频作为特质或场合状态的测量、
注意力权重作为测量证据。验证性(T4,开箱 #1,2026-07-07):预注册成功
规则未通过(2/7)——仅第一人称 → 神经质与政治/新闻选择 → 开放性两条
关系获锁箱级确认。适用范围:英语、单一平台 + 学生作文。

V8 时代新增事实(台账永远是正典):

- 真实文本的直接预测以空结果为诚实答案:V8 不变分数对大五的直接预测
  r=−.005,低于冗余变量对照。所有亮眼数值都必须携带其评级
  (EXPLORATORY / POST-HOC 等)。
- 作者识别 AUC 不等于人格效度——已实证:在个体结构为零的合成世界中
  仍可得到作者 AUC .864。
- 拒绝是功能而非缺陷:类型系统、比较许可与支撑审计会把"不允许的比较"
  作为异常抛出;R→V 桥在设计空值下误接受 0/200。

## 来源

自私有开发仓库 `project persona` 的提交
`154822a`, `05be394`, `cad83d5`, `c27727b`, `1c417fa`, `8447541`,
`5189168`, `b9f65a6`, `0650936`, `5485a02`
(+ 记录于 `docs/FREEZE_NOTES.md` 的冻结提交)冻结而来。在指南所记录的
builder/auditor 协议下以 AI 辅助研究方式构建;多数审计轮次捕获并修正了
真实的伪影——审计记录本身即是方法的一部分。V8 时代的开发史与教训见
`docs/V8_DEVELOPMENT_REPORT_JA.md`(日文)。
