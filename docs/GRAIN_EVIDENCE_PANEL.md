# GRAIN 观测与分析面板 v1 / Evidence Panel

用同一个入口接入自由选题、独立写作、对话、评分器坐标和有动作日志的任务。面板不会拟合因子或改动 GRAIN 的冻结评分。它在下游把三件事分开：参与证据、数值证据和有分母的行动证据。

## 快速运行

在 GRAIN 仓库根目录执行，无需模型下载或联网：

```sh
python -m examples.evidence_panel examples/evidence_panel_demo.json /tmp/grain-evidence.json
```

样例跨论坛、独立写作和任务：A 在论坛不可见，但在写作和任务里有记录；其中一篇写作没有数值读数，仍保留参与证据。论坛的构成改变与留存者变化分别输出。任务中有三条资格明确的记录，其中一条结局未知，选择率区间为 `[1/3, 2/3]`；再纳入一条资格不明记录时为 `[1/4, 3/4]`。命令拒绝覆盖已有输出。

| 所需操作 | 调用和证据读法 |
| --- | --- |
| 人有没有出现在这里，别处有没有记录？ | `participation(window, actor, target, history=...)`：正记录不依赖覆盖声明；只有目标采集完整才判断该处缺席；若别处范围不完整也不作整个 feed 缺席的断言。按历史窗列出曾见去处，不把第一次记录当作从来没去过。 |
| 同一人换了位置，读数如何变？ | `context_shift(before, after, actor, origin, destination, channel)`：分别出具原处基线、去处基线和去处后测。拆成同一时段的跨场景差与同一去处的后续变化；缺一角时保留仍然可算的差值，未知项是 `null`。 |
| 群体变化来自同人变化、发言份额还是换人？ | `compare(before, after, context, channel, weighting=...)`：用已观测的可计量样本，沿用 G-4 分解，给出参照重叠和两端分母。`actor` 每人等权，`event` 按该通道实际有数值的事件加权。没有共同作者时仍有总变化，无同人项。 |
| 可做的时候，到底做了多少？ | `opportunity_card(window, context, measure, actor=...)`：区分已知有机会且未做、结局缺失、资格不明与明确无机会。输出已记录机会上的可行比例区间，不是置信区间，也不估计日志没抓到的机会。 |
| 原参与者后来被看到在哪里？ | `participation_flow(before, after, target, history=...)`：锁定前窗已见名单，统计其后续状态和可见去处。一个人可去多处，每处只计一次。没有完整目标覆盖的去处单独计数，不冒充已知迁出。 |
| 同一批人在两个菜单上的行动是否变化？ | `paired_opportunities(before, after, origin, destination, measure, channel=...)`：只对两端都记有明确机会的同人给出平均变化区间，另列两端和配对日志覆盖。可附两端都有同一坐标通道的子样本，给出该子样本的行动区间及平均坐标变化；并列报告不等于估计个体相关。 |

## 接入自己的数据

- 快速接入原始论坛、作文或对话：调用 `grain_core.evidence_text.native_text_evidence`，传入已冻结的 GRAIN atlas 与 `TextRecord`，返回 `channels` 和 `EvidenceEvent`。坐标通道名包含 atlas 摘要，避免把不同参照的坐标混在一起。冻结词表完全没覆盖时 `values=None`，保留确有文本而无数值的事实。保留旧 `locate` 接口与原来的出词表标记。
- 嵌入、文本因子、资源分配、人工编码、自定义指标均可直接传入 `EvidenceEvent`。`channels` 声明每个通道的坐标顺序。同一事件被两个读者编码时用不同 `id`，共享同一 `observation_id`；参与只累计一次。空间、单位或版本不同就分通道，不在入口自动对齐。每项分析可用同一份输入、不同查询来重放。
- `window`、`context`、`actor` 是调用方指定的字符串。`window` 可以是批次、访视或写作条件，不需要秒数；需按来源约定自行指定可比较窗口和无未来信息的历史。输入里的 `Coverage(window, complete, context)` 描述采集，不描述个人是否看见或可参与。省略 `context` 表示当前声明范围内的整个 feed；数据抓取中断时不填完整。`Opportunity` 是独立账本，只有日志确认有机会且记录结局时才填 `true` 或 `false`；发言和机会都不自动创建行为结果。

```python
from grain_core.evidence_panel import Coverage, EvidencePanel, Opportunity
from grain_core.evidence_text import TextRecord, native_text_evidence

# Supply your saved native atlas; do not fit it on placement texts.
channels, events = native_text_evidence(atlas, [
    TextRecord("e1", "visit-1", "person-a", "essay", "I checked the route twice."),
])
panel = EvidencePanel(channels, events, coverage=[Coverage("visit-1", True, "essay")],
    opportunities=[Opportunity("m1", "visit-1", "person-a", "task", "try-other", True, None)])
print(panel.participation("visit-1", "person-a", "essay"))
print(panel.opportunity_card("visit-1", "task", "try-other", actor="person-a"))
```

JSON 输入遵循 [evidence_panel.v1.schema.json](../schemas/evidence_panel.v1.schema.json)，标识为 `grain.evidence-panel.v1`；`results` 以原始查询 ID 为键。每项比较给出实际分母／坐标，`null` 不转换为 0。输入 ID 可连接调用方日志，公开导出前由调用者匿名化。查阅 Python 接口无需 JSON Schema 运行时；CLI 会验证 JSON 结构和应用约束。

## 大文件的逐行输入

```sh
python -m examples.evidence_panel examples/evidence_stream_demo.jsonl /tmp/grain-stream.json \
  --queries examples/evidence_stream_queries.json
```

逐行文件第一条非空记录为 `{"schema_version":"grain.evidence-stream.v1","channels":{...}}`。后续每行是带 `type` 的单条 `event`、`coverage` 或 `opportunity`，字段同上。CLI 逐行模式另输出 `diagnostics`，包括各通道实际收到和计量的记录数、去重事件数及未声明完整覆盖的场景窗。默认单行上限 1,048,576 个字符，不加载全量行对象；用于累计和身份去重的索引仍占内存。`grain_core.evidence_stream.read_panel` 接收文本流，`text_csv` 支持声明列名；`iter_native_text_evidence` 每批最多 512 条并只编译一次冻结 atlas。不同资料若没做可信身份连接，应让 `actor`、事件 ID 和别名在各自的命名空间内。可用 `run_queries` 在同一个索引上反复执行不同查询。

### 多来源合并

将每份来源用 `namespace_source("source-a", channels, records)` 包装，再把各份返回值传给 `combine_sources(*sources)`。本地事件、场景、窗口与人员编号默认隔离，一个来源宣告采集完整不影响另一个来源。只对有可信对应的人员传 `actor_aliases`；要跨来源比较同一通道或行为指标，再用 `shared_channels`/`shared_measures` 显式声明。`combine_sources` 检查共享通道的名称及坐标顺序；坐标的测量意义是否相同由调用者确认。此接口不会凭文字或编号相似自动关联，不会自动跨来源去重同一真实事件。

## 性能与兼容性

共用入口只索引一次输入；重复查询不扫描原始事件，坐标对比只读指定的场景和通道。旧 `compute_selective_silence_panel` 在保持输入、输出及窗口定义不变的前提下，以稀疏的场景—作者索引替代逐场景重扫和桥接作者与所有场景的笛卡尔积。实例命令：

```sh
python -m examples.benchmark_evidence_panel d0e924f --repeat 5
```

[机器实测](../reports/evidence_panel_benchmark.json)是 Mac ARM64 单线程、两种固定稀疏负载，与基线逐项核对。2,400 条事件的中位时间从 22.4 降到 15.5 ms（1.44 倍），24,000 条从 573.0 降到 160.4 ms（3.57 倍），两组的最大数值差均为 0。加速属于该算子，不是整个 GRAIN、神经推理或 GPU 加速声明。

[真实载荷复算](../reports/evidence_real_replay.json)使用现有论坛汇总、Essays 和按键任务记录。相同结果下，逐行入口峰值约 52 MiB，整包约 77 MiB；逐行为 123 ms，整包为 89 ms（读取、索引与查询，不含生成基准输入）。小输入优先按方便选；当原始对象无法放进内存，再用逐行入口。两种都不是磁盘索引。
