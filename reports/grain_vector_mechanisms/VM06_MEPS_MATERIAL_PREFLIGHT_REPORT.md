# VM06：MEPS材料、说话者与完整输入审计

2026-09-06。**材料准备完成；模型机制复测尚未完成。**
用途是既有46人同场次、不同条件文本的无标签机制复测准备，不是训练、
外部心理效度检验，也不是从一场对话推断跨日人格稳定性。

## 已验证的材料结构

| 观察条件 | 有用户文本的人数 | 用户字符 | AI字符 | AI字符占比 |
| --- | ---: | ---: | ---: | ---: |
| 自由相谈 | 46 | 21417 | 161666 | 88.30% |
| MEPS Q1辅助对话 | 27 | 7874 | 57666 | 87.99% |
| MEPS Q2辅助对话 | 27 | 4998 | 37047 | 88.11% |
| MEPS Q3辅助对话 | 26 | 5164 | 40025 | 88.57% |

另有每人三个直接回答，共138个。总264个条件视图，其中126个聊天视图。
3个辅助聊天视图没有AI回答，不能补造assistant-only数据。字符统计含同角色
文本之间的换行，与历史strip/join计数略不同；不是参与者或来源发生变化。
AI字符占比是输入归属警报，**不等于已经量出AI对embedding的88%贡献**。

## tokenizer与上下文覆盖

对所有完整输入离线计数，未截断、未推断；三个模型共1539个输入条件。
revision和文件SHA均在manifest中固定。

| 观察器tokenizer | 三问直接回答均完整的人数 | 再加自由用户文本仍完整 | 完整自由对话可容纳 |
| --- | ---: | ---: | ---: |
| BGE-large-en-v1.5 | 44/46 | 34/46 | 0/46 |
| E5-large-v2 | 44/46 | 34/46 | 0/46 |
| multilingual-E5-large | 46/46 | 44/46 | 0/46 |

英文BGE/E5在日语直接回答中的UNK比例约20.2%至23.3%；多语E5三问为0，
自由用户文本仅1个UNK。分母排除特殊token，但包含前缀/角色标签token。
这说明输入覆盖差异，不证明多语语义等价或某模型具有人格推理能力。
完整自由对话多语E5中位2021、最多4367 tokens；不能512截断后宣称完整会话。

已实现逐说话者、逐轮无损分段：1600原始文本单元变1609窗口，每窗<=512。
366385原始字符全部覆盖，无重叠、无遗漏、无跨说话者合并；独立审计通过。
这是另一种观察设计，**不等价于完整上下文forward**。原文/token/embedding
均未写出，本轮模型forward次数为0。

## 本轮发现并修复的问题

1. 旧通用loader会读mood.csv，即使调用方丢弃`_mood`。现在默认不读，
   固定profile显式不读；历史报告添加勘误，旧数值未重写。新路线只允许
   三类文本/结构CSV，意外新增字段在读取行值前拒绝。混合Markdown容器
   仍需扫描/作哈希，但只保留回答段，不能称其字节从未被访问。
2. 所有1462条聊天时间实际为`HH:MM:SS.mmm`，不是完整ISO日期。r1将其
   列为未解析；r2正确标为clock-only；r3补字段白名单。未发现顺序下降，
   不补日期/时区、不据此构建跨日变化。另有5条同角色同文本重复、3条字符
   数与UTF16计数不符，均保留并标记，未静默删改。
3. 以完整AI对话为作者文本、直接截断长会话、或把英语tokenizer直接套日语，
   都会改变测量对象。当前修复提供可核对材料，不是心理测量能力提升的结果。

## 证据与复现

canonical私有目录位于`results/grain_vector_mechanisms/`：

- `vm06_meps_preflight_r3`，identity `2a1a13248f7bc74984ad1e4eab1f72aa80fd53993f4f26d146530bfa8121bebb`。
- `vm06_meps_windows_r3`：完整字符与角色分段。
- `vm06_meps_material_audit_r1`：输出SHA、模型间同文本、角色/跨度守恒复算。
- r1/r2保留为过程记录。r1至r3来源SHA、五张token/角色/覆盖表完全一致，
  r1与r3三张window表也相同。中间windows_r2因错误本地输入路径在任何数据
  读取/输出前退出；未冒充成功复测，canonical不指向它。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
DATA='/Volumes/mobile3/projects/project persona/data_sets/MEPS+AI_conv_experiment'
"$PY" scripts/run_grain_vector_mechanism_meps_preflight.py --input-root "$DATA" --output-dir "$R/new_meps_preflight"
"$PY" scripts/run_grain_vector_mechanism_meps_windows.py --input-root "$DATA" --preflight-dir "$R/new_meps_preflight" --output-dir "$R/new_meps_windows"
"$PY" scripts/audit_grain_meps_material.py --preflight-dir "$R/new_meps_preflight" --windows-dir "$R/new_meps_windows" --output-dir "$R/new_meps_audit"
```

## 下一研究问题

来源侧完整观察器/头和规则仍须先完成并封存。MEPS可独立进行不选心理方向
的材料/角色算子诊断，但不能提前称冻结机制已经迁移。下一实验分别检验：
人类文本本身的跨条件对应；AI回应的回声/内容对应；相同人类文本受到
真实/打乱AI上下文时的内部表征变化。角色池化与模型上下文计算必须分开。
同人相似并不排除共享提示、长度、AI辅助和写作习惯，不能直接命名为人格。
