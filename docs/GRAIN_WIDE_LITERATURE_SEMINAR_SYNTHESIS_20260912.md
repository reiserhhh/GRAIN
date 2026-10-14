# GRAIN 全体系广域文献研讨记录

日期：2026-09-12。目的：重建先行理论框架、判断当前贡献、找到已经走到我们下一步的研究，
据此修订领域级研究目标。不是新实验结果或发表保证。

## 实际协作

- Codex：读当前体系与开发史；独立检索心理测量、文本、因子/多视图、动力系统、
  模型解释和构造方法；逐项整理数学接口，核对关键原文与部分书目信息。
- Claude Code：`claude-fable-5-1`，`effort=max`；从有项目上下文的会话fork，
  使用只读工具独立检索全体系，重点补情境、选择、状态、测量制作和真实效标研究。
- round1/2允许Read、Glob、Grep、WebSearch、WebFetch；round3不提供工具。
  实际fork记录含WebSearch/WebFetch调用，不只是两模型凭记忆对话。
- 没有派发代码/训练任务；没有读取原始participant资料、密钥或锁箱；没有启动旧goal。

可复查输入：[R1](GRAIN_WIDE_LITERATURE_SEMINAR_R1_20260912.md)、
[R2](GRAIN_WIDE_LITERATURE_SEMINAR_R2_20260912.md)、
[R3](GRAIN_WIDE_LITERATURE_SEMINAR_R3_20260912.md)。
模型原始输出位于本地生成目录
`results/grain_vector_mechanisms/claude_wide_literature_seminar_20260912/round{1,2,3}.json`
及对应`round{1,2,3}_response.md`，均为成功返回；不将其中未经纠正的草案作为最终裁决。

## 三轮实际交换

| 轮次 | 输入与独立贡献 | 形成的决定 |
|---|---|---|
| R1 独立查找 | Claude补contextualized measures、contact/construal、random-item、state/trait、SIMPA、纵向NLP；Codex补semantic measurement、ECD、AI-GENIE、ACE、sJIVE、DSEM/SDE、JAM、concept bottleneck、残差记忆 | 基础部件已有大量直接前身；项目目标不能再等于部件集合 |
| R2 交叉反查 | Codex要求核查SIMPA/纵向NLP/情境研究，并反对用既有情境分类替代全部向量发现；Claude回读相关原文，也核对JAM监督路径 | 已知分类作锚点/对照；保留新结构；加法/交互作为竞争而非先验淘汰规则 |
| R3 最终纠正 | Codex指出共享空间不等于心理真值、MEPS不等距、selection IRT非外生、现有资料不是前瞻试验；Claude明确接受六项修正 | 不把统计分解当心理归属，不把构造示范当成两套成品量表；保留研究边界 |

## 一致结论

1. 保留用户终点：为情境化的文本人格、行为倾向与状态测量提供制作基础，而非只改进文本分类。
2. GRAIN本身仍可修订，slice、轴、层数、4W和路径接口不是不可改的公理。
3. 不再重复发明标准部件。用SIMPA/semantic projection对照文本解释，sJIVE对照共享结构，
   标准动态模型对照状态/响应，ECD对照构造逻辑，已有纵向NLP对照评价。
4. 保留选择与个人解释，不能因为某部分与情境相关就整体消除。
5. 理论增量必须是一项超出直接拼接的可检验机制/定量规律，不要求全部公式首创。
6. 心理效标可进入授权development；MEPS仍是validation-only，已查看的结果仍是回顾性证据。
7. 本轮提出的“机会/选择 + 学习到的证据 + 模型读者”三接口，是集中研究的问题描述，
   不是保证全领域此前没人处理过这三个条件。

## 已撤回/纠正的研讨草案

| 草案说法 | 最终采用 |
|---|---|
| 用DIAMONDS替换学习到的情境 | 作为理论锚点与基线；保留未分类向量证据 |
| 某ESM支持加法，所以交互应很弱 | 该研究提供竞争模型，不决定所有语言与时标的交互大小 |
| 情境化量表 .11/.24 可当MCD门槛 | 仅方向前例；不同效标、协议且原作有勘误，不能迁移数值阈值 |
| shared = cue validity；specific = reader bias/utilization | 二者均可有心理信号及伪迹，需通过独立操作判定 |
| sJIVE需要独立噪声多视图 | 那是借自另一类模型的解释限制，不能错贴为sJIVE拟合必要条件 |
| MEPS轮次等距；ME-TSO等于连续时间 | 两者均不成立；按观测时标选择模型 |
| 记录情境天然外生，文本情境纯属construal | 都需按来源和生成机制判断；来源不等于外生性证明 |
| 构造两套正式量表 | 构造示范与通用制作规则，不扩大为当前临床量表开发 |

## Codex保留的一项限制

Claude最终建议推导“无目标validation结果的效度上界及设计增益方向”。这可以作为候选，
但不是无条件可达到的终点：只有无标签文本时，相同文本分布可与不同心理目标联合分布兼容，
无法单独确定其效度。必须有授权开发效标、构念锚点或可检验的校准假设。
此外，selection IRT本来就能处理内生选择，ECD/adaptive design也并非所有变体都限于固定题。
我们不能靠把前人假设故意收窄来制造新颖性。

因此最终裁决是：先把最接近模型写进同一符号系统，找出一个真实的不可区分情形，
说明额外观察/机制怎样解决它，再形成一个有条件、非空洞的定量预测。
若标准组合已解决，就直接采用；不为了新颖性增加形式复杂度。

## 最终交付与下一步

- [全体系综述](GRAIN_WHOLE_SYSTEM_LITERATURE_REVIEW_20260912.md)：已有理论、公式对齐、贡献与研究终点。
- [27项机制矩阵](GRAIN_MECHANISM_PREDECESSOR_MATRIX_20260912.md)：每个角落的前身及采用/适配/研究路由。
- [母登记簿](GRAIN_LITERATURE_REGISTER.md)与[65项补充](GRAIN_LITERATURE_REGISTER_WIDE_20260912.md)：稳定出处及阅读深度。

下一项理论工作只有一个：完成最接近前人的共同模型比较，明确GRAIN要增加的
最小机制及其一个设计预测。用户批准实验后才开始有界机制模拟，不恢复过去长跑。

## 阅读限制

Claude遇到PubMed cookie、Wiley 403、PMC验证码及PDF提取失败，使用原始记录/摘要降级；
Codex也遇到部分出版商限制。A/I/F逐条登记，不声称130项都被两人精读。
没有安装、运行或锁定本轮定位的外部代码；许可证/版本未核处均标明。
目录覆盖改善了，但不是“不可能再找到更近的论文”的证明。
