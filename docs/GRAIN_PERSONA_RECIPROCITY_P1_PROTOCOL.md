# Persona 生成／识别交叉干预 P1 执行协议

日期：2026-09-07。承接 [双向读写与群体机制](GRAIN_PERSONA_RECIPROCITY_AND_GROUP_MECHANISMS_V1.md)。
本协议执行的是第一块机制拼图，不是真人人格评估，也不是完整 Big Five 实验。

## 对象与问题

窄行为对象：在社交场景中主动发起交谈，还是等待他人发起。
G 是 persona 条件下的第一人称表达任务；R 是根据行为记录判断作者是否主动。
检验两种任务的激活对比方向能否跨任务影响输出，及该影响是否具备语义特异性。

G 使用自然行为候选整句似然，不生成自由文本后让另一个 LLM 评分。首轮不是
Persona Vectors 的完整复现，而是可重放的候选表达实验。G 的外向／内向提示
是操纵条件，不把内向必然等待、外向必然主动作为关于真人的定律。

## 固定设计

- 模型：本地已有 Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- 校验原始权重和六个 tokenizer／配置文件 SHA256；不联网下载，不装新依赖。
- 主面板：Mac MPS FP32 eager；两个固定层 `block_10`、`block_16`，最后 prompt token。
- 开发：六个场景 × G/R × 高低两条件，共 24 prompts。
- 评估：另六场景 × 八个变体 × 高低两条件，共 96 prompts；不用于拟合方向。
- 全部场景和模板由代码明示，都是合成材料，没有真人文本或量表字段。
- 每个 source/site 方向为开发 high-low 激活差的均值，不拟合标签头、不选最优层。
- 加性剂量 `-1, +0.5, +1`；两个固定种子的等范数随机方向，各测 `-1,+1`。
- 相同干预方向跨目标场景复用，不能按某个目标的结果调整随机方向或剂量。
- 最多 5000 forward、1800 秒、MPS 已分配内存 9 GiB；两 CPU threads。

smoke 只用一个开发场景、两个评估场景、一个层和一个随机方向。它用于执行检查，
不是独立确认；其两个评估场景后续会在完整面板中复用，应披露为已技术性暴露。
所有区间均探索性，不依赖“未查看 lockbox”的宣称。本面板不接触旧 lockbox。

## 八个接收任务

| 变体 | 目的 |
| --- | --- |
| G_trait | 根据 outgoing/extraverted 或 reserved/introverted persona 比较两种表达。 |
| G_behavior | 改用行为偏好定义，不出现 trait 名称。 |
| R_direct | 从作者的行为记录判断是否主动接触。 |
| R_negative | 反向提问，交换 Yes/No 的语义计分，检查通用答案偏向。 |
| R_trait | 根据相同记录判断作者是否显得更外向；与行为事实识别分开。 |
| R_trait_negative | 反向人格归因问法，并交换 Yes/No 的语义计分。 |
| R_owner | 两个人的相反行为同时出现，交换行为归属，保持行为词袋。 |
| R_fact | 相同材料只问场地是否开放，两条件都应 Yes；检查无关事实漂移。 |

四主格 G->G、G->R、R->G、R->R 分层报告；其余变体解释效应边界，不合并成单一
“人格准确率”。最后 prompt 位置可能已经编码答案准备，不能仅凭该位置效果称为
证据整合电路；若反向提问或事实控制异常，下一诊断应定位更早的证据使用位置。

R 方向从行为事实识别的开发记录提取；R_trait 只作为独立接收任务，不参与方向
拟合。它的预期符号是操纵方向，不代表单次发起行为就能确定真人的外向性。
技术 smoke r1 尚未包含这两个归因变体；确认执行链有效后、未检视效应结果前，
补入完整面板，以免把事实读取误称为人格推断。完整源码与材料另行冻结。

## 计算与验收

计分是 s=log p(高倾向候选)-log p(低倾向候选)，所有候选 token 都计入。R 单 token
两候选用一次共享 forward。G 另保存每 token 平均分数，不能挑对自己有利的版本；
高低条件下相同候选的配对差是原生能力检查，避免候选句先验直接决定结论。

干预：h'=h+alpha*v。主效应 slope=(s(+v)-s(-v))/2；与同 source/site 随机方向
对照配对，再按场景求均值与 95% bootstrap CI。区间是 pointwise、未校正多重
比较；六个主题名的变体并非六种完全独立语言模板，不能外推到所有作者。

技术验收要求权重／模板一致、有限值、self-patch/restore 最大差 <=1e-5；
失败或超限必须写 FAILED/PARTIAL，不自动扩大预算或删去失败条件。

科学结论分层：各自任务是否具备方向性、跨任务是否有作用、反向问法／归属／
无关事实是否支持语义特异性。程序只输出证据，不自动选择一个层宣称成功。
即使控制通过，也只是共享计算候选，不证明真人属性被准确推断。

## 重放与文件

在仓库根目录、使用含 torch/transformers 的既有 venv：

```bash
python -m pytest -q tests/test_grain_persona_reciprocity.py
python scripts/run_grain_persona_reciprocity.py --prepare-only --output-dir results/grain_vector_mechanisms/reciprocity_prepared_new
python scripts/run_grain_persona_reciprocity.py --smoke --output-dir results/grain_vector_mechanisms/reciprocity_smoke_new
python scripts/run_grain_persona_reciprocity.py --output-dir results/grain_vector_mechanisms/reciprocity_full_new
```

`--model-dir` 可显式指定已存在的同 hash 权重目录。已有输出不覆盖；新目录可完整
重放，不支持从部分结果中挑剩余条件继续。断点文件只用于审计、不是静默 resume。

每轮保存 manifest、材料、源码快照、模型文件 hash、开发方向、随机方向、逐样本
基线／干预／数值控制、完整矩阵汇总和 report。激活与权重不提交到 git。

`weighted_effect_summary` 提供作者已平均后的加权均值、协方差、同号比例与有效
人数；仅数学原语已经实现。群体机制检验需要另行准备独立 GRAIN 文本视图和权重，
本轮没有用合成“作者”冒充该验证，也没有打开 MEPS 或 bridge 标签。

后台既有任务的代码、环境、队列和配置均不修改；新模型面板在空闲本地 GPU 上
单独执行。CPU 缓存读出仍占用自身线程，新面板不接管其租约。
