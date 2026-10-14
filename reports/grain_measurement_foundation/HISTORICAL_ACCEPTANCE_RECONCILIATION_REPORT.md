# 早期 Big5 / H2 / MCD 计划的证据归档复核

2026-09-13，C00/C23续批。仅读取历史预测、矩阵、脚本与结果；不重训、不恢复
LoRA/MBTI队列、不访问MEPS、不把旧报告的PASS自动等同所有原始成功标准完成。

## 1. 这次保留下来的有效成果

**MCD成果1的数值仍然成立。** 从20,950条预测重算75个条件/trait/fold单元，
3条主线及10个官方预计算参照逐项复算，最大差1.67e-16。

| 方法 | 五折相关先取平均、再平均五trait |
|---|---:|
| 官方 LR-N | .245975209 |
| 官方 LR-NP | .292628374 |
| MCD-only | .231721039 |
| N + MCD | .298039320 |
| NP + MCD | .318089806 |

原始预测中每条件/trait的作者无跨test折重复，N/MCD为1402作者、NP/MCD为1386，
不是旧生成路线的1401。因此保留“条件选择偏离对该官方包有增量”的历史证据；
不把高维GRAIN或新来源模块的结果冒充这次MCD的重现。
这是计分复算，不是重新训练检验特征构造、所有许可、清洗或超参数搜索。

存档的2,000次bootstrap样本重算：NP+MCD配对pooled增量分布均值.023604159，
95%区间[.013005109,.034031250]；原配对pooled点差约.023608。
表内折均值差为.025461432，不是同一个统计量，不能把该CI挂在折均值差上。
存档没有反向样本（0/2000），不把有限抽样计数解释成真实概率严格为0。
N+MCD对应分布均值.058301138，区间[.042245953,.074361890]。

## 2. Big5：数值闭合，不是完整base对照闭合

1,401作者的五fold test集合互斥；五个raw summary可复算。
逐trait剔除超出[0,1]的预测后，五fold和pooled都精确匹配FINAL的四位小数：
pooled mean r=.100540962。

fold4 raw均值.070420922；clip后.123722163；逐trait剔除后.129742427。
只有一行的N/A/C/O四个预测越界，E不越界；不同规则造成旧差异，非训练损坏。
原`BIG5_RECONCILIATION_REPORT.md`的PASS可保留为**该clean规则下产物复现通过**。

但现存base-control摘要和预测都只含fold0。其240个可解析E预测中239个越界，
套同一clean规则后E仅剩1个样本，Pearson不可计算。故不允许把五折LoRA的
.1005与base单fold/raw的.0624直接相减当作H1提升证据；旧“完整五折对照”要求
仍开放。若未来要恢复生成路线H1，应先核原输出/计分标度，再统一五fold及覆盖，
不是简单新增训练。本研究loop不为此自动开启已退居历史的昂贵队列。

## 3. H2：数据边界正确，历版指标必须带协议

本地三个分区ID实查：Big5-only1037、MBTI-only8678、bridge364；各自唯一，
两两交集均0。此为保存的分区证据，不代替重新检查每个训练调用的使用情况。

| 历史数值 | 实际对象 / 口径 |
|---|---|
| .5015 balanced accuracy | 旧修正脚本把0/1 Ridge输出按`prediction > 0`分类 |
| .5939 balanced accuracy | 后来cross-framework脚本按评估批次预测均值分类 |
| .6087 AUC | BGE直接头的排序指标；上述阈值改变本身不改变AUC |
| .5621 element_r | 初版修正的4x4桥接矩阵、当时编码口径 |
| .539018 element_r | 后来正极编码、5x4的BGE直接头矩阵 |
| .758450 / .775678 | E5 / hybrid的5x4矩阵；同样由保存矩阵复算 |
| .5263, p=.1022 | 独立命题：同人配对对比错配的个体桥接检验，存档而非本次重跑 |

5条路径共100个矩阵单元算术复核通过；AUC/分类是保存的结果，并未从原始
逐人预测重新拟合。批次均值阈值使分类依赖同批样本，不是固定个体部署阈值；
.5015->.5939也不是新训练获得+.0924。

同一个旧JSON还保存两种legacy结构数值：同轮重建.1731，另一个历史快照.6963。
它们不是同一比较，应保留出处而非择一“修好”。所谓-.09引用更早.5915，
不等于这轮legacy .5151到corrected .5015的差，更不能单独归因于数据边界。
旧legacy函数按bridge_fold做OOF；违反“bridge不参与构造”的项目边界与
“同一用户同时进入同一fold训练和测试”的泄漏不是同一断言。

后来的hybrid报告将跨364作者的典型pair Spearman均值.1670命名为same-user
coherence；该计算依旧是跨作者相关，并非个人内部相容性检验。它不推翻
.5263 / p=.1022的个体配对结果。DDF/LDF桥接路径另外标记，不替代直接文本头。

## 4. 对全部计划闭环的处置

- C23 Big5产物差异：**算术问题已闭合**；完整base-control成功标准仍缺，不重训。
- C23 H2数据分区：**保存分区已核**；阈值/维度/编码/比较快照已解释并分列。
  跨批固定阈值、逐人预测冷重建及hybrid选权的外部独立性仍需各自核清。
- C23 MCD成果1：**原预测与结果复算通过**；清洗、组件消融、计数/subreddit控制、
  Neuroticism反例、原生冷构造/迁移需继续查后续已有实验，不重新宣布全部未做。
- C00仍有历史计划尚未逐条验收；本次仅结束上述明确要求。下一步查MCD后续
  组件/层级实验的证据继承，连接C24到现有“条件保留、选择与响应分开”的制作理论。

不能把所有旧计划重跑一遍当作理论闭环；应保留问题、找准证据、复算成立范围，
对已被后续协议替代的实验记录替代原因，真实缺口才进入新实验。

## 5. 产物与重跑

`results/grain_vector_mechanisms/all_plan_closure_v1/historical_acceptance_r2/`：
`manifest.json`锁37个输入及脚本hash，`summary.json`区分各项完成范围。
七张CSV含Big5三规则、MCD75单元/三臂/官方参照/存档bootstrap、H2矩阵复算。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/reconcile_grain_legacy_acceptance.py --output results/grain_vector_mechanisms/all_plan_closure_v1/historical_acceptance_replay
"$PY" -m pytest -q tests/test_grain_legacy_acceptance.py
```

每次使用新目录，不覆盖原实验。r1发现Mac的AppleDouble旁文件被列入输出清单，
r2仅修复清单过滤；7张科学CSV及summary与r1逐字节一致，见`replay_tables.json`。
零模型调用、零训练，4个新单测及70项相关测试集合通过。
总goal ACTIVE，无自动commit/push或锁箱操作。
