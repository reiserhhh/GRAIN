# 后台旧队列暂停与研究优先级裁决

核对时间：2026-09-07 19:11-19:13 JST。根据用户指示评估剩余计算的研究价值。

## 实时状态

| 队列 | 实际状态 | 完成量 | 进程与恢复点 |
| --- | --- | --- | --- |
| Windows native direct traits | PAUSED_BUDGET_BETWEEN_SHARDS | 47/128片，2679作者，52815真实forward | owner48952已退出，child为空，next_shard47 |
| Windows own/matched/none history | PAUSED_BUDGET_BETWEEN_SHARDS | 65/128片，3640作者 | owner628已退出，child为空，next_shard65 |
| Mac cached secondary readouts | PAUSED_BUDGET | 297/315项 | owner15034不存在，child为空，next_case为SN_cont_placebo_mask_final_norm_decision_disposition |

通过直接 SSH `huoke@192.168.1.20` 读取远端两份 queue_status.json 并检查
相关 Python 进程，列表为空。Mac ps 检查15034不存在。没有发送 kill、修改
数值结果、清除锁、启动后继或重新运行实验。

直接评分末片为046，57作者，1118forward，片内记录的算术误差3.55e-15。
远端最终状态写于18:49:29，launcher于18:49:30正常返回0。实际队列时间
26610.438秒，约7小时23分30秒。8小时预算并未耗尽到最后一秒，因为原代码
要求剩余时间至少容纳2400秒单片预算加30秒余量，才会启动下一片；这是正常
预算保护，不是卡住或执行失败。

## 对“再跑是否改变结论”的回答

**当前决定：旧三条队列保持暂停，不自动续跑。不是全量完成，也不是统计无效性
或等价性已经成立。** 后续数据可能改变某个效应估计或模型排名；我们没有做
序贯无效性检验，也没有读未完成 native 队列的真实标签来选择停止时机。

这个判断依据的是任务与科学问题的对应关系，以及已完成、独立验收的证据：

- 主比较45头/225保存fold已完成。相同624名Big5作者下，pooled mean Pearson
  为BGE .176915、E5 .202323、Qwen body .150330。它不支持这个冻结Qwen观察器
  的普通body聚合普遍优于embedding；不能推广成所有LLM都无优势。
- 54项投影/固定层敏感性已经完成；无需继续靠无限换层找一个赢家。
- 已完成90case语义检查：原提示17/18行为对正确，Big5反转计分编码后0/10，
  MBTI反向列举后7/8。固定答案分数混入编码影响；增加同类作者不隔离该机制。
- 已完成P1能改变人格归因也能影响无关判断，支持继续区分概念、角色与答案
  准备，不支持把所有跨任务干预响应直接称为人格通路。
- 297项缓存结果可保存为探索材料，但315项完整配对报告仍未生成。查看部分
  单头点估计不等于完成该族比较，不据此挑选最佳层/条件或宣布全部无增量。

继续旧队列的主要收益是填齐native性能比较、history对比和缓存读出表；它们
本身不实施角色/机会条件互换、局部功能扰动或路径替换。因此不应把完成它们
作为新机制工作的前置条件。剩余direct81片按本次吞吐粗估仍需约12.7小时，
只是资源量级估计，不是完成时间保证。

## 证据保留与未闭环范围

远端原始队列、已完成片、manifest、源码快照、失败历史全部保留。当前Mac
native目录只有1个完整shard；本地queue_status.json还写着旧RUNNING/2片，
不得用它覆盖或代替远端47片的最终状态。尚未完成其余46片本地备份和独立
逐片重验；本轮仅核对远端状态和进程，不把远端汇总称为新的本地完整验收。
History本地有65个片目录，此前完整独立验收记录保留，本轮不声称重验所有数组。

完整native与history源头评估仍未完成。原complete-only分析门槛不改、不绕过，
不因暂停而改成COMPLETE。桥接/MEPS心理标签未新读取。暂停的是执行优先级，
不是抹掉未回答的问题，也不影响已经完成的主比较。

## 下一步

1. 优先执行创新研讨的P1B材料与技术预检，然后在最多2000 actual forward内
   区分角色归属、同层不同位置、问法/编码及自由-强制条件。它不依赖旧队列补齐。
2. 资源允许时先做已产出47片的无神经计算备份/核验；不是重新采集或读取标签。
3. 缓存剩余18项、direct81片、history63片只在某个明确问题确需完整比较时
   单独重新排期；当前不恢复任何队列，不削弱原验收条件。

来源：远端D:/projects/GRAIN-vector-direct-vm05与GRAIN-vector-history-vm05下
同名results队列；本地vm05_secondary_cached_readouts_r1/batch_status.json；
vm05_direct_launch_round37_r1/launch.log与exit.json；
VM05_PRIMARY_NEURAL_SOURCE_REPORT.md、VM05_DIRECT_SEMANTICS_REPORT.md、
PERSONA_RECIPROCITY_P1_REPORT.md。新的P1B仍是计划，未在本轮执行。
