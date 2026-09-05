# VM05完整来源取材协议

2026-09-06。此阶段只完成取材，不运行心理关联，不证明官方可比性。

用现有prepared official表的user_id及official_fold（bridge为bridge_fold）
建立来源清单，读取列严格限于这两项；不读取心理分数或类别。
原始正文来自既有PANDORA all_comments_since_2015.csv。

## 必须固定的分组

strict bridge文件具有优先权，不等于官方Big5/MBTI名单交集。
Big5-only=Big5来源减去全部MBTI来源和bridge；MBTI-only反之；其他both
但不在strict bridge的成员另留reserved，不拿来补训练量。
各轴的official_fold分别保存，不能统一为四轴共同fold。

全量遍历所有官方来源作者，不再只抽240人；不按已有模型结果或分数选人。
跨列/文件的一对一ID检查及分组零重叠必须通过。相同语料曾多次研究，
此包是后续探索性外部联系材料，不是新未接触lockbox。

## 文本规则

沿用已测试的36-comment取材仪器，3个时间排序视图，每视图12评论；
英文、每评论40-100 whitespace words、author/body/id/link/time字段完整。
每人用同一hash规则保留最多192候选评论，再以固定hash选36个独立讨论、
非重复正文，按时间排序为3视图。全候选池的跨作者完全归一化重复文本
排除；这不能排除语义近重复。正文不改写，不mask，泄漏线索另记标志。
不够评论或不能严格分时段的作者记录失败原因，不截短、不换标准补人。

列出每类来源、每fold进入前/后的计数、评论数、泄漏标志、时间跨度和
旧机制240人的重合；该重合不作为排除条件，但必须披露。词数限制和
活跃作者条件改变了母体，不能把此子集结果叫完整原论文benchmark复现。

## 后续使用边界

输出只是源材料及fold链接。训练内TF-IDF/PCA/scaler、固定模型采集预算、
GRAIN观察算子适用域，以及量表计分/校准仍需在下一阶段单独冻结。
三视图32/36评论与单prompt1536tokens不是自动等预算比较；不能把12评论
的LLM和36评论的GRAIN放同表后声称能力差距。此处不改变任何已有计分器。
strict bridge只接受冻结表示变换和源头拟合头的后验评估，不参与挑层/方向。

## 验收与输出

一次有界流式扫描（上限600秒检查）；若不足标PARTIAL，不作为全量完成。
输出vm05_full_source_cohort_r1：manifest、eligibility.csv、source_counts.csv、
fold_counts.csv、各源author_views.json、data_inventory、source_snapshot。
所有正文及作者级引用留在ignored结果目录；公开仅计数、方法和聚合结论。
原文件大小/mtime扫描前后不变，源头/评论/讨论无交叉错误，严格分组测试通过。
