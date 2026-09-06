# VM05强稀疏基线：内存生命周期修复

2026-09-06。原`vm05_sparse_mbti_r2`在SN raw的五个外折已保存后、
全来源选参与拟合阶段触发8GiB/time保护。监督器终止子进程，确认PID33463
不存在；父状态为FAILED_STOP。子目录的RUNNING仅是最后一次外折快照，
不表示仍有计算。旧实现未保存失败时峰值，不能编造精确最大RSS。
五个fold模型与日志保留；因为没有全来源头和完整summary，不作完成验收。

发现`select_head`在下一个内折和最终fit期间仍保留上一个内折的
`fit_x/valid_x`及变换器。修复只在内折结束后释放这些对象并gc；不改
TF-IDF、样本、seed、C/alpha范围、inner/outer切分、精度或评分。
单元测试以weakref确认六个内折矩阵在全来源fit之前已释放。
独立合成复算旧14505a7函数：dense/word-char乘MBTI/Big5四条件，
全部inner分数、选择参数相同，预测最大差0。此为数值等价证据，不是
真实SN内存修复成功的替代，仍要完成真实任务才验收。

下一执行新目录`vm05_sparse_mbti_r3`，只选SN/TF/JP九条件，保留原EI三臂。
8GiB上限不提高，4小时一包，默认模型协议不变；所有新源码另行归档。
失败时父状态新增最后child观测与具体时长/RSS，避免再丢诊断。
发布scorer/Windows冻结aa51b06源码不修改。旧r2不覆盖或伪装续跑完成。

## 真实SN raw闭环（2026-09-06）

新r3的SN raw已完成5外折、全来源选参和保存完整head/summary，6393人，
监督采样峰值6,756,466,688 bytes，低于不变的8GiB保护。监督记录1597.9秒，
子任务summary1574.2秒（计时起点不同，不混用）。因此这一个条件的真实内存
修复已经闭环，不代表其余八条件或所有未来数据规模都已通过。

OOF AUC=.6698766，条件于已拟合OOF预测的95%bootstrap区间
[.648656,.691100]，mean-fold AUC=.6722909。0.5阈值下预测全N，
balanced accuracy=.5000，真实N比例=.8901924；排序与可用决策不是一回事。
不在OOF/bridge上事后寻找阈值，也不能用该区间替代尚未做的强稀疏null家族。

`vm05_sparse_sn_raw_replay_r1`对5个保存模型逐折独立重播，最大预测差
1.1102230246251565e-16；未refit、未读新来源标签/bridge结局。
`vm05_sparse_sn_raw_summary_r1`只验收一个条件并列出八个pending；
活跃队列已进入SN cue_mask，不重新启动或覆盖原失败目录。
