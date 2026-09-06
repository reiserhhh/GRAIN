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
