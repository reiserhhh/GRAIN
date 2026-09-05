# VM05 强稀疏MBTI参照执行补充 V1

2026-09-06。补完SAME_EVIDENCE_DESIGN中既定word/char参照，非新增结果择优方法。
本批在任何该通道MBTI成绩出现前冻结，不修改运行中的source控制/null/CUDA代码。

## 设计

- 输入同一common_views、每人相同36评论，MBTI-only全部6393共同作者；
  EI/SN/TF/JP各按自己的官方repeat0五折，inner作者3-fold seed731。
- raw/cue_mask/placebo_mask都跑，四轴乘三臂共12条件。
- word(1,2)+char(2,5) TF-IDF，各max_features10000、min_df2、sublinear_tf，
  所有词表训练折内fit；Logistic C=[.01,.1,1,10]内层AUC选参；
  不做PCA、不因稀疏维数高改为16维、不用已见测试成绩选C或阈值。
- 主批使用完整共同来源，与原6393人词法及待完成的observer主表配对。
  旧参照排除敏感性仍用同6294作者；现有模型只限制OOF到6294的比较和
  真正排除99人重新训练的比较必须分别命名，不能混为一项。
- 用户级OOF、各轴AUC/PR-AUC/BA/Brier、固定.5分类及逐fold结果全保留。
  生成full-source head封存供后续独立bridge，但仍不打开bridge标签。
- 这是同证据参照，不是PANDORA原论文全数据/全特征复现。

## 执行与资源

入口scripts/run_grain_vector_mechanism_sparse_baselines.py复用未变的
run_grain_vector_mechanism_external_links.py。不启动语言模型训练或新embedding。
Mac24GiB内存；统计头CPU线程4，子进程RSS上限8GiB、默认整批7200秒。
每10秒记录自有child PID、RSS、elapsed；达到资源限制只终止本次自有进程，
保留部分输出，禁止影响其它研究或交易服务。无psutil也能用系统ps采样RSS。
输出目录用独占文件锁防止双启动。

材料、代码、12条件及资源规则写入batch_manifest；完成条件独立load_oof复算，
显式--resume-completed只能跳过已完整验证的条件，不能把半成品默认为完成。
预算不足1800秒不再启动下一条件；单条件不能完成时保留并诊断，不缩小数据
冒充完成、不替换同材料对照。全12条件结束前只报告部分进展。

Canonical目标：results/grain_vector_mechanisms/vm05_sparse_mbti_r2。
r1在任何拟合前因ExFAT自动产生的._.execution.lock被误判旧结果而停止，
保留PRELAUNCH_FILESYSTEM_GUARD_FAILED记录。937f914只修正新队列的
AppleDouble识别，未改变词表/统计头/数据；r2已重新启动并记录独占PID/RSS。
控制/null批次及CUDA队列各自拥有的运行代码保持冻结。强基线较弱或较强都完整
报告；其目的在于避免“胜过薄弱旧投影”被偷换成“LLM拥有独特推论机制”。
