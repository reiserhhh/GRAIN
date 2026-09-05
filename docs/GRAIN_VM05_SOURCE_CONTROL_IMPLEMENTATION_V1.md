# VM05 来源控制与完整置换实现补充 V1

2026-09-06。承接已冻结的 FULL_RUN_AND_CONTROLS_V1 第4/5/8项。
本补充在102条件批次与199次经验null启动前记录实现选择，不新增按结果挑选的头、
切分、维度或模型。Windows现有采集代码保持不动；bridge心理结果不读取。

## 来源控制

- MBTI四轴分别使用官方五折，剔除旧冻结参照中的99位作者；共同来源从6393
  变为6294。这里只排除旧参照，不把余下作者称为新lockbox；旧E1暴露另列。
- 四轴各跑 raw/cue_mask/placebo_mask，每臂 retain、purge，以及
  random_count 的731..735五个seed，共84条件。
- Big5-only没有这99人的重叠；word+char参照三臂各做purge与五seed等人数
  随机删作者，共18条件。原624人retain结果保留为参照，合计102个新条件。
- purge从训练集剔除与当前held-out任意讨论串相交的整个作者。测试作者/
  评论不改变；内层调参独立重复。数据不满足最少20训练作者则失败并保留。
- random_count不只是外层同人数：每个内层折也与purge准确匹配训练人数。
  为此用真实purge外层作者池作只读参考，按相同KFold索引计算其内层可留
  人数；随机外层池再按该人数删除。不使用心理值决定任何删除。
- full-source头同样对bridge的讨论ID做purge或等人数随机控制；这只看文本
  来源ID，绝不读bridge心理分数。训练人数及剩余共享讨论作者数逐层输出。
- 不把purge与随机删除的差称为纯因果效应：作者组成及语境分布也会变。
  排除旧参照前后的比较先将原OOF限制到同6294作者，再计算配对差。

## 标签盲缓存与置换

冻结词法算子的cue/placebo特征仅复用作者原文hash、算子hash、源代码hash、
特征名和顺序均一致的缓存。不是全样本新拟合SVD/PCA。

199次author-label permutation：seed=731+j，j=0..198。Big5一次打乱完整
五维标签向量，保留量表间相关；每个MBTI轴保持同源作者顺序。每次重新拟合
全部候选回归/分类头、重作内层选择，再生成五折OOF。只缓存独立训练折内的
不看标签词表/scaler/PCA，以避免重复相同计算。稀疏和密集、回归和二分类
均用synthetic data核对与原嵌套流程逐值一致；真实observed OOF最大差须<=1e-10。

第一20次的均值若Big5>.10或AUC>.55，作为正null异常警报停止，不作为科学成功
门槛；不能直接resume绕过。其余运行到199次，报告经验上尾
p=(1+null>=observed的次数)/200，最小可分辨p=.005。条件不同的检验分别列出，
不能把多项最小p当作确认性结论；后续完整任务家族统一多重比较解释。

初始null针对未purge的同一来源关联，不是条件独立或讨论串因果检验。
只读取已完成源头OOF中存下的标签，不再打开其他心理标签。
observed和每次permutation的预测、参数和SHA均保留，可独立复算及显式断点继续。

## 执行与验收

入口：
- scripts/run_grain_vector_mechanism_source_controls.py
- scripts/run_grain_vector_mechanism_external_null.py
- scripts/run_grain_vector_mechanism_external_links.py（新增控制参数）

默认单批7200秒，最大28800秒；串行条件，CPU线程4。预算不足在条件/置换边界
暂停。失败或孤立输出不覆盖；查因后另建修正目录。每个输出目录只允许一个
执行者，恢复时使用显式resume标志并核对相同身份。

验收按实际产物：全102条件完成才称来源控制批次完成；完整199才称该null完成。
核对每个purge外/内/bridge-source无共享训练作者，随机删除逐折人数准确匹配，
逐人OOF与summary一致。报告保留pooled和mean-fold、BA/校准/类别比例；
条件于已拟合OOF的bootstrap不得表述为全流程重拟合不确定性。

本轮只改善研究判别力，不修改GRAIN发布分数，不训练任何语言模型，
不把旧词法中间特征等同于GRAIN整体，也不因单一TF轴较强提前开bridge。
