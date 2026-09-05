# VM05 全量运行与对照冻结补充 V1

2026-09-06。承接 GRAIN_VM05_SAME_EVIDENCE_DESIGN_V1；本页先于心理标签读取/拟合头。
不改变 GRAIN 发布评分器，不读取 MEPS 心理值，不训练/下载语言模型。

## 已完成的前置与范围

- vm05_preflight_r1：7,343来源作者中7,266完整适用；624 Big5-only、
  6,393 MBTI-only、249 bridge。各人36评论、三个12评论时间视图。
- common material hash:
  f01dbe318e8b4cdf2b0546e1853316f04c16543d693877b0b2efd449eff8f316。
- Qwen/BGE/E5各12人技术检查通过；288项跨机token签名完全相同；
  Qwen重播/前缀不变性最大误差0，encoder重播在1e-6内。
- 原缓存预计11.582GB超过10GB，禁止直接全量启动。改为相同数组无损别名，
  每个字段恢复逐值验证通过，预计6.462GB。不是降精度/减少材料。
- 原估计全量12.46 GPU小时只是12人线性预算估计，不能当实际耗时。
  固定24个hash分片、每片三个观察器串行；单次队列最多8小时，每观察器
  分片最多30分钟，统一10GB cache上限。八小时后在分片边界暂停，显式
  继续已验收分片；失败或不完整分片停下保留，不自动重跑覆盖。
- 不同时加载多个GPU模型。训练/重计算用Windows .20现有CUDA及缓存；
  不停交易服务、不重配Linux .30、不调用付费API。

## 对照材料与解释

raw保留当前来源原文，但源库已经存在TYPE_MENTION，不能宣称未经上游处理。
cue_mask按冻结正则遮蔽命中词；placebo_mask遮蔽同评论等量非命中词，
选词seed固定、位置不重叠、完整评论数/词数不变。三个版本都完整进模型。

parent 11,395命中评论：8,944含TYPE_MENTION，2,585含其他正则线索，
两者重叠134。标记不携带原16型编码，但可能携带自报/社群选择信息。
本消融仅检验这组词/标记依赖，不保证删除全部人格报告或全部泄漏。
不能把它的正效应或负效应直接解释为全部人格信息。

## 在标签前冻结的分析规则

1. 正式源头/held-out作者按官方各任务fold；inner 3-fold seed731。
   主层固定block10正文mean/std，block04/16/final为敏感性，不挑外层最好层。
   三个12评论视图合为作者，绝不当作三个被试。正文状态没有读取后置问题。
2. BGE/E5以36条comment向量的mean/std聚合；Qwen以3组正文池的均值及
   方差合成全部正文token的mean/std，权重来自实际正文token数。
   不把二者权重差异隐瞒为纯架构差；补充Qwen等view权重作为固定敏感性。
   任务末位与两问差是次级特征，不掺入正文主候选。
3. TF-IDF word(1,2)/char(2,5)各max_features=10000，min_df=2、
   sublinear_tf=True；词表在训练折内fit。稀疏联合Ridge/Logistic头作为参照，
   不用稀疏PCA强迫变为16维。密集PCA/scaler与超参按原设计在inner重新fit。
4. 冻结GRAIN正式profile仅在all36 ready作者报告；拒绝者不强行计分。
   所有方法在相同ready子集重新作配对比较，另报完整共同总体中间特征头。
   99位旧参照作者均在MBTI-only；主结果公开既有观察器重用，
   加做排除旧参照成员的同折敏感性；旧E1共168位暴露单列，不称新lockbox。
5. shared-thread主分析保留原36评论；预定敏感性删除与test任何thread
   相交的整个train作者，不修改test或其文量，也不重用其标签。
   同时按相同train作者数作5个固定seed(731..735)随机删作者对照；
   删除只依赖材料ID，不依赖标签/结果。inner fold也执行相应条件排除。
   同一人数不保证相同作者分布，因此称“讨论来源敏感性”，不宣称纯因果效应。
6. 源头outer OOF完成、头与参数规则封存后才读取bridge标签。bridge绝不
   训练两个主头或选择表示；实际Big5输入的DDF仅作后验结构参照。
   沿用各来源模型全量fit后进入249 bridge；结构CI按作者抽样，保留20个格子
   而非把20格当独立被试。结构较高不意味着个体准确。
7. 主比较block10 vs BGE按Big5 mean r与MBTI mean AUC分开，GRAIN增量
   只在有支持同作者集评价。配对作者bootstrap至少1000次，CI条件于已拟合头；
   外层折/三个view敏感性不能当独立重复实验。多层、raw/cue/placebo完整列出。
   保留所有效应与负结果，不采用r>.5的行业外门槛。
8. 置换negative control须在各source内按完整作者标签向量打乱，并重新
   执行inner选择；先20次管线sanity，再199次作为完整经验null。
   完整199次未完成时明确标注，而不是以一次近零当检验完成。
   Big5尺度0..1/MBTI二元极性必须验证，缺失/错误编码不隐式填充。

## 接下来的实现

run_grain_vector_mechanism_external_queue.py负责冻结源代码/材料/模型的全量提取。
run_grain_vector_mechanism_external_links.py负责独立来源头/OOF及bridge封存门。
计划不会因先完成某些分片就把它们升为全量headline。结果未见前不另加层搜索。
理论问题依旧是：内部表示的外部可读信息，能否超过一般embedding和GRAIN，
而非把控制LLM的Yes/No输出当作已经控制人格推断。
