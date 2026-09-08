# GRAIN 待研究拼图与受控世界导航实验计划

**计划与文献研究完成；以下模型实验全部 PENDING，未开始。** 当前任务只授权
整理、查阅论文和详细规划。M27–M31维持已交付状态；不建立新goal、不启动
collector、不派发agent、不安装依赖。恢复实验需要下一条明确的执行指令。

## 1. 当前终点与两条路线

最终目标是从新作者文本中独立计算心理相关判断，并解释有效运算如何形成。
当前公式可替代后缀读出，仍依赖冻结LLM前缀。新计划保留两条路线：

```
                    受控事件世界 + 多种语言表达
                                  |
                     条件化证据与作用路径定位
                           /                \
 A: T_state(x,q,c) -> P_hat,Q_hat       B: T_projection(x,q,c) -> z_hat1,z_hat2
                           \                /
                            冻结计分函数 f_q
                                  |
             真实作者文本：模型响应 / 外部关联 / 运行依赖
                                  |
                   局部地图有用 -> 扩展下一块，而非无界填表
```

A提供更丰富的中间计算解释；B只重建固定读出所需信息。受控世界是共同平台，
不是替代其中一条路线。完整P/Q恢复可作为A的目标，不应成为B的入口条件。
对已有目标的低维投影可计算，不等于原始文本具有同样的低维充分统计量。

文献、数学依据与使用边界见[22项原始研究综述](GRAIN_NAVIGATION_LITERATURE_REVIEW_V1.md)。
旧[两路线计划](GRAIN_TEXT_TO_FUNCTION_TWO_ROUTES_PLAN_V1.md)保留，本文补充第三个
共同平台及完整依赖，不重写历史结果。

## 2. 全部当前pending清单

范围为M31后的主线与此前明确延续的数学构造问题；历史实验报告里的所有
“下一步”不自动变成当前任务。下表每项状态均为PENDING。

| ID | 课题 | 对应已有证据/缺口 | 文献接口 | 优先级/依赖 |
|---|---|---|---|---|
| NAV-01 | 受控世界与人格归因导航材料 | P1绑定/条件实验；尚未纯合成构造后返回真人 | 因果抽象、受控归纳头研究 | P0，共同起点 |
| NAV-02 | 路线A：上游运算与P/Q生成 | 现有缓存无上游梯度，生成过程依赖LLM | QK/OV、EAP/EAP-IG、transcoder | P0，01/04 |
| NAV-03 | 路线B：有效投影独立构造 | 已有34组G/g，但仍要前缀产生输入 | IG、active subspaces、模块化符号提取 | P0，01，复用02发现但不等完整A |
| NAV-04 | 窄目标的证据变化响应 | M30B有水平/外部运输，未完整检验证据响应 | 因果抽象、RAM | P0，01，和02共用材料 |
| NAV-05 | E/EI措辞与输出编码分离 | M28响应r=.047，构念名近但程序未必近 | 线性表示/可读性、Function Vectors | P1，01；阻碍跨查询比较时提前 |
| NAV-06 | 共享/专属路径与GRAIN群体分层 | M31协方差已分账，尚未上游共同机制交换 | 因果抽象、功能向量、CKA | P1，02/03/05 |
| NAV-07 | 状态与条件的作用分配 | 自由相谈强，课题回答弱；时点与材料不同 | CAPS、状态分布、对话机会、RAM | P1，04/10，不把状态强行稳定化 |
| NAV-08 | 构造参照与作者常模分开 | 2构造作者、12核心常模；新目标无通用常模 | 固定参照、MTMM、抽样不确定性 | P1，03/10之后，不阻塞局部发现 |
| NAV-09 | 共享文本前缀/KV计算 | 34查询重复前缀；现有use_cache=False | SGLang/RadixAttention、因果前缀 | P1工具项，有明确节约才做 |
| NAV-10 | 合成→自然文本的循环连接 | 实际PANDORA→MEPS运输，不等于合成运输 | RAM、跨材料机制复现、MTMM | P0，每块局部图后立刻接入 |
| NAV-11 | 最小独立程序与依赖清单 | 前缀仍加载完整权重；暂不能LLM-free | 因果抽象、模块化计算、符号提取 | P0终点，02或03 + 10 |
| NAV-12 | 条件局部图册、方向与交互补项 | M20/M21有限场运输弱；通用算子未完成 | 梯度子空间、局部近似、带条件函数 | P1，02/03局部有效但换条件失配时触发 |

P0表示直接减少文本→判断链中的LLM依赖；P1表示由具体发现触发的开发，不是
把全部检验先做完才能研究主线。12项是工作分解，不是12条已经成立的假设。

## 3. 共同材料设计

### 3.1 第一轮目标与来源

默认首轮以`extraversion`、`TF_cont`、`adapted_dejected5`三种查询为候选：
分别覆盖连续特质、二元偏向和已有较强外部连接的状态目标。这是基于已见结果
选择的探索性起点，不是新的预注册独立验证。三个key已与`direct_traits.py`
和`mood_catalog.py`核对；执行前冻结对应M31卡片与prompt版本。其他31项仍在目录中，按模块
复用逐批接入，不要求第一张地图覆盖所有34项。

先保留当前冻结观察器、模型/层/模板/答案集合和旧G/g对照。构造只使用新的受控
事件材料与既有允许的非MEPS参照；不使用MEPS文本或量表拟合任何新算子。
外部连接优先沿用已有PANDORA source-only及46人已授权相谈记录；这些已经
暴露的数据用于探索性迁移检查，不重新包装成未见过的确认集。

### 3.2 场景真值与人格归因分离

建议世界记录字段：

```
scene_id, semantic_family, actor, recipient, action, opportunity,
voluntary_or_required, negation, quotation_owner, evidence_reliability,
event_time, query_referent, query_occasion, realization_id, split
```

不是一次交叉所有字段。首轮操纵角色归属与自愿/条件，随后由结果触发否定、
时间、冲突证据等模块。输出中分开`world_facts`与`model_attributions`，禁止
把人格标签直接嵌入世界真值，禁止用“模拟外向80分”作为已知真人标签。

建议首个有界材料包：12场景骨架×2角色配置×2机会条件×2表达版本=96文本；
三个查询=288原生读取。按整个场景骨架分6构造/3开发/3保留组，不能把同一
场景的改写拆入训练与测试。此设计的主保留是场景/组合，不冒称表达模板
完全未见；必要时冻结后另加一个新renderer的小包，检验模板运输。

文本必须包含高/低方向、混合证据和无法判定情形；“证据不足”首先作为材料
属性记录，不能为了让模型一致就删去。保持冻结问题区域token定义，避免
将模板更换引入的坐标漂移误当新的作者机制。

### 3.3 真正应保存的地图

每个节点/边保存语义角色、层/头/通道位置、条件、作用方向、实际干预幅度、
函数/原生响应、参考材料与运行hash。群体位置和人类量表相关另列。可视化
最后生成，图上距离不是证据；优先能重算一条边，再追求画大图。

### 3.4 两种参照来源：真正测试“虚拟世界造公式”

设置独立的参照来源因子，避免只在合成材料上测试真人参照公式：

- `R_real`：现有两个PANDORA参照作者的冻结G/g，作为连续性对照，不重拟合。
- `R_synth`：只从6个构造场景族中，按预定事件覆盖选择两个合成参照，使用
  同一函数构造方法重新计算G/g。所有参照、系数和选择规则在真人连接前冻结。

`R_synth`不能按外部人格相关挑选参照或混入MEPS材料。每个查询先做一个
预定参照对，发现明确覆盖缺口后才增加另一个局部参照；不得无限搜索“最好”
参照。初始288次是材料的原生读取数，合成参照梯度另记调用与时长，仍受
阶段预算约束。保留场景只评估，不参加参照选择。

先比较两种函数对模型判断和编辑响应的恢复，再分别连接相同真人文本；A/B
构造程序也记录它使用哪组固定读出。这样才区分“合成输入上的机制发现”与
“纯合成研究参照→真人运输”。观察器仍有预训练知识，后者不代表从零知识
或无任何真实文本先验推导人类心理。

## 4. 工作包执行卡

以下脚本/模块名全部是**拟新增，尚未实现**；不可把这些路径当现成命令启动。
所有输出放在ignored的`results/grain_vector_mechanisms/navigation_v1/<ID>/<run_id>/`。

### NAV-01：世界生成器与基础路线观测

研究问题：同一目标可由哪些不同事件入口推动？对象归属和机会条件在哪里改变
判断？文献借用因果抽象的已知高层变量及受控研究的关系拆分，而非规定心理答案。

拟新增`prepare_grain_controlled_navigation.py`，函数`build_worlds`、
`render_world`、`split_scene_families`；`run_grain_navigation_baseline.py`负责
复用旧observer与固定函数。先手工规则renderer，不依赖另一个LLM无限生成。

输出`worlds.jsonl`、`design_manifest.json`、`native_and_function.csv`、
`response_by_condition.csv`、`reference_source_manifest.json`。分别保存
R_real/R_synth的函数卡与参照出处。检查角色归属、逐条件效果、样本数和不同表达版本
的响应，不用假人格accuracy。交付一个在保留场景上有可追踪响应的材料族即可
进入02/03；无响应的目标保留并换更相关的证据族，不因此扩模型/层数大网格。

### NAV-02：上游路径定位与路线A

拟新增`grain_research/vector_mechanisms/upstream_routes.py`及
`run_grain_upstream_routes.py`，提供`capture_differentiable_prefix`、
`rank_route_candidates`、`patch_route_group`、`replay_state_program`。
模型参数冻结，只有必要激活/输入表示启用梯度；不能把detached缓存反传当成
上游证据，也不能移植Qwen2.5不匹配的其他模型头权重。

先从f_q反向排序候选分支，再验证至多8组候选和2个有实质意义的组合。
保留MLP/残差/归一化影响，不只消融注意力头。若单头移除没影响，要检查
替代路径，不据此宣称该信息不存在。把投影响应、原生响应和P/Q误差分开。

输出`edge_candidates.csv`、`patched_effects.csv`、`upstream_graph.json`、
`state_reconstruction_metrics.csv`、`runtime_dependencies.json`。发现实际干预
可改变目标的路径组后立即尝试独立复现这部分运算，不等待全网络审计。
只在完整前缀提供输入时能运行的子图标为prefix-dependent intermediate。

### NAV-03：两项有效投影的独立程序

拟新增`run_grain_projection_program.py`和
`grain_research/vector_mechanisms/projection_program.py`，函数
`collect_upstream_jacobians`、`fit_candidate_operator`、`score_from_text`。
首轮比较三种机制候选：明确条件的加权和、保留成对关系的门控和、从02提取
的稀疏计算组合；不是Ridge/LightGBM的预测排名赛。

固定G/g，只以构造材料的z1/z2和证据变化学习/推导上游候选。若需要active
subspace，用较早层或共同语义字段的Jacobian，避免在已线性化的末端发现
一个必然的低秩。完整PQ重建损失只作A对照，B按有效投影和编辑响应比较。

输出`operator_spec.json`、`projection_predictions.csv`、
`response_transfer.csv`、`dependency_costs.json`。保留至少一个可解释、
跨场景有效的局部程序就进入10；若仅能读静态分数而不能读变化，记录缺的
交互类型，不用更多人类标签调出表面高相关。

### NAV-04：窄目标的自然证据响应

拟新增`run_grain_narrow_evidence_response.py`，提供
`make_local_evidence_edits`、`evaluate_frozen_response`。复用NAV-01与非MEPS
构造文本，分别改主体、否定、当前/过去、支持/相反证据；用真实自然文本
局部编辑作下一层，而不是只在随机hidden扰动上测试。

输出目标、编辑类别、原生变化、函数变化、三分量变化及作用相关/误差。
编辑方向的已知性来自事件操作，不把“应该更低落”写成真人真值。先确认
哪些类别跟随，哪些是模型本身弱反应，再决定02/03优先处理哪段路径。

### NAV-05：查询身份和输出编码

拟新增`run_grain_query_identity.py`，函数`build_query_factorial`、
`align_answer_codes`、`compare_query_responses`。用同一材料与对象，比较
“外向程度的高低判断”和“E/I偏向判断”两种语义框架，再各自比较两个二元
答案编码，形成可比的2×2诊断。它是新仪器诊断，不替代原连续Big5或原MBTI。

保留题意、编码、token区间和输出标尺；新问题区域要独立构造参照函数，不能
直接复用不同token行的G。输出`query_identity_matrix.csv`、配对变化表及
`coordinate_compatibility.json`。定位哪种改变导致路线改变，比要求所有
问法不变更重要；不通过挑选最有利问法改写已发布结果。

### NAV-06：共享、专属与群体条件化

拟新增`analyze_grain_navigation_sharing.py`，函数
`build_response_kernel`、`test_shared_route_exchange`、`compare_group_routes`。
以共同事件干预上的功能响应比较查询，先复用现有M31符号协方差；随后
检查一个候选上游替换是否同时改变两个目标。

输出`response_kernel.csv`、`shared_specific_edges.csv`、
`group_condition_effects.csv`。节点重叠、CKA、余弦和有符号干预作用分列。
群体定义先用既有GRAIN参考或独立文本A，文本B作响应比较；维持总体主结果，
不以全样本标签分组来制造群体内人格同质性，也不强制添加identity乘法增益。

### NAV-07：状态、机会与时间

拟新增`analyze_grain_state_opportunity.py`，函数
`cross_occasion_links`、`partition_material_and_occasion`。复用M30C的13改编
条目与四时点，只做已授权本地外部联系；不将其用于模型拟合或挑选路径。
先在同一criterion上交叉比较课题回答/自由相谈，然后保持mid/post角色区分。

输出完整view×occasion×query矩阵、配对bootstrap、变化关联及已有状态控制
后的关联。不会从一次会话造出跨日重测，也不把任何有时间差的分数都叫人格
变化。模拟世界另外验证时间标记与事件先后怎样影响读出，二者不混成同一证据。

### NAV-08：参考图与作者位置

拟新增`build_grain_reference_atlas.py`，函数`freeze_reference_population`、
`score_against_reference`、`bootstrap_reference_sensitivity`。分别记录G/g
构造参照、作者常模和地图覆盖范围。先保留旧2作者函数和12人核心常模作
控制；新参照来自非MEPS的允许材料，以覆盖条件而非人类标签选取。

输出`reference_manifest.json`、`reference_sensitivity.csv`、
`batch_invariance.json`。新批次加入不改变旧人的既定分数；不同参照下的位置
变化照实报告。不强求正态分布、跨人群同一系数或部署级常模后才允许研究。

### NAV-09：共享前缀工具项

拟新增`benchmark_grain_shared_prefix.py`，函数`find_exact_common_prefix`、
`cache_context_only`、`branch_query_suffix`。现有脚本`use_cache=False`，必须
新增隔离实验，不覆盖原collector。前缀必须包括相同token、位置、mask和
运行设定；带目标专属指令的前缀不能强行复用。

输出同一材料、同一dtype下的时间、峰值内存、缓存命中及P/Q和读出误差。
先做3查询小包；收益不足或改变数值时保留旧路径，不迁移整个服务栈。
共享缓存仍是LLM计算，不计入11的运行独立性成功。

### NAV-10：每张局部图后的真人哨点

拟新增`evaluate_grain_navigation_transfer.py`，函数
`freeze_program_before_linkage`、`evaluate_native_and_function`、
`join_author_criteria`。复用已有M29 source与M30B相谈目录、hash和作者连接。
预测先冻结，再连接外部标签；不在46人文本/标签上调程序。

每轮四类对照：原生模型、旧固定前缀+函数、新A、新B；函数及A/B再分
R_real/R_synth参照来源，不把合成参照结果混入旧公式成绩。缺失分支明示，不
用缓存向量输入伪装新文本独立运行。分别报告模型一致性、编辑响应、真人
Pearson/Spearman与MBTI AUC，以及调用依赖。PANDORA两种来源不能拼成
same-user bridge；SN64人的四个负例仍保留样本量说明。

人类关联可支持发展，但外部数据已暴露，后续适配产生的结果应标记为新探索。
第一次见到局部机制保留有用关系即可扩大邻近条件，不必等全目标显著。

### NAV-11：独立运行交付

拟新增`export_grain_independent_program.py`，函数
`trace_runtime_dependencies`、`run_raw_text_only`、`export_operator_bundle`。
只有完成输入依赖的分支才打包：新文本输入，不能提供缓存PQ、全模型返回值
或人工场景真值。允许固定tokenizer/词表/提取权重，但必须列出来源及大小。

输出`program.json`、`coefficients.npz`、`runtime_manifest.json`、
`raw_text_replay.csv`、`BUILD_AND_SCOPE.md`。分别标为full-prefix、extracted-
neural-program或explicit-operator；只省后缀、只换小黑箱都不混报为完全机制
解释。范围是指定目标/材料下端到端有效计算，不宣称全人格已被短公式穷尽。

### NAV-12：局部图拼接与条件交互

拟新增`analyze_grain_local_route_atlas.py`，函数`fit_condition_gates`、
`evaluate_chart_overlap`、`decompose_route_interactions`。由02/03发现明确
条件失配时触发。比较单图、简单平均和按可观察条件门控的局部图，不重启
旧M20/M21方向数/alpha穷举。所有门控来自独立构造材料。

输出`local_charts.json`、`overlap_consistency.csv`、
`conditional_signed_terms.csv`。检查同一编辑跨分区时的跳变与作用方向。
有限矩形差分可测交互，不必先计算完整高维Hessian；交互存在时先保留，
不是把它定义成误差删除。没有共同光滑场时允许分段公式作为有效成果。

## 5. 最小恢复顺序与资源

| 阶段 | 内容 | 有界交付 | 开发时间预算 | 神经预算 |
|---|---|---|---|---|
| A | 01 + 04基础包 | 96文本、288原生读取、双来源参照函数、首张响应表 | 2–4小时 | 先限20分钟，参照梯度另记；完整单元后可暂停 |
| B | 02定位 + 03首个候选 | 少量头/通道候选，至少一个A或B局部实现 | 4–8小时 | 每次20分钟；首轮累计上限60分钟 |
| C | 10哨点 + 12按需补图 | 首次真实文本运输对照与失配类型 | 1–3小时 | 优先缓存；新前缀按冻结作者包限20分钟 |
| D | 05/06/07/08/09按发现触发 | 跨查询、状态、参照或速度的一个具体决定 | 各1–3小时 | 不默认并行开启所有包 |
| E | 11打包 | 新文本独立运行示例、依赖账本、局部公式 | 1–3小时 | CPU重放优先 |

以上是恢复后的规划预算，不是当前已执行进度或速度保证。两条路线并存，
不意味着两个神经采集器同时占设备。每个完整场景/作者单元后记录剩余
科学决策价值；继续只会填表时暂停，而不是把暂停称为统计无效性证明。

默认路由沿用最新用户约束：192.168.1.20 Windows用于重负载/训练型计算，
可用VRAM预计小于10GB；192.168.1.30 Linux Intel16GB用于兼容的评估。
本轮没有训练任务。白盒梯度不能通过Ollama/llama文本API完成；远端若不能
直接访问同一checkpoint与autograd，保留已验证Mac MPS小包，不假设端口可达
就等于支持内部干预。跨设备dtype/backend差异另列，不能与机制改进混合。

## 6. 现有代码与依赖接点

| 已有组件 | 可直接复用的内容 | 不可假设已具备 |
|---|---|---|
| `grain_research/vector_mechanisms/direct_traits.py` | 目标、模板、候选答案与token化 | 任意新问法与旧G兼容 |
| `prefix.py::capture_prefix` | block16边界与layer17 QKV缓存、禁止后缀trace | 上游梯度、独立裁剪模型、KV复用 |
| `factor_functions.py::joint_gradient` | 冻结后缀对P/Q的偏导 | 原始文本到P/Q的全路径解析 |
| `run_grain_narrow_functions.py` | 独立参照构造与状态/关系目标 | 可以用MEPS训练新函数 |
| M31公式与协方差账本 | 34系数、参照hash、已有作用与外部关系 | 普适人体测量真值或完全LLM-free |

本地可用性检查：`torch`、`transformers`存在；`nnsight`、`pyvene`、
`transformer_lens`、`sglang`、`pysr`当前未安装。本轮未安装任何包。
先复用现有Torch hooks；若其实现成本超过明确收益，再选择一个工具，不
整套迁移。官方入口：[NNsight](https://nnsight.net/documentation/)、
[pyvene](https://github.com/stanfordnlp/pyvene)、
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)、
[transcoder代码](https://github.com/jacobdunefsky/transcoder_circuits)、
[符号模块研究代码](https://github.com/MilesCranmer/symbolic_deep_learning)。
列出工具不代表已经验证当前Qwen/Transformers版本兼容。

每个包统一保存config、source/code/model hashes、角色/场景划分、调用数、
预测、干预、错误、报告和决策。报告按目标与条件分列正负项，不强制每项高相关。
精度、响应、运行独立性是三栏；心理相关不套用任意工程行业的r>.9门槛。

## 7. 暂不恢复但不丢失的扩展

| 扩展 | 论文接口/未来最小问题 | 暂缓原因 |
|---|---|---|
| Persona写入/读取互逆 | Persona Vectors + 因果抽象：写入某方向是否改变对他人同类证据的读取？ | 先获得一条读取程序；不假设写入与读取互逆 |
| 跨模型/语言普适性 | CKA与功能干预：比较作用而非直接比较不同模型坐标 | 当前先闭合一个观察器；既有日语正例保留 |
| 新真人、跨日与临床验证 | CAPS、状态分布、RAM、MTMM指导同人/同条件研究 | 需要新的研究设计与使用授权，非本轮理论发现前置门槛 |
| 旧LoRA/MBTI训练、超参网格、lockbox | 历史benchmark与封存协议保留 | 不减少当前前缀机制缺口；没有重启授权 |
| 全34目标×所有变体全量地图 | 逐张局部图运输成功后再扩大覆盖 | 避免在第一块现实连接之前耗尽资源 |

下一次恢复时，先读取本计划、文献综述、M31账本，再完成NAV-01的可执行
冻结配置。当前没有任何包可以因“pending”状态而自动运行。
