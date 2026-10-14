# GRAIN 情境化测量制作手册 V1

2026-09-13。M14共同制作规格，配套[实测报告](../reports/grain_measurement_foundation/LOCAL_INSTRUMENT_CONSTRUCTION_REPORT.md)。
这是方法开发的两个可执行实例，不是现成临床量表。目标是让制作步骤可检查、
计分参数有来源、每个读数可追溯，而不是把每段文本压成一个任意回归分数。

## 1. B1–B7：同一制作方法，两个不同对象

| 步骤 | 共同规则 | 社交条件轮廓 | 任务当次过程 |
|---|---|---|---|
| B1 定义目标 | 先给对象、比较人群、时间范围和正负证据 | 安静/嘈杂聚会中的本人愉快报告期望；不是外向性 | 任务完成报告的个人水平、条件/伙伴响应、当次/窗口偏移；不是已命名情绪 |
| B2 定义表达设计 | 记录谁在何种机会下说什么，哪些是分配、哪些是选择 | 两个已分配条件，每条件独立事件 | 12个观察轮次，两条件×两种伙伴输入，记录发生轮次 |
| B3 构造证据单位 | 事件ID、主体范围、原文地址、条件、时间不可混在一个平均里 | 两条件内分别累积独立本人事件 | 保留事件→轮次→条件/伙伴绑定；同轮独立报告提高精度 |
| B4 选择模型 | 目标决定参数与不变性，不根据聚类外观命名 | 两个条件期望；水平及半差是派生量 | 三个个人系数加有序状态；状态协方差使用来源学习的标量OU工作模型 |
| B5 校准局部工具 | 构造侧学单位/参照/误差，应用者不参与拟合 | 条件化事件校准、两单元方法矩、作者bootstrap | 同类校准、复制报告噪声、REML人口矩与恢复率、作者bootstrap |
| B6 验证并解释 | 新作者计分后才开真值；按对象给误差、排序、覆盖、地址 | 经验两单元均值对照；重复/缺条件/范围未知 | 逐人OLS/残差对照；保留时钟、测试重排与改变时间绑定 |
| B7 迁移并再制作 | 继承运算与制作规则，不照搬不同对象系数 | 无时钟排序目标；需自己的条件、锚点和来源样本 | 状态对象必须有相应时钟/重复观察，不能由轮序直接改称天数 |

B1–B7承接[情境测量基础计划V3](GRAIN_CONTEXTUAL_MEASUREMENT_FOUNDATION_PLAN_V3.md#5-未来研究者必须能照做的制作流程)。
本包由同一执行者完成；独立制作复建仍是D02。真人心理/行为含义仍由D01提供。

## 2. 共同计分公式

原文和目标查询进入固定观察器，得到正/负命题的蕴含概率之差：

\[
v_e=p(\mathrm{entail}\mid x_e,q^+)-p(\mathrm{entail}\mid x_e,q^-),\qquad
y_e=\frac{v_e-c_0-c_1z_e}{d_0+d_1z_e}.
\]

`c/d`只用来源32条有明确事件方向的公开材料/目标估计；应用时固定。
标定后的事件读数不是概率，可能稍超出[-1,1]，不得私自clip或把它当人格百分位。
这里的方向标签定义事件单位，不是应用作者的潜在参数或问卷标签。

对指定个人，选定参数向量、条件设计及目标：

\[
\theta\sim(m,\Sigma),\quad y=H\theta+e,\quad \operatorname{Cov}(e)=R,
\quad G=\Sigma H^\top(H\Sigma H^\top+R)^{-1},
\]
\[
\hat s_t=t^\top[(I-GH)m+Gy]
=\underbrace{t^\top(I-GH)m}_{\text{来源参照项}}
+\underbrace{\sum_e k_{t,e}y_e}_{\text{有地址的个人证据项}},
\quad C=\Sigma-GH\Sigma.
\]

这继承高斯条件化/线性最小均方估计，不声称新发明了条件高斯定理。
实现用Cholesky而非显式求逆；独立审计另用事件级精度矩阵公式核对。
人群平均是参照，不是噪声删除目标；缺一条件时的借力必须与该人的实测条件区分。

12个来源作者bootstrap构造给出`score=mean(score_b)`，以及：

\[
V_t=\operatorname{mean}_b(t^\top C_b t)
+\operatorname{Var}_b(\hat s_{t,b}).
\]

第一项是工作模型条件方差，第二项是有限来源构造差异；`score ± 1.96 sqrt(V)`
是工作区间，不是包含一切误差的精确心理区间。本轮固定的锚点/观察器没有被
bootstrap，语义错归属、系统性措辞偏差不会自动进入这个方差。
报告中社交响应和任务个人水平的覆盖不足，必须和有效的状态读数一起保留。

### 2.1 个人条件对象

`theta=(mu_quiet,mu_noisy)`；`level=(mu_quiet+mu_noisy)/2`，
`condition=(mu_noisy-mu_quiet)/2`。两条件是目标空间，不是四个独立发现因子。
同一事件重复记录不增加观察数；不同独立事件即便用了相同句式仍是不同机会。
两者必须靠记录协议区分，不能由字符串相同/不同自行判定。

### 2.2 当次过程对象

\[
y_{u,t}=a_u+b_uz_t+c_up_t+s_{u,t}+e_{u,t},\qquad
\operatorname{Cov}(s_t,s_v)=P\exp(-\lambda|t-v|).
\]

`theta=(a,b,c,s_1,...,s_T)`；当前状态为`s_T`，当前表达为
`a+b*z_T+c*p_T+s_T`。窗口状态是记录时点的梯形加权和，不是未经采样的真实连续积分。
同一整段文本可有不同目标核：用于水平的证据权重不等于用于当前偏移的权重。
伙伴输入在本例由设计记录；此处未重新证明自然对话中输入的外生性，相关识别条件见
[互动响应报告](../reports/grain_measurement_foundation/INTERACTION_RESPONSE_REPORT.md)。

### 2.3 参照、补采与最低条件

`source_relative_position`使用冻结来源均值/方差；应用者单独或成批进入，分数不重中心化。
当前表达/窗口状态没有一个与设计无关的统一来源常模，本版本相应字段为null。
下一条独立报告的工作方差收益是：

\[
\Delta V_t(h)=\frac{(t^\top C h)^2}{h^\top C h+\sigma_e^2}.
\]

输出按来源构造平均，当前不预测新增材料后构造方差如何改变。完全重复已有事件收益为0。
社交候选为两条件；过程候选仅为当前条件/当前轮次的新独立报告，不是已完成的
“未来未见条件自适应设计”。2个过程时点只满足最低计算入口，不保证三个个人系数
都由数据识别；检查`empirical_profile_design_rank`及明示来源借力。
本例6条/条件与12轮×16条是实验预算，不是通用的足够文量定理。

## 3. 输入与输出协议

原始JSON顶层必须含`instrument_id`、`clock_unit`、`records`。
每条record恰有以下八个字段，不能放入`score`、`label`或作者真值：

```json
{
  "author": "new_author_01",
  "event": "occasion_01_report_01",
  "text": "At the quiet gathering, I enjoyed spending time with the other guests.",
  "scope": "self_event_report",
  "condition": -1,
  "time": 0.0,
  "partner": 0.0,
  "replicate": 0
}
```

这只是schema示意，不是本轮另一个验证案例。条件编码/含义见bundle的`spec`。
`scope`及条件由观察协议确认，不是NLI自动认证；未知范围写`unknown`，保留地址但不给值。
同作者同event原文/条件/时间冲突会报错，须采用已有M02显式修订模型，不覆盖旧事件。
本实例不自动处理更正链或全部开放共指。过程同一轮必须有明确且一致的条件/伙伴输入。
`replicate`标识试点复制组；它本身不是额外人格特征，也不是独立性的语义证明。

输出四文件：`reader_results.json`固定模型概率；`evidence.json`原文/字符地址/校准值；
`predictions.json`每人各目标分数、工作区间、来源/证据分账及逐事件权重；
`manifest.json`模型revision、输入/输出hash、实际推理量和访问检查。
`evidence.json`保留原文，**真实数据应用时为私有产物，不能直接公开**。

## 4. 从现有材料复建

在GRAIN仓库根目录运行，复用Persona本地环境；本次不安装依赖或下载权重。
配置在`configs/grain_local_instrument_construction.json`，读者固定为
`cross-encoder/nli-deberta-v3-base@6c749ce3425cd33b46d187e45b92bbf96ee12ec7`。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
OUT='results/grain_vector_mechanisms/mechanism_todo_v1/local_instrument_rebuild'
"$PY" scripts/run_grain_local_instrument_construction.py prepare --output "$OUT"
"$PY" scripts/run_grain_local_instrument_construction.py construct --output "$OUT"
"$PY" scripts/run_grain_local_instrument_construction.py apply --output "$OUT"
"$PY" scripts/run_grain_local_instrument_construction.py evaluate --output "$OUT"
"$PY" scripts/run_grain_local_instrument_construction.py natural --output "$OUT"
"$PY" scripts/score_grain_local_instrument_cold.py \
  --bundle "$OUT/instruments/social_condition_profile_v1_91401.json" \
  --input "$OUT/cold_inputs/social_condition_profile_v1.json" \
  --output "$OUT/cold/social_condition_profile_v1"
"$PY" scripts/score_grain_local_instrument_cold.py \
  --bundle "$OUT/instruments/task_occasion_process_v1_91401.json" \
  --input "$OUT/cold_inputs/task_occasion_process_v1.json" \
  --output "$OUT/cold/task_occasion_process_v1"
"$PY" scripts/audit_grain_local_instrument_construction.py --output "$OUT"
```

这些命令是复建说明，不能指向既有r1以覆盖冻结产物；阶段要求新目录。
`prepare`依赖已审的`condition_accumulation_r1/public_scored_bank.csv`和来源NLI记录；
`natural`依赖`relational_evidence_r1/natural/private_relations.jsonl`，不是发布包所必需的
公共数据。缺私有台账可略过natural，不补造其应用结果；主要数值构造、冷计分及审计独立运行。
源文件hash或模型校验不一致时先记录版本变化，不修改hash来伪造相同运行。

应用已有bundle只需运行两条冷入口之一并替换`--input`为符合协议的新人材料。
入口在读取应用原文前预加载固定模型资产，随后在数据访问约束内执行全部新forward；
无缓存分数/真值输入。详见[运行修正说明](../reports/grain_measurement_foundation/LOCAL_INSTRUMENT_EXECUTION_REPAIR.md)。
实际冷运行仍使用神经NLI编码器，**不是完全无神经模型的D0路线**。

## 5. 在另一个情境制作，而不是盲套现成系数

1. 写自己的B1–B3卡。决定是目标已知的事件反应，还是匿名配置；当前CLI只实现前者两例。
2. 在来源材料固定正负命题及条件化锚点，使用`calibration`/`restore_units`校准单位。
3. 用独立来源人的合格事件读数构造pilot。两单元输入`learn_cell_moments`为
   `[author,2,event]`；过程`fit_pilot`输入`[author,time,replicate]`的复制组均值，以及记录到的z/p/t。
4. 学习人口/噪声/时间参数并按作者重采样，封存`main`、`bootstrap`、pilot作者列表、
   查询/模型/时钟/锚点信息。`process`还须记录每轮实际独立报告数以便缩放观察噪声。
5. 在未用于构造的材料上用`score_records`或冷入口计分。先锁分数，再检查目标/外部连接；
   用不同措辞、材料量、条件缺失区分语义偏移、采样误差与人群参照差异。
6. 完成B6/B7说明后才做相应意义声明。没有当前目标所需机会/时间/主体字段时，改测有定义的
   可观察对象或补记录；不是把所有自然文本都强标为这两个实验条件。

本runner是指定两例的可重复制作程序，**不是输入任意语料即可自动发明心理量表的通用编译器**。
`fit_pilot`当前受限于共同试点设计；异长/缺测pilot、非平稳过程或内生伙伴输入需要按对应机制
换观测模型，而不是把函数能运行当成识别条件已满足。M05的查询K是另一种有作用依据的
观察量，本例没有把不同单位的K/logit与NLI事件值强行相加；后续条件整合见第8节，
不是改写本双实例的已冻结bundle。

## 6. 新条件与观察预算：M10增补

[实际设计报告](../reports/grain_measurement_foundation/NEW_CONDITION_DESIGN_REPORT.md)
将B2/B3/B5进一步连接：独立来源文本估计响应协方差与事件噪声，固定目标的
Schur补增益决定读取哪一份已可用报告；副本共享同一观察误差，增益严格为0。
来源仅见旧条件时，`X_source d=0`但`target'd!=0`的方向不能用更多旧报告识别。
遇到新条件先判断需要补哪类来源观察，再优化个人计分；本包64名补采来源作者使
隐藏条件差分风险下降70.1%，不是新应用作者标签或免费oracle修复。

接口：`fit_source`学习二次响应及相关状态；`fit_expanded`学习补采5单元；
`moments/choose/score_weights`只读来源模型、目标、条件/时间/事件ID，不读应用值。
其中5单元函数只支持已补采的5条件；真实新语义条件需另有证据，不自动由时长插值解决。
当前设计是已有文本读取，不把时间倒序读取误称前瞻施测；实时时间/可用性要进入候选限制。
保留模型错设带来的乐观工作方差，不将其当精确人格后验。

## 7. 验收及后续责任

有限计算验收：两例可来源构造、冻结、新人计分、实际原文冷运行、独立数式复算、
给出证据和来源分账。RMSE、作者排序、覆盖必须分别读，不能互相替代。
本轮允许保留工作区间偏窄和部分排序退化的结果，不为通过而重新选择种子或材料。
自然台账273事件在这两个设计下0合格，不是旧PANDORA/Essays/GRAIN所有分数失效。

D01承接真实心理/行为解释；D02承接独立执行者照此手册制作；D03承接理论论文与
正式发布。源代码/方法说明可准备交接，私有原文、凭据、权重与未授权数据不可自动发布。

## 8. 同原文的多观察与个人函数：M04/M05增补

[共同观察实测](../reports/grain_measurement_foundation/JOINT_CONDITIONAL_OBSERVATION_REPORT.md)
补齐B3–B5的新接口：GRAIN、查询矩、组内K与NLI共同观察一件事时，先从来源锚点
估计 `o=b(z)+L(z)m+error`，保留各单位与共同R。解码后同时保存 `W L` 分辨矩阵、
奇异值和事件地址，不把秩3就当三种心理构念都可稳定测量。

当目标包含两个反应的联合关系，先检查事件是否配对；`mean(q1*q2)`不是
`mean(q1)*mean(q2)`。本包保留配对时联合条件r=.3578，边际不变但错开配对后.0408。
这是编写机制文本与模拟参数的结果，提供观察设计规则，不代替D01心理含义验证。
查询组内K与此事件乘积是两种不同观察，前者有有限原生作用却没修好本包弱联合方向。

制作时：`calibrate`只拟合来源角点；`build_meter`冻结各条件的单位/协方差/解码；
`read_events`读取新人；`conditional_function`从本人端点得到水平/斜率及材料协方差；
`predict_function`给支持区间内未见数值条件。个人响应共享随后由第10节补齐；开放语义条件仍需M01。
来源误差R只描述角点内措辞变化，不能冒充事件抽样、来源估计和心理真值的全部不确定性。

遇到满秩弱逆，不优先增加惩罚参数搜索：先问目标所需的关系是否进入观察；
若新措辞使 `mhat=C_z m+d_z`，检查C/d随条件的变化如何将水平与响应互相混入。
精确复制通道不增加信息，可逆单位变换不改变解码值，均已独立验算。
完整多通道未胜NLI+事件乘积三通道，因此不要求每份量尺都使用全部GRAIN/query工具。
该接口是可选构造层，不能因拼接更多工具就自动升级解释范围。

代码/命令/全部失败结果见实测报告第9节；主r1和结果后follow-up分开保留。

## 9. 从当前分数到可更新证据状态：M02增补

[事件历史实测](../reports/grain_measurement_foundation/EVIDENCE_HISTORY_REPORT.md)
给B3/B5增加来源与版本层。输入保留作者内`record_id/event_id/block_id`、条件、
读数及显式关系；这些字段不是文本相似度的自动结论。新事件、独立复述、精确副本、
更正、撤回和引用摘要分别处理。新时点发生相反反应不自动覆盖旧事件。

来源阶段`fit_source`估计个人均值/方差与块/事件/报告方差；应用阶段`active_records`
决定有效主张，`score_records`输出个人函数、来源offset、各条权重及目标协方差。
新增证据优先使用`evidence_history_innovation.square_root_append`的QR创新形式；
原`append_innovation`在近重复报告下数值相消，保留冻结代码但不推荐新运行使用。
更正/撤回仍重建活跃账，不能当新独立似然直接追加。

信息按目标`c`的方差下降`c'(P-P_new)c`及完整信息矩阵计，而非统一“有效文本条数”。
已知依赖下，精确副本增益0；撤回可能扩大工作方差；值更正不必改变设计方差。
仅去重事件仍会忽略跨事件共同状态。本包水平r相同但覆盖.9688对.8750，提醒制作时
不要用排序相关代替证据校准。来源学习二阶矩的区间仍是工作区间，真实覆盖须另检。

当前分数和协方差不足以支持所有未来定向更正。交接包至少保留能重构来源影响的
状态；完整数值账+依赖图是当前可运行方案，不宣称最小/定长记忆。`restore_summary`
恢复历史元数据，数值附录仍须调用者提供/链接，不能把几句自然语言摘要当无损替代。
真实作者的数值账、来源和文本地址仍属私有衍生资料，不随公开方法文档发布。

此机制在576来源/768新应用模拟参数组、冻结文本读数上完成；没有新增真人量表效度。
自然台账中的跨评论事件识别与更正链仍须M01，互动含义留M09，心理解释留D01。

## 10. 共享模式而非共享答案：M04增补

[响应共享报告](../reports/grain_measurement_foundation/RESPONSE_SHARING_REPORT.md)
把“哪些人相似”变成目标相关问题。先用来源人的跨条件配对观察学习响应核K，
不只拿条件文本距离当K；来源均值、个人平均偏离和条件响应分别保留。
`learn_response → compile_function → apply_function`给一人的完整条件函数，
三个目标只是该函数的泛函，而非三个各自训练的head。

来源两半必须独立机会；`Cov(A,B)`用于信号、`Cov(A-B)/2`用于一个半组的观察误差。
同事件复述横跨两半会把事件误差当成稳定函数，先接M02依赖账或换适合的误差模型。
本实现只支持共同五条件的均衡两半，异长/缺测、共享事件/时变函数需要相应扩展，
不能以任意文本随机切半替代设计。源码保留原负特征值及PSD投影，不用删掉记录来提高信度。

共享`Phi/Lambda`不共享`xi_u`的取值：来源给模式，本人的观察确定该模式上的位置。
新人仅有中心平均时，本包不能恢复其跨条件反应；新增两端后才出现隐藏响应信息。
按响应轮廓匹配比按平均值匹配有用，但在条件特有世界不成立，不能加统一群体乘数。
如果条件函数近似线性，标准随机斜率已够用；条件与响应排序不同，再学更一般的响应K。

`P=I-11'/C`只做精确分账/距离，不删除原平均。即使条件独立，`P sigma²I P`
也有负非对角元素；不要把这个结构约束解释为抑制性心理机制。
对来源未覆盖的新语义条件不自动外推；自然证据身份留M01，心理与行为意义留M13/D01。

## 11. 先声明证据身份，再解释个人位置

[身份接口报告](../reports/grain_measurement_foundation/EVIDENCE_IDENTITY_REPORT.md)
区分意愿、行动报告、偏好、能力、当前体验与当次请求。它们是可并存的观察通道，
不是六个人格因子；NLI蕴含率不等于实际行为发生率。否定意愿仍可能是意愿通道，
其方向/实现程度还须另外绑定，不能把所有支持某通道的句子都记作正向特质。

制作时依次保存：原文角色/来源与地址、命题/模态/时间、功能/前文依赖、观察目标、
采样或机会分母、来源读数校准、个人函数、候选意义。开放多标签输出不自动归一为
一个互斥类型概率向量；只在声明单身份生成设计时使用本包simplex decoder。

同内容分数可能对应相反身份目标。先问目标是否由现有观察识别，再谈精度。
数值满秩仍可能来自微弱、措辞不稳的方向；本包保留其失败而不据结果重选秩。
具体事实NLI可正常、抽象身份元描述却失败，说明“给对象取一个泛化名称再询问模型”
不是自动获得可靠运算器。后续接具体事件绑定，不增加人格名称提示来替代。

互动中区分`pi_u(o)`机会份额与`f_u(o)`条件反应。未知机会只报观察轮廓；比较
同人/异人的反应时使用明确参照机会或分别呈现两项。缺失前文/回应时间保持缺失。
解释卡不能把未来意愿写成既有行动、把当次情绪表达写成跨时稳定特质，或把被
提问的机会写成个人主动选择；这些是所测对象的区别，不是压低相关的防御条款。

## 12. 计算解释闭环，不由分数替证据命名

[M13解释链](../reports/grain_measurement_foundation/EXPLANATION_CHAIN_REPORT.md)提供可运行例：
原文哈希/字符地址、匿名slice均值、查询源项/组间/组内分账、有限条件改变、候选意义卡。
计分器只读原文；解释身份由明确的材料记录给出，不让高分自动选择意愿或已实施行为。
自然材料没有可信身份记录时保留候选，交M01/独立编码，不能把模板解释当事实识别。

同一原文的数字必须引用全部输入，而候选含义可指向具体句段；数值相加是计算分账，
不是各句独立因果贡献。四slice均值与token加权坐标使用不同分母，分别记录。
历史心理关联要带对象ID和运算式：Cov(F03,旧Openness查询)的关联不属于F03本身。

24段机制材料上改变时限比改变请求表达极的平均影响更大。制作新局部测量时
需要匹配条件/校准条件函数后再解释个人差异；不要直接扣一个平均差作通用纠偏。
文本时限编辑响应与本人在两个条件中的实际响应是两种对象，不交换名称。
计算追溯和独立心理解释分开验收，先使用前者，后者按D01取得证据。

## 13. 具体回应绑定与读者选择偏移

[命题回应报告](../reports/grain_measurement_foundation/PROPOSITION_RESPONSE_REPORT.md)增加
`frames → bind_reply → response_coordinates → response_profile`。学习型UD提供候选
边，有限声明规则保留主体/模态/条件；源片段定义条件坐标，不从留出标签重估。
裸Yes/No必须绑定唯一具体命题，不把模糊前问、多问、引语自动视为同一个机会。

把两个身份槽分开：发出请求的是`speech_source`，请求别人实施行为的主体是
`proposition_actor`。没有明确受话者的自然`you`不得自动绑定到真实伙伴。
现包只接本人意愿/过去报告目标；当前请求、did省略和修复仍待下一接口。

若A表示某事件能被读取，简单平均得到的是E[z|A=1,c]，与目标E[z|c]相差
Cov(A,z|c)/P(A=1|c)。源句型全对不意味着新句型的缺失随机。
本包第三人称零贡献事件更易漏读，结果虽r近乎相同但MSE上升53.6%。

输出至少保留所有机会、成功读数和未分配条件；未解析不是反对或零分。
有明确[-1,1]单位界时，S总和/N已读/m缺失给完整有限材料均值范围
[(S-m)/(N+m),(S+m)/(N+m)]。未知条件可将未分配U纳入更宽包络。
范围不解决已读值错误/条件误分，不是人口真值CI。源校准、独立编码或新增
定向观察可再缩小范围，不用删掉未解析行制造稳定量尺。

## 14. 同一句话可以改变两本账

[对话双账](../reports/grain_measurement_foundation/DIALOGUE_CLAIM_LEDGER_REPORT.md)连接
原文→命题/话语→版本关系→个人有限分数。纠正过去报告更新我们对旧事件的认识，
同时“让我纠正”可能实施当前话语许可请求；两种记录的时间对象不同。
别把所有当前请求计入同一个求助指标，也别因修复不是任务求助就丢掉整句话。

制作时声明查询目标、被请求的行动和发话者。功能标签可并存，不强制单类，也不把
同轮多标签当多次独立机会。明确复述是alias，更正是replace，撤回删除主张而非负值。
不同前问提供问题局部参照，并不自动证明实际事件不同；物理事件同一性另记。

先使用新身份通道的有限单位，独立来源完成校准后才接个人参数/区间。
旧NLI读数、当前±1主张、话语次数不可共享未核单位或同一常模。
规则覆盖外的自然省略、含糊角色与未映射条件保持缺失，交对应新观察而非用平均人补齐。

现有心理/状态证据按[D01对象盘点](../reports/grain_measurement_foundation/INTERPRETATION_EVIDENCE_INVENTORY_REPORT.md)
使用：旧函数的相关不能移交给新话语指标。先有目标相符的解释编码，再决定名字。

## 15. 跨agent复建反馈：函数定位与未写明的规则（2026-09-13增补）

[D02跨agent构造记录](../results/grain_vector_mechanisms/mechanism_todo_v1/claude_todo_closure_20260913/d02_independent_construction/D02_CROSS_AGENT_CONSTRUCTION_LOG.md)
按§4命令重建两实例（与冻结r1逐值一致，审计PASS），再只依据本手册与库函数为一个
新的两条件对象（任务完成短/长）独立构造并冷计分。构造者带有开发上下文，**不是盲第三方**。
下表补回构造者必须查实现或自行补规则的地方；冻结bundle与r1数值不因此改变。

| 手册中的名称 | 实际模块 | 未写明而需补的规则 |
|---|---|---|
| `calibration` / `restore_units` | `grain_research/pilot_transport.py` | 参数序为(reference, observed, condition)；reference是编写的±1事件方向，observed是NLI边际 |
| `learn_cell_moments` | `grain_research/choice_design.py` | 输入形状[author, 2, event]，各作者每单元事件数相等 |
| `fit_pilot` | `grain_research/pilot_transport.py` | 输入[author, time, replicate]均值，z/p/t共同时间表 |
| `evidence_rows` / `score_records` / `score_person` | `grain_research/contextual_instrument.py` | 边际字典键为`sha256(query+"\0"+text)`；`clock_unit`必须等于`spec.clock_unit` |
| `calibrate` / `build_meter` / `read_events` / `conditional_function` / `predict_function` | `grain_research/joint_conditional_observation.py` | §8增补对象 |
| `learn_response` / `compile_function` / `apply_function` | `grain_research/response_sharing.py` | §10增补对象 |
| `fit_source` / `fit_expanded` / `moments` / `choose` / `score_weights` | `grain_research/new_condition_design.py` | §6增补对象 |
| `fit_source` / `active_records` / `square_root_append` | `grain_research/evidence_history.py`、`evidence_history_innovation.py` | §9增补对象；同名`fit_source`与新条件包不同 |

bundle必需键（手册此前未列）：`spec`（含`id/kind/query/clock_unit/target_names/target_description/conditions/allowed_scope/positive_hypothesis/negative_hypothesis`）、
`calibration`、`reader{model,revision}`、`seed`、`pilot_reports_per_occasion`（过程实例用）、`main`、`bootstrap`（≥2）、`pilot_authors`、`construction_input`。
bootstrap只规定“按作者重采样12次”，未固定随机数规则；不同构造者的构造方差项会不同，
应把种子规则写入spec后再比较区间。冷入口假定bundle位于`<run>/instruments/`且`<run>`可能含
`reader_cache.json/raw/truth/application`（用于访问拒绝探针），不在该布局下时探针为空但仍可运行。
新对象若无真实来源作者，只能声明模拟pilot；这是构造者规则，不是手册授权的观察设计。

## 16. 条件来源与开放条件分配（2026-09-14增补）

[全计划逻辑研究账](../reports/grain_measurement_foundation/GRAIN_ALL_PLAN_LOGIC_FINDINGS_V1.md)：
B2/B3 在自然材料上的两个事实——自然问答对中问句含条件从句仅14%（57/400），
声明目标（求助）自然出现率0/400。因此制作时**条件来源应设计为记录的线程/社区/时间元数据**，
不能指望从问句从句读取；能从从句读取时，用`grain_research/open_condition.py`：
来源从句向量→一维LDA后验π_e→期望设计行[1−π_e, π_e]与膨胀方差进入目标核（`score_person_open`）。
结果：水平目标与oracle相同，条件半差r损失约四分之一；高维各向同性后验欠置信（反例）；
高置信的从句级误分不能靠后验膨胀表示，须另加来源留出错误率项（待做）。
自然问答的可计分份额（极性回答子集）约.44，见`grain_research/dialogue_ledger_generic.py`的
极性前缀绑定规则（省略含否定者标记不翻转）。
