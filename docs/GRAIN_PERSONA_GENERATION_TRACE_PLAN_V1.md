# Persona 条件生成、盲读回与内部路径定位：实验计划 V1

2026-09-14。用户新构想；Codex 与现有 Claude 会话只读研讨。此文件是设计，不是新实验结果；不自动启动长生成或训练。

## 目标与既有工作的差别

先在已知 persona 操纵的模型世界中定位“指令如何成为表达、表达如何支持属性读出”。不用真人标签做第一步。生成条件是已知标签，但不是已验证的真实人格，也不保证模型服从。

已有 `GRAIN_PERSONA_RECIPROCITY_P1_PROTOCOL.md` 使用预写候选整句似然和 G/R 交叉干预；不是自由文章生成后、剥离 system prompt 的盲读回。P1B-G 已检验角色/机会/问法和局部向量迁移，发现查询依赖与无关事实漂移。新实验复用这些教训，不原样重跑，也不恢复旧 P1U 队列。

新对象：

`P (persona assignment) -> H_gen -> X (generated text) -> H_read -> S_q (trait readout)`

条件 C=任务、情境、输出长度约束与交互角色；W=persona 提示措辞；U=采样随机性。

`X ~ p_theta(X | P,C,W,U)`；研究的是 `p(P | X,C)` 和该链接的干预敏感度，不是从 X 唯一还原 W。不同指令可产生同样文本，读写不预设互逆。

## 分阶段执行

### A. 小型自由生成与盲读回

- 先选外向倾向、神经质倾向两个轴（已有对应读出）；一次只改变一个轴，高/低加中性对照，不将16种类型混成第一轮。
- 固定主题、场景、叙述人称、篇幅范围、模型/模板、采样配置。保留表达选择差异，不把生成后行为全部回归掉。
- Pilot：2轴×2极×4任务情境×2措辞族×2种子=64篇，加4情境×2种子中性=8篇，以及2种非人格指令×4情境×2种子=16篇，共88篇；每篇上限192新tokens。非人格指令长度匹配，用于检查普通文体/指令影响。相同随机种子是配对手段，不保证分叉后的词级对齐。
- 若能生成非平凡差异，扩展为5个条件（4极+中性）×6任务情境×3措辞族×3种子=270篇。任务与提示族分别保留发现、开发、最终评估部分；层/头/方向选择只用发现与开发。
- 提示只规定特质或含义，不直接列出想让分类器识别的标点/代词清单。包含“人格词明确命名”和“不出现人格词的含义描述”两类；禁止在文章里自报人格标签。
- 不按成功程度丢弃文章；保留服从性、显式标签出现、截断和主题偏离审计。原文与标签词遮蔽版都评估，后者仅为线索敏感性分析。
- 盲读者新建上下文，仅见文章和固定查询。绝不带入作者system prompt、生成KV cache、文件名或条件ID。
- 比较TF-IDF、现有GRAIN、冻结向量+线性探针、LLM查询读出；探针训练按任务/措辞族分离，所有预处理在发现侧fit。

主要结果：分条件文本差异、跨措辞/情境balanced accuracy/AUC、条件内置换对照、trait选择性、同persona采样方差与高低persona位移。生成种子不是独立人类作者。

### B. 写入路径定位

- 使用本地可hook的开放权重模型，优先已有 Qwen2.5-1.5B-Instruct/revision；先检查依赖和权重身份。API文本生成不用于声称注意力头定位。
- 自由生成测总体行为效应；固定同一续写前缀做teacher-forced对照，以对齐token位置。二者分别报告，不把后者当成自由生成的总中介效应。
- persona词span、末端共有提示位置、早期生成位置分别记录；跨prompt长度比较时匹配角色与token对齐，记录位置编码差别，不能按绝对索引乱patch。
- 残差流先粗定位，之后只拆出候选层的MLP输出与attention head输出贡献。attention权重图仅用于定位候选，不等同作用大小或思考链。
- 做高低条件激活替换、去除候选方向、正负小剂量和恢复；对照零改动/self-patch、等范数随机方向和非persona任务。
- 主要统计为留出条件上的配对输出变化和区间；不因总效应接近零而用归一化恢复率制造大值。

### C. 读回路径与跨任务作用

- 从生成文本盲读条件，定位文本证据位置到trait查询分数的变化；标签字面词与内容/来源归属分别检查。
- 生成方向 v_G 与读回敏感度 r_R 比较的是 `r_R^T v_G` 及有限干预，不要求两者余弦接近1。
- 在同一模型的兼容位置检验 G->G、G->R、R->G、R->R。跨层需明确传输映射；不同模型hidden坐标不直接相减。
- 能读出但不能控制、能控制但不能读出、二者都有但方向不同，均是不同机制结果，不一律写成失败。

### D. GRAIN 接入

- GRAIN 描述生成文本的行为配置，生成/读回激活提供另一视图。比较它们各自是否携带分配信息，以及是否解释同一干预的变化。
- 用不重叠任务文本A定义GRAIN邻域，在B文本检验路径效应是否更相似；邻域参数只在发现侧确定。
- 先报告条件内配对作用的分布，再聚合；避免平均向量相消被误读为无机制。
- 不要求GRAIN已经是完整人格量表；此实验也不把模型模拟者当作真人个体。
- 增加一个干预端点：在相同任务/种子下重新生成短续写，分别计算原文与干预文的GRAIN坐标，得到 `delta_g_intervention`。teacher-forced的delta log概率仍单独报告，不能将标量概率差直接投影到GRAIN向量上。
- 使用已有来源参照冻结的GRAIN坐标；先核验最低文本量/有效slice要求。192 tokens不够时只报告可计算通道，不强行输出完整GRAIN分数；较长多文本GRAIN桥接另定批次。
- 小样本位移距离使用来源侧确定的低秩空间或收缩协方差，不直接求高维样本协方差逆。真实作者参照几何仅作兼容性比较，不自动提供心理效度。
- 共通通道与特异/残差通道都是候选信号；不预设“人格只能在共通部分”。增加自由选题支臂可在后续保留choice机制，首轮固定任务只回答条件内表达问题。

## 资源与停止点

本轮只研讨与写计划。执行从88篇pilot和少量保存激活开始，按批落盘；不要保存全层全token全头张量后再找问题。机器路由沿用192.168.1.20训练/重计算、192.168.1.30评估，但内部干预必须先核验PyTorch/hooks支持；Intel llama服务不能自动视作可hook后端。已有Mac本地开放模型可作低成本pilot，不新增LoRA。

阶段结果没有区分性时先检查操纵是否改变表达、任务是否容许差异、读出是否只见输出；一次有界诊断后决定换一个任务/操纵，而非无限扩大样本。若识别有效而因果干预无作用，保存识别结果并修订作用路径假说。

## 产物约定

新目录 `results/grain_vector_mechanisms/persona_generation_trace_v1/`；保存设计manifest、提示族划分、生成原文与token、checkpoint身份、选定位置激活、盲读预测、干预效果、GRAIN桥接分析。新脚本候选为 `prepare_grain_persona_generation_trace.py`、`run_grain_persona_generation_trace.py`、`analyze_grain_persona_generation_trace.py`。不覆盖旧P1产物。

## 直接先行研究

1. Chen et al. (2025), Persona Vectors: https://arxiv.org/abs/2507.21509 ; official code https://github.com/safety-research/persona_vectors 。可复用正反条件激活差及steering工具；不能把其模型character traits当作真人Big Five证据。
2. Poonia & Jain (2025), Dissecting Persona-Driven Reasoning via Activation Patching: https://aclanthology.org/2025.findings-emnlp.1335/ 。persona对客观任务的MLP/MHA作用定位；对象不等同本文的心理trait自由生成。
3. Feng et al. (ICLR 2026), PERSONA: https://proceedings.iclr.cc/paper_files/paper/2026/hash/f66ce276436c4cd151535b03841ac713-Abstract-Conference.html ; https://github.com/xcfcode/persona 。已有trait向量提取/组合/干预，不能宣称首次发现persona方向。我们的候选增量是自由生成后的盲读回、跨任务干预对应和GRAIN条件化群体结构；原创性需由结果与逐项比较确认。

## 研讨状态

Claude已完成三轮只读讨论，核对了旧P1及先行研究，并在第三轮明确接受数学空间和解释边界修订。双方同意把自由生成/盲读回与旧候选似然实验分开。Codex修正了Claude原提案的样本量算术，缩小为88篇pilot，并拒绝把残差通道等同伪影、把真实作者几何重叠等同迁移有效。研讨记录见 `GRAIN_PERSONA_GENERATION_TRACE_SEMINAR_20260914.md`。

## 执行状态（2026-09-14，任务 CLAUDE-PERSONA-GENERATION-TRACE-20260914-01；持续更新）

结果与产物：`results/grain_vector_mechanisms/persona_generation_trace_v1/{pilot,expanded}/`；可读报告
`reports/grain_vector_mechanisms/PERSONA_GENERATION_TRACE_REPORT_V1.md`；代码 `grain_research/vector_mechanisms/persona_trace.py`、
`scripts/{prepare,run,analyze}_grain_persona_generation_trace.py`；测试 `tests/test_grain_persona_generation_trace.py`（8 项）。
本地写手/读者模型 = Qwen2.5-1.5B-Instruct（MPS FP32，`model_manifest.json` 记录权重哈希）；外部盲评 = DeepSeek `deepseek-flash`
（经 `/models` 核验的服务端 id；`.env` 配置名 `deepseek-v4-flash` 不是服务端 id，已记录），只发送合成文本，用量记入 `api_accounting.json`。

| 阶段 | 状态 | 要点 |
|---|---|---|
| A 88 篇 pilot + 盲读回 | 完成 | 无标签泄漏/拒答；外向性分配可被文本-only 读回（本地 AUC .703 p=.003；外部盲评 .711 p=.006），神经质更弱（.602/.705）；读者交叉轴 <.5 = 评者 halo；位移在均值上出现、单篇与采样离散同量级 |
| B 写入路径定位（teacher-forced） | 完成 | persona 位置残差补丁：block_00–08 恢复 1.00，block_12–14 .74–.84，block_16 .15/.32，block_20 起 0；发展侧复现；随机等范数 .2–.4（移除份额）、自补丁 0；attn/MLP/单头单层增量 ≤ .11，差异是累积的 |
| B′ 读取侧（路径补丁，span 保持低极） | 完成（层 + 头） | 下游读取集中在第 16 层（单层 .40，发现 .55）、第 20 层 .19、第 12 层 .12，单层和 .91，全层 1.000；主承载头 L16.head07 .21、L20.head05 .12；首版原位 K/V 替换被 span 内自传播污染（layer_00 = 1.0），存为 `*_v0_confounded` |
| C 读回路径与跨任务作用 | 完成（有界） | 读回是条件性的：任务（opinion/advice 允许，neighbor/email 不允许）× 措辞（named 承载，anchored 不承载）；GRAIN4 探针措辞族留出 .58–.66（anchored 测试 p=.012），情境留出 .51 |
| D GRAIN 端点 | 完成（pilot） | block_14 prefill 补丁后自由重生成 96 篇：外向性盲读 +.256（参照 +.229，随机 +.094）；GRAIN 坐标位移干预 > 随机（两轴 4/4 对），合并方向余弦 .82，但单轴方向估计不稳 |
| 扩展 270 篇（3 措辞族 × 6 情境 × 3 seed） | 完成 | 外向性整体 AUC .59/.62（p .007/.003），pilot 同款子集复现 .68/.72；效应由 named 措辞（.76/.74）与 opinion/advice 任务（.84/.82、.70/.77）承载，anchored 措辞与 neighbor/email 任务≈chance；神经质整体 chance；评估情境无区分；12 篇 AI 声明（11 篇 review seed 11）已计数、主分析不剔除 |

未做/边界：不追踪 DeepSeek 内部；无 commit/push/tag/锁箱；无旧队列；神经质轴的读者端无信息，端点对该轴不可判。
下一步（未执行）：在无读回的情境/措辞（neighbor、anchored）上重复 block_14 端点，检验"补丁搬运了状态但文本不变"的边界；
更多对（而非更多 seed）以稳定 GRAIN 方向估计；对 L16.head07 / L20.head05 做评估侧情境复核。

结论（2026-09-14）：实验按 handoff 完成到可判定的程度。外向性 persona 的写入—读取—端点链条在 Qwen2.5-1.5B 上成立且有界
（named × 自我表达任务）；神经质与 anchored/neighbor/email 组合为明确的失败反例，已保存。
