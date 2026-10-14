# 证据身份：内容相近不等于测量对象相同

2026-09-13。M01/M09新接口及有限反例交付；原有11项有限验收不变。
[预先固定的计划](../../docs/GRAIN_EVIDENCE_IDENTITY_PLAN_V1.md)。

## 1. 本轮达成什么

**同样谈“求助”，意愿、过去行动报告、偏好、能力、当次体验和正在提出的请求，
必须保留为不同的证据坐标。** 它们可以共同出现，不是六种互斥人格。
本轮实际完成原文到十个固定查询读数、话轮/前文地址、来源事件关系、受控个人
表达组成函数。没有改原GRAIN分数，没有把过去时自动解释成真实行为发生。

最直接的数学对照来自实际文本读数：两组模拟表达分布的内容分数均为
**0.9933139334**，但“行动报告份额减意愿份额”为 **+0.165707/-0.165707**。
它们并不是人格完全相同；是这个内容标量没有保留区分该目标所需的信息。

相反，通用“证据身份”NLI提问并未成为可靠的替代读者。它在来源内能形成区分，
却在新措辞上失效；简单的来源TF-IDF参照保留更多身份信息。该阴性结果原样保留，
不重写提示词、不删困难例、不调整秩或正则去抢救本次结果。

## 2. 实际执行与边界

| 材料/运算 | 本轮数量与用途 |
|---|---|
| 公开机制文本 | 24单身份句：6类各2来源/2留出措辞；12来源/否定/引语/混合挑战；16对话回复，共52条 |
| 自然原文 | 固定hash选12名作者/12评论的12句，1,326字符、46既有事件候选；不看问卷或NLI结果选样 |
| 主运行 | 800实际本地NLI对，11.224秒；64个新原文记录，16对话另有reply-only对照 |
| 结果后定位 | 6个单次诊断对，2.040秒；不改变原实验、源标定或主结果 |
| 模拟计分 | 192个新作者参数组，每人48独立抽取的表达；来源/留出是配对措辞，不当384个独立人 |
| 训练/隐私 | 仅来源12句拟合TF-IDF及读数均值；无神经训练、无下载、无MEPS使用、无私人原文外发 |

观察器为固定`cross-encoder/nli-deberta-v3-base`，revision
`6c749ce3425cd33b46d187e45b92bbf96ee12ec7`，CPU/float32/4线程，禁止截断。
自然文本仅重新读取已有许可PANDORA材料；没有新解析器调用或自然标签拟合。
角色`Respondent/Partner`由记录提供，未假称从无角色文本中自动发现。

## 3. 从原文到不同个人函数

接口在[实现](../../grain_research/evidence_identity.py)中：

`requests(raw turns) -> fixed NLI -> evidence_record -> declared target operation`。

每条记录包含当前原文及前文的hash/字符范围，独立查询的蕴含/中立/矛盾、
已有事件图链接。前一话轮只是**候选依赖**，没有自动断言它就是被修正的精确片段。
地址是整话轮，不是attention归因或词级因果支持。原文身份和心理解释分别保存。

六个身份坐标的总和不限制为1；一句“偏好求助，但觉得无法开口”可以支持多个。
开放自然文本**不**套六类混合解码器，不输出被迫归类的人格分数。

在额外声明的、一次表达恰有一个实验身份的有限世界中，才有：

\[
\pi_u\in\Delta^5,\quad
B_{mj}=E[h_j(X)\mid m,\text{source}],\quad
E[\bar h_u]=B^\top\pi_u,\qquad
\widehat\pi_u=\arg\min_{p\in\Delta^5}\|B^\top p-\bar h_u\|^2.
\]

`pi`是给定采样设计下的表达身份组成，不是实际行动发生率。
目标`q'pi`能由观察与归一条件确定，当且仅当
`q`在`[B';1']`的行空间中。若不在，存在`v`使`B'v=0,1'v=0,q'v!=0`；
足够小的`pi+/-epsilon*v`就是同观察、不同目标的例子。
这是标准线性代数，不是本研究新发明的识别定理。

本包的具体内容查询只有两个独立约束，不能识别全部六种组成；上面的精确碰撞
使用实际来源B构造，不是人为把内容读者设为常数。六个通用查询的数值秩虽为6，
条件数约**13,734**，最弱奇异值`.000366`；数值满秩并不保证语义可分或措辞稳定。
TF-IDF相应条件数4.44。本轮不据这些结果重新选择维数。

## 4. 计分结果：不得当人格预测相关

以下是192个模拟表达组成向量的六坐标平均结果，一种子；没有人类外部效标。

| 观察/源标定 | 来源措辞 mean r / MSE | 留出措辞 mean r / MSE |
|---|---:|---:|
| 单一内容NLI | .310 / .011055 | .182 / .022080 |
| Word TF-IDF | .915 / .002734 | **.619 / .011507** |
| 六个通用身份NLI | .644 / .012747 | .166 / .092630 |

直接知道每次真实实验身份时，48次抽样本身的平均MSE是`.002648`。
因此TF-IDF仍有措辞损失，NLI没有因“更语义化”自动胜出；其能力/当前请求维度
留出r=.769/.572，但体验维度r=-.442，不能只摘正面的维度报告。
该有限decoder也不能从“喜欢谈求助”推出“真实会求助”，不把两者算成一个偏差项。

## 5. 对话上下文有作用，但抽象身份读者不可靠

15条有对应预定查询的对话中，原假设得到NLI最大类别支持的数量为
reply-only **8/15**，含前文 **12/15**。这是候选支持，不是独立人工标签准确率；
多种功能可以并存，不能用top-1错分率解释多标签能力。另一条接受对方观点的
例子不属于“接受帮助”查询，单列保留。部分`Yes, please.`回答事实问题的材料
语用上生硬/可多解，不给这几条强加唯一正确含义。

仅看较自然的同一`Yes.`：前问为提供帮助时，接受帮助蕴含`.9949`；前问问能力时，
能力蕴含`.9893`；问未来意愿时，意愿蕴含`.9963`。但问过去是否求助时，
行动报告蕴含只有`.000404`。显式修正、反驳分别有`.9930/.9962`的相应候选支持。
模型改变了读数，不等于每个改变都遵循正确的证据绑定。

**单次结果后诊断**定位到抽象改写，而非交换label来救结果：

| 同一原文 `I asked you for help yesterday.` 的查询 | 蕴含 | 矛盾 |
|---|---:|---:|
| 具体：`The respondent asked for help yesterday.` | .998065 | .000089 |
| 具体否定 | .000028 | .999927 |
| 抽象：`The respondent reports an action that they themselves already performed.` | .000406 | .998313 |

另外三条普通蕴含/矛盾/中立控制均符合设计标签；抽象查询再次运行复现失败。
这不证明整套NLI没有能力，而是**本组元描述不应直接充当事件类型运算器**。
后续若发展观察器，应接现有带主体、模态、时态的具体事件命题，不重扫泛化查询。

互动计分还有另一层：给定机会o的反应`f_u(o)`与机会出现份额`pi_u(o)`分开。
两个作者都为`f=(.8,.2)`，机会份额为`(.9,.1)`和`(.1,.9)`，合并读数为`.74/.26`；
同一参照份额`(.5,.5)`下都为`.50`。公式继承条件标准化；这不是新真人效应。
没有观察到机会就不制造分母。以往M09共同输入/反馈识别结果仍有效，本轮没有重跑。

## 6. 自然应用、解释与验收

12句自然材料已生成120个查询读数、46事件候选链接和原文地址。
观察器对其中11句支持偏好、7句支持意愿，而行动报告0句；鉴于上述失效，
**这些只用于检查接口，不作为真实比例或作者画像**。材料无独立语义编码，
缺失前文时依赖字段保留未观察，不靠单句补造回复对象。

| 父项要求 | 本轮交付 | 保留的实际缺口 |
|---|---|---|
| M01原文证据/条件接口 | 多标签身份、原文地址、来源标定/内容参照、精确目标碰撞及留出损失 | 开放条件含义、实现状态和嵌套来源不能靠这组通用NLI自动解决 |
| M09互动接口 | 显式角色/前文候选、同回复异前问、机会/反应分账，继承已有数值识别 | 精确依赖片段、行动报告与当次言语行为的可靠自动区分 |
| M13解释接点 | 先指定所测证据身份、目标及分母，再说明个人函数 | 下一包对匿名GRAIN配置与窄响应给完整解释卡及证据改变对照 |

本轮有限实验COMPLETE，M01/M09父项仍partial；已完成11项，不将语义读者失败
掩盖为完成数增加。M13可以继续，不受当前观察器必须完美这一条件阻塞。
下一步先完成可计算解释链，随后以事件绑定规则承接语义缺口；不再追加本包精度实验。

## 7. 文件、运行与验证

产物：[本轮目录](../../results/grain_vector_mechanisms/mechanism_todo_v1/evidence_identity_r1/)。
公开刺激`public_bank.csv`；查询读数`readings.csv`；源decoder/识别条件
`source_decoder.json,identifiability.json,content_collision.json`；模拟逐人值/指标
`simulated_composition_predictions.csv,composition_metrics.csv`；
`dialogue_comparison.csv,identity_contrasts.csv,mechanism_summary.json`。
`private_*`含许可原文/事件地址，保持gitignored，不发布；其余结果也不自动推送。

新代码：[evidence_identity.py](../../grain_research/evidence_identity.py)、
[runner](../../scripts/run_grain_evidence_identity.py)、
[独立数值复核](../../scripts/audit_grain_evidence_identity.py)、
[测试](../../tests/test_grain_evidence_identity.py)。

```bash
PY="/Volumes/mobile3/projects/project persona/.venv/bin/python"
"$PY" -m pytest -q tests/test_grain_evidence_identity.py tests/test_grain_source_event_graph.py tests/test_grain_relational_evidence.py
# 新目录复跑，依次 prepare / observe / analyze；不覆盖已有r1。
"$PY" scripts/run_grain_evidence_identity.py prepare --output /tmp/grain_identity_replay
"$PY" scripts/run_grain_evidence_identity.py observe --output /tmp/grain_identity_replay
"$PY" scripts/run_grain_evidence_identity.py analyze --output /tmp/grain_identity_replay
"$PY" scripts/audit_grain_evidence_identity.py audit --output /tmp/grain_identity_replay
```

42测试PASS；800概率/输入对、12自然原文地址、3来源矩阵、1,152凸优化最优性条件、
36指标独立计算PASS，最大KKT残差`1.386e-6`；7冻结文件未变。复核采用同agent的
另一条数值计算路径，不冒充D02独立研究者复建。Ruff及diff whitespace检查通过。

## 8. 继承出处与研究位置

来源相对的事实性与嵌套范围继承
[Sauri & Pustejovsky,2012](https://aclanthology.org/J12-2002.pdf)；
话轮功能、功能依赖与修正的参照片段继承
[Bunt et al.,2020](https://aclanthology.org/2020.lrec-1.69.pdf)。
本地不是完整FactBank分类器或ISO标注实现。新工作是把这些区别接到GRAIN的
观察量、来源标定、个人目标、缺失分母和解释规则，并实际检验错误压缩/语义改写
如何改变所测对象；不把句法分类、约束最小二乘或行空间条件署名为新数学定理。
