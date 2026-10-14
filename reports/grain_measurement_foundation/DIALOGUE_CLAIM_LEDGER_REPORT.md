# 对话证据不是句子均值：M01/M09到M02的连接

2026-09-13。本轮实际完成原文解析、版本计分、作用对照和数值复核。
**新增可用机制：同一段话可以既更新旧证据，又实施新的言语行为；两者必须分别记账。**
这不是一次人格相关系数优化，也没有新增真人测量结果。

## 1. 实际材料和结果

14条公开编写对话，共81轮、33种不同原文，固定本地Stanza解析耗时3.253秒（含加载）。
8条主对话交叉两条件、两命题模式和两种措辞；另6条挑战。
条件坐标沿用上一包来源TF-IDF词表、IDF和原型，只transform；无训练、NLI调用、
新人口常模、MEPS访问或外部API。运行前冻结14个文件，原结果不覆盖。

| 检查 | 实测 | 意义 |
|---|---:|---|
| 主对话操作 | 72/72 | 前问、回答、复述、纠正、新参照、撤回进入正确版本操作 |
| 主对话逐前缀计分 | 72/72 | 包括8个尚无证据的空值；其余64个为有限均值 |
| 挑战操作 | 9/9 | 无前文、负问、冲突复述、复杂省略不补造主张 |
| 挑战话语计数 | 6/6 | 本人直接请求与他人引语分开 |
| 原始主对话通用请求计数 | 48/72 | 24个前缀的预期与计数范围不一致，保留原结果 |
| 结果后目标分账 | 求助9条、话语许可4条 | 13条候选全部保留，不以筛掉问题来提高指标 |

72个前缀不是72名被试，33种原文也不是独立自然语料样本。
规则使用明确的英语引用短语及辅助词，预期操作只进入评价器，不进入生产解析函数。
这说明所声明的机制可运行，不是学得了通用语言理解。

## 2. 为什么需要历史，而不是更多平均

同条件同模式的主序列如下：

| 当前输入 | 活跃主张 | 有限分数 |
|---|---|---:|
| Yes, I did. | 旧事件正向主张 | +1 |
| To repeat my previous answer: Yes, I did. | 同一主张的来源别名 | +1 |
| I need to correct my previous answer: No, I did not. | 替换旧版本 | -1 |
| 新时点问题后的Yes, I did. | 旧事件负向+新参照正向 | 0 |
| I withdraw my previous answer. | 删除刚刚的主张，保留旧版本 | -1 |

用同一组原文辅助词极性，忽略版本的全部均值、最后极性和版本账的64个非空前缀MAE
分别为**.958333、.875、0**。这是明确编写目标下的机制对照，不是自然预测精度提升。
复述不增独立证据；纠正改变对旧事件的认识；新时点才新增问题局部事件参照。
新问题本身仍不证明真实世界里发生了不同事件，输出保留`physical_event_identity_verified=false`。

令A_t为当前活跃主张集，m为意愿或过去报告，c为来源定义的条件：

\[
\hat f_{m,c}(t)=\frac{\sum_{j\in A_t}1[m_j=m,c_j=c]z_j}
 {\sum_{j\in A_t}1[m_j=m,c_j=c]}.
\]

无分母时输出缺失。alias保持A；revise替换指定父主张；retract只删除父主张。
同一事件的不同报告相关性仍由M02的工作协方差处理，本次有限均值不是新的概率后验。
旧NLI源模型不能不经单位标定就直接作用于新的±1命题通道。

## 3. 本轮有用的新细节：修复本身也是话语行为

原始失败来自四条主对话中的**Let me correct my previous answer**：
解析得到`let`请求候选，同时显式引用规则执行`revise`。随后6个前缀都累计了这个候选，
形成24项不匹配；不是24个独立解析错误，只有两种相关原文。

原计划将“通用请求候选数”对照“任务求助次数”，范围不一致。
该表达可能是在请求发言许可，并非完全错误的语言观察。
结果后伴随模块按明确目标分别记`task_assistance_candidate`、
`discourse_permission_candidate`和`explicit_previous_answer_repair`。
不删除原候选、不更改原r1，目标求助的72个主前缀与编写预期一致。
**这是一项结果后的范围修订，不作为新的独立泛化检验。**

“Please help me”中发话者是本人，被请求去帮助的行为者是对方；后者是否实际帮助未知。
本人当次请求、关于本人意愿的报告、实际行动报告是不同计量对象。
同一话轮的多个功能可以并存，不能强制softmax成单一人格类型或当作独立观测。

这里继承[ISO对话依赖/语义内容分层](https://aclanthology.org/2020.lrec-1.69/)，
不是重新发明对话行为分类。省略仅继承
[Dalrymple等的语义约束思路](https://arxiv.org/abs/cmp-lg/9503008)，未实现其高阶合一算法。
集合更新与有限平均是标准运算；本地贡献是将原文依赖、双重行为身份和个人计分更新接通。

## 4. 复核和范围

56项相关单元测试通过；14对话缓存重播完全一致；114个唯一原文地址核对。
另一套直接字典替换与和式重算324个通道前缀，并核对3个MAE。
14个原冻结文件与18个manifest条目hash一致（9个研究文件及9个卷元数据旁文件），
伴随分账0次新增神经调用。旁文件不算额外实验或数据产物。
这是本agent另写公式复核，不是D02的独立研究者复建。

M01/M09多了可用接口，但父项仍保持部分完成：开放自然文本的事件同一性、
一般省略/模态、未覆盖条件和互动目标仍需限定适用范围与自然验证。
没有把8条公开模板对话提升为通用测量器；不继续围绕这8条做提示词或精度网格。
同一目标的声明、来源和观察类型优先于“高分意味着什么”的解释。

## 5. 文件与复跑

- [冻结计划](../../docs/GRAIN_DIALOGUE_CLAIM_LEDGER_PLAN_V1.md)
- [原文双账](../../grain_research/dialogue_claim_ledger.py)
- [结果后话语目标分账](../../grain_research/dialogue_speech_scope.py)
- [主运行器](../../scripts/run_grain_dialogue_claim_ledger.py)
- [公式复核/分账入口](../../scripts/audit_grain_dialogue_claim_ledger.py)
- [单元测试](../../tests/test_grain_dialogue_claim_ledger.py)、[分账测试](../../tests/test_grain_dialogue_speech_scope.py)

`results/grain_vector_mechanisms/mechanism_todo_v1/dialogue_claim_ledger_r1/`保存
public_bank、parses、dialogue_ledgers、checks、ablations、summary、freeze与manifest；
`scope_followup/`另存原文话语账、逐前缀目标计数、独立数值重播与自己的hash。
原始24项范围不匹配仍在checks.csv，未被新结果取代。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
"$PY" scripts/run_grain_dialogue_claim_ledger.py --output results/grain_vector_mechanisms/mechanism_todo_v1/dialogue_claim_ledger_new
"$PY" scripts/audit_grain_dialogue_claim_ledger.py --source results/grain_vector_mechanisms/mechanism_todo_v1/dialogue_claim_ledger_new
"$PY" -m pytest -q tests/test_grain_dialogue_claim_ledger.py tests/test_grain_dialogue_speech_scope.py tests/test_grain_proposition_response.py tests/test_grain_evidence_history.py tests/test_grain_source_event_graph.py
```

运行位置为独立GRAIN仓库；输出目录必须不存在。上述审查程序依赖预先已安装的本地工具；
没有自动下载模型。旧结果重放无神经调用，第一次主运行需要本地固定Stanza。
