# 真实文本的主体、来源与事件图

2026-09-13；C03/C16的标准句法参照已实际接入，连接C06原文地址与C14语义函数。
规格：[Source/Event Plan](../../docs/GRAIN_SOURCE_EVENT_GRAPH_PLAN_V1.md)。

## 1. 发现与用途

**“谁是语法主语”“谁提供这个说法”“条目正在问谁的哪个行为”必须拆开。**
这次不再扩展手写引语模板：使用冻结Stanza模型解析完整真实评论，形成原文可寻址
的依存关系，再通过透明规则保存事件参与者、来源链与范围。GRAIN各切片的计分不变。

699条评论、2,844个解析句、51,097个解析词，产出7,595个事件候选。
其中745个带报告/认知来源嵌套、343个位于意愿谓词之下。这些不是都应相加的
“人格证据”：例如想做、被别人描述、自己做过，属于不同的证据通道。

最有决策价值的真实对照：旧80条来源模型复核中，7条被标为`other`的句子有6条
仍然含作者自述事件。因此“句中存在I/自述事件”不足以完成条目归属。
**下一步的必要对象是条目q与事件e的链接，而不是一个全句self/other开关。**
旧类别是模型意见、不是人工金标准；这里报告共存现象与接口缺口，不报分类正确率。

## 2. 真实执行范围

| 样本 | 原评论 | 有事件候选的评论 | 事件候选 | 单数作者主语候选 | 来源嵌套 | 意愿嵌套 |
|---|---:|---:|---:|---:|---:|---:|
| 每作者固定取一条，共624人 | 624 | 623 | 6701 | 1388 | 642 | 295 |
| 旧80条模型复核所在评论 | 80 | 80 | 945 | 352 | 109 | 52 |

两组重合5评论，总计699，不相加当704。624条按comment ID hash及seed916选择，
来自前批冻结native共同预算，不看分数或标签。复核组是此前被筛出的来源样本，
不能拿其频率代表人群。没有事件候选的1条评论仍保留，不称“没有心理信息”。
作者级分数没有由这每人一条评论计算；这是证据接口与覆盖研究。

使用Stanza1.11.0 English EWT、`tokenize/mwt/pos/lemma/depparse`，pos/lemma/depparse
选固定nocharlm包。只下载官方版本权重，合计243,289,162bytes，逐文件校验；
额外依赖Stanza/emoji放在ignored独立目录，未改主venv。CPU2线程、原文不上传。
这是**神经句法解析＋规则化图转换**，不是完全非神经计算，也不是新训练的LLM。

两次独立加载模型的完整解析分别40.07/39.90秒，五类核心输出逐字节一致。
后续纠错只读缓存；最终`source_event_r4`9.78秒、0新parser调用。无问卷标签载入、
无拟合、无MEPS使用、无锁箱打开、无commit/push。

## 3. 输出的数学对象

原文x经冻结解析器产生`D(x)=(V,E_dep,pi)`，pi是词到原文字符地址。
每个事件候选保存：

\[
e=(p,\ A_{\rm grammatical},\ S_{\rm assertion},\ H_{\rm embedding},
   M_{\rm local},\ C_{\rm adverbial},\ \pi_e).
\]

- p：谓词中心，包括动词及有系词/主语的非动词谓词，不等同独立实际事件。
- A：显式主语、被动主语、继承候选、可能的宾语控制、复数we或未解析。
- S：陈述/认知来源候选；保留整个嵌套链，不把最近主语自动当唯一真实来源。
- H/M：父谓词与局部否定、情态、意愿、频率线索。父句否定不直接翻转子句事实。
- C：状语从句及mark边，例如if/when/because；不是自动发现真实实验条件。
- pi：原始字符与中心词地址，再接到五视图单位；缩约词展开保留原包围token地址。

本批实算这些图；`factuality`与`query_relevance`都明确保存为`not_inferred`。
没有把名词/情绪词直接当人格因子，也没有根据语法将想做的事当已做的事。

### 总体输出清单

| 描述性属性 | 事件候选数 |
|---|---:|
| 显式语法主语 | 5014 |
| 继承主语候选 | 715 |
| 主语控制歧义 | 163 |
| 其余主语未解析 | 1703 |
| 被动主语 | 279 |
| 报告/认知来源链 | 745 |
| 意愿嵌套 | 343 |
| 局部否定线索 | 714 |
| 情态助动词线索 | 811 |
| 带状语从句关联 | 2106 |
| 频率线索 | 121 |

除主语来源分类外，各行可以重叠；2106不是2106个独立情境。某个状语从句同时
连接主事件和从事件。频率词不自动证明trait，过去时不等同评论时间，完整事件
区间可能不连续；必须保留词ID列表，不能只靠包围字符跨度重新拼出“独立句”。

## 4. 真正修正的适配错误

真实句子中出现“我被告知自己如何”。解析器已正确给出`nsubj:pass`，但r1/r2的
规则把其主语用作说法来源。这是我们的适配错误，不是模型不能区分语态。

修正：被动报告优先找明确`by/obl:agent`来源；无来源则保留`passive_source_unknown`。
全量缓存重算改变8个事件来源，6个原误归为作者本人、2个误归为其他语法主语。
不改变该事件“谈到谁”，只改“这个说法来自谁”。其中一个引语自述候选随之变为
来源未解析。新增两项被动来源回归测试。

另修正命名：全部`advcl`不能叫“条件”；字段改为`adverbial_clauses`，角色为
governing/subordinate event，保留原mark给下一层语义解释。没有修改实际解析边。
这两项是运行后发现的开发修正，旧r1/r2和过渡r3保留；不包装成事先证实的结论。

## 5. 与完整GRAIN/语义条目相接

| 原五视图 | 可以精确找到中心词的事件 /7595 |
|---|---:|
| 整篇 | 7477 |
| 原评论 | 7477 |
| 句群160 | 7477 |
| 跨评论128 | 7475 |
| 评论内128 | 7474 |

未接到冻结预算的事件仍保留原文。中心词被保留不等于完整语义证据都在该单位，
须结合上一批sentence coverage/完整原评论读取。五视图地址链接不产生五个独立事件。

抽象例子：“我很感谢你帮助别人”含作者的感谢事件和他人的帮助事件。
若条目问“作者愿不愿帮助别人”，不能拿前者的I为后者授权。
类似地，作者被人称赞某特性，是“他人描述作者”的证据，不自动等于作者自我评价。

下一层构造应以`(作者u, 查询q, 事件e, 证据通道c, 条件z)`作累积单位，
保持GRAIN表示phi与同址语义权重`w_q(e)`的连接。自述、他述、意愿、假设和旁人
语境分别积累；相关性由查询决定。`w_q(e)`尚待执行，不把语法覆盖率当它的准确率。
这落实了“保留信息并按条件解释”，而不是把不是自述的内容都当噪声删掉。

## 6. 前人方法的继承位置

[Qi et al.2020 Stanza](https://aclanthology.org/2020.acl-demos.14/)提供已经训练好的
词法/依存模型，本次真实使用其固定实现；不是只在参考文献中列名字。
[UD ccomp](https://universaldependencies.org/u/dep/ccomp.html)、
[xcomp](https://universaldependencies.org/u/dep/xcomp.html)、
[advcl](https://universaldependencies.org/u/dep/advcl.html)提供边类型，控制与共指仍需
区别处理，不能见到xcomp就机械继承上层主语。
[Sauri & Pustejovsky2012](https://aclanthology.org/J12-2002.pdf)把来源视角、极性和
确定程度分开并沿嵌套结构处理。本批继承这种对象分工，**不是完整De Facto复现**：
原完整词汇资源/合成事实判定未接入；当前有限报告/认知/意愿词表是显式适配假设。
论文也区分文本声称与世界事实；句法成功本身不会证明作者实际做过什么。

## 7. 验算、复跑与剩余工作

独立计算脚本（非独立研究者）核对51,097个词地址、7,595条事件路径、37,975个
事件×视图行；两次完整解析输出相同，被动来源改动逐项保留。13项新测试及
相关集合共55项通过，Ruff通过。原参与者文本和事件表全部在ignored results。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
uv pip install --python "$PY" --target results/grain_vector_mechanisms/resources/stanza_py111 --no-deps 'stanza==1.11.0' 'emoji==2.15.0'
# 首次环境准备；已有库存时不重复下载：
"$PY" scripts/fetch_grain_syntax_reference.py
"$PY" scripts/run_grain_source_event_graph.py --output results/grain_vector_mechanisms/all_plan_closure_v1/source_event_next
# 仅重建图，可加：--reuse-parses results/grain_vector_mechanisms/all_plan_closure_v1/source_event_r2
"$PY" scripts/audit_grain_source_event_graph.py
"$PY" -m pytest -q tests/test_grain_source_event_graph.py
```

正式本批输出`source_event_r4/`：contract、private_parses、private_graphs、private_events、
private_annotation_event_join、private_event_unit_join、old_annotation_support、
audited_sample_support、summary和audit。审计脚本核当前r4及保留的r1/r2重放。

下一步C03/C14/C16：实际比较全句语义匹配与**查询条件化的事件选择/来源解释**，
优先复用现成NLI/语义角色资源，不靠本批80条模型意见调出“高准确率”。
开放共指、跨句话语来源、目标行为的语义角色、事实性与独立解释效标仍须分别解决。
当前已补成真实文本语法证据接口；总计划与goal保持ACTIVE。
