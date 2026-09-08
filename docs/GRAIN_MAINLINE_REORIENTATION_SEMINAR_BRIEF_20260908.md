# GRAIN 主线重定向研讨：从局部读出回到理论连接

2026-09-08。Codex 主持，Claude 独立只读讨论。此任务不运行模型实验，不修改
发布计分器，不恢复旧队列。S25-S26 已完成；S27 只是计划，goal 当前暂停。

## 用户要求与终点

用户醒来后怀疑自主 loop 又围绕精度指标打转，要求双方回顾上下文、全部主要
结果与设计指标，共同决定怎样完成理论构筑。请独立判断，不默认赞同 Codex。

目的不是一个 finished clinical scale，也不是只做文本分类或把 N 的 r 调高。
核心是：真实作者的文本/上下文/记忆，怎样形成可作用于特质判断的向量关系；
能否提炼共同数学构造及大小特质各自的函数，逐步省略 LLM 的语言化推导。
允许使用固定神经编码器，不要求从字符开始全手工计算。人格心理/行为链接
负责解释测到了什么；它不是每个机制实验必须提高的单一优化目标。

GRAIN 的 where/choice、what、who/群体参照、when 不能因转向 LLM 而消失。
topic/situation 不预定义为应删除的噪声。共享路径不等于相同向量方向；
persona 写入与作者特质读出可能有关，但不是自动互逆。
发现 -> 发展/连接 -> 有针对性的验证；不追求无限防御，也不伪造确认性证据。

## 已核对的阶段证据

- 历史 MCD、动态/残差、4W 作为方法基础保留；不是每换模型就从零重来。
- VM03 已做同 token 多重集的角色绑定四格：独立加性表示失败，联合表示及
  乘积核可恢复受控关系。分布式 patch 明显、单 marker patch 很弱。
  新实验必须超出这个合成规则世界，不能重复宣称首次发现关系交互。
- D1: 213 Big5-only、2328 MBTI-only 的完成缓存。native Big5 mean r=.1993，
  配对 E5=.1372；TF native AUC=.6600。它们是开发源，不是新 lockbox。
- D2/D3B: 真实评论选择和向量干预，跨作者共享响应算子有效；GRAIN 邻域存在
  某些作用关联，但条件增益模型没有一致增量。不是所有 identity 加权都成功。
- S0-S4: teacher-only E5/PCA/Ridge 学生 TF AUC=.7173，高于 native .6591，
  低于标签监督 E5 .7537；N 学生 r=.0673 vs native .1765。
  简单 E5 内容/上下文对评论敏感度的解码未胜长度/位置对照。
- S6-S17: 均值保持的 token 配置有作用；query-conditioned KV pairing、注意力
  质量和条件二阶矩可以描述其中部分。仍使用预训练 contextual prefix。
- S18: 统计 beta 的零空间仍能携带模型作用；读出拟合不等于实际使用。
- S19-S20: E_u = integral grad F_u dot d_u；固定24参照作者的平均梯度 g_N，
  在另24作者重建实际效应 r=.9431。189参照外作者 z=mean_v g_N^T d_uv 与
  source N r=.2000，p=.0058，无问卷拟合。无需 upper decoder，仍需 prefix。
- S21-S23: 同人同材料4个数字查询，N/确认、Feeling、数量内容有不同功能。
  经验函数 Gram K_rs=g_r^T Sigma_D g_s；N/确认 .715，不能叫因子相关。
  固定 g_N 的内容正交部分 N r=.1946，总分 .2000；不是纯人格分离。
- S24: 实际配置差的两阶 head allocation 可由冻结 g_N 重建 r=.9543；
  该 allocation 并非全部原始配置效应 C。
- S25-S26: 189人配置读数 r=.1160 vs 配对读数 .2000；96.2% 标签协方差
  保留但方差变成2.749倍。新增余项跨视图 r约.39，N r=-.006。
  这是标准协方差恒等式的有用实例，不是新的普适人格定理。
- S27 计划在24人、240 forwards内定位 C 在后续 query residual 的携带位置；
  最后一层等于 C 是构造恒等式。未执行，需重新裁决其机会成本。

## 读哪些材料

只读 docs/、reports/ 中的聚合材料及必要源码；不要读 .env、results/、原文、
作者ID、逐人量表、MEPS、bridge、lockbox，不执行、不写文件、不调用别的agent。
按需选择下面章节，总阅读约6-10个定向片段，不必全库枚举：

1. docs/GRAIN_PROJECT_RETROSPECTIVE_20260905.md 第1-4节。
2. docs/GRAIN_TWO_SEMINAR_THEORY_NETWORK_V1.md 第5-7节，特别H1-H8。
3. reports/grain_vector_mechanisms/S0_S4_FUNCTIONAL_LOOP_REPORT.md。
4. docs/GRAIN_RELATIONAL_READOUT_THEORY_V4.md 第10-14节。
5. docs/GRAIN_SHARED_AND_TRAIT_SPECIFIC_PATHS_V1.md 第4-6节。
6. docs/GRAIN_QUERY_CARRIER_S27_PLAN.md。
7. reports/grain_vector_mechanisms/VM03_RELATIONAL_BINDING_REPORT.md（需要时）。

默认根目录是 /Volumes/mobile3/projects/Sliced Utterance In-Context Assessment。
阅读旧状态时按时间定位，不能把旧 next 当作现在必须执行的命令。

## 第一轮请交付

中文约1800-2600字，清楚区分实证、标准数学、提出的新机制。

1. 重述用户真正终点。给出有限、可实现的理论v1完成定义，不能要求证明所有
   人格/所有LLM/所有语言，也不能把任何数学恒等式当作终点已经完成。
2. 评价S5-S26是否偏航：指出最有用的三个新连接和仍缺的两三个连接。
3. 独立裁决S27：继续/附属/暂缓，为什么？比较至少一个不同于Codex的路径。
4. 设计恰好一个下一研究包：研究问题、已有资产、数学对象、最小新观测、
   结果怎样选择分支、何时停止。不要再给八条可以无限跑的队列。
5. 恢复分层指标：数学机制可复用性、原生响应保留、作者/群体结构、外部
   心理链接、推理成本各自是验收什么，而不是用一个r代表全部。
6. 挑战Codex：上次提出的“自然关系编辑 -> 多查询共享 -> 轻量替代”是否
   只是重复VM03/S1/S4？若是，需要哪一个真正新增的条件才能改变结论？

允许援引先前核对的一手文献，链接与命题要对应。不要求新文献大搜集。
你返回后Codex将核对证据与提出针对性反例，再请你第二轮作最终取舍。
