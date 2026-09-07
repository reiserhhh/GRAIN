# GRAIN 高等数学机制研讨：第一轮

2026-09-08。Claude Fable5.1/max 只读；Codex独立查文献并主持第二轮。

## 用户目标，优先于旧会话计划

找到作者文本到人格倾向判断的可复用数学程序，不是追求所有Big5/MBTI因子同样
高精度。拟人化、依恋等窄倾向也可有路径。不同特质可能部分共享、部分专属；
量表相关可能映射到共享路径。长期希望固定文本向量加数学模型替代LLM的推导。
先发现、再发展、后集中补强。局部有效机制就是成果，不做无边界防御程序。

用户希望跨领域高等数学启发，包括近期猜想证明的思考方法，不仅套现成ML。
请提出有实验落点的大胆想法，明确标准定理、借来的策略和我们新假说的区别。

## 已完成结果（以这些为准，旧会话状态已过时）

- D1：真实source作者native Big5 mean r=.1993(n213)，TF AUC=.6600(n2328)。
- D2：40TF+24N作者，两视图。query-selected4评论/full12评论的作者排序r=.6104/.8193。
- 真实证据移除的block16末查询向量位移d，在原输入上patch会引起部分自然判断变化。
- D4：训练作者SVD rank8+ridge近似M(alpha)，原留作者r=.5314/.5584。
- D3B：64开发，32新作者。shared每查询跨作者公式r=.3917/.3694，MSE/zero=.8699/.4987。
  GRAIN48->PCA4的双线性gain，r=.3646/.3999但MSE=.8938/.5735，未优于简单模型。
- N的selected donor响应r约.375，control donor约0，条件化信号不均匀。
- 不同任务rank8位移子空间重叠=.2724，但功能向量cos=.0895。
  来自不同作者群；这是几何比较，不是量表相关或27%人格信息。
- 公式精确部分：Delta s=M(1)+R；M(alpha)=integral grad g(h+lambda*d)^T d dlambda。
  近似：Mhat=a_q^Td。R包含其余路径与交互，不自动是噪声。
- 现有位移SVD优化的是向量方差，不直接优化功能；同人同材料多查询尚未采集。
- 旧P1U/完整128片队列暂停；新1736次forward已全部完成。不要重启任何实验。

## 只读范围

根目录：/Volumes/mobile3/projects/Sliced Utterance In-Context Assessment。
只读这5份汇总，不全仓库扫描，不读取原始文本、任何results逐人数组、量表、env、凭证：
1. docs/GRAIN_MAINLINE_MECHANISM_THEORY_V1.md
2. docs/GRAIN_SHARED_AND_TRAIT_SPECIFIC_PATHS_V1.md
3. reports/grain_vector_mechanisms/D3B_MAINLINE_CLOSURE_REPORT.md
4. docs/GRAIN_QUERY_PATH_MATHEMATICS_V1.md
5. docs/GRAIN_DISCOVERY_FIRST_RESEARCH_POLICY.md

允许公开论文WebSearch/WebFetch。禁止Bash、代码执行、写文件、MCP、其他代理。
不输出隐藏思考，只返回正式分析与可复核推导。不要修改当前终端或另开实验。

## 第一轮问题

独立提名至少5种数学思路，其中至少2种不只是常见线性回归/因子分析换名字。
可从微分/信息几何、算子理论、控制/可观测性、调和分析、多尺度证明、核方法、
统计学习、张量分解等方向出发。我们不预设必须使用这些。

每种必须说明：
1. 它改变了哪个核心数学对象，具体公式是什么？
2. 标准理论的假设、近期原始论文/定理来源（链接；没核实明确写待核实）。
3. 如何解释目前空间重叠但功能不同、gain不改善、selected/control差异？
4. 最小实验与观察量，用多少作者/查询/forward，能改变哪项下一步决定？
5. 哪一部分只是借用证明思路，不能直接把定理搬到LLM？

特别评估：位移PCA是否选错优化对象？共享路径是否应在响应函数而非欧氏坐标中
定义？如何让不依赖LLM的向量学生学到功能而非复制无用的hidden variance？
如果用交换子/闭环/holonomy，要指出固定标量函数精确梯度闭路积分为0的情形，
避免把不同操作造成的新状态混成同一保守场的非零闭路。

给出明确前三优先级，以及一个零神经forward可算的机制见证。中文约2000字。
Codex会将自己的文献核对和反例交给你第二轮修订，而不以意见一致代替实验。
