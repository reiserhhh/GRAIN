# HM22：参考敏感度与查询条件向量位移

2026-10-08，HM21 后的下一有限包，尚未执行。
只用冻结 native endpoint logits/gradient 和旧 compiled 作用；无新神经调用、
API、训练、真人数据或参数网格。新命名空间
`results/grain_math_response_v1/query_action_transport_r1`，另有verification。

## 需要改变的科学决策

HM21 的 nonlinear readout 解释完整属性应用 NA/低落共同作用差距约15.6%，
多数差距仍在 \(L=g_q(r)^Td_q\)。但这一项仍有参考计分敏感度，不可笼统
命名为“上游共享路径”。拆开敏感度与 logit 位移，决定下一步追踪哪一层。

## 冻结定义与步骤

1. 核验 HM21/HM20 原始hash/代码/2172地址/724状态/7arms，保留
   8/4/4 family、ownership、顺序、机会结构、no-op与失败。先冻结本包代码。
2. 只使用原 **source full_property** 的64状态参考 gradient，按原等family
   测度计算三查询 \(\bar g_q\)。这是先验指定设计的参考矩，不是拟合人格
   标签或挑选最佳anchor。报告每查询不同 reference record 的实际数、重复
   地址及权重；重复记录不是独立样本。参数在读held摘要前冻结。
3. 在全部2172地址计算
   \(B_q=\bar g_q^Td_q\)、\(G_q=(g_q(r)-\bar g_q)^Td_q\)，
   验证 \(y_q=B_q+G_q+C_q\)。不修改旧 compiled coefficient 或移除余项。
4. 对全角色/全arms计算三部分 RMS、内积及9项有符号协方差。
   对比 HM21 的L/C分账与旧 forward/odd compiled 差距。
   将 \(\operatorname{Cov}(B_q,B_s)=\bar g_q^T\Sigma_{qs}\bar g_s\)，
   \(\Sigma_{qs}=\operatorname{Cov}(d_q,d_s)\)，直接矩阵复算。
5. 保留十分类公共平移 gauge，使用已知head商空间，但不重做
   HM10B1 输出商空间发现。十分类答案轴不是十个人格因子；跨查询数值
   单位相同也不保证语言或心理意义相同。
6. 对 source 定义、全部primitive/table/跨查询矩阵使用另公式复核。
   synthetic tests覆盖固定/变化敏感度、协方差反号、gauge、alphabet单位、
   no-op和非零相互抵消。失败保留，不调rank/anchor/alpha救援。

## 判定和终止

若固定 \(\bar g\) 下的B仍携带同方向主要共同结构，下一步追踪共享的
query-conditioned logit 位移支持与 context-binding 运输。
若 G 贡献主要符号/幅度，下一步保留参考状态的观察律，避免将其当作者信号。
混合结果分查询/设计保留，不做总体赢家排名。

本包以改变上述决定为交付，不把9项恒等分账本身当新发现，不继续细分
只为填表。将一个明确下一机制写回主线，而不是另起精度队列。
若本包仍不能区分，需提出最小的新干预信息要求，不能无界复拆同一缓存。

原18/100/44/natural authors/个人响应与心理效度保持全范围。
不调用Claude/自动化、不commit/push/tag/lockbox，不访问MEPS私文或恢复旧队列。
