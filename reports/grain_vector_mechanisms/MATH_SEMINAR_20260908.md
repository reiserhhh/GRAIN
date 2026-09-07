# 数学研讨结论：从方差方向转向功能路径

2026-09-08。结论：下一步优先识别“证据位移×输出敏感度”，再做同人多查询
的共享/专属响应分解，最后训练保持证据效应的轻量学生。不是继续盲目加维度。

## 1. 本轮实际完成

- Codex独立核对当前D1–D3B结果及主要数学论文，再与Claude进行两轮讨论。
- Claude使用Fable 5.1、effort max，从已有研讨会话开只读分支。
  第一轮6次Read、5次WebSearch；第二轮2次Read。没有执行或写文件工具。
- 第一轮约303秒、12turns；第二轮195秒、3turns，均成功返回正式答复。
  Codex独立立场在读取Claude第一轮正式答复之前落盘。
- 实际编写并运行5个有限数学算例及其测试；前4个送Claude交叉讨论，第5个在
  最终整理时补充。没有新神经forward，没有新真人预测结果。
- 对现有代码的复核确认：位移SVD先保大方差；capture处于inference_mode；
  旧缓存未记录梯度。旧E5总均值不能替代编辑后embedding。

私有对话正式输出、工具元数据和算例JSON保存在
`results/grain_vector_mechanisms/claude_math_seminar_20260908/`，不提交原始stream。
只引用正式答复，不提取或传播模型隐藏推理。没有打开问卷、MEPS原始文本或env。

## 2. 最有价值的三条数学升级

### A. 保留会改变判断的方向，不只保留变化大的方向

控制与模型降阶领域的CoBRAS同时考虑状态分布与输出梯度；2026年进一步用于
湍流爆发、同步与海浪极端事件的机制发现。它直接回应本项目的问题：低方差
方向可能高敏感，单独PCA/SVD可以丢掉关键响应。[1][2]

我们的最小对象是同case的d与g：M≈g^T d。候选投影以
`mean[(g^T(I-P)d)^2]`比较；再用真实全剂量patch测非线性影响。
边际二阶矩C_d、C_g的平衡值得尝试，但必须保留联合配对。
这是从已有方法中借用目标与算法思路，不直接套用其所有系统假设与误差界。

### B. “共用了一部分路径”应在同人同编辑的作用上定义

AJIVE提供同一受试者多数据块的联合/个别分解；2025年的理论研究进一步分析
信噪和子空间错位何时限制共享估计。[3][4] 我们把每个特质查询作为一个块，
同一作者和相同文本编辑作配对。输出三张不同的图：评论证据共享、带符号的
功能响应相关、联合/专属作者配置。它们不再被一个几何cosine混为一谈。

这能接上用户的新假说：大特质与小特质可能共用某类证据，但对它的读出方向、
强度或独有证据不同。路径共享与问卷相关之后再作关联，不先假设二者相等。

### C. 学生学习“文本改变时判断如何改变”，而不只是拟合一个数字

Sobolev训练和Jacobian匹配把函数值与局部响应一起作为学习目标。[5][6]
我们的可计算版本用同一输入编辑E_j连接两种编码器：

\[
L=E(f_q(x)-s_q(x))^2
+\eta E\{f_q(E_jx)-f_q(x)-[s_q(E_jx)-s_q(x)]\}^2.
\]

这直接服务“少用LLM推导，保留向量到特质的计算关系”。固定编码器仍可来自
预训练模型；所谓轻量替代是省略推理生成/下游大模型计算，不是禁止神经编码器。
DeepONet的输入分支×查询分支可作后续结构参考，小样本先做线性/核学生。[7]

## 3. 最新证明给我们的思考方法

Wang–Zahl的三维Kakeya结果及Guth 2026年的讲解，重要之处不仅是结论，而是
从极端/最难配置中追踪粗细尺度近等式、进而逼出结构的思路。[8][9]
转译到GRAIN：先找全局公式解释最差、但不同视图仍重复的局部模式，检查哪一
种条件化解释了它。不是试图把1536维文本空间当成三维Kakeya线段集合。

Marton猜想特征2情形的证明用熵距离和结构化下降推进论证。[10]
我们借“有明确改进量的结构搜索”，不把有限群定理套到实值embedding。
本项目可直接使用的数学骨架其实更朴素而精确：嵌套条件期望的勾股分解。

令F_l包含目前已观察的信息，m_l=E[Y|F_l]。增加新信息后：

\[
E[(Y-m_l)^2]-E[(Y-m_{l+1})^2]=E[(m_{l+1}-m_l)^2]\ge0.
\]

证明只需m_{l+1}-m_l与Y-m_{l+1}正交，再展开平方。
意义：残差里找到结构，就是找到了新增信息对应的函数增量；不必删除topic、
situation或旧信号，也不必假定所有误差最终为零。有限样本估计仍要报告实际
改进量。新增信息应累积嵌套，不能用LLM层号自动代替这个条件。

另读Mori-Zwanzig深度学习表示与路径signature研究。[11][12] 前者提示投影
遗漏可能变成记忆项，后者刻画同端点不同顺序；这两条只有在S1暴露相应剩余
模式后才进入下一包，不让文献清单变成另一轮无边界实验。

## 4. 已计算的五个数学见证

这些是可复现构造，不是PANDORA新结论。

| 算例 | 实际结果 | 对实验的直接影响 |
|---|---|---|
| 方差与功能分离 | PCA保留99.9900%方差，响应MSE仍1；功能方向MSE 0 | 按响应而非方差选方向 |
| 联合配对 | d和g的边际二阶矩完全相同；配对响应可全1或全0 | 作者/case配对不能打散 |
| 坐标变换 | 响应、二阶矩乘积谱误差<4e-15；欧氏cos从0变为-0.9998 | 声明度量，优先功能配对量 |
| 路径顺序 | 相同端点两路径有向面积+0.5/-0.5；固定梯度闭路积分0 | 用顺序特征，不错称非零保守环流 |
| 新增信息 | 精确总体MSE 5.25→4.25→0.25→0；增量能量1/4/0.25 | 交互结构可藏在上一层残差中 |

最后一个构造为三个独立均匀符号A/B/C、Y=A+2AB+0.5C；终点误差0是因为
玩具世界所有生成变量都已观测，不是实测人格误差最终为0的证据。

脚本：`scripts/check_grain_math_seminar_20260908.py`。
测试：`tests/test_grain_math_seminar_20260908.py`。
本轮验证：5项测试通过；两个新增Python文件通过Ruff格式与检查。

## 5. 真正的分歧与修正

Claude和Codex共同接受：标量线性读出不意味着只有一个潜在心理维度；普通
作者响应矩阵秩不等于Ho–Kalman最小状态维数；control r低不能直接推出核空间；
边际梯度/位移谱不能替代配对作用；固定标量梯度积分不能产生非零闭路效应。

保留Claude的有效反驳：公共编辑应有内容依据，不必固定文本位置；先用三个
查询选中评论的并集，再跨查询重用同一编辑。输入梯度按评论聚合能给“显影剂”
提供低成本入口；学生若配对缓存齐全可以立即零forward先导，不必等待S1。

最终整理时进一步校正：

- 2个编辑不在数学上自动使CP退化；例如秩2时k-rank之和仍可能满足唯一性
  充分条件。暂缓CP是样本/解释/优先级决定，不是一个“2编辑不可能”的定理。
- 24作者×2视图×2编辑构成每查询24×4块，不是24×8。重复行不当独立作者。
- S0没有梯度缓存，g^T d比较必须等待S1；S1中的输入梯度也不是免费。
- 梯度乘输入需要声明参照；用参照差与实际编辑校准，不把离散删除效应等同
  连续一阶项，也不将全部非线性精确度归给一次反传。
- 全体梯度相似可以意味着成功发现共同算子，不因为同作者余弦不高于异作者
  就关闭整条线。作者特异性和可复用性分别回答。
- 条件期望恒等式作用于同一个Y，教师响应与真实心理标签不能偷换。

## 6. 决议与下一步

执行规范见`docs/GRAIN_FUNCTIONAL_PATH_NEXT_PLAN_V1.md`。

1. **S0零forward**：既有96作者按作者CV比较位移SVD、full-Ridge、PLS。
2. **S1小面板**：24作者×2视图×3查询，共同编辑和局部梯度；先8作者测资源，
   上限60分钟。计划726F+144B，具体GPU时间以新可微capture实测为准。
3. **S2轻量学生**：先学响应，再比较共享项与特质专属项；缺失编辑embedding
   就利用S1补齐。不重开长队列，不为增加特质数量而复制庞大训练任务。

最值得追的新对象是“同一证据在不同查询下的敏感度分布”，而不是一个无条件
通用的人格向量。共同定式可以成立，具体特质的函数与证据覆盖仍允许不同。

## 一次文献与使用位置

1. Otto, Padovan, Rowley. *Model Reduction for Nonlinear Systems by Balanced
   Truncation of State and Gradient Covariance*. CoBRAS，原始算法与局限：
   https://arxiv.org/html/2207.14387v4
2. Zolman, Mokbel, Otto, Brunton (2026-06). *Uncovering Extreme Event Mechanisms
   for Prediction and Control with Sensitivity-Balanced Projections*：
   https://arxiv.org/html/2606.05618v1
3. Feng, Jiang, Hannig, Marron. *Angle-Based Joint and Individual Variation
   Explained*：https://arxiv.org/abs/1704.02060
4. Yang, Ma (2025). 共享子空间估计及AJIVE理论：
   https://arxiv.org/abs/2501.09336
5. Czarnecki et al. (2017). *Sobolev Training for Neural Networks*：
   https://arxiv.org/abs/1706.04859
6. Srinivas, Fleuret (2018). *Knowledge Transfer with Jacobian Matching*：
   https://arxiv.org/abs/1803.00443
7. Lu et al. *DeepONet: Learning nonlinear operators for identifying differential
   equations based on the universal approximation theorem of operators*：
   https://arxiv.org/abs/1910.03193
8. Wang, Zahl (2025). *Volume estimates for unions of convex sets, and the Kakeya
   set conjecture in three dimensions*：https://arxiv.org/abs/2502.17655
9. Guth (2026-04). *The Kakeya conjecture, after Wang and Zahl*，尤其粗细尺度
   近等式和结构部分：https://arxiv.org/html/2604.03416v1
10. Gowers, Green, Manners, Tao (2023). *On a conjecture of Marton*，熵距离归纳：
    https://arxiv.org/html/2311.05762v2
11. Venturi, Li. *The Mori-Zwanzig formulation of deep learning*：
    https://arxiv.org/abs/2209.05544
12. Ceylan, Proemel. *Global universal approximation with Brownian signatures*，
    2025首版、2026-07修订：https://arxiv.org/abs/2512.16396

Active subspaces梯度协方差来源另见Constantine, Dow, Wang：
https://arxiv.org/abs/1304.2070 。以上按研究适用性选用，不按数学名词复杂度排序。
