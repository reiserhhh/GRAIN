# 相同材料上的内容、选择与证据质量

2026-09-11，NEXT70 / 第59轮。对应[报告](../reports/grain_vector_mechanisms/NORMALIZED_EVIDENCE_REPORT_V1.md)。

## 1. 两系统的数学接口仍是同一段材料

GRAIN提供作者相对参照位置及同材料上的方向D_f；查询程序F_q回答沿该方向
改变向量时，特定推论怎样改变。统一接口是

    tau_qf(X) = <gradient_H F_q(H(X)), D_f(X)>.

这不是将两个分数相加，也不是凭token权重给作者贴标签。方向作用按token
地址相加可归回slice；有限改变仍须积分或另算交互。同一个词在别的上下文
下可能有不同作用，正负项与抵消都需保留。

## 2. 本轮可复现的标量与导数

z_t为源冻结的8维BODY坐标，p_t是原三维位置描述；16个Gaussian中心与
带宽sigma不变。每个核k定义

    log w_kt = -||(z_t,p_t)-c_k||^2/(2 sigma^2)
    Z_k = (1/T) sum_t w_kt
    ell_k = log Z_k
    a_kt = w_kt / sum_j w_kj
    m_k = (1/T) sum_t w_kt z_t
    r_k = sum_t a_kt z_t = m_k/Z_k
    s_kt = gradient_z log w_kt = -(z_t-c_k,z)/sigma^2.

两个读法的完整scalar分别为

    F_raw = F_local + sum_k [v_k^T(m_k-mbar_k) + b_k(ell_k-ellbar_k)]
    F_rel = F_local + sum_k [v_k^T(r_k-rbar_k) + b_k(ell_k-ellbar_k)].

每组系数分别拟合，但数量/源目标/alpha完全匹配。导数从同一标量得到：

    grad_z_t F_rel = grad F_local
      + sum_k a_kt [v_k + s_kt {v_k^T(z_t-r_k)+b_k}]

    grad_z_t F_raw = grad F_local
      + sum_k (w_kt/T)[v_k+s_kt(v_k^T z_t)]
      + sum_k a_kt b_k s_kt.

三个加项可分别记为直接内容、选择变化、log质量变化；它们是公式内的计算
分解，不是相互独立的心理构念。尤其**raw也有跨token作用**：logZ的导数
已经取决于全部token；不能称它为完整的零交互对照。原始加权矩本身是加性。

此处质量Z是**平均核响应**，不是文本长度、事件出现次数或完整choice分布。
它只保证在这组有限摘要内，归一化没有不可逆删去m的幅度；不能声称保留了
所有作者选择信息。数值实现使用logsumexp，不可表示质量直接报错，不裁剪。

## 3. 本轮补齐的逻辑

当Z>0时，(m,ell)与(r,ell)可逆，所以归一化不创造新信息。但线性读头与
各向同性Ridge不在这个非线性坐标变换下保持不变；不同结果可能来自函数
表达与正则几何共同变化。本次不能唯一归因于softmax的心理合理性。

共同log权重偏移c使r不变、ell增加c、m乘exp(c)。保留ell的完整预测器
通常仍会变化。对特定摘要的比例不变，不等于整体测量不受机会或分布影响。

实测给出具体用途区别：源N的相对选择改善跨段GRAIN响应(.969572->.948326)，
确认寻求没有同方向改善；源外完整方向作用则是raw更好，较local误差低
3.60%/5.41%。完整评分依旧没有同步进步。因此下面三件事要分开发展：

1. 在一组材料里建立相对选择机制。
2. 选出的变化投影到GRAIN方向后是否有用。
3. 方向作用能否组成对新作者完整评分的正确函数。

这保留局部机制的成果，不让整函数失败抹去它，也不把前两步自动当第三步。
下一步先对冻结程序拆出content/selection/mass的带符号作用；不扩中心数。

## 4. 与前人方法及解释层的关系

[Ilse、Tomczak与Welling (2018), §§2.3-2.4](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf)
用可学习加权汇总把多个实例连接到一个集合标签，区分实例级和嵌入级路径。
这支持同一文本集合可以有内部证据选择与整体读出两个层次。其权重由神经
网络学习；本轮核固定、拟合连续原生评分/导数，另保留log质量，并非原样
复现他们的分类系统，更没有从高注意力自动得到临床解释。

精神分析/交流分析辅助解释可使用同一材料地址、邻接对话角色与固定数值作为
输入。关系事件的编码单位可能跨多个slice，不能只选一个最高权重token。
参考既有[同材料解释理论](GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)
中的防御机制、CCRT与TA工作；本轮不新运行临床提示、不改变评分、不将
`confirmation_seeking`等模型查询当已验证的人格量表或诊断。

当前真实作者文本已经接入，但H16仍由固定神经前缀产生。这是数值读出
机制的有限发展，不是全部编码器已被移除。
