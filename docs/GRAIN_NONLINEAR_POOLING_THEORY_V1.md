# 内容响应、证据选择与总量：从二次池化回到查询读取

2026-09-11，第58轮NEXT69。实测见
[报告](../reports/grain_vector_mechanisms/NONLINEAR_POOLING_REPORT_V1.md)。

## 1. 已解决的数学缺项

NEXT68的v=mean(z,p*z)使未改token的响应只能为a+p*b。NEXT69保留同一
局部scalar与二次读头，却先映射每个token：

\[
\phi_k(z,p)=\exp[-\|[z,p]-c_k\|^2/(2\sigma^2)],\qquad
v=T^{-1}\sum_t\phi(z_t,p_t)-c_\mathrm{source},\qquad
F=F_\mathrm{local}+\tfrac12v^TM_qv.
\]

16个中心取原64个冻结中心中的0,4,...,60，未根据目标挑选或重标尺度。
当前作用从同一scalar微分得到。记J_j=D_z phi(z_j,p_j)，则

\[
J_{j,k}=-\phi_k(z_j,p_j)(z_j-c_{k,z})^T/\sigma^2,
\qquad \nabla^2_{z_j,z_i}F=T^{-2}J_j^TM_qJ_i\quad(i\ne j).
\]

对未改B、固定last/布局，有限半差密度精确为
\(\delta k_j=J_j^TM_q\delta v\)。这次J_j随B的具体内容变化，旧位置限制已移除。
实测候选响应能量中57.81%/53.26%不在旧位置子空间内；这是**候选输出形状**，
不是真实响应解释率。已验证分数与导数同源、混合导数对称与原位置零空间突破。

## 2. 从“能够变化”到“变对方向”仍有一段

源坐标响应误差降幅只有.657%/.437%，小于线性二次的2.668%/2.568%。
N的GRAIN作用误差较零低3.794%，确认寻求反而高5.821%。新8完整分数近似
匹配local，没有完整晋级。保留全表，不以变化形状增多代替目标作用学得更好。

这不推翻非线性前置的函数性质；它结束的是**当前16核特征、固定正则与源支持
下的池化二次候选族**。特征变化也会改变Ridge所偏好的函数，不把这轮解释成
与惩罚完全无关的纯表达能力定理。不追加参数扫描或新的泛化保证。

## 3. 论文定位：分布表征与特定任务的读取不同

[Muandet et al., Kernel Mean Embedding of Distributions, 2017](https://arxiv.org/abs/1605.09522)
将概率分布映射到RKHS，并讨论核选择与后续统计推断。原文§3.3.1区分具有
可识别性的完整特征核嵌入与具体预测用途。本实验只是16个Gaussian核评估均值，
不是完整特征核的无限维嵌入；不能借用后者的单射性保证宣称信息已经齐全。
即使表征包含差异，仍需找到与指定查询相应的读取函数。

先前原生证据提供了更直接的下一步：
[NEXT50](GRAIN_SOURCE_QUERY_SELECTION_THEORY_V1.md)已在真实模型内发现可复用
查询引子和key选择作用。标准归一化注意力见
[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)。下一候选借鉴它的
**相对竞争与被读内容分开**，不是继续扩展任意二次多项式。

## 4. 保留总量，不把选择信息归一化掉

对固定核中心k，设\(w_{kt}>0\)，定义

\[
Z_k=T^{-1}\sum_t w_{kt},\quad m_k=T^{-1}\sum_t w_{kt}z_t,
\quad r_k=m_k/Z_k,\quad\ell_k=\log Z_k.
\]

Z描述这类证据的平均核质量，不是未经验证的心理重要性；r描述在该核选择下
的向量位置。保留\((r,\ell)\)时，\(m=e^\ell r\)，所以与\((m,\ell)\)之间
有可逆的数值重表达。单独保留r才会丢掉共同幅度。这个区分与GRAIN一直保留
choice/机会条件一致，而不是为了去噪删去选择。

若用r读取局部值\(v_t=b^Tz_t\)，记\(a_t=w_t/\sum_iw_i\)及
\(s_t=\nabla\log w_t\)，链式法则给出

\[
\nabla_{z_j}(b^Tr)=a_j\{b+(v_j-b^Tr)s_j\},\qquad
\nabla_{z_j}\ell=a_js_j.
\]

第一项为内容读取，第二项为证据相对当前汇总的选择作用。前段变化可以通过
分母与r改变后段读法，且允许增强/抵消。所有项仍来自同一个scalar。公式是
标准商法则/softmax微分；我们要研究的是它在同一作者GRAIN方向上的数值作用。

这不是RKHS条件均值算子估计，也不等于已经恢复LLM注意力头或心理诊断。
当前只完成推导，实际同材料比较按[NEXT70计划](GRAIN_NORMALIZED_EVIDENCE_READER_PLAN_V1.md)
执行。精神分析/交流分析候选解释继续走已有原文地址与证据绑定接口。
