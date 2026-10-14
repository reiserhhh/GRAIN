# GRAIN 查询条件作用律：固定向量作用与参考敏感度

2026-10-08。HM22 已执行、冻结、全表异公式核验。承接 HM21 的已知
输出计分微积分，研究对象是受控读者的干预作用，不是新的人格维度。

## 1. 本轮得到的机制区分

固定来源计分敏感度后，部分跨查询共同结构仍存在；其他关系会因为
参考状态的敏感度变化而反号。共同结构不能统一归给“上游向量路径”，
也不能全部归给输出非线性。下一步需要保留完整输出向量及参考观察律。

完整属性应用 NA/低落：固定敏感度作用协方差为原生协方差的94.1031%。
但参考敏感度增量为-27.8081%，计分曲率增量为+33.7050%，两者抵消。
固定作用接近最终作用，不能推出这两个增量可以忽略。

## 2. 定义、参考与精确分账

沿用原十分类数值 alphabet，

\[
f(z)=a^T\operatorname{softmax}(z),\quad
g(z)=\nabla f(z),\quad z\sim z+b\mathbf1.
\]

对查询 \(q\) 的实际参考/目标候选 logits \(r_q,t_q\)，令
\(d_q=t_q-r_q\)，使用其零和公共平移商表示。来源参照为

\[
\bar g_q=\sum_{i\in S_{\mathrm{full}}}w_i g(r_{q,i}),
\qquad \sum_i w_i=1.
\]

\(S_{\mathrm{full}}\) 固定为原 source full_property 的64状态/8词汇
family；权重使各family等重。三个查询各64个不同参考记录，没有重复
地址。192是查询参考记录数，不是192个新独立作者。

来源参照在读取held结果前冻结，不由目标标签、最佳anchor或rank决定。
它固定的是**计分函数的参考敏感度**，没有删除或中心化文本情境信息。

\[
\begin{aligned}
B_q&=\bar g_q^Td_q,\\
G_q&=(g(r_q)-\bar g_q)^Td_q,\\
C_q&=f(t_q)-f(r_q)-g(r_q)^Td_q,\\
y_q&=B_q+G_q+C_q.
\end{aligned}
\]

\(B\) 表示在固定来源敏感度下看到的查询条件位移；\(G\) 表示参考
计分敏感度的变化；\(C\) 是 HM21 已验算的非线性计分余项。三者都保留。
这里的 \(G\) 不是作者动态，\(C\) 不是心理噪声；字母 \(B_q\) 也不是
历史待识别的个人响应算子 \(B_u\)。

## 3. 跨查询结构与完整向量观察

在每个预定角色/设计的family测度下，记
\(\Sigma_{qs}=\operatorname{Cov}_w(d_q,d_s)\)，则

\[
\operatorname{Cov}_w(B_q,B_s)=\bar g_q^T\Sigma_{qs}\bar g_s.
\]

本包保留72个完整10×10跨查询位移协方差块，公共logit平移不改变
这个收缩。十分类答案坐标不是十个人格因子；旧九维head商空间结论
继续保留，未重新发现或重新选择坐标秩。

令 \(K_{XY}=\operatorname{Cov}_w(X_q,Y_s)\)，\(X,Y\in\{B,G,C\}\)：

\[
\begin{aligned}
K_{\mathrm{fixed}}&=K_{BB},\\
\Delta_{\mathrm{sensitivity}}&=K_{BG}+K_{GB}+K_{GG},\\
\Delta_{\mathrm{link}}&=K_{BC}+K_{GC}+K_{CB}+K_{CG}+K_{CC},\\
K_{\mathrm{native}}&=K_{\mathrm{fixed}}+
\Delta_{\mathrm{sensitivity}}+\Delta_{\mathrm{link}}.
\end{aligned}
\]

这些是有符号的协方差分账，不是正交方差分解，比例可超过100%或为负。
若旧编译作用为 \(P_q\)，还有

\[
K_{\mathrm{native}}-\operatorname{Cov}_w(P_q,P_s)
=\underbrace{K_{BB}-\operatorname{Cov}_w(P_q,P_s)}_{\text{固定作用差距}}
+\Delta_{\mathrm{sensitivity}}+\Delta_{\mathrm{link}}.
\]

## 4. 数据中的异质性

完整属性应用32状态/4family：

| 查询对 | Native协方差 | 固定作用BB | 敏感度增量 | 曲率增量 |
|---|---:|---:|---:|---:|
| NA/低落 | +8.24505e-7 | +7.75885e-7 | -2.29279e-7 | +2.77900e-7 |
| NA/信任 | +3.20744e-6 | +4.55543e-6 | -1.14788e-6 | -2.00110e-7 |
| 低落/信任 | -7.10021e-6 | -6.28631e-6 | -4.81959e-7 | -3.31944e-7 |

NA/低落 receiver forward 总差距1.78610e-6中，固定作用差距占97.2778%；
敏感度与曲率分别占-12.8369%/+15.5590%，净增量仅2.7222%。
因此该组主要需要改善向量作用运输，但不能用净抵消抹去参考观察律。

完整属性验证 NA/信任：BB=-8.44497e-7，敏感度增量=+1.33448e-6，
曲率增量=+6.17548e-8，最终native=+5.51734e-7。关系从负转正。
应用 mean_only NA/信任还出现BB=+1.53897e-6到native=-1.18152e-6
的反向例子，不能把反号全部解释为近零数值抖动。

全部72项重叠设计协方差中61项BB/native同号，11项不同；不是72次
独立复制或心理显著性检验。全部7臂、所有角色及失败保留，没有挑赢家。

## 5. 对理论与实现的升级

应分开两个运算：

\[
\text{文本状态/干预}
\xrightarrow{\Phi_q}\text{查询条件化的完整输出向量}
\xrightarrow{f,\,r_q}\text{标量观察作用}.
\]

\(\Phi_q\) 是否能由来源向量独立运输，是下一机制问题；\(f\) 已知。
把这两层直接揉成一个标量回归，可能掩盖参考依赖和共享作用的来源。
这是本轮结果引出的待检验推断，不是已证明所有编译误差都由标量压缩造成。

完整概率向量和logit商坐标可互换，标量 \(a^Tp\) 则会丢失方向。
同时 \(f\) 在概率坐标中恰好线性，在logit坐标中一般非线性：曲率取决
于所用坐标表示，不应被命名为一种固有的“人格曲率”。这是标准分类
观察律，不是本项目首创。

## 6. 验算、交付与下一包

全部2172地址/724状态/7臂/312 no-op核验；0新拟合或前向。
三个source anchor、72矩阵块、2172动作、72统计、72协方差、432差距
逐项用不同算式复核。动作差5.25e-16、协方差差4.34e-19；小分母比例
另列最大差4.09e-10，未修改主结果或容差。
268项关联测试、28文件通过。同agent异公式，不是第三方盲复建。

- 实现：`grain_research/construction/math_query_action_transport.py`。
- 主结果：`results/grain_math_response_v1/query_action_transport_r1`。
- 复核：`results/grain_math_response_v1/query_action_transport_verification_r1`。
- 下一计划：`docs/GRAIN_OUTPUT_VECTOR_TRANSPORT_PLAN_V1.md`，HM23尚未执行。

下一包保持旧来源输入核与alpha，预测完整零和候选logit位移，再经已知
readout计分；只改变被学习的对象，不搜索精度。held原生参考logits仍是
观察器缓存输入，必须明确：这是机制探针，不是已经脱离LLM的部署计分器。
原18/100/44、自然作者、个人响应与心理意义继续保留，不从本包自动验收。

## 理论出处与借鉴边界

[Wainwright与Jordan2008，§3.5.2–3.6](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf)
给出指数族自然参数/完整均值参数的关系，启发保留向量观察而非只保留一分。
本轮阅读到相关定理和例子，不声称305页全读或所有附录证明完成。

OpenAI/math固定commit `adc7f1241b42e322a6451854ab7e4b4c146bf78a`，
metric/connection稿07-global.tex190–222再次核读。借鉴观察、参考和单位
共同运输；没有将PDE唯一延拓或光滑流形假设套用到文本人格。
