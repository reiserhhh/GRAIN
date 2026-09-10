# GRAIN方向的前向传输与源固定读出

2026-09-10，NEXT40，第29轮。接续[实际注意力分解](GRAIN_FIXED_SEED_ATTENTION_THEORY_V1.md)。
本轮发现：在同一作者材料上，GRAIN方向可以先转成attention输出变化，再与
其他作者提供的固定方向相乘，复现一部分指定查询的局部作用。无需读取当前
作者的输出梯度。这里解释的是模型如何读材料，不把局部导数直接命名为人格。

## 1. 两种计算方向给出同一个作用

令H是最后attention的归一化输入，C(H)是最终位置各head输出的拼接，
A=D_H C为当前上下文处的导数，g为该输出的下游余切向量。固定logit对比
目标L与NEXT39相同。D_f是共同正文token上已有GRAIN坐标定义的方向，
在求导时固定，而不是同时重新估计因子。则：

\[
J=A^Tg,\qquad \lambda_f=A D_f,\qquad
\tau_f=\langle D_f,J\rangle=\langle\lambda_f,g\rangle.
\]

原来的读法是从输出向正文反传J，再与GRAIN相乘；现在可以先把GRAIN
方向向输出传输，得到lambda。两者是标准JVP/VJP对偶，不是新定理。
[JAX官方说明](https://docs.jax.dev/en/latest/jacobian-vector-products.html)
给出这两类线性映射；[You Only Linearize Once](https://arxiv.org/abs/2204.10923)
讨论了线性化与转置在自动微分中的关系。本实现是固定attention的NumPy
解析式，没有引入JAX，也没有依赖论文中的软件实现。

关键开发选择是保留当前上下文A，只将g换成源作者平均gbar：

\[
\widehat\tau_f=\lambda_f^T\bar g.
\]

gbar是冻结构造资产，不是当前人的分数或标签。它最初仍来自原模型源作者
反向计算；当前q/k/v也仍是原模型前向表征。消除的是当前目标反向依赖，
不是已经消除完整编码器或推论出一个人的问卷数值。

## 2. lambda可以直接从数值特征算出来

沿用NEXT39的head m、KV组b(m)、旋转R、完整上下文权重a和输出o。
对选中的非最终query正文位置集合B，令：

\[
\begin{aligned}
\dot v_{bj}&=W_{V,b}D_f(j),\\
\dot k_{bj}&=R_j W_{K,b}D_f(j),\\
\dot s_{mj}&=(R_lq_m)^T\dot k_{b(m)j}/\sqrt d,\\
\lambda_{f,m}^{value}&=\sum_{j\in B}a_{mj}\dot v_{b(m)j},\\
\lambda_{f,m}^{key}&=\sum_{j\in B}a_{mj}(v_{b(m)j}-o_m)\dot s_{mj},\\
\lambda_f&=\operatorname{concat}_m(\lambda_{f,m}^{value}+\lambda_{f,m}^{key}).
\end{aligned}
\]

所有a和o仍由完整可见上下文计算，不对B重新归一化。此处dq=0仅因为
方向在这个输入平面不扰动最终query；更早层的扰动会改变最终query，不能
照抄这个局部假设。实际12query heads、2KV组、d=128和RoPE都保留。

给定四个GRAIN方向，拼接的lambda仅有4×12×128=6144个数。若只需四个
作用，数学上无需保存整个token梯度场。本轮程序仍同时计算完整场进行
对偶核验，因此没有声称测得了取消场计算后的加速倍数。

## 3. 本轮识别出的可共享部分

只用旧4名作者构造源g；旧4的评价用留一作者、其余3人均值，新8用固定
4人均值。另一个事先声明的程序跨两个查询平均源g。没有选择每query
较好者组成一个事后混合模型。两种程序全部保留。

新8作者的GRAIN作用与原模型作用相关，同query源g为N=.809、确认=.955，
跨query源g为.919/.869。作用平方误差相对零预测为.383/.098与.157/.244。
各因子先减作者均值后相关仍为.798/.956与.915/.866，不只是四因子固定
均值差造成的相关。32个数来自8作者×4方向，不是32独立被试。

这说明：当前局部计算可以由共享的下游读出与材料特异的attention传输
共同形成。两查询的共享读出也与固定相同数字输出方向有关；不将这种
数值共享直接翻译成两个人格构念共享了多少心理机制。

## 4. 读出方向的欧氏误差不是作用误差

记delta_g=gbar-g，lambda按因子作为矩阵Lambda的行：

\[
\widehat\tau-\tau=\Lambda\delta_g,\qquad
\|\widehat\tau-\tau\|^2=\delta_g^T\Lambda^T\Lambda\delta_g.
\]

这是当前四方向诱导的半正定几何，不是一个完整心理空间。g的欧氏误差
变小，不保证GRAIN方向上的误差也变小。新8的N确有这一实例：pool后
g相对平方误差均值由.0198增至.0357，但作用误差/零预测由.383降至.157。
因此不能继续以“哪个g最接近”代替“哪些响应被正确读取”。现有数值的
作用误差恒等式最大差6.11e-16，原生float32梯度与前向式的作用差3.35e-7。

这提供下一步定位残差的具体工具：先投影到实际GRAIN响应，再判断误差
来自什么方向；不是把所有未解释场能量一概当人格或噪声。也不是据这一
个观察立即拟合新的加权g。

## 5. 如何从局部接口接回完整路径

当前D_f定义在最后层归一化BODY输入；它不同于早先块16方向。下一步让
块16方向经过真正的后续层传播，并保留最终query与残差的变化。记X为
最后block的输入，r为最后位置attention之后、MLP之前的残差状态：

\[
r(H_{16})=X_l(H_{16})+W_O C(N_1(X(H_{16})))_l,
\quad t=\nabla_rL,
\quad \tau_f=t^T D_{H_{16}}r[D_f].
\]

则g=W_O^Tt，完整响应为dot_r=dot_X_l+W_O dot_o_l。直接从源作者
提取t，再用当前材料的前向JVP与源平均t相乘，才能检验是否同时保留
更早层上下文、query变化与残差路径；不要通过反演W_O猜t。

见[NEXT41完整传输计划](GRAIN_COMPLETE_TRANSPORT_JVP_PLAN_V1.md)。其目标
是块16局部作用，不是完整有限分数差。已完成的同材料接口和
[有证据地址的辅助解释](GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md)
分别保留；模型中的“确认寻求”query不是临床诊断或已经建立的量表。
