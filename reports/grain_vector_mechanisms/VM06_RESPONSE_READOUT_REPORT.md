# RC5: 从向量几何到输出功能的精确读出分解

2026-09-06。44人、98完整forward、92次额外LM-head计算完成，独立标量/
材料审计通过。没有训练、心理标签或新层位搜索。

## 结论

**均值向量相似，不等于模型的输出功能相同。**
真实与错配前文下，平均预测位置向量的余弦仍达.9918；逐位置余弦平均.9745。
但真实前文的tail平均logp优势为.41975 nat/token。它可以精确写成
`.75668-.33693`，或`1.07246-.65271`。前者分开目标词分数和全词表归一化，
后者分开matched端线性敏感性和有限位移的KL余项。

数学恒等式本身不是原创发现。实证价值在于：它在本项目相同真人回复上
定位了“余弦很高但输出改变”的计算层次，说明单独看池化距离或局部线性
敏感性都会漏掉部分功能变化。它仍未定位人格特异方向。

## 输入、位置与标准数学

复用RC1的44份完整回复与实际/长度匹配前文。使用最后归一化之后、
直接进入LM head的逐位置状态h；不把中层向量硬套最终读出，也不先平均
后预测。正文状态0..T-2预测观测词2..T；排除由prefix预测的第一词，以及
没有对应观测下一词的末正文状态。共1808个tail token，独立统计单位仍44人。

令a/m为actual/matched，z=W h+b，A(z)=logsumexp(z)，p=softmax(z)：

```text
Delta ell_y = delta z_y - [A(z_a)-A(z_m)]
G_m = delta z_y - E_{p_m}(delta z)
G_a = delta z_y - E_{p_a}(delta z)
Delta ell_y = G_m - KL(p_m || p_a) = G_a + KL(p_a || p_m)
G_a <= Delta ell_y <= G_m
```

对应的局部隐藏空间曲率是`W.T(diag(p)-p p.T)W`，方向曲率为
`Var_p(delta z)`。因此功能几何取决于读出W和当前位置的分布p，不只是
隐藏空间的原始长度/角度。有限位移的KL并不自动等于二阶局部近似。
这属于指数族/Bregman的标准关系，不是新造心理公理。
[Banerjee et al., JMLR2005](https://www.jmlr.org/papers/v6/banerjee05b.html)。

## 实测结果

先对每个人的token求均值，再对44人等权平均；1000次人级bootstrap，
seed731，条件于这批回复与donor。均值统计的精确等式成立，各项区间不应
相减来伪造差值区间。

| 量 | 人级均值 | 描述95%区间 |
|---|---:|---:|
| 原回复logp优势 | .419749 | [.303632,.552001] |
| 目标词logit变化 | .756678 | [.536695,1.002170] |
| log-partition变化 | .336929 | [.209890,.473843] |
| matched端梯度项G_m | 1.072463 | [.827183,1.338494] |
| KL(matched || actual) | .652713 | [.515609,.799317] |
| actual端梯度项G_a | -.198688 | [-.273916,-.124163] |
| KL(actual || matched) | .618437 | [.503325,.738753] |
| matched方向曲率 | 1.437508 | [1.078343,1.854657] |
| 逐token状态余弦均值 | .974518 | [.970693,.978289] |
| 两个平均预测状态的余弦 | .991758 | [.990211,.993280] |
| 逐token位移范数 | 34.777952 | [31.809588,37.525778] |

38/44人的净logp优势为正；matched端G_m则44/44正。actual端G_a为37/44人负，
但这不表示原上下文有害：净位移效果是沿路径的有限差，不是终点局部导数。
matched端线性外推明显高估净作用，actual端线性值甚至平均负，KL恰好给出
两者与真实有限差之间的差距。不能据此宣布“65%是噪声”或“某百分比是人格”。

范数34.78属于这个模型的任意坐标尺度，不与其他模型范数直接比较。
池化余弦.991758与旧RC1的.992005略不同：本实验只纳入预测已观测tail的
T-1个位置，旧值使用整个正文T个位置。这不是新模型或材料漂移。

## 数值问题和验收

首次非私人英文检查失败：把LM head拆成8行小矩阵后，与原评分的误差
4.254924e-5超过1e-5。没有读私人文本，没有放宽容差。诊断改回原始T行
矩阵，并保持每个tail位置的原始行号，误差降至约4.4e-7。
原始第0行的prefix状态不在body缓存中，故置一个不用于评分的0行；线性
读出逐行独立，后续所有行保持相同。随后才分8行进行CPUfloat64词表归约。

这是事前仪器修正，不是结果筛选。fdfb342冻结数学问题，167cbae实现；
e450c27在私人forward前记录并修正形状问题。旧technical_r1保留FAILED_STOP；
technical_r2的EN/JA检查通过后才开始正式包。

- 全包151.88秒，MPS采样6,174,857,728bytes，低于不变9GiB。
  10技术+88私人完整forward；4技术+88私人head-only调用。
  私人CPU词表归约242批，技术5批；不是242次完整模型运行。
- 全部1808位置精确softmax/KL恒等式、非负性和端点凹性通过，最大标量
  恒等式误差1.92624e-14；独立head逐token重播误差<=1.21336e-6。
- 原RC1原始forward重播误差<=4.44e-16；float64重新归约后的人均tail
  与旧float32均值差<=6.96688e-7，低于事前1e-5，不把舍入差解释成效应。
- 独立audit重算44人/1808位置覆盖、标量等式、层/位置计数、来源identity、
  人级均值与全部区间。它未独立重跑神经网络/整个词表概率，后者只在
  runner内瞬时验证；保存的是标量，不是原文、token ID或向量。
- 209相关测试通过，包含零变换、共同logit偏移、旋转后同步搬运读出器、
  相等范数但相反/零效果、位置错位及篡改审计；不是全历史测试。

## 理论意义和剩余范围

RC1证明前文兼容效应，RC2测部分状态搬运，RC3测同词袋次序扰动，RC4
分开状态与持续前文访问，RC5解释最终读出。它们共同反对把“平均文本
embedding的距离”直接当成“模型对个人的理解程度”。但这不是证明LLM
已有隐含Big5，而是为这种假说建立可重播的计算层次。

现有观测是短回复对前一AI发言的兼容性，不是全人生记忆、作者特质或
模型自述的真实思维。KL在这里是两个下一词分布的差，不是人的不确定性
或心理距离。还需源集合人格标签的外部读出、同ready增量和独立材料检验。
也不能因为末层解码恒等式成立就把中层套用同一W；中层解码偏差是已有
方法研究的问题。[Belrose et al., 2023](https://arxiv.org/abs/2303.08112)。

这组固定回复RC机制包现在关闭，不再进行开放式层位/方向/阈值搜索。
下一阶段按GRAIN_VECTOR_REMAINING_REQUIREMENTS.md完成源集合及固定/自由、
Essays/X的机制复测，最后形成VM07，不以本报告替代心理测量主目标。

## 交付

canonical：`results/grain_vector_mechanisms/vm06_response_readout_r1`，identity
`15ed75e0bbe420a9a8a71c61665692ca784a11cf6c2b8b379bf03727c371d16e`。
audit：`vm06_response_readout_audit_r1`；有效器械：`vm06_response_readout_technical_r2`。
使用新目录复现，不覆盖完成包：

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
"$PY" scripts/run_grain_response_readout.py --model-dir "$R/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306" --input-root '/Volumes/mobile3/projects/project persona/data_sets/MEPS+AI_conv_experiment' --preflight-dir "$R/vm06_meps_preflight_r3" --parent-dir "$R/vm06_response_context_r1" --output-dir "$R/new_response_readout"
"$PY" scripts/audit_grain_response_readout.py --run-dir "$R/new_response_readout" --parent-dir "$R/vm06_response_context_r1" --output-dir "$R/new_response_readout_audit"
```
