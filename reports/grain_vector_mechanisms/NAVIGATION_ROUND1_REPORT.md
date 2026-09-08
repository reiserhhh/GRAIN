# GRAIN NAV第一轮：合成参照、实际路径与独立计算

2026-09-09。Goal V6仍ACTIVE；本报告交付第一轮发现，不表示NAV-01–12全部完成。

## 1. 发现与意义

1. 只用合成研究参照构造的公式可以接回真人：自由相谈低落状态r=.569，旧真人参照r=.564。构造与评分仍借用冻结模型的前缀；合成参照不等于从零知识推导心理。
2. 实际内部替换发现上游作用：候选覆盖注意力头与MLP，而非注意力权重可视化。每目标8个构造集选出的路线，在保留场景上有非零原生作用。
3. 第一种无Transformer运行候选已有T/F外部联系：三个固定输入投影算子AUC=.677–.705；外向性与状态没有同样运输。该差异指导补上下文运算，而非优化T/F排行榜。

## 2. 受控世界与双参照

12场景骨架×2角色×2机会×2表达=96文本，三个查询=288原生读取。6/3/3场景族划分；两个参照都来自构造场景。额外54次参考梯度/前向，共342次、约96秒。所有场景事实与人格归因分列。

| query | reference | n | r | mae | rmse |
|---|---|---|---|---|---|
| TF_cont | R_real | 24 | 0.9450 | 0.0309 | 0.0385 |
| TF_cont | R_synth | 24 | 0.9759 | 0.0179 | 0.0230 |
| adapted_dejected5 | R_real | 24 | 0.9959 | 0.0280 | 0.0317 |
| adapted_dejected5 | R_synth | 24 | 0.9974 | 0.0330 | 0.0372 |
| extraversion | R_real | 24 | 0.9876 | 0.0400 | 0.0485 |
| extraversion | R_synth | 24 | 0.9916 | 0.0201 | 0.0235 |

表中r是与模型评分的相关，24条保留文本来自3场景，不是24名真人。角色、机会、表达变化单独保存：合成参照的机会响应r为E=.974、TF=.878、低落=.790；低落角色响应仅.566。水平一致不代替响应检验。

## 3. 真人参照运输

| dataset | query | reference | n | function_external_r | function_r_lower | function_r_upper | function_auc |
|---|---|---|---|---|---|---|---|
| PANDORA_source | TF_cont | R_real | 64 | 0.2072 | -0.0505 | 0.4226 | 0.6196 |
| PANDORA_source | TF_cont | R_synth | 64 | 0.2112 | -0.0389 | 0.4204 | 0.6116 |
| PANDORA_source | extraversion | R_real | 213 | 0.1767 | 0.0523 | 0.2997 | n/a |
| PANDORA_source | extraversion | R_synth | 213 | 0.1661 | 0.0405 | 0.2909 | n/a |
| direct_answers | adapted_dejected5 | R_real | 46 | 0.1092 | -0.1475 | 0.3491 | n/a |
| direct_answers | adapted_dejected5 | R_synth | 46 | 0.1316 | -0.0970 | 0.3604 | n/a |
| free_chat | adapted_dejected5 | R_real | 46 | 0.5645 | 0.4088 | 0.7053 | n/a |
| free_chat | adapted_dejected5 | R_synth | 46 | 0.5690 | 0.4137 | 0.7118 | n/a |

预测先保存，再连接既有问卷。没有用MEPS文本/标签构造系数；既有46人和source-only队列是已暴露的探索性外部连接，不是新独立确认。PANDORA只有E213人、TF64人，非官方完整CV。

## 4. 上游实际运算

先用冻结读出梯度排序，再只在构造场景选候选；保留场景不参与选头。候选限到第12层，避免直接替换第16层读出当作上游解释。30个角色/机会对，300组实际干预，351次神经调用，约85秒。

| query | cases | function_native_patch_r | mean_abs_native_patch | mean_abs_native_natural | patch_natural_r |
|---|---|---|---|---|---|
| TF_cont | 6 | 0.9639 | 0.0302 | 0.0421 | 0.6417 |
| adapted_dejected5 | 6 | 0.7702 | 0.0148 | 0.0467 | 0.4678 |
| extraversion | 6 | 0.9567 | 0.0062 | 0.0405 | 0.7543 |

层/头编号从0计数。E与低落共享候选L12H0、L11H7、L11H0；TF的首个注意力候选是L12H3，并有较多MLP候选。这里是同名运算位置的交集，后续要做跨目标共同替换，才展开共享机制结论。

E的8路合并只造成较小原生变化；高作用相关不能写成已恢复整个自然变化。全部带符号组合交互已保存，不把非加性删除为噪声。当前还没有完成A的独立P/Q生成器。

## 5. 输入投影候选及公式

```text
z_1 = <G, P-P_R>; z_2 = g' (Q-Q_R); f = b+z_1+z_2
J_ja = d z_a / d E at synthetic reference text j
A_jtk(x) = softmax_k[tau*cos(E_jt,E_xk) - lambda*(pos_jt-pos_xk)^2]
E_tilde_jt(x) = sum_k A_jtk(x) E_xk
z_hat_a(x) = mean_j[z_a(x_j) + <J_ja, E_tilde_j(x)-E_tilde_j(x_j)>]
f_hat(x) = b+z_hat_1(x)+z_hat_2(x)
```

三种预先固定候选为uniform(0,0)、semantic(8,0)、semantic_order(8,4)。没有真人监督拟合，没有依据人类相关调tau/lambda；它们是输入梯度加固定对齐的局部近似，不是已验证的完整前缀等价。

12次构造梯度约6秒；真实输入token的369单元全部运行，无P/Q缓存输入，无Transformer运行，进程没有import Torch。运行保留约0.93GB预训练token embedding表和小型参照梯度，不能称为没有预训练知识。

| dataset | query | method | n | full_function_r | function_external_r | function_auc |
|---|---|---|---|---|---|---|
| PANDORA_source | TF_cont | semantic | 64 | 0.2128 | 0.3425 | 0.6887 |
| PANDORA_source | TF_cont | semantic_order | 64 | 0.1480 | 0.3503 | 0.7047 |
| PANDORA_source | TF_cont | uniform | 64 | 0.3084 | 0.3276 | 0.6767 |
| PANDORA_source | extraversion | semantic | 213 | -0.0301 | -0.0255 | n/a |
| PANDORA_source | extraversion | semantic_order | 213 | -0.0219 | -0.0225 | n/a |
| PANDORA_source | extraversion | uniform | 213 | -0.0049 | -0.0931 | n/a |
| direct_answers | adapted_dejected5 | semantic | 46 | 0.1504 | 0.0099 | n/a |
| direct_answers | adapted_dejected5 | semantic_order | 46 | 0.2954 | -0.0418 | n/a |
| direct_answers | adapted_dejected5 | uniform | 46 | -0.2848 | 0.1621 | n/a |
| free_chat | adapted_dejected5 | semantic | 46 | 0.3635 | 0.1465 | n/a |
| free_chat | adapted_dejected5 | semantic_order | 46 | 0.1026 | -0.0214 | n/a |
| free_chat | adapted_dejected5 | uniform | 46 | 0.2232 | 0.2242 | n/a |

T/F三种AUC的bootstrap区间分别约[.531,.805]、[.544,.818]、[.559,.836]（uniform/semantic/order）；与历史原生AUC的增益区间仍跨0。最有用的新事实是无真人拟合的独立算子保留了关联，而非宣称已胜出官方基准。
T/F的独立分数与完整函数的r仅.148–.308，即人类关联与原生机制复现不是同一结果。低落和E则提示仍缺上下文化、角色或长度/位置相关运算。保留完整负结果，不把T/F结果外推到全目标。

## 6. 下一轮已确定的机制问题

- NAV-02/03：把已定位的QK/OV与MLP上下文化作用接入程序，比较各目标为何需要不同运算；不扩tau/alpha精度网格。
- NAV-04/05/06：补自然文本证据变化、查询措辞/编码和跨目标共同路线交换。
- NAV-07/08/09/12：现有时点表继续分账，做固定参照/缓存/局部组合的有界实验。
- NAV-10/11：新程序每次改动后做同一real-text连接，并交付真实依赖与可重放入口。

## 7. 重放与状态

脚本依次为prepare_grain_controlled_navigation.py、run_grain_navigation_baseline.py、evaluate_grain_navigation_transfer.py、run_grain_upstream_routes.py、run_grain_projection_program.py、evaluate_grain_projection_external.py。已存在冻结输出时不得覆盖；collect按完整单元恢复。
数据位于results/grain_vector_mechanisms/navigation_v1/，作者级文件均在ignored结果目录。公共报告只有聚合指标。既有报告和旧lockbox保持原状。
