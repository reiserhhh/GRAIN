# GRAIN逻辑研究：真实评论的排列、输入可区分性与目标选择

2026-09-10。NEXT08b有限实验完成；持续goal ACTIVE。

## 1. 新发现

**当前E5评论均值确实把某种LLM读取差异压成了同一个输入。**
12作者相同12条评论原文，仅反转列表：T/F原生判断变化RMS=.1048，N=.0341；
两查询均12/12作者变化超过2e-6。原始端点和同次重复误差都是0。
因而这些变化不是靠给同一均值换一个更强的回归器就能重现的。

但“把顺序加回来”也不是完整解决方案：源拟合位置加权项对新反转的配置
变化r=-.236(T/F)/.050(N)，MSE/零变化约1.22/1.23，没有改善。
它在源作者的T/F目标拟合r从.656升至.685，却把自评关联r从.249降至.203。
**输入区分能力、正确条件运算、作者效标保留，是三个连续但不同的机制。**

## 2. 数据与实际操作

同一旧12位源不重叠研究作者，T/F和N各自original/reverse24对。
评论原文、评论数、同一E5向量多重集保持；显示顺序和编号归属变化。
总提示token长度不变；自然渲染后的边界token有0-4个计数L1差异，均值3。
这是完整评论的呈现顺序干预，不是纯位置编码干预或人的性格变化。

第一次prepare在token多重集检查处停止，未执行任何新神经计算或源拟合。
随后在神经结果出现前记录tokenizer-only修订，保留全部作者和这些差异。
现行包为`logic_input_order_v1/paired_r2/`。旧12作者面板已使用过，不称新盲评。

固定模型Qwen2.5-1.5B，MPS FP32/eager，不生成文本；52次完整前向
（48端点+4重复），耗时82.03秒。无API、无新E5调用、无LLM训练。
保存了P/Q与native读数并复算固定f的所有部分；全部重放误差为0。

## 3. 新的数学连接：输入等价类而不只是线性拟合

若phi(T)=phi(JT)，任何只看phi的函数g均满足g(phi(T))=g(phi(JT))。
这也适用于非线性g。当前独立E5均值、query加权池、无序分布池都具有这一性质；
不是说E5看不见单条评论内部的词序，而是聚合不保留评论之间的排列。

令t+=(t(T)+t(JT))/2，t-=(t(T)-t(JT))/2，则任意不变预测p满足：

    mean_pair (t-p)^2 = (t+-p)^2 + (t-)^2.

这里的群只有两元素C2。下面是实际配置c与已冻结E5均值程序的分账：

| query       |   pair_mse |   even_error |   odd_floor |   odd_floor_fraction_of_pair_mse |
|:------------|-----------:|-------------:|------------:|---------------------------------:|
| TF_cont     |   0.004518 |     0.003675 |    0.000843 |                         0.186603 |
| neuroticism |   0.000435 |     0.000391 |    0.000044 |                         0.100190 |

奇项下界分别占该反转配对MSE约18.7%/10.0%。这是该轨道实验的误差下界，
不是原始PANDORA预测误差或+.141未提取效标贡献的解释比例。
其他组件表内为计算下界使用了oracle配对均值，不是另一个已训练预测器。

这一构造结合[Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf)的集合不变性与
[Janossy pooling](https://arxiv.org/abs/1811.01900)的排列平均思想；并非原创通用
数学定理。新的经验连接在于它被应用到当前真实作者材料、同一E5输入和同一
查询函数，补上了旧S6内部block16配置实验不能直接替代的输入接口。

## 4. 哪些计算组件在运动

| query       | component     |   mean_absolute_change |   rms_change |   positive |   native_r |
|:------------|:--------------|-----------------------:|-------------:|-----------:|-----------:|
| TF_cont     | configuration |                 0.0438 |       0.0581 |          8 |     0.3836 |
| TF_cont     | score         |                 0.0382 |       0.0457 |         11 |     0.5285 |
| TF_cont     | native        |                 0.0781 |       0.1048 |         10 |     1.0000 |
| neuroticism | configuration |                 0.0085 |       0.0132 |          7 |     0.6430 |
| neuroticism | score         |                 0.0142 |       0.0189 |          5 |     0.2759 |
| neuroticism | native        |                 0.0206 |       0.0341 |          7 |     1.0000 |

T/F反转使10/12原生读数升高，是同一作者在另一种呈现下被模型读成更偏F，
不是作者变成另一种人格。N配置变化与native变化r=.643，但完整f变化仅.276。
完整f是固定参考下的近似函数，不是原生F，所以其组件组合还可能包含近似误差。

按共同sd(Delta f)sd(Delta native)归一化，可加的相关贡献为：

| query       | component     |   contribution_to_full_native_r |
|:------------|:--------------|--------------------------------:|
| TF_cont     | mean          |                         -0.2431 |
| TF_cont     | configuration |                          0.5674 |
| TF_cont     | final         |                          0.2041 |
| neuroticism | mean          |                          0.1620 |
| neuroticism | configuration |                          0.4307 |
| neuroticism | final         |                         -0.3168 |

N的最终读取部分抵消了另外两项的部分关联；T/F的组合则有另一种作用。
这为NEXT10提供了同一材料、同一干预下的组合记账，不以某一组件的r最大
直接改写完整判断。配对区间和全部组件结果保存在CSV。

## 5. 一个有顺序的输入，为什么仍未闭合读取

增加v(T)=mean_j s_j h_j，s_j从-1到1；v(JT)=-v(T)。源训练只学习原pooled
程序误差的无截距Ridge项，不改base，固定alpha=.01*n_train_views。
五折按作者，63 T/F、213 N作者，三个抽样视图；没有用Y学习或挑选参数。
同一旧828条pooled目标/预测重放误差<1e-16。新模型及held预测先冻结再看端点。

配置反转响应：T/F r=-.236，N=.050；MSE/零变化=1.222/1.228。
当前新增项明确能区分顺序，却未学到该顺序的正确条件作用，故不晋升。
源作者均值的真实效标关联为：

| query       | target   |   mean |   ordinal |
|:------------|:---------|-------:|----------:|
| TF_cont     | c        | 0.1907 |    0.1563 |
| TF_cont     | r        | 0.2489 |    0.2029 |
| neuroticism | c        | 0.0415 |    0.0417 |
| neuroticism | r        | 0.0611 |    0.0600 |

这些是已使用源样本的探索性连接，不是新人格性能比较。完整teacher指标与
配对差值另存，不能只展示T/F拟合r增加而略去自评关联下降。

## 6. 最重要的逻辑修正与下一步

复制LLM的所有顺序敏感性，未必符合稳定作者测量的目标。如果目标是对一袋
记录给稳定人格位置，任意展示顺序可以作为显式呈现条件，研究不变或平均目标；
如果目标是作者的真实表达过程，时间和互动次序则应该保留。两种用途都需要，
不是删除动态或把每个模型变化都认作人格。

[输入等价类与目标理论](../../docs/GRAIN_INPUT_QUOTIENT_AND_TARGET_THEORY_V1.md)
已将上述三类用途与4W/GRAIN接合。有限反转结果留下两个可操作方向：

1. 条件运算：用独立源作者的真实original/reverse成对端点，比较同一ordinal输入
   和已有早层P/Q输入，以区分缺少配对监督与缺少上下文化表示。固定小型因子设计，
   不在这12人上训练或对Y做精度网格。
2. 目标选择：对同一材料保留呈现不变部分和顺序响应部分。先作为两个测量对象，
   不把两者合成一个“更像人格”的新总分。T/F原源+.141效标剩余仍需独立定位。

## 7. 交付与重放

方案、核心、runner、六个新单元测试已实现；49个相关测试通过。
公开汇总：`input_order_tables_v1/`；私人数值/PQ/模型仅存gitignored结果目录。
没有改发布评分器、lockbox或MEPS用途；没有push。

```bash
"../project persona/.venv/bin/python" scripts/run_grain_input_order.py prepare --output-dir results/grain_vector_mechanisms/logic_input_order_v1/paired_r2
"../project persona/.venv/bin/python" scripts/run_grain_input_order.py run --output-dir results/grain_vector_mechanisms/logic_input_order_v1/paired_r2
"../project persona/.venv/bin/python" scripts/run_grain_input_order.py analyze --output-dir results/grain_vector_mechanisms/logic_input_order_v1/paired_r2
"../project persona/.venv/bin/python" scripts/report_grain_input_order.py
```

已有目录不能重新prepare；完整acquisition再次run会直接返回，不增加调用。
没有公开的原始材料不能从公共仓库独立生成本面板，需要授权本地缓存。
