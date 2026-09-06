# VM06-RC2：上下文信息的逐层可用性

2026-09-06。**固定层位实验与独立标量审计完成。**
协议9cd659a、实现0e6b728在本轮真人forward前提交。
这是RC1同一44人材料的功能追问，不是独立样本复现或人格效度实验。

## 核心结果

真实前文下的当前回复状态，在较晚层搬运到错配前文时，可以恢复更多
实际回复的条件似然优势。block10改善小，不是干预器完全无效：block16、
block22改善明显更大，最后block与final norm精确恢复。

但**群体平均随深度上升，不等于每个人都沿同一条单调积累路径**。
本轮说明的是指定输入和干预下的状态可用性，不是人格信息所占比例，
更没有证实“第16层开始理解人格”。

## 1. 做了什么

完整沿用RC1的44条真实用户回复、其紧邻AI前文和长度匹配错排。
每条当前用户token不变；真实前文状态作为donor，错配前文为接收forward。
依次在零起点block4/10/16/22/27或final_norm，替换全部当前正文token状态，
每次只改一个站点，不同时替换多个层，不搬运AI上下文位置。

每人两次原始forward、六次自替换、六次实际状态搬运。正文第一词由正文
之前的位置预测，不在本次搬运的作用范围内，因此主统计仅用第2至最后词。
没有生成解释，没有训练头，没有使用MEPS量表。

## 2. 定量结果

主单位为每人tail-token平均log probability，之后参与者等权平均。
原始真实前文优势为 **+.41975 [.30363,.55200] nat/token**，38/44正方向。

| 搬运站点 | 相对错配改善 | 条件性95%区间 | 正方向人数 | 群体均值之比 |
| --- | ---: | --- | ---: | ---: |
| block4 | +.01162 | [-.00571,.02509] | 28/44 | .0277 |
| block10 | +.02431 | [-.01639,.06162] | 28/44 | .0579 |
| block16 | +.16806 | [.11745,.21964] | 35/44 | .4004 |
| block22 | +.34317 | [.24580,.45771] | 39/44 | .8176 |
| block27（最后block） | +.41975 | [.30363,.55200] | 38/44 | 1.0000 |
| final_norm | +.41975 | [.30363,.55200] | 38/44 | 1.0000 |

均值之比为mean(patch-mismatch)/mean(actual-mismatch)，不是平均个人恢复率，
不是“人格/语义信息百分比”。末两个1.0是数学上应有的正控制，不是独立发现。
未对个人接近零或负的分母做除法，也不把作用强制限制在0至1。

预先固定的相邻层配对差：

| 后一站点 - 前一站点 | 均值 | 条件性95%区间 |
| --- | ---: | --- |
| block10 - block4 | +.01270 | [-.02123,.04372] |
| block16 - block10 | +.14375 | [.09013,.20415] |
| block22 - block16 | +.17510 | [.10357,.26595] |
| block27 - block22 | +.07658 | [.04384,.11741] |
| final_norm - block27 | .00000 | [.00000,.00000] |

全部区间：1000次、seed731、以参与者为单位，条件于这些固定回复及既定
donor分配。六个站点不是六组独立参与者；不是人口p值或多重比较校正的
确认性检验。不根据本轮大小选出一个“最佳人格层”。

## 3. 重要细节：没有一条共同的单调个人路线

以下为完成预设分析后的描述性检查，不作为新增预注册结果：

- 44人中只有13人的block4→10→16→22→27改善序列在1e-6容差内单调不降。
- block22搬运在11人身上比真实前文的原始结果更高，在5人身上比错配更低。
- 本来就有6人的真实前文tail似然低于错配；末层精确恢复后仍是这6人，
  不能把他们删掉以制造全体一致的效果。

因此这个过程不能被简化为每层给每个人增加固定份额的单一信号。
可能有非线性交互、个别token竞争、混合状态的离分布效应；本轮没有
区分这些来源。最后合格回复中位仅24.5 tokens，可能包含结束语或短回应，
不能代表完整复杂自我开示。

## 4. 数学上学到了什么

令C_l为前文所有位置的状态，U_l为当前正文所有位置的状态，M/P为mask及位置。
剩余计算是F_l(C_l,U_l;M,P)，不是只靠正文均值的f_l(mean U_l)。本轮估计：

    delta_l = ell_tail(F_l(C_l^m, U_l^a; M_m, P))
              - ell_tail(F_l(C_l^m, U_l^m; M_m, P))

上标a/m分别为真实/错配条件。C_l与U_l并未一起搬运，因此这是部分状态的
interchange，而不是完整计算状态替换。早层小作用并不能说明早层不编码
真实上下文，较晚层作用也不能证明存在唯一高阶语义或人格变量。

final_norm后，对于当前token t的状态u_t与下一词x_(t+1)：

    log p(x_(t+1) | u_t) = W_x u_t - logsumexp(W u_t)

后续没有跨位置attention。把真实u_t逐个精确替换后，tail logp必然恢复。
block27之后也只有逐位置norm与读出，所以同样恢复。这是本轮正控制为何
成立的原因，不能将架构恒等式冒充新心理学定理。

对GRAIN的实际升级方向是：将**可读信息、上下文利用、读出充分性**分开。
正文均值向量很接近，仍可能在词汇读出重要方向上发生有用改变；一个
中间正文向量包含信息，也未必在保留错误前文时足以复原输出。
应研究条件状态与正文状态的组合算子，再问这些算子是否与作者差异/外部
心理指标有关。Where/What/相邻When在本实验有操作定义，Who仍未识别。

本设计沿用激活替换的标准思路；方法/指标会改变定位结论，故冻结站点、
错排、计分和控制，不作全层最佳值搜索。
[Zhang & Nanda, ICLR 2024](https://arxiv.org/abs/2309.16042)。
若要声称对应某个高层心理变量，还必须明确可检验的抽象因果模型；
状态搬运本身不提供这种命名。
[Geiger et al., NeurIPS 2021](https://arxiv.org/abs/2106.02997)。

## 5. 验收与复现

- 真实缓存Qwen2.5-1.5B-Instruct、FP32/MPS/eager/noKV；无下载或远端MEPS传输。
  44人、264站点行、628forward（含12器械），832.3秒，MPS已分配采样6.175GB。
  独立technical-only另12forward，日英末层tail误差均0；不是跨语言效度检验。
- 264次自替换逐token误差全0；264次正文搬运对第一词预测的改变全0；
  block27与final_norm共88次真实tail最大误差全0。
- 88次原始forward重做RC1实际/错配条件，mean/tail标量差<=4.44e-16。
  block10平均效果也原样重现；与RC1重用同材料，不能声称独立样本复现。
- 独立audit验证输出SHA、冻结材料身份、44人完整六站点配对、全部效果/区间及控制。
  未独立重跑628forward，也未从未持久化的激活重新计算神经状态。
- 152项专项测试通过，包含漏行、错人、首词受影响、末层错误、虚假汇总攻击。
  新故障fixture曾因pandas整数列不能写入小数而先失败，已显式float修复并
  全部复测；不影响模型执行。两项既有SWIG弃用警告保留，非全历史suite。

canonical：`results/grain_vector_mechanisms/vm06_response_depth_r1`，identity
`991a2673c0e65aa8105ed0f5616c86e34c6cf87f9ea6a27879297397122c92b4`。
审计：`vm06_response_depth_audit_r1`；器械：`vm06_response_depth_technical_r1`。
原文、token IDs、激活未保存；输出均在ignored研究目录，只公开聚合报告。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
"$PY" scripts/run_grain_response_depth.py --model-dir "$R/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306" --input-root '/Volumes/mobile3/projects/project persona/data_sets/MEPS+AI_conv_experiment' --preflight-dir "$R/vm06_meps_preflight_r3" --parent-dir "$R/vm06_response_context_r1" --output-dir "$R/new_response_depth"
"$PY" scripts/audit_grain_response_depth.py --run-dir "$R/new_response_depth" --parent-dir "$R/vm06_response_context_r1" --output-dir "$R/new_response_depth_audit"
```

## 下一步决策

本局部问题闭环：弱block10效果不是整个干预器失效，指定较晚层的正文状态
更能恢复真实前文优势。未闭环的问题是**恢复来自怎样的上下文计算**。
下一包应在固定材料上区分词袋/词序贡献与持续前文访问，不继续扩大层位搜索；
同时研究逐位置读出几何，避免再用均值余弦代替输出作用。所有新控制先冻结。
完整PANDORA来源观察器、强稀疏对照、同ready比较和未开bridge仍需完成；
不得用这项无标签对话机制实验代替人格效度或宣布整个研究目标完成。
