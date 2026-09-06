# VM06-RC1：相同用户回复，真实与错配上下文

2026-09-06。**本地上下文利用实验完成；人格推论机制的整体验证未完成。**
设计先于真人forward提交3b4b463，器械与FP32修正dfb835b。

后续更新：本文末层正控制与固定层位追问已由
[RC2](VM06_RESPONSE_DEPTH_REPORT.md)完成。本文保留RC1当时的结果与待办边界，
不把后续发现回写成原先已经验证的结论。

## 结果回答了什么

真实上一轮AI回应，使相同人类回复比长度匹配的错配回应更容易被模型预测。
但把真实上下文下的block10用户token状态搬到错配上下文，只带来很小且
不确定的平均改善。因此不能把上下文作用归结为一个独立可搬运的中层
“用户向量”。这提供了上下文利用的局部证据，不是新增人格效度。

## 材料与执行

- 只读MEPS自由相谈文本白名单。46人中44人符合最后完整合格回复规则；
  检查111个候选对，67个因长度规则不符跳过，未截断/改写、未使用问卷。
  每人一条，44个不同回复；16--243 tokens，中位24.5，共1852 tokens。
  这不是每人完整会话，也不代表所有复杂自我开示。
- 真实AI前文、全局长度匹配错排、固定随机错排、无AI前文四臂。
  三种替代臂保持当前用户token逐个相同，输入embedding差异为0。
  匹配/随机错排的平均绝对log长度差=.03438/.58686；无人作自己的donor，
  无重复使用donor，也无真实与donor完全相同的上下文。
- 原生Qwen模板，当前正文起点固定768、执行长度1024，prefix右对齐并
  attention-mask左pad，显式position_ids；无KV cache，不生成回答。
- Qwen2.5-1.5B-Instruct固定revision，七个文件逐SHA与Windows缓存一致后
  本地复用；没有把MEPS传出Mac。FP32/MPS/eager，270个forward，346.8秒，
  已分配显存采样最大6.175GB。向量只留内存，输出为编码标量与哈希。

## 1. 上下文确实影响回复预测

每人先取所有用户token平均log probability，再等权平均参与者；单位nat/token。

| 上下文 | 平均logp |
| --- | ---: |
| 真实AI回应 | -3.78991 |
| 长度匹配错排 | -4.22634 |
| 随机错排 | -4.20966 |
| 无AI前文 | -4.61183 |

| 配对差 | 均值 | 描述性95%区间 | 正方向人数 |
| --- | ---: | --- | ---: |
| 真实 - 长度匹配 | +.43643 | [.31067,.57494] | 39/44 |
| 真实 - 随机 | +.41975 | [.30498,.54154] | 42/44 |
| 真实 - 无前文 | +.82192 | [.66090,.99555] | 44/44 |

区间为1000次固定seed731配对bootstrap，条件于这些回复与既定错排。
不包含重新抽取参与者/新对话/重配donor的不确定性，不提供人口或人格效度p值。
无前文还改变了对话轮结构，不能单独隔离AI语义；主对比采用相同轮结构的
长度匹配错排。词汇重合、问答兼容、指代解析和顺应都仍可解释该效果。

## 2. 向量改变不等于单点机制恢复

真实相对长度匹配前文，正文token均值向量的平均余弦/相对L2变化：

| 站点 | 余弦 | 相对L2 |
| --- | ---: | ---: |
| 输入embedding | 1.00000 | .00000 |
| block10 | .98536 | .16118 |
| block16 | .97743 | .20169 |
| final_norm | .99201 | .12127 |

不能因final_norm余弦很高就判断上下文无用。余弦是正文**均值**几何，
语言模型输出来自逐位置读出，两者不是同一统计量。

功能诊断在错配forward中替换block10全部当前用户token状态为真实上下文
donor状态。只评分第2至最后用户token，因为第1token由正文之前的位置预测：

- 真实相对错配的完整tail差：+.41975 [.30363,.55200]，38/44正向。
- block10搬运相对错配的tail差：+.02431 [-.01639,.06162]，28/44正向。
- 44人的错配自替换逐token logp误差均0。

本轮没有充分证据认定block10正文状态是独立因果瓶颈。也不能由此认定
正文状态完全没用，或作用只在某个未测层。后续层仍能读取错配AI位置的
状态，当前操作没有搬运整段上下文或完整计算状态。尚欠末层读出正控制和
更精确的层/角色路径定位，不能把这项局部弱恢复写成唯一通路证明。

## 3. 对GRAIN数学路线的含义

单位置或均值状态并不是Transformer计算的完整状态。可写为：

    H_X^(l+1) = F_l(H_C^l, H_X^l; mask, positions)

C为上下文位置，X为当前文本位置。当前输入相同，但H_C变化能够持续
改变H_X；只替H_X、不替H_C，不能保证重建原有输出。
这是一项结构性解释候选，下一实验要直接区分持续上下文读取和末层读出，
而非把本轮数据当作其唯一证明。

若h是某一预测位置的final_norm状态、W为输出矩阵，则精确有：

    log p(x | h) = W_x h - log sum_v exp(W_v h)
    gradient_h log p(x | h) = W_x - sum_v p(v | h) W_v

因此有用的变化由读出方向加权，不由向量距离大小单独决定；聚合均值还会
混合不同位置的梯度方向。应分别测量“能从向量读出什么”“什么输入改变
向量”“什么状态干预改变输出”。不能把可逆白化、余弦靠近或某个平均向量
变化直接当信息增减。本轮与VM04的局部弱搬运相呼应，但不是跨任务效应等价。

对应4W：Where是受控会话前文；What是固定用户回复；When是相邻轮次顺序；
Who尚未由此识别。本研究仍服务人格解释器的方法基础，但此处不作人格命名。

## 4. 器械与证据边界

最初tokenizer接口返回BatchEncoding而不是list，r1在任何forward/真人读取
之前拒绝，修复兼容后保留失败记录。BF16器械r2的日语位置平移测试未过：
logp差.02657、最小层余弦.99830。未放宽门槛，FP32 r3把日/英差降到
1.19e-6/3.58e-6、余弦>=.999999999996，自替换0。两例不能判断语言能力。
正式运行重做这六个器械forward后才读真人文本。

`vm06_response_context_audit_r1`核对全部44成员、重算两种错排和五个配对差/
区间，并验证输出SHA及528几何行的覆盖、范围和聚合。未从原始激活重算
余弦；这是标量/分配独立审计，**不是**
独立重跑270个模型forward。完整原始激活不持久化，复测需要在受限本地
重新加载允许文本。bridge、MEPS量表、训练标签均未用于本实验。

## 复现与下一步

canonical：`results/grain_vector_mechanisms/vm06_response_context_r1`，identity
`6e21037163c52808fc6429601d17aae1668c2dc2fce451021586df68780a7b58`。
器械`vm06_response_context_technical_r1/r2/r3`保留，r3通过。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
"$PY" scripts/run_grain_vector_mechanism_response_context.py --model-dir "$R/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306" --input-root '/Volumes/mobile3/projects/project persona/data_sets/MEPS+AI_conv_experiment' --preflight-dir "$R/vm06_meps_preflight_r3" --output-dir "$R/new_response_context"
"$PY" scripts/audit_grain_response_context.py --run-dir "$R/new_response_context" --output-dir "$R/new_response_context_audit"
```

下一优先补final_norm正文状态搬运的精确tail读出正控制，排除局部弱恢复
只是干预实现问题；再按固定少数层比较状态利用深度。之后检验词汇重合/
词序与持续上下文读取，不搜索最能“显著”的人格方向。完整来源头、bridge
及冻结跨材料外部机制路线仍须按主计划完成，不能被本次局部实验替代。
