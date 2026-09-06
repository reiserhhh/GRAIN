# RC3: 同词袋、不同顺序的上下文作用

2026-09-06。状态：模型运行及独立标量/分配审计完成。
协议9372ff1先冻结，049233f实现，随后才运行私人文本。

## 结论

**上下文作用不能仅由原始 token 计数解释；仍不能把顺序效应直接叫作人格推理。**
同样44份完整用户回复、相同实际/长度匹配错配前文，原顺序的平均
logp优势为.4364 nat/token，八token块打乱后为.3025，逐token打乱后为.0671。
两个固定seed方向一致。原文字不变的token词袋还保留部分兼容信息，
但native优势大幅依赖组织方式。这是冻结小型Qwen的上下文利用实验，
不是作者人格分数、心理量表效度或跨人泛化提升。

## 方法与主结果

只在已分词的前一条AI回复内部交换完整token ID，不decode/re-encode。
当前人的完整回复、所有role边界、位置、长度和mask不变。八token块保留
块内顺序；逐token打乱破坏局部子词组织。每一前文在其actual和donor用途
使用相同排列，seed固定731/1731，不挑seed、不删除难例。

令每人实际/错配前文的回复平均对数概率差为D，原顺序与打乱顺序的
差中之差为I，两个前文臂共同的平均降幅为Q：

```text
D_s = ell(actual_s) - ell(matched_s)
I_s = D_native - D_s
Q_s = [(ell(actual_native)-ell(actual_s))
       +(ell(matched_native)-ell(matched_s))] / 2
```

下表主指标为全部回复token的平均logp；括号为固定材料/配对下1000次
participant bootstrap描述区间。两个seed先在人内平均，N仍44。

| 前文顺序 | D:实际-错配 | I:相对native优势下降 | Q:共同退化 |
|---|---:|---:|---:|
| native | .4364 [.3107,.5749] | 0 | 0 |
| block8, seed731 | .3139 [.2165,.4153] | .1226 [.0606,.1862] | .0539 [.0140,.0988] |
| block8, seed1731 | .2911 [.1932,.3951] | .1453 [.0842,.2157] | .0714 [.0235,.1259] |
| block8, 两seed均值 | .3025 [.2079,.4007] | .1340 [.0806,.1958] | .0626 [.0208,.1099] |
| token, seed731 | .0540 [.0062,.0978] | .3824 [.2693,.5160] | .2864 [.2026,.3726] |
| token, seed1731 | .0801 [.0291,.1327] | .3563 [.2532,.4731] | .2789 [.2032,.3619] |
| token, 两seed均值 | .0671 [.0245,.1136] | .3694 [.2642,.4944] | .2827 [.2041,.3651] |

block8均值I为34/44人正，token均值I为41/44人正。token打乱后D为28/44人正。
去掉第一预测token的tail敏感性结果同方向：native D=.4197，block8
D=.2914、I=.1283 [.0728,.1963]；token D=.0697、I=.3501 [.2412,.4794]。
所有条件都报告，不把两个seed当88位独立被试，也不把这些条件区间当
新增独立确认集的显著性。

## 器械与审计

- 446 forward（6非私人检查+44x10），591.14秒；FP32/MPS/eager，无KV。
  MPS采样分配峰值6,174,857,728bytes；模型/模板同RC1，没有训练。
- 440模型行、352排列记录覆盖完整；原始actual/matched重新运行与RC1
  all/tail标量最大误差0。相同正文输入embedding均值精确不变。
- 所有前文精确保留token多重集、长度、非前文位置；反排列恢复通过。
  没有边界交叉token。44个前文的两个seed均非相同排列且没有不变条件。
  block8/token平均改变位置比例分别.9675/.9870。
- 独立审计重算分配、派生seed、排列hash、来源identity、标量、全部效果与
  区间。它没有独立重跑神经网络，也没有重新读raw token核算词袋；后者由
  runner精确断言与合成攻击测试保护。两种证据不能混写。
- 170项相关单元测试通过；不是整个历史仓库全套测试。私人文本/向量/ID
  不进入git；MEPS只在Mac解析许可文本，未访问量表字段。mixed回答容器
  仍由原loader扫描/哈希后提取许可文本，不声称完全未读容器字节。

## 数学解释与边界

置换token词袋不变，不代表语法、子词有效性、位置匹配或输入分布不变。
I消掉了两臂退化的共同项，但不能消掉与原回复兼容度有关的差异退化。
因此不能说84.6%的效应就是推理或人格；这只是两个均值之比，非信息分解。
残留D也不能纯称topic：模型可从剩余统计关系重建一些次序。

这与已有研究的争论一致，但不是复现同一任务：Sinha等研究的是打乱顺序
的masked-LM预训练及下游任务；Abdou等指出子词切分前后打乱并不等价，
且长度与unigram分布仍可能保留可恢复顺序。我们是冻结自回归Qwen在固定
真实回复上的条件likelihood扰动，不可把两个设置的结论直接互换。
[Sinha et al., 2021](https://aclanthology.org/2021.emnlp-main.230/)，
[Abdou et al., 2022](https://aclanthology.org/2022.acl-long.476/)。
上述文献用于运行后的方法定位，不追写成事前冻结的理论预测。

当前只测前一AI回复的作用，不是完整人的记忆。44份回复中位24.5tokens，
存在简短结束语；材料重复使用，区间不涵盖新材料/新donor的不确定性。
RC1证明前文兼容，RC2证明正文状态的层位充分性，RC3证明对词序扰动敏感。
三者均不能独立证明心理构念；GRAIN的源集合头与未开bridge才负责后续
外部连接，不能用本实验替代。

## 交付与下一步

canonical：`results/grain_vector_mechanisms/vm06_response_order_r1`，identity
`e4f465784dea72cab55c8572050b4c17ee5b4cfd483274a34dbfd2a943864709`。
audit：`vm06_response_order_audit_r1`；technical：`vm06_response_order_technical_r1`。
代码：`response_order.py`、`run_grain_response_order.py`、`audit_grain_response_order.py`。
在新输出目录复现，勿覆写完成包：

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
"$PY" scripts/run_grain_response_order.py --model-dir "$R/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306" --input-root '/Volumes/mobile3/projects/project persona/data_sets/MEPS+AI_conv_experiment' --preflight-dir "$R/vm06_meps_preflight_r3" --parent-dir "$R/vm06_response_context_r1" --output-dir "$R/new_response_order"
"$PY" scripts/audit_grain_response_order.py --run-dir "$R/new_response_order" --parent-dir "$R/vm06_response_context_r1" --output-dir "$R/new_response_order_audit"
```

下一项是固定层后的正文对前文attention边访问阻断，不新增token打乱搜索。
要区分已存入正文状态的信息与后续持续访问；先验证layer0阻断后的上下文
独立性、末层无操作与self/sham，再接私人材料。不是移除topic/personality，
也不改变正式GRAIN计分或开放新心理标签。完整源集合/强稀疏对照仍在进行。
