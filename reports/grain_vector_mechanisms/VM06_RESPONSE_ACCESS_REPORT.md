# RC4: 已保留正文状态与持续前文访问

2026-09-06。44人、834forward、1113.02秒全部完成；独立标量/材料审计通过。
协议659d638、实现72dbc3b均先于私人forward。无心理标签、无训练。

## 主要发现

**中层正文向量不能默认被当作已经独立完成的上下文摘要。**
在block10之后让正文再也读不到前文，真实前文的相对优势变为-.0537，
描述区间包括0；到block16/22之后才切断，则分别保留.0839/.3155。
原始完整模型的优势为.4197 nat/token。更晚的正文状态在这个切断条件下
能保留更多与回复兼容有关的作用，但不是“人格信息百分比”。

尤其值得保留的反例：中途切断造成的整体损伤大于从第一层就不让正文
读前文。这说明改变计算路径会产生状态与后续计算不匹配，不能把切断
损失当作一项干净、可加的人格成分或纯粹的信息量。

## 方法和完整结果

原始token、role、位置、长度保持不变。只在指定block之后，把所有正文
query对所有prefix key的attention边阻断；prefix query和body-body边不变。
包括role-prefix key，因为这些位置也可能带前文信息。层序号从0开始；
-1=第一层前就切断，27=没有改变。只分析回复第2至末token的tail平均logp：
第一词由prefix位置预测，理论上不会被body-query的改动改变。

```text
D_k = ell(actual,cut_k) - ell(matched,cut_k)
R_k = ell(actual,full) - ell(actual,cut_k)
Q_k = ell(matched,full) - ell(matched,cut_k)
I_k = D_full - D_k = R_k - Q_k
```

均值单位nat/token；95%区间是固定44回复/固定donor下1000次人级bootstrap，
非独立新样本确认、非多重比较后的心理显著性。

| 切断位置 | D:保留的实际-错配差 | I:剩余访问的差异作用 | R:实际臂损伤 | Q:错配臂损伤 |
|---|---:|---:|---:|---:|
| 第一层之前 | .0000 [0,0] | .4197 [.3036,.5520] | .6422 | .2225 |
| block10之后 | -.0537 [-.1351,.0109] | .4735 [.3182,.6557] | 1.9800 | 1.5065 |
| block16之后 | .0839 [.0269,.1311] | .3359 [.2064,.4969] | 1.0617 | .7258 |
| block22之后 | .3155 [.2117,.4286] | .1042 [.0638,.1454] | .4127 | .3085 |
| block27之后(原始) | .4197 [.3036,.5520] | .0000 [0,0] | 0 | 0 |

block10/16/22的D正值人数分别22/33/38，I正值人数37/36/36。
R在这三个中途切断条件下均44/44人正。R/Q完整区间见sealed summary。
I_10大于原始D_full，已直接说明I不是可分配的“剩余信息比例”。

同时把实际前文下的完整正文状态搬入错配前文forward，比较仍可访问前文
和同步切断前文两种版本（同一组人/站点，不是新的独立重复）：

| 搬运站点 | 全访问时patch增益 | 切断访问时patch增益 | 二者差 |
|---|---:|---:|---:|
| block10 | .0243 [-.0164,.0616] | -.0537 [-.1351,.0109] | .0780 [.0097,.1498] |
| block16 | .1681 [.1175,.2196] | .0839 [.0269,.1311] | .0842 [.0319,.1451] |
| block22 | .3432 [.2458,.4577] | .3155 [.2117,.4286] | .0277 [.0005,.0563] |

相同正文状态的功能作用依赖后续还能访问什么前文。不能把RC2的patch
效果理解成该站点固有、与计算条件无关的个人信息含量。

## 器械验收

- FP32/MPS/eager，无KV，同RC1模型/模板/44回复及donor。6原数值检查+
  EN/JA各18条件=42技术forward；每人18条件，共834forward。
- 44人的第一层前切断均使两臂tail完全相同。这是计算图保证的端点，不是
  数据证明“人与人没有区别”。输入正文固定是前提。
- block10/16/22共132次patch+cut恢复actual同cut的tail误差全0。
  这是“剩余输入状态相同”的恒等式，不作为机制效应或新因子发现。
- 88次mask/self sham误差全0，所有改变均保持首token不变。原始RC1和
  原RC2 patch结果重新运行；记录最大误差<=4.44e-16。
- 独立audit验证792私人标量行、36非私人访问条件行、全材料/源identity、
  每层边数、全部均值/区间及保存的控制结果。没有独立神经或完整attention
  矩阵重跑。模型过程的hook检查与独立标量验算须区分。
- MPS分配采样6,174,857,728bytes，低于9GiB；189相关tests在RC4交付时通过，
  加入后续读出测试后的209项亦通过（不是全历史套件）。

## 对GRAIN理论的约束

正确对象是条件化计算，而不是一个脱离读出和可用上下文的向量距离：

```text
output = F_remaining(U_k, C_k, attention_access)
```

U_k是当前正文的状态，C_k是所有prefix状态。其联合路径对输出有作用，
但本实验删边也改变softmax归一化和输入分布；后层损伤可能包括预期前文
突然不可用的计算失配。因此D_k只叫**该干预下的保留作用**，不是互信息、
唯一中介路径或对人的人格推理。测量理论不能把这个模型作用直接赋给作者。

材料仍是44人的最后一份合格短回复，中位24.5token；前文来自AI而非人的
完整自传。我们衡量的是原回复的条件兼容，不是生成的新心理判断，也不是
跨人群可靠性。这组证据支持把“表示可读”“持续读取”“读出使用”拆开，
不支持把一般上下文依赖命名为心理构念。

## 复现与下一步

canonical：`results/grain_vector_mechanisms/vm06_response_access_r1`，identity
`7e802bdadf5b0a056ea4b861126927028896f87349e413a731743fd5d6c6651e`。
audit：`vm06_response_access_audit_r1`；器械：`vm06_response_access_technical_r1`。

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
"$PY" scripts/run_grain_response_access.py --model-dir "$R/local_observers/989aa7980e4cf806f80c7fef2b1adb7bc71aa306" --input-root '/Volumes/mobile3/projects/project persona/data_sets/MEPS+AI_conv_experiment' --preflight-dir "$R/vm06_meps_preflight_r3" --parent-dir "$R/vm06_response_context_r1" --depth-dir "$R/vm06_response_depth_r1" --output-dir "$R/new_response_access"
"$PY" scripts/audit_grain_response_access.py --run-dir "$R/new_response_access" --parent-dir "$R/vm06_response_context_r1" --depth-dir "$R/vm06_response_depth_r1" --output-dir "$R/new_response_access_audit"
```

随后完成已冻结的逐位置LM-head/KL读出分解。该步骤结束后不再在这44份
短回复上无界追加层位/切断搜索，转回完整源集合比较、同ready增量与封头后
的bridge外部联系。无效度标签的RC系列不能替代这些主目标。
