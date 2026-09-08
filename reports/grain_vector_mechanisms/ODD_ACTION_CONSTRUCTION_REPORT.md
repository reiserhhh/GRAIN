# M20：双向敏感度与奇作用构造

2026-09-08，COMPLETE。代码`98f0a2d`。70新后缀前向/反向对、138.94秒；
复用74正路节点，0前缀、0模型训练。拟合7.03秒，全部预测冻结后才连接旧源Y。

## 发现

奇作用可以用偶敏感度场与带方向的状态相乘表示；但本轮位置/状态有限函数
只读到一部分。N联合作用函数仍有源作者关联r=.4067，不过它与实际奇作用
的r仅.1152，幅度也明显偏小。这保留了一个可计算的作者关联，不把它描述成
找回整条实际通路。三个函数族和两种损失全部报告，无源Y择优。

每query4位reference的view0拟合，20位非reference评估，评分按三视图平均。
下面是同一20人上的结果。效应r测模型奇作用；源r测已暴露量表，不是同一指标。

| 查询/函数 | 奇作用r | MSE/零作用 | 源Y r |
|---|---:|---:|---:|
| TF position field | .2165 | 1.0898 | -.0469 |
| TF position action | .0220 | 1.4111 | -.1773 |
| TF state-even field | -.0084 | .9670 | -.0419 |
| TF state-even action | .1338 | .9548 | .2337 |
| TF combined field | .1925 | 1.0403 | -.0625 |
| TF combined action | .1281 | 1.2277 | -.0331 |
| N position field | .2317 | 1.0170 | .3307 |
| N position action | .1511 | 1.0450 | .3462 |
| N state-even field | .1244 | 1.0030 | .1390 |
| N state-even action | .0207 | .9931 | .3335 |
| N combined field | .1968 | 1.0321 | .2537 |
| N combined action | .1152 | 1.0104 | .4067 |

参照组n4的三视图均分不是view0训练拟合误差，两者不可混报。所有角色完整
数值、bootstrap和有符号源协方差余项留在CSV/JSON，没有省掉负结果。

## 数学与数值

固定评论均值C、查询xbar和d=E-C，令g+/-为两方向的积分敏感度：

```
odd = .5 <d,g+ + g->
even = .5 <d,g+ - g->
Gamma_even = T center_comment((g+ + g-)/2)
odd = mean_token <d_t,Gamma_even,t>
```

三个偶函数族为32位置余弦、64投影状态平方及两者合并；与d相乘后为奇函数。
全部144材料的反射奇性误差为0。端点重放最大8.10e-8，中心化误差约1e-9。

然而三/五节点积分只有3/16奇作用、5/16偶作用达成既定5%或1e-5容差；
奇/偶积分gap RMS分别.000858/.001181。所有单元保留，不增加节点或改门槛。
这不能用来否定精确积分恒等式，也不能把拟合不足全归于数学不可运输。

## 下一决定与重放

先做M21唯一零神经对照：以精确端点约束积分总作用，保持同样函数族，区分
数值监督误差与表示/拟合缺口。之后整合框架，不再启动方向数/alpha搜索。

`scripts/run_grain_odd_action.py prepare/collect/fit/evaluate`；私有结果在
`results/grain_vector_mechanisms/odd_action_construction_v1/`。prepare仅用于新目录；
已完成collect不应重复。`integration_diagnostics.json`、`transport_metrics.csv`、
`source_metrics.csv`、`summary.json`为主要审阅入口。
