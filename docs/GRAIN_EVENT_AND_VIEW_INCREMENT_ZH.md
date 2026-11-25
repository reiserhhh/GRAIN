# 使用手册：从对话查证一个行为，或判断双视角分解值不值得做

[English](GRAIN_EVENT_AND_VIEW_INCREMENT.md) | **简体中文** | [日本語](GRAIN_EVENT_AND_VIEW_INCREMENT_JA.md)

两个入口彼此独立，均不重算或改写原生因子。从当前主分支的仓库根目录运行。建议使用 Python 3.12，新环境先执行 `python -m pip install -r requirements-lock-main.txt`。以下示例不需模型、语料或 API。

## 对话：纠正以后，本人采用了新前提吗？

```sh
python -m grain_core.event_response examples/event_response.json
```

[示例输入](../examples/event_response.json)中，搭档将 red 纠正为 blue，焦点人答 “I'll use blue.”。输出 `events[0].status = "support"` 表示确实说出了采用新前提的回应，并非已经执行。指定对话、焦点人、竞争前提和纠正轮，轮次从 0 开始，`cutoff` 当轮可见。只处理紧接着的本人回应。返回原句地址、说话者与截止点；据此区分明确反证（`refute`）、答了但不足以判断、尚未看到本人回答。后两者均不当作反证。

内置识别只覆盖显式英语句式。自由改写或其他语言可接由独立读者预先编码的触发与回应，均需标明版本并匹配前缀指纹；普通语义候选不投票，已编码证据冲突则待核。可用 `--liquid` 附上同一输入的冻结轨迹。后续独立动作只进入 `later_checks`，不会改写先前判断。输出含原话，私密对话的 JSON 不应直接共享。

条件卡按本人确实作答的机会分层。若某层有 n 次，支持 s 次，反证 f 次，可能的响应比例为 [s/n, 1−f/n]；未观测层为 [0,1]。示例的已观测层权重为 0.75、未观测层为 0.25，故 `reference_rate_bounds = [0.75, 1.0]`。这是未定事件的补全范围，不是人格概率或抽样置信区间。同一行动不可重复计数。

## 数值：两条历史共同排除了什么解释？

```sh
python -m grain_core.dual_view_increment examples/dual_view_increment.json
```

输入两个同坐标、同截止、步长对应的序列；缺交易日或机会时先断开，不插值冒充观测。日期可用两列相同的 ISO 日期，工具只核对匹配与先后，不核对交易日历。不提供校准时返回两原值、均值与有向差。若以交叉矩校准，`previous_levels` 应取独立校准期；隐含共同或私有方差非正则返回 `unsupported_calibration`，不强行估速率。

[示例](../examples/dual_view_increment.json)中 `rates`、`budgets` 按“共同、第一视角、第二视角”排列；另给逐步静态半宽、外界和校准标识。依次读 `static`（当期）、`common_only`（只加共同趋势）、`simple_lag`（两条历史分别传播再求交）、`full`（必须共用同一条历史状态）。例中后两者分别为 [−0.4, 2.0] 与 [0.8, 1.2]。

| `vs_simple_lag` | 下游应如何处理 |
|---|---|
| `simple_lag_sufficient` | 简单方法给出同样的区间，保留简单答案。 |
| `narrows_only` | 范围缩小，针对事先指定阈值的判断不变。 |
| `changes_threshold_decision` | 只有联合约束排除了阈值；记录它排除了哪一种解释。 |
| `null` | 看 `status`：缺校准、校准不受支持或约束不相容均无完整分解值；不要填零。 |

共同成分为正和次日开盘上涨是两个命题。要评价后者，须在相同截止、日期和可覆盖样本上，将简单双视角历史与加入分解后的方法配对比较。
