# GRAIN参照功能框架v1：如何构造和重放

2026-09-08。研究方法组件，独立于历史发布评分器。理论以
[参照函数框架](GRAIN_REFERENCE_FUNCTION_FRAMEWORK_V1.md)为准。

## 1. 固定什么

固定授权材料、作者单位、采样预算、观察器/分词器/模板/精度/层、查询与读数
方向、参考作者和干预规则。本轮为本地Qwen固定checkpoint，block16边界，
TF、N、confirmation-seeking三个查询，12评论视图；不是全PANDORA新benchmark。
确切model路径在`run_grain_real_author_path.py`，哈希在各manifest/observer。

人体量表不参与参照矩阵构造。reference、函数估计人和评估人分开；外部联系
只重用已有源结果。确认倾向查询只作功能实例，目前没有对应真人量表链接。
具体材料、author ID和结果只放`results/`，不向第三方LLM/API发送。

## 2. 从文本到数学读数

```text
授权作者评论列表 + 固定query
    -> question_mask / token_view：冻结模板、token身份、查询区mask
    -> capture_prefix：固定模型block0..16；取得block16状态
    -> Z=boundary[query_mask]，h=boundary[-1]
    -> b + <G,Z-Zbar> + g'(h-hbar)
    -> 同式计算编辑材料；两者相减是自然变化候选
    -> 三query的水平/变化组成6坐标
    -> 用固定reference中心/尺度获得群体相对位置
```

现有prefix实现还提取layer17 Q/K/V后提前退出，但不执行该层attention输出、
MLP、后续blocks、最终norm/lm_head。它不生成回答或推理文本；仍依赖预训练
前缀知识，完整模型权重目前仍驻内存。不是“无LLM编码器”的算法。

输入以同一作者的`bodies: list[str]`交给现有`question_mask(tokenizer,bodies,q)`。
它返回文本、spans、encoded和mask。空文本、超3072token、query token变化等
应报错或先按冻结材料规则采样，不能静默改变模板后沿用旧矩阵。

## 3. 构造参考函数

本轮先在两位参考作者的完整材料和固定E1编辑上计算P区积分敏感度，在两个
证据背景平均；Q使用完整原生末token梯度。参考完整状态和完整分数均值给出
Zbar、hbar、b。每个query独立保留G/g和token身份，不把词数不同的查询强行对齐。
两人只是本次构造见证，不是推荐的总体常模样本量。

已有来源入口按依赖顺序：full-state edit / query-region functional
-> `run_grain_level_response.py prepare/run/analyze`
-> `analyze_grain_level_configuration.py`
-> `run_grain_prefix_source_connection.py prepare/run/evaluate`。
不要为重放结论重新运行这些神经采集器；它们用于在授权新材料上重新构造。
参考输入、代码、模型、token identity和运行成本均在manifest；改变观察器要新建
参考，不将旧系数宣称成坐标无关常数。

## 4. 统一API

```python
from grain_research.vector_mechanisms.reference_system import FrozenReferenceSystem

system = FrozenReferenceSystem.load(
    "results/grain_vector_mechanisms/complete_level_response_v1"
)
# Z/h must come from the frozen observer, not an arbitrary embedding encoder.
readout = system.score(query, Z, h, question_token_ids, observer_fingerprint)
change = system.response(
    query, Z, h, edited_Z, edited_h, question_token_ids, observer_fingerprint
)
```

`observer_fingerprint`是冻结observer JSON的SHA256。调用者须验证真实编码
配置，不应把旧fingerprint随意贴给新向量。库检查ref文件哈希、token身份、
观察器声明、shape和有限值；它不访问人体标签、不重新拟合、不裁剪外推分数。
输出P/Q贡献、完整候选值、参照值、自然变化和锚定候选，各自保留单位。

`reference_coordinates(values, center, scale)`以固定参考生成位置，不随本次
评分的同批人改变。`query_function_gram(values)`在同一组材料上给出查询功能
协方差，保留共享与专属信息；不是对不同query的token序列做裸向量相关。

## 5. 最小重放

从独立仓库根目录，Python3.11+需numpy/pandas/scipy/matplotlib和现有torch环境。
本机解释器为相邻Persona的`.venv/bin/python`。不联网安装、不读新数据。

```bash
python -m pytest tests/test_grain_reference_system.py tests/test_grain_endpoint_constraint.py -q
python scripts/run_grain_framework_replay.py --output results/grain_vector_mechanisms/my_reference_replay
```

该入口读取36份已有P/Q包和固定reference，不运行神经模型，不读取问卷。
它核对108分数/响应、自然账本、6维作者参考配置，计算9组作者bootstrap指标。
输出`summary.json`、`manifest.json`、`complete_metrics.csv`、`functional_gram.json`、
私有逐材料预测和`private_complete_readout_scatter.png`。输出目录必须不存在。
图像含个体采点，所以仅本地审阅，不将它当作公共匿名数据自动发布。

## 6. 研究组件与开放扩展

`configuration_parity.py`/`odd_action.py`定义并计算方向/偶对称作用；
`endpoint_constraint.py`说明单作用约束下内部场的不唯一性。它们提供机制
分析与有限读出，不默认加到P/Q总分里。

如要继续压缩条件/方向函数，先定义目标是全值、自然变化、干预作用或人体
外部联系中的哪一个，再选择能区分新机制的设计。M20/M21既有六函数结果
保留，不在已暴露20人上继续挑alpha。新增参照、更多查询与persona写入/读出
双向性属于下一研究阶段；不混入本次v1交付或旧锁箱确认。
