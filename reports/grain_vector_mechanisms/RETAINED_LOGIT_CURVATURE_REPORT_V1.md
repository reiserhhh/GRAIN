# T/F读取曲率：主要在内部函数，不在最后的概率换算

2026-09-10，第十七轮，NEXT24有界实验完成。**内部读取场确实会沿输入变化；
但本次逐位置径向/切向二次近似没有读好它，不替换旧读法。** 这是模型机制
研究，不是新一轮Big5/MBTI准确度评估。未读问卷标签或临床材料。

## 1. 新发现及其数值

沿用4名参考作者、两种正文条件M0/Mu的8条保留状态R路径，每条用原3或5
积分点，0新神经调用。R是非正文、非查询的隐藏位置，包含上下文携带信息；
它不同于上轮直接扰动的GRAIN正文方向，不能把两种局部作用混成同一实验。

取p(t)=F(M,R0+t*d;x0,L)，u=logit(p)，将端点变化减去起点切线，精确拆成
最终sigmoid的曲率与内部log-odds函数的曲率。两条正文路径先在作者内平均：

| 对象 | RMS |
|---|---:|
| 原生概率变化 | .009927 |
| 总非线性余项 | .013657 |
| 最终sigmoid曲率 | .000081 |
| 内部函数余项 | .013657 |
| log-odds变化 | .056524 |
| log-odds非线性余项 | .078179 |

余项可以大于总变化，因为它与切线项抵消；不是超过100%的心理方差。
内部余项远大于当前积分误差（概率最大.0001313、logit最大.0007060）。
变换后整个梯度场的端点变化/起点范数均值为.443。
分解恒等式误差0.0e+00；概率始终处于.2005–.2511，
没有概率饱和后强行clip或低概率放大。旧积分误差复核最大差2.65e-8，来自
本轮float64收缩与旧缓存计算顺序，保留原误差、不补采节点。

因此，简单把固定线性输出改成sigmoid，不是这条R路径的主要答案。

## 2. 一次有限近似，不以评价结果调参

用4参考作者的logit梯度增量拟合一个受限二次势函数：每个保留位置只允许
沿参考向量和垂直参考向量的两种伸缩。136位置×2系数，每种正文条件一套，
固定.01收缩，不搜索维数/参数。源平均起点梯度保留；先冻结48条预测再读取
20名旧评估作者的原生端点。

source_origin使用源平均起点logit；oracle_origin使用当前作者真实b00/b10。
后者仅用于分辨基线依赖，不是独立运行方法。比较目标统一为Phi_R：两个
正文条件下概率变化的平均，不是完整F，也不是人的人格真值。

| 方法 | 作用RMSE | MSE/零响应MSE | 响应相关r | 同号 |
|---|---:|---:|---:|---:|
| frozen_full | 0.017274 | 0.8536 | 0.269 | 14/20 |
| origin_linear__oracle_origin | 0.017806 | 0.9069 | 0.297 | 13/20 |
| origin_linear__source_origin | 0.017863 | 0.9128 | 0.295 | 13/20 |
| radial_quadratic__oracle_origin | 0.019654 | 1.1049 | 0.326 | 12/20 |
| radial_quadratic__source_origin | 0.019803 | 1.1217 | 0.323 | 12/20 |
| reference_mean | 0.018117 | 0.9389 | nan | 16/20 |
| support_calibrated | 0.016866 | 0.8137 | 0.277 | 14/20 |

新二次函数的source-origin响应相关从.295到.323，但MSE从.9128到1.1217
倍零响应误差。配对作者bootstrap的MSE改善为-7.30e-5，95%区间
[-1.60e-4,-7.81e-6]。真实起点版本也没有救回，故不把相关上升包装成提升。
这是旧探索面板上的函数迁移比较，2000次bootstrap不等于新增2000名作者。

两套二次函数的参考梯度增量残差能量比仍约.9995；272系数中124个设计
能量为0，按冻结规则设0。参考梯度源自四个人，不能按隐藏维数当样本数。
梯度场总能量和某条位移上的作用另算：大残差也可能正交于位移，不能只凭
.9995判断其人格价值或断言所有二次函数都失败。

## 3. 理论推进，而非再开校准网格

现在要表示的不是一个固定方向，而是随状态变化的读取函数。完整Hessian
不应从四条路径强行估计；本次模型只试了按位置分块、块内径向/切向各向
同性的特殊形式。数学反例表明，非对角耦合可以有非零作用，却被这个形式
漏掉；这定位一种候选机制，不把反例当真实网络的实证证明。

现有可用线性/support读法保留。NEXT24本次候选关闭；按主线转入NEXT25
条件能量与NEXT26完整输入。只有所选完整接口确实需要时，再测试一个明确
的交叉方向/位置耦合项，不继续任意曲率或准确度搜索。

与用户补充的接点一致：上轮GRAIN×token配置给材料侧坐标，本轮揭示读取
函数侧的非线性。两者接合要保留位置、条件和作用目标，而非将所有token
乘一个永远固定的人格权重。精神分析/TA只沿用已有证据绑定解释接口，本轮
没有诊断或自动临床推断。详见[数学与文献](../../docs/GRAIN_RETAINED_LOGIT_CURVATURE_THEORY_V1.md)。

## 4. 交付与重算

41项相关测试（本轮16项）通过；Ruff通过。2个诊断、3个模型/预测、5个
评估文件逐字节重放，共10个。首次standalone pytest入口缺少本库import路径，
已改用python -m pytest；没有为通过测试改动数值或删案例。
所有逐作者数据、隐藏状态、梯度和模型留在ignored results；仅公开聚合表。

```bash
PY="../project persona/.venv/bin/python"
"$PY" scripts/diagnose_grain_retained_logit.py --name logic_retained_logit_user_replay
"$PY" scripts/run_grain_retained_logit_surrogate.py freeze --name logic_retained_logit_user_model
"$PY" scripts/run_grain_retained_logit_surrogate.py evaluate --name logic_retained_logit_user_model
"$PY" -m pytest -q tests/test_grain_retained_logit.py tests/test_grain_action_geometry.py tests/test_grain_query_functional.py tests/test_grain_native_material_reader.py
```

需要已有私有缓存，不下载资料/模型。计划abffe81、诊断eed065c、候选冻结
8f057e2；均本地提交，未push。全系统goal仍ACTIVE，不声称穷尽全部理论。
