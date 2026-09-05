# VM05进入条件：样本用途与观察器资源

日期：2026-09-06。本页为实施盘点，不是新的预测结果或正式完整协议。

## 当前机制样本不足以直接承担完整心理效度检验

对现有prepared official CSV仅读取user_id，用相同ID_SEED连接240作者。
没有读取Big5/MBTI数值、类别比例、fold结果或新问卷。
结果保存在results/grain_vector_mechanisms/vm05_membership_inventory_r1；
脚本为scripts/run_grain_vector_mechanism_external_inventory.py。

| 子集 | discovery144 | calibration48 | mechanism holdout48 | 合计240 |
|---|---:|---:|---:|---:|
| Big5成员（含both） | 22 | 5 | 5 | 32 |
| 保守Big5-only | 16 | 3 | 3 | 22 |
| strict bridge成员 | 6 | 2 | 2 | 10 |
| MBTI-only，每轴相同成员数 | 122 | 43 | 42 | 207 |

这里是文件成员覆盖，不是标签完整率、统计功效或每类数目。Big5-only排除
所有MBTI来源成员，MBTI-only排除所有Big5来源成员，避免把不在strict
报告集的both-label作者重新混进源头训练。主代码仍需读取标签后验证缺失与极性。

不能把240误当成240个Big5带标作者，更不能在10个bridge上重估并宣称
H2结构恢复。现有144人的无标签表征可复用为仪器/几何开发，不替代主外部样本。

## 执行次序

1. 先完成VM04独立确认；不根据其效应重新挑层、作者或语义问题。
2. 冻结VM05采集表示的小集合、单一文本预算和统计头规则，再按官方源头
   成员扩展Big5-only与MBTI-only原评论材料，保留strict bridge为后验测试。
   仅按来源及文本可用性扩展，不按分数高低挑“容易”的人。
3. 重新记录各阶段人数、标签缺失、官方fold覆盖和编码。所有投影/统计头/
   超参在训练及inner folds拟合；不要从240人整体fit新投影后直接称外推。
4. TF-IDF、BGE/E5、LLM内容池化、上下文差、GRAIN及组合使用同文本预算。
   冻结完整GRAIN目前拒绝12评论视图；报告其适用率。必须统一增加观察预算
   或分别报告完整GRAIN与未受限中间特征，不能偷偷放宽门槛后声称GRAIN增量。
5. 当前207个MBTI-only成员可以先做小型仪器对接，但不代替全源头benchmark。
   不在22个Big5-only和10个bridge上进行开放式高维搜索。

## 第二观察器资源预检

本机Mistral-7B-Instruct-v0.3 revision
c170c708c41dac9275d15a8fff4eca08d52bab71的3个safetensors均存在，
文件合计14,496,080,928字节。仅核对文件存在/大小，尚未逐tensor校验、加载
或运行hooks。Mac物理内存24GiB；Windows .20总显存约12GB，其他业务常驻
后可用约6.9GB。因此不能默认此7B模型可以直接BF16全量反向。

先做独立资源/数值预检，必要时将第二观察器限于现有推理端的行为端点。
量化行为对照不是相同坐标系白箱复现；不能混合向量或把能力差全归因于规模。
不要停止其他项目服务、强行加载造成OOM、下载新权重或付费API后不记账。
目前无需用户的新idea，明确的实施队列仍是独立确认、扩样外部联系、第二观察器。
