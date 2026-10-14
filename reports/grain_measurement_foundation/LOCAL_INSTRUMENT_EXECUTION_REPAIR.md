# M14 冷执行依赖预加载修正

2026-09-13。`local_instrument_r1`的来源构造/应用/评估已经冻结。
首次社交实例冷执行在导入Transformers时失败：文件allowlist不允许读取
依赖包metadata，导致tqdm的已安装版本被解释为None。尚无NLI forward。
错误为 `Unable to compare versions for tqdm>=4.27: need=4.27 found=None`。

不重装依赖、不改模型/参数/原文/已保存分数，不重开数值实验。
新增 `scripts/score_grain_local_instrument_cold.py` 作为正式冷入口：
先实例化同一固定本地tokenizer/model以完成依赖加载，再调用原冻结engine，
仍使用原来的严格数据allowlist。预加载无forward、不读应用原文。
总180秒预算包含预加载；manifest增加wrapper hash和预加载时间。
失败输出目录保留为 `cold_failed_preload/social_condition_profile_v1`。
这是实现隔离问题的修复，不是观察器精度救援；成功与否以修复后的实际冷执行为准。

预加载后的尝试仍在首次forward前被Hugging Face隐式凭据读取拦截。
后续入口固定HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE并关闭隐式token读取，
不增加凭据或应用数据allowlist。空失败目录保留为cold_failed_wrapper_preload
与cold_failed_implicit_auth；前者输出在上下文切换时丢失，后者已捕获完整报错。

离线flag仍未阻止该版本Hub在第二次模型实例化时读取凭据。最终修正复用预加载
资产，不在guard内重新加载：wrapper绑定与原infer_pairs相同的batch=8、无截断、
float32 CPU softmax计算，所有应用文本及forward仍在原guard内。原冻结文件不变，
wrapper hash另记。离线flag失败空目录保留为cold_failed_offline_auth。
这不是“无模型调用”冷运行：每个文本/查询仍执行实际新forward，随后独立审计其
分数与历史固定读者容差一致性，不能用既有margin充当实际冷运行。
