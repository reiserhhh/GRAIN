# HM24 r2：显式公共文本缓存地址

2026-10-09。科学设计完整沿用冻结的 `GRAIN_DECODER_FIBER_PLAN_V1.md`，
没有基于首个结果更换方向、半径、样本或查询。r1完成第0单位(7forward、
1backward)后发现第二查询缓存没有L16_A_actor；r1失败、代码与结果保留。

原缓存将公共body只存于首查询npz，后续query只存query状态。
r2准备阶段显式记录 reference_body_npz 与 reference_npz，并使用公共
query的donor正文npz。运行时正文从前者读取，query仍从原对应缓存读取。
完整整篇input_ids与原query评分不变，每次live状态逐字节核验不放宽。

新命名空间 `decoder_fibers_r2`，新runner与alternate verifier各自冻结。
第0单位从r1按地址、npzhash原样继承，额外登记旧jsonhash；不重新调用。
状态总调用数继承7，剩余47×7=329，整个设计合计仍336forward/48backward。
原1–4token单射事实、48固定family/query、所有±patch和结果解释边界不变。
仅修复输入布局，不修改旧神经结果或追求更好输出。
