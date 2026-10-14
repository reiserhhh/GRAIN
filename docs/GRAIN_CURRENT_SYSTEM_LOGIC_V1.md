# GRAIN 当前系统逻辑重建 V1

2026-09-14 接手更新：E1 的自然条件入口加入来源拟合的社区×问句地址；这里是已回复上下文而非完整曝光机会。条件读出加入 `p(1-p)[(dm)^2+dSd^T]` 的可选二阶矩；统一 epsilon 修正得到失败反例。绑定层须分开作者自身与第三人内容，后者不计为作者 0 分。证据见 [接手报告](../reports/grain_measurement_foundation/CODEX_EXECUTABLE_GAP_CLOSEOUT_20260914.md)。

2026-09-14。任务 `CLAUDE-ALL-PLAN-LOGIC-20260914-01` 第二交付。
依据：31边逻辑图谱、机制轮报告、Claude闭环审计与本轮对照表。
每条边写：输入→输出、公式/算法、**学得(L)/规定(S)** 的量、假设、证据类型。
证据类型：**MI**=数学恒等式；**SIM**=声明世界中的模拟机制；**NAT**=自然文本上的观察效应；
**PSY**=心理解释/外部效标。四类不得互换；一条边可同时带多类。

## 0. 总图

```text
自然记录 X(u,t) ──► 角色/线程/时钟 ──► 原文单位与地址 ─┬─► 句法框架(通道/主体/条件/引用)  [E1]
                                                      ├─► 固定NLI读数 v_e / 校准单位 y_e   [E2]
                                                      ├─► 冻结前缀状态 H, 4维GRAIN g, 查询矩 K [E3]
                                                      └─► 词法/语义向量(TF-IDF, E5, distilroberta) [E4]
版本账(new/alias/revise/retract) [E5] ──► 活跃主张集 A_t ──► 有限均值 / 来源学习矩 (m,V,R) [E6]
条件 z(记录/设计/读取) [E7] ──► 目标核 ŝ_t = tᵀ[(I−GH)m + Gy] [E8] ──► 工作区间 [E9]
观察设计: Schur增益 / 补采 [E10]；来源pilot REML (m,V,p,λ,R) [E11]
查询路径: C/P/Q 分账、编译投影、门控 [E12]；K算子有限变化 [E13]
解释链: 地址→运算→变化→候选意义 [E14] ──► 外部效标 (PANDORA 34函数 / MEPS 46人) [E15]
制作: bundle(spec, calibration, main, bootstrap) → 冷入口 → 新人计分 [E16]
```

## 1. 文本 / 身份 / 机会 / 选择

**E1 原文→句法框架**。输入：原文与话轮角色；输出：事件框架 {predicate, actor∈{speaker,partner,other,unresolved,source_scoped}, channel∈{intention, past_action_report, preference, ability, current_experience, prospective, current_request, unspecified}, conditions(adverbial clause 地址), sign, question}。
算法：固定Stanza EWT依存解析（S：模型revision）＋显式词表规则（S：INTENT/PREFERENCE/ABILITY/STATE/REQUEST/SUPPORT）。
假设：UD关系近似命题结构；引语/来源域内不归本人。证据：SIM/NAT-接口（公开16/16、留出12/16；自然12句46框架）。
不学得任何参数；**这是规定词表，不是学得的语法**。

**E2 原文→NLI读数→校准单位**。v_e = p(entail|x,q⁺) − p(entail|x,q⁻)；y_e = (v_e − c₀ − c₁z)/(d₀ + d₁z)。
S：读者 `cross-encoder/nli-deberta-v3-base@6c749ce`；L：c,d 由来源32句最小二乘（reference=编写±1，observed=边际，conditional=True）。
假设：来源角点覆盖应用措辞；读者对措辞的漂移被线性校准吸收。证据：NAT-编写文本（RMSE .00043）；反例：抽象身份查询失效（CE2）。

**E1'/E5 版本账**。A_t 由显式引用短语（S：REPAIR/REPEAT/RETRACT正则）驱动：new/repeat/alias/revise/retract；`active_records` 重建活跃集。
有限均值 f̂_{m,c}(t)=Σ_{j∈A_t}1[m_j=m,c_j=c]z_j / Σ 1[…]。MI：alias不增证据；revise替换父主张；retract删除。
证据：SIM/NAT-编写（72/72）；MAE .958/.875/0（忽略版本/末极性/版本账）。
**当前限制**：问题必须是唯一、正向、非来源域的 help-request 目标（REQUEST∩SUPPORT）；条件必须映射到两个冻结原型；否则 `no_unique_question / condition_unmapped`。

**机会与选择**。π_u(o) 机会份额与 f_u(o) 条件反应分开（MI：机会标准化）；选择树尾部信息（SIM：校正后RMSE .633 vs 未校正1.47）。
自然：机会集合O未观测；线程结构 parent_id 提供"被提问"这一类可观测机会（本轮WP1/WP3新增）。

## 2. 表示与语义证据

**E3 冻结前缀与GRAIN坐标**。H∈R^{T×1536}（block16）；g∈R⁴ 由四slice校准z（S：29校准作者定义单位）；查询矩 M_d = T⁻¹p_d(g)ᵀH；组内 K_q = T⁻¹(RV_q)ᵀ(RV_q)，R=(I−P_g)H，V_q 来源rank4 SVD（L：来源4作者）。
F_q = c_q + ⟨B_{q,2},M_2⟩ + tr(A_qK_q)（L：A_q,c_q 来源锚定）。MI：∇_H F_within = 2RV_qA_qV_qᵀ/T。
证据：NAT-历史作者（组内方向RMSE −24.8%/−22.3%；有限变化 −33.3%/−25.1%）；反例：同单元反序保持所有矩与K但改变原生输出16/16（CE6）→ K是辅助量，不充分。

**E4 词法/语义向量**。TF-IDF(1–2gram, L：来源12句) 保留身份信息（留出r .619）；通用NLI元描述 .166（CE2）。paraphrase-distilroberta整句向量把行为混入条件坐标（条件分句隔离后 1.000/.875）。

**表示充分性（MI）**。T=R∘Φ ⇔ Φ(D₁)=Φ(D₂)⇒T(D₁)=T(D₂)；R*(Φ(E))−R*(E)=E[(E[T|E]−E[T|Φ(E)])²]。
NEXT90/91：当前矩可见作用 N/C .451/.327（新8）而源4 .008/.051；余项 query 余弦 .656；同G cell 内配置承载原生作用（NAT-历史）。

## 3. 匿名GRAIN结构与查询条件化路径

**E12 C/P/Q 分账**。block16状态分三区；对64条低→高证据边做2³组合替换；P分账73.5–80.5%（NAT-编写，560调用）。
路线A：P4/Q4→来源状态估计→P16/Q16→固定读者；路线B：C4/P4/Q4→来源chart直接投影。MI：两表达式差1.55e-15（同一估计函数的复合）。
D0 token程序：0 Transformer，T/F AUC .705（NAT-PANDORA 历史）；4块程序为 extracted-neural-program。
学得：来源系数（Ridge，α固定三档）；规定：block边界、查询模板、答案编码。
反例：同词袋归属交换18/18作用为0（顺序敏感≠主体绑定）。

**E13 门控/层累积**。层25 SwiGLU乘积法则：源固定门控/RMS算子新8 r .997/.998；丢输入或门控项 SSE/zero .201–.385（NAT-历史）。

## 4. 共享 / 私有响应函数

**E6/E8 目标核**（S：高斯工作模型；L：m,Σ,R）。
θ~(m,Σ)，y=Hθ+e，Cov(e)=R；G=ΣHᵀ(HΣHᵀ+R)⁻¹；ŝ_t = tᵀ(I−GH)m + Σ_e k_{t,e}y_e；C=Σ−GHΣ。
两单元轮廓：θ=(μ_quiet, μ_noisy)，level/condition 为派生量；过程：θ=(a,b,c,s₁..s_T)，Cov(s_t,s_v)=P e^{−λ|t−v|}。
MI：来源项+证据项精确加回（误差≤1e-8）；不同目标核对同一证据权重不同。
SIM：当前表达RMSE .2446→.1189、r .5575→.6145；社交半差r .4898→.2052（CE9：收缩损失排序）。

**响应共享（E6'）**。来源两半独立机会：Cov(A,B)→信号，Cov(A−B)/2→误差；K=Φ Λ Φᵀ 学得，ξ_u 个人；
SIM：交叉世界隐藏差分r .8414/MSE .044659 vs 常数差分.156086；平滑世界随机斜率等价（.014255 vs .014327）；
去均值 P=I−11ᵀ/C 产生负协方差（MI，不是心理机制）。

**互动/共同输入（E18）**。y_it=a_i+b_iz_t+(c_i+d_iz_t)u_t+s_it+e_it；IV：W=[1,z,r,zr]，X=[1,z,u,zu]，β̂=(WᵀX)⁻¹Wᵀy；控制函数 v。
SIM：共同原因 c RMSE .113→.054；弱分配 5.71（CE5）；恒定伙伴只识别 a+c。NAT：观察器66/72 但拒绝命题3/24（CE3）；Qwen JA 2/6（CE4）。
**规定**：r 必须是记录的分配；自然文本不补造。

## 5. 时间 / 状态 / 互动

**E19 目标核的时间版本**。ds=As dt+L dW；Φ_Δ=e^{AΔ}，Q_Δ=P−Φ_ΔPΦ_Δᵀ；T̂_l=lᵀm+k_l(y−Hm)。
SIM：不规则文本当前表达RMSE .30905→.14319；窗口覆盖 .69271/.96875。NAT：276作者短切片均值≠整篇0参照；MEPS 623对元数据只审角色。
S：动力学参数（M08）→ L：独立pilot REML（M11）：m/V/p/λ/R；bootstrap 3达上限/1触界保留。

**E7 条件 z**。当前来源：记录（cold input record.condition∈{−1,1}）、设计（模拟）、读取（frames条件从句→冻结两原型TF-IDF）。
**没有开放条件的学得分配**——这是自然应用的直接断点（R08）。

## 6. 参照计分与不确定性

**E9 工作区间**。V_t = mean_b(tᵀC_bt) + Var_b(ŝ_{t,b})；score±1.96√V；S：12次作者bootstrap（种子规则未写入spec——D02发现）。
source_relative_position=(score−tᵀm)/√(tᵀΣt)。MI：批次不变（单人计分不重中心化）。
**不是心理95%CI**；覆盖实测 .766–.984 分目标报告。

**E10 观察设计**。ΔV_t(h)=(tᵀCh)²/(hᵀCh+σ_e²)；副本增益0；来源学习设计 vs 覆盖优先 −25.9%/−30.0%；隐藏方向恰增 Var(d)=.034133（MI+SIM）；补采64来源 → .010940。

**E11/E20 运输**。覆盖 .333/.557→.969/.964（独立pilot）；条件化计量器 y'=c(z)+d(z)y 混合 a/b（MI 展开）；表观恢复率 .6748→.2960。

## 7. 解释

**E14 解释链**。证据卡（原文哈希/字符地址）→ 运算（slice均值U₀₃=¼Σg_{s,03}；查询 Q=b+T⁻¹Σp₂(g_t)ᵀβ₂h_t+T⁻¹Σr_tᵀAr_t）→ 配对变化（请求极+.013705, 8/12；时限+.076500, 12/12）→ 候选意义（声明材料身份，不由分数推）。
MI：FA/查询展开 3.55e-15/4.44e-16。PSY：历史 S 协方差 r=−.263890 不转给 U（对象不同）。

**E15 外部效标**。34函数×效标 492行（弱、座标特异）；MEPS 46人：低落/负性/合并 r .564/.422/.555（重放，三合成分数共享条目）；关系族最小q .1216。
PSY 层：**全部是旧对象的关联；新对象（版本账/当次请求/条件响应参数）无对应效标**。

## 8. 构造与运输

**E16 制作链**。prepare(bank+margins)→construct(calibration; learn_cell_moments | fit_pilot; bootstrap)→apply(score_records)→evaluate→cold(actual NLI forwards)→audit。
bundle 必需键（D02回填）。手册§4重建与r1差0；手册-only第三对象冷分 vs 缓存差1.44e-12。
S：spec（id/kind/query/clock_unit/targets/conditions/allowed_scope/hypotheses）；L：calibration, main, bootstrap。

## 9. 学得 vs 规定 的总账

| 量 | 学得(L)来源 | 规定(S) | 自然文本可得？ |
|---|---|---|---|
| 事件框架/通道/主体 | — | UD模型+词表 | 是（候选，无金标准） |
| 证据单位 v_e→y_e | 校准c,d（来源32句） | 读者、正负命题 | 是（需命题） |
| 条件 z | 两原型TF-IDF（来源） | 记录/设计 | **否（开放条件未学得）** |
| 版本账操作 | — | 显式引用正则 | 部分（需前置问题） |
| 目标核 m,Σ,R | 来源pilot | 高斯工作模型 | 需来源作者 |
| 动力学 p,λ | pilot REML | OU形式 | 需重复机会 |
| 查询函数 A_q,c_q,V_q | 来源4作者 | block16、模板 | 是（需前缀） |
| 解释身份 | — | 材料记录 | 否（需编码） |

## 10. 从逻辑重建得到的结论

1. 每条自然应用链都在 **E7（条件）** 和 **E1'（前置命题/引用）** 处断开：条件靠记录，命题靠唯一问题。
   机制本身（E5/E6/E8/E9）在有记录时可运行且已被独立复算。
2. **目标窄化**是M01/M09自然覆盖为0的一个独立原因：ledger只接受 help-request 谓词；
   E1 已能产出任意谓词的通道/主体框架，但 ledger 未使用它们。
3. 读者层（E2）在编写材料内可靠，在抽象元描述与拒绝命题上高置信失败；这些是**通道选择**问题，不是精度问题。
4. 查询路径（E12/E13）与GRAIN矩（E3）的关系已到"范围内成立+已知盲点"的状态；再扫网格不改变主线判断。
5. 心理含义层（E14/E15）计算侧已闭合；新对象效标为人类依赖，不能由计算替代。

据此，agent可执行的真正缺口是：E7的开放条件分配（含不确定性传播到E8/E9）、E1'的目标泛化与自然前置结构、
自然机会账（E1/π_u）。它们在依赖上先于任何"学习共同条件响应函数于自然文本"的尝试。
