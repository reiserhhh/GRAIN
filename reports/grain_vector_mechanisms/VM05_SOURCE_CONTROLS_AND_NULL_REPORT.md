# VM05 来源控制、完整置换与 OOF 结构诊断

2026-09-06，第十二轮更新。**本报告是阶段进展，不是VM05完成或LLM人格机制确认。**

**当前状态以第7节为准：102来源条件/2448成员选择/510个fold模型已全部验收；
13个199次null已独立重算，余2个仍运行。同材料MBTI稀疏参照已启动。**
第1--6节保留第十一轮的历史快照与数学推导，以下“仍运行”不是当前102条件状态。

第十一轮状态：
84个MBTI来源条件完成；Big5 raw六个控制完成，cue/placebo还在串行运行。
12个MBTI条件各199次置换完成；三个Big5稀疏条件的完整null仍在运行。
Windows全量Qwen/BGE/E5提取持续运行，bridge心理结果未打开。

## 1. 本轮纠正了哪些问题

1. 随机删作者不能仅匹配外层人数。现已逐外层、逐内层及bridge前全来源拟合
   匹配真实purge的训练人数；各次删除不看心理标签。独立代码重建成员。
2. 旧参照排除前后不能直接比较不同人数。原OOF先限制到同6294作者，才与
   排除99人后重新训练的头配对。旧E1暴露仍存在，不称新lockbox。
3. 完整null不能只打乱最终预测。每次打乱完整作者标签向量，并重新拟合
   所有候选头及内层选择；仅缓存各训练折内不看标签的变换。
4. 汇总相关下降不等于折内相关同幅下降。新增总协方差与AUC配对分解，
   分清相关、输出尺度及跨fold比较。不改预测，不在测试集重新校准。
5. 历史source manifest有的没有新增feature_input_sha256字段。初次null启动
   因此在拟合前报错；修复为显式记录历史缺项并保留原材料/算子及逐值OOF
   重播。正式Big5资格null使用新增完整manifest，未覆盖失败实验或改结果。

实现补充在[GRAIN_VM05_SOURCE_CONTROL_IMPLEMENTATION_V1](../../docs/GRAIN_VM05_SOURCE_CONTROL_IMPLEMENTATION_V1.md)。
全流程没有训练/下载语言模型、付费API或bridge/MEPS心理值访问；Ridge/Logistic统计头是本轮拟合对象。

## 2. MBTI来源控制：同6294作者、四轴各自官方五折

旧V7词法投影24维的作者mean/std共48列，再用训练折PCA16及Logistic读出。
不是完整当前GRAIN正式分数，也不是LLM直接人格判断。

| 轴 | raw保留讨论 AUC | raw隔离讨论 AUC | cue隔离讨论 AUC | cue隔离讨论 BA@.5 |
| --- | ---: | ---: | ---: | ---: |
| EI | .5883 | .5709 | .5506 | .4999 |
| SN | .5936 | .5833 | .5792 | .5000 |
| TF | .6775 | .6706 | .6712 | .5857 |
| JP | .5704 | .5617 | .5587 | .5242 |

比较真实purge与五个相同训练人数随机删作者的**平均分数**，不是blend新模型：

| 轴 | raw AUC差 | 配对作者95% CI | cue AUC差 | 配对作者95% CI |
| --- | ---: | --- | ---: | --- |
| EI | -.01128 | [-.02041,-.00142] | -.01469 | [-.02520,-.00386] |
| SN | +.00184 | [-.00963,+.01293] | +.00654 | [-.00448,+.01717] |
| TF | -.00436 | [-.00720,-.00116] | -.00524 | [-.00814,-.00224] |
| JP | -.00441 | [-.01132,+.00187] | -.00468 | [-.01141,+.00182] |

以上是条件于已拟合OOF的探索性区间，未做全流程重拟合，不把seed当被试。
EI/TF下降在mean-fold指标也同方向，并非纯粹pooled比较假象。
purge+cue条件中cue-placebo：EI -.02144 [-.03054,-.01281]；
SN -.00436 [-.00947,+.00055]；TF +.00085 [-.00129,+.00284]；
JP -.00279 [-.00667,+.00111]。

**解释：**旧参照作者的重用不足以解释这些关联；同作者配对差极小。
EI较依赖当前词线索和讨论来源，TF保留较强读出但也有小的讨论来源依赖。
这不是“EI全是噪声、TF全是人格”。选择社群/讨论本身也可能与人格相关；
purge改变训练作者分布，不是对人格的因果干预。作者选择与表达信息不可混为一谈。

## 3. Big5 raw：汇总下降还有读出尺度的成分

同624作者。原word+char raw baseline pooled r=.13697、mean-fold=.16246；
purge pooled=.07990、mean-fold=.15297。五个等人数随机删除的平均pooled
r=.10966（范围.08806..12644）、平均mean-fold=.15760。
purge相对五seed平均的pooled差=-.02976，CI[-.05255,-.00783]；
mean-fold差仅-.00463，五折中2折为正。

不能据此说“删除共享讨论导致人格信息损失.057”。相关的汇总同时受到每折
预测分散程度及均值影响。对每个trait，令w_k=n_k/n，c_k=w_k s_yk s_pk/(s_y s_p)：
\[
r_{\rm pooled}
=\sum_k c_k r_k+\frac{\operatorname{Cov}_w(\bar y_k,\bar p_k)}{s_y s_p}.
\]
进一步令s=\sum c_k，\bar r_w=\sum w_k r_k：
\[
r_{\rm pooled}
=\bar r_w+(s-1)\bar r_w+\sum_k(c_k-sw_k)r_k+r_{\rm between}.
\]
四项分别是按样本量加权的fold相关、分散程度对齐项、fold相关重新加权项、
fold均值间协方差项。这是总协方差恒等式，不是另造心理潜变量或新成绩。

| raw设置 | pooled r | 加权fold r | 分散对齐项 | 相关重新加权项 | 均值间项 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原参照 | .136969 | .161004 | -.010889 | -.002009 | -.011137 |
| 真实purge | .079897 | .151571 | -.036236 | -.020369 | -.015069 |

因此下降不主要是我最初考虑的“折间均值偏移”，更大部分在预测尺度和权重。
原五折alpha=[1,10,1,1,1]，purge=[10,100,1,10,1]；
这与正则化/拟合变化相容，但尚非其唯一原因的因果证明。
真实pooled预测质量下降仍成立；mean-fold不能事后替换它来宣布成功。

同样，AUC由同fold及跨fold的正负作者比较共同构成。本数据约80%的配对
发生在不同fold，故源头读出必须同时审查跨fold尺度。所有分解逐值重构，
当前91个输入run/119 trait行最大误差1.58e-16。没有改原始OOF或拟合测试校准。

## 4. 置换：关联存在不等于高预测精度

旧词法Big5 PCA16的资格null：199次，observed r=.05679；
null均值=-.01700，95%分位[-.07003,.02917]，经验上尾p=.010。
seed731与此前完整独立重拟合的r=-.03799188595826826完全相同。
缓存版observed重播最大差8.33e-17。

12个MBTI retained条件（四轴x三臂）各199次，共2388次，全部独立重算
预测文件/置换索引/SHA后通过；上尾经验p均达到.005的分辨率下限。
不能说p=0，也不能用相同p给强弱信号排序。15条件家族还有3个Big5 sparse
null未完；以未完项p=1作保守占位，当前Holm为.075、探索性BH为.00625。
即使15项最终都在.005下限，Holm也最低.075。不要偷偷缩小家族或把这个
计算分辨率问题写成“无关联”。当前不为获取星号而追加置换或更改主门槛。

初始Big5单null仅作弱中间特征/实现资格，不冒充三个强稀疏参照的完整检验。
置换检验的是无条件来源关联；不证明原文没有语义自报泄漏或讨论因果混淆。

## 5. 实际验收与尚未完成

- 84个MBTI模型条件全部完成；420个保存的fold模型在原held-out材料重播，
  最大误差1.11e-16，0次重新拟合、0次新标签读取。
- partial_r2独立成员审计包含这84个条件及1个Big5 raw purge：
  共2040次选择记录重建，60组随机/purge对照逐24次选择人数匹配。
- Big5 raw六条件均已算完；其余12个cue/placebo条件和最终102条件审计未完。
- 12个MBTI完整null已审计，三个Big5稀疏null仍在运行，不称15条件完成。
- Windows冻结队列仍在采集；shard_00/01三观察器都已备份回Mac并核对全部
  cache SHA，共每模型606作者。shard_01压缩包两机SHA相同。
- 专项101 tests通过，scoped ruff/diff通过；未重跑完整历史项目测试。
- 未改正式GRAIN计分、锁箱、MEPS用途或另一项目服务；未push。

下一顺序：完成运行中的来源控制/null与向量采集；补同材料MBTI word/char
强参照（当前只有旧词法头，不能作为唯一竞争者）；完成固定层/固定规则的
LLM与BGE/E5头、正式GRAIN ready子集及增量；全部来源头/控制封存后再bridge。
还需把外部可读性与模型实际使用的位置/方向连接，再进行VM06/07。
预测尺度/内层Pearson选参不约束尺度的一致性，列为读出诊断问题；
如研究源头校准，必须训练侧拟合、全候选同规则、另立分析，不能救本轮test成绩。

## 6. Canonical产物与复现

全部在私有results/grain_vector_mechanisms/：
- vm05_source_controls_r1：102条件批次，尚在运行；每例manifest、选择日志、
  OOF、各fold模型/full-source head及运行记录。
- vm05_source_control_audit_partial_r2：85条件部分审计、64配对对比。
- vm05_source_control_replay_mbti_r1：84条件/420个fold模型重播。
- vm05_big5_raw_count_contrast_r1.json：raw purge相对五seed对照，来源identity齐全。
- vm05_source_oof_partition_partial_r2、vm05_big5_raw_oof_partition_r1：原预测的恒等式分解。
- vm05_null_big5_grain_raw_r1：资格null199次；vm05_null_ei_grain_raw_noref_r1
  是单条件资格复跑，不计作额外独立科学重复。
- vm05_null_family_r1、vm05_null_family_audit_partial_r1：15条件批次/12已完审计。
- vm05_full_vectors_r1：Mac局部备份；whole_queue_complete=false。

脚本：
run_grain_vector_mechanism_source_controls.py、
run_grain_vector_mechanism_external_null.py、run_grain_vector_mechanism_null_batch.py、
summarize_grain_vector_source_controls.py、summarize_grain_vector_null_family.py、
summarize_grain_vector_external_replay.py、summarize_grain_vector_oof_geometry.py。

控制批次显式--resume-completed；null批次显式--resume。先核对进程与
run_status，不能为观察超时重启活跃任务。源码8eb26e9/0c5414b封存计分，
235cea7/e558d1f/c619693补独立审计与数学分解。新增审计不覆盖Windows aa51b06采集代码。

## 7. 第十二轮闭环更新

全部102条件已完成。vm05_source_control_audit_final_r1独立重建2448次成员
选择，75组随机/purge逐24处人数完全匹配，72个配对对比可复算。
vm05_source_control_replay_big5_r1补齐18条件/90个保存fold模型，最大重播
误差8.33e-17；加上此前MBTI的420个模型，合计510模型通过。
无重新拟合、无新心理值/bridge访问。

| Big5臂，同624作者 | purge pooled r | 五seed随机平均r | 配对差与95% CI | mean-fold差 |
| --- | ---: | ---: | --- | ---: |
| raw | .079897 | .109658 | -.029761 [-.052555,-.007828] | -.004630 |
| cue | .079707 | .112023 | -.032316 [-.055187,-.010095] | -.005701 |
| placebo | .065000 | .109392 | -.044392 [-.067206,-.021366] | -.001387 |

cue-raw purge差-.000190[-.005105,.004207]，cue-placebo purge差+.014707
[-.002563,.032718]。点估计不能证明人格词掩码提升预测；pooled变化仍需
与预测尺度/折内变化分开。以上均为探索性、条件于已拟合OOF的作者bootstrap。

Big5 raw sparse完整199 null也完成：observed=.136969，null均值=-.014611，
95%分位[-.066029,.033284]，p=.005，observed重播误差8.33e-17。
vm05_null_family_audit_partial_r2已独立重算13条件/2587次，余cue/placebo
未完，15条件家族及Monte Carlo分辨率规则不变。

新补同材料MBTI强稀疏参照vm05_sparse_mbti_r2，四轴/三臂共12条件运行中。
r1在任何拟合前因ExFAT的AppleDouble锁元数据误判而停；937f914修复并加
回归测试，未改统计头。不要把已出现的单fold分数当全四轴结论。

本轮末EI raw的五折及full-source head完成：6393作者，AUC=.672221，
95% CI[.655621,.688377]，mean-fold AUC=.672640。同材料旧中间投影
AUC=.587588，配对差+.084633[.068027,.101767]。但BA@.5=.503630，
预测正类比例仅.003285，真实正类比例.207727；**排序改善不等于阈值分类
已可靠**。当前只是12条件中的1个；其cue/placebo、其他轴、完整强稀疏null
与来源控制仍需完成，不能让旧中间特征的控制替它背书。

正式GRAIN几何通道完成160条件，另外50个信息分解探索完成，详见
[几何信息报告](VM05_GEOMETRY_INFORMATION_REPORT.md)。不是全当前GRAIN理论
与LLM的最终比较，也不改变本报告全来源成绩。

Windows全向量采集继续。Mac已验收shard_00/01/02三观察器备份，每模型
909作者，整体仍未完成。shard02归档324423410字节，两机SHA一致：
077ba1cb7b1569457816f9482dc4e94c940026392722ed71e10a16cc77827e15。

下一：完整null/稀疏/observer源头比较、同ready神经观察器对照与增量，
封存规则再bridge；区分可读信息与LLM功能使用，再VM06/07。仍有可执行
研究问题，不需要用户先提供新idea。未改发布scorer、锁箱、MEPS用途或push。
