# GRAIN Theory Manuscript: Trilingual Crosswalk and Editorial Contract

Status: internal review companion for the English, Japanese, and Chinese
version 1 manuscripts. This file is not part of the submitted article.

## 1. Canonical files

| Language | File | Editorial role |
|---|---|---|
| English | `GRAIN_THEORY_MANUSCRIPT_EN_V1.md` | Submission master unless a Japanese- or Chinese-language journal is selected |
| Japanese | `GRAIN_THEORY_MANUSCRIPT_JA_V1.md` | Independently written Japanese academic review text |
| Chinese | `GRAIN_THEORY_MANUSCRIPT_ZH_V1.md` | Independently written Chinese academic review text |

The Japanese and Chinese versions preserve the English argument, equations,
evidence, and claim boundaries, but are not sentence-by-sentence translations.
Terminology follows established usage in psychology, psychometrics,
longitudinal modeling, and computational social science.

## 2. Argument crosswalk

| Section | English function | Japanese section | Chinese section |
|---|---|---|---|
| Abstract | Scope, contribution, and boundary | 要旨 | 摘要 |
| 1 | Problem and contribution | 序論 | 引言 |
| 2 | Prediction, situated behavior, state-trait, operators | 概念的基盤 | 概念基础 |
| 3 | Formal context-conditional multilevel model | 文脈条件付き多水準モデル | 情境条件化的多层模型 |
| 4 | Development and validation workflow | 開発と妥当化の手順 | 开发与效度验证程序 |
| 5 | Audited evidence and nonclaims | 経験的動機と現在の証拠境界 | 经验动机与当前证据边界 |
| 6 | Eight falsifiable propositions | 反証可能な命題 | 可证伪命题 |
| 7 | Implications for psychological science | 心理科学への含意 | 对心理科学的意义 |
| 8 | Limitations and next study | 限界と今後の優先課題 | 局限与研究重点 |
| 9 | Conclusion | 結論 | 结论 |

## 3. Controlled terminology

| English | Japanese | Chinese | Usage note |
|---|---|---|---|
| natural-language psychological measurement | 自然言語に基づく心理測定 | 自然语言心理测量 | Preferred umbrella term |
| individual language texture | 個人言語テクスチャ | 个体语言纹理 | Reproducible multiscale organization of local language events; never a single embedding |
| context-conditional | 文脈条件付き | 情境条件化 | Avoid project metaphors such as rind or skin |
| observation operator | 観測作用素 | 观测算子 | Covers segmentation and representation |
| condition selection | 条件選択 | 条件选择 | Distinct from conditional response |
| conditional response | 条件付き反応 | 条件内反应 | Response under observed or assigned conditions |
| opportunity structure | 表現機会構造 | 表达机会结构 | What could have been expressed |
| reference population | 参照母集団 | 参照总体 | Must travel with a score |
| generalization universe | 一般化可能性の範囲 | 概化全域 | Generalizability-theory sense |
| construct validity | 構成概念妥当性 | 构念效度 | Never inferred from prediction alone |
| measurement invariance | 測定不変性 | 测量不变性 | Must be empirically tested |
| between-person / within-person | 個人間／個人内 | 个体间／个体内 | Never interchanged |
| person-specific | 個人固有 | 个体特异 | Requires reproducibility and uncertainty |
| multitrait-multimethod | 多特性・多方法 | 多特质-多方法 | External validity design |
| uninterpreted technical variable | 未解釈の技術変数 | 尚未解释的技术变量 | Used before psychological interpretation |
| abstention / no-score rule | 出力保留規則 | 不予报告规则 | Preferred over forced extrapolation |
| large language model | 大規模言語モデル | 大语言模型 | Interpretive aid, not scoring authority |

## 4. Equation crosswalk

| Equation | Object |
|---:|---|
| 1 | Person-specific condition selection |
| 2 | Conditional expression distribution |
| 3 | Versioned segmentation and representation |
| 4 | Behavior-to-use inferential chain |
| 5 | Minimal event decomposition |
| 6 | Person-specific transition kernel |
| 7 | Reference-relative measurement candidate |
| 8 | Between-person coefficient |
| 9 | Pooled fixed-effects within-person coefficient |
| 10 | Person-specific coefficient |
| 11 | Projection-specific cross-level transport diagnostic |
| 12 | Stable component, drift, and error decomposition |
| 13 | Residual-source decomposition |
| 14 | Prospective behavioral linkage model |
| 15 | Deterministic scoring and LLM-supported explanation |

## 5. Claims that must remain identical

1. GRAIN is an analysis method for developing future measures, not a
   finished scale, test, diagnosis, or personality taxonomy.
2. Technical reproducibility, measurement comparability, construct validity,
   and use validity are separate claims.
3. Context has two potentially distinct roles: selected behavior and a
   condition for response.
4. Individual, relational, and population outputs are not interchangeable.
5. Common units do not establish measurement invariance.
6. The empirical program contains strong author-reproducible structure but
   only limited Big Five linkage; author identification is not personality
   validity.
7. Synthetic recovery establishes finite-model identifiability, not human
   prevalence or psychological meaning.
8. Stable scoring is deterministic and versioned; an LLM may explain a frozen
   evidence packet but may not alter scores or override abstention.
9. Current evidence does not support clinical diagnosis or individual
   decision-making.

## 6. Remaining submission decisions

- Select target journal and apply its manuscript category, length, heading,
  anonymization, and generative-AI policies.
- Supply author names, affiliations, corresponding-author details, funding,
  conflicts of interest, and CRediT roles.
- Create an anonymous public archive for code, aggregate reports, protocols,
  and the claims ledger; do not redistribute restricted corpora.
- Preserve `GRAIN_THEORY_EVIDENCE_PROVENANCE.md` as the internal map from
  manuscript numbers to primary repository reports.
- Decide whether the empirical evidence table remains in the main text or
  moves to an online supplement.
- Obtain an external Japanese academic-language review if submitting the
  Japanese version; the same applies to Chinese-language submission.
- Run a final DOI, page-number, and in-text-citation audit against the selected
  journal style.

## 7. Addendum (2026-09-13): mechanism-round supplement files

| Language | File | Role |
|---|---|---|
| English | `GRAIN_MECHANISM_SUPPLEMENT_EN_V1.md` | Supplement master; paragraph order shared by all three |
| Japanese | `GRAIN_MECHANISM_SUPPLEMENT_JA_V1.md` | Parallel Japanese text, same tables and claims |
| Chinese | `GRAIN_MECHANISM_SUPPLEMENT_ZH_V1.md` | Parallel Chinese text, same tables and claims |
| Index | `GRAIN_MECHANISM_THEORY_PACKAGE_V1.md` | Inherited theorems / local findings / counterexamples / scope, with artifact paths |
| Provenance | `GRAIN_MECHANISM_EVIDENCE_PROVENANCE.md` | Artifact location of every number in the supplement |

Additional controlled terminology:

| English | Japanese | Chinese | Usage note |
|---|---|---|---|
| evidence identity | 証拠同一性 | 证据身份 | Intention, reported action, preference, ability, current experience, current request: coexisting channels, not six traits |
| versioned claim ledger | 版本付き主張台帳 | 版本化主张账 | New / alias / revise / retract operations; not a probability posterior |
| working interval | 作業区間 | 工作区间 | Working-model variance plus finite source-construction variance; never a psychological 95% CI |
| source-constructed instrument | ソース構成測定具 | 来源构造工具 | Parameters learned from independent source authors; applied authors never fitted |
| target kernel | 目標核 | 目标核 | Affine readout for one declared target; different targets weight the same evidence differently |
| speech source / proposition actor | 発話源／命題行為者 | 发话者／命题行为者 | Who speaks versus who is proposed to act |

Claims that must remain identical, added to Section 5: (10) simulated parameter recovery is not human
prevalence or validity; (11) explicit reference rules are not a learned universal grammar; (12) a
cross-agent rebuild with development context is not a blind third-party replication.

## 8. Addendum (2026-11): liquid-trajectory, Riccati-Möbius, and interpersonal-dynamic supplement files

| Language | File | Role |
|---|---|---|
| English | `GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md` | Version 2 supplement master; heading structure shared by all three |
| Chinese | `GRAIN_LIQUID_RICCATI_SUPPLEMENT_ZH_V2.md` | Parallel Chinese academic text, same sections, metrics, and boundaries |
| Japanese | `GRAIN_LIQUID_RICCATI_SUPPLEMENT_JA_V2.md` | Parallel Japanese academic text, same sections, metrics, and boundaries |
| Provenance | `GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md` | Source map linking standalone modules/tests and companion `project-persona` formal/empirical outputs |

Additional controlled terminology for Version 2:

| English | Japanese | Chinese | Usage note |
|---|---|---|---|
| constrained Riccati-Möbius flow | 制約付き Riccati-Möbius 流 | 受约束 Riccati-Möbius 精确流 | Exact bounded $2\times 2$ ZOH matrix exponential integrating the continuous scalar Riccati ODE; preserves `dt = 0` and subdivision semigroup consistency |
| dual-pathway episodic observer | 二重経路エピソード観測器 | 双通路情景观测器 | Couples fast continuous decay across idle delays with content-addressable cosine retrieval; retrieval weights are not causal token interventions |
| dynamic set inversion | 多時点動的集合反転 | 动态多时点集合反演 | Cross-view lag-1 rate separation + HiGHS trajectory polytope bounds; degenerates when shared and private rates coincide |
| interpersonal evocation and plasticity | 対人喚起と対人可塑性 | 人际唤起与人际可塑性 | Partner-elicited behavioral shifts and contingent slope adaptation measured across strictly disjoint partner panels |
| machine-checked formal verification | 機械検証済み形式証明 | 机器核验形式化证明 | Lean 4 theorem verification under declared axioms (`propext`, `Classical.choice`, `Quot.sound`); never replaces empirical human validation |

Claims that must remain identical across EN/JA/ZH, added to Section 5: (13) machine-checked Lean 4
theorems certify deductive soundness under stated premises, not psychological construct validity;
(14) discrete liquid ODE approximations must be verified against time-subdivision invariance and
dual-pathway retrieval cannot be misread as causal token intervention; (15) disjoint public human
cohorts (RealPersonaChat, ESConv, PSE-Database, Bar-Anan) cannot be merged into one latent person
or used for clinical diagnosis; (16) pooled cross-validated `R²`, within-panel `ΔR²`, strict
cross-panel generalization `r`, and `Self - Peer` discrepancy variance must be reported separately
alongside null findings.
