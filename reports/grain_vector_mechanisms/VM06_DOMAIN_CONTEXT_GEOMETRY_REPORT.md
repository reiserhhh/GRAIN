# VM06: domain context, independent geometry and lexical reuse

2026-09-06. **Complete bounded domain packet and independent saved-data audits.**
The full vector/personality program, source-only comparisons and bridge remain open.

## Design and scope

The frozen context protocol (511367a) selected96 Essays documents,55 English X
accounts and16 Japanese X accounts without reading psychological or market
outcomes. This is167 cases, not all2,467 Essays or all19,954 X accounts.
Essays context is the first three raw sextiles and current text the sixth, with
two intervening sextiles omitted by design. X context uses three earlier posts,
current text the sixth of six chronological spread posts, with two gap posts.
Complete current/context spans meet the pinned Qwen token envelope; no silent
truncation. Exact copied texts shared across accounts and within-account exact
duplicates are excluded, but near copies, repeated templates and automated
accounts are not thereby eliminated.

Matched and random other-source contexts are frozen bijective derangements
within corpus/language/symbol cells. Matching minimizes log context-length
differences; symbol is collector metadata, not experimental topic assignment.
One case per account; Essays is within-document continuity, not retest.
All source text is assigned the **user** role. The same current content tokens
and positions are used in six conditions: own, matched, random, none, own-block8,
matched-block8. Block8 permutes context-content tokens only, preserving the bag.
Fixed FP32 Qwen/eager execution has no KV reuse and retains the native readout.

The primary endpoint is average observed-token log probability after the first
current token. This measures how much a context helps this frozen model explain
the supplied continuation. It is not generated text quality, personality-score
accuracy, human intelligence or a psychological trait estimate. All intervals
below are nominal95% fixed-case bootstrap intervals (1,000 draws, seed731),
conditional on this material and donor assignment. Shared donors/selected cells
limit population inference; no confirmatory multiple-testing claim is made.
Scoring is teacher-forced: preceding current-text tokens are supplied in every
condition. This is not a free-generation forecast of an unseen whole document.

## 1. Source context has functional value

| Material | N | Own minus matched tail logp [CI] | Positive cases | Own minus none | Native minus block8 advantage [CI] |
|---|---:|---:|---:|---:|---:|
| Essays, English | 96 | .16815 [.14006,.19451] | 92 | .25392 | .04851 [.03569,.06331] |
| X, English | 55 | .93911 [.65858,1.25144] | 48 | 1.42119 | .16430 [.10575,.22632] |
| X, Japanese | 16 | .27123 [.03704,.62521] | 12 | .73853 | .09229 [.00870,.19378] |

Own-minus-random differences are .16511/1.01204/.30815, respectively. After
block8 permutation, own-minus-matched remains .11964/.77480/.17894. Thus token
content alone retains part of the advantage, while ordered context contributes
additional function under this intervention. Block shuffling also degrades
naturalness; the contrast does not identify a pure order or personality fraction.
Do not rank language ability by these absolute logp gaps: materials, baselines,
tokenization and selected account counts differ.

Current input embeddings are exactly identical across conditions. Final pooled
state cosine own/matched is .99360/.99044/.99448, despite nonzero readout effects.
This repeats RC5's distinction between pooled-vector closeness and model function,
without needing to posit an undiscovered psychological direction.

## 2. Independent semantic geometry tracks some functional differences

Protocol e4eac66 was frozen before inspecting the private context-effect table.
The same raw contexts/current texts are independently encoded by cached BGE,
English E5 and multilingual E5. English encoders see151 English cases only;
multilingual E5 sees all167. Shared contiguous character windows fit all three
tokenizers, preserve all characters and use the same character-count aggregation
weights. This is normalized window pooling, not full-dialogue attention.
Native BGE CLS and E5 mean pooling differ, so comparisons concern complete
observer packages, not architecture size in isolation.

For each case,

`g_i = cos(E(current_i), E(own_context_i))
       - cos(E(current_i), E(matched_context_i))`;
`d_i = tail_logp(own_context_i) - tail_logp(matched_context_i)`.

| Observer | Material | Same-source percentile | corr(g,d) [CI] | Partial r, context-length difference only |
|---|---|---:|---:|---:|
| BGE | Essays en | .8208 | .3496 [.1318,.5300] | .3491 |
| E5 | Essays en | .8024 | .4469 [.2383,.5922] | .4474 |
| Multilingual E5 | Essays en | .8091 | .4115 [.2077,.5516] | .4149 |
| BGE | X en | .8190 | .7212 [.5847,.8348] | .7397 |
| E5 | X en | .8305 | .8987 [.8235,.9494] | .9062 |
| Multilingual E5 | X en | .8446 | .8145 [.6902,.8984] | .8283 |
| Multilingual E5 | X ja | .7633 | .8063 [.5873,.9304] | .8134 |

The percentile asks whether a source's context is closer than other contexts in
its frozen cell. It is not MBTI AUC or independently repeated author reliability.
Spearman associations are .236/.375/.329 in Essays and .711/.796/.773 in English
X; Japanese multilingual E5 is .782. No undefined bootstrap draws occurred here.
High X associations suggest shared content continuity is important. They do not
show that the embedding causes the Qwen effect, or that either measures Big5.

## 3. Exploratory content-reuse follow-up narrows the explanation

After seeing the above results, 334efea fixed a bounded follow-up before computing
overlap values. This is explicitly **post-hoc exploratory**, not preregistration.
No new neural calls or data selection. Native Qwen content tokens are marked
when they belong to a contiguous four-token sequence appearing in a context.
Neither-context coverage means outside sequences found in **either** context,
not merely outside their intersection. Cases need at least four selected tail
tokens. The remaining tokens may still share unigrams, shorter phrases, semantic
content, formatting and topics; this is not exhaustive removal of content reuse.

| Material | Mean neither-context coverage | Eligible cases | Effect outside either context's repeated 4-grams [CI] | Own-only repeated 4-gram effect [eligible N] |
|---|---:|---:|---:|---:|
| Essays en | .9654 | 96 | .16222 [.13465,.18792] | .46763 [47] |
| X en | .7247 | 53 | .34793 [.18367,.54943] | 2.69010 [29] |
| X ja | .8509 | 16 | .05947 [-.03679,.14247] | 1.34179 [8] |

Repeated phrases are a substantial contributor to X's large aggregate effect.
Essays and English X retain a positive outside-4-gram contrast. Japanese X's
remaining contrast is underresolved, not evidence that its residual effect is
zero or that Japanese analysis failed. These columns have different eligibility
and weights; their means are not an additive causal decomposition of the total.
Partition membership uses the observed continuation; it is a retrospective
diagnostic of saved likelihoods, not a deployable predictor-selection rule.

The second follow-up conditions g and d on an intercept, stratum indicators,
log current length, log own/matched context-length ratio, and differences in
unigram/four-token coverage. Both variables are residualized by the same OLS
design, then correlated. All nuisance fits are descriptive on these cases and
refitted within each case-bootstrap draw. They are **not** a new held-out head,
causal mediation test or operational proposal to delete personality-related cues.

| Observer | Material | Conditional r [CI] | Residual df |
|---|---|---:|---:|
| BGE | Essays en | .3199 [.0814,.5038] | 91 |
| E5 | Essays en | .3970 [.1587,.5547] | 91 |
| Multilingual E5 | Essays en | .3519 [.1298,.5064] | 91 |
| BGE | X en | .3264 [-.1013,.5921] | 45 |
| E5 | X en | .4902 [.1279,.7431] | 45 |
| Multilingual E5 | X en | .2726 [-.0201,.5469] | 45 |
| Multilingual E5 | X ja | .5998 [-.7504,.9461] | 10 |

Japanese X has one undefined draw among1,000, excluded rather than replaced by
zero. Its wide interval and low residual df preclude a stable adjusted claim.
The joint conditioning cannot attribute the decrease to any one covariate.
In-sample partial associations are not semantic/personality-specific effects.

## Consequence for GRAIN's mechanism theory

There are now separable observations of **source matching**, **context-dependent
model function**, and **independently encoded geometry** on exactly the same
texts. They are related, not interchangeable. Ordered context and content
continuity both matter; a very large cross-case r can partly reflect lexical and
collector-cell structure. Some association survives the declared descriptive
controls, but the residual is an unresolved mixture, not automatically reasoning
or personality. No universal scalar "personality direction" follows.

The useful next object is a provenance-qualified, task-specific response to
evidence, retaining context/role/content channels rather than subtracting them
wholesale. External source-only heads must determine whether this sort of
readable information links to psychological labels. That program remains open;
these results neither replace nor select its frozen source/bridge candidates.

## Execution, audit and reproduction

- Context:1,008 forwards (6 public+1,002 private),1,339.99 seconds, sampled
  MPS6.175GB. Public self-patch errors0; compact/fixed readout max4.06e-6 under
  unchanged1e-5. The first public-only attempt failed because one context had15
  rather than16 tokens; zero neural forwards in that failed packet. Only the
  public example was expanded before the complete run; the failure is retained.
- Geometry:469 case-observer pairs,938 pooled vectors,356 common windows,
  1,020 forwards,176.83 seconds. Maximum sampled MPS2.248GB. Public repeat and
  padding-replacement errors0; compact/fixed max1.57e-7. Pinned weight/tokenizer
  hashes, native position buffers and source snapshots verified.
- Context audit:167 cases,1,002 conditions,334 order manipulations,36 endpoints
  and36,000 bootstrap draws replayed. Scalar max4.44e-16, geometry7.11e-15,
  aggregate1.11e-16. Float64 recomputation can differ from saved float32 means
  by9.80e-7; this numerical distinction is recorded rather than hidden.
- Geometry audit:1,068 tokenizer/window checks,938 pooled vectors,469 case
  effects,7,000 bootstrap draws. Independent correlation algebra replaces OLS
  for the length partial; maximum aggregate discrepancy3.06e-15.
- Overlap audit:668 masks independently rebuilt by exhaustive sliding-window
  comparison,668 case partitions and19,000 bootstrap draws. Independent SVD
  projection replaces nuisance OLS; maximum discrepancy2.39e-14. No audit
  independently repeats neural extraction. All private texts/tokens/vectors are ignored.

Private canonical directories under `results/grain_vector_mechanisms/`:
`vm06_domain_context_material_r1`, `vm06_domain_context_r2`,
`vm06_domain_geometry_r1`, `vm06_domain_overlap_r1`; audits have the corresponding
`vm06_domain_*_audit_r1` names. Material replay independently reruns preparation
with the same algorithm and produces the identical case hash, not an independent
selection method. Context identity starts `3be6da50`, geometry `7b2b7f7d`,
overlap `376c1819`; full hashes are in private manifests.

Run entry points: `prepare_grain_domain_context.py`, `run_grain_domain_context.py`,
`run_grain_domain_geometry.py`, `analyze_grain_domain_overlap.py`.
Each accepts explicit parent/output directories; use a fresh output directory.
Auditors: `audit_grain_domain_context.py`, `audit_grain_domain_geometry.py`,
`audit_grain_domain_overlap.py`. The protocol documents list all fixed operators.
Model revisions, datasets and installed dependencies are unchanged; no paid API,
new model training, bridge-label access or released GRAIN scorer change.
