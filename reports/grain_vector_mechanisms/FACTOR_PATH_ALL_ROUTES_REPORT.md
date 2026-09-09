# GRAIN factor / query-path all-route results

Date: 2026-09-10. **All four declared routes executed.**

## 1. Decision

The useful join is presently **factor interpretation through query paths**, not
a universal accuracy-enhancing concatenation. Actual pre-existing GRAIN factors
connect with model judgments; one of eight tested native-factor/query directions
exceeds this run's matched shuffled-direction control. Real questionnaire links
are smaller. Extraversion is a useful joint-prediction candidate, not an all-trait
promotion. GRAIN factors and query scores remain distinct objects.

| Route | Finding | Next use |
| --- | --- | --- |
| GFP01 factor/path association | 108 primary cells; 29 BH q<.05 links | Anonymous factor-to-query interpretation map |
| GFP02 actual activation intervention | 1/8 cells have positive paired absolute-gain CI over one norm-matched shuffled null | Follow native:F03/T-F as a model-direction candidate |
| GFP03 real criteria | Strongest individual-factor absolute r=0.217; 0/108 BH q<.05 links | Candidate associations, retain codes rather than assign personality names |
| GFP04 prediction | Extraversion GRAIN 0.113, path 0.189, joint 0.238; no uniform benefit | Trait-specific optional joint head, not replace GRAIN |

## 2. Materials and the actual system

Final association/CV set: 276 authors ({'big5_only': 213, 'mbti_only': 63}), three upstream
12-comment views each. Four whole-document, four native-comment and four
fixed-128-cross factors are scored by frozen historical TF-IDF/SVD/factor-posterior
bundles. Their basis used101 discovery and29 calibration authors. No factors,
factor names or GRAIN reference norms are fitted to the present labels. Each
factor is namespaced by operator; equal numeric factor IDs do not imply the
same axis across operators. These are actual historical factor scores, **not**
the newer query mean/configuration/final scores renamed as GRAIN.

Identity correction: source_r1 initially compared raw historical usernames to
new pseudonyms. The exact cohort hash mapping reveals one overlap (MBTI-only).
It is excluded in final source_r2, leaving213 Big5-only and63 MBTI-only authors.
The original source_r1 results and code snapshot are retained; its raw-string
overlap claim is superseded. Native interventions use 277 source authors for
label-free direction fitting; all12 evaluation authors are disjoint from both
that source and all180 historical factor-study authors. No MEPS or bridge fit,
lockbox opening, private-text cloud request, or long training was performed.

| operator | views_scored | units | median_units | median_tokens | min_units | max_units |
| --- | --- | --- | --- | --- | --- | --- |
| whole | 828 | 828 | 1.0000 | 864.0000 | 1 | 1 |
| native | 828 | 9936 | 12.0000 | 864.0000 | 12 | 12 |
| fixed_128_cross | 828 | 5997 | 7.0000 | 864.0000 | 5 | 11 |

The frozen factor support test checks text quantity, not a certificate of
universal psychometric transport. Original timestamps are absent from these
masked source-view caches, so view agreement is not cross-day reliability.
Factor-view median correlations range
0.167–0.643.
The old optional geometry gate was not widened. All source comments were retained
by minimum-length scoring; upstream masking stays unchanged. Term-flag rates are
reported in `factor_freeze.json`; a flag is not itself verified score leakage.

## 3. Association and interpretation

Largest factor / frozen-function associations (all108 primary tests retained):

| factor | query | n | pearson_r | r_lower | r_upper | q_bh |
| --- | --- | --- | --- | --- | --- | --- |
| whole:F02 | TF_cont | 63 | -0.5480 | -0.7006 | -0.3474 | 0.0001 |
| whole:F01 | TF_cont | 63 | -0.4995 | -0.6650 | -0.2873 | 0.0005 |
| native:F03 | EI_cont | 63 | -0.4675 | -0.6410 | -0.2486 | 0.0011 |
| fixed_128_cross:F02 | EI_cont | 63 | -0.4662 | -0.6401 | -0.2470 | 0.0011 |
| fixed_128_cross:F02 | JP_cont | 63 | 0.4142 | 0.1855 | 0.6003 | 0.0049 |
| fixed_128_cross:F03 | TF_cont | 63 | 0.4129 | 0.1839 | 0.5993 | 0.0049 |
| native:F04 | conscientiousness | 213 | 0.3868 | 0.2662 | 0.4954 | 5.63e-07 |
| whole:F01 | EI_cont | 63 | 0.3760 | 0.1414 | 0.5706 | 0.0117 |

Largest real questionnaire associations:

| factor | query | n | pearson_r | r_lower | r_upper | q_bh |
| --- | --- | --- | --- | --- | --- | --- |
| whole:F04 | extraversion | 213 | -0.2169 | -0.3414 | -0.0850 | 0.1562 |
| native:F02 | TF_cont | 63 | 0.2157 | -0.0339 | 0.4399 | 0.8843 |
| whole:F02 | TF_cont | 63 | -0.2144 | -0.4388 | 0.0352 | 0.8843 |
| fixed_128_cross:F02 | TF_cont | 63 | -0.2141 | -0.4386 | 0.0356 | 0.8843 |
| native:F02 | EI_cont | 63 | 0.1907 | -0.0599 | 0.4187 | 0.8843 |
| whole:F01 | JP_cont | 63 | -0.1716 | -0.4022 | 0.0796 | 0.8843 |
| native:F02 | JP_cont | 63 | -0.1554 | -0.3882 | 0.0961 | 0.8843 |
| fixed_128_cross:F02 | SN_cont | 63 | 0.1549 | -0.0966 | 0.3878 | 0.8843 |

The former concerns the model's trait judgment; the latter concerns measured
human criteria. A factor may align strongly with a model's reading strategy but
weakly with a questionnaire. That distinction is an empirical outcome, not a
reason to relabel all shared text structure as personality.

For a source-only cross-fitted GRAIN predictor of the function value, m_hat_G,
the additional explanatory accounting is exact:

    corr(m,Y) = cov(m_hat_G,Y)/(sd(m)sd(Y))
                + cov(m-m_hat_G,Y)/(sd(m)sd(Y)).

No Y enters the fit or hyperparameter choice of m_hat_G. The terms are signed
covariance contributions, not percentages, orthogonal causal components, or
mediation estimates. This post-hoc analysis asks which part of the model's
external link is shared with this GRAIN factor family.

| query | n | grain_to_function_oof_r | total_function_criterion_r | grain_linked_covariance_contribution | remaining_covariance_contribution |
| --- | --- | --- | --- | --- | --- |
| EI_cont | 63 | 0.4006 | -0.0123 | 0.0248 | -0.0371 |
| JP_cont | 63 | 0.2250 | 0.0635 | 0.0003 | 0.0632 |
| SN_cont | 63 | -0.0893 | 0.0604 | 0.0008 | 0.0596 |
| TF_cont | 63 | 0.6020 | 0.2008 | 0.0372 | 0.1637 |
| agreeableness | 213 | 0.2968 | 0.1855 | 0.0398 | 0.1457 |
| conscientiousness | 213 | 0.4623 | 0.0634 | -0.0103 | 0.0736 |
| extraversion | 213 | 0.2398 | 0.1767 | -0.0188 | 0.1955 |
| neuroticism | 213 | 0.4714 | 0.1008 | 0.0308 | 0.0700 |
| openness | 213 | 0.2527 | 0.1764 | 0.0101 | 0.1663 |

## 4. Actual vector interventions

Four source-standardized native factors predict P/Q state directions jointly
with Ridge alpha100. Move +/-0.25 of each direction while keeping original
context and all other token states unchanged, then run the actual fixed model
suffix. Controls use source-author-permuted factor directions with matched norm.
One direction-null realization is not a permutation significance test.

| query | factor | real_mean_effect | real_lower | real_upper | real_positive_fraction | null_mean_abs | abs_gain | abs_gain_lower | abs_gain_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF_cont | native:F01 | -0.0025 | -0.0030 | -0.0021 | 0.0000 | 0.0039 | -0.0014 | -0.0017 | -0.0011 |
| TF_cont | native:F02 | -0.0002 | -0.0004 | -5.03e-05 | 0.2500 | 0.0016 | -0.0013 | -0.0016 | -0.0010 |
| TF_cont | native:F03 | 0.0024 | 0.0020 | 0.0030 | 1.0000 | 0.0011 | 0.0014 | 0.0009 | 0.0020 |
| TF_cont | native:F04 | -0.0023 | -0.0031 | -0.0016 | 0.0000 | 0.0031 | -0.0008 | -0.0011 | -0.0005 |
| neuroticism | native:F01 | -4.61e-05 | -0.0001 | 2.85e-05 | 0.4167 | 0.0003 | -0.0002 | -0.0003 | -0.0001 |
| neuroticism | native:F02 | 2.73e-05 | -4.03e-05 | 9.32e-05 | 0.5833 | 7.57e-05 | 1.63e-05 | -4.75e-05 | 8.42e-05 |
| neuroticism | native:F03 | -0.0005 | -0.0005 | -0.0004 | 0.0000 | 0.0006 | -0.0001 | -0.0002 | -1.58e-05 |
| neuroticism | native:F04 | -0.0009 | -0.0010 | -0.0007 | 0.0000 | 0.0007 | 0.0001 | -5.16e-05 | 0.0003 |

Effects are half the plus-minus difference in the model's conditional score.
For native:F03/T-F, positive sign fraction is 1.000; mean effect 0.002444 versus null
absolute effect 0.001091 (about 2.24x). The effect is small in score units but is
an actual, repeatable finite model intervention. It is **not** a change to the
author's trait. On the final source set this factor's T-F criterion r is
0.0088, so this example presently identifies a model-reading direction, not a
validated T-F construct. 5 of eight real directions are weaker than the
matched null in absolute response; generic state sensitivity is not enough.
No cross-query vector axis equivalence is assumed: each query gets its own
literal-token-compatible fitted direction and displacement units.

Cost: 408 native calls, 254.0s;
384 signed suffix edits,24 exact full-state replays, no model training.
Affine predicted effects and curvature are also saved; affine/native agreement
alone is not substituted for actual replay.

## 5. Prediction comparison and metric diagnosis

Nested author CV: 5 outer folds,3 inner folds; Ridge alpha1/10/100/1000 selected
only by inner MSE. Scalers are train-only. All four arms share the same splits;
GRAIN12/path3/joint15/interactions51. Frozen function/native are descriptive
unfitted references, not additional supervised-CV heads. Continuous scores are
not clipped or automatically calibrated probabilities.

Pooled out-of-fold Pearson r:

| query | frozen_function | grain | interaction | joint | native | path |
| --- | --- | --- | --- | --- | --- | --- |
| EI_cont | -0.0123 | -0.2189 | -0.2019 | -0.1940 | 0.0484 | -0.1963 |
| JP_cont | 0.0635 | -0.3248 | -0.3837 | -0.4311 | -0.1102 | -0.4177 |
| SN_cont | 0.0604 | -0.2339 | -0.0898 | -0.1362 | 0.1600 | 0.0436 |
| TF_cont | 0.2008 | -0.2178 | -0.0998 | -0.0979 | 0.1919 | -0.0711 |
| agreeableness | 0.1855 | 0.0653 | 0.0836 | 0.1032 | 0.1685 | 0.1244 |
| conscientiousness | 0.0634 | 0.0285 | -0.0183 | 0.0572 | 0.1294 | 0.0499 |
| extraversion | 0.1767 | 0.1127 | 0.1775 | 0.2380 | 0.2109 | 0.1890 |
| neuroticism | 0.1008 | -0.1122 | -0.1006 | -0.1175 | 0.1765 | -0.1352 |
| openness | 0.1764 | -0.0077 | 0.1428 | 0.1356 | 0.1501 | 0.1209 |

Pooled axis AUC:

| query | frozen_function | grain | interaction | joint | native | path |
| --- | --- | --- | --- | --- | --- | --- |
| EI_cont | 0.4951 | 0.4216 | 0.3464 | 0.3840 | 0.5229 | 0.3219 |
| JP_cont | 0.5144 | 0.3025 | 0.2654 | 0.2469 | 0.4532 | 0.2541 |
| SN_cont | 0.5678 | 0.2415 | 0.4746 | 0.3686 | 0.6992 | 0.5169 |
| TF_cont | 0.6154 | 0.3784 | 0.4064 | 0.4574 | 0.5956 | 0.4439 |

Extraversion participant-paired bootstrap (1000 draws):

| candidate | reference | delta_r | delta_r_lower | delta_r_upper |
| --- | --- | --- | --- | --- |
| joint | path | 0.0490 | -0.0464 | 0.1450 |
| joint | grain | 0.1253 | 0.0131 | 0.2467 |

Joint changes E over GRAIN alone by 0.125 [0.013,0.247]; over path-only by
0.049 [-0.046,0.145]. Interaction terms do not improve E in this packet. Neither the largest
single gain nor these CIs establish all-trait benefit; intervals resample fixed
OOF predictions, not an entirely retrained pipeline each draw.

Important metric observation: fold-specific training means alone can correlate
negatively with held labels. Do not interpret the negative pooled MBTI r as a
negative personality relationship. Here is the preserved supplementary ranking
and calibration diagnosis, without changing any predictions:

| query | arm | within_fold_centered_r | mean_within_fold_auc | auc_valid_folds | minority_count | training_mean_pooled_r | mse_gain_vs_training_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EI_cont | joint | -0.1198 | 0.4506 | 5 | 12.0000 | -0.1570 | -0.0013 |
| EI_cont | path | -0.2608 | 0.2712 | 5 | 12.0000 | -0.1570 | -0.0005 |
| JP_cont | joint | -0.3069 | 0.3616 | 5 | 27.0000 | -0.3310 | -0.0130 |
| JP_cont | path | -0.2799 | 0.3205 | 5 | 27.0000 | -0.3310 | -0.0056 |
| SN_cont | joint | -0.0234 | 0.6414 | 3 | 4.0000 | -0.2388 | -8.07e-05 |
| SN_cont | path | 0.0968 | 0.7273 | 3 | 4.0000 | -0.2388 | 0.0005 |
| TF_cont | joint | 0.0456 | 0.6595 | 5 | 26.0000 | -0.3120 | -0.0143 |
| TF_cont | path | 0.0883 | 0.6377 | 5 | 26.0000 | -0.3120 | -0.0030 |

SN has only four minority authors and one single-class test fold. Its pooled
AUC is particularly unstable; this tiny source panel is not the official MBTI
benchmark. One label permutation per arm gives the following pooled-r range:

| arm | min | mean | max |
| --- | --- | --- | --- |
| grain | -0.4089 | -0.1768 | -0.0417 |
| interaction | -0.3981 | -0.1211 | 0.0604 |
| joint | -0.4037 | -0.1704 | -0.0058 |
| path | -0.3692 | -0.1572 | 0.0445 |

These negative controls are not high positive leakage signals; negative null
correlations also expose the pooled-fold intercept issue. No signs or AUCs
were flipped after inspection. No threshold, alpha grid, factor or observation
operator was chosen using held labels. The 277-to276 identity correction changes
MBTI fold allocation as well as membership; it is not an accuracy improvement.

## 6. Reproduction and next decision

Plan: `docs/GRAIN_FACTOR_PATH_ALL_ROUTES_PLAN_V1.md`.
Private CPU: `results/grain_vector_mechanisms/factor_path_routes_v1/source_r2`.
Private native: `results/grain_vector_mechanisms/factor_path_routes_v1/native_r1`.

```bash
"../project persona/.venv/bin/python" scripts/run_grain_factor_path_routes.py --output-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_source
"../project persona/.venv/bin/python" scripts/run_grain_factor_path_interventions.py --source-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_source --output-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_native
"../project persona/.venv/bin/python" scripts/analyze_grain_factor_path_routes.py --source-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_source --native-dir results/grain_vector_mechanisms/factor_path_routes_v1/new_native --report-path results/grain_vector_mechanisms/factor_path_routes_v1/new_report.md
```

Fresh commands use the corrected276-person direction source. To reproduce the
historical native_r1 exactly, use its original277-source directions and
source_r1/code_snapshot.zip; the report deliberately records that distinction.
Never overwrite completed result folders. Replaying `--stage analyze` requires
no neural calls. Quantitative tables, predictions, state directions and per-unit
packets stay local/gitignored; only aggregate findings and code are public.

Priority now: connect **the same factor** across all three matrices,
A=factor/model association, J=actual intervention sensitivity, B=factor/real
criterion association. Whole:F02/T-F has the stronger observational query link;
native:F03/T-F has the cleaner current intervention candidate; whole:F04/E has
the stronger external candidate. They are not interchangeable discoveries.
Choose the next bounded experiment to connect those identities rather than
expand a predictive grid. Keep the E joint head as an optional use case.

This pilot closes four executable routes on12 existing factors, not all GRAIN
versions or all possible psychological constructs. MEPS small traits, dynamic
response factors and population-specific construct naming remain future work.
Theoretical precedents: [TCAV](https://proceedings.mlr.press/v80/kim18d.html),
[concept bottlenecks](https://proceedings.mlr.press/v119/koh20a.html), and
[causal abstraction](https://www.jmlr.org/papers/v26/23-0058.html).
