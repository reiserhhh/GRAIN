# GRAIN dynamics, personality-query paths and measurement channels

Date: 2026-09-10. Finite execution complete. Discovery-led mechanism experiment,
not a new full-PANDORA benchmark, newly validated psychological scale, or release
scoring change. **The useful new result is an operational dynamic-factor to model
judgment connection, alongside evidence that model-readable and criterion-linked
information need not be the same part of a factor.**

## 1. Question-by-question decisions

| question | answer |
| --- | --- |
| GRAIN changes connect to model queries? | Yes, selected F02/T-F co-change plus actual held-author intervention |
| More model linkage guarantees more real-trait linkage? | No monotonic pattern across all nine queries; level and dispersion differ |
| Within-text dynamics carry candidate real-criterion information? | Yes, exploratory F02 dispersion/T-F and F04 drift/extraversion links |
| Simply add every dynamic feature? | No universal held-author gain in these fixed feature families |
| Can the gap be allocated mathematically? | Predictive MSE gain: yes; true trait/action/reporting-error separation: not with these channels |

The source has 276 reference-disjoint authors: 213 Big5-only and 63 MBTI-only.
The same nine recorded criteria are used throughout. Source labels were already
exposed in earlier work; all current supervised estimates are exploratory nested
CV. The native follow-up reuses 12 authors disjoint from source and historical
factor construction. They are not a newly recruited or newly blind cohort.

## 2. Two time scales, not one ambiguous movement axis

Restored 9,936 exact masked comments and their original
timestamps/subreddits. Every author contributes three 12-comment views from
36 distinct discussions. Median author span is 1068.9
days; median within-author comment gap is 12.6
days. These are sampled comments, not adjacent conversational responses.

The sampled-comment branch scores four three-comment blocks per view with the
frozen whole/native bundles: 3,312 blocks and eight coordinates. The separate
within-comment branch retains 9,165 of
9,936 comments (92.2%),
four lossless contiguous lexical-token spans each: 36,660 slices.
All 276 authors remain; at least 7 eligible comments
per view. These are GRAIN lexical/punctuation tokens, not Transformer tokens.
No old factor basis or reference norm is refitted.

Four points support short-path descriptors, not individual AR parameters or a
complete flow/gust process decomposition. Moving a frozen whole factor to short
spans is an explicitly new observation use; the within-comment CV baseline
therefore includes its four new slice means. Neither branch replaces the released
GRAIN dynamic measurement system. This is a precisely bounded interface test.

## 3. Dynamic GRAIN coordinates can actually affect the model

For each author, difference the existing three period-level GRAIN scores and
the same frozen query function f(P,Q). Resample or permute whole authors, never
pretend the two transitions are independent participants.

| adjustment | authors | r | lower | upper | p | q |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 63 | -0.3356 | -0.4838 | -0.1698 | 0.0004 | 0.0432 |
| context_adjusted | 63 | -0.3352 | -0.4832 | -0.1659 | 0.0004 | 0.0432 |

The whole:F02/T-F link survives the primary 108-cell BH family at the reported
Monte Carlo resolution. Context adjustment includes text length/format,
community dispersion/change, span and median temporal gap. Intermediate
function-part tests are secondary. Primary permutation count was increased from
499 to 4,999 for every primary cell because .002 was too coarse a p-value floor;
authors and tested directions were not changed. This is exploratory localization,
not a preregistered isolated test.

Then fit a source-only displacement direction from changes in whole:F02 to
changes in P/Q, without questionnaire labels. Fix alpha=100 and step=0.25 before
the new native run. Patch only query-region states and the disjoint final state;
keep the original text and model weights fixed. Actual native results:

| Quantity | Value |
| --- | ---: |
| Authors / new calls | 12 / 60 |
| Signed half-plus-minus effect on conditional p(F) | -0.006419 |
| Authors moving in the same negative direction | 12/12 |
| Mean absolute real-direction effect | 0.006419 |
| Mean absolute norm-matched shuffled-direction effect | 0.001318 |
| Real/null absolute-effect ratio | 4.871 |
| Paired absolute-effect excess | 0.005101 |
| 95% author-bootstrap interval for excess | [0.004467, 0.005867] |
| Maximum cached-state replay discrepancy | 0.0e+00 |

The alphabet is T=0, F=1. The negative half-contrast means the full +step versus
-step contrast is about -1.28 percentage points in conditional p(F), not a
personality-accuracy improvement. One shuffled direction is a matched negative
control, not a full random-direction significance distribution. This establishes
a usable model-sensitive displacement; it does not assert that this vector edit
is a natural human personality intervention or a unique attention circuit.

## 4. Model-readable and real-criterion-linked information partly separate

Within-comment whole:F02 dispersion has real T/F r=0.375,
95% CI [0.140, 0.570], n=63. Its change-to-query
co-change is only r=-0.021. By contrast, the period-level F02 change
has the intervention-backed model relation above. These are two statistics of
the same coordinate, not interchangeable measures.

Leading exploratory within-comment criterion links:

| query | feature | n | pearson_r | r_lower | r_upper | p | q |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TF_cont | disp::whole:F02 | 63 | 0.3751 | 0.1404 | 0.5699 | 0.0025 | 0.3969 |
| extraversion | drift::whole:F04 | 213 | 0.1839 | 0.0507 | 0.3106 | 0.0071 | 0.5778 |
| conscientiousness | disp::whole:F01 | 213 | 0.1667 | 0.0330 | 0.2945 | 0.0149 | 0.7360 |
| TF_cont | roughness::whole:F03 | 63 | -0.2935 | -0.5046 | -0.0493 | 0.0196 | 0.7360 |
| openness | roughness::whole:F02 | 213 | 0.1532 | 0.0191 | 0.2818 | 0.0254 | 0.7360 |
| neuroticism | roughness::whole:F01 | 213 | -0.1513 | -0.2800 | -0.0172 | 0.0273 | 0.7360 |

None survives the full 162-cell criterion family, so retain them as localized
candidates rather than naming factors after them. In particular, the r=.375
dispersion link is not yet a validated T/F subscale. Across the twelve period
coordinates, ranks of path-link strength versus real-criterion strength vary by
query (rho range -0.343 to +0.441). There is no general monotonic relationship.
The within-comment scan likewise yields no BH-supported all-query motion link.

The same F04 drift has different meanings at different clocks: sampled-comment
drift/extraversion r=-.162, within-comment drift/extraversion r=+.184. Their signs
cannot be pooled as one universal "motion trait". Most dynamic descriptors have
low cross-period correlations, with median values near zero; that is a property
of these sampled long-span materials, not a test of same-session repeatability.

## 5. Prediction comparisons answer the adjustment question, not the mechanism

All arms use identical author folds within branch. Ridge alpha is selected by
three inner folds, with train-only scaling, inside five outer folds. The base
is GRAIN12 (sampled-comment) or GRAIN12 plus four slice means (within-comment).
Extra blocks are query contributions, observed context, and dynamic+dispersion.
The latter is compared with dispersion alone and one fixed point-order shuffle.
No human criterion is read at inference time.

### sampled-comment branch: author-held Pearson r

| query | baseline | +query | +dispersion | +dynamic | +shuffled | +all |
| --- | --- | --- | --- | --- | --- | --- |
| EI_cont | -0.2189 | -0.1940 | -0.2018 | -0.2338 | -0.3527 | -0.2291 |
| JP_cont | -0.3248 | -0.4311 | -0.3071 | -0.4523 | 0.0847 | -0.4556 |
| SN_cont | -0.2339 | -0.1362 | -0.2730 | -0.0454 | 0.0559 | -0.0505 |
| TF_cont | -0.2178 | -0.0979 | -0.0763 | -0.1178 | -0.0011 | 0.0252 |
| agreeableness | 0.0653 | 0.1032 | -0.0554 | 0.0833 | 0.0342 | 0.1685 |
| conscientiousness | 0.0285 | 0.0572 | 0.0268 | 0.0177 | 0.0687 | 0.0276 |
| extraversion | 0.1127 | 0.2380 | -0.0513 | -0.0316 | -0.1214 | 0.0892 |
| neuroticism | -0.1122 | -0.1175 | -0.0871 | -0.0354 | -0.0526 | -0.0129 |
| openness | -0.0077 | 0.1356 | -0.1065 | -0.0789 | 0.0471 | 0.0552 |
### within-comment branch: author-held Pearson r

| query | baseline | +query | +dispersion | +dynamic | +shuffled | +all |
| --- | --- | --- | --- | --- | --- | --- |
| EI_cont | -0.1878 | -0.1630 | -0.1461 | -0.1176 | -0.1127 | -0.0719 |
| JP_cont | -0.2931 | -0.4141 | -0.1752 | -0.2083 | -0.1074 | -0.2422 |
| SN_cont | -0.2300 | -0.1346 | -0.2922 | -0.2886 | -0.2504 | -0.2882 |
| TF_cont | -0.1681 | -0.0436 | -0.0251 | -0.2033 | -0.0236 | -0.0947 |
| agreeableness | 0.0575 | 0.1043 | 0.0208 | 0.0347 | 0.0524 | 0.1313 |
| conscientiousness | 0.0661 | 0.0652 | 0.1006 | 0.0427 | 0.0263 | 0.0656 |
| extraversion | 0.1138 | 0.2022 | 0.0227 | -0.0245 | -0.0348 | 0.0853 |
| neuroticism | -0.1369 | -0.1391 | -0.1539 | -0.1197 | -0.1274 | -0.0828 |
| openness | -0.0008 | 0.1316 | -0.0382 | -0.0698 | -0.0858 | 0.0474 |

The pure query addition preserves a useful E route in the sampled-comment
branch (.113 -> .238) and an O route (-.008 -> .136). Adding all dynamics is not
a general improvement: E falls to .089 with all additions, versus .238 with
query alone. Within-comment order additions do not beat the order-shuffled
comparator with a positive paired-r interval on any of nine outcomes. Small
per-feature criterion associations do not guarantee multivariate prediction gain.

These are small mechanism cohorts, not the prior 8,678-user MBTI baseline.
MBTI minority counts are EI=12, SN=4, TF=26, JP=27. Pooled CV r/AUC can be
distorted by fold-specific intercepts; the saved metric files include centered
within-fold r, valid-fold AUC and training-mean MSE. For example, sampled-comment
SN baseline pooled r=-.234 coexists with within-fold r=-.028 and AUC=.487.
Do not reinterpret a negative pooled r as a reversed personality mechanism.
Two fixed label permutations per baseline/full arm are sanity checks, not a
calibrated significance test. No result is rescued by flipping predicted signs.

## 6. Exact predictive-gap accounting, and the missing behavior channel

For feature-block subset S, let M(S) be held-author MSE. Average each block's
marginal improvement over all six block-addition orders. The resulting three
Shapley allocations sum to M(empty)-M(all), with maximum accounting discrepancy
4.34e-19. Negative terms indicate worse prediction; they are not deleted.

| query | context | dynamic_and_dispersion | query_path |
| --- | --- | --- | --- |
| EI_cont | 0.0021 | -0.0003 | 0.0018 |
| JP_cont | -0.0060 | -0.0071 | -0.0088 |
| SN_cont | -0.0002 | 0.0011 | 0.0001 |
| TF_cont | 0.0056 | -0.0088 | -0.0027 |
| agreeableness | 0.0003 | 0.0013 | 0.0012 |
| conscientiousness | -0.0012 | -0.0003 | 0.0003 |
| extraversion | -0.0014 | -0.0019 | 0.0030 |
| neuroticism | 0.0004 | 0.0004 | -0.0003 |
| openness | 0.0003 | -0.0011 | 0.0010 |

Units are squared criterion units; compare within a criterion, not across
Big5 and binary MBTI. This is allocation of held-out predictive gain, not
percentages of true personality, causal variance or label noise.

MEPS numeric channel inventory (no new model fitting or questionnaire tuning):

| view_kind | condition | participants | user_turns_total | assistant_turns_total | median_user_chars | role_order_preserved |
| --- | --- | --- | --- | --- | --- | --- |
| direct_answer | meps_q1 | 46 | 46 | 0 | 227.5000 | 46 |
| direct_answer | meps_q2 | 46 | 46 | 0 | 206.5000 | 46 |
| direct_answer | meps_q3 | 46 | 46 | 0 | 204.0000 | 46 |
| free_chat | free_chat | 46 | 419 | 457 | 469.0000 | 46 |
| split_chat | meps_q1 | 27 | 131 | 125 | 182.0000 | 27 |
| split_chat | meps_q2 | 27 | 80 | 79 | 171.0000 | 27 |
| split_chat | meps_q3 | 26 | 86 | 85 | 156.0000 | 26 |

There are user/assistant utterances, participation counts, task answers and
already-authorized pre/post self-reports. Participation is observed conversational
behavior. It is not an independent everyday-action criterion. For example:

| feature | scale | n | pearson_r | p | q |
| --- | --- | --- | --- | --- | --- |
| user_turns | post_disclosability_lv1 | 46 | 0.3092 | 0.0365 | 0.9990 |
| assistant_turns | post_disclosability_lv1 | 46 | 0.3078 | 0.0374 | 0.9990 |
| user_chars | pre_leak_concern | 46 | -0.3060 | 0.0386 | 0.9990 |

These 46-person correlations are exploratory and none of 418 scanned
participation/criterion links survives BH. Assistant counts also reflect the AI,
and missing assistance records are not automatically coded as zero use when
opportunity has not been established. No raw survey, duplicate coding archive,
Study 1 join or analysis-master table was opened.

The reason the broader "saying versus doing" gap cannot be numerically isolated
is identification, not insufficient model complexity. With only observed text
T and questionnaire Y, a world with unobserved action B=Y and a world with
B independent of both produce the same observed (T,Y) distribution. They imply
different text-action relationships. More elaborate functions of (T,Y) do not
select between them. Big5/MBTI self-report must not be renamed real action.

## 7. Updated mathematical interface and next useful experiment

Keep four distinct objects for each factor: level, dispersion, ordered shape,
and model sensitivity. Let H denote the concatenated P/Q state and G the frozen
factor coordinates. Source fitting provides Delta H = D Delta G + R. For a
frozen affine reader f(H)=b+g^T H, the following identity is exact:

    Delta f = g^T D Delta G + g^T R.

For the actual nonlinear native reader F, the exact differentiable path identity
is Delta F = integral_0^1 grad F(H+t Delta H)^T Delta H dt. Our finite native
patch tests a local action along D, not the entire integral or a natural text edit.
Separately, D_time(G+c_author)=D_time(G), so dynamic matching alone cannot recover
reference-relative author offsets. Preserve the static anchor.

This round's decision is **a factor-specific, statistic-specific interface**, not
universal concatenation. The next sharply defined mechanism question is why F02
level/change is model-readable while its within-text dispersion is more linked
to the recorded T/F criterion in this sample. Compare level-aligned and
dispersion-aligned state directions on the same material, retaining the current
whole:F02 direction as a positive model control. Do not restart a broad accuracy
grid or discard GRAIN coordinates simply because a query does not read them.
An action-gap experiment requires separately declared same-author action outcomes
and opportunity/timing, possibly from independently coded existing MEPS answers;
new human recruitment is not automatically required, but outcome definition is.

## 8. Reproduction, literature and scope

Run from the independent GRAIN repository with the parent project's environment:

```bash
"../project persona/.venv/bin/python" scripts/run_grain_dynamic_path_channels.py
"../project persona/.venv/bin/python" scripts/run_grain_within_comment_channels.py
"../project persona/.venv/bin/python" scripts/run_grain_dynamic_direction_intervention.py
"../project persona/.venv/bin/python" scripts/report_grain_dynamic_channels.py
"../project persona/.venv/bin/python" -m pytest tests/test_grain_dynamic_channels.py tests/test_grain_factor_path_routes.py tests/test_grain_functional_profiles.py tests/test_grain_full_state_edit.py -q
```

Default outputs are immutable on fresh preparation: use new output locations for
CPU reruns, or the explicit analysis/resume options to recompute analyses of
existing caches. The one native script intentionally refuses an existing output
directory; preserve/relocate a prior private packet before an intentional rerun.
Do not repeat a completed native run merely to generate this report.

Public aggregate report: this Markdown source and its offline HTML rendering.
Private numerical artifacts: `results/grain_vector_mechanisms/dynamic_path_channels_v1/`
with `within_comment/` and `native_direction/` children. Individual predictions,
raw material references and model arrays remain gitignored. Exact result hashes
and current code snapshots are in `delivery_manifest.json` and `code_snapshot.zip`.
Tables are used instead of charts because this deliverable primarily requires
exact nine-outcome lookup and a distinction between incompatible units/clocks.

The method choices are aligned with, rather than inferred as established by:

- [Fleeson (2001)](https://pubmed.ncbi.nlm.nih.gov/11414368/): trait-related
  behavior distributions can have distinct location and variability properties.
- [Campbell and Fiske (1959)](https://doi.org/10.1037/h0046016): different methods
  and different constructs must be distinguished when interpreting covariance.
- [Jain and Wallace (2019)](https://aclanthology.org/N19-1357/): a displayed
  attention pattern is not itself evidence of explanatory importance; here the
  native follow-up tests an actual state intervention instead.

This completes the five bounded computational questions. It does not assert
that all latent behavioral mechanisms are identified, or that every historical
GRAIN dynamic-factor family was tested. Released scoring, B0/B4, existing
lockboxes and parked long-running queues are unchanged.
