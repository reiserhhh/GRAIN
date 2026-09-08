# GRAIN: query actions and fixed-reference author configurations

2026-09-09. M25/M25B/M26 completed, following M23/M24/M24B.

## Main finding

The same frozen mathematical function reads both complete judgments and natural
evidence changes on three matched queries, then places authors against a fixed
reference. Configuration action within a query and shared action between queries
are different objects. The latter is better recorded as signed, condition-dependent
cross-path covariance than forced into one universal axis.

| Query | Native judgment r [author-bootstrap 95% CI] | Native edit-response r [CI] | Response MSE / zero response |
|---|---|---|---|
| T/F | .871 [.707, .942] | .796 [.686, .875] | .381 |
| Neuroticism | .821 [.666, .930] | .678 [.524, .782] | .640 |
| Confirmation seeking | .911 [.845, .957] | .827 [.697, .909] | .320 |

These are agreement with a fixed model's judgments, not human questionnaire
correlations. Earlier source-label links from M10/M22/M23 remain separate evidence.

## Design and actual cost

- Existing S1 material: 24 authors, two nonoverlapping views, three queries,
  full/E1/E2. Each view has identical edits across queries.
- Twelve M8 evaluation authors remain evaluation authors. The other twelve fit
  author norms and shared-query projections. No author crosses roles.
- Functional G/g matrices still come from the OLD two reference authors; fitting
  twelve-author norms does not refit these action matrices.
- 432 P/Q cells: 72 M8 cells reused, 360 new prefix-only calls in 266.31 s.
  Final-state vectors exactly replay S1. No suffix, gradients or generation.
- Per query: 24 full-state observations and 48 edits on twelve held authors.
  Bootstrap resamples authors, keeping their views and edits together.
- Zero exactly duplicated comment texts between views for every author. This is
  cross-text sampling, not temporal retest or independent new collection.
- No new human labels, questionnaires, lockboxes or external APIs were used.

## Reusable construction

For fixed query q and observer, P is the ordered question-token matrix, Q the
final token, and delta denotes displacement from the old functional anchor:

```
delta_P = 1 * delta_m + delta_C; sum_token(delta_C) = 0
f_q = b_q + <sum_token(G_q), delta_m>
          + <G_q - mean_token(G_q), delta_C> + g_q' delta_Q
    = b_q + M_q + C_q + Q_q
Delta f_q = <G_q, P_edit - P_full> + g_q'(Q_edit - Q_full)
author_position = (author_function - fixed_reference_mean) / fixed_reference_SD
```

This split is exact for the frozen affine function. Agreement with the nonlinear
native model is empirical. Unclipped values preserve amplitude errors.

### Shared function is conditional

Reference-only OLS predicts each standardized configuration response from the
other two queries, after reference edit/view centering. Held-author correlations
are -.217/-.147/-.066; residual energy exceeds total energy through cancellation.
Do not deploy this projection as a general shared personality score. Its exact
shared-plus-private interface is a declared linear decomposition; the negative
transport results are retained rather than selecting a favorable fit.

The useful paired-query relation is:

| Paired natural responses | Native r | Functional r | Configuration-only r |
|---|---:|---:|---:|
| T/F and N | .140 | .132 | .185 |
| T/F and confirmation | -.066 | -.063 | -.105 |
| N and confirmation | .374 [.089, .588] | .322 [.017, .547] | .066 |

M25B splits each centered native response into M, C, Q and a retained native
remainder U. For matched authors and material condition c:

```
Gamma_qr[a,b | c,R] = Cov_R(response_q,a, response_r,b | c)
Cov(native_q, native_r | c) = sum_a sum_b Gamma_qr[a,b | c,R]
```

This is a signed covariance identity, not independent variance allocation or a
newly proved theorem. Its empirical use localizes the functional terms carrying
the observed cross-query relation. Maximum accounting error: 4.34e-19.

For pooled N/confirmation, normalized by the two native response SDs:
- Q x Q contributes +.10835; C x C contributes only +.02396.
- All modeled M/C/Q cross-terms sum to +.24006 of native r=.37352.
- Remaining terms sum to +.13345, including U x U +.11461 and
  U_N x Q_confirmation -.09130.

These are native-correlation contributions, not percentages of causal variance.
A large within-query C action (M24) can coexist with small between-query C x C.
T/F-N native r is +.305/+.155/+.390/-.558 across view0-E1/view0-E2/view1-E1/view1-E2;
N-confirmation stays positive (.133/.385/.680/.401). Retain query- and condition-
specific functions rather than fitting one common direction.

### Author position and response are separate

Twelve reference AUTHOR averages define fixed norms. On twelve held authors:

| Profile | Dimensions | Same-author cross-view AUC | Author-permutation p |
|---|---:|---:|---:|
| Full query levels | 3 | .634 | .061 |
| Configuration edit responses | 6 | .467 | .678 |
| Concatenation | 9 | .514 | .409 |

Confirmation-query level has cross-view r=.658 [.012,.957]; T/F and N level r
are .077/.076. Keep the former as an author-position example in this panel.
Edit sensitivity is not interchangeable with enduring position. Concatenation
dilutes the useful profile, so it is inspectable but not promoted over levels.

### Compact use: suffix-free, full coefficients retained

SVD ranks 1/2/4/8/16/full were fixed before analysis. Reference-only selection
required <=.10 relative RMS error on full configuration and BOTH edit responses.
All queries select full-rank fallback: 36/62/62. Rank16 E2 errors are
.181/.185/.159; the criterion was not silently relaxed.
Here full means no truncation of the token matrix; the centered coefficient's
algebraic rank is at most T-1, not literally T independent directions.

The delivered bundle keeps 245,760 configuration coefficients plus mean/Q and
fixed references. Its NPZ is 3,106,840 bytes. Dot products replace suffix and
generation, not the frozen neural prefix. All 432 states/48 profiles replay to
3.89e-16; batch error is exactly0. This is not an LLM-free encoder or complete
conditional-field compression.

## Mainline decision and delivery

The five-piece goal has a finite executable construction: coordinate location,
actual intervention, same-material cross-query structure, fixed author position,
and reusable readout. Preserve the full functional matrices, condition-indexed
cross-path covariance, query-specific remainders, and separate position/response
channels. Do not resume score-only or nonlinear grids to erase negative arms.

Wider populations, query wording, observers, traits and more compact conditional
fields are next-stage developments. No automatic collector remains. This completes
the bounded construction, not all human personality theory. Old MCD/4W, source
associations and directional findings remain foundations; release/lockboxes unchanged.

Build/API: `../../docs/GRAIN_FUNCTIONAL_PROFILE_BUILD_V2.md`.
Private artifacts under ignored `results/grain_vector_mechanisms/`:
`shared_author_profiles_m25_v1`, `functional_author_construction_m25_m26_v1`,
`functional_covariance_m25b_v1`, `functional_profile_bundle_m26_v1`.

Sixteen new mathematical/API tests pass. Broader GRAIN run:790 passing before
the final two covariance tests were added; those two plus fourteen new and twelve
old CPU tests then passed separately (28). Ruff/diff checks pass. Artifact audit:
12/12 disjoint authors,432 unique cells, reference-center/SD replay errors
1.39e-17/3.12e-17. Existing unrelated edits remain untouched.
