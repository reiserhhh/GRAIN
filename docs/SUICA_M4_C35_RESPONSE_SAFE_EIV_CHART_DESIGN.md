# GRAIN M4-C.3.5 Response-Safe Cross-View EIV Chart Design

## Status

`DESIGN_CANDIDATE_NOT_FROZEN`

M4-C.3.4 isolated a truth-open condition-chart contribution but did not
produce an observable chart repair. This document specifies the only next
route currently licensed by that attribution result. It is not a
preregistration, source freeze, or empirical result.

## Question

Can the chart benefit localized by M4-C.3.4 be recovered from replicated
pre-response observations alone, without response, author, mechanism, or
synthetic-truth access?

The existing generator provides the required tensor:

\[
X_{s,u,i}
\in
\mathbb R^p,
\]

with source \(s\), reference author \(u\), shared condition \(i\), and
pre-response feature coordinate. Conditions are aligned across sources and
authors, while their observation noise is independently generated.

## Proposed object

Split reference authors into two fixed halves \(A,B\). Let
\(\bar X_s^A,\bar X_s^B\) be condition-indexed source prototypes. Define the
cross-half signal covariance:

\[
\widehat S_s
=
\frac12\left[
\operatorname{Cov}(\bar X_s^A,\bar X_s^B)
+
\operatorname{Cov}(\bar X_s^B,\bar X_s^A)
\right],
\]

and the cross-source moment:

\[
\widehat C_{12}
=
\frac14
\sum_{h,h'\in\{A,B\}}
\operatorname{Cov}(\bar X_1^h,\bar X_2^{h'}).
\]

After retaining only the positive, response-free support of
\(\widehat S_1,\widehat S_2\), compute:

\[
\widehat S_1^{-1/2}
\widehat C_{12}
\widehat S_2^{-1/2}
=
U_r\Lambda_rV_r^\top.
\]

The response-safe quotient chart is:

\[
\widehat\phi_{\mathrm{RS}}(i)
=
\frac12
\left[
U_r^\top\widehat S_1^{-1/2}\bar X_{1i}
+
V_r^\top\widehat S_2^{-1/2}\bar X_{2i}
\right]
\pmod{O(r)}.
\]

It identifies the gauge class of a cross-source, cross-author reproducible
condition subspace. It does not identify named coordinates or a latent
condition truth.

Rank \(r\) may use only:

1. singular values above a frozen condition-permutation 99th percentile;
2. author-split principal-angle stability; and
3. a fixed numerical rank floor.

No response or downstream relation endpoint may select rank.

## Primary estimand

The C3.4 gate-zero creation target, response/choice edges, oracle denominator,
worlds, authors, and `K=8 excitation` endpoint remain fixed:

\[
q_{\mathrm{RS}}
=
\frac{
\Gamma^0(L[\widehat\phi_{\mathrm{RS}}])
-
\Gamma^0(L[C_0])
}{
\Gamma^0(L[C^\star])
-
\Gamma^0(L[C_0])
}.
\]

Set `S=0` and `P=0`. The experiment tests chart identification only.

## Allowed and forbidden information

Allowed:

- reference calibration pre-response tensors;
- fixed author-half membership;
- condition alignment across the two declared numeric sources;
- reference selection pre-response tensors for frozen stability checks; and
- mechanism pre-response tensors after every chart choice is frozen.

Forbidden:

- author identity as a feature;
- response, choice, future state, future menu, or generated-event indicators;
- mechanism or world labels during fitting;
- synthetic truth, oracle basis, or oracle chart before final evaluation;
- Big Five, MBTI, or any external label;
- relation geometry, creation loss, or headroom during rank or model choice.

## Fixed comparators and attacks

The only model comparators are:

1. the current discovered chart `C0`;
2. same-rank source-1 PCA fitted on the same calibration prototypes; and
3. the response-safe EIV chart.

The oracle chart is an unopened ceiling, not a candidate. Required attacks:

- condition-index shuffle in one source before cross-view fitting;
- source-pair shuffle;
- reference-author-half resampling;
- response-field permutation, which must leave the chart unchanged;
- orthogonal gauge and common-shift checks;
- no-creation worlds;
- author and response leakage worlds;
- support-shift and topology-mismatch worlds; and
- exact condition aliases.

Exact observational aliases must remain quotient-identical. A chart may not
claim to recover distinctions absent from the allowed sigma-field; downstream
evaluation must return `MECHANISM_UNDERRESOLVED` rather than inventing latent
coordinates.

## Candidate gates

Before source freezing, the following thresholds require an independent
implementation audit:

- \(q_{\mathrm{RS}}\ge .60\), clustered LCB above `.50`, and positive
  improvement in at least `6/8` repetitions;
- gain over same-rank single-view PCA at least `.10`, clustered LCB above
  zero;
- oracle headroom at least `.05`, clustered LCB above zero;
- \(q_w\ge .35\) in every target world with oracle headroom at least `.05`;
- signed mean gain no worse than `-.01` in non-target main worlds;
- held author-split largest principal angle at most 15 degrees;
- retained singular values above the frozen permutation 99th percentile;
- source/condition shuffle gain at most `.01`;
- no-creation false-success rate at most `.05`;
- gauge error at most `1e-6`;
- common-shift pair-distance error at most `1e-10`; and
- zero false latent recovery in exact-alias worlds.

An alias false promotion takes precedence and yields
`M4_C35_NO_GO_ALIAS_UNSAFE`. Any other primary-gate failure yields
`M4_C35_NO_GO_RESPONSE_SAFE_CHART_IDENTIFICATION`.

## Difference from rejected chart routes

This is not another PCA/Isomap/landmark candidate tournament and not the
rejected shared isotropic RBF repair. It has no kernel bandwidth, Nyström
smoothing, or downstream endpoint search. Its sole new assumption is that
independent source and author replicates can identify a reliable
errors-in-variables common condition subspace.

## Boundary

Even a complete pass would support only an observable condition-chart
candidate in the registered finite synthetic worlds. It would not close
M4-C.2 without a separately frozen transport confirmation, authorize M4-D,
validate natural text, or identify personality.
