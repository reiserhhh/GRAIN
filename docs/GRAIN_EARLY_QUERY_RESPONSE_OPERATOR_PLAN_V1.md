# HM34: Source-Only Early-Query Response Operator

Status: planned after verified HM33, not executed. No new neural calls are
required for the first stage. This is a mechanism compilation, not a prediction
hyperparameter search. Preserve every HM33/HM32/HM31/HM30 primary dependency.

## Why This Is the Next Decision

HM33 shows that P, the earlier query-suffix states, carries much of the
conditional property action, especially for the trust target. Text-side B
still acts, particularly on negative affect. Do not ignore B or fit a universal
H/P/Q model and call its remaining error noise.

The next question is narrower and constructive: can one source-learned
linear operator map actual P displacements to the common change in property
covectors on cold contexts? Keep target-specific operators and their units.
This tests a piece of the text-to-query bridge that can be computed without
the downstream decoder once H/P/Q states have been extracted.

## Fixed Data and Mathematical Law

Use all48 HM33 sixteen-corner cubes, three queries, four wordings and the
original source/validation/application roles. Do not re-acquire any corner.
For each query q and pair/wording c:

    d_c=(vec(P_0)-vec(P_1))/2,
    g_c=E_(H,B,Q)[(alpha_(H,B,P0,Q)-alpha_(H,B,P1,Q))/2].

g_c is precisely the P character coefficient. It is not a personality factor.
The pair's P token IDs and sequence length are identical. Across different
queries, P spaces may have different dimensions; do not concatenate them
without an explicitly defined map. Position/layout history remains part of
the encoded P states.

Source columns are the8 pair/wording contrasts, pairs0/1 times four wordings.
With D=[d_1,...,d_8], G=[g_1,...,g_8], use the minimum-norm linear extension

    L_q d=G (D^T D)^+ D^T d.

The numerical rank rule is fixed machine tolerance100 times epsilon times
the8x8 Gram matrix dimension times its largest eigenvalue. The alternative
primal SVD uses the square root of this same threshold, not another rank rule.
It is not selected by application accuracy. Keep the spectrum, truncation,
interpolation error and input metric. This clarifies the implementation before
freezing, without observing any held response or choosing a tolerance by error.
No ridge/alpha/rank/kernel/prompt/layer search and no query-specific rescue.
Use only source captures/gradients while fitting; save all three source models
and their hashes before loading held P coefficients or target actions.

All compatible linear operators differ by N(I-Pi_D). Report the cold
input's supported and unsupported components. The unsupported component
remains unidentified; it must not be defined as noise or as a zero true effect.
Check this source-equivalence identity with an explicit alternate implementation.

## Actual Outputs, Not Just Interpolation

1. Predict the P coefficient on every source, validation and application cube.
   Compare full gradient units, the two original task projections, and a zero
   response reference. Report per-query errors and signed magnitudes.
2. For every fixed H/B/Q address, predict the P-only half-difference by the
   same L_q d_c. Retain the observed HP/BP/PQ and all higher interaction
   terms as explicit discrepancies, never relabel them measurement noise.
3. For both original arms, pair with the exact realized H half-delta:

       predicted P-action contrast=<L_q d_c,delta_H>.

   Compare against actual [(A at P0)-(A at P1)]/2 at every fixed H/B/Q.
   These768 finite contrasts are not overall personality prediction accuracy.
   Decompose derivative-to-finite-action error separately from shared-operator
   error, including signed cancellation and the interaction with H-specific
   deltas. A P main-gradient fit does not automatically preserve raw-action
   P coefficients when the physical H deltas differ.
4. Reconstruct the pure P character by independent paired averaging, not
   only reading the existing Fourier table. Verify all targets and roles.
5. Keep B/P/Q action bounds from HM33 as context, not a threshold to tune.
   No source role or application coefficient may be hidden in a model input.

Use exact physical coordinates. A coordinate change must transport D, G,
the input metric and their pairing together; Euclidean pseudoinverse in a
fresh arbitrary frame is not intrinsically invariant.

## Tests and Acceptance

Before freeze, test known linear P response, non-linear/interaction counter-
examples, rank-deficient/source-unidentified directions, zero inputs, source
isolation, paired token addresses, alternate SVD/primal Gram equivalence,
and invariance under transported orthogonal frames. Preserve failures.

Freeze a new runner/helper/verifier/spec/tests/plan and fresh result directory.
At most three source operator fits,0 new model/API/private calls or neural
training. Independent-formula replay of every coefficient/prediction/contrast,
not test count or file existence, is finite acceptance. Keep all roles, arms,
queries and all source/held failures. Do not promote original18/100/44 scope.
Alternate Gram/primal-SVD replay has a fixed absolute tolerance1e-8, set before
freezing; report the actual maximum errors separately from this ceiling.

## Adaptive Continuation

If cold common P response is preserved, use it as a target-specific reading
operator and develop its interaction with H and the retained B contribution.
Do not declare all personality inference compiled.

If it fails, do not search a larger kernel. In a subsequent separately frozen
bounded packet, inspect the missing mixed response using

    alpha(P1)-alpha(P0)
      =integral_0^1 [d_P d_H F](P0+t(P1-P0))(P1-P0) dt,

at fixed H/B/Q and layout. Actual Hessian-vector observations would distinguish
nonlinear P response from absent source support. They are not yet authorized
by this zero-call stage; propose the concrete addresses/budget first. Existing
H-path integrals are not this P-path mixed-derivative experiment.

The graph-refinement manuscript's joint-signature lesson motivates preserving
state pairing. Its global-identification hardness theorem is not invoked.
The rational-formula manuscript motivates preserving the original observation
domain, not rationalizing softmax. Own finite source identities must be tested.

No API/private MEPS/Claude/automation/new neural training/commit/push/tag/
lockbox/old queue. The existing goal remains ACTIVE. Natural-author B_u and
psychological meaning are separate open parts of the original mainline.
