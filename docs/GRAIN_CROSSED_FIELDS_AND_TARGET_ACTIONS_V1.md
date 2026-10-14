# Crossed Fields and Target-Conditioned Actions

Canonical evidence: HM30 crossed_wording_context_r1 and its alternate verifier.
This develops GRAIN's where/what observation link; it is not a new psychological
factor discovery or a replacement for author-level measurement.

## Orthogonal Decomposition, Without Deleting Conditions

For a finite fully crossed, equally weighted context c and wording p bank, write

    g[c,p,q] = mean_g[q] + g_context[c,q] + g_wording[p,q] + g_interaction[c,p,q].

The context and wording components sum to zero, and the interaction sums to zero
over either variable. Let Jc and Jp be uniform averaging matrices. The four
projectors Jc tensor Jp, (Ic-Jc) tensor Jp, Jc tensor (Ip-Jp), and
(Ic-Jc) tensor (Ip-Jp) are orthogonal idempotents summing to identity.
Reconstruction and energy additivity follow by elementary product-space algebra.
The construction is standard commuting-projection FANOVA; see Ginsbourger et al.
2014, https://arxiv.org/html/1409.6008, Section2 equations1-4.
The chosen finite product measure is a design, not a natural-text independence
assumption. Retain all terms rather than automatically dropping a condition mean.

The same decomposition applies to lambda=O4^-T g with the declared moment metric.
Under m'=Rm, transport lambda'=R^-T lambda and dual metric G_inv'=R G_inv R^T.
An untransported Euclidean norm is not a frame-invariant magnitude comparison.

## Whole-Field Energy Is Not Target Action Importance

For a declared unit direction u[q,a], local physical action is <g,u>.
With W[q]=sum_a u[q,a] u[q,a]^T, an action-sensitive seminorm is

    ||g||^2_W = g^T W g = sum_a <g,u[q,a]>^2.

This is a positive-semidefinite task seminorm, not a full psychological metric.
Its kernel contains directions not tested by these actions, not proven noise.
For a direction fixed over cells, contraction commutes with the four projectors,
but the energies change under contraction: a small whole-field component can
dominate a particular action. Finite endpoint action additionally contains
path-dependent nonlinear terms; local derivatives are evaluation references.
HM30 retains actual float32 displacement, its norm, odd action and even change.

Empirically the application whole-gradient wording shares are88.38%,95.36%,76.12%
for negative affect, low mood and AI trust. Yet trust's unit-displacement mean
action has65.79% interaction and only22.59% wording share. This directly argues
against choosing observation components solely by overall vector variance.
There is no conversion of those figures to personality importance or variance.

## Source-Only Compilation and Its Remaining Dependency

Fit source fields with uniform source contexts:

    lambda_global[q] = O4^-T mean_source[c,p] g[c,p,q],
    lambda_wording[p,q] = O4^-T mean_source[c] g[c,p,q].

On a new context and SAME declared wording, compute

    predicted_A = <lambda_wording[p,q], O4 DeltaH_actual>.

The fifteen fields were frozen before all held observations. All12 held query/
arm comparisons improve RMSE over the global field, application10.57%-59.90%.
Application low-mood mean signs improve6/8 to8/8, but trust mean remains7/8.
This is positive conditional compilation, not a universal field or a new-wording
predictor. Raw post-layer16 text vectors still require the frozen observer.

Paired queries share cell addresses but use query-specific hidden directions;
their covariance is not a same-physical-direction causal overlap. Signed context,
wording and interaction covariance terms may cancel. Keep target units separate.

## Next Mechanism, Not Another Accuracy Sweep

Current g depends on both H (the property representation) and c (all unpatched
states, including question states). Crossed wording/context materials locate
variation, but cannot decide whether c acts beyond information already in H.
At identical four-token property H, compare g(H,c0,q) and g(H,c1,q) using actual
cross-recombined states. Measure finite actions with identical realized delta.
This separates context written into H from context outside H modulating its
interpretation. The latter requires explicitly conditional cotangent fields;
the former motivates mathematically transported lexical fields rather than lookup.

New plan: GRAIN_CONTEXT_FIBER_RECOMBINATION_PLAN_V1.md, not executed in HM30.
Synthetic reader hybrids are mechanism interventions, not new author reports.
Original natural-author, B_u, external-meaning and18/100/44 requirements remain.
