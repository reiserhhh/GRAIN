# NEXT64: matched context-modulated scalar token program

2026-09-11. Planned, not executed. NEXT63 measured a genuine same-BODY read-in
interaction. This step develops ONE conditional family and a matched additive
control, then returns to complete held-author scoring and GRAIN actions.

## Fixed source and comparison

Source4, both original and NEXT63 last-replacement states/gradients/scores. Two
queries separately: neuroticism and confirmation_seeking. All4 sources construct
each query program. Existing exposed new8 is application only. No humanY, MEPS,
new material, neural/API calls, alpha or feature-count search.

IMPORTANT: the shifted source targets use other-source mean last states including
every member of source4. Do not reuse them in a fake leave-one-author-out split.
No new source4 held-out score table this round. Source intervention predictions
are in-fit diagnostics, not independent replication. Current new8 targets remain
denied until all predictions freeze.

Use three displayed models, only two newly fitted per query (4fits total):
1. Frozen NEXT62 all4 Hermite program, unchanged historical reference.
2. Additive Hermite program refitted to BOTH original/shifted source views.
3. Context-product extension trained on the IDENTICAL augmented source views.

This separates the benefit of additional source intervention targets from the
benefit of a conditional input/function. Original per-query source score/density
scales and alpha=.01 remain fixed. Each source author weight1/4; each of its two
views weight1/2. All fitting targets are model L and BODY gradient, not questionnaires.

## One explicit conditional scalar

Keep NEXT62 all4 BODY U8, centering/SD,64 token landmarks, position coordinates,
Gaussian bandwidth, original scalar/Hermite feature layout. No refitted BODY basis.
Let phi_m(v)=[k_m(v),psi_m1(v),...,psi_m8(v)]. Landmarks remain16 per source.

Build a SEPARATE source-only context coordinate from the four ORIGINAL last states:
center at their mean; use centered right singular vectors, up to3 nonzero axes
(relative cutoff1e-10, deterministic signs); scale each retained coordinate by
its source population SD, using1 only for SD<1e-8. This is at most3-dimensional
because four centered states have rank<=3, not a psychological dimensionality claim.
Apply this fixed projection/scaling to both shifted and new8 last states.

Context centers c_m are the original last coordinates of the author supplying
token landmark m. Context bandwidth is the median strictly positive pairwise
distance among the four ORIGINAL source context coordinates. No native targets
choose a context axis, bandwidth, center or truncation. Rank0/undefined bandwidth
is a documented stop, not a selected replacement rule.

    kappa_m(c)=exp(-||c-c_m||^2/(2 sigma_c^2))
    gamma_m(c)=kappa_m(c)-kappa_m(0)
    Lhat=L_additive(theta)+mean_t sum_m gamma_m(c) e_m^T phi_m(v_t).

Zero e exactly recovers the additive family; at source-reference c=0 the added
BODY scalar/gradient contribution vanishes. The mixed derivative need not vanish.
Candidate has1169 coefficients versus593 control, explicitly not matched capacity.
The last gradient includes gamma's analytic derivative and the original last term;
the BODY gradient differentiates phi, with c held fixed. GRAIN D remains BODY-only.
Both scores and responses must come from this one scalar. No arbitrary extra
vector head, current native gradient or target-dependent gain.

Stream the source normal equations in small token blocks; no full hidden-width
by-center arrays or giant stacked SVD. The question is useful context dependence,
not another unregularized source capacity table. 120sCPU cap across4fits, application,
one replay and audit; estimate after the first query's two fits with30s reserve.
Preserve bounded outputs if overcap. No automatic smaller candidate rescue.

## Application and theory decisions

Report every new8 query/factor for all three programs: fullL r/RMSE/bias/SD and
SSE/sourceconstant; fullBODY field error; full and allocation GRAIN actions with
r/RMS/signs/SSEzero; paired query contrasts. Original new8 is still exposed
development material, not a new blind human validation cohort.

Source report: original/shifted fullL and gradient loss, predicted native finite
delta_tau, additive zero and nonzero conditional response. This shows what the
function can learn on supplied interventions; only new8 tests source-disjoint use.
Report raw new8 last displacement retained/outside source context span without
using it to exclude examples. Preserve the existing BODY-complement cancellation
warning; an improved GRAIN projection is not automatic full-field reconstruction.

Complete-function promotion remains: BOTH queries beat the MATCHED augmented
additive control's fullactionerror, have fullactionSSEzero<1 and scoreSSE/source
constant<1. Useful local/context effects survive even if fullpromotion fails.
Do not choose a factor after results and claim the whole model succeeded.

If source interaction is learned but new8 fails, close this candidate and inspect
which original-context/fullscore contribution remains omitted, not resweep alpha
or dilute the objective into source fidelity. No additional donor/dose experiments.

## Tests and outputs

Finite difference BODY/last and their mixed derivative; symmetric Hessian/line
integral; zero-extra nesting; frozen descriptor checks; source-only context PCA;
equal-author/equal-view weights; forbidden-current targets; exact prediction and
metric replay. Freeze primary code first. New context-modulated token module/
runner/tests, private `logic_context_modulated_potential_v1/`, public
`CONTEXT_MODULATED_POTENTIAL_REPORT_V1.md`, theory/atlas/goal/parent sync, local commit.
The source4 original+shifted materials may teach the extractor; MEPS/lockbox/release
are untouched. H16 remains the explicit neural-encoder input dependency.
