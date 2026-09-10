# NEXT57: source-only scalar function and GRAIN-response extraction

2026-09-10. Planned, not executed. NEXT56 closes the hidden-state substitution
family: offset changes the reader, removing it preserves contrast but not score.
No further bias/gain/rank/layer rescue on this panel.

## Objective and distinction from earlier packets

Learn one explicit H16->original fixed logit L function; derive its GRAIN actions
from the same function. Current application runs no suffix, reads no current
teacher outputs/gradients and adds back no native remainder. H16 remains neural.
NEXT05/08 matched E5 configuration residuals, NEXT32/33 conditioned B/pairs,
NEXT53 hidden-state values/derivatives, NEXT55/56 retained an actual suffix.
This targets original whole L and its H16 derivative without that suffix.

## Frozen function and source construction

Same12 authors/two queries/four directions; source4 leave-one-out, new8 all4.
Use the existing pointwise source mean mu and PCA32 U from NEXT53; no new PCA,
rank or feature grid. Existing BODY B and last-token address define64 features:

    phi(X)=concat(mean_{t in B}[(X_t-mu)U], (X_last-mu)U)
    Dphi[D_f]=concat(mean_{t in B}[D_ft U], D_f,last U).

The last-token derivative is zero under current BODY perturbations. It can aid
score levels but does not independently teach query-state dynamics. BODY means
are pooled contextual states, not word counts. Log source directional ranks/norms
as diagnostics, without selecting new features from new8 outcomes.

For each query/source reference, fit feature center/SD on source authors only;
SD<1e-8 becomes1. Keep columns. Set source target ellbar=mean L_s and
sL=max(std L_s,1e-6). Intercept is unpenalized:

    lhat(X)=a+w^T z(X)
    Lhat(X)=ellbar+sL*lhat(X)
    tauhat_f=sL*w^T Dz(X)[D_f].

Targets are original source L/tau from NEXT51 native captures, not NEXT55/56
failed candidates. Held-source/new8 targets are evaluation-only. Human Y is absent.
Two candidates with identical features, normalizations, weights and penalty:

    value: mean_s (lhat_s-(L_s-ellbar)/sL)^2 + alpha||w||^2
    joint: value loss + mean_{s,f}(w^T Dz_sf-tau_sf/sL)^2.

alpha=.01, derivative weight1, averaging over people and four directions explicit.
Stable augmented least squares/Ridge:5 references x2 queries x2 objectives=20
solves. No held-data calibration. Methodological precedent is derivative-informed
function approximation: [Czarnecki et al., 2017](https://arxiv.org/abs/1706.04859).

## Budget and decision

- 180s total CPU,20 solves,0 new neural/API/Y. Freeze script/tests/plan before
  construction. Source teacher reads restricted by membership; prediction reads
  only H16/GRAIN/source parameters. One cached replay, no neural restart.
- 48 scores/192 actions plus paired query contrasts. Report all authors/queries/
  factors, source4 and new8 separately;32 action cells are not32 people.
- L r/mean bias/SD/RMSE/SSE-source-constant; tau r/RMS/signs/SSE-zero. Preserve
  NEXT56 I/V contrast as context, not a new chosen teacher target.
- Compare joint to matched value-only with signed score/derivative error changes,
  source design ranks and same-function derivative checks. Keep negative results.
- Same utility rules: both new8 queries require r>0 and respective SSE ratio<1;
  separate score-only and response-only. Passing removes the suffix for that
  function, not H16. Retain associations below the gate at their observed scope.

If this fixed family fails, close it and use explicit design/value/response
residuals to select an input-mechanism change or a retained independent route.
Do not open successive alpha/pooling sweeps on these12 authors. The next decision
concerns the text-to-function interface, not filling a larger metric table.

## Deliverables and checks

`scripts/run_grain_complete_function_extraction.py`, reusable scalar fit/apply
module, tests, cache auditor, aggregate report/theory and dependency updates.
Private `results/grain_vector_mechanisms/logic_complete_function_extraction_v1/`.
Test affine finite differences, zero last-token derivative, excluded-source
scalers/target origins, matched normalization, signed error accounts/query joins.
No release/scorer/lockbox/MEPS/clinical changes or remote push.
