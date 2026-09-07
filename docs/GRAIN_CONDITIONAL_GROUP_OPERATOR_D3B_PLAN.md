# D3B: conditional group response operator

2026-09-08. Next development packet after the complete D2-D4 discovery results.

## Why this is the next mainline step

D2 measures vector-mediated real-text effects. D4 rank8 predicts new-author
intervention responses (TF r=.5314; neuroticism .5584). D3 shows group connections
in particular response coordinates but not one uniform response-profile geometry.
Test whether independent GRAIN coordinates parameterize the operator's gain.

## Finite design

1. Treat all64 D2 authors as development material now that the first held results
   have been inspected. Preserve the original split/results as historical evidence.
2. Fit, within each trait, a shared rank8 response basis and a conditional operator
   `M_hat=[b + W*phi(z)]^T U^T(alpha*d)`. phi is a training-only4-coordinate reduction
   of the frozen GRAIN whitened profile. Compare shared-only and conditional models.
3. Choose ridge from {1,10,100} inside3-fold author-grouped development CV. All
   preprocessing, basis fitting and model selection stay inside those inner folds.
   Do not use D2's old held designation as untouched confirmation for this revision.
4. Prepare20 new TF and12 new neuroticism source authors, excluding D1, all D2
   authors and reference members. Use the same fixed selection/questions/site,
   two views, and nine conditions. Treat ALL these authors as final response holdout.
5. Budget580 new forwards, maximum15 minutes initially; assess cost after the first
   complete author. No new labels, new model download or long-queue restart.
6. Compare held-author response r and MSE for the shared and GRAIN-conditioned
   operators; retain all conditions. If group conditioning helps, formulate the
   varying-gain mechanism. If the shared operator is the useful model, formulate
   the common operator with individual differences carried in its input trajectory.

## Deliverables / finite endpoint

Saved shared and conditional models, development-CV table, new-author predictions,
response metrics, and an integrated D1-D4 theory record. This is a mechanism
development decision, not a requirement to eliminate all remaining error.

Status: COMPLETE, 2026-09-08.32 fresh authors,64 views,580 forwards,649.63seconds.
Shared response r=.391710/.369447; MSE/zero=.869855/.498677 (TF/N).
Conditioned response r=.364596/.399893; MSE/zero=.893827/.573477.
Decision: retain the shared per-query operator; group gain does not improve MSE.
All conditions, paired author intervals and reproduction commands are in
`../reports/grain_vector_mechanisms/D3B_MAINLINE_CLOSURE_REPORT.md`.
The finite mainline goal is closed; no new private data use or external resource.
