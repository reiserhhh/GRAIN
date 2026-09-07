# S9: connect internal configuration to the source teacher remainder

2026-09-08. Preparation can proceed while S8 runs; do not run two neural collectors
simultaneously on the Mac. This is the next direct mainline link, not an extra
embedding benchmark.

## Authorized material

All213 N source authors used by S4, plus64 T/F source authors chosen by a fixed
hash, as a bounded positive-control sample. Same three cue-mask views of12
comments. No bridge, MEPS, new questionnaires, lockboxes, threshold calibration,
or sample selection by effect/outcome. Existing S4 all-data mean/PCA student's
OOF predictions remain fixed. Preserve the full2328-author TF S4 result separately.

Inspecting the source revealed an expected observer difference: D1 used pinned
Qwen2.5-1.5B CUDA/bfloat16 with padding3072, whereas S1-S8 use the same checkpoint
on MPS/float32 without padding. Consequently S9 recollects both the unchanged
score and the all-comment mean-collapsed score under the same new observer.
Exact prompt-token hashes must match D1 before any forward pass.

## Three-part accounting

For each author, equal-average the same three views. Let S_old be the D1 CUDA
teacher, S_new the new FP32 origin, M the FP32 all-comment mean-projection score,
and f the frozen S4 student. Then

S_old - f = (S_old - S_new) + (S_new - M) + (M - f).

After all planned predictions are saved and hashed, join the already authorized
source Y and report Cov(Y,.) for each term, with paired-author bootstrap CIs.
This separates observer-condition difference, configuration response, and the
remaining mean-readout/student difference. It does not label a term as pure
noise or independent personality variance. S_new-M is a function intervention,
not a human-trait intervention. It includes all token configuration, not just
S7's estimated nonlinear permutation component.

Clarification before outcome inspection: the projected endpoint retains query
states already contextualized in earlier layers. Thus M-f is the **projected
suffix** remainder, not pure mean-embedding error. The CSV name `mean_remainder`
refers to the intervention endpoint, not information-theoretic sufficiency.

## Execution and resource decisions

Prepare and verify277 authors/831 views without labels or neural inference.
Maximum1662 new forwards, no gradients/training/API. Sequential actor; total
one-hour compute cap, resumable complete-author files. First run48 authors as
a resource checkpoint; do not inspect Y there. Use origin compatibility, actual
effect sizes versus numeric precision, liveness and estimated remaining cost to
decide whether completing the planned sample is scientifically worthwhile.
No outcome-based early-stopping claim. Evaluator refuses an incomplete cohort.

Outputs include frozen membership/material, prompt/input/cache hashes, old
observer descriptors, per-view scores, exact author-level components, aggregate
metrics and covariance intervals. Private materials and predictions remain ignored.

## Mathematical basis

This applies the exact linearity of covariance and the conditional-projection
accounting in GRAIN theory V2/V3. The configuration intervention is the projection
and path-integral object established by S6, informed by
[Sundararajan et al. (ICML2017)](https://proceedings.mlr.press/v70/sundararajan17a.html).
The task is to locate which measurable function carries the source association;
the algebra does not assume that every correlated component is psychologically causal.
