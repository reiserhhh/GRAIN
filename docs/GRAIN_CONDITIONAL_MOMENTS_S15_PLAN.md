# S15: comment mass versus conditional joint moments

2026-09-08. After S14. Discovery experiment; no questionnaire-trained reader.

## Question

S14's common-query reader retains C recovery. Can a comment-level covariance
formula approximate it, or does the useful relational operation require higher
within-comment moments? Separate that question from approximation of comment mass.

Use the same source authors/views, frozen S14 author-out common queries, S12
KV/PCA C readers and full token support. Four fixed arms, with no fitted weights:

1. exact: the full within-comment KV pairing at the common query;
2. uniform_moment: weight each group's Cov_G(V,K)q by |G|/n;
3. exact_mass_moment: weight the same covariance by exact attention mass P_G;
4. gaussian_mass_moment: use mass from log Z_G approximated by
   log|G|+E_G[q^TK]+Var_G(q^TK)/2, with the same conditional covariance.

Keep all outside-comment denominator terms exact, including query/template
tokens. The fourth arm approximates a finite log normalizer; no empirical
Gaussianity is presumed. No direction is designated psychological noise.

The equations follow conditional expectation and exponential-family cumulants
in [Wainwright & Jordan2008](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf),
as derived in `GRAIN_RELATIONAL_READOUT_THEORY_V4.md`, section8. No claim of
inventing the general moment expansion or that it is exact on finite token clouds.

## Bounded execution and decision

First24 source authors,72 prefix calls, no outcomes: replay exact S14 scores and
report covariance-arm fidelity/MSE to that exact reader. This is a resource/
mechanism checkpoint, not a significance stopping rule. If approximations show
severe scale or ordering failure, first analyze that discrepancy rather than
automatically expanding the table. If useful or the comparison remains genuinely
decision-relevant, complete the fixed213-author cohort (639 prefix calls total).
Maximum25minutes of collection; one neural process;0 backwards/LM-head calls.

For a full cohort only, freeze every arm's author scores and then join the
already-exposed source N outcomes. Report fidelity to exact common-query scores
and teacher C, MSE, source associations and paired deltas. No Y-based choice of
query, order, arm or tuning parameters. Pilot outcomes are not joined.

The frozen intercept remains in all arms. Fidelity MSE includes bias and scale
error; report normalized error relative to the exact reader's author variance.
Do not repair a bad approximation by post-hoc fitting on its evaluation authors.

## Outputs

`results/grain_vector_mechanisms/conditional_moments_s15_r1/`: immutable manifest,
source hashes, private per-author arm scores, progress, pilot diagnostics and
optional full analysis. Report `S15_CONDITIONAL_MOMENTS_REPORT.md` states whether
the24-author checkpoint or full213-author cohort was actually executed.
No new labels, protected datasets, LoRA, downloads, external APIs or push.

## Resource decision after the frozen24-author pilot

72 calls,67.17seconds, no questionnaire join. Exact reader replay passed.
Uniform-mass covariance r=.3683, MSE/exact-author-variance=902.60;
exact-mass covariance r=.7730, error ratio=.7419; Gaussian-mass covariance
r=.6540, error ratio=3.5132. The exact-mass arm has meaningful internal fidelity,
while the other arms expose a normalization mechanism. Complete the already
fixed213 cohort, all four arms unchanged, rather than select a new weight or
calibration on this pilot. This is a scientific resource-value decision, not
an interim significance claim or external-personality selection.
