# S11: can a useful intervention contrast become a cheap student target?

2026-09-08. After S9/S10; zero new neural calls. S9's N contrast is label-related
(r=.2116), and native configuration beats the projected endpoint in paired r.
Before more collection, test whether the existing E5 mean representation can
recover this identified component instead of only copying the complete teacher.

## Frozen finite comparison

Use all213 S9 N authors, same E5 cue-mask cache, same S4 five outer author folds.
No T/F refit on the small64-author subset: preserve its existing positive-control
results and avoid comparing a64-author fit against the2328-author S4 fit.

Targets: new FP32 teacher S, projected endpoint M, contrast C=S-M. For each target,
fit two predefined coordinate families: PCA64/Ridge and PLS-rank2/Ridge. Ridge
alpha1/10/100 is selected by three inner author folds and target MSE only.
No new ranks, embeddings, blends, Y-based fitting or selection. Report every arm,
plus each family's independently fitted M-student + C-student as a reconstruction
of S. Use the same source material; cache all fitted affine maps and verify replay.

All OOF rows must be frozen before source Y is joined. Report target fidelity,
external r, paired intervals against the frozen S4 student and relevant native
target, and retained covariance. This is a post-S9 **exposed-source discovery
continuation**: the choice of C is informed by S9's outcome connection, so the
new external comparisons are not independent confirmatory evidence.

## Mathematical leverage and a necessary algebraic control

Let P be a fixed linear learning operator, including the intercept. Then
P(S)=P(M+C)=P(M)+P(C). Merely renaming two targets cannot improve a fixed linear
student. Explicitly verify this with common PCA geometry and alpha10.

Any difference for independently tuned/reduced components must come from different
target-dependent regularization or response coordinates, not newly created input
information. A C-only student can also avoid copying the rest of S, but its
outcome utility is an empirical question. This is the next test of the conditional
expectation/projection reasoning in theory V3, not another unlimited baseline grid.

The response-coordinate choice reuses
[Cook, Helland & Su, JRSSB2013](https://academic.oup.com/jrsssb/article/75/5/851/7075921),
which motivates preserving response-relevant rather than highest-variance
directions; we use finite PLS, not the paper's likelihood envelope estimator.
The old full-score student and S5 experiments supply controls for changing the
target versus changing the coordinate method.

Budget: two CPU threads, ten-minute cap, zero forwards/backwards. If neither
family recovers useful C information, stop this mean-vector family and move to
query-conditioned/joint-vector features; do not tune on Y to rescue the table.
