# NEXT82: a material-conditioned reader field from values and directional teaching

2026-09-11. PLANNED, not executed. NEXT81 located a BODY-to-reader input that the
old d(c) field forces to zero. Develop that field now, rather than another
existence-only test or context weight grid. No new native capture this card.

## 1. Fixed learning panel and data split

Use source4 ALL for construction and new8 for application. Do not call a
source4 leave-one-out comparison valid here: NEXT81's clamped states used the
all-four mean, which includes any nominally excluded source. Source results are
training descriptions. New8 remains reused discovery material, not a new lockbox.

Teach the differential reader d=r_N-r_confirmation only; keep common reader m
fixed to NEXT80 all4. Reuse exactly:

-8 value observations: four original and four fixed-last source states;
-16 BODY derivatives: four frozen GRAIN directions at each clamped source;
-4 context derivatives: NEXT80 all4 source directional secants at original states.

No current new8 r/target is read during construction/application. No human labels,
MEPS, APIs or extra neural work. Old function/source membership stays fixed.

## 2. Three nested input contracts, not a parameter sweep

Represent BODY in shared hidden coordinates, maintaining the old factor/address
contract and freezing GRAIN values when differentiating H:

```text
C = concat(H_N,last, H_C,last) / sqrt(2)
M0 = mean_t H_BODY,t
Mf = mean_t (G_tf-mean_t G_tf)(H_BODY,t-M0), f=1..4
x_A = [C]
x_B = [C, M0]
x_C = [C, M0, M1, M2, M3, M4].
```

These moments retain a BODY input, not its entire token configuration or all
layout. Each block is divided by its original-source median pairwise distance,
then the concatenation by sqrt(number of active blocks). Zero-distance blocks
are explicitly skipped and recorded. No recipient PCA, whitening or rank choice.
No additional feature centering or whitening is applied. Derivatives use the same feature function:
DM0=mean(D_BODY), DMf=mean((G_f-mean G_f) D_BODY). For NEXT80 context v, both query
last tangents equal v; for NEXT81 BODY directions, C tangent is exactly zero.

Three fixed Gaussian kernels k(x,z)=exp(-||x-z||^2/2), one per input contract.
The context-only arm structurally cannot fit BODY derivatives, while the others
can. There is no bandwidth/alpha/factor/rank search.

## 3. One compatible derivative-observation learner

Use the Gaussian value/first-derivative Gram matrix, evaluated analytically:

```text
D_v^x k(x,z) = -v^T(x-z) k(x,z)
D_v^x D_w^z k(x,z) = [v^T w-(v^T(x-z))(w^T(x-z))] k(x,z).
```

Set d0 to the mean of the four ORIGINAL source differential readers. The value
targets are d-d0; derivative targets are unchanged. The predicted field is
`d_hat(x)=d0+kernel_residual(x)`, retaining the original source mean away from
the kernel support. There is no separately fitted intercept.

For residual-value/BODY/context observation groups g, set the common source-only weight
`w_g = 1/sqrt(n_g * mean_g ||target||^2)` (use1 only for an all-zero target group,
with explicit flag). These same weights are used in all three arms. Fit the
weighted28x28 kernel system with `lambda=.001*trace(K_weighted)/28`; reject a
zero/nonfinite trace rather than tune. Three vector-valued solves maximum.
Retain zero BODY kernel rows in arm A as structural residuals; never divide by
their zero derivative norms. Output original-coordinate d vectors.

Check analytic kernel derivatives, Gram symmetry/PSD tolerance and recovered
field derivatives on synthetic data. Report source value and both derivative
losses separately, kernel rank/condition, and old context-only structural zero.
No GP posterior or human uncertainty claim is attached to this regularized fit.

## 4. Application, matched outputs and decision

On new8 compute d_hat and its four GRAIN BODY derivatives using only inputs and
source packages. Preserve all three arms. Freeze before opening native readers
or evaluation packets. Compare d vector error and fixed-original-native-evidence
scalar/K/V projections against old constant/kernel/motion baselines, with both
centered and raw errors. No significance claim on8 or implicit winner promotion.

Also compose the fixed four selector/value functions using an explicit anchor:

```text
Fhat_q(H; A)=rhat_q(A)^T phat_q(H), rhat_N/C=m +/- d_hat/2.
```

At H=A, report the anchored BODY partial `rhat^T Dphat`, which has the old native
K/V component target. Separately report the diagonal derivative adding
`Drhat^T phat`; it is a property of the numerical scalar, not that old native
component target. Check both with matching finite differences, holding A fixed
only for the anchored partial. No silent stop-gradient/global-function claim.

If BODY inputs recover source derivative variation and improve new8 projected
reading, retain a usable material-conditioned field. If source variation is
recovered but transfer fails, inspect its source support/observable projection
once; do not run a hyperparameter rescue grid. If even source variation is absent,
derive the fixed moment-map nullspace before adding token structure. An input
family result is not proof that four GRAIN moments exhaust the material channel.

## 5. Budget and outputs

At most90 cumulative numeric CPU seconds, three source solves, zero neural/API/
human-label calls. One application/evaluation freeze and exact cached replay.
Preserve old code, scorer, lockbox and all negative arms. Same source code/data
provenance and explicit application read allowlists as previous finite cards.

Suggested `material_reader_field.py`, runner and tests; private
`logic_material_reader_field_v1/`; aggregate `MATERIAL_READER_FIELD_REPORT_V1.md`.
Report fit/dependency counts, source derivative losses, new8 vector/projected
errors, all four numerical compositions, anchored versus diagonal actions and
replay/cost. This develops the common same-material interface, not a new clinical
scale or a declaration of complete theory. Goal ACTIVE.

Precedent: [Solak et al.,2002](https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html).
The kernel formulas and explicit anchored/diagonal distinction, not citation
alone, define the computation to be tested.
