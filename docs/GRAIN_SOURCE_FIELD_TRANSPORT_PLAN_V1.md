# HM27: Is the Transported Action a Shared Field or a Contextual Field?

Status: ready to freeze, not executed. HM26 compilation is verified, not a final model.

## Scientific Question

Hermite source observations improve cold endpoint actions. Before increasing
complexity, compare shared physical covectors with the conditioned potential.
This distinguishes new information from unnecessary nonlinear approximation.

## Frozen Candidate Design

Reuse all48 HM25 paths/9 nodes, original source8/validation4/application4
families and three queries. No new model/API/private calls, no alpha/rank grid.
Read source gradients only when constructing anchors; held gradients are
evaluation only. Original blocks contain <=4 tokens; use the fixed degree0-3
map O and its checked Moore-Penrose pullback O^dagger.T g. Directions and
covectors must be in the same physical moment coordinates. Do not average
unaligned raw token tensors with different lengths.

Compare exactly these controls, preserving all of them:

1. Source start covector mean: average the8 source-reference pullbacks.
2. Source path-mean covector: Simpson-integrate each source path then equal-family
   average; it is a shared approximation, not the target's actual integral.
3. Constant source endpoint-action mean: state-blind information control.
4. The frozen HM26 full24 value-only and Hermite potentials.
5. Zero action: physical amplitude baseline, not unobserved psychological zero.

Before any fit/held evaluation, add two non-tuned identification controls:
source start and source path-mean COMMON covectors obtained by jointly solving
all source token-gradient constraints. For each family, minimize mean token
gradient squared error, with equal family mass. The fixed four moment columns
and machine rank threshold are not selected by outcomes. Keep the original
per-family-pseudoinverse averages as separate candidates rather than replacing
them after seeing results. There are8 candidates in total.

If O_T is the4-by-T moment map, each token gradient g_T identifies the affine
set {lambda: O_T.T lambda=g_T}. For T<4 the extension is not unique. The average
of independently minimum-Euclidean-norm extensions is not generally invariant
under a nonorthogonal coordinate change. Fix R=diag(1,2,3,4) before evaluation:
compare proper lambda'=R^-T lambda, x'=R x with a fresh Euclidean pseudoinverse
in that frame. Original units remain the primary; frame probing is not a model
selection grid. Joint source constraints identify a quotient, not necessarily
every ambient coefficient. Cold displacement outside joint source support is
explicitly reported, not filled with claimed true information.

Cold candidates1/2 act on actual moment displacement by dot product. Save
source coefficient arrays, per-family dispersion, all actions, cross-query
action covariance and per-role errors/signs. No held rotation, coordinate
selection or fit. Explicitly distinguish the moment-space source extension
from exact full-context decoder dependence.

## Decision

If the source path mean transports as well as Hermite, retain a simpler shared
action law and seek fresh condition changes before claiming a nonlinear field
is necessary. If Hermite's conditional action materially exceeds both source
anchors, retain a learned context-dependent potential and identify which
context coordinates carry that increment. If both fail, next data acquisition
must change source support or missing-context observation, not precision grids.
Any small/negative result remains part of the same controlled scientific scope.

## Acceptance

Source anchors frozen before held native values load; all48 endpoint addresses
retained; cached logits and projection-unit checks; synthetic coordinate
transport, constant-field and curved-field counterexamples; alternate direct
SVD/linear-system pullbacks and every covariance/error replay. No overall
psychological/theory completion status from this finite diagnostic.

Save source coefficients/constraint spectra/family dispersion;384 candidate
actions,72 query-role metrics,72 paired-query covariance accounts and192 fixed
frame comparisons. Source family model-fit residuals are physical gradient
errors, not psychometric residual variance. No held derivatives in any anchor.
