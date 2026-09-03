# V8 Residual Geometry Correspondence

Status:
`OPENED_PANEL_AXIS_FREE_FALSIFICATION`

## Question

Opportunity-filtered M concordance survives in PANDORA, Essays, and X-market,
but no filtered global linear axis replicates. The next estimand is therefore
not another factor axis. It is the relation geometry among authors:

\[
R_{c0},R_{c1}
\longrightarrow
\widetilde K_{c0},\widetilde K_{c1}.
\]

Does the same author set preserve pairwise geometry across technical
replicates after declared opportunity conditioning?

## Frozen kernels

For residual author coordinate \(R_r\), the linear and RBF kernels are

\[
K_r^{lin}=R_rR_r^\top,
\qquad
K_r^h(i,j)=
\exp\left(-\frac{\|R_{ir}-R_{jr}\|^2}{2h^2}\right).
\]

The base bandwidth is the within-context D0 median residual distance. The
registered multipliers are \(h\in\{.5,1,2\}\). No D1/D2 bandwidth tuning is
allowed.

Replicate-specific declared-opportunity profiles define

\[
M_{Z_r}
=
I-K_{Z_r}(K_{Z_r}+\lambda nI)^{-1},
\qquad \lambda=.1,
\]

and the conditioned relation matrix is

\[
\widetilde K_r
=
\mathcal U\left(M_{Z_r}HK_rHM_{Z_r}\right),
\]

where \(\mathcal U\) is within-context U-centering: the diagonal is removed
and off-diagonal row/column means are subtracted with the finite-sample
\((n-2)\) correction. This prevents mechanically large self-similarity
diagonals from producing high alignment after author correspondence is
permuted.

The primary kernel relation correspondence is an off-diagonal
relation-concordance statistic:

\[
\operatorname{KRC}_{rel}
=
\frac{\langle\widetilde K_0,\widetilde K_1\rangle_F}
{\|\widetilde K_0\|_F\|\widetilde K_1\|_F}.
\]

The U-centered matrices need not be positive semidefinite. Consequently KRC
is not interpreted as an ordinary positive-definite kernel correlation. The
corpus statistic is a context-stratified block-diagonal direct sum. Its
Frobenius inner product weights contexts by retained off-diagonal pair mass;
its estimand is a random retained within-context author pair, not an equally
weighted average context. It contains no cross-context author relation and
does not identify one unified corpus geometry.

Local geometry is measured by overlap of conditioned-linear nearest-neighbor
graphs at fixed fractions \(.05,.10,.20\). This precedes, and gates, any use
of diffusion maps or other manifold coordinates.

## Null and gates

The null uniformly permutes the complete replicate-two author relation matrix
within observed context. Fixed points and the identity correspondence remain
in the support. Permuting the already-conditioned matrix is algebraically
equivalent to synchronously permuting replicate-two residual rows, nuisance
rows, residual-maker rows/columns, and the resulting relation matrix because
every fitted rowwise and kernel operation is permutation equivariant. D0
freezes bandwidths and nuisance maps. D1/D2 use 1,999 null worlds and 999
context-stratified paired-author bootstraps.

The bootstrap resamples paired authors and U-centers each resampled
off-diagonal block. Its interval is conditional on the D0-fitted background,
coordinate residualizer, nuisance relation operator, and bandwidth; it does
not include full-pipeline refit uncertainty. Conditioning is relative to the
registered nuisance profile and ridge value. It never establishes that all
opportunity variation has been removed.

All corpus, split, and kernel cells share one maximum-T family.

- `DISTRIBUTED_LINEAR_GEOMETRY`: linear KRC has positive excess, maxT
  \(p\leq.05\), and positive bootstrap lower bound in both D1 and D2.
- `AXIS_FREE_SHORT_SCALE_KERNEL_CORRESPONDENCE`: one registered sub-median
  RBF scale passes in both panels.
- `AXIS_FREE_MULTI_SCALE_KERNEL_CORRESPONDENCE`: at least two registered RBF
  scales pass in both panels.
- `AXIS_FREE_SINGLE_SCALE_KERNEL_CORRESPONDENCE`: one non-short registered
  RBF scale passes in both panels.

These three statuses license reproducible relation correspondence, not a
nonlinear increment, unless the paired RBF-minus-linear gate also closes.
- `NONLINEAR_RESIDUAL_GEOMETRY`: one RBF scale passes those gates and its
  paired excess over linear KRC also passes in both panels.
- `PERSISTENT_LOCAL_RESIDUAL_GEOMETRY`: at least two adjacent registered
  neighborhood fractions pass in both panels.
- `SCALAR_CONCORDANCE_ONLY`: signed trace survives, but no relation-geometry
  family passes.

The current panels are opened. A positive result is exploratory and requires
fresh D3 authors before promotion. Single-context disaggregation and the
subsequent transport diagnostics reuse these D1/D2 authors and must not be
counted as independent confirmations. A context that fails this RGC family is
reported as `NO_CONTEXT_RELATION_GEOMETRY_RESOLVED`; the context diagnostic
does not rerun or establish the separate signed-scalar concordance claim.

## Methodological alignment

The relation-matrix estimand follows the same general logic as centered
kernel alignment and representational-similarity analysis: compare relations
among observations rather than demanding coordinate-wise axis identity. RBF
KRC is related to kernel dependence methods such as HSIC. Multiscale graph
correlation motivates the registered local-scale follow-up, but GRAIN retains
its context-restricted exchangeability null rather than importing an
independent-observation asymptotic test.

This experiment cannot identify personality, emotion, cognition, a latent
factor, causality, cross-corpus common coordinates, diagnosis, or clinical
validity.
