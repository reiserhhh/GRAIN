# NEXT75: source response-informed coordinates for the same scalar selector

2026-09-11. PLANNED after NEXT74. Develop the derivative geometry of the selector;
do not restart a generic prediction-head, alpha or attention-attribution grid.

## Decision to make

NEXT74 learns primal selection (new8 KL falls26-34%) but its same-function GRAIN
tangent is miscalibrated. Even all4-source joint derivative residual loss is~1.
Is the existing variance-only feature map missing the response directions, or is
the coupled fitting geometry responsible? First answer this on sources, then test
one source-response-informed coordinate construction at unchanged width.

## Part A: source-only finite accounting

Reuse the exact NEXT74 source statistics. For each of five references and both
queries, eigendecompose the derivative Gram with a fixed numerical pseudoinverse
tolerance. Report source target energy projected into/outside the derivative
design, numerical rank and the residual of the fixed joint solution. Also report
the value cost at the minimum-norm derivative projection, as a diagnostic only.
Never apply this unconstrained projection to new8 or select an alpha with it.
These are10 sufficient-statistic accounts, not additional predictive candidates.
Implementation pin: Gram eigenvalues below1e-10 times the largest are excluded.
Report this numerical range explicitly; zero design columns remain zero.

## Part B: one derivative-informed source projection

Use the full H16 coordinates before the old variance-only PCA. Standardize from
permitted source H16 only, and differentiate that fixed scaling. Remove only the
same sequence constant gauge. Build a source-weighted cross-moment matrix from:

    C=[E Z^T Y_N /s_N, E_f DZ_f^T DY_Nf /t_N,
       E Z^T Y_C /s_C, E_f DZ_f^T DY_Cf /t_C].

Y/DY are NEXT74 teacher-minus-identity logits/derivatives under the split-specific
source anchor. C has1536 rows and48 columns (two12-head targets and their directional
counterparts). Use its top32 left singular vectors as a source-supervised projection,
then the same causal-context/position design and one joint alpha=.01 objective.
Fit new source-only feature centers/scales and retain zero-pinned constant gauge.
If rank32 is unsupported, report the deficiency and stop rather than silently
filling arbitrary directions or searching ranks. Preserve source4LOO/new8all4.
Source input mean/SD use equal-author/query weights on allowed tokens; source
target normalizers s/t are scalar RMS across allowed tokens/heads (and all4
directions for t). Rank uses max(matrix.shape)*machine_epsilon*sigma_max.
Backscale the left singular vectors by the source input SD, so the existing
H16 feature function differentiates the exact same standardized projection.

This is supervised cross-moment dimension reduction, **not** recovery of a complete
neural Jacobian from four directional observations or proof of a true active
subspace. It changes the projection, not the mathematical scalar/derivative rule.
Ten new coefficient solves; retain all NEXT74 comparators, both queries/all4
factors/both readouts. A derivative-informed source projection is motivated by
[O'Leary-Roseberry et al.,2022](https://doi.org/10.1016/j.cma.2021.114199),
whose full-Jacobian/PDE assumptions are not imported into this partial-tangent task.
Metadata and author summaries were checked; read the full mathematical assumptions
and the [2023 corrigendum](https://doi.org/10.1016/j.cma.2022.115673) before importing
any error bound. This plan borrows the design motivation, not that bound.

## Outputs, execution and next decision

- Freeze code/plan/tests before outcome application. Source-only fit, pure H16+
  value packet application, then target evaluation; numeric current K/dK/Q denied.
- Source projection-capacity table, complete selector/query/factor/contrast tables,
  amplitude, signed fidelity and errors. Freeze outputs before native target reads.
- Source exclusion, common-function derivative, full-to-reduced map derivative,
  masked gauge and deterministic replay tests. Explicit canonical artifact names.
- Zero new neural/API/Y/MEPS;10fits plus10source capacity accounts;180CPU seconds
  including one replay. No new8 model selection. Stop after this one candidate.
- Primary question: can response-informed coordinates improve derivative and
  action error while retaining the already found selector benefit? If yes, move
  toward removing the separate native-value dependency. If not, identify whether
  missing nonlinear/token coupling or an incompatible target support is indicated;
  do not substitute arbitrary derivative shrinkage or broaden the grid.

Overall goal remains ACTIVE; this plan has not yet executed. The clinical/TA
interpretation work remains a source-backed interface, not a parallel training run.
