# NEXT79: a context-conditioned source differential reader

2026-09-11. PLANNED, not executed. Develop the missing conditional readout located
by NEXT78, not another attribution table or value/selector/alpha grid.

## 1. Fixed question and construction

Can available H16 query context select an appropriate source differential reader
and preserve the opposing modulation lost by the constant-reader substitute?
Keep the common source reader, all four NEXT77 numerical selector/value maps,
queries, factors, source4LOO/new8all4 and the original fixed endpoint targets.

Use the paired context c=(H16_N,last+H16_C,last)/2. Never whiten/project it with a
new8 reference or fit a PCA on all authors. For each source reference S:

    m_S = mean_i ((r_iN+r_iC)/2)
    d_i = r_iN-r_iC
    h_S^2 = median_{i<j} ||c_i-c_j||^2
    pi_i(c) = softmax_i(-||c-c_i||^2/(2h_S^2))
    d_hat(c) = sum_i pi_i(c) d_i
    rhat_N(c)=m_S+d_hat(c)/2; rhat_C(c)=m_S-d_hat(c)/2.

Unconditional comparator uses pi=1/|S| and must exactly reproduce NEXT77 readers.
One conditional construction, five source references, no coefficient solve or
bandwidth search. Zero source distance uses uniform weights and records a
degenerate-support flag. Gaussian weights use stable log normalization.
This is a normalized vector kernel smoother, not a claim of a new RKHS theorem.

## 2. Source/application/evaluation boundary

- Source construction reads only permitted source H16 last rows and NEXT51
  source endpoint gradients. Filter membership before computing m, h or d.
- Application reads current paired H16/layout, source-only reader package and
  existing NEXT77 four-function predictions. Current native r/g/K/V/Q/targets
  remain denied. No human labels, MEPS or new neural/API/model queue.
- Preserve a scalar function, not a patched derivative. Since the declared
  GRAIN directions have D[last]=0, Dc[D_BODY]=0 and the BODY derivative remains
  rhat(c)^T Dp. Verify this boundary and full scalar finite differences using
  the original NEXT77 functions. For context directions include Drhat*Dc;
  test the kernel reader derivative synthetically rather than detaching it.
- Evaluation opens cached native targets only after predictions freeze. Keep
  scalar, key/value/total, query-specific/pair-mean, N/C contrast and each factor.
  Report direct differential-reader vector error and projected response error;
  neither alone automatically guarantees the other.

## 3. Mechanism and decision outputs

Compare constant vs conditional readers on identical four functions. Report
raw and author-centered scalar errors, directional response errors/cross terms,
per-factor fidelity, source weights/entropy/effective source count, and current
context distance relative to source distances. Weight diagnostics do not choose
a new bandwidth. Retain all outcomes, not just a winning function/query.

Record one source-range diagnostic after prediction freeze: form the affine span
of source d_i around their mean and project the current native differential
reader into it. The orthogonal residual is a lower bound on vector error for
any weighting of these same source readers. This is an evaluation-only capacity
account, not a fitted recipient coefficient or a new candidate. If that missing
range dominates, the next useful action is richer source-context teaching, not
bandwidth tuning. A vector-space bound does not by itself bound endpoint action
error; retain the existing projected metrics separately.

The key new observable is whether differential-response modulation moves toward
the native target, without losing useful scalar author variation. If both show
a usable improvement, retain the conditional field as the next component. If
not, close this smoother and identify whether source response range or missing
context dependence prevents it; no automatic kernel sweep. Universal theory
closure, whole-L replacement and human personality validity are separate claims.

Suggested `context_differential_reader.py`,
`scripts/run_grain_context_differential_reader.py`, focused tests and aggregate
report. Private `logic_context_differential_reader_v1/` under vector_mechanisms;
source packages/readers/author outputs remain private. Public source4/new8
aggregate tables only. Source provenance, exact baseline replay, scalar/body
derivatives and one full cached replay required. Budget90CPU seconds including
replay, five source constructions, zero new neural/API/Y. Overall goal ACTIVE.

Rationale/primary references:
[NEXT78 theory](GRAIN_ENDPOINT_QUERY_CONTRAST_THEORY_V1.md),
[Nadaraya 1964](https://www.mathnet.ru/eng/tvp356),
[Kadri et al. 2016](https://jmlr.org/papers/volume17/11-315/11-315.pdf).
