# NEXT71: content, selection and mass contributions in frozen readers

2026-09-11. PLANNED after NEXT70; no new training or selector search.

## Decision this changes

NEXT70 normalized selection helps source N cross-response; raw+logmass helps
new8 GRAIN actions more, without improving whole scores. Before building another
selector, identify whether the usable effect comes from the content pathway,
the evidence-selection pathway, the mass pathway, or the refitted local prefix.
This is a function-mechanism account, not a capacity/regularization sweep.

## Frozen inputs and computation

Use only audited NEXT70 four models and48packets, matched-local cached controls,
source4fourviews/new8original, bothqueries/fourGRAINdirections. No new authors,
native calls, fits, humanY, MEPS, rankings or public text. Up to60CPU seconds for
account plus replay. Freeze the implementation before evaluating the account.

For each arm compute exact coordinate gradient pieces:

    rel: C_t=sum a*v; S_t=sum a*s*[v dot(z-r)]; M_t=sum a*b*s
    raw: C_t=sum w/T*v; S_t=sum w/T*s*[v dot z]; M_t=sum a*b*s.

Keep the existing fitted local gradient L. Sum L+C+S+M must match the frozen
gradient and all four GRAIN actions before joining native targets. The pieces
are computational derivative terms, not assumed separate integrable scalars.
No label thresholds and no per-author choice of components.

Source B response: differences of each piece across plus/minus. L change must
be exactlyzero on unchanged B. Report signed target projection, each component
energy, all cross-terms and total SSE. Check exact reconstruction. Different
component signs are useful evidence of competition/cancellation, not errors.

New8 actions: separate fitted local change versus matched-local (dL), C,S,M.
For baseline error e0 and total change d=sum_j d_j:

    E_new-E_old = 2<e0,d> + ||d||^2.

Report both raw matrices and symmetric credits
`credit_j=-(2<e0,d_j>+<d_j,d>)/||e0||^2`, whose sum equals relative error reduction.
Credits may be negative or exceed1. They are a declared symmetric algebraic
allocation, not unique causal shares or percentages of personality information.
Keep native whole scores as the existing report; do not fit another score head.

## Deliverables and exit

- Numeric module, small synthetic identity/gradient tests, replayable runner.
- Public aggregate component/action/cross-term tables; private arrays ignored.
- A short mechanism report identifying what explains the already-observed gain.
- Mainline decision: preserve useful component(s) at their observed scope. If
  selection is not the useful component, stop generic normalization development
  and reconnect source-compiled query-directed selection rather than grid kernels.
- Always finish after the exact account; do not grow an oracle upper-bound study.
  Overall goal remains ACTIVE. Old clinical explanation/scorer/lockbox rules stay.
