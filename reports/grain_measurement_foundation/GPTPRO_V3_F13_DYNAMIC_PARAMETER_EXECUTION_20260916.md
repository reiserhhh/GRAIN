# F13 dynamic parameter uncertainty: execution in progress

## New computational layer

The prior dynamic target run held source and personal parameters fixed. This
packet actually re-estimates the source prior/noise, then the personal parameters,
before propagating uncertainty to current state and future observations.

1. Resample 300 source authors with replacement; duplicated copies get distinct
   random-effect identities and event addresses. Preserve each chronology.
2. Run the original two-round source ML/MAP procedure on the first 48 events.
   Save full fit history, original source indices, failures and noise status.
3. For each application author and 16/32/48 prefix, run the original two-start MAP
   estimator using that particular source model; future responses remain unused.
4. Approximate personal parameter uncertainty in (a,b,eta) by inverse central
   Hessian at the MAP (step 1e-4). This is explicitly a Laplace approximation,
   not an exact posterior. Failed MAP/nonpositive curvature is unsupported;
   there is no eigenvalue flooring or outcome-driven rescue.
5. Draw 200 joint parameter vectors; apply exact conditional scalar filtering
   and joint future covariance for each vector. Retain conditional event/state
   and across-parameter covariance separately. Source-model aggregation is the
   next layer and has not yet run.

For source fit b, personal draw j and target vector T:

    m_b = average_j E[T | psi_bj, prefix]
    V_b = average_j Var[T | psi_bj, prefix]
          + Cov_j(E[T | psi_bj, prefix]).

The subsequent source mixture will use average_b V_b + Cov_b(m_b). It must not
discard failed fits/prefixes and silently renormalize as if the full protocol
had completed. Method uncertainty from the Laplace approximation is not captured
by those components; it stays a declared limitation and comparison obligation.

## Execution

Source-only producer:
`scripts/run_grain_gptpro_f13_dynamic_source_bootstrap.py`.
Personal worker:
`scripts/run_grain_gptpro_f13_dynamic_personal.py`.
Grid consumer:
`scripts/run_grain_gptpro_f13_dynamic_grid.py`.
Core:
`grain_research/gptpro_foundation_v3/f13_dynamic_parameter.py`.

Planned source grid: 3 worlds x 3 seeds x 200 actual refits = 1,800.
Planned personal grid: 1,800 source fits x 200 authors x 3 budgets x 200 draws.
Repeated budgets/draws are NOT new authors. Both processes use one BLAS thread;
each has a 5,400-second wall-time cap checked between atomic tasks. Source fits
are durably saved individually; application tasks checkpoint every ten prefixes.
Resume is explicit and checks code/input hashes. Failed partial workers require
inspection, not automatic deletion/restart.

Results:
`results/grain_gptpro_foundation_v3/F13/dynamic_source_full_r1`,
`dynamic_personal_grid_r1`. Smoke and single-source integration results retained
in `dynamic_source_smoke_r1`, `dynamic_personal_smoke_r1`,
`dynamic_personal_full_r1`.

Verified progress snapshot: 90 source fits completed without source task errors;
16 source models consumed by the personal grid. First standalone integration:
600 prefixes /120,000 parameter draws, zero unsupported curvature. These counts
are snapshots, not final results. Read current checkpoint files and live process
handles for subsequent status. Source session77881 (pid99697), consumer94950
(pid274) were confirmed running; neither is a completed job.

## Checks and restrictions

Eleven related tests pass: bootstrap duplicates remain independent copies;
application/duplicate-event input is rejected; batched parameter filters agree
with dense Gaussian conditioning; all three target covariances agree with the
joint temporal law; total covariance agrees with second moments; Laplace Hessian
agrees with a dense likelihood derivative. Ruff passes. Full source-refit audit,
all-personal-draw replay, source mixture and final coverage remain open.

The shared-shock author bootstrap is conditional on one shared episode, not
independent-episode uncertainty. That world cannot inherit independent-author
Wilson inference. No new data generation/API/neural training/private texts,
MEPS fitting, remote execution, lockbox, commit or push. No final coverage
improvement is claimed yet. Full F13/F08/F14-F17 and card contracts remain open.
