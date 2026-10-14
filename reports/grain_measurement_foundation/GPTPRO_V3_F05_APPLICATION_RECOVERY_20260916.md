# F05 Frozen-Source Application Comparison

2026-09-16 JST. Original V3 recovery goal remains active. Full application run
finished; shared-input channel, profile diagnostics and final acceptance remain.

## Execution

All nine source r2 bundles were reused without re-estimating dynamics. A source
LMM comparator was newly fitted on the same source first48 observations only.
Application has 200 authors per world/seed and three seeds. Five methods:
personal MAP dynamics, frozen population dynamics with personal state filtering,
static OLS, source LMM, and an explicitly separate oracle Kalman comparator.

Prefixes16/32/48 estimate personal parameters. Sequential forecasts through the
last16 keep parameters fixed at48, update states using preceding observations,
and predict one/four steps when available. Conditions come from the prescribed
clock; future outcomes are not inputs. Observed-only workers exit before the
terminal scorer reads latent targets and computes the oracle.

Results contain 162,000 parameter/state rows and 297,000 forecast rows. These
are repeated measurements of 1,800 application author instances, not 459,000
independent people. All reported application optimizer fits succeeded; explicit
failure routing is tested and full availability tables are saved.

## Results at Prefix48

Current-state MSE, lower is better:

| World | Frozen population dynamics | Personal MAP | Oracle |
|---|---:|---:|---:|
| core | .893724 | .175852 | .145755 |
| near_root | 1.263210 | .992954 | .142266 |
| shared_shock | .864901 | .178882 | .159201 |

In core, rho MSE improves from .042176 to .030037. The paired delta is -.012139
[-.015949,-.008424]. In near_root, rho MSE instead increases from .006364 to
.009945, delta +.003582 [.002601,.004575], despite reduced state error.
Near-root intercept improvement is inconclusive: -.139981 [-.314675,.036501].

Thus the original requirement to separate parameters, state and forecasts
matters. A useful expression forecast does not prove accurate personal inertia
or intercept recovery. This is simulated latent state, not named human emotion.

Four-step observed-outcome MSE:

| World | Frozen population | Static OLS | Source LMM | Personal MAP | Oracle |
|---|---:|---:|---:|---:|---:|
| core | 1.831342 | .647981 | .646551 | .633022 | .577123 |
| near_root | 2.418767 | 3.002798 | 2.981377 | 1.524288 | 1.291479 |
| shared_shock | 1.984201 | .666898 | .666217 | .662821 | .696101 |

The shared-shock oracle's realized forecast loss being higher here is not an
oracle contradiction. Authors share episode innovations; this is only three
independent episode streams, not 600 independent future shocks. Conditional
mean error remains lower for the oracle (.004082 versus MAP .047446). The joint
visible-history covariance channel has not run and remains necessary. Original
author bootstrap intervals are descriptive under that dependence, not reliable
unconditional episode-level uncertainty.

## Coverage Bug Found and Repaired

The initial summary had coverage .001667 because mixed numpy.bool_ and None
produced an object-type grouped reduction. Individual true/false indicators and
all predictions were intact. A separate arithmetic audit reconstructed 162
coverage cells with explicit numeric 0/1 conversion. No model was rerun and no
original output was overwritten. The runner and regression test are corrected.

Canonical repaired summaries are in `F05/application_audit_r1/`, not the original
`application_full_r1/state_summary.csv`. At prefix48, actual MAP/oracle coverage:
core .898333/.941667; near_root .565000/.956667; shared_shock .881667/.925000.
These are plug-in state intervals; missing parameter uncertainty is F13 work.

## Evidence and Outstanding Work

- `results/grain_gptpro_foundation_v3/F05/application_full_r1/`: frozen individual
  predictions, errors, optimizer attempts, source LMM fits and 12 paired contrasts.
- `results/grain_gptpro_foundation_v3/F05/application_audit_r1/`: repaired state
  summaries, old/new differences, successful-only summaries and fit availability.
- Arithmetic replay checks all state squared errors, interval indicators and
  both forecast errors, with 39 input artifact hashes verified. This does not
  replace an independent reconstruction of all predictions or latent laws.
- Eighteen F05 tests pass, including future outcome poisoning, optimizer failure
  fallback, dense filter equations and nullable indicator aggregation.

Next: independent prediction/CI replay; actual joint shared-shock and missing-time
contrasts; twenty source-fixed likelihood profiles/Hessians; original card and
DAG packaging. Do not label these agent-solvable gaps as human pending. No
background process remains, and no API, neural training, private data, lockbox,
commit or push was used.
