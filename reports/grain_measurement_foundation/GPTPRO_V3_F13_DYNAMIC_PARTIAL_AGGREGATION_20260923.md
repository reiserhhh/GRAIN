# F13 three-level dynamic uncertainty: partial aggregation

## Recovery of the actual execution state

Both earlier processes terminated. Source fitting stopped normally at the 5,400s
budget after 1,488/1,800 tasks with zero task errors; the consumer finished those
same 1,488 and then recorded AWAITING_SOURCE_FIT. Old pids99697/274 are absent.
Resumed the original hash-checked protocols, rather than regenerating inputs or
repeating the finished tasks. New source session94813/pid59980 and consumer
session80490/pid59991 were verified advancing beyond task1,520.

## Completed-cell aggregation

New entrypoint `scripts/aggregate_grain_gptpro_f13_dynamic.py` combines all three
covariance components without dropping off-diagonals:

    V = E_source[E_parameter(V_event,state)]
        + E_source[Cov_parameter(m)] + Cov_source(E_parameter(m)).

Every source mixture has 200 components. Any missing source task leaves the
entire world/seed cell incomplete. Any unsupported author-prefix contribution
leaves that author-prefix unscored, rather than renormalizing surviving draws.
Total covariance is checked against independent second-moment algebra for every
scored prefix. The finite inner sample also contributes Monte Carlo uncertainty
to component estimates; the source component is not an unbiased pure population
variance estimator or an exact Bayesian posterior variance.

Snapshot `results/grain_gptpro_foundation_v3/F13/dynamic_aggregate_r1` contains all
three seeds for core/near_root and one shared_shock seed: seven complete cells,
4,200 author-prefix combinations, 12,600 target rows, no unsupported prefixes.
The remaining shared_shock cells were incomplete (128/200 and0/200 at snapshot).
Each person contributes three budgets; these are not independent new authors.

## Results from complete independent-world cells

Coverage averaged descriptively over three seeds and three budgets:

| World | Target | Full finite mixture coverage | Width | MSE |
|---|---|---:|---:|---:|
| core | current state | .910556 | 1.524032 | .196404 |
| core | four-step observation | .938333 | 3.074919 | .679830 |
| core | four-observation average | .923889 | 2.001196 | .311189 |
| near_root | current state | .795556 | 2.537315 | 1.061002 |
| near_root | four-step observation | .928333 | 4.156975 | 1.312811 |
| near_root | four-observation average | .928333 | 3.027294 | .702553 |

Previously fixed-parameter plug-in four-observation average coverage was
.881667/.895556 for core/near_root. Added parameter uncertainty improves those
intervals, but does not deliver universal calibration. At the frozen per-cell
Wilson gate core passes19/27, near_root10/27. Failures retained. No inference
uses these pass fractions as a global success threshold.

For near_root current state, mean variance components are event/state .138358,
personal parameter .287149, source .006036. For core these are .118904/.032348/
.000375. Source resampling is a small component here; more bootstrap draws do
not directly repair the substantial remaining near_root state undercoverage.
Future observations and current latent state are different targets, and better
future coverage does not establish accurate latent-state decomposition.

Next mechanism diagnosis should distinguish finite-prefix intercept/state
confounding, source prior bias, and non-Gaussian/multimodal parameter uncertainty
missed by Laplace. These are hypotheses, not identified causes from this table.
Do not tune Hessian floors, inflate intervals post hoc or increase bootstrap
counts until the failed cells pass.

## Evidence and remaining work

Full covariance JSON, component-bearing predictions, metric rows, unsupported
ledger and SHA256 input list are saved. Two aggregation tests check the complete
second-moment identity and nonfinite rejection.22 related tests/Ruff pass.
This verifies aggregation algebra, not a full independent re-fit/replay of every
new source fit and personal draw. Those audits and final all-nine-cell reporting
remain. Shared-shock coverage remains descriptive because authors share shocks.

The two resumed producer/consumer jobs remain active at this report's snapshot.
F13 overall, F08 target uncertainty, other specified scenarios and all-plan
contract integration remain open. No new human data, API, neural training,
private text access, lockbox, commit or push.
