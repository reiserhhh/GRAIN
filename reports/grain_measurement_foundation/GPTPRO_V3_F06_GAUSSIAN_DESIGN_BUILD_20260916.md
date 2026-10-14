# F06 B: MAP-local dynamic measurement design

Date: 2026-09-16. Status: implementation and smoke audit complete; full run in progress. This is not a final scientific acceptance report.

## Estimator and design

The unknown personal parameter is psi=(a,b,eta,beta), with rho=.99*tanh(eta). Noise variances and the first three population-prior coordinates come from the immutable F05 core source fits. The protocol's independent beta~N(.4,.2^2) prior extends that model. The nonreactive control sets true beta=0 without telling the estimator. No application outcome chooses the prior or fitting procedure.

Given a realized prefix, let c_0=0 and c_t=rho*c_(t-1)+h_(t-1). Then

    m_t = a + b*z_t + beta*c_t
    C_ij = q*rho^|i-j|/(1-rho^2) + r*1[i=j].

The exact Gaussian prefix MAP objective and its analytic gradient are implemented. Each prefix uses the two frozen starts rho=.2/.8 and the original L-BFGS-B limits. A failed pair of fits returns the source mean and source parameter covariance; it is flagged, not dropped from summaries.

For the working Gaussian covariance, use expected information AT the fitted MAP:

    I_jk = (d_j m)' C^-1 (d_k m)
           + .5 tr(C^-1 (d_j C) C^-1 (d_k C))
    V_psi = (prior_precision + I)^-1.

Only eta changes C. This is not an exact Bayesian posterior or an observed-Hessian Laplace claim. Analytic derivatives include both the rho-dependent stationary covariance and accumulated intervention forcing. Gradient checks include rho=0 and negative rho, not only the positive simulation range.

Conditional on psi, the final state has Gaussian mean m_s(psi) and variance v_s. Let g=d m_s/d psi. The joint approximation retains parameter/state cross-covariance:

    V_joint = [[V_psi, V_psi*g],
               [g'*V_psi, v_s + g'*V_psi*g]].

Ignoring these off-diagonal terms would incorrectly treat parameter and state estimation as separate channels.

After y_t, select (z_(t+1),h_t). The next-state transition row for joint coordinates (a,b,eta,beta,s) is

    F_state = (0,0, d_rho/d_eta * m_s, h_t, rho).

Predict V_minus=F*V_joint*F'+q*e_s*e_s'. For x=(1,z,0,0,1), the observation update is

    V_plus = V_minus - V_minus*x*x'*V_minus/(x'*V_minus*x+r).
    gain_j = V_minus[j,j] - V_plus[j,j].

The adaptive strategy maximizes the declared target's expected variance reduction, not an oracle's realized error. Four target variants are run separately: a,b,beta,state. The state-design target is the NEXT observed state; terminal evaluation scores the current state after its observation. Because candidate interventions can change predictive variance, maximum reduction is not generally equivalent to minimum remaining variance. This distinction is retained rather than changing the criterion after seeing results.

## Policies and paired worlds

- Fixed: alternating z, h=0.
- Random: uniform pre-drawn selection among four (z,h) corners.
- Coverage: fixed lexical four-corner cycle.
- Scalar information: one variant for each of the four target coordinates.

The first observation has no prior intervention (h_-1=0); the final action is zero because it cannot inform a response inside the budget. Prefix readouts at 8/16/24 report effective past actions separately from an action selected for the next response.

All arms share each author's parameters, initial state, process innovations, measurement noise and random candidate stream. The paired no-action state evolves under exactly the same innovations. State displacement and squared displacement are recorded independently of fitted scores. Zero-effect controls must have exactly zero physical displacement even if actions are selected.

Full counts are 300 source/100 validation/200 application per seed, three seeds, two reactivity worlds, seven actual paths (three reference policies plus four adaptive targets), 24 observations. Reference paths are not duplicated for each target; every path reports all four target losses. Shared nuisance models are already source-fitted in F05 and frozen; these new source/validation roles do not tune selection or learn another prior.

## Verification already completed

- Five new tests pass: analytic MAP gradients against central differences, conditional state means/variances against the scalar filter, state-parameter derivative checks, exact joint covariance conditioning, positive definiteness and no-action beta-prior preservation.
- All 26 repaired F06 tests pass, .67 seconds; Ruff passes.
- Smoke: 24 simulated individuals, 2,688 prefix fits and 2,688 target metric rows, about 4.10 seconds. Both starts failed on zero prefixes.
- Separate smoke audit reconstructs physical states and observations from exogenous inputs, calculates fitted state means via whole-vector Gaussian conditioning, independently computes gains through a block covariance Schur complement, checks all policy choices including the first source-prior choice, and replays target errors and perturbation accounting. Maximum discrepancy 7.7716e-16; all 24 author hashes verified. Audit r2 adds the first-choice check; r1 remains preserved.

The posterior Fisher/gradient formulas are unit-tested, not independently recomputed at every full-run prefix by the current audit. Aggregate paired intervals and final Pareto/production-rule synthesis also remain open. These boundaries are not silently promoted to full acceptance.

The aggregate-audit companion is now implemented in `scripts/summarize_grain_gptpro_f06_gaussian_design.py`. It refuses incomplete role counts; reconstructs metric/seed tables from compressed author packets; independently replays the stored paired intervals; reports target-matched adaptive-versus-fixed/coverage/random contrasts; and constructs a descriptive two-cost Pareto table without treating point dominance as statistical superiority. It also exports common records with `condition={z,h}` and explicit `action_timing=after_observation_affects_next_state`; h must not be misused as a cause of the same observation. The smoke companion `gaussian_smoke_summary_audit_r2` verifies 192 original intervals, 96 target-specific contrasts and 2,688 common records, with maximum interval difference 2.22e-16. Two reporting tests pass. The full companion run remains pending the live computation, and r1's differently named h field is preserved rather than overwritten.

## Running and recovery

Files:

- `grain_research/gptpro_foundation_v3/f06_gaussian_design.py`
- `scripts/run_grain_gptpro_f06_gaussian_design.py`
- `scripts/audit_grain_gptpro_f06_gaussian_design.py`
- `tests/test_grain_gptpro_f06_gaussian_design.py`
- `scripts/summarize_grain_gptpro_f06_gaussian_design.py`
- `tests/test_grain_gptpro_f06_gaussian_summary.py`

Smoke: `results/grain_gptpro_foundation_v3/F06/gaussian_smoke_r1` and `gaussian_smoke_audit_r2`.
Full run: `results/grain_gptpro_foundation_v3/F06/gaussian_full_r1`.

Each author is written atomically as a compressed packet, then registered by SHA-256 in `checkpoint.json`. Resume validates code, source-manifest and all completed-author hashes. An incomplete temporary author is never counted. The per-invocation time ceiling is 1,800 seconds and results are retained if it is reached.

```sh
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 '../project persona/.venv/bin/python' scripts/run_grain_gptpro_f06_gaussian_design.py --source results/grain_gptpro_foundation_v3/F05/source_full_r2 --output results/grain_gptpro_foundation_v3/F06/gaussian_full_r1 --max-seconds 1800
# Only after the live process has ended, if a time-limit checkpoint remains:
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 '../project persona/.venv/bin/python' scripts/run_grain_gptpro_f06_gaussian_design.py --source results/grain_gptpro_foundation_v3/F05/source_full_r2 --output results/grain_gptpro_foundation_v3/F06/gaussian_full_r1 --max-seconds 1800 --resume
```

Do not start a second writer while the first is live. Full computation and scientific findings have not yet been accepted. The overall research goal and original task-contract/DAG work remain open. No API, neural training, private raw text, MEPS fitting, remote execution, lockbox, commit or push is involved.
