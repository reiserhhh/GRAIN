# F06: observation policy as evidence and as intervention

Date: 2026-09-16. Results: `results/grain_gptpro_foundation_v3/F06/policy_sensitivity_r1`.
Audit: `F06/policy_sensitivity_audit_r1`. Scope: the conditional counterexample route explicitly permitted by F06-05, not the full population policy comparison in F06-04.

## Mechanism established

An observation action has two distinct roles: it can reveal the present state through the mechanism choosing it, and it can change the next state through its intervention effect. These terms must not be combined or silently omitted.

With y_t = a + b*z_t + s_t + e_t and s_(t+1) = rho*s_t + beta*h_t + w_(t+1), the joint likelihood is

    integral p(s_0) product_t [p(y_t | s_t,a,b,z_t)
                              pi(h_t | s_t,observed_history)
                              p(s_(t+1) | s_t,beta,h_t)] ds

The final transition is omitted at the endpoint. The action h_t is chosen AFTER y_t; it never causes y_t or s_t retroactively.

- Recorded policy h_t=1[y_t>0]: its probability factor is a parameter-free indicator once y_t is observed. Legal records use the standard sequential Gaussian likelihood with lagged h. Impossible records have zero likelihood. This is sequential likelihood under a policy, not a claim that the distribution conditional on the entire future action sequence is an unrestricted multivariate normal.
- Hidden-state-dependent policy h_t~Bernoulli(sigmoid(gamma*s_t)): the Bernoulli factor must be integrated jointly with the state. Here the action is observed; its latent selection driver is hidden. Changing a final action changes the state inference even though it cannot change the already-produced physical state.
- gamma=0: the policy factor is exactly 2^(-T), independent of parameters. Omitting it changes only a likelihood constant, so parameter optima are unchanged. gamma=1 and 3 need not behave that way.
- No actions: beta never enters the data law. Its data likelihood is flat, even if a prior produces a numerical estimate. Scores explicitly say `prior_only_no_data_information`, not successful beta recovery.

## Design and numerical execution

Three seeds, one predeclared independent exogenous instance per seed. Each is run under zero/recorded/hidden policy and zero/distributed beta, producing 18 cases of 24 observations. These are three paired counterexamples, NOT 18 independent people or the default 600-author application comparison. No cases were selected based on outcomes.

The policy profile uses the full frozen beta=-.5:.05:1, rho=0:.05:.95 grid. Hidden-policy sensitivity uses gamma=0,1,3, with true simulation gamma=3. The previously frozen F05 core source models supply nuisance variances and the population prior; beta gets the declared N(.4,.2^2) extension. Each prefix obtains a/b from its own working action-conditioned MAP fit, then holds them fixed throughout its profile. This intentionally limited sensitivity analysis is not a joint confidence set or an unbiased causal estimator. No grid or source parameter is selected using hidden truth.

The observed-only worker runs in a separate process without opening truth. Physical states/counterfactuals are read only for terminal comparisons. Every physical comparison shares initial state, innovations, measurement noise and policy uniforms.

Grid [-8,8]/401 is tried first; omitted initial and transition probability are explicitly checked against .001. Only the predeclared expansion [-12,12]/601 is permitted. No cells failed in these cases; the maximum omitted mass was 5.3933513e-6. Unit tests separately force expansion and unresolved failure. A reported finite-grid approximation is not silently treated as an exact integral.

## Results

Conditional beta optima for hidden-policy cases:

| Seed | True beta | Ignoring state selection (gamma=0) | Joint policy gamma=3 |
|---|---:|---:|---:|
| 20260915, zero effect | .000 | .15 | .00 |
| 20260916, zero effect | .000 | .25 | .10 |
| 20260917, zero effect | .000 | .30 | .30 |
| 20260915, distributed effect | .252 | .30 | .25 |
| 20260916, distributed effect | .247 | .55 | .45 |
| 20260917, distributed effect | .458 | .50 | .45 |

The joint factor changes the inferred response parameter and improves its point error in five of these six cases, but does not remove working-nuisance and short-prefix uncertainty. Several likelihood envelopes remain broad. The known true gamma need not maximize a finite prefix's conditional likelihood, and final-state error does not improve uniformly. Gamma remains a declared sensitivity axis, not an application-selected winner.

All six zero-action cases are exactly beta-flat. In the recorded policy cases, the distributed-effect instances select 1, 2 and 24 actions respectively. This poor variation explains why even the correct recorded-policy likelihood can leave beta weakly determined, with some conditional optima at the grid boundary. Recording a policy solves one likelihood problem; it does not create excitation or eliminate confounding with a/b.

The exact same-noise physical difference obeys

    delta_s_0 = 0
    delta_s_(t+1) = rho*delta_s_t + beta*h_t
    delta_s_t = beta * sum_(j<t) rho^(t-1-j) h_j

When beta=0 all perturbations are exactly zero even though actions occur. With distributed beta, squared state displacement sums are:

| Seed | Recorded policy | Hidden policy |
|---|---:|---:|
| 20260915 | .06709 | 1.18103 |
| 20260916 | .14997 | 1.93500 |
| 20260917 | 11.95654 | 8.59667 |

Thus a policy cannot be ranked by information alone: the same rule interacts with the person's baseline and response dynamics to produce different intervention burdens. These numbers are simulated units, not psychological harm estimates.

## Reproducibility and audit

- Full runner: 18,600 profile rows, about 19.37 seconds local CPU, no API/neural calls.
- A separate audit integrates hidden-policy path likelihoods and terminal moments BACKWARD, rather than rerunning the forward filter. The recorded/zero arms are checked against whole-vector Gaussian covariance calculations. All 18,600 rows replay; maximum difference 4.2633e-14; 135 archived file hashes verified.
- The finite-grid gamma=0 likelihood differs from the untruncated analytic Gaussian by at most 5.3984e-6. This separate approximation discrepancy is retained, not confused with the near-machine-precision same-grid algebra check.
- Eight new state-policy tests plus seven family-policy tests pass (15 total, .57 seconds). Checks include an independent continuous quadrature of the hidden-action update, lagged causal timing, impossible recorded actions, exact no-action beta flatness, truth-field and duplicate-event rejection, and one-expansion-only truncation behavior.

Files:

- `grain_research/gptpro_foundation_v3/f06_policy_state.py`
- `scripts/run_grain_gptpro_f06_policy_sensitivity.py`
- `scripts/audit_grain_gptpro_f06_policy_sensitivity.py`
- `tests/test_grain_gptpro_f06_policy_state.py`

```sh
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 '../project persona/.venv/bin/python' scripts/run_grain_gptpro_f06_policy_sensitivity.py --source results/grain_gptpro_foundation_v3/F05/source_full_r2 --output results/grain_gptpro_foundation_v3/F06/new_policy_sensitivity
env OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 '../project persona/.venv/bin/python' scripts/audit_grain_gptpro_f06_policy_sensitivity.py --input results/grain_gptpro_foundation_v3/F06/new_policy_sensitivity --output results/grain_gptpro_foundation_v3/F06/new_policy_sensitivity_audit
```

## What this adds to GRAIN and what remains

For text-based measurement, the prompt or interlocutor response is part of the observation mechanism, not merely contextual metadata. A future implementation needs the timing of the reply, the known selection rule, and the target distinction between present state and response to an intervention. A deterministic hidden-state threshold is the gamma->infinity limit; it imposes sign constraints on state and is not the same as h=1[y>0] under noisy y. This limit is described, not fitted here.

F06 remains open for the original full Gaussian four-policy comparison, all a/b/state/beta error tables, scalar IRT reference, information-versus-perturbation Pareto analysis, and original task-contract/DAG integration. Those are agent-executable tasks, not human-side blockers. No original parent card has been promoted by this counterexample packet. The six human-study preparations and all other outstanding F00-F17 requirements remain in the active goal.

No private raw data, MEPS fitting, neural training, remote use, lockbox, commit or push. No live process remains from this packet.
