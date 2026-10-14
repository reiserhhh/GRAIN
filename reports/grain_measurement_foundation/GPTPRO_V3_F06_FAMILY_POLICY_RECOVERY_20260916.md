# F06 Panel A: adaptive family discrimination

Date: 2026-09-16. Canonical output: `results/grain_gptpro_foundation_v3/F06/family_full_r2`.
Independent-formula replay: `F06/family_audit_r2`. Same-agent numerical audit, not independent human replication.

## Finding

Choosing informative conditions improves finite identification of a simulated response function. At 24 observations the adaptive policy lowers probability MSE by 35.8% relative to random selection. This establishes an implementable observation-design component, not human personality accuracy or new LLM mechanism evidence.

Application results pool 600 authors over three seeds. Source/validation/application counts per seed are 300/100/200; the nine generative hypotheses and uniform prior are declared rather than fitted. Every policy shares the same author and pre-drawn outcome uniforms. Future outcome uniforms are reserved for terminal evaluation. Allocation across the nine atoms differs by at most one author per role; uncertainty is conditional on this approximately balanced simulation design.

| Policy, budget 24 | Family accuracy (3 axes) | Atom accuracy (9 axis/strength pairs) | Probability MSE | Future Brier | Posterior entropy |
|---|---:|---:|---:|---:|---:|
| Fixed corner | .3400 | .2017 | .038660 | .231527 | 1.919533 |
| Random | .8667 | .5333 | .008626 | .202540 | 1.150864 |
| Coverage | .8483 | .4917 | .009550 | .203947 | 1.129101 |
| Epistemic variance | .9217 | .5583 | .005540 | .198464 | 1.006632 |
| Expected entropy gain | .9217 | .5583 | .005540 | .198464 | 1.006632 |

Adaptive minus random, 2,000 paired author bootstraps stratified by seed:

- Probability MSE: -.003085, 95% CI [-.004359, -.001876].
- Independent future Brier: -.004076, CI [-.006644, -.001675].
- Posterior entropy: -.144232 nats, CI [-.173576, -.118407].

The objective is family discrimination; selecting more distinct conditions is not intrinsically better. Adaptive selection visits about 2.335 distinct corners at budget 24, versus 8 for coverage. The sign-complement symmetry makes several corners redundant. Full nine-atom recovery is materially harder than identifying the three axes.

## Why the two adaptive policies are identical here

For hypothesis h=(k,beta), p_h(c)=sigmoid(beta*c_k), with c_k in {-1,+1}. Thus p_h(c) is either q_h or 1-q_h. For any posterior w:

    V(c) = pbar(c)*(1-pbar(c)) - sum_h w_h*q_h*(1-q_h)
    IG(c) = H_Bern(pbar(c)) - sum_h w_h*H_Bern(q_h)

The subtracted terms are independent of c. Both remaining terms are strictly increasing functions of closeness to .5, so both criteria rank all candidates identically. With identical lexical tie resolution and matched uniforms, induction gives identical complete trajectories. Acquisition values are different, but selected actions coincide. This is exact redundancy within this frozen symmetric panel, not a copied-policy implementation and not universal equivalence of epistemic variance and information gain.

At the fixed all-negative corner, all axes have the same law for each beta. The axis posterior consequently remains uniform after every observation. The .34 accuracy reflects the finite allocation plus lexical tie choice, not axis information. More responses in that one condition cannot resolve this ambiguity.

## Execution and correction

- All 1,800 simulated authors executed five strategies: 216,000 step records and 27,000 budget-level metric rows.
- The separate audit recomputed every action, matched outcome, whole-likelihood posterior, post-update entropy, all metric rows and 48 paired intervals. Maximum numerical discrepancy: 1.1102230246251565e-15; 10 archived output hashes verified.
- Seven focused tests pass, including likelihood replay, fixed-corner nonidentifiability, epistemic versus total predictive variance, lexical ties and the two-policy ordering theorem. Ruff check and formatting pass.
- `family_full_r1` and failed `family_audit_r1` are retained. The first independent replay exposed posterior class ties resolved differently by floating-point accumulation under plain argmax. Canonical r2 uses explicit lexical resolution within 1e-14 for posterior decisions as for acquisitions. No outcome-dependent change to design or selection was made. Probability losses and policy trajectories are unchanged.

Code:

- `grain_research/gptpro_foundation_v3/f06_family_policies.py`
- `scripts/run_grain_gptpro_f06_family_policies.py`
- `scripts/audit_grain_gptpro_f06_family_policies.py`
- `tests/test_grain_gptpro_f06_family_policies.py`

Reproduce in a new output directory:

```sh
'../project persona/.venv/bin/python' scripts/run_grain_gptpro_f06_family_policies.py --output results/grain_gptpro_foundation_v3/F06/new_family_run
'../project persona/.venv/bin/python' scripts/audit_grain_gptpro_f06_family_policies.py --input results/grain_gptpro_foundation_v3/F06/new_family_run --output results/grain_gptpro_foundation_v3/F06/new_family_audit
```

## Remaining F06 work and production implications

This packet closes the numerical panel-A comparison, not F06 or its dependent task cards. Parent requirements remain open for:

1. Gaussian intervention panel with separate a/b/state/beta targets and source-frozen nuisance estimates.
2. Scalar IRT reference using the prescribed complete grid, not these family metrics.
3. Same-noise reactivity counterfactuals and recorded versus latent-state-dependent policy likelihoods, including truncation accounting and sensitivity profiles.
4. Event-identity/duplicate-update protection, standardized common records, execution request/input manifests and original dependency-chain integration. Policy code has a truth-free selector/updater interface; the simulator necessarily receives hidden truth. Strict task-boundary packaging is not yet closed.
5. F09 higher-order/process rivals and OOF calibration are not covered by the exact nine-law panel.

For a future text-based measure, conditions should be selected for the estimand they distinguish, not merely for diversity or response variability. A condition can be reliable yet nonidentifying. The next Gaussian/reactivity panel must test the cost of changing the state while trying to measure it; panel A has no such dynamics.

No API, neural training, private raw data, MEPS fitting, lockbox access, commit or push occurred. No background job remains from this packet. Overall research goal remains ACTIVE.
