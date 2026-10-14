# F06 scalar IRT/CAT reference

Date: 2026-09-16. Canonical run: `results/grain_gptpro_foundation_v3/F06/irt_full_r1`.
Audit: `F06/irt_audit_r1`. This is the independent scalar reference required by F06-02/F06-04, not the Gaussian-state panel or empirical text-scale calibration.

## Result

With 24 responses, posterior-integrated information-gain selection reduces scalar parameter MSE by 16.0% relative to random item selection and 10.8% relative to a fixed central item. The gain concerns parameter measurement; independent future binary responses retain substantial irreducible variation, and their realized Brier difference is not clearly resolved in this sample.

| Policy | Parameter MSE | Parameter RMSE | Mean posterior variance | Probability MSE | Future Brier | Interval coverage |
|---|---:|---:|---:|---:|---:|---:|
| Random | .169728 | .411980 | .183392 | .006328 | .194625 | .9800 |
| Uniform coverage | .181449 | .425968 | .182937 | .006770 | .194423 | .9683 |
| Fixed difficulty 0 | .159870 | .399838 | .171736 | .005846 | .193310 | .9750 |
| Fisher at posterior MAP | .146864 | .383229 | .155514 | .005562 | .193386 | .9667 |
| Expected information gain | .142642 | .377679 | .155407 | .005370 | .193352 | .9667 |

Paired differences at 24 responses, information gain minus reference, 2,000 author bootstraps stratified by seed:

- Parameter MSE vs random: -.027086, 95% CI [-.043040, -.010614].
- Parameter MSE vs fixed center: -.017229, CI [-.030730, -.002971].
- Probability MSE vs random: -.000958, CI [-.001536, -.000356].
- Future Brier vs random: -.001273, CI [-.003581, .001234].
- Future Brier vs fixed center: +.000042, CI [-.001973, .002039].

Information gain and MAP Fisher are separate implemented criteria. Their scores and trajectories need not coincide in this panel; the numerical difference is not used to announce one universally superior. Fixed central questions are already a strong reference under a centered population prior.

## Frozen law and completion of unspecified details

The original protocol specifies p(y=1|a,d)=sigmoid(a-d), d in {-1,0,1}, a=-3:.1:3 and budgets 8/16/24. It does not specify the scalar prior, true-parameter sampling distribution, or which fixed item to use. Before any outcomes, `protocol.json` records the following explicit completion assumptions:

- A discrete standard-normal prior on the prescribed 61-point grid. Simulated a is drawn from that same prior. These are exact finite-model calculations, not an approximation claim about unbounded human traits.
- The fixed policy uses d=0; coverage cycles through -1,0,1; random choices use a separate pre-drawn stream.
- Fisher uses the current posterior MAP, with I(a,d)=p(1-p). Information gain integrates the complete posterior rather than substituting a point estimate.
- All candidates and posterior ties use fixed increasing order with tolerance 1e-14. No application outcome selects a prior, grid, policy or criterion.
- Counts are 300 source, 100 validation and 200 application per seed, for seeds 20260915/16/17. The item law and prior are declared, so no source fit or validation tuning is required or claimed. Final comparison uses the 600 application individuals.

Every policy shares a person's latent a, pre-drawn response uniforms and independent future-outcome uniforms. Different chosen questions can legitimately generate different responses under the shared uniforms. Future evaluation averages the three fixed difficulties, not the policy's selected-item distribution.

## Equations and implications for measurement design

For an observed item, the finite posterior is

    w_new(a) proportional to w_old(a) * p(a,d)^y * (1-p(a,d))^(1-y).

The response score derivative is y-p, hence expected squared score is p(1-p). The exact information-gain criterion is

    IG(d) = H(w) - sum_y P(y|d,w) H(w_after(d,y)).

The ledger reports mean, variance and equal-tail grid interval AFTER the last update. Duplicate event IDs are rejected; repeated difficulty on a genuinely new independent response is not treated as duplicate evidence.

Unlike F06's directional family panel, a single scalar item difficulty is identifying in the infinite-response limit: a=d+logit(p). In the earlier family panel a fixed corner leaves three axes observationally identical. Therefore "always vary conditions" and "always fix conditions" are both inadequate universal rules. The required variation depends on which unknown function or parameter is being measured.

Posterior entropy may rise after a realized response even though expected information gain is nonnegative. The full run contains 477 entropy-increase steps. A preliminary test incorrectly expected ten positive responses followed by a negative response to increase entropy under this specific prior; it actually decreases entropy. Tests now retain that counterexample and separately demonstrate a valid bimodal case where surprise increases entropy. No simulation design was altered to force the increase.

Adaptive interval coverage at 24 responses is .9667, Wilson 95% interval [.9491,.9783]. This is prior-predictive simulation coverage in a known discrete model, not repeated-sampling coverage for every fixed human trait. Equal-tail rounding on a finite grid can yield conservative coverage; it is not evidence of clinical reliability or of the correctness of a learned reader.

## Execution and evidence

- Smoke run completed first. Full run: 1,800 simulated individuals, five policies, 216,000 steps and 27,000 budget-level rows; about 13.29 seconds.
- Every observation is also exported in the F00 common-record schema. Policy branches have distinct event IDs, with shared author IDs for paired comparisons. All 216,000 records validate; there is no fabricated textual evidence (`synthetic:no_text/irt_trial`).
- Separate audit reconstructs posteriors from cumulative item success/failure likelihoods, independently enumerates both outcomes for EIG, and verifies all selected actions, responses, scores, budget posteriors and common-record correspondence.
- All 96 paired intervals replay; 16 archived hashes verify; maximum numerical discrepancy is 7.3830e-15. This is a same-agent independent-formula audit, not third-party reconstruction.
- Six scalar tests plus the existing 15 repaired F06 tests pass: 21 total, .57 seconds. Ruff check and formatting pass.

Code:

- `grain_research/gptpro_foundation_v3/f06_irt.py`
- `scripts/run_grain_gptpro_f06_irt.py`
- `scripts/audit_grain_gptpro_f06_irt.py`
- `tests/test_grain_gptpro_f06_irt.py`

```sh
'../project persona/.venv/bin/python' scripts/run_grain_gptpro_f06_irt.py --output results/grain_gptpro_foundation_v3/F06/new_irt_run
'../project persona/.venv/bin/python' scripts/audit_grain_gptpro_f06_irt.py --input results/grain_gptpro_foundation_v3/F06/new_irt_run --output results/grain_gptpro_foundation_v3/F06/new_irt_audit
```

## Remaining scope

F06 now has separate numerical evidence for family discrimination, scalar CAT, and the permitted hidden-policy sensitivity counterexamples. The full Gaussian four-policy panel with separate a/b/state/beta targets, reactive versus nonreactive worlds, source-frozen priors, and information/perturbation tradeoffs remains executable work. Original card-level execution requests, dependency manifests and acceptance mapping also remain open. These are not transferred to human validation and no parent task is automatically promoted by this report.

No API, neural training, private raw text, MEPS fitting, lockbox access, remote use, commit or push occurred. No live job remains from this packet. The full research goal remains ACTIVE.
