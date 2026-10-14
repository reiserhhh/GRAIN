# F07 corrected observation-channel information

Date: 2026-09-16. Results: `results/grain_gptpro_foundation_v3/F07/information_r1`; independent-formula audit: `information_audit_r1`.

## What was repaired

The legacy `f07.py` uses level as an arbitrary logit shift and defines the flat world as a constant .5 response. That is not the frozen protocol's five opportunity/cost/reader channels. It remains preserved and is not the canonical implementation of these requirements.

New `f07_channels.py` implements the actual parameters: opportunities m=(1,1,1,3,6), costs tau=(4,6,12,40,80), latent p=sigmoid(a+b*z), presence c and symmetric reader flip epsilon. The observed law is [missing, positive, negative], not a binary score in which absence is imputed as zero.

In the baseline channel,

    s = epsilon + (1-2*epsilon)*p
    I_per_opportunity = c * (1-2*epsilon)^2 * (grad p)(grad p)' / [s(1-s)]
    I_per_token = (m/tau) * I_per_opportunity.

The reader changes the denominator as well as the numerator. Multiplying the original clean information by only (1-2*epsilon)^2 is incorrect.

For off-target contamination, s becomes .8*s+.1. For MNAR, c(theta,z)=c0*(.5+.5*p) and the full information decomposes as matrices:

    I_total = (grad c)(grad c)'/[c(1-c)]
              + c*(grad s)(grad s)'/[s(1-s)].

The first term describes the selection/presence channel. It can be nonzero even when the content is completely randomized (epsilon=.5). That is information under a known selection model, not evidence that the available content measures the target well.

Both a and b are unknown. Slope information uses the Schur complement I_bb-I_ab^2/I_aa after summing the design matrix. At a single fixed z, I_bb is positive but the efficient slope information is zero: a and b are confounded. Missing intercept information produces null rather than a fabricated zero estimate.

## Computed curves

At the frozen a=0,b=.5 reference, the balanced per-token efficient information is:

| World/variant | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | Shape |
|---|---:|---:|---:|---:|---:|---|
| U, baseline | .000539 | .001078 | .001796 | .008081 | .016163 | increasing |
| D, baseline | .048488 | .028211 | .009913 | .003785 | .000598 | decreasing |
| P, baseline | .002694 | .006566 | .007351 | .003785 | .000149 | peak at 3 |
| P, MNAR | .001502 | .003807 | .005149 | .005672 | .003948 | peak at 4 |
| Flat, baseline | .003048 | .003048 | .003048 | .003048 | .003048 | calibrated flat |

MNAR changes the maximizing level because selection also carries parameter dependence. A per-token optimum therefore depends on what is observed and how its availability is generated, not just on the content reader.

The flat world calibrates c_l to equalize BASELINE per-token information at the reference parameters, as explicitly recorded. It is constructed, not discovered. Its equality is not preserved after off-target contamination or parameter-dependent presence. In the MNAR extension its curve rises from .001701 to .002559. Calibration is conditional on the observation model.

These are channel levels, not five psychological degrees of openness. No continuum of human psychological openness has been established by these equations.

## Execution and checks

- 120 budget-level matrix rows: four worlds, three variants, five levels, fixed 400-token or fixed 32-opportunity budgets. Odd counts retain their actual imbalance in z when eliminating the intercept.
- 360 analytic derivative comparisons at steps 1e-4,1e-5,1e-6; maximum relative error 1.2456e-8, meeting the original multi-step derivative rule.
- 360 actual multinomial simulations of 100,000 independent opportunities each. Counts and Monte Carlo information/SE are stored; no arbitrary MC-error cutoff is used as a correctness criterion.
- Independent presence/content decomposition reconstructs all 120 information matrices and all 360 MC analytic references; maximum discrepancy 5.3291e-15; eight archived output hashes verified.
- The independent five-category Gaussian ordinal reference uses the prescribed cutpoints and SD=1. It has a separate law and derivatives; a binary flip mechanism is not invented for ordinal responses. Its latent scale/noise differ from the binary model, so a raw information comparison is not a claim that ordinal response is universally better.
- Six tests pass, including normalization, missingness derivatives, random-content/informative-presence counterexample, fixed-condition confounding and explicit flat calibration. Ruff passes.

## Artifacts and remaining work

- `grain_research/gptpro_foundation_v3/f07_channels.py`
- `scripts/run_grain_gptpro_f07_information.py`
- `scripts/audit_grain_gptpro_f07_information.py`
- `tests/test_grain_gptpro_f07_channels.py`

```sh
'../project persona/.venv/bin/python' scripts/run_grain_gptpro_f07_information.py --output results/grain_gptpro_foundation_v3/F07/new_information
'../project persona/.venv/bin/python' scripts/audit_grain_gptpro_f07_information.py --input results/grain_gptpro_foundation_v3/F07/new_information --output results/grain_gptpro_foundation_v3/F07/new_information_audit
```

Still open: exact MAP source/validation/application simulations under these categorical laws, full finite-sample bias/coverage, F08 measured synthetic-reader confusion matrices, and original task/DAG integration. This mathematical packet does not replace those experiments. Real response costs or human reader error rates remain future H04 inputs. Overall goal remains ACTIVE; F06 full dynamic computation continues independently.

No API, neural training, private raw text, MEPS fitting, remote execution, lockbox, commit or push occurred.
