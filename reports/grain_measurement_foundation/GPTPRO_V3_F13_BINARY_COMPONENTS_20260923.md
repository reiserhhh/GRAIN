# F13 next-probability versus next-outcome uncertainty

## Mathematical object

For a fixed source transition table p_s and current working history-state weights
w_s, the state-specific next probability P=p_S has mean mu=sum_s w_s p_s and

    V_probability = sum_s w_s (p_s-mu)^2
    V_outcome_given_state = sum_s w_s p_s(1-p_s)
    Var(Y_next) = V_probability + V_outcome_given_state = mu(1-mu).

These are different uncertainty objects. A known probability .5 and an unknown
probability equally likely0 or1 have identical next-outcome variance .25, but
probability variance0 versus.25. Binary predictive variance cannot establish
precision of a probability estimate.

The implementation also reports central discrete .025/.975 quantiles of P.
They are conditional finite-state intervals, not confidence intervals for the
unknown deterministic predictive mean or human psychological traits. Learned
reader weights define a working law; this decomposition does not calibrate it.

## Actual cached-data execution

Reused `F08/nli_downstream_full_r1`: two worlds, three seeds, complete/mixed
panels and oracle/soft/hard channels. Six source bundles cold-replayed against
source observations, then all388,800 B1 prediction records reproduced exactly.
No new reader, model fit, API or synthetic text generation. The source simulator
was deterministically regenerated only for numeric source-bundle verification.

Variance identity maximum discrepancy2.23e-16. Every prediction identity and
ordering checked against the original saved prediction stream. Output includes
both validation/application roles; the following descriptive averages use only
application authors (events are not treated as independent author samples).

| World/panel/channel | History-state probability variance | Conditional outcome variance |
|---|---:|---:|
| XOR/complete/oracle | 0 | .188147 |
| XOR/complete/soft | .000347 | .188147 |
| XOR/mixed/oracle | .039097 | .188134 |
| XOR/mixed/soft | .039891 | .188232 |
| IID/mixed/oracle | .0000264 | .249902 |
| IID/mixed/soft | .0000248 | .249915 |

For a sequential XOR process, unqualified observations leave meaningful
uncertainty about recent states and hence next probability. For an IID process,
knowing those states contributes almost nothing. Even the oracle channel does
not reveal unqualified latent observations. A zero complete/oracle state component
does not imply zero uncertainty in the learned source transition probabilities.

## Correction and evidence

Initial r1 failed with KeyError(world): the old observation schema stores source
world/seed in its simulator event address, not separate columns. The revision
uses the same explicit routing convention as the existing F08 audit. Addresses
select source bundles only; no ID/seed enters the scorer. Failed r1 remains.

Canonical results: `results/grain_gptpro_foundation_v3/F13/binary_components_r2`.
Core: `grain_research/gptpro_foundation_v3/f13_binary.py`.
Runner: `scripts/run_grain_gptpro_f13_binary_components.py`.
Three new tests validate exact joint enumeration, same-mean/different-uncertainty
counterexample and invalid weights.27 related tests and Ruff pass.

## Remaining work

This closes fixed-source B1 variance object definition and cache integration,
not the full F08 uncertainty task. Source-author bootstrap, personal-parameter
uncertainty in B2/M, reader calibration uncertainty and full interval evaluation
remain. No Beta posterior is fabricated from fractional pseudo-counts. The
F05 source/personal long runs continue separately; do not restart them.
Full F13 and the overall100-card goal remain open. No private-text/MEPS access,
neural training, lockbox, commit or push.
