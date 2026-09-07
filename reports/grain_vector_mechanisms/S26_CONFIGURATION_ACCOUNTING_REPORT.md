# S26: the observed N drop is mainly variance dilution, not erased signal

2026-09-08. COMPLETE,213authors/three views. Zero new neural calls, no fitting
or sign choice,0.81seconds numerical analysis after source validation. Code and
plan e5d51e2 were committed before S25 outcomes were analyzed.

## Finding

The new full head-configuration displacement contains the original pairing
component. Applying the same frozen functional gradient to their exact retained
sum shows that most source covariance survives, while additional variation
reduces the standardized correlation.

| Component | Primary189 N r | Bootstrap95%CI | All213 N r |
|---|---:|---|---:|
| Full configuration-head score | .116028 | [-.042374,.253217] | .110449 |
| Pairing score | .199990 | [.061117,.330554] | .174895 |
| Remaining score | -.006002 | [-.166754,.139391] | .000390 |

Primary remaining p=.93468; all213 p=.99548. These estimates place the added
scalar component near zero N association in this sample. They do not establish
independence, a universal zero effect, or irrelevance to other traits.

## Exact account

For native weights p and values V, d_pair=sum_G sum_i(p_i-P_G/|G|)V_i.
With Delta_h=h_native-h_projected, retain r=Delta_h-d_pair. Equivalently,
r is the native group-mean value reading plus outside-body reading minus the
actual projected head. Thus z_config=z_pair+z_remaining; no component is removed.

Primary189 source covariances:

.00005873447 = .00006105542 - .00000232094.

The full configuration score retains96.20% of pairing's signed source covariance,
but its variance is2.74935times larger. Specifically,

Var(config)=2.5223292e-6,
Var(pair)=9.1742746e-7,
Var(remaining)=1.4721100e-6,
Cov(pair,remaining)=6.6395905e-8.

These satisfy Var(config)=Var(pair)+Var(remaining)+2Cov(pair,remaining).
The observed Pearson ratio is exactly

r_config/r_pair = [Cov(Y,config)/Cov(Y,pair)] * sqrt[Var(pair)/Var(config)]
                = .961986 * sqrt(1/2.749350) = .580168.

This accounts for.116028/.199990 without choosing a regression coefficient.
It is a descriptive identity for this sample, not a significance test showing
the methods differ in the population. S25's paired difference CI spans zero.
All213 source covariance similarly gives.00005509497=.00005494256+.00000015241.
Largest source-covariance identity error5.0e-20; variance identityerror6.4e-22.

## What the additional vector contains

The remaining scalar is reproducible across the three random text views:
primary pairwise r=.38798/.38836/.39130. That is higher than the original
pairing scores' .10036/.04804/.25739, despite its near-zero N association.
Therefore stability alone is not a criterion-specific signal selector.

In native Euclidean coordinates, primary RMS vector norms are
configuration1.17817, pairing.65908, remaining1.19800; mean pair/remaining cosine
is-.30145. Yet their scalar covariance under the frozen functional reader is
small and positive. Coordinate-level cancellation and source-prediction
covariance cancellation are different quantities. Do not label the larger or
more stable vector either pure noise or a newly established personality factor.

## Decision

Keep the paired relational readout as the useful N route. The larger full head
configuration is not a replacement. Retain remaining structure separately for
future trait-specific explanations; do not erase it from GRAIN globally.

The actual full-model S9 C still carries source association through more than
this head. Next trace when its effect becomes carried by the final query state
across subsequent decoder boundaries. That closes the missing source pathway
more directly than perfecting this head's already accurate local approximation.
Normalization-order math is ready as a bounded explanatory candidate, but defer
a large normalization grid while the later-query transport question is open.

## Reproduction

Output: results/grain_vector_mechanisms/configuration_accounting_s26_r1/.
Includes frozen retained vectors/scores, source connections, metrics, view
consistency and exact accounts. Vector addition error5.55e-17, S25 score replay0,
S20 score replay9.99e-17. Run scripts/run_grain_configuration_accounting.py with
parent venv Python and a new --output-dir after complete S25. Private artifacts
remain ignored.94focused tests pass; no model training or protected data access.
