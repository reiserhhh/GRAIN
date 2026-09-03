# GRAIN M4-C.3.3 Opportunity-Excitation Information Protocol

## Status

This protocol and its source hashes were frozen after implementation smoke
tests and before opening the eight-repetition endpoint. A one-repetition
runtime benchmark is development-only and excluded from inference.

## Question

M4-C.3.2 established a positive Fisher-Wiener loop gain but recovered only
25.4% of oracle creation headroom. M4-C.3.3 asks whether the remaining
headroom is limited by observable creation information rather than by another
condition chart, hazard family, or author mechanism.

## Fixed objects

The following do not vary across frontier arms:

- response-safe discovered condition chart;
- Fisher-Wiener estimator and `epsilon_scale=1e-6`;
- 16-author mechanism population;
- physical response and choice edges from the `K=1` passive anchor;
- maximum-passive oracle endpoint;
- five main synthetic worlds;
- naturally generated evaluation paths; and
- author-relation Spearman geometry.

No rank, kernel, bandwidth, model, or endpoint is selected after opening.

## Information grid

\[
K\in\{1,2,4,8\},
\qquad
I\in\{\text{passive},\text{orthogonal excitation}\}.
\]

The nested calibration/selection occasion pairs are:

```text
K=1:  5 / 2
K=2: 10 / 4
K=4: 20 / 8
K=8: 40 / 16
```

Every occasion contains 120 events. The maximum trajectory is generated once
per world/repetition. Lower budgets are literal author-wise occasion prefixes,
not separately sampled datasets.

## Designed excitation

Natural response scale is estimated from calibration/selection paths before
evaluation. The intervention cycles without additional random draws through:

\[
\{+e_1,-e_1,+e_2,-e_2\}
\]

at amplitude one reference response standard deviation. It enters the
response state inside the path generator:

\[
r_{uot}^{exc}=r_{uot}^{natural}+s_{ot}.
\]

The same planted creation law regenerates the next opportunity menu, and the
intervened state propagates through later calibration/selection events.
Evaluation paths are copied unchanged from the passive natural process.
Therefore this is a designed creation-input intervention, not a covariate
rewrite with stale outcomes.

## Information statistic

For the fixed feedback block:

\[
x_{uokt}=\widehat\phi(c_k)\otimes r_{uot},
\]

\[
\mathcal I_{u,K,I}
=
\sum p_{uokt}(1-p_{uokt})
x_{uokt}x_{uokt}^{\top}.
\]

\(J_{K,I}\) is the median author minimum positive eigenvalue of
\(\mathcal I\). Trace, effective rank, and condition number are secondary.

## Primary endpoints

\[
\Gamma_{K,I}
=
\rho_S\left(
\operatorname{pdist}(\widehat L^{FW}_{K,I}),
\operatorname{pdist}(L^\star)
\right),
\]

\[
\eta_{K,I}
=
\frac{
\Gamma_{K,I}-\Gamma^D_{K,I}
}{
\Gamma^{O\text{-swap}}-\Gamma^D_{K,I}
}.
\]

The dose slope is estimated within each world/repetition after centering
\(\Gamma\) and \(\log J\); repetition is the bootstrap cluster. The endpoint
contrast is `K=8 excitation` minus `K=1 passive`.

## Frozen gates

- information ratio at least `4`, with repetition-cluster LCB above `1`;
- geometry-on-log-information slope LCB above `0`;
- endpoint geometry gain at least `.03`, with clustered LCB above `0`;
- high-information headroom recovery at least 50%;
- high-information geometry at least `.70`;
- at least `6/8` repetitions with positive Fisher gain and geometry `.70`;
- held-out hazard-loss degradation no greater than 2%;
- split-half author-permutation gain no greater than `.01`;
- no-creation false success no greater than `.05`;
- gauge difference no greater than `1e-6`;
- direct common-shift pair-distance error no greater than `1e-10`; and
- truth-open condition-alias information-loss detection at least `.95`.

No-creation controls run only the high-information endpoint; a dose curve is
unnecessary for their false-success test.

## Deterministic sharding

Repetitions `0-7` may run as four non-overlapping two-repetition shards. Seeds
are functions of the absolute repetition number. Aggregation refuses missing
or duplicate metric cells and recomputes all gates from concatenated rows.

## Stop rule

If information rises by at least four-fold but the dose slope is not positive,
or the high-information endpoint still recovers less than 50% headroom, the
claim “more observable opportunity is sufficient” is rejected. Remaining
error is routed to chart bias, hazard misspecification, or heterogeneous
author mechanisms. Further kernel/embedding replacement remains stopped.

## Frozen hashes

```text
97dca978632d767cebb0aa92f4881d6d6d351e4fe67349e8ba29d17e3a1d51a2  configs/m4_opportunity_excitation_frontier.json
adbc6529a95524a766a07151ea29ccca91d7c1e9149c6ea661800060e94bcc8e  scripts/run_suica_m4_opportunity_excitation_frontier.py
caea51aa78ece44d44272bfd35dd012d4a361e72ed5321c9417c89b7c9de97cc  scripts/aggregate_suica_m4_opportunity_excitation_frontier.py
a355f951013754a725c54050b7856c5c87f2f1d3d17bb607f3df5bacb32537a6  suica_core/m4_opportunity_excitation.py
01174c4b6e936fd958e59200a2ce2bd3b8860c3101cd2e88502bd5f5b752f5d4  suica_core/m4_creation_information.py
a7efa3b7aa0f9244b1d3cea3901f677030aef4001bbec2d5676ea20b237e5c0c  suica_core/m4_fisher_wiener_creation.py
6b6dde15bea661bfd8660f83014b3e60457cd9a6f6e6a926d41ca20f81c6a1e3  tests/test_m4_opportunity_excitation.py
ef9d912da735885b07aede1d06d6f7bd90a7ea4e89631a6d627167799b3ca1d9  tests/test_m4_creation_information.py
6ca10bf175b841d85aec79fbf40df6733467aeecda882f53279eb980b45b6da6  tests/test_m4_opportunity_excitation_frontier.py
```

## Opened result

All eight absolute repetitions were aggregated after the four deterministic
shards completed. Every frozen source hash above remained byte-identical.
The opened decision is:

```text
M4_C33_NO_GO_INFORMATION_LIMIT
```

The observable information intervention worked in the narrow causal sense:

- endpoint information ratio: `13.0378`, clustered LCB `12.4051`;
- geometry-on-log-information slope: `.02935`, LCB `.02223`;
- `K=8 excitation - K=1 passive` geometry: `+.08022`, LCB `.06111`;
- high-information geometry: `.75232`;
- high-information Fisher gain: `.11375`;
- held-out hazard degradation: `.148%`;
- permutation gain: `-.38260`;
- null false success: `0`; and
- gauge, common-shift, and truth-open alias controls passed.

It was not sufficient for the preregistered closure:

- recovered oracle headroom was `.44026`, below `.50`; and
- only `5/8` repetitions had positive gain and geometry at least `.70`,
  below `6/8`.

The post-hoc decomposition preserves that frozen verdict. Passive opportunity
growth from `K=1` to `K=8` contributed `+.0611`
[`.0364`, `.0824`] geometry. Orthogonal excitation at `K=8` added a smaller
`+.0192` [`.0033`, `.0372`]. Mean geometry peaked at `K=4 excitation`
(`.7585`) rather than continuing upward. World-specific recovered headroom
ranged from `-.2157` in `history_gated_ecology` to `.8104` in
`selection_creation_compensation`. Therefore more observable opportunity is
useful but not a uniform repair; the residual is routed to mechanism family,
latent history, chart bias, or author heterogeneity.

Artifacts:

```text
70a896d72d9a60034e2a7c49a511013be277f2fbbe8662dd0be28c2175e9c0a2  results/m4_opportunity_excitation_frontier/metrics.csv
1d434c45d88b6bf0e34f5543ae4431f6067cc8a0c814575fc2adaa58d025d3bc  results/m4_opportunity_excitation_frontier/decision.json
b4353f500e05ebbed7ae2a404e14a4807e3ec7d6ad30060262541e6733622164  results/m4_opportunity_excitation_frontier/alias_audit.csv
ebd6005edcd21cbfce2da7519d5f66c4dd6c2a6a91a33017b749df78dbab3b0f  results/m4_opportunity_excitation_frontier/mechanism_decomposition.json
```

The frozen aggregator wrote the metrics and decision before its report-only
Pandas 3 groupby call rejected a mixed serialized column. The frozen source
was not changed. `scripts/render_suica_m4_opportunity_excitation_report.py`
coerces only the six displayed report columns and renders from the already
written aggregate artifacts; it does not recompute any gate or estimand.

## Boundary

This protocol can identify only a finite synthetic creation-information
frontier. It cannot identify personality, validate natural text, reopen
M4-C.2, or authorize M4-D.
