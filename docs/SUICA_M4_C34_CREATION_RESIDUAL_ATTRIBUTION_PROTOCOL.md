# GRAIN M4-C.3.4 Creation Residual Attribution Protocol

## Status

This protocol and the source hashes below were frozen after target-aligned
unit/smoke tests and one development-only runtime benchmark, but before
opening the eight-repetition endpoint. The earlier gate-averaged prototype
was rejected before freezing because it changed the C3.3 estimand; none of its
endpoint values are evidence. The frozen endpoint has now been opened once;
its result is recorded below without changing the protocol or hashes.

## Question

C3.3 showed that observable creation information is useful but insufficient.
C3.4 asks whether the remaining relation-loop loss is attributable to:

1. condition-chart distortion (`C`);
2. a misspecified creation observation likelihood (`S`); or
3. cross-author Fisher-Wiener pooling bias (`P`).

## Fixed primary estimand

Every cell estimates the same gate-zero direct creation derivative:

\[
A^0_{u,kd}
=
\left.
\frac{\partial}{\partial r_d}
P(G_{u,t+1,k}=1\mid r,g=0,z_0)
\right|_{r=0},
\]

where \(g=1\{H_{t,0}>0\}\) and \(z_0\) is the original C3.3 zero reference
for current-generated state, duration, and other nuisance inputs.

\[
L^0_{u,csp}=\widehat A^0_{u,csp}R_u^DD_u^D,
\]

\[
\Gamma^0_{csp}
=
\rho_S\left(
\operatorname{pdist}(L^0_{csp}),
\operatorname{pdist}(L^{\star,0})
\right),
\]

\[
q^0_{csp}
=
\frac{\Gamma^0_{csp}-\Gamma^0_{000}}
{\Gamma^{0,O\text{-swap}}-\Gamma^0_{000}}.
\]

All eight cells share the same target, frozen discovered response/choice
edges, oracle-swap ceiling, and denominator. Gate-rate averaging is forbidden
because it mixes the operator with an exposure distribution.

## Three-factor cube

### Chart (`C`)

- `C=0`: response-safe discovered chart.
- `C=1`: truth-open oracle chart, used only for attribution.

The oracle chart cannot become a deployable method.

### Observation likelihood (`S`)

- `S=0`: the current joint gate logistic. All history strata share nuisance
  terms and direct response coefficients; gate-response interactions are
  fitted jointly. The output remains the gate-zero derivative.
- `S=1`: target-aligned history-stratified likelihood. Gate-zero and gate-one
  strata receive separate intercept, return, duration, and feedback
  coefficients. Only the gate-zero derivative enters the primary loop;
  gate-one information is a sufficiency diagnostic.

When observed generated/external sources are mutually exclusive and at least
80% of authors have nondegenerate gate-zero source data, `S=1` uses a
two-part hurdle model. With \(Z=E_{t+1}\lor G_{t+1}\):

\[
P(G=1\mid r,g=0,z_0)=\pi_0(r)q_0(r),
\]

\[
A^{0,S=1}
=
q_0(0)\,\partial_r\pi_0(0)
+\pi_0(0)\,\partial_rq_0(0).
\]

An empirical risk-rate multiplier is not permitted.

### Pooling (`P`)

- `P=0`: leave-one-author-out split-half Fisher-Wiener pooling.
- `P=1`: author-local high-information fit.

For `S=1`, pooling touches only response blocks required by the gate-zero
primary estimand. In hurdle worlds, risk and source-allocation response blocks
are shrunk separately. Gate-one coefficients do not enter the primary
shrinker.

## Frozen worlds and roles

- five main worlds from C3.3;
- target observation-law worlds:
  `endogenous_source_partition_matched` and `history_gated_ecology`;
- three no-creation/null worlds;
- 16 mechanism authors;
- `K=8 excitation`, 40 calibration and 16 selection occasions;
- 120 events per occasion;
- naturally generated evaluation paths;
- the C3.3 `K=1 passive` discovered response/choice anchor; and
- eight fresh absolute repetitions.

The three non-target main worlds enforce noninferiority.

## Attribution statistics

The eight \(\Gamma^0_{csp}\) values receive a complete Möbius decomposition.
Shapley values allocate \(\Gamma^0_{111}-\Gamma^0_{000}\) across `C/S/P`.
This is a relation-geometry value function; the terms are not additive
operator errors.

The `S` main effect is:

\[
\Delta_S
=
\frac{1}{4}
\sum_{c,p}
\left(
\Gamma^0_{c,1,p}-\Gamma^0_{c,0,p}
\right).
\]

Repetition is the bootstrap cluster.

## Frozen gates

- oracle headroom at least `.05`, clustered LCB above zero;
- joint stratified/hurdle information full rank in at least 95% of
  author-cells;
- gate-zero source-at-risk validity at least `.80`;
- mean `S` effect across the two target worlds at least `.03`, clustered LCB
  above zero, and positive in at least `6/8` repetitions;
- full diagnostic cell recovered headroom at least `.80`;
- full recovered headroom at least `.50` in each target world;
- mean full-cell degradation no worse than `.01` in the other three worlds;
- leading Shapley factor margin clustered LCB above zero;
- comparable generated-hazard loss degradation no greater than 2%;
- split-half author-permutation gain no greater than `.01`;
- no-creation false success no greater than `.05`;
- gauge difference no greater than `1e-6`;
- direct common-shift pair-distance error no greater than `1e-10`; and
- truth-open condition-alias information-loss detection at least `.95`.

## Stop rule

If full-cell recovered headroom is below `.80`, the `C/S/P` decomposition is
declared incomplete and all `K>8`, kernel, embedding, and excitation expansion
remain stopped. If the `S` gate fails, the claim that omitted history/source
observation law is the dominant residual explanation is rejected. Only a
single factor whose Shapley lead clears the frozen margin may enter a fresh
confirmation; discovery itself cannot reopen M4-C.2 or authorize M4-D.

## Deterministic sharding

Repetitions `0-7` may run as four non-overlapping two-repetition shards.
Seeds depend on the absolute repetition. Aggregation refuses missing or
duplicate metric, decomposition, control, or alias cells and recomputes every
gate from the concatenated rows.

## Frozen hashes

```text
b747450881fc2f8b384ebe77c9c54ace6cab19e348931767ffefd78a0a4ab20f  configs/m4_creation_residual_attribution.json
3976d4bd5e041e972a43bf956f3a7655c9fdcfab81b38850c253108b5674f724  scripts/run_suica_m4_creation_residual_attribution.py
2db0cb54fc29e3884d52352985d7bf9827f784226c4694220412fef58d25f37b  scripts/aggregate_suica_m4_creation_residual_attribution.py
8b00584be16f148e0bcd4db742025dd864e6ffd6e4d016db0a609688e9779088  suica_core/m4_creation_residual_attribution.py
a7efa3b7aa0f9244b1d3cea3901f677030aef4001bbec2d5676ea20b237e5c0c  suica_core/m4_fisher_wiener_creation.py
a355f951013754a725c54050b7856c5c87f2f1d3d17bb607f3df5bacb32537a6  suica_core/m4_opportunity_excitation.py
71e8768d52e5c9700f6367768f5708756ac036988e515740b57221afee58cfb9  tests/test_m4_creation_residual_attribution.py
5dbf6ac8405cb668ff03a46022f4ceec16e5476869241562409ed099f506489c  tests/test_m4_creation_residual_aggregation.py
```

## Frozen result

The eight-repetition endpoint was opened once and aggregated from four
non-overlapping shards. The frozen decision is:

```text
M4_C34_NO_GO_THREE_FACTOR_INCOMPLETE
```

The oracle swap exposed mean headroom `.11244` with clustered LCB `.07407`.
The three factors did not jointly close that headroom:

| cell or allocation | result |
| --- | ---: |
| discovered baseline geometry | `.77493` |
| `C`-only geometry | `.86324` |
| `C`-only global recovered headroom | `.78538` |
| full `C+S+P` geometry | `.81322` |
| full global recovered headroom | `.34055` |
| Shapley `C` | `+.07268` |
| Shapley `S` | `-.04336` |
| Shapley `P` | `+.00897` |
| `C` leading-margin clustered LCB | `+.03956` |

`S` failed as an explanation rather than merely failing to help. Across the
two target worlds its mean effect was `-.08441` with clustered LCB `-.15813`,
and it was positive in `0/8` repetitions. The full-cell recovered headroom was
`.24071` in `endogenous_source_partition_matched` and `-2.75241` in
`history_gated_ecology`. Thus a separately fitted history/source observation
law is not the dominant omitted mechanism for the fixed gate-zero estimand.

`P` was small, whereas the truth-open `C` factor was the only factor with a
separated positive Shapley contribution. This localizes most recoverable loss
to condition-chart distortion, but it does not provide a deployable repair:
the `C=1` chart reads truth and is forbidden outside attribution. Moreover,
the truth-open alias audit detected information loss in `7/8` repetitions
(`.875 < .95`), so chart-alias sensitivity is not yet uniform.

The information-rank, source-at-risk, hazard, author-permutation,
no-creation, gauge, and common-shift controls passed. The signed mean
full-cell gain in the three non-target worlds was `+.06652`, satisfying their
noninferiority rule. These controls make the negative `S` result interpretable
but do not override the failed full-recovery, target-recovery, and alias gates.

## Routing consequence

The `C` lead clears the protocol's single-factor routing requirement. A fresh
experiment may therefore study a **response-safe, truth-blind condition-chart
estimator** against the frozen `C` attribution ceiling. It may not reuse the
oracle chart as a predictor, reopen isotropic RBF smoothing, add more `K`,
kernel, embedding, or excitation searches, or advance M4-D. Its first burden
is to recover the oracle chart's author-relation benefit while passing a
stronger alias battery on unseen worlds. The bounded design candidate is
specified in `SUICA_M4_C35_RESPONSE_SAFE_EIV_CHART_DESIGN.md`.

## Boundary

This is a finite synthetic residual-attribution experiment. It cannot identify
personality, validate natural text, license an augmented gate/source operator,
close M4-C.2, or authorize M4-D.
