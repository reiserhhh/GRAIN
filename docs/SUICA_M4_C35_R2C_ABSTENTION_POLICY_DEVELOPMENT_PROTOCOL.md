# GRAIN M4-C.3.5-R2C Abstention-Aware Policy Development Protocol

## Status

R2C is a fresh-seed development study. It is not a confirmation and cannot
produce `GO`. It follows the R2B result
`M4_C35_R2B_PARTIAL_CONDITIONAL_EFFECT`, whose only failed registered gate was
zero native RCCA refusal.

No maximum abstention rate is registered here. This study may estimate an
appropriate future budget, but may not choose a budget and confirm it on the
same outcomes.

## Scientific object

For each pre-response condition cell \(i\), freeze

\[
A_i=\mathbf 1\{\text{the RCCA applicability check accepts cell }i\}.
\]

The routed chart is

\[
\Pi_i=A_iR_i+(1-A_i)B0_i.
\]

`R` is the response-safe RCCA chart and `B0` is the old response-safe chart.
Acceptance uses only pre-response support evidence. A refused cell remains in
the primary analysis and receives `B0`; it is never deleted.

The primary eligible-world estimand is the full planned-population policy
gain

\[
\Delta_\Pi
=
\mathbb E\left[
  \Gamma_{\Pi}-\Gamma_{B0}
\right]
=
\mathbb E\left[
  A(\Gamma_R-\Gamma_{B0})
\right].
\]

The accepted-stratum forced-`R` gain is secondary and cannot replace
\(\Delta_\Pi\).

## Frozen strata

Eligible effect worlds:

- `endogenous_creation_expansion`
- `endogenous_source_partition_matched`
- `source_rotated_feedback`

History safety sentinel:

- `history_gated_ecology`

Compensation boundary:

- `selection_creation_compensation`

Null worlds:

- `linear_null_ecology`
- `linear_exogenous_selection`
- `fast_return_equal_marginal`

## Fresh development sampling

- Root seed: `2603644629`
- Author-permutation seed: `613234704`
- CKA-permutation seed: `303676477`
- Raw bootstrap seed: `683248717`
- Policy bootstrap seed: `2806419682`
- Independent repetitions: `2`
- Null draws: one per null world and repetition
- Cluster bootstrap draws: `1000`
- Bootstrap unit: complete repetition

These seeds were fixed before opening any R2C response or oracle endpoint and
do not occur in R2 or R2B.

## Phase separation and seal

Phase A:

1. Generate only sanitized pre-response condition tensors.
2. Fit and serialize `B0`, `Br_var`, `Br_rep`, and `R`.
3. Record no response, endpoint, author mechanism truth, oracle chart, or
   geodesic truth.
4. Seal config, this protocol, runtime, source, every basis bundle, and the
   complete Stage-A manifest.
5. Reconstruct the RCCA applicability profile from pre-response tensors,
   verify exact Stage-A basis replay, freeze every \(A_i\), and seal a second
   policy manifest.

Phase B:

1. Require both frozen manifest hashes.
2. Verify config, protocol, source, runtime, pre-response digests, bundles,
   and exact policy identity.
3. Open response and oracle truth only after both manifests exist.
4. Score `Pi` as exact `R` for accepted cells and exact `B0` for refused cells.

Any replay or identity failure invalidates the development run.

## Development diagnostics

The primary report includes:

- full-population eligible policy gain and cluster LCB
- policy recovery of oracle headroom
- policy efficiency relative to the same-estimator oracle
- per-world policy gain and acceptance rate
- history-sentinel safety
- compensation-boundary `Pi-Oest` fidelity
- hazard degradation
- null false-success rate
- acceptance rate by registered stratum
- accepted-stratum forced-`R` gain as a secondary diagnostic
- refused-stratum forced-`R` gain and `Pi-R` routing gain, to test whether the
  refusal rule actually avoids harmful forced use
- exact policy and chart-basis replay errors

Candidate thresholds reproduce the R2B effect and safety scale. They are
design diagnostics only. Passing them does not establish confirmation.

## Claim boundary

This study may establish only that an entirely pre-response RCCA/B0 routing
policy is mathematically executable and characterize its finite-synthetic
development behavior. It does not establish:

- an externally justified abstention budget
- confirmation of policy value
- universal chart replacement
- natural-text transport
- personality, behavioral, or clinical validity
- M4-D readiness
