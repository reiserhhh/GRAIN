# GRAIN M4-C.3.5-R2B Conditional Response-Safe Chart Protocol

## Status

This protocol is frozen before any R2B confirmation response or oracle
endpoint is opened. R2B is a new estimand. The two R2 development repetitions
are design evidence only and are excluded from every confirmation interval.

The original R2 universal-replacement proposition is closed as
`M4_C35_R2_PARTIAL_CHART_EFFECT`: its `world_consistency` gate failed because
the selection-creation compensation sentinel had mean `R-B0 = -0.00662` and
two-repetition cluster LCB `-0.03395`, below the frozen `-0.01` boundary.
R2B does not weaken that boundary or reuse the old decision name.

## Scientific question

Let

```text
Delta_R(w) = Gamma_R(w) - Gamma_B0(w)
H_O(w)     = Gamma_oracle-swap(w) - Gamma_B0(w)
F_C(w)     = Gamma_R(w) - Gamma_Oest(w)
```

`B0` is the old response-safe chart. `R` is the pre-response RCCA chart.
`Oest` uses the truth-open oracle chart with the same downstream estimator.
The oracle swap is a scoring denominator only.

R2B asks three different questions rather than forcing one sign constraint:

1. In accessible-effect worlds, does `R` recover positive chart headroom?
2. In the history-gated sentinel, is `R` non-inferior to `B0`?
3. In the selection-creation compensation boundary, is `R` non-inferior to
   `Oest`, even when a more faithful chart need not outscore `B0` after
   opposing mechanisms cancel?

## Frozen worlds

Accessible-effect worlds:

- `endogenous_creation_expansion`
- `endogenous_source_partition_matched`
- `source_rotated_feedback`

Safety sentinel:

- `history_gated_ecology`

Structural boundary:

- `selection_creation_compensation`

Null worlds:

- `linear_null_ecology`
- `linear_exogenous_selection`
- `fast_return_equal_marginal`

Each null world has five independent draws per repetition. Eight repetitions
therefore contain 40 main cells and 120 null cells.

## Frozen sampling and inference

- Root seed: `2223719102`
- Author-permutation seed: `537424274`
- CKA-permutation seed: `2839251173`
- Bootstrap seed: `3120088265`
- Independent repetitions: `8`
- CKA permutations per main cell: `199`
- Cluster bootstrap draws: `5000`
- Bootstrap unit: complete repetition
- Ratio draws with non-positive denominator are undefined, not clipped
- Minimum valid ratio-bootstrap fraction: `.975`

## Phase separation

Phase A:

1. Generate only the pre-response condition view.
2. Fit and serialize `B0`, `Br_var`, `Br_rep`, and `R`.
3. Serialize no ecology response, endpoint, author mechanism parameter, oracle
   basis, or geodesic truth.
4. Seal the config, protocol, numerical runtime, source tree, every basis
   bundle, and the complete Stage-A manifest.

Phase B:

1. Require the preregistered Stage-A manifest SHA-256.
2. Verify config, protocol, R1 decision, source, runtime, pre-response digest,
   and bundle hashes before scoring.
3. Open response and oracle truth only after all eight repetitions are sealed.
4. Permit deterministic repetition shards, but aggregate exactly one complete
   `0..7` repetition set.

Any hash, digest, coverage, leakage, or alias failure makes the confirmation
invalid. It cannot be repaired in place.

## Confirmation gates

Accessible effect:

- Mean eligible `Delta_R >= .05`, cluster LCB `> 0`
- Mean recovery `>= .60`, LCB `>= .35`
- Mean accessible efficiency `>= .80`, LCB `>= .50`
- Eligible `R-Oest` LCB `>= -.01`
- Every eligible world has headroom LCB `> 0`
- Every eligible world has recovery LCB `>= .30`
- Every eligible world has at least `7/8` positive repetitions
- Aggregate eligible gain is positive in at least `7/8` repetitions

Controls:

- `R-Br_var` and `R-Br_rep` gain LCBs are both `> 0`
- CKA gain LCBs are `> 0`
- Every main-cell CKA permutation `p <= .01`
- History-sentinel `Delta_R` LCB `>= -.01`
- Compensation-boundary `F_C` LCB `>= -.01`
- Hazard relative-degradation cluster UCB `<= .02`
- Author-permutation gain `<= .01`
- Native RCCA refusal rate `= 0`
- Source-shuffle zero-rank/refusal rate `>= .95`
- Support-shift refusal rate `= 1`
- Latent-alias false-recovery rate `= 0`
- Same-rank/same-trace contract pass rate `= 1`
- Gauge error `<= 1e-6`
- Common-shift error `<= 1e-10`

Null specificity:

- A null cell is a false success when `R-B0 >= .03`
- Wilson 95% upper bound must be `<= .05`
- With 120 null cells, at most one false success is permitted

All gates are conjunctive. Failure closes this estimand under its registered
name. Seeds, worlds, endpoints, thresholds, or arms cannot be changed after
opening Phase B.

## R1 semantic replay audit

R1 was originally frozen with generator SHA-256
`e76ef8e10d88d6c413ba04877b52dac00de5576885d2baeb0702e4fab797fb43`.
The current generator adds a standalone Phase-A pre-response entry point and
has SHA-256
`2d023bfbe26e7b87cdd68e93fd22ae27835e3ba5676eee4fec6617e9d055d19a`.

The complete R1 confirmation was replayed with the current generator and the
original frozen config. The original and replayed artifacts are byte-identical:

```text
metrics.csv      e87af3371be2cb6b6be786dd438ab127d9b4cf89454737b1d8f549f49bc2a0c1
controls.csv     59492603a98bc8c2e64179a796be78f5d18f8ebb0671a07f2f4a6ae05eed83cd
alias_audit.csv  fcd131d83721ed1bc956787378898ea55ab1a7d4ddafa4d45e976879283f6ff3
decision.json    32f5991d26049d79594eabba650f1478efd97f57b8e67b1e4124b2d4c8648b68
```

This licenses the source change as a semantics-preserving extension of the R1
world generator. R2B receives a new source seal and does not reuse the old
generator hash.

## Frozen pre-Phase-A hashes

```text
382f0aa5de306d716927e2da4dc21158822c54145f502c3b6c7133539e72b2fd  configs/m4_response_safe_chart_replacement.r2b_confirmation.json
2d023bfbe26e7b87cdd68e93fd22ae27835e3ba5676eee4fec6617e9d055d19a  suica_core/m4_chart_ecology_generator.py
df0da358a67391d84ae10324e591cc56487a17e81677b6cc0e134519f165d4f9  suica_core/m4_response_safe_chart_bundle.py
a28913a326505cbf889eed42d115f3c30fa976f59835332a82019bacfe3f1a72  suica_core/m4_response_safe_chart_replacement.py
0ea35d0748c8ebdde8ba7a767c5f40759f2f967d7379d1e67feb72d5d26f8967  suica_core/m4_response_safe_rcca_chart.py
f166c14f1bd4a630b873bfdbeea0a6b68db2eb2797b6b6dd7fb7f4a0e812045a  scripts/prepare_suica_m4_response_safe_chart_replacement.py
39760e807c03184d3b7f21901a74fb1c68134aededd12e60cc7e0b9e9474c2ff  scripts/run_suica_m4_response_safe_chart_replacement.py
7419c0f057792623fa2c3aa871d32ec000d8c151b21c841d1122e057e84ee41a  scripts/run_suica_m4_response_safe_chart_replacement_sealed.py
d2f7b0034f66f5289dfc42c7d64b822f4293260e582cc2c1090c7c0dbb905323  scripts/aggregate_suica_m4_response_safe_chart_replacement.py
```

The Stage-A manifest additionally seals every Python file under
`suica_core/`, the four R2 scripts, runtime versions, and each serialized
chart bundle.

## Claim boundary

A GO would establish only a finite-synthetic conditional response-safe chart
effect: observable repeatability/source alignment identifies a chart that
recovers accessible downstream geometry in three registered worlds, remains
safe in one history sentinel, and tracks the same-estimator oracle chart in
one compensation boundary.

It would not establish:

- universal five-world chart replacement
- causal mechanism recovery
- natural-text transport
- personality, clinical, or behavioral validity
- M4-D readiness
