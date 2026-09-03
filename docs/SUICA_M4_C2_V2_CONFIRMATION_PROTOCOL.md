# GRAIN M4-C.2 V2 Confirmation Protocol

## Status

Frozen before the fresh-seed V2 confirmation run.

The V1 development result remains `M4_C2_NO_GO_IDENTIFICATION`. V2 is a new
estimand and cannot retroactively change that result.

## V1 failure and V2 repairs

V1 exposed two design defects:

1. The selection/creation pair matched expected menu rates but not realized
   union-menu paths.
2. Condition alias used an absolute predictive-skill threshold instead of
   measuring information lost between an oracle state and the observable
   quotient chart.

V2 replaces the matched creation arm with
`endogenous_source_partition_matched`. Selection and source-partition arms
share candidate conditions, environments, and the exact union-menu path.
Source partition changes only the external/generated mark inside that fixed
envelope. A separate `endogenous_creation_expansion` world retains the claim
that generation can expand the total opportunity set.

Condition alias is now a truth-open synthetic information-loss audit:

\[
S_\chi =
\frac{1}{2}S_{\chi,\mathrm{choice}}
+\frac{1}{2}S_{\chi,\mathrm{hazard}},
\qquad
\Delta_{\mathrm{alias}} = S_O-S_D,
\qquad
R_{\mathrm{retain}} = S_D/S_O.
\]

It passes only when:

\[
S_O\ge .50,\qquad
\Delta_{\mathrm{alias}}\ge .20,\qquad
R_{\mathrm{retain}}\le .70,\qquad
\mathrm{LCB}_{95\%}(\Delta_{\mathrm{alias}})>0.
\]

This is not an operational alias detector. Without oracle state, the system
may report an underresolved observed mechanism but cannot attribute the cause
to condition alias.

## Frozen gates

- Exact union-menu, environment, and candidate-condition equality: maximum
  difference `0`.
- Union-menu and entropy MMD: at most `1e-10`.
- Outside-option gap: at most `.03`.
- Selected-condition total variation: diagnostic only, because it carries the
  selection effect.
- Fast/slow equal-marginal MMD and outside gap: at most `.03`.
- Oracle mechanism macro-F1: at least `.90`.
- Discovered mechanism macro-F1: at least `.85`; drop from oracle at most
  `.08`.
- Active support and sign recovery: at least `.85`.
- Choice, creation, and loop action geometry: at least `.70`.
- Return and recovery geometry: at least `.75`.
- History-gate margin: at least `.55`.
- Alias and source tests: at least `.95`; this makes condition alias an `8/8`
  requirement.
- Null false-positive rate: at most `.05`.
- Basis and response invariance: exactly `1`.
- At least `6/8` repetitions must pass every substantive gate.

## Development-only power check

Twenty full-size development seeds, disjoint from the V2 confirmation bank,
gave alias oracle skill:

- median: `.5794`
- mean: `.5838`
- minimum: `.5027`
- proportion at or above `.50`: `20/20`

These values justify freezing the DGP but are not confirmation evidence.

## Frozen hashes

```text
2515e141cb1497fba4581814b02c87c16023a2585d5b9a5c3e6b8be3e093595c  configs/m4_chart_ecology.json
cb70935b704cfe1ec37d75c426236bfa7e40f3bd8fdffbdf9458f5533afb6e65  scripts/run_suica_m4_chart_ecology.py
e76ef8e10d88d6c413ba04877b52dac00de5576885d2baeb0702e4fab797fb43  suica_core/m4_chart_ecology_generator.py
ae58be2cbf6707fb76f78ddbfc89df47f2fb6ff5395c4041737e48e55d4bf93c  suica_core/m4_chart_ecology_estimator.py
d6d09c6e80510179f56eee584c0f8d1a4653f5a11183d06a89bb668f024a9cb7  suica_core/m4_chart_ecology_audit.py
```

No threshold, DGP strength, or estimator setting may change after the
confirmation run begins. A failure remains a V2 failure.
