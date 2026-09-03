# GRAIN M4-G — Objective Redesign Line: registrations and outcomes

Opened 2026-08-03, immediately after the M4-F panel design line closed
(`docs/SUICA_M4_F_PANEL_DESIGN_SYNTHESIS.md`). This document holds every
registration and outcome of the line, in order, append-only. The claims ledger
controls; nothing here may exceed its tier.

## Why this line exists

Two independent arcs converged on the same remaining problem.

- **M4-D Leg 14 → M4-E2**: the discovery objective carries a systematic frame
  displacement whose common offset is DISTRIBUTED (largest single piece is the
  residual at .40–.45), world-specific in direction at the permutation null,
  and not removable by any single term — removing the dominant term DOUBLES
  the gap. Verdict `OFFSET_SPREAD_NO_SINGLE_OBJECTIVE_TERM`. Its registered
  hand-off named the **scale family** as the largest IDENTIFIABLE carrier
  (~1/3 of mass in the registered sequential order, ~1/2 standalone,
  dominated by the `n2` component: the principal column-scale modes
  `{A v_i v_i^T}` — first-order motion of the consensus under re-scaling its
  principal column directions, i.e. **the freeze whitening's unregularized
  1/sqrt(eig) amplification**). The supervised block is irrelevant. M4-E2 also
  disclosed that its own sub-40% clause is knife-edged and order-sensitive:
  the honest reading is "no term above ~half the mass under any ordering".
- **M4-F1..F9**: the panel side is closed. A finite panel can be made
  self-consistent and cannot be made trait-certifying. If a trait-level object
  is reachable at all, the remaining lever is the objective, not the data.

M4-E2 DECOMPOSED the offset. This line INTERVENES on it.

## Standing rules inherited from M4-F (all four apply to every leg here)

1. Designed-null and decorrelation-style gates state their aggregation rule
   and multiplicity treatment BEFORE the run.
2. Every leg pre-states the scale at which its target is measurably non-zero
   (citing the prior leg's persisted numbers) and the minimum detectable
   effect its design affords; a null at the target's noise floor is reported
   UNDERPOWERED, never as a null.
3. Every leg verifies from the generator/operator that its manipulation has a
   NON-ZERO causal channel at every tested parameter value; a null on an inert
   knob is reported VACUOUS, never as a null.
4. Every gate bounds MATERIALITY through an equivalence form — the confidence
   interval must lie inside a margin justified from the generator or the
   effect size — never nil significance on a known-nonzero quantity.

---

## M4-G1 registration (2026-08-03, BEFORE run) — is the scale family an actionable lever, or only a carrier?

**Question.** M4-E2 showed the whitening's amplification family carries the
largest identifiable share of the offset. Carrying mass is not the same as
being fixable. Does REGULARIZING that amplification actually reduce the
discovery objective's frame displacement — and if it does, does the reduction
transfer to truth, or does it merely improve the objective's self-consistency
while leaving (or worsening) what the objective recovers?

The second half is not optional. This program has twice measured
self-consistency and truth moving in opposite directions: Leg 12 (a cleaning
intervention improved every loop statistic while lowering truth-referenced
attribution .7264 → .6677) and M4-F5 (the gauge rose 63x while trait-like
recovery plateaued). An offset reduction that does not transfer is a cosmetic
result and must be reported as one.

**Design.** Reuse Leg 14 / M4-E2's persisted world and objective path exactly;
the ONLY thing that varies is the whitening applied inside the discovery
objective:

- `baseline` — the deployed unregularized whitening (1/sqrt(eig)). Must
  reproduce the persisted offset and gap (gate G1).
- `shrinkage` — 1/sqrt(eig + lambda) over a registered ladder
  lambda / median(eig) in {0.01, 0.1, 1.0}.
- `truncated` — drop the smallest-eigenvalue directions (precisely those the
  1/sqrt(eig) map amplifies most) over a registered ladder retaining
  {90%, 75%, 50%} of spectral mass.
- `identity` — no whitening at all: the extreme control that distinguishes
  "regularize the amplification" from "whitening is the problem".

Same worlds, same repetitions, and the same paired-by-world statistics M4-E2
used, so arms are compared within world.

**Metrics, both mandatory at every arm.**
1. OFFSET/GAP — M4-E2's own persisted objective metric, unchanged.
2. TRUTH-REFERENCED RECOVERY — computed by the identical path on the world's
   known structure, in the M4-F5 style (register the operationalization in
   Part 0 before compute; report at least two variants with their sensitivity,
   since this is the leg's main researcher degree of freedom).

**Leans.**
(a) ACTIONABLE: at least one shrinkage or truncation setting reduces the
    offset by >= 25% relative to baseline, paired-by-world CI excluding zero.
(b) IT TRANSFERS: at the setting that minimizes the offset, truth-referenced
    recovery IMPROVES over baseline, paired CI excluding zero — under BOTH
    truth variants, or the lean MISSES.
(c) NOT SIMPLY "WHITENING BAD": the `identity` arm is worse than the best
    regularized arm on the offset metric.

**PIVOT-IF:** no arm reduces the offset by >= 25% with a paired CI excluding
zero -> THE SCALE FAMILY IS EXONERATED AS A LEVER. It carries mass without
being actionable, the redesign target moves to the residual block (.40–.45 of
the mass, the largest single piece), and the next registered question becomes
the objective's FUNCTIONAL FORM rather than its scaling.

A second, separately recorded outcome: if (a) HOLDS but (b) MISSES, the
finding is that the scale family is a **cosmetic** lever — offset reduction
without truth transfer — which is a result about the objective metric itself
and must be reported as prominently as a success would have been.

**Gates.**
- **G0 POWER** — pre-state the baseline offset/gap level from M4-E2's
  persisted artifacts and the minimum detectable paired difference this design
  affords. If the realized paired CI half-width exceeds 12.5% of the baseline
  offset (half of lean (a)'s 25% bar), the leg is UNDERPOWERED and adjudicates
  nothing.
- **G1 ANCHOR** — the `baseline` arm reproduces M4-E2 / Leg 14's persisted
  offset and gap to <= 1e-12 on identical world seeds.
- **G2 CHANNEL LIVENESS** — for every non-baseline arm, demonstrate the
  whitening operator actually moved: report the spectrum or condition number
  of the applied whitening beside baseline's, with an equivalence-form
  statement of what counts as "materially different". An arm that does not
  move the operator is INERT and its result is reported VACUOUS, not as a null.
- **G3 TRUTH-PATH INVARIANCE** — the truth-referenced path must be the
  identical objective path differing only in its input; demonstrate with a
  degenerate equality check (M4-F5 and M4-F9 both achieved exactly 0.0).
- **G4 MATERIALITY FORM** — every gate in this leg is stated as an
  equivalence/margin bound; no gate may be a nil-significance test on a
  known-nonzero quantity. State this compliance explicitly.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_g1_whitening_intervention/`; report
`reports/SUICA_M4_G1_WHITENING_INTERVENTION_REPORT.md`.

## M4-G1 outcome (2026-08-03, appended)

**PIVOT DOES NOT FIRE. Lean (a) HOLDS, lean (b) MISSES decisively under both
truth variants, lean (c) MISSES. Verdict:
`COSMETIC_LEVER_OFFSET_WITHOUT_TRANSFER`.** G0 and G2 first, as the standing
rules require: G0 POWER is a mixed picture (3 of 6 candidate arms
underpowered at the n=3-world replication ceiling) but lean (a)'s HOLD rests
on `truncated_90` alone, which IS adequately powered (half-width 98.1% of
the 12.5%-of-baseline-offset bar, 52.8% reduction, paired 95% CI [5.28,
8.46]) — not on underpowered noise. G2 CHANNEL LIVENESS passes for every one
of the seven non-baseline arms by a wide margin (smallest relative
condition-number change 43.3%, vs a 10% materiality bar). G1 ANCHOR and G3
TRUTH-PATH INVARIANCE both pass at floating-point-epsilon / exact 0.0.

At the arm the registration's own rule selects for lean (b) — `truncated_50`,
the candidate with the single lowest mean offset (5.19, a 59.9% reduction) —
truth-referenced recovery does not transfer, it collapses: `e_arm_true` rises
from baseline's ~0.567 to ~0.928 at BOTH truth-recovery budgets (4x and 8x
events, regenerated from the frozen world law per Leg 4 Part 4b's own
machinery), paired 95% CI entirely on the "worse" side at both
(n=384 author-reps each). Lean (c) MISSES outright: `identity` (zero
amplification) has the single LOWEST offset of all eight arms tested,
including every regularized candidate — not merely "not worse," an
inversion of the registered expectation.

A disclosed, non-gating companion computed identically for every arm
(paired, author-level, both budgets) sharpens the finding into a
dose-response: `shrinkage_0.01` and `shrinkage_0.1` — the two mildest
settings, offset reductions 10.9% and 24.9% — show small but statistically
real truth-recovery IMPROVEMENTS (CIs entirely positive at both budgets).
`shrinkage_0.1` misses the registered 25% actionable bar by 0.14 percentage
points (24.86%), a knife-edge disclosure in the same spirit as M4-E2's own
sub-40% clause. `shrinkage_1.0` (42.1% reduction) is ambiguous (CI includes
zero). Every arm at or past 50%+ reduction (the three truncated arms,
identity) shows large, decisive worsening. Spearman(offset, recovery error)
across the 7 non-baseline arms = −0.786 (descriptive, n=7, not a registered
test): lower offset associates with WORSE recovery, not better. **The scale
family carries mass and is a real, adequately-powered lever on the offset
metric — but only its mildest, sub-actionable settings preserve or help
truth; every setting that clears the registered "actionable" bar makes
recovery worse, and the setting with the single lowest offset (identity) is
the second-worst for recovery of all eight arms tried.**

Full numbers, gates, per-arm table, and the dose-response table:
`reports/SUICA_M4_G1_WHITENING_INTERVENTION_REPORT.md`. Artifacts:
`results/m4_g1_whitening_intervention/{decision.json, gates.json,
offset_rows.csv, gap_rows.csv, truth_recovery_rows.csv,
g2_spectrum_evidence.csv, g3_check_rows.csv}`.

**Hand-off.** Neither the PIVOT's clean exoneration nor a usable estimator
fix is licensed: the scale family plainly IS an offset-metric lever (lean a),
and is plainly NOT a truth-transferring one past its mildest settings (lean
b). The open question this leaves for the next registration in this line is
sharply framed by the dose-response table: whether a NARROWER shrinkage
ladder concentrated between this leg's "real but sub-bar" (~0.01–0.1) and
"ambiguous" (~1.0) points can find a setting that is both actionable (>=25%
offset reduction) and truth-transferring, or whether — as the monotone
Spearman pattern suggests — those two properties are in this objective's
structure fundamentally in tension, in which case the redesign target moves,
as the PIVOT branch would have directed, to the residual block or the
objective's functional form.

## M4-G1 planner adjudication note (2026-08-03, appended) — the target metric is now the suspect

The leg is adjudicated as the agent reported it: lean (a) HOLD on
`truncated_90` (the one arm that clears 25% AND is adequately powered,
52.8% reduction, paired CI [5.28, 8.46]), lean (b) MISS decisively under both
truth variants, lean (c) MISS by inversion, pivot does not fire, verdict
`COSMETIC_LEVER_OFFSET_WITHOUT_TRANSFER`.

**But the arm table says something sharper than "cosmetic", and it points at
the metric rather than at the lever:**

| arm | offset | width | truth error @4x |
|---|---|---|---|
| baseline | 12.999 | 13 | 0.5562 |
| shrinkage_0.01 | 11.571 | 13 | 0.5512 |
| shrinkage_0.1 | 9.755 | 13 | 0.5539 |
| shrinkage_1.0 | 7.503 | 13 | 0.5751 |
| truncated_90 | 6.129 | 4 | 0.8801 |
| truncated_75 | 5.572 | 3 | 0.9085 |
| truncated_50 | 5.190 | 2 | 0.9574 |
| **identity** | **4.352** | 13 | 0.8656 |

The offset falls MONOTONICALLY as the whitening is weakened, and the single
lowest offset in the whole table belongs to `identity` — no whitening at all —
whose truth error is 56% worse than baseline's. Across the seven non-baseline
arms the offset and the recovery error are anti-correlated at Spearman −.786.
**Minimizing the registered target makes the objective worse at its job.**

Two mechanisms could produce that, and they have very different consequences:

1. **Units.** `offset_norm` is a norm in a space whose metric the whitening
   itself sets. Weakening the whitening shrinks every distance in that space,
   so the measured offset falls without anything improving. If so, the metric
   cannot be compared across arms with different whitening scales — and
   M4-E2's attribution of ~1/3–1/2 of the offset mass to the scale family may
   itself be a units statement rather than a structural one.
2. **A real trade-off.** The displacement genuinely trades against recovery,
   and the objective cannot have both.

There is also a plain **width confound** sitting on top of either: the
truncated arms carry width 4/3/2 against baseline's 13, and a norm over fewer
dimensions is mechanically smaller.

The two mechanisms are cheaply separable and the next leg separates them. Note
what is at stake: under mechanism 1, this line's stated target — and part of
the prior leg's decomposition — is disqualified as an optimization objective.

One more thing the leg found that the registered leans could not score, and
which survives either mechanism: the two MILDEST arms (shrinkage .01 and .1)
are the only ones whose truth recovery genuinely IMPROVES, both with CIs
entirely on the better side — and shrinkage_0.1 misses the registered 25%
actionable bar by 0.14 points (24.86%). The bar selected exactly the arms that
hurt. That is recorded here as a registration critique, not as a rescue: no
lean is re-scored on it.

## M4-G2 registration (2026-08-03, BEFORE run) — is the offset metric measuring the world or its own units?

**Question.** Does `offset_norm` inherit the whitening's units (and the
retained width), or is it scale-invariant? The answer decides whether M4-G1's
anti-correlation — and part of M4-E2's decomposition — is an artifact or a
finding.

**Design, in two registered diagnostics.**

- **D1, the pure-scale ladder (decisive).** Take the deployed baseline
  whitening W and form arms `c*W` for c in {0.25, 0.5, 1.0, 2.0, 4.0}. This
  changes NOTHING structural: identical eigenvectors, identical relative
  spectrum, identical condition number, identical width. It is a pure change
  of units. Measure `offset_norm` and the truth-referenced recovery at every
  c, on the M4-G1 world/objective path, unchanged.
- **D2, the width companion.** Report `offset_norm / sqrt(width)` alongside
  the raw value for all eight M4-G1 arms, to separate the dimension-count
  effect from the scale effect. Declared as a diagnostic, not a new target.

**Leans.**
(a) UNITS: the log-log slope of `offset_norm` against c has a CI excluding 0
    (report the point estimate and state plainly whether the CI contains 1).
(b) TRUTH IS SCALE-FREE: truth-referenced recovery is invariant across the c
    ladder — equivalence form, all pairwise differences inside +/- 0.02,
    a margin justified as ~4% of baseline truth error (.556).
(c) NORMALIZATION REPAIRS THE ORDERING: under a scale-normalized offset
    (registered in Part 0 before compute, together with its justification),
    the Spearman correlation between offset and truth error across M4-G1's
    eight arms is no longer negative, and `identity` is no longer the minimum.

**PIVOT-IF:** the log-log slope CI includes 0 -> THE METRIC IS SCALE-INVARIANT.
M4-G1's anti-correlation is then a real property of the objective: reducing
displacement genuinely costs recovery, M4-E2's scale-family attribution stands
as structural, and the redesign faces a true trade-off rather than a units bug
— which becomes the line's next registered question and a substantially harder
one.

**Gates.**
- **G0 POWER** — M4-E2's anchored world count is only 3 (M4-G1 reported 3 of
  6 candidate arms underpowered at that ceiling). Run the c ladder on MORE
  worlds (target 8) since it is a fresh measurement, and state explicitly
  which comparisons are E2-anchored (3 worlds) and which are fresh (8). Report
  the realized MDE; underpowered comparisons adjudicate nothing.
- **G1 ANCHOR** — c=1.0 reproduces M4-G1's persisted `baseline` offset and
  truth recovery to <= 1e-12 on identical world seeds.
- **G2 CHANNEL LIVENESS** — show that each c actually changes the applied
  whitening operator while leaving eigenvectors, relative spectrum, condition
  number and width unchanged; this is what makes D1 a pure-units manipulation
  rather than a structural one. Report all four invariances explicitly.
- **G3 TRUTH-PATH INVARIANCE** — degenerate equality check, as in M4-G1.
- **G4 MATERIALITY FORM** — every gate an equivalence/margin bound; state
  compliance per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_g2_metric_units/`; report
`reports/SUICA_M4_G2_METRIC_UNITS_REPORT.md`.

## M4-G2 outcome (2026-08-03, appended)

**PIVOT DOES NOT FIRE. The log-log slope of `offset_norm` against a pure
whitening-scale multiplier c is 0.8796, paired-by-world (n=8, fresh) 95% CI
[0.8386, 0.9206] — materially nonzero, adequately powered (half-width 8.2%
of the 0.5 bar), and sub-linear (the CI excludes both 0 AND 1). Per the
registration's own branch: `offset_norm` is DISQUALIFIED as an optimization
target in its raw form.** Lean (a) HOLDS. Lean (b) MISSES decisively — truth
recovery is not scale-free; it improves monotonically and precisely as c
rises (error .786→.422 at budget=4x across the ladder on the 6 numerically-
valid worlds, 18 of 20 pairwise checks decisively outside the ±0.02
equivalence margin, up to 15x). Lean (c) HOLDS decisively: under the
registered scale-normalized offset (offset divided by the geometric mean of
an arm's own per-direction scale factors), the Spearman correlation between
offset and M4-G1's truth error flips from M4-G1's disclosed −0.786 to
**+0.833** (identical at both truth budgets, no disagreement), and
`identity` — M4-G1's spuriously "best" arm on raw offset — becomes the
single WORST-scoring arm once normalized, now consistent with its actual
poor truth-recovery. G0 primary (the slope CI, the decisive test) is
emphatically not underpowered. G1 ANCHOR and G3 TRUTH-PATH INVARIANCE both
pass at 0.0. G2 shows all four registered invariances (eigenvectors,
relative spectrum, condition number, width) hold to floating-point
exactness within every context while the whitening operator's Frobenius
norm scales with c exactly — D1 is confirmed pure units, not structure.
Verdict: **`METRIC_UNITS_DISQUALIFIED`.**

**Two world-selection corrections and one statistical-validity correction
were made mid-compute, all disclosed in full with both readings given, none
altering lean (a)/pivot's already-finalized numbers.** (1) `leg3._world_seed`'s
own pre-existing `matched_groups` table shares seeds across
`{linear_exogenous_selection, endogenous_source_partition_matched}` and
`{fast_return_equal_marginal, slow_hysteresis_equal_marginal}`; both pairs
produced bit-identical (0.0 diff) `offset_norm`/G2 evidence despite genuinely
different dynamic mechanisms — a deterministic, outcome-blind exclusion of
each pair's second member was applied BEFORE any hypothesis-relevant number
existed. (2) The replacement `linear_exogenous_selection` was then found
degenerate (`D_true` below `FLIP_TOLERANCE`) on all 256 of its
(rep,view,author) combinations — G3 inapplicable — and was replaced by
`topology_mismatch` (checked non-degenerate from the archive before running
it), again before any hypothesis-relevant number existed for the world in
question. (3) AFTER lean (a)/pivot were finalized, lean (b)'s naive
all-8-world computation produced nonsense means (10^8-10^9 scale); the
arm-invariant diagnostic `e_orc_true` showed two worlds
(`linear_null_ecology`, `fast_return_equal_marginal`) with median ~5e9 vs
0.18-0.74 in the other six — a pre-existing `_relative_error`
near-zero-denominator fragility on these two low/null-signal worlds' small
`D_true` at the regenerated truth budgets, unrelated to the c-ladder (their
OFFSET data remained sane throughout). Lean (b) was recomputed on the valid
6-world subset (adopted) with both readings persisted; the MISS verdict
holds under both.

**Consequence for M4-E2's scale-family attribution, stated precisely as the
registration requires**: M4-E2's decomposition was computed once, at a
single fixed operator's own units, and never compared offset magnitudes
across operators of different intrinsic scale — as a description of which
displacement directions dominate AT that one fixed operator, it is not
directly tested or overturned here. What IS undermined — demonstrated
directly by lean (c)'s sign flip from −0.786 to +0.833 — is treating raw
`offset_norm` as comparable, scale-free evidence ACROSS operators whose own
intrinsic scale differs, which is exactly the cross-arm design M4-G1 used
(arms whose geometric-mean-scale spans 32.6x to 1.0x): M4-G1's own headline
"identity has the lowest offset" finding is shown here to be substantially a
units artifact of that specific comparison, not evidence about the arms'
relative structural quality.

Full numbers, gates, the c-ladder table (offset and both truth variants per
c), the per-world log-log regression table, the D2 width-companion table,
and the full lean (b)/(c) tables (both naive and adopted readings):
`reports/SUICA_M4_G2_METRIC_UNITS_REPORT.md`. Artifacts:
`results/m4_g2_metric_units/{decision.json, gates.json, offset_rows.csv,
truth_recovery_rows.csv, g2_invariance_evidence.csv, g3_check_rows.csv,
scale_norm_rows.csv, d1_loglog_per_world_slopes.csv, d2_width_companion.csv,
lean_b_pairwise_equivalence.csv,
lean_b_pairwise_equivalence_naive_all8_worlds.csv,
lean_b_truth_validity_diagnostic.csv}`.

**Hand-off.** The raw discovery objective (`offset_norm`) is disqualified as
an optimization target — M4-G1's own cross-arm comparisons built on it
(including its central "identity is best" finding) inherit this
disqualification. A scale-normalized version (Part 0.1 of this leg) is
demonstrated, on M4-G1's own 8 arms, to repair the ordering into one
directionally consistent with actual truth-recovery quality. The open
question for the next registration in this line is whether that
normalized metric (or a refinement of it) is viable as a REPLACEMENT
optimization target in its own right — checked here only on the 8-arm,
3-world setting M4-G1 already ran, not on a fresh design of its own — and,
separately, whether the genuine (non-units) scale-vs-recovery relationship
this leg found on the pure c-ladder (bigger c: worse raw offset, but BETTER
truth recovery, opposite-signed to M4-G1's own non-uniform-regularization
finding) reflects something about this objective's functional form that a
normalized-offset redesign would need to reconcile.

## M4-G2 planner adjudication note (2026-08-03, appended) — one artifact killed, one lever uncovered

The verdict stands as adjudicated: slope .8796 [.8386, .9206] on a manipulation
whose four invariances were confirmed at exactly 0.0, so **`offset_norm` is
DISQUALIFIED as an optimization target in its raw form**. Lean (c) shows the
repair works: under the pre-registered scale-normalized offset, Spearman with
truth error flips from M4-G1's −.786 to **+.833**, and `identity` moves from
raw-offset minimum to normalized-offset maximum. The width companion rules out
the dimension-count explanation (identity's offset/sqrt(width) = 1.207 is
still the minimum). M4-E2's own decomposition is NOT overturned — it never
varied c — but M4-G1's cross-arm raw-offset comparisons, including its
"identity is best" reading, inherit the disqualification.

**Lean (b)'s MISS is the more interesting half, and it is a finding, not a
failure.** Truth recovery was registered as scale-free and is not: it improves
MONOTONICALLY along the ladder, .786 → .646 → .525 → .451 → .422, with 18 of
20 pairwise CIs outside the ±.02 equivalence band. Since c is provably a pure
change of units — eigenvectors, relative spectrum, condition number and width
all identical to 0.0 — recovery cannot depend on it unless something
DOWNSTREAM of the whitening compares a scale-carrying quantity against a FIXED
ABSOLUTE CONSTANT. Multiplying the whitening by 4 then effectively weakens
that constant, and recovery improves by 20% of its own magnitude.

That is a third instrument finding in this line's short life, and unlike the
first two it points at a repair rather than a retraction: if a fixed constant
is being compared against a quantity whose scale is arbitrary, making it
RELATIVE should deliver the c=4 improvement at c=1 — a real gain rather than a
units trick. The next leg tests exactly that, and is registered to fail loudly
if the scale dependence is not localizable in identifiable constants.

Process note, recorded because it is the standard here: the executing agent
hit three mechanical problems (two colliding world seeds; a degenerate
replacement world; a near-zero-denominator blowup in two worlds' relative
error) and disclosed each with its resolution — the first two fixed before any
hypothesis-relevant number existed, the third found only AFTER lean (a) and
the pivot were finalized, recomputed on the valid 6-world subset with both
readings persisted and the MISS holding under each.

## M4-G3 registration (2026-08-03, BEFORE run) — turn the units artifact into a repair

**Question.** M4-G2 proved something downstream of the whitening compares a
scale-carrying quantity against a fixed absolute constant, because a pure unit
change moved truth recovery .786 → .422. Can that dependence be localized in
identifiable constants, and does making them SCALE-ADAPTIVE deliver the c=4
recovery at c=1?

**Part 0 audit (before any compute, and gated).** Enumerate every constant in
the objective path that is compared against, added to, or thresholds a
scale-carrying quantity — regularizers, convergence tolerances, rank or
support thresholds, floors, clip bounds. Publish the inventory in the report
with file and line references. This inventory IS the leg's hypothesis space;
constants discovered later may be reported but may not be scored.

**Design.** Reuse M4-G2's world set (8 worlds, the valid-subset rule from its
lean (b) carried over verbatim) and objective path.

- `baseline` — c=1.0 as deployed (anchor).
- `c4_reference` — c=4.0, the improvement to be matched (also an anchor: must
  reproduce M4-G2's persisted c=4 recovery to <=1e-12).
- `adaptive_<k>` — one arm per inventoried constant, with that constant alone
  made relative to a declared data-scale statistic (registered in Part 0),
  everything else at c=1.0.
- `adaptive_all` — every inventoried constant made relative simultaneously.

**Leans.**
(a) LOCALIZABLE: at least one single-constant `adaptive_<k>` arm recovers
    >= 50% of the c=4 truth-recovery gain at c=1 (gain measured as
    baseline_error − c4_error, per budget), paired-by-world CI excluding zero.
(b) DEFINITIONAL CHECK: the winning adaptive arm's truth recovery is invariant
    across the c ladder {0.25, 1.0, 4.0} — equivalence form, all pairwise
    differences inside +/- .02. This is what distinguishes a genuine
    scale-adaptive repair from a lucky reparameterization.
(c) NO NEW COSMETIC TRADE: the scale-normalized offset at the winning arm does
    not worsen relative to baseline (equivalence form, margin registered in
    Part 0).

**PIVOT-IF:** no single-constant arm reaches 50% of the gain AND `adaptive_all`
also fails to -> the scale dependence is NOT localized in identifiable
constants. The redesign target then moves to the objective's FUNCTIONAL FORM,
and this line's remaining hypothesis (that the displacement problem is a
scaling problem at all) is retired.

**Gates.**
- **G0 POWER** — pre-state the c=4 gain from M4-G2's persisted artifacts and
  the design's MDE; underpowered comparisons adjudicate nothing.
- **G1 ANCHOR** — `baseline` and `c4_reference` reproduce M4-G2's persisted
  values to <=1e-12 on identical world seeds.
- **G2 CONSTANT LIVENESS** — for every inventoried constant, demonstrate it
  actually carries scale dependence: perturb it alone at c=1 and show truth
  recovery moves, in equivalence form against a registered margin. A constant
  that does not move recovery is INERT, is reported as such, and its adaptive
  arm's null is VACUOUS rather than a null.
- **G3 TRUTH-PATH INVARIANCE** — degenerate equality check, as in G1/G2.
- **G4 MATERIALITY FORM** — every gate an equivalence/margin bound; state
  compliance per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_g3_scale_adaptive/`; report
`reports/SUICA_M4_G3_SCALE_ADAPTIVE_REPORT.md`.

## M4-G3 outcome (2026-08-03, appended)

**This outcome lands outside every registered branch, and is reported as
such — neither HOLD, nor a clean MISS, nor PIVOT FIRES.** Part 0's audit
found six Category A constants on the truth-recovery call graph
(`hazard_ridge`, an IRLS convergence tolerance, an IRLS weight floor, a
logit clip bound, a finite-difference probe epsilon, and the basis's own
literal intercept column) plus two Category B constants provably confined
to the offset/GPA path alone (excluded from adaptive arms by call-graph
inspection, not by empirical test). **`hazard_ridge` is confirmed, on
four converging lines of evidence, as the (or the dominant) IDENTIFIABLE,
LOCALIZED source of M4-G2's proven scale dependence**: (i) a provable a
priori argument (an L2 penalty of the form `ridge·n·I` breaks the exact
diagonal-reparameterization invariance unregularized IRLS otherwise has —
textbook reason ridge implementations standardize features first); (ii)
the adaptive ridge value (`hazard_ridge_deployed · mean(raw retained
eigenvalues)`) is consistent and directionally stable across all 64
sampled contexts (0.00023–0.00089, always 4.6–18% of the deployed 0.005,
never crossing zero); (iii) its well-powered author-level companion CI
(n≈745, the SAME grain lean (b) itself uses) decisively excludes zero,
recovering 69–72% of the c=4 gain at both budgets ([0.0399,0.0628]@4x,
[0.0488,0.0727]@8x); (iv) the other five constants are cleanly,
adequately-powered nulls (`tolerance`/`weight_floor`/`clip_bound`/
`probe_epsilon`, CIs within ~1e-4 of exactly [0,0]) or a small real
negative (`intercept`), at the identical grain, ruling out "everything
moves a little" as an alternative story. **But `hazard_ridge`'s own
REGISTERED test — paired-by-world (n=6) — is severely underpowered: half-
width ≈0.08 against a 25%-of-gain bar of ≈0.02, a >4x gap, disclosed in G0
from M4-G2's own persisted numbers BEFORE any adaptive-arm compute ran**
(the same design also leaves M4-G2's own raw c=4 gain barely/not
significant at this grain — not a property invented for this leg). Per
the standing rule that a null at the noise floor is reported UNDERPOWERED,
lean (a) is UNDERPOWERED for `hazard_ridge`/`adaptive_all` (not MISS), and
the **PIVOT is therefore UNDERPOWERED, not FIRES and not DOES-NOT-FIRE** —
FIRES requires every arm to be a *clean* MISS, and an underpowered
comparison cannot supply that. **Separately, and decisively (well-powered,
n=745, all 3 pairwise c-comparisons × 2 budgets, every CI outside ±0.02):
LEAN (b) MISSES.** The adaptive-ridge formula does not deliver invariant
recovery across c∈{0.25,1,4} — it produces an inverted-U, worst at c=0.25,
best at c=1.0 (its own calibration point), worse again at c=4.0 (though
still better than c=0.25): mean diffs +0.084/+0.093 (0.25 vs 1, both
budgets) and −0.034/−0.024 (1 vs 4, both budgets), every CI clear of the
margin by 1.5–5x. Lean (c) HOLDS trivially and exactly: `hazard_ridge`'s
basis is bit-identical to baseline's (0.0 max abs diff, all 8 worlds × 8
reps × 3 roles — verified structurally, not merely argued), so its offset
cannot have moved. G1 ANCHOR passes at exactly 0.0 (8,192 truth rows + 16
offset rows vs M4-G2's persisted `c_1.0`/`c_4.0`). G3 passes at exactly
0.0 (72 checks, all 9 arms × 8 worlds).

**Verdict: `hazard_ridge` is the identified, load-bearing localized
constant, but no formula tested here is CERTIFIED as a genuine, general
repair** — lean (b), the gate explicitly built to distinguish a real
repair from a lucky reparameterization, MISSES decisively and
mechanistically-interpretably (the formula fixes ridge's ABSOLUTE value
from context-level raw eigenvalues alone, not its ratio to each c's own
realized design scale, so it only lands well at its own calibration point,
c=1). The hypothesis that M4-G2's scale dependence is a scaling problem is
**neither retired nor confirmed solved**: `hazard_ridge` is now the
identified target for any future repair attempt, which must register a
c-DEPENDENT (not merely context-dependent) functional form and MUST re-run
lean (b) as its own certification gate — reusing this leg's localization
finding is legitimate; skipping lean (b) would not be.

Full numbers, gates, the Part 0 constant inventory with file/line
references, the per-arm summary table, and the full lean (a)/(b)/(c)
tables: `reports/SUICA_M4_G3_SCALE_ADAPTIVE_REPORT.md`. Artifacts:
`results/m4_g3_scale_adaptive/{decision.json, gates.json,
truth_recovery_rows.csv, author_level_truth_rows.csv, context_meta.csv,
offset_rows.csv, e_orc_true_validity_diagnostic.csv,
lean_a_per_constant.csv, g2_liveness_per_constant.csv,
lean_b_pairwise_equivalence.csv}`.

**Hand-off.** The next registration in this line, if this hypothesis is
pursued further, must do two things this leg deliberately did not do
(doing either after seeing lean (b)'s MISS would be a post-hoc rescue):
(1) register a scale-adaptive ridge formula that is a function of each
ARM's own realized design scale (not only the deployed baseline's raw
eigenvalues), so that it is well-defined and, by construction, testable
for genuine c-invariance rather than single-point calibration; (2)
reconsider the registered statistical grain for truth-recovery gates —
this leg's own G0 and M4-G2's own lean (b) both found the author-level
(n≈745) grain adequately powered where the world-level (n=6) grain was
not, so a future leg's PRIMARY test should either use a finer-grained
pairing (e.g. paired-by-world×repetition) or explicitly register a larger
world set. Absent either, the PIVOT question this leg's own registration
posed remains formally undecided, even though the weight of (non-
adjudicating) evidence leans toward "does not fire."

## M4-G3 planner adjudication note (2026-08-03, appended) — the constant is found; the repair is not certified

**What is established.** The scale dependence M4-G2 proved is localized in ONE
constant: `hazard_ridge` (deployed 0.005,
`suica_core/m4_chart_ecology_estimator.py:341-342`), on four converging lines
of evidence — a priori math, an adaptive value consistent across all 64
contexts at 4.6–18% of deployed, a decisive well-powered author-grain effect
(CI [.040,.063] @4x, [.049,.073] @8x), and clean nulls on the other five
inventoried constants (tolerance, weight floor, clip bound, probe epsilon all
INERT at both grains; intercept small and NEGATIVE). Its adaptive arm recovers
89.5%/95.5% of the c=4 gain at world grain (69.2%/72.3% author grain, CI
excluding zero) at c=1, with the basis bit-identical to baseline — so lean (c)
holds trivially and no cosmetic trade was made.

**Why that is not yet a repair.** Lean (b), the gate registered precisely to
separate a genuine scale-adaptive fix from a lucky reparameterization, MISSES
decisively (n=745, all six pairwise checks outside ±.02). The adaptive arm's
recovery is an inverted-U across c: worst at .25, best at 1.0, worse again at
4.0. The mechanism is visible in the formula: the declared data-scale
statistic is `RAW_SCALE = mean(raw retained eigenvalues)`, which is computed
BEFORE whitening and is therefore c-INVARIANT, while the quantity the ridge
regularizes lives AFTER whitening and scales with c. So the fix is calibrated
at the deployed scale and cannot follow the scale it was built to track. It is
context-adaptive, not scale-adaptive.

**Pivot: outside every registered branch, and correctly reported as such.**
Firing requires every arm to be a clean MISS; `hazard_ridge` and
`adaptive_all` are UNDERPOWERED at the registered world grain, not MISS. The
agent flagged that underpowering from M4-G2's persisted gain BEFORE running
any adaptive arm, and declined to resolve the branch by picking a side.

**Fifth planner registration defect, and its rule.** G0 required a pre-stated
MDE but did not specify the analysis GRAIN, so the line's default (paired by
world, n=6–8) was inherited — and it is structurally underpowered for effects
of this size by more than 4x, while the author grain (n≈745) is adequately
powered and was available all along. **Standing rule (fifth): the analysis
grain must be chosen for power against the registered bar and justified before
the run; inheriting the line's default grain is not a justification.**

## M4-G4 registration (2026-08-03, BEFORE run) — the c-covariant ridge, at a grain that can see it

**Question.** M4-G3 localized the scale dependence in `hazard_ridge` but its
adaptive formula keyed off a pre-whitening statistic and so could not follow
c. Does a ridge keyed off a POST-whitening scale statistic — one that scales
with c by construction — deliver c-invariant recovery without losing the gain?

**Design.** Reuse M4-G3's world set, objective path, arms plumbing and truth
variants. **The registered analysis grain is the AUTHOR grain (n≈745)**, per
the fifth standing rule; world-grain numbers are reported as a companion and
adjudicate nothing.

- `baseline` — deployed ridge at c=1 (anchor to M4-G3's persisted value).
- `g3_raw_scale` — M4-G3's adaptive arm, the known point fix (anchor).
- `covariant_<k>` — one arm per registered post-whitening scale statistic,
  ridge = alpha * stat, with the statistic computed on the whitened quantity
  the ridge is actually added to. Register the candidate statistics AND the
  alpha calibration rule in Part 0 before compute; alpha must be fixed by a
  rule (e.g. matching the deployed ridge at c=1), never tuned on outcomes.
- Each arm evaluated across c in {0.25, 1.0, 4.0}.

**Leans.**
(a) C-COVARIANCE: at least one `covariant_<k>` arm is c-invariant at the
    author grain — all pairwise differences across c inside ±.02.
(b) NO LOSS: that arm's recovery at c=1 is not worse than `g3_raw_scale`'s
    (equivalence form, margin registered in Part 0).
(c) SPECIFICITY: the same covariant rule applied to one of M4-G3's INERT
    constants does NOT produce c-invariance — the control that shows the ridge
    is the mechanism rather than the rescaling being cosmetic anywhere.

**PIVOT-IF:** no covariant candidate achieves c-invariance -> the dependence
is not a simple ridge-scaling issue, M4-G3's localization is incomplete, and
the redesign target moves to the objective's FUNCTIONAL FORM. That is the
consequence M4-G3's own pivot could not deliver, and this leg is registered to
deliver it either way.

**Gates.**
- **G0 POWER** — at the AUTHOR grain, against the c=4 gain persisted by
  M4-G3; state the MDE and the bar before adjudicating. World-grain companion
  numbers may not be used to adjudicate anything.
- **G1 ANCHOR** — `baseline` and `g3_raw_scale` reproduce M4-G3's persisted
  values to <=1e-12 on identical world seeds.
- **G2 COVARIANCE LIVENESS** — for every `covariant_<k>`, demonstrate the
  realized ridge value actually scales with c (report ridge(c=.25),
  ridge(c=1), ridge(c=4) and their ratios). A candidate whose ridge does not
  move with c is by definition not covariant: report it as INERT and its
  result as VACUOUS, not as a null.
- **G3 TRUTH-PATH INVARIANCE** — degenerate equality check.
- **G4 MATERIALITY FORM** — every gate an equivalence/margin bound; state
  compliance per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_g4_covariant_ridge/`; report
`reports/SUICA_M4_G4_COVARIANT_RIDGE_REPORT.md`.

## M4-G4 outcome (2026-08-03, appended)

**The PIVOT FIRES, cleanly — the one thing M4-G3's own pivot attempt could
not do.** Two post-whitening statistics were registered (`covariant_var`,
the direct post-whitening analogue of `RAW_SCALE`; `covariant_msq`, the more
literal uncentered-second-moment reading of "the quantity the ridge is
added to"), both provably homogeneous-degree-2 in c
(`POST_SCALE_k(context,c) = c²·POST_SCALE_k(context,1)`, an exact algebraic
identity from `whitening = c·whitening0`, confirmed empirically to 10+
significant figures in the pooled G2 gate: realized ridge ratios exactly
16.0/0.0625 at c=4/c=0.25 vs c=1, in direct contrast to `g3_raw_scale`'s own
exactly-1.0 ratio, M4-G3's own flaw reproduced as a clean number). At the
registered AUTHOR grain (n=745, well-powered — the decisive comparison's
half-width, 0.0056, is barely over half the 0.01 power bar, not a marginal
call), **both candidates cut the raw c-dependence by 89.1% relative to doing
nothing to `hazard_ridge`** (the specificity control, `weight_floor` treated
by the identical rule, reproduces `baseline`'s own raw swing almost exactly
and is decisively confirmed NOT c-invariant on all six checks — the
reduction is specific to `hazard_ridge`, not a generic artifact) **— but the
residual 10.9% is itself decisively outside the registered ±0.02 margin**
(CI [0.026, 0.037] @4x on the deciding c=0.25-vs-c=4.0 pair, clearing the
margin by 30–36% of its own width, both arms, both budgets). Lean (a)
MISSES cleanly for both arms; lean (c) SPECIFICITY is decisively CONFIRMED;
lean (b) (disclosed companion, officially unreached since lean (a) never
HOLDS) also MISSES decisively — at c=1 the covariant arms sit at
`baseline`'s own worse level, not `g3_raw_scale`'s better one, exactly the
pre-registered mechanistic prediction (alpha calibrated to match deployed
ridge at c=1, not `g3_raw_scale`'s much weaker value). G1 anchor and G3
truth-path invariance both pass at exactly 0.0.

**Per the registration's own consequence for a firing pivot: the dependence
is not a simple ridge-scaling issue, M4-G3's localization is incomplete, and
the redesign target moves to the objective's FUNCTIONAL FORM.** The
mechanism was registered, not discovered post-hoc: Part 0 disclosed, before
compute, that `_hazard_design` mixes columns built from the whitened basis
(which does satisfy the c-scaling identity, and which a covariant ridge can
compensate) with columns built from raw world data
(`generated_current`/`duration`) and from the *unscaled* intercept
sub-column crossed with `response_next` (`feedback_0_d`/`gate_0_d`) — these
do not scale with c and are not compensated by any scalar ridge, however it
is keyed. G2's own numbers make this concrete: the ridge itself is exactly
covariant (16.0/0.0625, floating-point exact) while only 89.1% of the
recovery swing it was meant to fix actually closes — the gap lives
downstream of the ridge's value, in how the design is assembled from
mismatched column types.

**Two mechanical items, both caught and disclosed before any
hypothesis-relevant number was compromised.** (1) An early G1-anchor draft
attempted to independently re-verify `g3_raw_scale`'s c=0.25/4.0 values
against M4-G3's own persisted winner-ladder files — but `g3_raw_scale` is,
by this leg's own registered compute-scope reduction, never computed away
from c=1 in this leg's new compute, so the join correctly found 0 rows and
raised the assembly's own `expected != len(joined)` assertion; fixed by
removing the invalid anchor attempt (this arm's c=1 anchor, the only point
this leg newly computes for it, remains independently verified at exactly
0.0) — caught after all 8 worlds' truth-stage compute had already completed
and persisted, but before it affected any lean, gate, or truth-recovery row.
(2) `covariant_var` and `covariant_msq` turned out numerically
near-indistinguishable throughout (differ in the 3rd–4th decimal) — a
registered open question in Part 0 (whether a role's raw features are
zero-mean relative to `center`), resolved empirically rather than by
assumption, and reported as a real finding rather than suppressed.

Full numbers, gates, the Part 0 statistic/calibration/specificity-control
registration, the per-arm × per-c table, and the full lean (a)/(b)/(c)
tables: `reports/SUICA_M4_G4_COVARIANT_RIDGE_REPORT.md`. Artifacts:
`results/m4_g4_covariant_ridge/{decision.json, gates.json, calibration.json,
truth_recovery_rows.csv, author_level_truth_rows.csv, context_meta.csv,
e_orc_true_validity_diagnostic.csv, g2_ridge_vs_c.csv,
g0_and_lean_pairwise_rows.csv}`.

**Hand-off.** This line's original hypothesis — that M4-G2's scale
dependence is a pure scaling problem, fixable by making one absolute
constant relative to a data-scale statistic — is now closed out at the
functional-form boundary it was always going to meet: `hazard_ridge` is
confirmed, on two independent legs and four converging lines of evidence
(M4-G3) plus a fifth (this leg's specificity control), as the dominant
identifiable channel, and a genuinely covariant ridge closes the large
majority (89.1%) of it — but not all, and the shortfall is structural, not
a matter of finding a better statistic or a better alpha. The next
registration in this line, if pursued, should not test a third
`ridge = alpha · stat` variant (a fourth instance of a functional form this
leg and M4-G3 together have now closed out) but should examine whether a
column-type-aware treatment of the hazard design itself — separately
regularizing or rescaling the `generated_current`/`duration`/
`feedback_0_d`/`gate_0_d` blocks that do not share the whitened basis's own
c-scaling — can close the remaining 10.9%. Absent that, the line's own
honest position is that M4-G2's scale-dependence finding is majority-
explained and majority-repairable by a single constant, with a real,
quantified, structurally-located remainder.

## M4-G4 planner adjudication note (2026-08-03, appended) — the opaque problem now has a named mechanism

The pivot fires cleanly, and unlike M4-G3 it fires on adequately powered
evidence: the twelve comparisons deciding lean (a) all carry half-widths
<= .0056 against a .01 bar. Lean (a) MISSES on the c=.25-vs-4.0 pair
(CI [.026,.037] @4x, clearing the ±.02 margin by 30–36%), lean (b) MISSES,
lean (c) SPECIFICITY is CONFIRMED decisively (the inert-constant control shows
3.2–17.4x larger non-invariance on the same pairs). G2 is exact: the covariant
ridges realize c-ratios of exactly 16.0 and 0.0625 in floating point, matching
the algebra, while `g3_raw_scale`'s ridge ratio is exactly 1.0 — M4-G3's
diagnosed flaw reproduced as a measured number rather than an inference.

**No repair is certified, and the registered consequence stands: the target
moves to the objective's FUNCTIONAL FORM.** But the leg did far more than fire
a pivot. A scalar covariant ridge closes **89.1%** of the raw c-dependence,
and the residual 10.9% has an identified structural cause:

> `_hazard_design` mixes columns of different scale provenance — c-scaled
> columns built from the whitened basis alongside c-invariant ones
> (`generated_current` / `duration`, taken from raw data;
> `feedback_0_d` / `gate_0_d`, crossed with the unscaled intercept).
> **No single scalar ridge can be scale-consistent across a design matrix
> whose columns do not share a scale.**

That is worth stating plainly against where this line started. M4-E2 handed
over "a distributed, world-specific frame displacement, largest single piece
the residual at .40–.45, no single term removable". Four legs later the same
phenomenon reads: the discovery objective regularizes a design matrix with
heterogeneous column scaling using one scalar constant, so its behaviour
inherits the whitening's units and no scalar fix — adaptive or covariant — can
be scale-consistent. The problem moved from a statistical mystery to a named
structural defect with an obvious candidate remedy.

M4-G3's localization is sharpened rather than overturned: `hazard_ridge`
remains the dominant channel; it is simply the wrong SHAPE of knob.

## M4-G5 registration (2026-08-03, BEFORE run) — per-column scaling: does the named defect have the obvious fix?

**Question.** M4-G4 named the mechanism: one scalar regularizer applied to a
design matrix whose columns do not share a scale. Does making the
regularization PER-COLUMN — so each column is regularized in its own units —
complete the repair that scalar covariance could only take to 89.1%?

**Design.** Reuse M4-G4's worlds, objective path, arms plumbing, truth
variants and gate helpers. **Registered analysis grain: AUTHOR (n≈745)**, per
the fifth standing rule; world-grain numbers are companions and adjudicate
nothing. Every arm evaluated across c in {0.25, 1.0, 4.0}.

- `baseline` (anchor), `g3_raw_scale` (anchor: the best point fix at c=1),
  `covariant_var` (anchor: M4-G4's best scalar covariant, 89.1% closed).
- `column_standardized` — standardize each `_hazard_design` column to unit
  scale before the ridge is applied, so the single ridge acts in common units.
- `diagonal_ridge` — a per-column ridge proportional to each column's own
  scale statistic (registered in Part 0), leaving the design untouched.
- A specificity control carried over from M4-G4: the same per-column rule
  applied where it should do nothing.

Register in Part 0, before compute: the column-scale statistic, the
calibration rule fixing overall strength (it must reproduce the deployed ridge
at c=1 in the homogeneous-column limit), and the exact column inventory of
`_hazard_design` with each column's scale provenance (c-scaled vs c-invariant)
stated from the code, with file:line references.

**Leans.**
(a) COMPLETION: at least one per-column arm is c-invariant at the author grain
    — all pairwise differences across c inside ±.02, i.e. it clears the bar
    M4-G4's scalar covariant missed by 30–36%.
(b) NO LOSS: that arm's recovery at c=1 is not worse than `g3_raw_scale`'s
    (equivalence form, margin registered in Part 0).
(c) QUANTITATIVE IMPROVEMENT: the residual c-dependence it leaves is smaller
    than the scalar covariant's 10.9%, reported on the same statistic M4-G4
    used so the two numbers are comparable.

**PIVOT-IF:** no per-column arm achieves c-invariance -> the scale
heterogeneity is NOT confined to the design matrix's columns. The defect is
then deeper than column scaling — in the hazard model's structure rather than
its parameterization — and the line's next question becomes whether the
displacement is IRREDUCIBLE at this level of the objective, which would be a
result about the objective's construction rather than about any knob in it.

**Gates.**
- **G0 POWER** at the AUTHOR grain against M4-G4's persisted effects; state
  the MDE and bar before adjudicating; world-grain companions adjudicate
  nothing.
- **G1 ANCHOR** — `baseline`, `g3_raw_scale` and `covariant_var` reproduce
  M4-G4's persisted values to <=1e-12 on identical world seeds.
- **G2 COLUMN-SCALE LIVENESS** — report the spread of column scales BEFORE
  and AFTER each per-column treatment (e.g. max/min ratio across columns), at
  every c. A treatment that does not materially reduce the spread is not the
  registered manipulation: report it INERT and its result VACUOUS.
- **G3 TRUTH-PATH INVARIANCE** — degenerate equality check.
- **G4 MATERIALITY FORM** — every gate an equivalence/margin bound; state
  compliance per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_g5_per_column_ridge/`; report
`reports/SUICA_M4_G5_PER_COLUMN_RIDGE_REPORT.md`.

## M4-G5 outcome (2026-08-03, appended)

**The PIVOT DOES NOT FIRE. Lean (a) COMPLETION HOLDS decisively for BOTH
per-column arms — not a knife-edge, but floating-point exact.**
`column_standardized`'s recovery is bit-identical (0.0 paired difference,
all 745 authors, both budgets, all three c-pairs) across c ∈ {0.25, 1.0,
4.0}; `diagonal_ridge`'s is invariant to 1–4×10⁻¹⁴ (ordinary floating-point
roundoff). Both are seven-plus orders of magnitude inside the registered
±0.02 margin, at the registered AUTHOR grain (n=745), on comparisons that
carry essentially zero residual uncertainty (paired differences ≈0, so the
standard error itself is ≈0 — the strongest possible non-underpowered case,
not merely an adequately-powered one). **Lean (c) QUANTITATIVE IMPROVEMENT
HOLDS decisively for both arms, at both budgets: the residual c-dependence
M4-G4 left at 10.9% of the raw swing is closed to 0.0% (`column_standardized`,
exact) and ~3–8×10⁻¹² % (`diagonal_ridge`, floating-point zero) — 11–12
orders of magnitude smaller than M4-G4's own residual, on the identical
statistic.** Lean (c) SPECIFICITY is separately, decisively CONFIRMED: the
identical mechanism applied to `weight_floor` (M4-G3-inert) produces large,
non-invariant swings (0.06–0.30, all six checks OUTSIDE the margin,
reproducing M4-G4's own persisted specificity-control numbers exactly) — the
repair is specific to `hazard_ridge`, not a generic artifact. G2 shows both
mechanisms unambiguously LIVE: `column_standardized` collapses the design's
own column-variance spread from 38.8×–908.1× (growing with c, as the c² vs
c⁰ column split predicts) to *exactly* 1.0× at every c; `diagonal_ridge`'s
realized ridge spread exactly tracks that same 38.8×–908.1× heterogeneity,
up from the deployed scalar's perfectly uniform 1.0×. G1 anchor (32,768
rows across 4 anchor arms, including `specificity_weight_floor` at all
three c — an additional disclosed anchor beyond the registration's three
named arms, required by this leg's adopted reading of "carried over from
M4-G4") and G3 truth-path invariance (112 checks) both pass at exactly 0.0.
The column inventory (1,050 columns, 8 worlds × 4 hazard models, classified
by direct empirical comparison of designs built at c=1 vs c=2) confirms the
Part 0.1 analytical claim with zero exceptions: every column of
`_hazard_design` is exactly degree-0 or exactly degree-1 in c, no column
ever mixes both.

**But lean (b) NO LOSS MISSES decisively for both arms.** At c=1, both
arms' recovery is *worse* than `g3_raw_scale`'s by 0.051–0.062 (both
budgets, CI entirely on the worse side) — almost exactly M4-G4's own
persisted c=4 gain (0.059/0.068) — landing within 0.0002–0.0019 of
`baseline`'s own (worse) c=1 level, not `g3_raw_scale`'s (better) one.
**The mechanism is traceable directly to a Part 0 design choice, disclosed
there though its lean (b) consequence was not explicitly forecast**: both
`column_standardized`'s ridge (`ridge_deployed`, unmodified) and
`diagonal_ridge`'s calibration constant (fixed so the homogeneous-column
limit reproduces `ridge_deployed` exactly) are anchored to the *deployed*
value, never to `g3_raw_scale`'s much weaker, specially-fit one — the
identical pattern M4-G4 itself predicted and confirmed for its own scalar
covariant arms, now found, unprompted, to recur in this leg's per-column
arms. Achieving invariance was therefore never, by this registered
construction, also going to recover `g3_raw_scale`'s specific gain.

**This outcome satisfies neither of the registration's two named branches,
stated plainly rather than forced or softened.** "Repair CERTIFIED" requires
lean (a) HOLD *and* no loss — lean (b)'s decisive MISS fails that. "PIVOT
FIRES" requires every arm to MISS lean (a) — neither does, and lean (a)'s
HOLD is not a marginal or underpowered one. The honest, load-bearing
finding: **invariance — the property this whole `M4-G` line has chased
since M4-G1 — is now fully and completely solved, by two independently-derived
mechanisms that agree with each other and with a provable, pre-registered
mathematical argument (Part 0.5) exceeding M4-G4's own base-route-only
guarantee; recovering `g3_raw_scale`'s specific numerical gain is a
distinct, separable, still-open calibration-anchor question, not a further
symptom of the column-heterogeneity defect.** G2's own evidence makes the
separation mechanistic, not merely observed: the ridge/design manipulation
is unambiguously LIVE and the resulting fit is unambiguously invariant,
while the *level* that invariant fit settles at is a direct, provable
consequence of which absolute constant (`ridge_deployed`) the per-column
rule was registered to reproduce in the homogeneous limit.

**Two mechanical items, both caught and fixed during the pre-compute
`--stage inventory` smoke-testing pass, before any hypothesis-relevant
truth-recovery number existed.** (1) `diagonal_ridge`'s naive formula
(`ridge_j = K·var_j`, no fallback) gave `condition_0` — a literal-1 column
structurally identical to the ridge-exempt `intercept` — a ridge of exactly
0, reintroducing the `intercept`/`condition_0` collinearity the deployed
scalar ridge's own uniform nonzero value was incidentally preventing;
`np.linalg.solve` raised `LinAlgError: Singular matrix` on the
`feedback`/`gate` routes. Fixed by falling back to the plain deployed ridge
for any non-exempt, degenerate column, mirroring `column_standardized`'s own
"leave a degenerate column unscaled" rule — making the two arms'
degenerate-column treatment consistent by construction, not by coincidence.
(2) An early diagnostic mis-labeled `_fit_logistic_columnstd`'s first return
value (`beta`, the un-rescaled coefficient in original units, expected to
vary with c) as `gamma` (the standardized-space fit Part 0.5 predicts is
c-invariant), producing an apparently large discrepancy that contradicted
the independently-verified fact that the standardized design itself was
bit-identical across c. Traced to the mislabeling by directly re-deriving
`design_std` equality (0.0 diff) before touching the comparison code; fixed
by adding an explicit `coefficient_std` return value. Neither item touched
any world's actual `--stage truth` compute; all 8 worlds' truth-stage runs
completed with zero Tracebacks, RuntimeErrors, LinAlgErrors, or Killed
signals.

Full numbers, gates, the Part 0 column inventory/statistic/calibration-rule
registration (including the two-mechanism provable-invariance argument),
the per-arm × per-c table, and the full lean (a)/(b)/(c) tables:
`reports/SUICA_M4_G5_PER_COLUMN_RIDGE_REPORT.md`. Artifacts:
`results/m4_g5_per_column_ridge/{decision.json, gates.json,
calibration_source.json, column_inventory.csv,
g2_column_scale_spread_evidence.csv, provable_invariance_spot_check.csv,
truth_recovery_rows.csv, author_level_truth_rows.csv, context_meta.csv,
e_orc_true_validity_diagnostic.csv, g0_and_lean_pairwise_rows.csv}`.

**Hand-off.** This line's central question — is M4-G2's proven scale
dependence a scaling problem at all, and if so, where does it live — is now
closed with a complete, mechanistically-verified answer: it lived entirely
in `_hazard_design`'s mixture of c-scaled and c-invariant columns under one
shared scalar regularizer, and a genuinely per-column treatment (either of
two independently-derived constructions) removes it completely, to
floating-point exactness, confirmed at the row level across 745 authors and
specific to the named constant by a decisive control. What remains open is
narrower and different in kind from anything this line has asked before:
not a structural-defect question but a calibration-selection one — whether
a per-column rule anchored to `g3_raw_scale`'s own (better, but itself only
context-adaptive, not c-covariant, per M4-G3's own diagnosed flaw) level,
rather than to the deployed ridge's homogeneous-limit value, can be both
invariant and at least as good, or whether `g3_raw_scale`'s specific gain
was itself an artifact of its own single-point calibration that a
principled, non-tuned per-column rule has no obligation to reproduce. If
pursued, the next leg in this line should register that calibration
question directly rather than testing a third structural variant of the
per-column family this leg has now closed out.

## M4-G5 planner adjudication note (2026-08-03, appended) — invariance solved exactly; strength is the remaining variable

**Two decisive HOLDs and one decisive MISS, none of them ambiguous.**

Lean (a) HOLDS to machine precision: `column_standardized` is BIT-IDENTICAL
across c (0.0), `diagonal_ridge` invariant to 1–4e-14, both backed by the
pre-registered provable linear-reparameterization argument and both
route-general rather than base-route-limited. Lean (c) HOLDS: M4-G4's residual
10.9% closes to 0.0% and ~1e-12% respectively — eleven to twelve orders of
magnitude. G2 shows the mechanisms are live and enormous (column-scale spread
38.8x/59.6x/908.1x before standardization, 1.0000000000002x after).

The column inventory is itself a result worth keeping: across 1,050 columns,
8 worlds and 4 models, **every column of `_hazard_design` is exactly degree-0
or exactly degree-1 in c — never mixed, zero ambiguous cases.** The
heterogeneity M4-G4 named is therefore exactly binary, which is precisely why
a per-column treatment can remove it exactly rather than approximately.

**So the scale problem this line has chased since M4-G2 is SOLVED, provably
and exactly. And it did not buy the recovery.** Lean (b) MISSES decisively:
both invariant arms sit at BASELINE's recovery level (.5583) rather than
`g3_raw_scale`'s better one (.5068), a gap of .051–.062 that is about the size
of M4-G4's whole c=4 gain.

The reason is visible and was predicted by M4-G4 for its own arms: the
registered calibration for both per-column arms anchors overall strength to
the DEPLOYED ridge, while `g3_raw_scale`'s advantage came from being WEAKER
than deployed (M4-G3 measured its adaptive value at 4.6–18% of deployed). So
the per-column arms inherited deployed's strength — and with it deployed's
recovery — while gaining exact invariance.

**That separates two things this line had been treating as one.** Scale
invariance is a property of the ridge's SHAPE (per-column vs scalar) and is
now solved. Recovery quality is a property of its STRENGTH, and remains
untouched. The registration's two named branches both assumed the repair
would arrive as a single package; it did not, and neither branch applies
cleanly. Recorded as an outside-every-branch outcome, exactly as reported.

## M4-G6 registration (2026-08-03, BEFORE run) — shape and strength together, or a real trade-off

**Question.** M4-G5 solved invariance with the ridge's SHAPE at deployed
STRENGTH. M4-G3 found the recovery gain at a WEAKER strength with the wrong
shape. Does the combination — per-column shape at the weaker strength — give
both, or do invariance and recovery genuinely trade off?

**Design.** Reuse M4-G5's worlds, objective path, arms plumbing, column
inventory, truth variants and gate helpers. **Registered analysis grain:
AUTHOR (n≈745).** Every arm evaluated across c in {0.25, 1.0, 4.0}.

- Anchors: `baseline`, `g3_raw_scale`, `column_standardized` at deployed
  strength (all must reproduce M4-G5's persisted values to <=1e-12).
- `colstd_alpha_<k>` — `column_standardized` at an overall-strength ladder
  alpha/deployed in {0.05, 0.10, 0.20, 0.50, 1.00}, chosen to bracket the
  4.6–18% band M4-G3 measured. The ladder is registered here and may not be
  extended after seeing results; if the optimum sits at an endpoint, say so
  and register any extension as a separate leg.

**Leans.**
(a) INVARIANCE IS STRENGTH-FREE: every `colstd_alpha_<k>` arm remains
    c-invariant within ±.02 (expected to be exact by M4-G5's
    reparameterization argument — this lean tests that the argument does not
    depend on strength).
(b) RECOVERY RECOVERED: at some alpha, recovery is not worse than
    `g3_raw_scale`'s at c=1 (equivalence form, margin registered in Part 0),
    paired CI at the author grain.
(c) TUNING IS ALSO SCALE-FREE: the alpha that minimizes recovery error is the
    SAME at c=0.25, 1.0 and 4.0. This is the sharpest of the three — if the
    best alpha itself moves with c, then invariance of the OUTPUT was not
    enough and the objective's TUNING remains scale-bound.

**PIVOT-IF:** no alpha in the registered ladder is both c-invariant and at
least as good as `g3_raw_scale` -> INVARIANCE AND RECOVERY GENUINELY TRADE
OFF at this objective. That is a structural limit of the current construction,
not a tuning failure, and the line's next question becomes whether a different
regularization FAMILY (not strength, not per-column shape) can hold both — or
whether the displacement is irreducible here, which is the question M4-G5's
pivot branch pointed at.

**Gates.** G0 POWER at the author grain against M4-G5's persisted gap
(.051–.062); G1 triple anchor to <=1e-12; G2 STRENGTH LIVENESS (report the
realized ridge magnitude per alpha and confirm it moves as registered; an
alpha whose realized ridge does not move is INERT and VACUOUS); G3 truth-path
invariance; G4 materiality-form compliance stated per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_g6_shape_and_strength/`; report
`reports/SUICA_M4_G6_SHAPE_AND_STRENGTH_REPORT.md`.

## M4-G6 outcome (2026-08-03, appended)

Same design as the registration above, executed: eight arms (`baseline`,
`g3_raw_scale` at c=1 only; `column_standardized` at deployed strength plus
`colstd_alpha_<k>` for k in {0.05,0.10,0.20,0.50,1.00}, all at c in
{0.25,1,4}), reusing `g5._forced_route_derivative_columnwise` completely
UNCHANGED for every `column_standardized`-family arm -- the only new
quantity is the scalar `ridge_used(alpha) = alpha * deployed_ridge` passed
into it. A cheap `--stage smoke` pre-flight (one representative context
per world, no full sweep) verified the alpha-ridge arithmetic and the
`colstd_alpha_1.00`/`column_standardized` identity BEFORE the full 8-world
truth sweep launched, and passed cleanly on its first run; the full sweep
then completed on all 8 worlds with zero Tracebacks, RuntimeErrors,
LinAlgErrors, or Killed signals (~40.6 minutes wall-clock, 8 concurrent
against 10 cores, ~1.28x M4-G5's own ~31.75 minutes despite a ~1.43x
larger per-world (arm,c) count, since this leg's single mechanism needs
only one `_hazard_design` build per call, unlike `diagonal_ridge`'s two).

**PIVOT DOES NOT FIRE. Lean (a) HOLDS EXACTLY for all five registered
alphas, without exception** (0.0 paired difference, all 745 authors, every
c-pair, both budgets) -- directly confirming, empirically, the leg's own
pre-registered Part 0.1 argument that the reparameterized penalized
objective is c-independent for ANY fixed scalar ridge, not merely the
deployed one. **Lean (b) HOLDS for three of five alphas** (0.05, 0.10,
0.20 -- all comfortably WITHIN the +/-0.02 margin at both budgets) and
**MISSES for the remaining two** (0.50 AMBIGUOUS/OUTSIDE; 1.00 OUTSIDE
both budgets, reproducing M4-G5's own persisted 0.0515/0.0605 MISS
bit-for-bit, since `colstd_alpha_1.00` is bit-identical to
`column_standardized` by construction -- confirmed to 0.0 by an
assembly-time structural check, 12,288 rows). **Three alphas hold BOTH
leans jointly**, decisively defeating the PIVOT-IF condition. The
headline alpha -- lowest pooled error among the three -- is
`colstd_alpha_0.10` (10% of deployed, near the midpoint of M4-G3's own
measured 4.6-18% band): mean `e_arm_true` 0.5083/0.4946 (4x/8x) against
`g3_raw_scale`'s own persisted 0.5068/0.4949, statistically tied and
numerically ahead at 8x. **Lean (c), the sharpest lean, HOLDS under the
adopted (mean, per-budget) reading**: the error-minimizing alpha is 0.10
at c=0.25, 1.0, AND 4.0, for both budgets separately and pooled -- an
INTERIOR optimum (error rises to both sides), so no ladder-endpoint
question arises under this reading. A disclosed median-based robustness
check names a DIFFERENT best alpha (0.05) -- also perfectly scale-free in
its own right, six-for-six across (c,budget) -- but this reading's own
optimum sits exactly at the ladder's lower endpoint, so per the outer
task's own instruction this is disclosed plainly: any exploration below
alpha=0.05 would need a separately-registered leg, not an extension of
this one. G0 shows the deciding comparisons carry essentially zero (lean
a) or comfortably small (lean b, half-widths 2.8-3.9x smaller than the
+/-0.02 margin) residual uncertainty. G1 anchor (three anchors plus one
bonus internal-consistency check, 45,056 rows total) and G3 truth-path
invariance (160 checks) both pass at exactly 0.0. G2 confirms the alpha
multiplier is genuinely live at all 15 (alpha,c) cells, matching the
registered alpha to 0.0 absolute difference and spanning exactly the
registered 20x ladder ratio.

**Verdict: shape (M4-G5's per-column standardization) and strength (this
leg's weaker-than-deployed alpha) together DELIVER the repair.** The
scale-dependence chased since M4-G2, localized to `hazard_ridge` by M4-G3,
decomposed into shape (solved exactly by M4-G4/M4-G5) and strength
(untouched by M4-G5, since its calibration anchored to the deployed
value) is now closed on BOTH axes at once: a band of alphas (roughly the
lower fifth of the registered ladder) achieves exact c-invariance AND
recovery matching `g3_raw_scale`'s own best level, and the
error-minimizing member of that band is the SAME member at every c
tested, under the reading methodologically consistent with every other
lean in this line. Full artifacts: `scripts/run_suica_m4_g6_shape_and_strength.py`;
`results/m4_g6_shape_and_strength/` (decision.json, gates.json,
error_surface_alpha_x_c_x_budget.csv, argmin_by_c_and_budget.csv,
truth_recovery_rows.csv, author_level_truth_rows.csv, context_meta.csv,
smoke_check.csv, and partials); `reports/SUICA_M4_G6_SHAPE_AND_STRENGTH_REPORT.md`.

## M4-G6 planner adjudication note (2026-08-03, appended) — the repair is delivered

**All three leans HOLD; the pivot does not fire; the verdict is
`SHAPE_AND_STRENGTH_HOLD_AT_SOME_ALPHA`.** This is the first fully
constructive result of the line.

The alpha x c author-grain error surface (n=745, pooled) makes both halves
visible at once:

| alpha | c=0.25 (4x/8x) | c=1.0 (4x/8x) | c=4.0 (4x/8x) |
|---|---|---|---|
| 0.05 | .5123/.4963 | .5123/.4963 | .5123/.4963 |
| **0.10** | **.5083/.4946** | **.5083/.4946** | **.5083/.4946** |
| 0.20 | .5122/.5019 | .5122/.5019 | .5122/.5019 |
| 0.50 | .5314/.5258 | .5314/.5258 | .5314/.5258 |
| 1.00 | .5583/.5554 | .5583/.5554 | .5583/.5554 |

Every row is identical across c — lean (a) holds EXACTLY at all five strengths
(0.0 paired difference, no exceptions), confirming empirically that M4-G5's
reparameterization argument does not depend on strength. Lean (b) holds at
alpha .05/.10/.20 and misses at .50/1.00, the latter reproducing M4-G5's own
miss bit-for-bit. Lean (c) — the sharp one — HOLDS: alpha = .10 minimizes
error at every c, at both budgets, and pooled, and it is an INTERIOR optimum
rather than a ladder endpoint. G2 is exact (realized ridge matches the
registered alpha to 0.0 at all 15 cells), G1 anchors reproduce M4-G5 at
exactly 0.0 over 32,768 rows, G3 is 0.0 over 160 checks.

**Headline: `column_standardized` at alpha = 0.10 x deployed ridge is exactly
c-invariant AND ties `g3_raw_scale`'s recovery (.5083/.4946 vs .5068/.4949,
numerically ahead at 8x) — so the objective can be made scale-free without
paying for it.** Disclosed qualification, stated rather than papered over: a
median-based reading names alpha = .05 instead, which sits at the ladder's
lower endpoint; the ladder was registered as non-extendable, so any test below
.05 requires its own leg.

Process note, also carried into the report: the leg's original monitor was a
`tail -F` pipeline that by construction can never signal completion, which is
why the agent had to be woken. It was killed and not rebuilt. All eight worlds
computed correctly and independently, verified after the fact from each log's
completion marker and partial-file presence.

## M4-G7 registration (2026-08-03, BEFORE run) — take the repair back to the problem that opened the line

**Question.** M4-G6 delivered a repair judged on truth-referenced recovery and
c-invariance. But this line was opened to attack something else: the discovery
objective's systematic FRAME DISPLACEMENT — Leg 14's finding, characterized by
M4-E2 as distributed, world-specific, and not removable by any single term.
Does the certified repair actually reduce THAT?

This is the line's closing question, and it is registered to be able to end in
an honest partial win: scale-invariance and recovery could both be real gains
that are simply ORTHOGONAL to the displacement.

**Design.** Reuse M4-E2 / Leg 14's own worlds, objective path and persisted
anchors — the same ground the displacement was originally measured on.

- `deployed` — the baseline objective as shipped (anchor: must reproduce
  M4-E2's persisted values to <=1e-12).
- `repaired` — `column_standardized` at alpha = 0.10 x deployed ridge, the
  M4-G6 headline (anchor: must reproduce M4-G6's persisted recovery to
  <=1e-12 where the paths coincide).

**Metrics (all three mandatory).**
1. The SCALE-NORMALIZED offset registered in M4-G2 — the raw `offset_norm` is
   disqualified and may not be used as the primary.
2. Leg 14's displacement gap, on its own persisted definition.
3. Truth-referenced recovery, both M4-F5-style variants, to confirm the
   reduction is not bought at recovery's expense.

**Leans.**
(a) DISPLACEMENT FALLS: the repaired objective's scale-normalized offset is
    lower than deployed's, paired-by-world CI excluding zero.
(b) THE ORIGINAL GAP FALLS: Leg 14's displacement gap is lower under the
    repair, paired CI excluding zero.
(c) NOT COSMETIC: truth-referenced recovery does not worsen under either
    variant (equivalence form, margin registered in Part 0).

**PIVOT-IF:** neither (a) nor (b) shows a reduction with a CI excluding zero
-> SCALE-INVARIANCE AND RECOVERY ARE ORTHOGONAL TO THE DISPLACEMENT. The
repair stands as a real and certified improvement to the objective's
conditioning, the displacement is confirmed as a separate still-open defect,
and the line closes with a partial win labeled exactly as such — not as a
solution to the problem it was opened for.

**Gates.** G0 POWER at the registered grain, stated with the target level from
M4-E2's persisted artifacts and the MDE, with the grain justified rather than
inherited (fifth standing rule); G1 dual anchor to <=1e-12; G2 REPAIR LIVENESS
(the repaired arm must differ from deployed in the applied regularization —
report the realized per-column ridge profile beside deployed's scalar); G3
truth-path invariance; G4 materiality-form compliance per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_g7_repair_vs_displacement/`; report
`reports/SUICA_M4_G7_REPAIR_VS_DISPLACEMENT_REPORT.md`.

## M4-G7 outcome (2026-08-03, appended)

Same design as the registration above, executed: two arms (`deployed`;
`repaired` = `g5._forced_route_derivative_columnwise` with
`treatment="column_standardized"` at `ridge_deployed = 0.10 x
deployed_ridge`, M4-G6's headline, called UNCHANGED) x the 3
`HIGH_GAP_WORLDS` (Leg 14's/M4-E2's own worlds, not the G-line's expanded
8-world `D1_WORLDS`) x 8 repetitions. Part 0 proved, before any compute,
that metrics 1 and 2 (`offset_norm`/its M4-G2 scale-normalized form, Leg
14's `disp_v2`) are pure functions of `context["v2_basis"]` alone --
`_forced_route_derivative_columnwise`'s signature takes a fixed basis and
never returns one, and `context["v2_basis"]` is built once, before either
arm exists as a concept, from a config whose `hazard_ridge` never varies
between arms. All three foreground world passes and the assemble stage
completed with zero Tracebacks or RuntimeErrors, all on the first attempt
except one non-hypothesis-relevant diagnostic (below).

**PIVOT FIRES. Lean (a) MISSES and lean (b) MISSES, both at an EXACT 0.0
paired difference, not a near-miss.** Lean (a) (M4-G2's registered
scale-normalized offset, paired-by-world, n=3 -- the metric's own native
grain): per-world reduction 0.0/0.0/0.0, CI exactly [0.0, 0.0]. Lean (b)
(Leg 14's own `disp_v2`, paired-by-repetition, n=24 -- chosen over the
naive world-grain default per the fifth standing rule, since `disp_v2` is
already a per-rep quantity; world-grain companion, n=3, agrees): reduction
0.0 at all 24 (world, repetition) pairs, CI exactly [0.0, 0.0] at both
grains. **Leg 14's displacement gap closes by exactly 0% of its own
magnitude.** This is not a power failure: G0 shows neither comparison is
underpowered against its own registered bar (12.5% of deployed's own mean
scale-normalized offset for lean (a); Leg 14's own 10% `PIVOT_REDUCTION_BAR`
for lean (b)) -- the realized half-widths are exactly 0.0 because the
underlying per-unit difference is exactly 0.0, a direct, verified
confirmation of Part 0.1's structural argument (`structural_identity_check`
in `decision.json`: `pass_exactly_zero: true`), not sampling luck. **Lean
(c) HOLDS at both budgets**: author-grain (n=384/budget) mean diff
(repaired minus deployed) is -0.000243 (4x) and -0.007886 (8x), both CIs
comfortably inside the registered +/-0.02 margin (upper bounds 0.0156 and
0.0084) -- one-sided WITHIN at both budgets, if anything favoring
`repaired`. G1 anchor (five independent persisted sources: M4-E2's
`offset_table`; Leg 14's `displacement_rows.csv` and `displacement_table`;
M4-G6's persisted `baseline` and `colstd_alpha_0.10` truth rows; M4-G1's own
independent `baseline` rows as a disclosed secondary reading, since M4-E2
itself persists no budget=4x/8x recovery for the `deployed` arm's metric-3
anchor to literally target) passes at 1.78e-15 -- floating-point noise,
comfortably inside 1e-12. G2 REPAIR LIVENESS confirms the repair genuinely
differs from deployed on BOTH axes the outer task asked for: the scalar
ratio is exactly 0.10 (cross-checked against M4-G6's own persisted G2
evidence for `colstd_alpha_0.10`), and the per-column ridge profile --
computed via `g5._standardize_design`, the exact floor-protected mechanism
(`DESIGN_VAR_FLOOR=1e-12`, `DEGENERATE_STD_FALLBACK=1.0`) the real fit uses
internally -- spreads 20.5x-281.8x across 24 sampled contexts against
deployed's exactly-flat 1.0x profile. G3 truth-path invariance passes at
0.0 (6 checks, both arms).

**Verdict: `PIVOT_SCALE_INVARIANCE_AND_RECOVERY_ORTHOGONAL_TO_DISPLACEMENT`.**
Per the registration's own consequence for this branch: M4-G6's certified
repair stands, unmodified, as a real and useful improvement to the
discovery objective's conditioning -- exactly c-invariant, at no cost to
truth-referenced recovery, independently reconfirmed here. But it does not
touch, at all, in any world or repetition, the frame displacement M4-D Leg
14 found and M4-E2 decomposed. The mechanical reason is now proven, not
merely inferred: every one of this line's six legs (M4-G1 through M4-G6)
intervened on the whitening scale or the estimator's ridge penalty, both of
which act strictly downstream of `context["v2_basis"]` -- the object Leg
14's displacement is measured on -- so no repair built entirely within that
downstream space could ever have moved it, regardless of how well-tuned.
**The M4-G line closes here on a partial win, labeled exactly as such**:
the objective's conditioning is repaired and certified; its systematic
frame displacement is confirmed real, unmoved, and handed off as belonging
to the basis-construction path (`_bases_from_whitening`/
`build_m4_discovered_basis`), not the regularization path this entire line
worked in.

**One mechanical problem, found on the first run, fixed before any
hypothesis-relevant number was affected.** The first G2 per-column
ridge-profile pass used raw, unfloored design-column variance; several
near-degenerate columns (e.g. `condition_0`) drove the reported spread
toward float overflow (`~1e+301`, observed). This diagnostic feeds only G2
REPAIR LIVENESS evidence, never a lean or the pivot, so nothing
hypothesis-relevant existed yet. Fixed by calling `g5._standardize_design`
directly (the real fit's own degenerate-column floor); all three worlds
were then re-run in full, and every substantive number -- metrics 1/2/3,
all gates, all leans -- reproduced identically before and after, as
expected since the fix touched only that one diagnostic.

Full numbers, gates, per-world/per-arm tables for all three metrics, and
the complete faithfulness chain: `reports/
SUICA_M4_G7_REPAIR_VS_DISPLACEMENT_REPORT.md`. Artifacts:
`scripts/run_suica_m4_g7_repair_vs_displacement.py`;
`results/m4_g7_repair_vs_displacement/{decision.json, gates.json,
offset_displacement_by_world.csv, displacement_by_rep.csv,
truth_recovery_rows.csv, author_level_truth_rows.csv,
g2_ridge_profile_rows.csv, g3check_rows.csv, context_meta.csv, and
partials}`.

## M4-H1 registration (2026-08-03, BEFORE run) — re-decompose the displacement in valid units

**The line that opened here closes with `docs/SUICA_M4_G_OBJECTIVE_LINE_SYNTHESIS.md`.**
M4-G7 proved the displacement lives in basis construction (`context["v2_basis"]`),
not in the ridge/whitening territory the M4-G line worked in. Before any work
starts there, one debt must be paid.

**Question.** M4-E2 decomposed the displacement into subspaces S1 (supervised
core), S2 (supervision span), S3 (normalization/scale modes) and S4 (residual),
and reported the scale family as the largest identifiable carrier (~1/3
sequential, ~1/2 standalone) with the residual the largest single piece
(.40–.45). It computed all of that on the RAW `offset_norm`, which M4-G2
disqualified as unit-dependent (log-log slope .8796 under a pure change of
units). **Does the attribution survive in valid units?**

**Design.** Re-run M4-E2's decomposition unchanged in every respect — same
worlds, same anchors, same S1–S4 construction, same registered sequential
order, same reverse order, same standalone variants — with ONE substitution:
the target quantity is M4-G2's registered scale-normalized offset instead of
the raw one. Report the raw-metric shares alongside as the disqualified
comparison, so the change (or its absence) is visible rather than asserted.

**Leans.**
(a) THE ATTRIBUTION MOVES: the scale family's share under the normalized
    metric differs from its raw-metric share by more than 10 percentage
    points, in either direction, in the registered sequential order.
(b) THE RESIDUAL SURVIVES: the residual block remains the largest single piece
    under the normalized metric (M4-E2's most consequential finding, tested
    for unit-robustness).
(c) WORLD-SPECIFICITY SURVIVES: the offset direction remains world-specific at
    the permutation null. This is a units-INDEPENDENT property, so it doubles
    as a control on the re-analysis: if it changes, the re-implementation
    differs from M4-E2's in some way not registered here.

**PIVOT-IF:** every subspace share moves by <= 10 points -> M4-E2's
decomposition was units-robust after all. Its picture stands as structural,
the debt is paid, and the basis line proceeds directly from it.

**Gates.** G0 POWER with the grain justified and the target level cited from
M4-E2's persisted artifacts; G1 ANCHOR — the RAW-metric re-run must reproduce
M4-E2's persisted shares to <=1e-12, which is what proves the only change is
the metric; G2 METRIC SUBSTITUTION LIVENESS — show the normalized and raw
targets actually differ per world (they must, by M4-G2); G3 truth-path
invariance where applicable; G4 materiality-form compliance per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_h1_normalized_decomposition/`; report
`reports/SUICA_M4_H1_NORMALIZED_DECOMPOSITION_REPORT.md`. This leg opens the
**M4-H basis-construction line**; subsequent M4-H registrations continue in
this document until it warrants its own.

## M4-H1 outcome (2026-08-03, appended)

**PIVOT FIRES -- not on a near-miss but on an exact algebraic identity,
proved in Part 0 before compute and confirmed empirically to floating-point
precision. Every subspace share moves by at most 1.67e-14 percentage points
between the raw and the scale-normalized metric (bar: 10 points), across all
3 worlds, all 3 registered orderings (registered/reverse/standalone), and
every within-S3 family split. Verdict:
`PIVOT_UNITS_ROBUST_ATTRIBUTION_STANDS`.** Lean (a) THE ATTRIBUTION MOVES
MISSES (max observed move 1.7e-14 points, nowhere near the 10-point bar).
Lean (b) THE RESIDUAL SURVIVES HOLDS (S4 remains the largest single piece,
registered order, all 3 worlds, unchanged to machine precision: .426948/
.401458/.446432). Lean (c) WORLD-SPECIFICITY SURVIVES holds under this leg's
own Part 0.2 operationalization (offset cosines at-or-below the permutation
null's 95th percentile in all 3 pairs, both metrics, cosine identity vs. raw
1.67e-16).

**Why this is a proof, not a coincidence (Part 0.1).** M4-G2's registered
`scale_normalized_offset(arm) := offset_norm(arm) / GM(s)` is a SCALAR
division of the already-computed scalar offset_norm by a scalar summary of
that arm's own whitening -- it does not touch the whitening, the discovered
basis, or the offset's direction. Applied to the offset MATRIX
(`Delta_normalized := Delta / GM_world`, GM_world = M4-G2's own formula
averaged over the world's 8 reps, the one registered aggregation choice this
leg introduces), every one of M4-E2's decomposition statistics -- sequential/
reverse/standalone shares, within-S3 family shares, width-mismatch share,
dominant share, top-1 singular share, and cross-world Procrustes cosines
(raw values and both disclosed nulls) -- is a ratio of squared or nuclear
norms of LINEAR IMAGES of Delta under FIXED, Delta-independent bases (S1/S2/
S3 are built from Leg 10 arm-B directions, response-target patterns, and the
consensus position `a_center` -- never from Delta itself). Projection onto a
fixed basis is linear, so every such statistic is HOMOGENEOUS OF DEGREE 0 in
any global per-world scalar applied to Delta -- an algebraic identity, not an
empirical finding to be discovered. The two candidate readings of "substitute
the target quantity" this leg disclosed (rescale the decomposed object vs.
treat the substitution as a pure relabeling of the reported scalar) are
consequently forced to the SAME numbers; this is the ambiguity's resolution,
not an evasion of it. Refit-based statistics (M4-E2's task-4 diagnostic,
dominant-component removal) are invariant for a separate, simpler reason
(the removal direction is a component normalized by its own norm, already
unitless) and are not one of this leg's three registered leans, so task 4 is
not re-run -- disclosed, not silently skipped.

**G0 POWER**: the three comparisons are deterministic recomputations on
M4-E2's 3 registered worlds (a census, not a sample) -- WORLD is the grain
because it is the only one at which a world's own decomposition share is
defined (GPA consensus collapses 8 reps into one Delta before any share
exists), matching M4-E2's own non-CI, world-level point-comparison design at
this identical grain. Realized margin: 4.5e14x the registered 10-point bar
(max share identity diff 2.22e-16, machine epsilon) -- not underpowered by
any reading. **G1 ANCHOR** passes on two independent chains: (1) the RAW
re-run reproduces M4-E2's persisted `decision.json` at 84/84 checks (every
registered/reverse/standalone/S3-family share field plus 6 scalar
companions, 3 worlds), offset_norm at 3/3 worlds, and cross-world cosines
against M4-E2's persisted `cosine_rows.csv`, ALL at exactly 0.0 (or
5.55e-17-scale floating-point noise for the cosines) -- proving the metric
substitution is the ONLY thing that changed; (2) GM_world's own 24 per-rep
inputs reproduce M4-G2's persisted `scale_norm_rows.csv` (arm=="baseline")
at exactly 0.0, with Leg 10's own `_debias_gate` (confirms the lambda=0
whitening rebuild reproduces `context["v2_basis"]` exactly) passing at 0.0
across all 24 reps -- proving the scale factors used correspond exactly to
the SAME whitening the offset was built from. **G2 METRIC SUBSTITUTION
LIVENESS** holds decisively: `scale_normalized_offset` differs from
`offset_norm` by 96.4-97.4% (relative) in every world, GM_world running
27.5-38.6 -- the substitution changes the reported magnitude by roughly
two-thirds to three-quarters of an order of magnitude while (per Part 0.1)
being unable to move a single share. **G3** is NOT APPLICABLE, stated
explicitly (no truth-referenced regeneration path in this leg). **G4**
materiality-form compliance stated per gate in the report.

The full registered/reverse/standalone share table under both metrics
reproduces M4-E2's own numbers, including its disclosed order-sensitivity
knife-edge (`selection_creation_compensation`'s registered-order S3 at
.3830, .017 under the .40 pivot-share bar, while reverse-order S3 exceeds
.40 in all 3 worlds at .446/.529/.498) exactly unit for unit under the
normalized metric -- the order-sensitivity is a property of the
decomposition's structure, not of the raw offset's units, and normalizing
was never going to resolve it. The within-S3 family split (n2 = principal
column-scale modes, the whitening's amplification family) reproduces at all
three granularities M4-E2 reported (n2 alone: .3946/.4772/.4569; n2 within
standalone S3: .7398/.7082/.7901; n2 within the registered-order sequential
S3 component: .8227/.8132/.8714), identically between raw and normalized to
<=2.2e-16 at every cell.

**The consequence for the M4-G line's own synthesis, stated precisely, as
that document required**: the debt flagged there -- "M4-E2's decomposition
attributed ~1/3-1/2 of the identifiable carrier mass to the scale family
using the RAW offset, which G2 disqualified... now a units statement of
unknown structural content" -- is paid. M4-G2's units disqualification was
about comparing the MAGNITUDE offset_norm ACROSS DIFFERENT WHITENING
OPERATORS (M4-G1's 8 arms, the pure c-ladder) -- a cross-object comparison.
M4-E2's attribution is a decomposition of ONE FIXED object into subspace
fractions -- a within-object proportion, already unitless by construction.
The two were never the same kind of quantity; this leg is what proves that
formally, rather than merely arguing it. M4-E2's numbers stand exactly as
originally reported: scale family ~1/3 of the mass in the registered order
(~1/2 standalone, n2-dominated), residual .40-.45 the largest single piece,
offset direction world-specific at the permutation null.

No mechanical problems: all three foreground world computations (~35-60s
each, 8 contexts per world) and the assemble step completed cleanly on the
first attempt, zero Tracebacks or RuntimeErrors.

Full numbers, gates, the complete subspace x ordering share table under
both metrics, and the permutation-null evidence:
`reports/SUICA_M4_H1_NORMALIZED_DECOMPOSITION_REPORT.md`. Artifacts:
`scripts/run_suica_m4_h1_normalized_decomposition.py`;
`results/m4_h1_normalized_decomposition/{decision.json, gates.json,
decomposition_rows_both_metrics.csv, subspace_rows_both_metrics.csv,
cosine_rows_both_metrics.csv, cosine_rows_raw.csv, cosine_rows_normalized.csv,
g1_share_anchor_rows.csv, g1_offset_anchor_rows.csv, g1_cosine_anchor_rows.csv,
gm_rows_with_g2_anchor.csv, share_identity_check.csv, lean_a_rows.csv,
pivot_rows.csv, and per-world partials}`.

**Hand-off.** M4-E2's map -- scale family the largest identifiable carrier at
~1/3-1/2 of the mass, residual .40-.45 the largest single piece, offset
direction world-specific, order-sensitive knife-edge at .3830 -- is now
confirmed structural rather than a units artifact, closing the one debt the
M4-G line's synthesis owed the record before the basis-construction line
could build on it. The M4-H basis-construction line (`context["v2_basis"]` /
`_bases_from_whitening` / `build_m4_discovered_basis`, per M4-G7's own
hand-off) proceeds directly from M4-E2's own numbers without needing to
recompute them.

## M4-H2 registration (2026-08-03, BEFORE run) — is the displacement in the basis's own normalization?

**Where the search now stands.** M4-G7 proved the displacement lives in basis
construction (`context["v2_basis"]`), not in the estimator's ridge or the
whitening scale the M4-G line worked in. M4-H1 then confirmed that M4-E2's map
of that displacement is valid as written — its shares are units-invariant by
an identity — so the search can use M4-E2's own numbers, which point at S3
(normalization/scale modes; .2953–.3830 registered, .4459–.5288 standalone)
with the n2 scale family dominating inside it (.74–.79 of standalone S3), and
S4 (residual) as the largest single piece.

**Question.** The M4-G line applied audit-then-intervene to the ESTIMATOR's
regularization and repaired it exactly. Does the same treatment applied one
level up — to the normalization inside BASIS CONSTRUCTION — reduce the
displacement itself?

**Part 0 audit (gated, before any compute).** Enumerate every normalization,
scaling, centering and reference choice inside the construction of
`context["v2_basis"]`, with file and line references, and classify each as a
candidate carrier or not with a stated reason. **This inventory is the leg's
hypothesis space**; steps discovered later may be reported but not scored.
This mirrors M4-G3's inventory discipline, which is what made that leg's
localization decisive.

**Design.** Reuse M4-E2 / Leg 14's worlds and anchors.
- `deployed` — the basis as shipped (anchor: reproduce M4-E2 and Leg 14's
  persisted values to <=1e-12).
- `basisvar_<k>` — one arm per inventoried step, that step alone varied
  (regularized, unnormalized, or alternatively centered as the step's nature
  dictates; the variant form for each step is registered in Part 0).

**Metrics (all three mandatory at every arm).**
1. Leg 14's displacement gap on its own persisted definition — PRIMARY.
2. M4-E2's S1–S4 shares, to test whether a reduction shows up as mass leaving
   the subspace the mechanism predicts rather than as a number moving for
   unexplained reasons.
3. Truth-referenced recovery, both M4-F5 variants — because this program has
   three times measured a target improving while truth worsened (Leg 12,
   M4-F5, M4-G1) and no reduction is accepted here without checking.

**Leans.**
(a) A CARRIER EXISTS: at least one `basisvar_<k>` reduces Leg 14's gap by
    >= 25% relative to deployed, paired-by-world CI excluding zero.
(b) MECHANISTICALLY CONSISTENT: at the winning arm, S3's share falls (or, if
    the winner is not an S3-type step, the share of the subspace its own
    mechanism predicts falls) — a number moving for the registered reason.
(c) NOT COSMETIC: truth-referenced recovery does not worsen under either
    variant (equivalence form, margin registered in Part 0).

**PIVOT-IF:** no arm reduces the gap by >= 25% with a CI excluding zero ->
the displacement is NOT in the basis's normalization either. The remaining
candidate inside basis construction is the CONSENSUS/ALIGNMENT step itself,
and that becomes the line's next registered question.

**Gates.** G0 POWER with the grain justified not inherited, citing Leg 14's
persisted gap level and stating the MDE; G1 ANCHOR to <=1e-12 on both M4-E2
and Leg 14 persisted values; G2 BASIS LIVENESS — every arm must actually
change the basis, reported as a subspace angle or basis distance against
deployed, with a materiality margin registered in Part 0 (an arm that does not
move the basis is INERT and its result VACUOUS); G3 truth-path invariance;
G4 materiality-form compliance stated per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_h2_basis_normalization/`; report
`reports/SUICA_M4_H2_BASIS_NORMALIZATION_REPORT.md`.

## M4-H2 outcome (2026-08-03, appended)

**PIVOT DOES NOT FIRE. A carrier is found, mechanistically CERTIFIED by lean
(b), and disqualified on safety grounds by lean (c) -- the same shape of
finding this line reached at M4-G1, now located one level up, in basis
construction rather than estimator regularization.** Part 0 audited every
normalization/scaling/centering/reference choice inside
`freeze_m4_condition_transform` (`m4_condition_manifold_estimator.py:545-
608`) and its callees with file:line references, classifying six as
candidate carriers (per-source robust standardization; reference-panel
centering, mean vs median; the eigenvalue rank-retention threshold; the
whitening scale in both a regularized and an unnormalized reading; the
constant mass column's scale) and four as excluded with a stated a priori or
scope reason (chart family/dimensions/neighbors/landmarks/bandwidth, shared
with the closed chart-selection line; source-averaging, a fusion choice not
a normalization; the covariance denominator, a priori bounded at ~0.52%
whitening effect for N=96 reference points, far below the 25% bar and
degenerate with M4-G2's already-disqualified units axis; the whitening's
numerical floor, verified never binding at 6.7e5-3.0e6x its own value).

**Lean (a) A CARRIER EXISTS HOLDS: three of six `basisvar_<k>` arms clear
the 25% bar with a paired CI excluding zero, at BOTH the adopted rep grain
(n=24, primary, per M4-G7's own precedent for this identical metric) and
the literal-text world grain (n=3, companion) -- the two readings agree on
every arm's classification, so the grain ambiguity Part 0 disclosed did not
end up mattering.** `basisvar_whitening_unscaled` (drop the 1/sqrt(eig)
whitening entirely -- M4-G1's own "identity" extreme control, re-run one
level up on the basis that actually builds `context["v2_basis"]`) cuts Leg
14's gap by **64.71%** (rep grain; 64.54% world grain), the largest of any
arm, CI [10.950, 12.423]. `basisvar_rank_tolerance_tight` (rank_tolerance
1e-6 -> 1e-3) cuts 42.15% (CI [6.725, 8.500]) but is disclosed as partly a
width artifact -- width falls from 13 to 6-10 depending on rep, and the
`disp_v2/sqrt(width)` companion (this line's own D2 convention) shows only a
20.8% width-independent reduction. `basisvar_whitening_shrinkage`
(1/sqrt(eig+lambda), lambda = 0.10 x median(retained eig), M4-G1's own
middle rung) cuts 28.93% (CI [4.635, 5.814]) at CONSTANT width -- a clean
signal. The other three arms (center_median +7.19%, source_scale_off
+0.89%, intercept_matched_scale -3.11%, i.e. worse) all miss. G0 shows none
of the six arms underpowered against the 25%-of-deployed-mean bar
(half-widths 0.062-0.888 vs a 4.51 bar). Per the registered winner rule
(largest reduction, fixed before results): **winner =
`basisvar_whitening_unscaled`.**

**Lean (b) MECHANISTICALLY CONSISTENT HOLDS, decisively.** At the winner,
S3's registered-order share collapses by an order of magnitude in 3/3
worlds (.3323->.0490, .3830->.0238, .2953->.0482; mean .3368->.0403) --
exactly the subspace Part 0 predicted before compute (every candidate step
is, by construction, an S3-normalization step), not a number moving for
unexplained reasons. A side effect visible in the full share table, not
adjudicated by any lean: the mass leaving S3 lands disproportionately in S1
(safety-complement), which rises from .200-.232 to .497-.554, while S4
(residual) stays roughly flat.

**But lean (c) NOT COSMETIC MISSES, decisively, at both budgets and both
grains.** Author-grain (n=384) truth-referenced recovery at the winner is
WORSE by +.2741 [.2568,.2913] (4x) and +.2774 [.2603,.2944] (8x), both
classified OUTSIDE the registered +/-.02 margin by 12-14x its own width; the
world-grain companion agrees in sign and magnitude (+.307/+.303). G0 flags
this comparison's half-width (.0172/.0171) as nominally over this line's own
strict 0.01 power bar -- disclosed plainly, and disclosed as not undermining
the classification, since the realized CI sits nowhere near the margin
boundary the bar exists to resolve.

**Disclosed dose-response companion (non-adjudicating, computed identically
for every arm that cleared lean (a), mirroring M4-G1's own established
practice): the MILDEST qualifying arm is the only one that is safe.**
`basisvar_whitening_shrinkage` (28.9% reduction) shows truth recovery
GENUINELY IMPROVING vs deployed, CI [-.0169,-.0054] at 4x, entirely on the
better side and comfortably WITHIN the no-loss margin -- while
`rank_tolerance_tight` (42.2%) and `whitening_unscaled` (64.7%, the winner)
both MISS decisively (+.093 and +.274 at 4x). Spearman(reduction_pct,
`e_arm_true`) across all six arms = **+0.600** (descriptive, n=6, not a
registered test) -- the same-signed relationship as M4-G1's own
Spearman(offset level, error) = -.786 on the downstream copy (a positive
correlation between REDUCTION and ERROR is the same relationship, sign-
flipped, as a negative correlation between remaining offset LEVEL and
error). Smaller n and a different intervention family here, so the
magnitude is not directly comparable, but the qualitative pattern --
minimizing the registered target past a mild point makes the objective
worse at its job -- reproduces exactly, one level up.

G1 ANCHOR passes at exactly 0.0 (not merely inside 1e-12) across three
independent chains: `deployed`'s per-rep `disp_v2` vs Leg 14's persisted
`displacement_rows.csv` (24/24 checks), `deployed`'s offset_norm and all
four share families vs M4-E2's persisted `offset_table` (3/3 worlds), and
`deployed`'s budget=1x `e_v2_true` (via this leg's own new budget-
regeneration machinery) vs Leg 14's persisted `gap_rows.csv` (3/3
spot-checks) -- meaningful because `deployed`'s basis is independently
RECOMPUTED via the same ingredients/whitening machinery every `basisvar_<k>`
arm depends on, not read off `context["v2_basis"]` directly. G2 BASIS
LIVENESS passes for all six arms by a wide margin (smallest 2.97x the 10%
materiality bar, `basisvar_intercept_matched_scale`; largest 12.66x,
`basisvar_whitening_unscaled`) -- no arm's result is VACUOUS. G3 TRUTH-PATH
INVARIANCE passes at exactly 0.0 across 21 checks (all 7 arms x 3 worlds).

**Verdict: `CARRIER_FOUND_MECHANISM_OR_TRANSFER_UNCERTIFIED`.** Per the
registration's own instruction for a found carrier: the step is named
(`basisvar_whitening_unscaled`, `m4_condition_manifold_estimator.py:580-
583`), the closure is quantified (64.71% of Leg 14's gap), and lean (b) DOES
certify it as the registered mechanism rather than an unexplained number
movement -- but the carrier that reduces the displacement most is exactly
the one lean (c) disqualifies. This is not the PIVOT branch (a carrier was
found, so the "displacement is not in the basis's normalization either"
conclusion does not apply) and not a clean HOLD (lean (c) misses too
decisively to call this settled) -- it lands precisely in the branch the
registration reserved for exactly this shape of outcome, stated exactly as
found.

One process note, disclosed as the standing convention requires: the first
`--stage oracle` invocation exceeded the harness's foreground timeout and
was auto-moved to a background task by the tool layer, not by any monitor
built to wait on it; the harness's own completion notification was read
once and every subsequent stage ran synchronously in the foreground with an
explicit long timeout, as the process rules require. One efficiency fix,
made before any hypothesis-relevant number existed: the per-(world,
repetition, budget) world-regeneration call (~6-13s, arm-invariant) was
initially recomputed once per arm; a disk cache cut arm-stage wall-clock
from ~190s to ~15-28s per (world, arm) after the first arm of each world,
verified to change no computed number (the `deployed` arm's numbers,
cache-independent by construction, remain bit-identical to M4-E2's and Leg
14's own persisted values).

Full numbers, gates, the six-step Part 0 inventory with file:line
references, the per-arm x per-world table for all three metrics, and the
dose-response companion table: `reports/SUICA_M4_H2_BASIS_NORMALIZATION_
REPORT.md`. Artifacts: `scripts/run_suica_m4_h2_basis_normalization.py`;
`results/m4_h2_basis_normalization/{decision.json, gates.json,
disp_rows.csv, g2_liveness_rows.csv, truth_recovery_rows.csv,
author_level_truth_rows.csv, g3check_rows.csv, offset_shares_by_arm.csv,
and per-(world,arm) partials}`.

**Hand-off.** `basisvar_whitening_unscaled` is a second confirmed instance,
at a different pipeline location, of this program's recurring cosmetic-
lever pattern. The dose-response points at the safe direction being mild,
not aggressive: `basisvar_whitening_shrinkage` (recovery genuinely
improved) is the natural next candidate for a leg that registers a narrower
shrinkage ladder around this one step, mirroring M4-G1 -> M4-G3's own
narrowing move but now on the basis-construction path. Separately: because
this leg found a carrier (even if disqualified on safety grounds), the
question the registration reserved for a firing pivot -- whether the
CONSENSUS/ALIGNMENT step is the remaining candidate inside basis
construction -- is not reached by this leg's own registered branches and is
not claimed here; it stays open precisely because this leg's own carrier
failed on safety, not because basis normalization was exonerated as a
location for the displacement.

## M4-H2 planner adjudication note (2026-08-03, appended) — a safe lever exists, and my "winner" rule hid it

**The adjudication stands as reported.** Lean (a) HOLDS (3 of 6 arms clear
25% with CI excluding zero); lean (b) HOLDS decisively (at the winner, S3
collapses an order of magnitude, .337 -> .040, in 3/3 worlds — exactly the
subspace the step's own mechanism predicts); lean (c) MISSES decisively
(recovery worse by +.274/+.277, 12–14x outside the margin). Pivot does not
fire. Verdict `CARRIER_FOUND_MECHANISM_OR_TRANSFER_UNCERTIFIED`. Gates are
exact (G1 at 0.0 across three independent chains, G3 at 0.0 on 21/21, all six
arms live at 2.97–12.66x the materiality bar), and the audit excluded four
candidates with stated reasons rather than silently.

**The pattern this reproduces.** M4-G1 found that aggressively reducing the
offset destroyed recovery while the mildest arms were the only ones that
helped. M4-H2 finds the same thing one level up, in the basis:
Spearman(reduction, error) = +.60 across all six arms, the same phenomenon as
M4-G1's Spearman(offset, error) = −.786. **Twice now, at two independent
levels of the machinery, driving the displacement/offset target hard has cost
recovery, and mild regularization has been the only safe direction.** That is
starting to look like a property of this objective rather than a coincidence
of either level.

**And here is the defect, which is mine and is the sixth of its family.** My
registration defined the leg's "winner" as the arm with the LARGEST target
reduction, then tested the safety check at that arm. Target-only winner
selection systematically picks the most aggressive arm — precisely the one
most likely to be cosmetic. It did so here: lean (c) was evaluated at
`whitening_unscaled` (64.71% reduction, recovery destroyed) while a QUALIFYING
arm existed that was both effective and safe —
**`basisvar_whitening_shrinkage` (lambda = 0.10 x median retained eigenvalue)
cuts the gap 28.9%, clearing the registered 25% bar, and its recovery
genuinely IMPROVES with the CI entirely on the better side.** The agent found
and disclosed this as a non-adjudicating dose-response check because the
registration gave it no way to score it. The same defect cost M4-G1 its own
best arm (shrinkage .1, the only helpful one, missed the 25% bar by .14
points).

**Standing rule (sixth).** When a leg carries both a target metric and an
anti-cosmetic check, the WINNING ARM must be defined jointly — the best arm
among those that pass BOTH — never as the extremum of the target. A
target-only winner is a selection rule biased toward cosmetic results.

**Note worth carrying forward.** The safe arm's strength — 0.10 x a median
eigenvalue scale — is the same 0.10 that M4-G6 certified one level down. Two
independent optimizations landing on the same mild ratio is either a
coincidence worth naming or a property worth testing; M4-H3 measures it rather
than assuming it.

## M4-H3 registration (2026-08-03, BEFORE run) — the safe lever's ladder, with the winner defined jointly

**Question.** M4-H2 showed the basis whitening's shrinkage is a real,
mechanistically-certified, and SAFE lever on the displacement at
lambda = 0.10 x median retained eigenvalue. Where is its optimum, how much of
the displacement can be removed safely, and does the safe region extend?

**Design.** Reuse M4-H2's worlds, anchors, arms plumbing, inventory and gate
helpers. Arms:
- `deployed` (anchor: reproduce M4-H2's persisted values to <=1e-12).
- `basis_shrinkage_<a>` for lambda / median retained eigenvalue in
  {0.02, 0.05, 0.10, 0.20, 0.50} — a registered, NON-EXTENDABLE ladder
  bracketing M4-H2's working value. If the optimum sits at an endpoint, say so
  and note that any extension requires its own leg.
- `whitening_unscaled` carried as the known-unsafe reference (anchor).

**Metrics at every arm**: Leg 14's displacement gap; M4-E2's S1–S4 shares;
both M4-F5 truth variants.

**Winner definition (registered, per the sixth standing rule).** The JOINT
winner is the arm with the largest displacement reduction AMONG those whose
recovery does not worsen (equivalence, ±.02, both variants). Every lean below
is evaluated at the JOINT winner, never at the target extremum. The report
must publish the full arm x {displacement, recovery, S3 share} table and show
the joint selection explicitly.

**Leans.**
(a) SAFE AND EFFECTIVE: a jointly-qualifying arm exists with >= 25%
    displacement reduction.
(b) MECHANISTICALLY CONSISTENT: at the joint winner, S3's share falls
    materially (margin registered in Part 0).
(c) ACTIVELY GOOD, NOT MERELY HARMLESS: at the joint winner, recovery is
    IMPROVED — CI entirely on the better side, not just inside the
    equivalence band. This is the stronger claim and is registered as such.

**PIVOT-IF:** no arm is jointly qualifying at >= 25% -> the safe region and
the effective region do not overlap above 25%. The displacement then has a
bounded, partial, safe fix whose ceiling must be reported as the finding, and
the line's next question becomes whether the unsafe remainder is reachable at
all without cost.

**Gates.** G0 POWER with justified grain, citing M4-H2's persisted gap and
recovery levels; G1 ANCHOR to <=1e-12 on `deployed` and `whitening_unscaled`;
G2 BASIS LIVENESS per arm against the registered materiality bar; G3
truth-path invariance; G4 materiality-form compliance per gate; **G5 JOINT-
WINNER COMPLIANCE — the report must demonstrate the winner was selected
jointly and show what a target-only rule would have chosen instead**, so the
sixth standing rule is visibly enforced rather than merely stated.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_h3_safe_lever_ladder/`; report
`reports/SUICA_M4_H3_SAFE_LEVER_LADDER_REPORT.md`.

## M4-H3 outcome (2026-08-03, appended)

**Executed as registered**, reusing M4-H2's worlds/anchors/arms-plumbing/
Part-0/gate-helpers throughout; this leg's only new code is a parameterized
near-duplicate of M4-H2's own shrinkage-whitening branch (varying
lambda/median(retained eig) instead of M4-H2's hardcoded 0.10) plus the
per-arm-metric and joint-winner-adjudicate orchestration around it. Gates
exact: G1 anchor `9.71e-17` (three independent chains, 48+6+3072 row-level
checks against M4-H2's own persisted CSVs, not decision.json summaries
alone); G2 all six arms live 2.60x-12.66x the bar; G3 `0.0` on 21/21; width
verified identical across all 7 arms in every one of 24 (world,rep) cells
(no width confound); a disclosed, non-gate check confirms
`basis_shrinkage_0.10` reproduces M4-H2's own `basisvar_whitening_shrinkage`
bit-for-bit (0.0 over 24 checks) -- this leg's re-derivation of M4-H2's
middle rung is exact.

**G5 JOINT-WINNER COMPLIANCE, the leg's central deliverable.** All five
ladder rungs (0.02-0.50) pass the recovery-safety filter (does not worsen,
both budgets); only `whitening_unscaled` fails. Reduction rises
monotonically with ratio: 16.04% -> 23.17% -> 28.93% -> 34.66% -> 41.56%
(rep grain), with NO sign of flattening by 0.50. **Joint winner (registered
rule -- largest reduction among the safe pool): `basis_shrinkage_0.50`,
41.56%.** **Target-only winner (what the pre-sixth-standing-rule rule would
have picked): `whitening_unscaled`, 64.71%, recovery destroyed (+.274/+.277,
OUTSIDE both budgets)** -- the defect the sixth standing rule exists to
prevent, reproduced on schedule. A disclosed ambiguity (is the reference
arm eligible for joint-winner selection, given the registration names it
separately from "the... ladder"?) was resolved (ladder-only) and verified
not to matter: both readings pick the identical winner, since the reference
fails safety either way.

**The joint winner sits AT the ladder's own upper endpoint** -- registered
NON-EXTENDABLE, not extended; the true safe ceiling is not yet located and
would need its own leg past ratio 0.50.

**Adjudication at the joint winner:** (a) SAFE AND EFFECTIVE HOLDS (41.56%,
CI [6.790,8.221] excludes zero, 1.66x the bar). (b) MECHANISTICALLY
CONSISTENT MISSES: S3 falls materially (>=10% relative, this leg's own
registered margin) in 2 of 3 worlds (18.6%, 16.1%) but RISES in
`source_rotated_feedback` (-2.2%, i.e. does not fall at all, by any
margin) -- and this is not winner-specific: no rung on the entire ladder
achieves any S3 fall in that world, though the 3-world MEAN does decline
monotonically with ratio (0.337->0.298). (c) ACTIVELY GOOD MISSES: recovery
at the winner is a genuine, well-powered null (CI [-.0142,.0057]@4x/
[-.0129,.0068]@8x, both straddling zero, half-widths 0.0099 comfortably
under this line's own 0.01 power bar) -- not worse, not distinguishably
better. **Verdict `SAFE_BUT_MECHANISM_NOT_CONFIRMED_AT_WINNER`.** Pivot
does NOT fire.

**A new shape of outcome for this line.** M4-G1 and M4-H2 both found lean
(b) HOLDING decisively and lean (c) MISSING decisively at their winners.
Here, for the first time, lean (a) alone holds cleanly while both (b) and
(c) land in a genuine middle ground -- partial-but-real mechanism (2/3
worlds), null-but-not-harmful recovery -- rather than either a clean HOLD
or a decisive MISS.

**Disclosed, non-adjudicating dose-response companion.** The registered
rule optimizes "largest reduction among safe arms", not "best arm on all
three leans". `basis_shrinkage_0.20` (34.66% reduction) independently
clears 25% AND its recovery IS improved (CI entirely negative, both
budgets) -- had the rule instead been "largest reduction among ACTIVELY
GOOD arms", leans (a) and (c) would both have HELD at `0.20`. Lean (b)
would STILL have MISSED there too (its own `source_rotated_feedback` S3
share also sits above deployed's) -- the mechanism-uniformity gap is a
property of the whole safe range, not an artifact of the registered rule's
specific pick. No post-hoc rescue is applied; the registered winner and
its HOLD/MISS/MISS stand as the adjudicated result. This distinction
between "safe" and "actively good" is new to the line -- M4-G1's and
M4-H2's own safe arms were also their most-reduced safe arms, so the two
notions never had to be told apart before.

No mechanical problems: smoke test, all three context+oracle stages, all
three G3 spot-checks, all 21 (world,arm) compute stages, and assemble
completed clean on the first attempt, entirely foreground with explicit
long timeouts, matching M4-H2's own reported wall-clock order of magnitude
(~187-191s/world context+oracle, ~21-28s/arm after cache warm-up).

Hand-off: the safe lever's ceiling within `{0.02,...,0.50}` is 41.56% with
no sign of flattening -- locating the true ceiling needs a new,
separately-registered ladder extending past 0.50. Separately, this leg's
"actively good" property (available at 0.05-0.20, traded away by the
registered rule in favor of raw reduction at 0.50) is a distinction this
line has not previously needed to draw. And lean (b)'s partial confirmation
is robust across the whole tested ladder, localized to
`source_rotated_feedback` specifically -- a world-specific mechanism
question this leg's design surfaces but does not resolve.

## M4-H3 planner adjudication note (2026-08-03, appended) — the rule worked, and exposed its own successor

**G5 earned its place.** A target-only rule would have picked
`whitening_unscaled` (64.71% reduction, recovery destroyed at +.274/+.277) —
the exact defect the sixth standing rule exists to prevent, reproduced on
schedule and shown in the report rather than argued about. The registered
joint rule excluded it and selected `basis_shrinkage_0.50`. Two disclosed
ambiguities in the rule's reading were computed both ways and changed nothing.
The rule is now demonstrated, not merely asserted.

**Adjudication at the joint winner.** Lean (a) HOLDS (41.56% reduction, CI
[6.790, 8.221], clearing the 25% bar by 1.66x). Lean (b) MISSES. Lean (c)
MISSES — recovery at the winner is a well-powered null (half-width .0099
under the .01 bar): not worse, not distinguishably better. Verdict
`SAFE_BUT_MECHANISM_NOT_CONFIRMED_AT_WINNER`, a new outcome shape for this
line, where previous carrier-found legs had (b) hold decisively and (c) miss
decisively.

**Three things this leaves, in order of how much they matter.**

1. **The ceiling is not located.** The joint winner sits AT the registered
   ladder's own non-extendable upper endpoint with reduction still rising. The
   safe lever's actual ceiling is therefore unknown, and the registration
   said in advance that locating it needs its own leg. It does.

2. **The mechanism does not generalize across worlds.** S3 falls materially in
   2/3 worlds (18.6%, 16.1%) but RISES in `source_rotated_feedback` (−2.2%),
   and no rung anywhere on the ladder achieves any S3 fall in that world. This
   is not a winner-specific knife-edge; it is a world where the lever reduces
   the displacement without touching the subspace the mechanism predicts. That
   is consistent with M4-E2's world-specific direction finding and it means
   the carrier's mechanism is at best partial.

3. **My joint rule, having fixed one selection bias, has a subtler one.**
   "Largest reduction among safe arms" is still target-priority within the
   qualifying set, so it picks the most aggressive qualifying arm — the one
   least likely to clear the STRONGER safety level. It did:
   `basis_shrinkage_0.20` independently satisfies both (a) at 34.66% and (c)
   actively-good, and is arguably the better arm overall, but the rule chose
   .50 because .4156 > .3466. The agent disclosed this rather than quietly
   scoring the nicer arm.

**Standing rule (seventh).** When the anti-cosmetic check has graded levels
(harmless vs actively good), the registration must state WHICH level defines
qualification, and the leg must report the best arm at EACH level. A single
winner hides the trade the levels are there to expose.

## M4-H4 registration (2026-08-03, BEFORE run) — where does the safe region end, and does the lever generalize?

**Design.** Reuse M4-H3's worlds, anchors, arms plumbing and gate helpers.
Ladder EXTENSION upward: lambda / median retained eigenvalue in
{0.50, 1.0, 2.0, 4.0, 8.0}, with 0.50 carried as the anchor (must reproduce
M4-H3's persisted values to <=1e-12) and `whitening_unscaled` retained as the
known-unsafe reference. Metrics unchanged: Leg 14's displacement gap, M4-E2's
S1–S4 shares, both M4-F5 truth variants.

**Winner definition (seventh standing rule).** Report TWO winners explicitly:
the best arm that is HARMLESS (recovery within ±.02 equivalence) and the best
arm that is ACTIVELY GOOD (recovery CI entirely on the better side). Every
lean below names which winner it is evaluated at.

**Leans.**
(a) CEILING LOCATED: at least one rung in the extended ladder is UNSAFE, so
    the harmless region's upper bound is interior to this ladder rather than
    an endpoint again.
(b) GENERALIZES OR NOT — a registered either-way test: at the harmless
    winner, does S3 fall in 3/3 worlds? A miss here confirms
    `source_rotated_feedback`'s non-response is STRUCTURAL rather than a
    strength effect, which is itself the finding and must be reported as one.
(c) THE LEVELS SEPARATE: the actively-good ceiling is strictly below the
    harmless ceiling, quantifying what pushing harder costs.

**PIVOT-IF:** no rung in the extended ladder is unsafe -> the safe region
extends beyond 8x median-eigenvalue shrinkage and has no located ceiling in
this design. That would mean displacement reduction is safe at essentially any
shrinkage, which is surprising enough that it demands a separate explanation
of why the deployed value sits where it does — and that explanation becomes
the line's next question rather than a footnote.

**Gates.** G0 POWER with justified grain, citing M4-H3's persisted levels;
G1 ANCHOR to <=1e-12 on `basis_shrinkage_0.50` and `whitening_unscaled`;
G2 BASIS LIVENESS per arm; G3 truth-path invariance; G4 materiality-form
compliance per gate; G5 DUAL-WINNER COMPLIANCE — publish the full arm x
{reduction, S3 per world, recovery both variants} table and both winners, and
state what a single-winner rule would have chosen.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_h4_safe_ceiling/`; report
`reports/SUICA_M4_H4_SAFE_CEILING_REPORT.md`.

## M4-H4 outcome (2026-08-03, appended)

**Executed as registered**, reusing H3's worlds/anchors/arms-plumbing/gate-
helpers/G5-table-machinery throughout (imported as `h3`, which itself
imports `h2`); this leg's only new code is arm-dispatch plumbing for the
four genuinely new ratios (1.0/2.0/4.0/8.0 -- literal reuse of H3's own
`_whitening_for_shrinkage_ratio`) plus the dual-winner selection/
adjudication orchestration the seventh standing rule requires. `deployed`,
`basis_shrinkage_0.50` and `whitening_unscaled` are computed by LITERAL
calls into H3's own dispatch function, not reimplemented. Gates exact: G1
anchor `9.71445146547012e-17` (registered-literal 2-chain AND disclosed
superset 3-chain incl. `deployed`, both against H3's own persisted CSVs,
full row-level joins); G2 all six arms live 6.2x-12.7x the bar, monotone in
ratio; G3 `0.0` on 21/21; smoke test additionally confirmed the bridge rung
bit-for-bit against H3's own dispatch (`0.0` exact) before any full-scale
compute ran. A mechanical reporting-layer correction, found AFTER the
hypothesis-relevant numbers existed: the script's own coarse width-
invariance flag reads `false` because it conflates between-repetition and
between-arm variation; the correct per-(world,repetition) check (computed
directly from `disp_rows.csv`) confirms 0/24 disagreements -- no width
confound, no adjudicated number changes.

**G5 DUAL-WINNER COMPLIANCE.** `basis_shrinkage_{0.50,1.00}` qualify
HARMLESS (does not worsen, both budgets); `basis_shrinkage_2.00` is
AMBIGUOUS at both budgets (its own CI straddles the 0.02 margin);
`basis_shrinkage_{4.00,8.00}` and `whitening_unscaled` are confirmed
UNSAFE (OUTSIDE, both budgets). **HARMLESS winner: `basis_shrinkage_1.00`,
45.79% reduction** (CI [7.515,9.025], rep grain). **ACTIVELY GOOD winner
within this leg's own newly-tested ladder: NONE** -- the actively-good
pool restricted to `{0.50,...,8.0}` is empty. What a SINGLE-winner rule
(H3's own sixth-standing-rule "joint winner," harmless-only) would have
chosen: the identical arm, `basis_shrinkage_1.00` -- reported alone, this
would have hidden that no arm in this leg's own range is actively good and
that the winner's own recovery classification has markedly less
statistical headroom than H3's winner (8x buffer to the margin only 10.1%
of the margin's width, vs H3's 65.9%) -- decisive, not underpowered at the
decision-relevant margin, but honestly disclosed as thinner. Target-only
winner (the original defect): `whitening_unscaled`, 64.71%, recovery
destroyed -- reproduced a third time, on schedule.

**Adjudication.** (a) CEILING LOCATED HOLDS: `basis_shrinkage_{4.00,8.00}`
are confirmed unsafe (+.0375/+.0397 and +.0605/+.0630, OUTSIDE both
budgets, CIs entirely above the margin) -- the harmless region's upper
bound (1.0) is interior to this ladder, not another endpoint. One rung
(`2.00`) is honestly reported AMBIGUOUS rather than forced to either side
-- the boundary is precise about where safety ends and where unsafety
begins, but the interval `(1.0, 4.0]` is not fully resolved by this
ladder's own grain. (b) GENERALIZES HOLDS, resolving H3's open either-way
test: at the harmless winner, S3 falls materially in 3/3 worlds --
`source_rotated_feedback` (the world that ROSE or stayed flat at EVERY
rung H3 ever tested) falls 11.1%, clearing the 10% floor, with a clean
monotonic decline across the whole ladder past the 0.50 knife-edge (0.2953
deployed -> 0.3019 @0.50 -> 0.2624 @1.0 -> 0.2335 @2.0 -> 0.2077 @4.0 ->
0.1845 @8.0 -> 0.0482 @unscaled). Per the registration's own either-way
framing, this IS the finding: the total non-response at every H3 rung was
a STRENGTH EFFECT, not a structural immunity -- refuting, not merely
failing to confirm, the "structural" alternative. (c) LEVELS SEPARATE
HOLDS under the adopted Reading B (combined ladder: H3's own persisted
0.02-0.20, cited not recomputed, + this leg's own 0.50-8.0, confirmed
monotonic in reduction end to end, 9 points, 16.0%->52.2%) -- actively-good
ceiling 0.2 (H3) sits strictly below harmless ceiling 1.0 (this leg), an
order of magnitude apart; Reading A (this leg's own ladder alone) is
honestly reported NOT COMPUTABLE (empty actively-good pool) rather than
forced to a verdict. **Verdict
`CEILING_LOCATED__MECHANISM_GENERALIZES_3_OF_3_WORLDS__LEVELS_SEPARATE`** --
all three leans HOLD, a clean sweep and a new outcome shape for this line
(every prior carrier-found leg left at least one lean missing or in a
genuine middle ground).

**Two disclosed nuances, neither smoothed over.** First, the harmless
winner's own recovery classification, while decisive against the actual
0.02 margin under BOTH one-sided (adopted) and two-sided (disclosed
robustness check) readings, carries markedly thinner statistical headroom
than H3's winner (Section G0 of the report). Second, the safe/unsafe
boundary contains one genuinely unresolved rung (`2.00`, AMBIGUOUS) rather
than a single clean crossing -- disclosed as a real resolution limit of
this ladder's own grain, not glossed into a false-precision "ceiling is
exactly 1.0" claim.

No mechanical problems affecting any hypothesis-relevant number: smoke
test, all 3 context+oracle stages (186.6s/189.2s/191.2s/world, matching
H2/H3's own order of magnitude), all 3 G3 spot-checks, all 21 (world,arm)
stages (21-28s/arm after cache warm-up), and assemble completed clean on
the first attempt, entirely foreground with explicit long timeouts (no
background jobs, no monitors).

Hand-off: the harmless ceiling within the ladder tested to date is ratio
1.0 (45.79% reduction), located with one honestly-disclosed unresolved
rung (`2.0`) between it and the confirmed-unsafe region starting at `4.0`
-- narrowing that interval further needs a tighter bracket or larger n at
that specific point, not registered or performed here. The actively-good
ceiling (0.2) and harmless ceiling (1.0) now diverge by a full order of
magnitude, not merely at the margin -- settling the distinction this line
first had to draw at H3. Most substantively, `source_rotated_feedback`'s
total non-response across H3's whole ladder is resolved as a strength
effect, not a structural one -- the mechanism is present in all three
`HIGH_GAP_WORLDS`, just at different thresholds of the same lever, and why
that specific world needs roughly double the push the other two need is a
new, narrower, unregistered question this leg's design surfaces but does
not resolve.

## M4-H4 planner adjudication note (2026-08-03, appended) — clean sweep, and H3's world-heterogeneity reading overturned

**All three leans HOLD** — the first clean sweep of any carrier-found leg in
this line. Verdict
`CEILING_LOCATED__MECHANISM_GENERALIZES_3_OF_3_WORLDS__LEVELS_SEPARATE`.

- **(a) CEILING LOCATED.** Ratios 4.0 and 8.0 are confirmed UNSAFE (recovery
  CIs entirely above the .02 margin at both budgets). Ratio 2.0 is reported
  **AMBIGUOUS** — its own CI straddles the margin — rather than forced either
  way, so the honest statement is that safety ends at <=1.0 and unsafety
  begins at >=4.0, with the interval between genuinely undetermined.
- **(b) GENERALIZES, at the HARMLESS winner (ratio 1.00).** S3 falls
  materially in 3/3 worlds. `source_rotated_feedback` — which rose or stayed
  flat at EVERY rung M4-H3 tested — falls 11.1% here, with a clean monotonic
  decline the rest of the way. **M4-H3's apparent structural non-response is
  REFUTED: it was a strength effect.** The registration made this an either-way
  test and the answer came back the other way; recorded as such.
- **(c) LEVELS SEPARATE**, under the disclosed combined-ladder reading (M4-H3's
  .02–.20 plus this leg's .50–8.0, monotone across all nine points): the
  actively-good ceiling (.2) sits a full order of magnitude below the harmless
  ceiling (1.0). The this-leg-only reading is honestly reported NOT COMPUTABLE
  (empty actively-good pool) rather than forced.

**The seventh standing rule earned its place immediately.** A single-winner
rule would have named the same arm — but reporting it alone would have hidden
two things the dual report makes visible: that NO arm in this leg's range is
actively good, and that this winner's recovery classification has far thinner
headroom than M4-H3's (CI upper bound 17.5% / 10.1% of margin, versus
65.9% / 71.7% there). And the target-only counterfactual picked
`whitening_unscaled` for a third consecutive leg.

**The lever is now characterized.** Safe to ratio 1.0 (45.79% displacement
reduction); actively good only to ratio 0.2 (34.66%); unsafe at >=4.0;
undetermined at 2.0; mechanism generalizes across all three worlds at the
harmless ceiling. That is a real, bounded, mechanistically-supported partial
repair of the displacement — the first this program has had.

**And it leaves an obvious question.** Even at the harmless ceiling, **54% of
the displacement remains**. The lever moves S3; M4-E2 said S4 (residual) was
already the largest single piece at deployed (.4015–.4464). So what carries
the remainder, and is it attackable at all? That is the next registration.

## M4-H5 registration (2026-08-03, BEFORE run) — what carries the surviving 54%?

**Question.** Under the safe lever, the displacement falls by ~46% and S3's
share collapses. Does the surviving displacement have a new dominant carrier
that can be attacked in turn, or has it become genuinely distributed — in
which case ~46% is the practical ceiling of subspace-targeted repair and the
line should say so.

**Design.** Reuse M4-E2's S1–S4 decomposition machinery and M4-H4's worlds,
anchors and arms plumbing. Decompose the RESIDUAL displacement under:
- `deployed` (anchor: reproduce M4-E2's persisted shares to <=1e-12),
- `basis_shrinkage_1.00` — M4-H4's HARMLESS winner,
- `basis_shrinkage_0.20` — M4-H3's ACTIVELY-GOOD winner,
both anchored to their persisted displacement values at <=1e-12.

**Leans.**
(a) A NEW DOMINANT CARRIER EXISTS: under the harmless winner, some subspace
    carries >= 40% of the SURVIVING displacement in >= 2 of 3 worlds.
(b) IT IS S4: that carrier is the residual block — the piece M4-E2 already
    found largest at deployed.
(c) STRENGTH-INVARIANT: the same subspace dominates under BOTH repaired arms,
    so the finding is a property of removing S3-type mass rather than of one
    particular strength.

**PIVOT-IF:** no subspace carries >= 40% in >= 2 of 3 worlds under the
harmless winner -> THE SURVIVING DISPLACEMENT IS GENUINELY DISTRIBUTED. There
is then no attackable dominant piece left, ~46% is the practical ceiling of
subspace-targeted repair, and the line's next question is whether a
distributed remainder can be attacked at all — or whether this is where
subspace-targeted work honestly stops. That outcome is to be written with the
same prominence as a discovery, per this line's standing practice.

**Gates.** G0 POWER with justified grain, citing M4-H4's persisted levels;
G1 ANCHOR to <=1e-12 on all three arms (shares against M4-E2 for `deployed`,
displacement against M4-H3/M4-H4 for the repaired arms); G2 DECOMPOSITION
LIVENESS — the residual decomposition must actually differ from deployed's
(it must, since S3 falls; report the per-world share deltas so this is shown);
G3 truth-path invariance where applicable; G4 materiality-form compliance per
gate. Report the full arm x subspace x world share table.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_h5_residual_carrier/`; report
`reports/SUICA_M4_H5_RESIDUAL_CARRIER_REPORT.md`.

## M4-H5 outcome (2026-08-03, appended)

**Executed as registered.** Zero new Part 0 audit and zero new basis-
construction code: all three arms (`deployed`, `basis_shrinkage_1.00` =
H4's own HARMLESS winner, `basis_shrinkage_0.20` = H3's own ACTIVELY-GOOD
winner) are literal calls into H3's/H4's own already-anchored dispatch
functions; this leg's only new code is a 2-way arm router, the G1 anchor-
chain assembly, the G2 DECOMPOSITION LIVENESS computation (new to this leg
-- share-based, not basis-distance-based), and the qualifying-subspace lean
logic. A disclosed, adopted scope reduction: the registration's own Design/
Metrics/Leans sections name no `TRUTH_BUDGETS`, so Metric 3 (truth
recovery, already adjudicated at both repaired arms by H3/H4 themselves) is
out of scope -- the oracle-stage regeneration loop is skipped entirely, and
G3 is satisfied instead by a lightweight world-build faithfulness spot-check
(budget=1.0 regen vs. flat-style refit, 3 arms, 1 per world) that does not
gate any lean. Gates exact: G1 registered-literal chain (shares vs M4-E2's
own persisted `decision.json['offset_table']` for `deployed`; displacement
vs M4-H4's/M4-H3's own persisted `disp_rows.csv` for the two repaired arms,
24 row-level checks each) at **exactly 0.0**; disclosed superset (full
shares for both repaired arms vs H3's/H4's own persisted
`offset_shares_by_arm.csv`, plus `deployed`'s displacement vs both) at
1.78e-15, three orders of magnitude under the 1e-12 tolerance; G2
DECOMPOSITION LIVENESS live in 6/6 (arm,world) cells at 2.3x-5.2x its own
10% bar (reused `LEAN_B_MATERIALITY_RATIO`); G3 world-build faithfulness
`0.0` on 9/9 checks. One mechanical issue, a mislabeled (not miscomputed)
disclosure string inside `gates.G0_power.rule3_vacuous_check` -- H4's/H3's
own G2 ratios were cited without first dividing by the materiality bar,
producing a misleading "0.70x the bar" instead of the correct "7.01x" --
found and fixed AFTER `--assemble` had already produced every lean/pivot/
verdict number; re-running `--assemble` (pure re-derivation from already-
persisted partials) reproduced an identical verdict and identical gate
booleans, confirming only the one prose string changed.

**The full arm x subspace x world share table (both orderings), the
central deliverable.** Under REGISTERED order (S1->S2->S3->S4, ADOPTED),
S4_residual is the only subspace to ever clear 0.40 at either repaired arm,
in any world (S3 comes closest, up to 0.3212, still 0.079 below the bar)
-- it rises from an already-elevated deployed baseline
(0.4015-0.4464, M4-E2's own "largest single piece" finding) to **0.5066-
0.5247** at the harmless winner and **0.4517-0.4921** at the actively-good
winner, in EVERY one of the 3 worlds, at BOTH arms. G2's own delta evidence
confirms this is not assumed: S4's absolute registered-order share rises in
all 6 (arm,world) cells tested (+0.5 to +12.3 percentage points), while S1
FALLS in all 6 (-0.3 to -5.2 points) and S3 falls in 5 of 6 (the one
exception, `basis_shrinkage_0.20`/`source_rotated_feedback`, is the same
world H3 found totally non-responsive at every rung up to 0.50 -- S3 still
does not fall there even under this leg's own fresh recomputation, but S4
still gains 0.5 points, the smallest movement in the table but not a null).
Under REVERSE order (S3->S2->S1->S4, disclosed companion), S3 -- not S4 --
is the larger of the two at `deployed` and at `basis_shrinkage_0.20` in
EVERY world, reproducing M4-E2's own disclosed reverse-order picture
exactly; only at `basis_shrinkage_1.00` does S4 stay larger in all 3
worlds, and in `source_rotated_feedback` specifically by a genuine
knife-edge (S4=0.4313 vs S3=0.4302, gap 0.0012 -- the closest of all 18
(arm,world,ordering) top1-vs-top2 comparisons by more than an order of
magnitude; every other one of the 18 has a gap >=0.0185, and 15 of the
remaining 16 clear 0.027, 14 of those 16 clear 0.08).

**Adjudication.** (a) A NEW DOMINANT CARRIER EXISTS -- **HOLDS**, both
orderings agree: S4 clears the 0.40 bar in 3/3 worlds (not merely the
registered 2/3) at the harmless winner, under registered AND reverse order.
The pivot does NOT fire under either ordering
(`pivot.orderings_agree_on_pivot_decision = true`). (b) IT IS S4 --
**HOLDS**, both orderings agree (S4 is the sole qualifier both ways, so the
tie-break rule never activates). (c) STRENGTH-INVARIANT -- **HOLDS under
the ADOPTED registered order** (S4 is again the sole qualifier, 3/3 worlds,
at the actively-good winner) **but MISSES under the disclosed reverse-order
companion** (S3, not S4, is the sole qualifier there, 3/3 worlds) --
`lean_c.orderings_agree = false`. This traces directly to the share table
itself: reverse order already has S3 durably ahead of S4 at `deployed` and
at the milder `basis_shrinkage_0.20` repair; only the stronger
`basis_shrinkage_1.00` repair pushes S4 past S3 under both orderings, and
even then by a knife-edge in one world. **Verdict
`NEW_DOMINANT_CARRIER_FOUND__IS_S4__STRENGTH_INVARIANT__ORDER_SENSITIVE_AT_ACTIVELY_GOOD_ARM`.**

**Read plainly, per the registration's own instruction to give this the
prominence a positive finding deserves (the pivot did not fire, so this is
not the distributed-remainder branch, but the finding still needs stating
without spin):** the safe lever's repair does not manufacture a new
carrier from nothing -- it takes the block M4-E2 already found largest at
`deployed` (S4, the residual) and makes it the clear majority of what
survives (51-52% of the harmless winner's own displacement, up from a
40-45% plurality pre-repair), in every world, under this line's own
adopted ordering convention. But S4 is *defined* as whatever S1+S2+S3
does not explain -- unlike S3, it has no operational, code-legible
construction of its own (M4-E2's own inventory named three families for
S3; S4 was never given one). The reverse-order disagreement at the milder,
actively-good arm is itself informative, not noise: it suggests S4's
apparent takeover may be a strength effect of pushing past the
actively-good ceiling specifically, not an intrinsic property of removing
S3-type mass. Whether S4 can be given an operational construction at all,
and whether that construction is itself attackable, is the line's next
question -- not registered or attempted here.

## M4-H5 planner adjudication note (2026-08-03, appended) — the survivor has a name, and no handle

**Adjudication.** Lean (a) HOLDS decisively under BOTH orderings: S4_residual
clears the .40 bar in 3/3 worlds at the harmless winner (registered order
.507–.525; reverse .431–.462). Lean (b) HOLDS, sole qualifier either way.
Lean (c) HOLDS under the adopted registered order but MISSES under the
disclosed reverse-order companion, and the agent traced that to its source
rather than leaving it as a discrepancy: reverse order already has S3 ahead of
S4 at deployed and at the milder repair, and only the stronger 1.00 repair
pushes S4 past S3 — by a genuine .0012 knife-edge in `source_rotated_feedback`,
the closest of all 18 (arm, world, ordering) comparisons by more than an order
of magnitude. Pivot does not fire under either ordering. Verdict
`NEW_DOMINANT_CARRIER_FOUND__IS_S4__STRENGTH_INVARIANT__ORDER_SENSITIVE_AT_ACTIVELY_GOOD_ARM`.

**What it means.** The repair does not manufacture a new carrier; it
ENTRENCHES the one M4-E2 already found largest. S4 goes from a 40–45%
plurality at deployed to a **51–52% majority** of what survives at the
harmless winner, rising in all 6 (arm, world) cells (+0.5 to +12.3 points;
+6.0 to +12.3 at the harmless winner), while S1 falls in all 6 and S3 falls in
5 of 6.

**And here is the wall.** S4 is the RESIDUAL block — defined by exclusion,
as whatever survives projecting out S1, S2 and S3. **Unlike S3, it has no
operational, code-legible construction to attack.** Every repair this program
has landed — the estimator's ridge shape and strength, the basis whitening's
shrinkage — worked because there was a named step in the code to vary. There
is no S4 step. Subspace-targeted repair has therefore reached its own ceiling
at ~46% removable safely, and the majority of the remainder lives in a block
that currently exists only as a subtraction.

That leaves exactly one question worth asking before this line closes, and it
is decisive either way.

## M4-H6 registration (2026-08-03, BEFORE run) — is the residual attackable at all?

**Question.** Does the S4 residual carry REPRODUCIBLE structure — something a
future basis could be built to capture — or is it repetition-specific, in
which case the surviving majority of the displacement is irreducible at this
level and ~46% is the program's definitive ceiling for objective-side repair?

**Design.** At `deployed` (reference) and `basis_shrinkage_1.00` (the harmless
winner), extract the S4 component per (world, repetition) using M4-E2's own
S1–S4 construction, unchanged. Then measure:
- WITHIN-WORLD reproducibility: agreement (cosine, absolute value, on the
  Procrustes-aligned components) of S4's direction across repetitions inside
  each world, against a **repetition-shuffled null** whose construction is
  registered in Part 0 before compute.
- ACROSS-WORLD agreement: the same statistic between worlds, which M4-E2's
  world-specificity finding predicts should sit at the null — a consistency
  control on the whole measurement.
- Report both against the null's own q95, in equivalence/margin form.

**Leans.**
(a) REPRODUCIBLE WITHIN WORLD: S4's within-world across-repetition agreement
    exceeds the repetition-shuffled null's q95, with a CI excluding it, in
    >= 2 of 3 worlds at the harmless winner.
(b) NOT ACROSS WORLDS: across-world agreement sits at or below the null's q95
    — the control. A miss here means the measurement is picking up something
    shared that M4-E2's world-specificity says should not be there, and the
    leg must say so rather than proceed.
(c) MATERIAL ENOUGH TO CAPTURE: where (a) holds, the within-world agreement
    is >= .5, i.e. large enough that a per-world basis could plausibly capture
    the reproducible part rather than merely detecting it.

**PIVOT-IF:** within-world agreement does not exceed the null in >= 2 of 3
worlds -> **S4 IS REPETITION-SPECIFIC AND NOT ATTACKABLE BY ANY BASIS
CONSTRUCTION.** The displacement's surviving majority is then irreducible at
this level, ~46% is the definitive ceiling of objective-side repair, and the
M4-H basis line closes on that statement. Per this line's standing practice
that outcome is written with the prominence of a discovery, because it is one:
it would convert an open problem into a measured limit.

**Gates.** G0 POWER — state the null's spread and the design's MDE before
adjudicating, with the grain justified not inherited; G1 ANCHOR — both arms
reproduce their persisted S1–S4 shares to <=1e-12; G2 NULL LIVENESS — the
repetition-shuffled null must actually destroy the structure it is meant to
destroy (report its own spread against the unshuffled statistic); G3
truth-path invariance where applicable; G4 materiality-form compliance per
gate. Note M4-E2's disclosed Procrustes-cosine null inflation and handle it
explicitly — two independent random matrices of this shape are not orthogonal
in this statistic, so the null must be the shuffled-rep null, never zero.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_h6_residual_reproducibility/`; report
`reports/SUICA_M4_H6_RESIDUAL_REPRODUCIBILITY_REPORT.md`.

## M4-H6 outcome (2026-08-03, appended)

**PIVOT FIRES. S4 IS REPETITION-SPECIFIC AND NOT ATTACKABLE BY ANY BASIS
CONSTRUCTION. Verdict
`PIVOT_S4_REPETITION_SPECIFIC_NOT_ATTACKABLE_BY_ANY_BASIS_CONSTRUCTION`.**
G0 and G2 first, as this leg's own registration required. **G0 POWER**: the
repetition-shuffled null (200 draws per arm, pooling 24 real S4_residual
components into 3 random pseudo-groups of 8 and averaging the 84
within-pseudo-group pairwise |Procrustes cosine| values per draw) has q95 =
0.3313 (harmless winner) / 0.3821 (deployed), median 0.3260/0.3779, std of
draw means 0.0028/0.0026. MDE, from the null's own pooled pair-level std
(0.0362/0.0281, a non-circular pre-adjudication proxy): 0.0140 at
within-world grain (n=28), 0.0051 at across-world grain (n=192) — an order
of magnitude below the 0.5 materiality bar's own distance from the null
(0.169–0.179), so the design is well-powered for a MATERIAL effect, though
not for one at the scale of the null's own ~0.003 noise floor, a caveat
carried honestly into the per-world reading. The disclosed, non-deciding
random-matrix-null companion (E2's own construction) sits substantially
HIGHER (q95 ≈ 0.46) than the repetition-shuffled null — an empirical
confirmation, on this leg's own data, of exactly the non-interchangeability
M4-E2's disclosure warned about. **G2 NULL LIVENESS**: both arms' null
distributions are genuine, non-degenerate, sub-ceiling (range ≈0.015, std
≈0.0028, median 0.326–0.378, nowhere near the trivial ceiling of 1.0) —
live.

**G1 ANCHOR** passes at exactly 0.0 (registered-literal: deployed's shares
vs. M4-E2's decision.json, `basis_shrinkage_1.00`'s displacement vs.
M4-H4's disp_rows.csv, 24 checks) and 9.02e-17 (disclosed superset: full
shares vs. H4's own CSV, deployed's displacement vs. H3's AND H4's own
CSVs) — bit-identical to H5's own persisted world-level numbers, confirmed
by direct CSV comparison. A third, NEW anchor family — required because no
predecessor leg ever computed a per-repetition delta — ties this leg's own
new construction to already-anchored `disp_v2` values (7.11e-15 vs. a 1e-9
tolerance, since these are two independently-ordered computations of a
provably symmetric Procrustes distance) and to `leg14._align` itself
(0.0 and 1.07e-14 vs. a 1e-12 tolerance, a pure floating-point identity).
**G3** (world-build faithfulness, scope-reduced citing H5's own precedent,
gates no lean): 0.0 on 6/6 checks.

**Lean (a) REPRODUCIBLE WITHIN WORLD — MISSES: 0 of 3 worlds** (registered
bar ≥2/3) at the harmless winner. Within-world mean |cosine| agreement
(0.318–0.345, n=28 pairs each) sits essentially on top of the null (q95
0.3313) in all 3 worlds; CI lower bounds fall 0.0009–0.0260 short. The miss
is decisive in 2 of 3 worlds but a genuine knife-edge in the third
(`source_rotated_feedback`, CI lower bound only 0.0009 below q95 — the
closest of all 6 (arm, world) comparisons by nearly an order of magnitude;
the second-closest, 0.0015, is the SAME world at `deployed`). Point
estimates disagree in sign across worlds (2 of 3 marginally above q95, 1
below even at the point estimate), so the near-miss is not corroborated by
a second world. Worth naming: `source_rotated_feedback` is the same world
M4-H3 found totally non-responsive to the shrinkage lever and M4-H4
resolved as a strength effect rather than structural immunity — here it is
again the outlier, at both arms, without crossing the bar. **Lean (b) NOT
ACROSS WORLDS (the control) — HOLDS**: pooled across-world agreement
(0.3232, n=192) sits below the null's q95 (0.3313), CI entirely below by a
real but modest margin (0.0030, ≈58% of the CI's own half-width) — the
control behaves exactly as M4-E2's world-specificity finding predicted,
confirming lean (a)'s miss is not an artifact of the measurement picking up
shared, non-world-specific structure. **Lean (c) — not applicable** (no
world qualified under lean (a)).

**Per-repetition S4 shares** (a genuinely new quantity this leg computes,
disclosed as distinct from H5's own world-level, GPA-consensus shares) run
substantially higher than the world-level figures — 0.72–0.79 mean per-rep
share vs. 0.40–0.45 (deployed) / 0.51–0.52 (harmless) at the consensus
level — an expected contrast: a single repetition's own raw displacement
has not been denoised by GPA-averaging across 8 reps, so more of it falls
outside what the consensus-fit S1/S2/S3 bases explain.

**Mechanical notes.** A synthetic unit test of the new per-repetition delta
construction (random matrices, no real data) was run BEFORE any real
compute and directly verified: the symmetric-alignment identity (native
delta's norm exactly reproduces the quotient distance), the
rotation-extraction gate (exactly 0.0), the norm-preservation gate
(1.4e-14), the algebraic commutation identity `align(B@R,A@R) =
align(B,A)@R` this construction depends on (5.8e-15), and the
gauge-invariance/equivariance contrast that motivates aligning each
repetition's delta into the world's own consensus gauge before projecting
it (a toy gauge-invariant basis gave identical shares on native vs.
gauge-aligned delta, 0.0156170419530426 both ways; a toy gauge-equivariant
basis, mirroring S3's own n2 construction, gave different ones, 0.0257 vs.
0.0340, exactly as the math predicts). No mechanical problems arose in the
real run: smoke test, all 3 worlds' G3 stages, all 6 (world,arm) compute
stages, and assemble completed clean on the first attempt, entirely
foreground with explicit per-call timeouts (no background jobs, no
monitors), total compute wall-clock under 2.5 minutes. Data-integrity
checks (row counts for all 12 output CSVs against their expected 2×3,
2×3×8, 2×3×28, 2×3×64 and 2×200 shapes) passed before any hypothesis-
relevant number was read.

**Hand-off.** The M4-H basis line closes on this statement, exactly as the
registration's own PIVOT branch specifies: the safe lever (`basis_
shrinkage_1.00`) removes 45.79% of the discovery objective's frame
displacement without harming truth-referenced recovery (M4-H4) — a real,
bounded, partial repair, the first this program has had; what survives is
dominated by S4, a block defined purely by exclusion with no operational
construction of its own (M4-H5); and this leg shows, at a scale it was
well-powered to detect had it existed, that S4's own direction does not
even reproduce across independent repetitions of the SAME world — the
across-world control confirms this is a real absence of structure, not a
measurement artifact. **~46% is the M4-H line's definitive ceiling for
subspace-targeted, objective-side repair of the displacement.** Closing
that further would require an intervention past the discovery objective's
own S1–S4 decomposition entirely — outside anything this line's thirteen
legs (M4-G1 through M4-G7, M4-H1 through M4-H6) have tried, and not
registered or attempted here.

## M4-H6 planner adjudication note + BASIS LINE CLOSED (2026-08-03, appended)

**The pivot fires, decisively, and the M4-H line closes on it.** Lean (a)
MISSES in 0/3 worlds — S4's direction does not reproduce across 8 independent
repetitions of the same world (decisive in two; a genuine knife-edge at −.0009
in `source_rotated_feedback`). Lean (b), the control, HOLDS: across-world
agreement sits entirely below the null, exactly as M4-E2's world-specificity
predicts, which is what rules out "the measurement cannot see anything" as an
explanation for (a). Lean (c) is not applicable. Verdict
`PIVOT_S4_REPETITION_SPECIFIC_NOT_ATTACKABLE_BY_ANY_BASIS_CONSTRUCTION`.

**The null handling is why this result can be believed.** The registered
repetition-shuffled null sits at q95 = .3313 at the harmless winner, while the
disclosed random-matrix companion sits at q95 ≈ .46 — empirically confirming
M4-E2's warning that the two are not interchangeable. Against a zero null
everything would have looked significant; against the random-matrix null
everything would have looked trivially sub-null. Only the registered null
answers the question that was asked.

**Consequence, recorded as the discovery the registration said it would be:**
the surviving majority of the displacement is repetition-specific residual.
There is nothing stable there for a basis to capture, so ~46% is the
definitive ceiling of subspace-targeted, objective-side repair. The
displacement problem opened by Leg 14 is closed as a measured limit; the full
arc is written up in `docs/SUICA_DISPLACEMENT_PROBLEM_RESOLVED.md`.

## M4-J1 registration (2026-08-03, BEFORE run) — do the certified repairs generalize beyond the three worlds that produced them?

**Why this is the next leg and not a deployment.** Two repairs are certified:
M4-G6's per-column standardization at alpha = .10 x deployed ridge (exactly
c-invariant, recovery tied-or-better, scale-free interior optimum) and
M4-H3/H4's basis-whitening shrinkage (safe to ratio 1.0 at 45.79% displacement
reduction; actively good to ratio .20 at 34.66%). Adopting either changes a
frozen operator, which under F16 creates a NEW operator needing its own study
ID and seal. Both were measured on M4-E2's THREE worlds. Before that decision
is worth putting to anyone, the repairs need a generalization test — and if
they fail it, the decision does not arise.

**Design.** Apply both certified repairs on the WIDER world set already used
by this program (M4-F/M4-H2's eight worlds), with M4-E2's three carried inside
it as anchors. Arms: `deployed`; `colstd_alpha_0.10` (the G6 repair);
`basis_shrinkage_0.20` (actively-good); `basis_shrinkage_1.00` (harmless
ceiling). Metrics: Leg 14's displacement gap; both M4-F5 truth variants; and
c-invariance of the G6 repair across c in {0.25, 1.0, 4.0}.

**Leans.**
(a) THE REDUCTION GENERALIZES: the basis-shrinkage repair at ratio 1.0
    achieves >= 25% displacement reduction in >= 75% of the wider world set,
    paired-by-world CI excluding zero.
(b) SAFETY GENERALIZES: recovery does not worsen in ANY world under either
    repair (equivalence, ±.02, both variants) — a single world where it
    worsens is a material finding about deployment risk and must be named.
(c) INVARIANCE IS WORLD-INDEPENDENT: the G6 repair's c-invariance holds
    exactly in every world, testing whether M4-G5's reparameterization
    argument is a property of the construction rather than of the three
    worlds it was demonstrated on.

**PIVOT-IF:** the reduction fails to reach 25% in >= 75% of the wider set ->
the repairs are WORLD-SET-SPECIFIC. The M4-E2 three-world results stand as
scope-limited, no deployment decision arises, and the finding is that the
repair's generality was the untested assumption all along.

**Gates.** G0 POWER with justified grain and the MDE stated; G1 ANCHOR — on
M4-E2's three worlds every arm reproduces its persisted value to <=1e-12,
which is what makes the five new worlds interpretable as an extension rather
than a re-derivation; G2 REPAIR LIVENESS per arm per world; G3 truth-path
invariance; G4 materiality-form compliance per gate; G5 DUAL-WINNER reporting
retained where a winner is named.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_j1_repair_generalization/`; report
`reports/SUICA_M4_J1_REPAIR_GENERALIZATION_REPORT.md`.

## M4-J1 outcome (2026-08-03, appended)

**PIVOT DOES NOT FIRE. Lean (a) HOLDS (7/8 worlds, 87.5% >= the 75% bar).
Lean (b) MISSES decisively for the basis-shrinkage repair at BOTH certified
ratios, in two explicitly named worlds, but HOLDS for the G6 repair. Lean
(c) HOLDS EXACTLY, 6/6 worlds, 36/36 checks all at 0.0, not a near-miss.
Verdict:
`REDUCTION_GENERALIZES__SAFETY_FAILS_IN_NAMED_WORLD__INVARIANCE_WORLD_INDEPENDENT`.**
This is not an outcome outside the registered branches -- it is exactly the
combination the registration's three independent leans were built to allow.

No new basis-construction or estimator code was written: every arm is a
literal call into already-certified machinery (`h3._basis_for_h3_arm`,
`h4._basis_for_h4_arm`, `g6._resolve_all_params`/
`g6._forced_route_derivative_for_arm`, `g2._whitening_for_c`).
`colstd_alpha_0.10`'s displacement was not independently recomputed through
a second GPA pipeline -- it was ASSIGNED from `deployed`'s own disp_v2 per
(world, repetition), reusing M4-G7's own structural proof that the ridge
never enters `context["v2_basis"]`, with the precondition (basis@c=1.0 ==
deployed basis) verified bit-identical in all 8 worlds (64 checks, 0.0) before
the assignment was trusted.

**G0/G1/G2 first, as the standing rules require.** G1 ANCHOR is EXACT --
not merely inside 1e-12 but 0.0 -- across 9,376 checks on the three
`HIGH_GAP_WORLDS` (disp_v2 vs Leg 14/H3/H4/M4-G7's own persisted values;
truth recovery vs H3/H4/M4-G6's own persisted values, including all three
`colstd_alpha_0.10` c's against M4-G6's own 8-world file). This is what
licenses the five `FRESH_COMPANION_WORLDS` as an EXTENSION, not a
re-derivation. G2 REPAIR LIVENESS passes for all 16 (basis-arm, world)
cells (smallest materiality ratio 48.9%, ~5x the 10% bar) and for
`colstd_alpha_0.10` on both counts (ridge arithmetic exact 0.0; whitening
Frobenius norm scales with c to 8.88e-16). G3 TRUTH-PATH INVARIANCE passes
at exactly 0.0 over 48 checks. G0 POWER: this leg's own leans are worded
PER WORLD, forcing a materially less-powerful grain than every prior leg in
this line (paired-by-repetition n=8/world for lean (a); paired-by-author
n<=256/world/budget for leans (b)/(c)) -- disclosed plainly, with 2/8 lean-a
worlds and 30/36 lean-b comparisons flagged UNDERPOWERED against their own
bars, but every one of the 8 lean-b OUTSIDE classifications sits 2.68x-10.3x
past the +/-0.02 margin regardless, and every lean-c check is EXACT (0.0
half-width), so no adjudicated classification rests on marginal power.

**Lean (a) THE REDUCTION GENERALIZES holds at 87.5% (7/8 worlds)**, every
holding world's paired CI (n=8 reps) excluding zero by a wide margin
(smallest gap `condition_alias_ecology` [7.55,9.68]). The one miss,
`history_gated_ecology` (21.5% reduction, CI [-1.25,5.19], underpowered AND
below the point-estimate bar), is a visibly different world: its deployed
disp_v2 (7.01) is under half of every other world's (13.18-22.73), and the
ratio-1.00 repair barely moves it.

**Lean (b) SAFETY GENERALIZES misses decisively for the basis-shrinkage
repair, at BOTH `basis_shrinkage_0.20` (M4-H3's own actively-good setting)
and `basis_shrinkage_1.00` (M4-H4's own harmless-ceiling setting), in
exactly two worlds, both truth budgets, all eight (arm,world,budget) cells
classified OUTSIDE the +/-0.02 "does not worsen" margin by 2.68x-10.3x --
`history_gated_ecology` (+.148/+.148 at ratio 0.20; +.206/+.206 at ratio
1.00) and `topology_mismatch` (+.054/+.056 at ratio 0.20; +.109/+.111 at
ratio 1.00), harm larger at the more aggressive ratio in both worlds, the
same target-vs-safety direction this program has found everywhere else, now
crossing worlds rather than crossing strength.** These two worlds are named
explicitly per the registration's own instruction, not averaged into an
overall pass. Elsewhere the basis-shrinkage repair is safe or genuinely
ambiguous at this leg's own per-world grain (11/24 WITHIN, 5/24 AMBIGUOUS of
the remaining cells). **`colstd_alpha_0.10` worsens in ZERO of the 6 worlds**
(8/12 WITHIN, 4/12 AMBIGUOUS, 0/12 OUTSIDE) and shows large, decisive
IMPROVEMENT in the identical two worlds that break the other repair
(history_gated_ecology -.22/-.24; topology_mismatch -.055/-.074) -- lean (b)
HOLDS for the G6 repair specifically.

**Lean (c) INVARIANCE IS WORLD-INDEPENDENT HOLDS EXACTLY**: every one of 36
checks (6 valid worlds x 2 budgets x 3 c-pairs) shows mean diff exactly 0.0,
CI [0.0,0.0] -- confirmed not a pooled statistical artifact but a row-level
algebraic identity (a sampled cell's `e_arm_true` is bit-identical,
`0.105885`, at c=0.25/1.0/4.0). M4-G5's reparameterization argument is a
property of the per-column-standardization construction, not of the three
worlds it was first shown on.

**Pivot does not fire** (requires lean (a) to miss; it holds at 87.5%). One
mechanical problem, caught and fixed before any hypothesis-relevant number
existed: an unquoted shell variable containing a space
(`/Volumes/.../project persona/.venv/bin/python3`) broke word-splitting on
the first parallel-launch attempt, failing all four worker processes on
their first command before any context was built; fixed by quoting, batch
re-launched clean. ~22 minutes total foreground compute (one world run
singly for timing calibration, then two batches of 4 and 3 worlds
concurrent on 10 cores, matching this program's established internal-
concurrency-inside-a-foreground-call precedent), zero Tracebacks or
RuntimeErrors on any computational content.

Full numbers, gates, the per-arm x per-world tables for both metrics, and
the c-invariance evidence:
`reports/SUICA_M4_J1_REPAIR_GENERALIZATION_REPORT.md`. Artifacts:
`results/m4_j1_repair_generalization/{decision.json, gates.json,
disp_rows.csv, truth_recovery_rows.csv, author_level_truth_rows.csv,
g2_liveness_rows.csv, colstd_ridge_liveness_rows.csv, g3_check_rows.csv,
lean_a_reduction_rows.csv, lean_b_safety_rows.csv,
lean_c_invariance_rows.csv}`.

**Hand-off.** The two certified repairs no longer stand or fall together.
`colstd_alpha_0.10` (M4-G6) generalizes cleanly on every axis tested here --
displacement-neutral by structural proof, exactly c-invariant world-
independently, and safe-to-actively-good in every one of the 6 worlds where
safety is measurable -- and is the stronger candidate for the F16
new-operator-ID deployment step this line has deferred since M4-G6. The
basis-shrinkage repair (M4-H3/H4) generalizes on its target but carries a
now-named, well-powered, two-world material safety risk that its original
3-world certification could not have surfaced (neither `history_gated_
ecology` nor `topology_mismatch` is a `HIGH_GAP_WORLDS` member) -- adopting
it as currently specified, at either certified ratio, is not supported by
this leg's own evidence without first understanding why those two worlds
respond oppositely to the lever. That mechanism question -- what
`history_gated_ecology` and `topology_mismatch` share that the three
original `HIGH_GAP_WORLDS` do not -- is the open question this leg leaves,
not resolved or speculated on here.

## M4-J1 planner adjudication note (2026-08-03, appended) — the two repairs no longer stand or fall together

**Adjudication.** Lean (a) HOLDS: 7 of 8 worlds clear 25% reduction (87.5%
against a 75% bar), the sole miss being `history_gated_ecology` at 21.5% with
a CI including zero — a world whose baseline displacement (7.01) is under half
of every other world's. Lean (c) HOLDS EXACTLY: 36/36 c-invariance checks at
0.0 in 6/6 worlds, verified row-level, so **M4-G5's reparameterization
argument is a property of the construction and not of the three worlds it was
demonstrated on**. Pivot does not fire. G1 anchored at exactly 0.0 across
9,376 checks, which is what makes the five new worlds an extension rather than
a re-derivation.

**Lean (b) splits the two repairs, and that is the leg's real content.**

- `colstd_alpha_0.10` (the M4-G6 repair) worsens recovery in **zero** of six
  worlds and improves it sharply in two.
- `basis_shrinkage_0.20` and `basis_shrinkage_1.00` (the M4-H3/H4 repair) both
  worsen recovery decisively — 2.7x to 10.3x past the margin, at both budgets
  — in **`history_gated_ecology` and `topology_mismatch`**, with the harm
  LARGER at the more aggressive ratio. Neither world is a `HIGH_GAP_WORLDS`
  member, so the original three-world certification could not have surfaced
  this.

**And the two repairs are anti-correlated exactly where it matters.** In those
same two worlds, basis-shrinkage drives recovery error UP (.459 -> .607/.665;
.309 -> .362/.418) while the G6 ridge repair drives it sharply DOWN
(.459 -> .239; .309 -> .253). The worlds where one repair is most harmful are
the worlds where the other is most helpful.

**Deployment picture, stated plainly.**

- **M4-G6's `colstd_alpha_0.10` generalizes essentially without
  qualification** — exact c-invariance everywhere, no measured harm anywhere,
  displacement unchanged by structural proof. It is the strong candidate for
  the F16 new-operator-ID step, and that step is a decision for the operator,
  not for this loop.
- **M4-H3/H4's basis-shrinkage repair is NOT supported for adoption as
  specified.** It generalizes on its target and fails on safety, in named
  worlds, at both certified ratios.

The executing agent declined to speculate about what those two worlds share.
That restraint is right, and it defines the next leg.

## M4-J2 registration (2026-08-03, BEFORE run) — is the harm boundary predictable, or is the basis repair undeployable in principle?

**Question.** What distinguishes `history_gated_ecology` and
`topology_mismatch` from the six worlds where the basis-shrinkage repair is
safe? This is not idle taxonomy: if the harm boundary is predictable from a
property measurable BEFORE applying the repair, the repair can be salvaged
behind a gating rule. If it is not, then one cannot know in advance whether a
new corpus is a harm case, and the repair is undeployable in principle
regardless of how well it performs on average.

**Part 0 discriminator inventory (registered before compute, fixed as the
hypothesis space).** Measure each of the following at `deployed` on all eight
worlds: (i) baseline recovery error; (ii) baseline displacement magnitude;
(iii) S3 share; (iv) S4 share; (v) an effective-rank / conditioning statistic
of the basis. Candidates discovered later may be reported but not scored.

**Leans.**
(a) BASELINE COMPETENCE PREDICTS HARM: baseline recovery error at `deployed`
    correlates with the basis-shrinkage repair's recovery BENEFIT across the
    eight worlds (Spearman, CI excluding zero) in the direction that means
    the repair helps where the objective does badly and harms where it
    already does well. Motivated directly by the observation that the two
    harm worlds carry the two lowest baseline errors of the eight.
(b) IT SEPARATES CLEANLY: the two harm worlds are exactly the two
    lowest-baseline-error worlds — a rank separation, not merely a
    correlation.
(c) DISPLACEMENT MAGNITUDE IS NOT THE DISCRIMINATOR: baseline displacement
    does NOT separate the harm worlds (`history_gated_ecology` is the lowest
    at 7.01 but `topology_mismatch` is mid-range at 13.18). A specificity
    check that the discriminator is competence rather than displacement size.

**PIVOT-IF:** no discriminator in the registered inventory separates the two
harm worlds from the six safe ones -> THE HARM BOUNDARY IS NOT PREDICTABLE
from any measured world property available before applying the repair. The
basis-shrinkage repair is then undeployable in principle, not merely
unsupported as specified, and that is the finding — to be written with the
prominence this line gives its measured limits.

**Gates.** G0 POWER with justified grain and MDE stated (n=8 worlds is small;
say so and state what it can and cannot detect); G1 ANCHOR — every reused
quantity reproduces M4-J1's persisted values to <=1e-12; G2 DISCRIMINATOR
LIVENESS — each candidate must actually vary across the eight worlds (a
constant discriminator is VACUOUS, not a null); G3 truth-path invariance where
applicable; G4 materiality-form compliance per gate.

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_j2_harm_boundary/`; report
`reports/SUICA_M4_J2_HARM_BOUNDARY_REPORT.md`.

## M4-J2 outcome (2026-08-03, appended)

**Executed as registered.** No new basis-construction or estimator code was
written. Discriminators (i) baseline recovery error and (ii) baseline
displacement were read directly (not recomputed) from M4-J1's own persisted
`author_level_truth_rows.csv`/`disp_rows.csv` — identity by construction.
Discriminators (iii) S3 share and (iv) S4 share were computed by a literal,
unchanged call into `h2._arm_offset_and_shares` (M4-E2/M4-H2's own S1-S4
decomposition), extended to the 5 `FRESH_COMPANION_WORLDS` H2 itself never
touched (H2 is a 3-world leg) via the identical S1/S2-pattern-then-shares
recipe H2's own `_run_arm` uses. Discriminator (v) — a condition-number
statistic of the retained eigenspectrum — was computed via a literal,
unchanged call into `h2._ingredients_for_arm(context, "deployed")`, the exact
ingredients function H3/H4's shrinkage dispatch consumes for every ladder
rung, independently cross-checked per repetition against
`leg10._freeze_ingredients` (a separately-written copy of the same
eigendecomposition) at exactly 0.0 abs diff over all 64 (world, repetition)
pairs. All 8 worlds computed in under 2 minutes total foreground wall-clock,
reusing M4-J1's own cached contexts read-only; zero mechanical problems.

**G0/G2 first, as the outer task required.** G0: n=8 `D1_WORLDS`, n=6
`VALID_TRUTH_WORLDS` for every truth-derived quantity — small, disclosed
explicitly along with what this grain can and cannot detect. Rank-2
separation is an EXACT, fully-powered fact at n=6 (not a sampling estimate);
a 6-point nonparametric bootstrap over Spearman's rho has only
**C(11,6) = 462 distinct resamples**, disclosed rather than hidden behind
20,000 draws — CIs that include zero are reported UNDERPOWERED, never as
confident nulls (the conditioning-number correlation's realized CI spans the
entire possible range, [-1.0, 1.0]). G2: all 5 registered discriminators
vary materially across the 8 worlds (registered bar: range ≥5% of median
|value|) — **no discriminator is VACUOUS.**

**G1 ANCHOR: PASS, stronger than M4-J1's own self-anchor.** Recomputed
`disp_v2` (a side product of the S3/S4 machinery) matches M4-J1's persisted
`disp_v2` at **exactly 0.0** over 64 checks, on **all 8 worlds** — not merely
the 3 `HIGH_GAP_WORLDS` M4-J1 anchored itself against, since M4-J1's own file
already covers all 8. S3/S4 shares and `offset_norm` match H2's persisted
`offset_shares_by_arm.csv` to 5.55×10⁻¹⁷ on the 3 `HIGH_GAP_WORLDS` (H2 never
computed the 5 fresh worlds — this is what licenses them as an extension).
G3: N/A, stated as such — this leg introduces no new truth-recovery
computation. G4: every gate/lean is an equivalence, exactness, rank-gap, or
CI-exclusion bound; compliance stated per gate in the report.

**Lean (a) BASELINE COMPETENCE PREDICTS HARM HOLDS**: Spearman rho = **0.886**
between deployed baseline error and `basis_shrinkage_1.00`'s recovery benefit
across `VALID_TRUTH_WORLDS` (budget 4.0), bootstrap CI **[0.185, 1.000]**,
excluding zero, in the registered direction (low baseline error → repair
harms; high baseline error → repair helps). Consistent across budget 8.0
(identical) and the milder `basis_shrinkage_0.20` ratio (rho=0.771, CI
[0.000,1.000] — same direction, a boundary-case companion).

**Lean (b) IT SEPARATES CLEANLY HOLDS**: `history_gated_ecology` (0.4588)
and `topology_mismatch` (0.3087) are exactly the two lowest baseline-error
worlds of six; the next-lowest safe world sits at 0.5527 — gap 0.0939, the
safe-world minimum 20.5% above the harm-world maximum. Exact, fully-powered
at this n.

**Lean (c) DISPLACEMENT MAGNITUDE IS NOT THE DISCRIMINATOR MISSES on its own
literal criterion — a genuine, disclosed, not-softened finding.** Applying
lean (b)'s own rank test to displacement, full-precision recomputation (not
the rounded report-table values the registration's motivating text was read
from) shows `history_gated_ecology` (7.008) and `topology_mismatch` (13.182)
are ALSO exactly the two lowest of six — gap 3.747, safe-world minimum 28.4%
above the harm-world maximum, a **larger** relative gap than baseline
error's own 20.5%. This directly contradicts the registration's own framing
("`topology_mismatch` is mid-range at 13.18"). The correlation companion
offers partial, not decisive, specificity: displacement's benefit-correlation
CI touches exactly 0.000 at its lower bound (rho=0.829) — a boundary case,
weaker than baseline error's comfortable 0.185, but not a clean miss either.
**Net: a 2-of-6 rank coincidence between two discriminators is, on its own,
weak evidence at n=6; this leg reports the miss plainly rather than
reframing lean (c) around the more favorable correlation reading alone.**

**The other three registered items show no signal at all.** S3 share, S4
share, and the conditioning-number/effective-rank pair separate by
**neither** the rank test nor the correlation test — the conditioning-number
correlation's CI spans the entire possible range, genuinely uninformative at
this n, reported as UNDERPOWERED rather than a confident null.

**PIVOT DOES NOT FIRE.** The registered condition requires every
discriminator to fail; baseline recovery error alone already separates,
decisively, by both tests — a single, pre-registered, primary, directional
test, sufficient on its own to resolve the pivot without needing the other
four items' both-direction, multiplicity-exposed checks. **The harm boundary
is predictable from at least one measured, pre-repair-applicable world
property; the basis-shrinkage repair is NOT undeployable in principle** in
the sense this leg's registered pivot tests.

**Verdict**:
`HARM_BOUNDARY_PREDICTABLE_BUT_NOT_SPECIFIC_TO_COMPETENCE__PROVISIONAL_GATE_ONLY`
— the honest middle finding the three independent leans were built to allow:
(a) HOLDS, (b) HOLDS, (c) MISSES on its own literal criterion. Not softened
toward either a clean "gateable" story or a clean "undeployable" one.

**Hand-off.** A gate of the shape "do not apply the basis-shrinkage repair
where deployed baseline recovery error is low" is directly, if provisionally,
supported — rank-exact and correlation-CI-excluding-zero at the primary
combination, direction-consistent across every budget/ratio combination
tested. But this leg cannot distinguish "low baseline error" from "low
baseline displacement" as the operative mechanism (the two discriminators are
too entangled in this 6-world sample), and the underlying question M4-J1's
own hand-off left open — WHY these two specific worlds respond adversely to
the shrinkage lever — remains unanswered. A gate built on this evidence is a
reasonable provisional heuristic, not a certified deployment control; it
needs validation on a larger, independent world set (one that can vary
baseline competence and baseline displacement independently) before being
trusted for a real F16 deployment decision.

Full numbers, gates, the Part 0 inventory, and the full 5-discriminator ×
rank+correlation table: `reports/SUICA_M4_J2_HARM_BOUNDARY_REPORT.md`.
Artifacts: `results/m4_j2_harm_boundary/{decision.json, gates.json,
part0_inventory.csv, full_inventory_separation_table.csv,
disp_v2_recomputed_rows.csv, g1_share_anchor_rows.csv, benefit_rows.csv,
per-world partials}`.

## M4-J2 planner adjudication note (2026-08-03, appended) — the gate exists, is not specific, and my own motivating fact was wrong

**Adjudication.** Lean (a) HOLDS: baseline recovery error at `deployed`
correlates with the basis-shrinkage repair's recovery benefit
(Spearman .886, CI [.185, 1.000], correct direction, 3 of 3 companions
positive and 2 of 3 decisively excluding zero). Lean (b) HOLDS: an exact
rank-2 separation with a 20.5% margin. Pivot does NOT fire. Verdict
`HARM_BOUNDARY_PREDICTABLE_BUT_NOT_SPECIFIC_TO_COMPETENCE__PROVISIONAL_GATE_ONLY`.

G0 is the model of how this line should state power: full power for one thing
(a rank-2 separation is deterministic once six values are measured — no
sampling uncertainty) and near-zero for another (a Spearman CI at n=6 admits
only C(11,6)=462 distinct resamples; one discriminator's CI spans the entire
[-1,1]). Where a CI included zero it was reported UNDERPOWERED, never as a
null. G1 anchored `disp_v2` at exactly 0.0 across 64 checks on all eight
worlds — stronger than M4-J1's own three-world self-anchor.

**Lean (c) MISSES, and the reason is my error, not the leg's.** I registered
the specificity control on the claim that displacement does not separate the
harm worlds, citing "`history_gated_ecology` is lowest at 7.01 but
`topology_mismatch` is mid-range at 13.18". Checked at full precision against
the persisted table, that is simply false: the eight worlds' displacements are
18.22 / 16.93 / 18.61 / 22.73 / 18.33 / **7.01** / 18.69 / **13.18**, so the
two harm worlds are the two LOWEST of the eight, and displacement rank-
separates them by an even larger relative gap (1.284) than baseline error's
own (1.205). The control I wrote to establish specificity instead demonstrated
its absence, and the agent said so plainly rather than softening it.

**Eighth planner defect, and a new kind.** The first seven were defects in
RULES — aggregation, power, channel, materiality form, grain, winner
definition, graded levels. This one is a defect in EVIDENCE: a factual claim
used to motivate a lean, not verified against the persisted data at full
precision before the registration was committed. **Standing rule (eighth):
every factual claim cited to motivate a lean must be checked against the
persisted artifacts at full precision before the registration is committed. A
wrong motivating fact does not merely weaken a leg — it makes the control test
the wrong thing.**

**Where this leaves deployment.**

- The basis-shrinkage repair is **NOT undeployable in principle**: a real,
  measured, pre-application gating signal exists.
- But the gate is **not certified**: baseline competence and baseline
  displacement are comparably predictive at n=6 and cannot be told apart as
  distinct mechanisms, while S3 share, S4 share and conditioning number show
  no signal at all. A competence-based gate is a reasonable provisional
  heuristic, not a deployment control.
- `colstd_alpha_0.10` (the M4-G6 repair) remains unaffected by any of this and
  is still the clean F16 candidate.

## M4-J3 registration (2026-08-03, BEFORE run) — can the two candidate gates be told apart?

**Question.** Baseline recovery competence and baseline displacement are
confounded across the eight worlds available: both rank-separate the harm
worlds, neither can be credited. Can they be DISSOCIATED — and if so, which
one actually predicts harm?

**Design.** Search the program's existing world generators for, or construct
by registered parameter variation, worlds that dissociate the two: HIGH
baseline error with LOW displacement, and LOW baseline error with HIGH
displacement. Apply `basis_shrinkage_0.20` and `basis_shrinkage_1.00` to them
and record whether harm follows competence or displacement. Anchor all eight
existing worlds unchanged.

**Leans.**
(a) DISSOCIATION ACHIEVED — a gate on the leg itself: at least two worlds
    exist or are constructible in which the two discriminators rank
    OPPOSITELY, verified before the harm outcome is computed. If dissociation
    cannot be achieved, the leg adjudicates nothing further and says so: the
    two candidates are not separable with the machinery available, and the
    gate stays provisional.
(b) COMPETENCE WINS: in the dissociating worlds, harm follows baseline
    recovery error rather than baseline displacement.
(c) THE GATE IS USABLE: a single threshold on the winning discriminator
    classifies every world — the original eight plus the new ones — correctly,
    with the threshold registered from the original eight BEFORE the new
    worlds' outcomes are computed.

**PIVOT-IF:** harm follows DISPLACEMENT rather than competence, or neither
predicts in the dissociating worlds -> the provisional gate's stated basis is
wrong. Either the correct gate is displacement (in which case say so and
re-specify), or the boundary is not a single scalar at all and no gate can be
certified from this design.

**Gates.** G0 POWER with the dissociating-world count stated and its limits
acknowledged; G1 ANCHOR — the eight existing worlds reproduce M4-J1/M4-J2's
persisted values to <=1e-12; G2 DISSOCIATION LIVENESS — the constructed
worlds' two discriminators must genuinely rank oppositely, shown before any
harm outcome is read; G3 truth-path invariance; G4 materiality-form
compliance; **G6 MOTIVATING-FACT VERIFICATION (new, from the eighth standing
rule) — every factual claim this registration cites must be re-derived from
the persisted artifacts at full precision and reported, including the
displacement table above.**

Tier: EXPLORATORY, label-free, synthetic. Artifacts:
`results/m4_j3_gate_dissociation/`; report
`reports/SUICA_M4_J3_GATE_DISSOCIATION_REPORT.md`.

## M4-J3 outcome (2026-08-03, appended)

**DISSOCIATION NOT ACHIEVED. Lean (a) MISSES cleanly under the leg's primary
reading (0 of 3 pairwise comparisons among the usable new worlds dissociate,
at either the bare or a registered 10% materiality bar). Per the
registration's own instruction, the leg adjudicates nothing further: leans
(b) and (c) are explicitly NOT ADJUDICATED, and the pivot's own precondition
("in the dissociating worlds") is unsatisfied, so it neither fires nor
cleanly does not fire. Verdict:
`DISSOCIATION_NOT_ACHIEVED__GATE_STAYS_PROVISIONAL`.** This is the leg's own
explicitly-sanctioned honest-negative branch, not an outcome outside the
registered leans -- the registration named this exact possibility ("the two
candidates are not separable with the machinery available... an acceptable,
honest outcome") and it is what happened.

**G0/G2/G6 first, as the standing rules require.** G0: a pairwise rank
inversion is a deterministic fact once both worlds' deployed-arm values are
measured (no sampling uncertainty), but a BARE inversion can be noise-level,
so this leg required both axes' relative gap to clear a 10%-of-value
materiality bar (reused unchanged from this program's own
`G2_MATERIALITY_RATIO` convention) before crediting a dissociation --
disclosed alongside the unfiltered count throughout. `found_slow_hysteresis_
equal_marginal` was confirmed pathological exactly as anticipated (baseline
error 7.91e9, the identical order-of-magnitude signature as M4-J1/M4-J2's
own `EXCLUDED_TRUTH_WORLDS`), leaving 3 usable new worlds and 3 primary
pairwise comparisons. **G2: 0/3 dissociate.** Both constructed hybrids'
displacement values track their MECHANISM donor's own native value almost
exactly (within 1-2%; e.g. `hybrid_lo_error_hi_disp` = mechanism(topology_
mismatch)@seed(condition_alias_ecology) measured (0.3252, 13.349) against
topology_mismatch's own native (0.3087, 13.182)) rather than moving toward
the seed-donor's extreme -- an honest empirical finding stated as a
hypothesis in Part 0.1 (unlike M4-G2's `offset_norm`, `disp_v2` compares
against `oracle_basis`, which branches on mechanism identity, not seed
alone) and now falsified for this specific lever. A disclosed,
non-adjudicating companion (full population: 8 `D1_WORLDS` with
`EXCLUDED_TRUTH_WORLDS` screened out + 3 usable new worlds, 36 pairs) finds
exactly 2 materially-dissociating pairs, both anchored on `history_gated_
ecology` -- but both against worlds ALREADY KNOWN (or shown, for the new
member) to share the identical HARM outcome (`history_gated_ecology` vs
`topology_mismatch`: M4-J1's own two harm worlds, compared against each
other; `history_gated_ecology` vs `hybrid_lo_error_hi_disp`: the hybrid
inherits topology_mismatch's own harmed status, +0.1378 vs +0.1091 at the
primary combination, same sign, comparable magnitude) -- both classify
`BOTH_HARMED` at all 4 tested (arm,budget) combinations, so even the most
permissive reading of "two worlds exist" yields nothing lean (b) can use. An
earlier draft of this companion check omitted the `EXCLUDED_TRUTH_WORLDS`
screen on the anchor side and found 11 "dissociating" pairs, 6 spurious
(driven by `fast_return_equal_marginal`/`linear_null_ecology`'s own ~1e9
pathology); caught and fixed in the same assembly pass before the corrected
result was used for anything. **G6: every cited fact verified CORRECT** --
unlike M4-J2's own motivating-fact error, this registration's eight-world
displacement table (`18.22/16.93/18.61/22.73/18.33/7.01/18.69/13.18`)
matches the persisted `disp_v2` values 1:1 at full precision with no
mismatches, and both cited rank-separation percentages re-derive within
0.05 points (error 20.46% vs cited 20.5%; displacement 28.43% vs cited
28.4%). `any_cited_fact_found_wrong = False`.

**G1 ANCHOR passes exactly** (8 `D1_WORLDS` read directly from M4-J1's own
files, identity by construction, PLUS an independent one-repetition
self-validation of this leg's own new mechanism/seed-donor context-builder
against M4-J1's persisted `history_gated_ecology` rep-0 value: disp_v2 diff
0.0, truth diff 2.22e-16 across 16 authors). **G3 passes exactly** (0.0 over
12 checks, 4 new worlds x 3 `BASIS_ARMS`). **G4 compliant**, stated per
gate.

**Order of operations, proven by file timestamp, not merely asserted.**
`dissociation_evidence.json` (G2/lean(a), deployed-arm only) was written at
2026-08-08T15:35:28 UTC; the earliest `partial_harm_*.csv` (the shrinkage
arms' harm outcome) was written at 2026-08-08T15:39:35 UTC -- a 4-minute
gap, and all 16 harm partials postdate the evidence file
(`evidence_predates_every_harm_partial = True`). `--stage harm_arms` was
first invoked only after `DISSOCIATION NOT ACHIEVED` had already printed,
computed solely to disclose the ONE potentially-informative companion pair
identified above, never to reverse the already-final lean (a) verdict.

**Two mechanical findings, both caught and disclosed before any
hypothesis-relevant number existed.** (1) `hybrid_lo_error_hi_disp`
(mechanism=topology_mismatch @ seed-family=condition_alias_ecology) hit
`chart.refused` at repetition 4 specifically -- an anticipated-in-principle
but not anticipated-in-location risk (Part 0.1 disclosed the construction's
fidelity was empirical, not assumed) -- resolved by catching the refusal
per-repetition and continuing at n=7 reps for that label alone, disclosed in
a `skipped_repetitions_*.json` file. (2) The full-population companion
check's pathology-screen omission, described above.

**Pivot: NOT APPLICABLE.** Its own registered precondition ("in the
dissociating worlds") is unsatisfied with zero dissociating worlds in the
primary population -- it neither fires (would need a clean miss inside an
existing population) nor cleanly does-not-fire (would need lean (a) to have
held). This is the same shape M4-G3's own pivot took when underpowered,
except this leg is NOT underpowered (G0's rank-inversion criterion is
deterministic and fully powered) -- it simply found no dissociation to test
the pivot's condition against.

Full numbers, gates, the full world x discriminator x harm table, the G6
re-derivation, and the disclosed companion analysis:
`reports/SUICA_M4_J3_GATE_DISSOCIATION_REPORT.md`. Artifacts:
`results/m4_j3_gate_dissociation/{decision.json, gates.json,
dissociation_evidence.json, dissociation_table.csv,
dissociation_pairwise*.csv, g1_selfcheck.json,
world_discriminator_harm_table.csv, anchor_deployed_table.csv,
anchor_harm_table.csv, new_world_deployed_table.csv,
new_world_harm_rows.csv, lean_b_competence_rows.csv,
lean_b_companion_full_population_rows.csv,
lean_c_threshold_classification_rows.csv, g3_check_rows.csv, per-world
partials}`.

**Hand-off.** M4-J2's own closing position stands entirely unchanged: the
basis-shrinkage repair's harm boundary is predictable but not specifically
attributable, a competence-and/or-displacement-keyed rule remains a
reasonable heuristic screen, and no certified deployment gate exists.
`colstd_alpha_0.10` (M4-G6) remains the clean, unaffected F16 candidate.
This leg's own contribution is negative but load-bearing: it closes off the
mechanism/seed-donor crossing as a viable lever for dissociating the two
candidates (displacement is overwhelmingly mechanism-driven, not
seed-driven, for this metric -- the opposite of what M4-G2 found for
`offset_norm`), and it surfaces, for the first time, that the one genuine
dissociation latent in the ALREADY-PUBLISHED 8-world data
(`history_gated_ecology` vs `topology_mismatch`) is exactly the pair that
cannot resolve the question, because both are already known to respond
identically. If this question is pursued further, the next registration
needs either a materially different construction lever, or must confront
directly the possibility that competence and displacement are not
separable scalars in this objective's world space at all.
