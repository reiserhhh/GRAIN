# GRAIN M4-G — The Objective Redesign Line, Closed

Status: **line closed 2026-08-03**, seven registered legs (M4-G1..M4-G7).
Tier: EXPLORATORY, label-free, synthetic throughout. The claims ledger
controls. Companion: `docs/SUICA_M4_G_OBJECTIVE_REDESIGN_PLAN.md` (every
registration and outcome in order), `docs/SUICA_M4_F_PANEL_DESIGN_SYNTHESIS.md`
(the preceding line).

---

## 1. What the line was opened for

Two arcs converged: the panel side was closed (M4-F — a finite panel can be
made self-consistent and cannot be made trait-certifying), and M4-E2 had
handed over the discovery objective's frame displacement as "distributed,
world-specific, largest single piece the residual at .40–.45, no single term
removable", naming the freeze whitening's unregularized 1/sqrt(eig)
amplification as the largest identifiable carrier. If a trait-level object was
reachable at all, the remaining lever was the objective.

## 2. What was measured

| leg | question | answer |
|---|---|---|
| G1 | is the scale family an actionable lever? | COSMETIC. Regularizing it cuts the offset (52.8% at the one powered arm) but truth recovery collapses at the offset-minimizing setting; `identity` — no whitening at all — has the LOWEST offset of eight arms while being 56% worse at recovery; Spearman(offset, error) = −.786. |
| G2 | is the offset metric measuring the world or its own units? | **UNITS.** Log-log slope .8796 [.8386,.9206] under a manipulation whose eigenvectors, relative spectrum, condition number and width were invariant at exactly 0.0. Raw `offset_norm` DISQUALIFIED as an optimization target; the scale-normalized form flips Spearman to +.833 and moves `identity` from minimum to maximum. Its lean-(b) miss was the lever: recovery is not scale-free (.786→.422), so something downstream compares a scale-carrying quantity to a fixed constant. |
| G3 | where is that constant? | `hazard_ridge` (deployed .005, `suica_core/m4_chart_ecology_estimator.py:341-342`), on four converging lines, with the other five inventoried constants cleanly inert. Its adaptive arm recovered ~90% of the c=4 gain — but keyed to a PRE-whitening statistic, so it could not follow c: context-adaptive, not scale-adaptive. |
| G4 | does a c-covariant scalar ridge fix it? | No — pivot fires cleanly. A scalar covariant ridge closes 89.1%; the residual 10.9% is structural, because `_hazard_design` mixes c-scaled columns (whitened basis) with c-invariant ones (raw `generated_current`/`duration`; `feedback_0_d`/`gate_0_d` crossed with the unscaled intercept). No scalar ridge can be scale-consistent across columns that do not share a scale. |
| G5 | does per-column regularization fix it? | **Exactly.** `column_standardized` bit-identical across c (0.0), `diagonal_ridge` to 1–4e-14, backed by a pre-registered provable reparameterization argument; the 10.9% residual closes to 0.0%. Column inventory: across 1,050 columns x 8 worlds x 4 models every column is exactly degree-0 or degree-1 in c, never mixed — the heterogeneity is exactly binary, which is why the fix is exact. But recovery did not improve: the arms calibrated to DEPLOYED strength, while G3's gain came from 4.6–18% of it. Invariance is the ridge's SHAPE; recovery is its STRENGTH. |
| G6 | shape and strength together? | **The repair.** All three leans hold. At alpha = .10 x deployed, `column_standardized` is exactly c-invariant AND ties the best prior recovery (.5083/.4946 vs .5068/.4949, ahead at 8x). Invariance holds exactly at all five strengths; the error-minimizing alpha is the SAME at every c and is an INTERIOR optimum — so the tuning is scale-free too, not just the output. |
| G7 | does the repair reach the displacement the line was opened for? | **No — and the zero is structural, not statistical.** Proved in Part 0 before compute and confirmed empirically at exactly 0.0: the repair's ridge parameter never enters `context["v2_basis"]`, the object both displacement metrics are computed from. Leg 14's gap closes by exactly 0%. |

## 3. The result

**Won, and certified:** the discovery objective's scale dependence is fully
diagnosed and repaired. The metric that measured it was itself disqualified
and replaced; the responsible constant was localized against five inert
alternatives; the reason a scalar fix could not work was named (binary column
heterogeneity) and proved; and the working fix — per-column standardization at
alpha = .10 x deployed ridge — is exactly scale-invariant, recovers as well as
the best prior tuning, and has a scale-free optimum.

**Not won, and now precisely located:** the frame displacement is untouched,
by construction. It does not live in the ridge or the whitening scale — the
entire territory this line worked in — but in **basis construction**
(`context["v2_basis"]`). That is a hand-off, not a shrug: seven legs of
elimination turned "distributed, world-specific, no single term removable"
into "not here, and here is where it must be instead".

**One correction the line owes the record:** M4-E2's decomposition attributed
~1/3–1/2 of the identifiable carrier mass to the scale family using the RAW
offset, which G2 disqualified. That attribution is now a units statement of
unknown structural content and must be recomputed under the normalized metric
before the basis line builds on it. That recomputation is the next
registration.

**Scope.** Synthetic worlds, EXPLORATORY tier. The repair is a property of the
deployed objective's conditioning; it licenses no claim about any corpus,
construct, person, or diagnosis.

## 4. Method: what the line cost and what it bought

Two further planner registration defects occurred and are recorded in the plan
document rather than repaired away: G3's registration demanded a pre-stated
MDE but not a GRAIN, so the line's default (n=6–8 worlds) was inherited though
it was >4x under the bar while the author grain (n≈745) was available; and
G1's 25% actionable bar selected exactly the arms that hurt recovery, while
the two mildest arms — the only ones that genuinely improved it — fell just
below the bar (shrinkage .1 missed by .14 points). The first produced the
fifth standing rule (justify the analysis grain for power; do not inherit it);
the second is recorded as a registration critique with no lean re-scored on it.

Executing agents held the line's standard throughout: G3 reported an outcome
that fit no registered branch rather than picking a side; G4 added a
specificity control against its own hypothesis; G5 and G7 each disclosed a
mechanical bug with an explicit statement of whether it was resolved before or
after any hypothesis-relevant number existed; G7 proved its own null
structurally in Part 0 before running, which is why its zero is a mechanism
rather than a measurement.

## 5. What is now open

1. **Re-decompose the displacement under the scale-normalized metric** — the
   correction M4-G2 forces on M4-E2's attribution, and the necessary first
   step of any basis-construction work. Registered as the next leg.
2. **Basis construction** (`context["v2_basis"]`) — where G7 proved the
   displacement must live.
3. **Deploy or shelve the repair** — the certified fix is a change to a frozen
   operator, so adopting it creates a NEW operator under F16 and needs its own
   study ID and seal. Not done here, deliberately.

---

## Dated correction (2026-08-03, appended after M4-H1)

Section 3 of this synthesis said M4-E2's attribution "is now a units statement
of unknown structural content and must be recomputed under the normalized
metric". **That was overcautious, and M4-H1 settled it: the attribution is
units-invariant by an identity, proved before compute and confirmed at
2.22e-16.**

M4-G2's normalized offset is `offset_norm / GM(s)` — a per-world SCALAR
division of an already-computed offset. Every M4-E2 statistic is
`||component||^2 / ||Delta||^2` against bases (S1/S2/S3) that do not depend on
Delta. Projection is linear, so every share is homogeneous of degree 0 in any
scalar applied to Delta, and raw and normalized shares are forced identical.

The distinction this draws is worth keeping: **M4-G2 disqualified the offset's
LEVEL as an optimization target, not M4-E2's SHARES.** Ratios cancel the units
that levels inherit. The debt named here is paid; M4-E2's picture stands as
structural, and the M4-H basis line proceeds from its own numbers:

- S4 (residual) is the largest single piece in 3/3 worlds (.4015–.4464,
  registered order);
- S3 (normalization/scale modes) carries .2953–.3830 in registered order and
  .4459–.5288 standalone/reversed — M4-E2's disclosed knife-edge reproduced
  exactly under both metrics;
- within S3 the n2 scale family dominates (.7398–.7901 of standalone S3,
  .8132–.8714 of registered-order S3);
- the offset direction remains world-specific (cosines at or below the
  permutation null's q95 in 3/3 pairs).
