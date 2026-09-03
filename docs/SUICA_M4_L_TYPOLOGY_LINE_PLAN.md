# GRAIN M4-L — The Typology Line (IDT-R: the reverse reading)

Line opened 2026-08-09 (evening) on the program owner's directive: study
the REVERSE direction — does removing identity make correct personality
GROUPING possible, and is the old "error = 0 ⇒ no identity" hypothesis
provable? Theory: `docs/SUICA_IDENTITY_THEORY_V1.md` appendix P (theorems
R1–R3, the geometry dichotomy, the completeness audit). Tier:
EXPLORATORY, label-free, synthetic, card-space. The claims ledger
controls. The defense phase charter stays shelved per the owner's
direction; this line takes precedence.

All standing rules 1–17 and conventions bind (round-trip parsing; 4-world
pilots with df-awareness; Part-0 verification of bit-identity claims;
chunked stages < 600 s; rule-16 full-object enumeration; rule-17
realizability with fallback ladders).

---

## M4-L1 — Instrument leg: a typed world, the identity tax on grouping, the dichotomy, and the completeness audit calibrated

**REGISTERED 2026-08-09, BEFORE RUN.** Planner: this document's author.
Executor: dispatched agent. This leg builds the TYPE layer, validates the
R-theorems' measurable shadows against Part-0 predictions, and calibrates
the completeness audit. It adjudicates instrument validity, not the
open taxometric question (that is L2+).

### Machinery

`scripts/run_suica_m4_l1_typed_world.py`, extending K2a's expressive
world IN THE LEG SCRIPT (suica_core untouched; rule-12 source naming).
Type layer added to the trait channel:

    mean_part_i = τ_{g(i)} + b_i

- g(i): G = 4 groups, equal sizes, assigned by seeded rng;
- τ_g: centroids in a k_τ = 3-dimensional random subspace S of the
  k = 48 latent space, pairwise separation set by knob Δ (realized
  separations reported);
- b_i (within-type identity), TWO GEOMETRIES per arm:
  **ISO** — isotropic in all 48 dims; **ALIGNED** — supported in S;
  identity share ρ_id := σ_b²/(σ_b² + σ_τ²) is the swept knob;
- occasion/state/noise channels as in K2a (φ_slow .90, n_occ 8,
  w_int 0), so cards and ρ_i come from the validated machinery.

Grouping instruments (pinned per rule 9/12, both computed everywhere):
PRIMARY = Lloyd's k-means on cards with known G, 64 seeded restarts,
best-objective; SECOND READING = spectral (top-k_τ PCA projection of
pooled cards, then k-means — the R2 "projection" operation). Agreement
metric: ARI against generator truth g (Hungarian-matched accuracy as
second reading).

### Arms (master_seed 20260822; 8 worlds × 512 authors per cell)

Grid: geometry ∈ {ISO, ALIGNED} × ρ_id ∈ {0, 0.15, 0.35, 0.55, 0.75}
at fixed Δ (chosen in Part 0 so the ISO ρ_id=0.35 cell sits mid-range,
rule-17 realizability: no cell saturated at ARI 0 or 1 in the pilot;
pre-declared adjustment ladder on Δ, one step, disclosed). Plus, on the
ISO and ALIGNED ρ_id=0.55 cells: an **oracle-removal companion**
(cards recomputed with b_i subtracted using generator truth — the upper
bound of the owner's conjecture) and an **oracle-projection companion**
(projection onto true S). 10 main cells + 4 companions.

### Part 0 gates

- **G0L (construction + anchors)** — reconstruction residual ≤ 1e-12;
  realized ρ_id and Δ within tolerance per cell; K2a anchor cell
  re-derived bit-exactly; per-cell ARI-relevant dims reported (rule 5).
- **G1L (rule 10)** — the type channel changes panels; ISO vs ALIGNED
  differ; b=0 arms differ from b>0 arms.
- **G2L (rule 3 + rule 17)** — liveness and realizability: pilot ARI in
  (0.05, 0.95) for the mid cells (else the Δ ladder fires, once,
  disclosed); type-subspace estimability check (top-k_τ eigengap > 0
  reported).
- **G3L (power, 4-world pilot)** — MDEs for every CI clause; rule-11
  satisfiability with directions; rule-13 spec (B=2000, seed=master,
  ≥10×B at boundaries).
- **G4L (Part-0 predictions)** — computed BEFORE arms from appendix P's
  R2 geometry: per-cell predicted ARI-degradation ordering in ρ_id;
  the ALIGNED error floor Φ(−Δ/(2σ_{b,u})) per boundary at each ρ_id;
  the ISO projection-gain factor; the audit's expected surviving-identity
  share per cell (= designed b-share through the validated T6″ machinery).
- **G5L (hygiene + rule 16)** — enumeration over the four leans' 2^4
  outcomes with the routing below; round-trip; chunked; rule-14
  self-check (all card-space, ARI-vs-ARI and share-vs-share; no
  cross-scale gate).

### Leans (aggregation: per-cell CIs vs Part-0 predictions, paired
world-block bootstrap B=2000; leans separate)

- **V-1 (designed sanity) [prior .85]** — ρ_id = 0 cells: PRIMARY
  grouping recovers g with ARI ≥ 0.95 in 8/8 worlds, both geometries
  (the world is groupable when the owner's 誤差=0 premise holds by
  construction).
- **V-2 (identity tax, R4) [prior .80]** — ISO cells: ARI strictly
  decreasing in ρ_id (ordering exact) AND per-cell ARI within CI of the
  Part-0 R2 prediction in ≥3/4 non-zero cells.
- **V-3 (the dichotomy, R2) [prior .75]** — at ρ_id ∈ {0.35, 0.55}:
  ALIGNED ARI < ISO ARI with pooled CI excluding 0; the ALIGNED cells'
  error consistent with the floor prediction (CI containment ≥1/2
  cells); oracle PROJECTION restores ISO ρ_id=0.55 by ≥ half its deficit
  to the ρ_id=0 level while restoring ALIGNED by ≤ a quarter of its
  deficit (the conjecture's validity boundary, measured); oracle REMOVAL
  restores BOTH to ARI ≥ 0.95 (upper bound sanity).
- **V-4 (completeness audit, R3) [prior .80]** — the T6″-based
  surviving-identity meter, applied to within-TRUE-group deviations:
  reads 0 within margin on ρ_id = 0 cells (designed null, margin from
  pilot) and tracks the designed b-share within CI in ≥6/8 non-zero
  cells (calibration of the audit as a measuring device).

### Routing (rule 16; precedence top-down)

1. V-1 MISS → **P1L: instrument defect or unrealizable grid** — STOP,
   report, no theory shadow claimed.
2. ≥2 of {V-2, V-3, V-4} MISS → **P2L: appendix P's measurable shadows
   fail** — the R-theorems keep their proofs but lose their claimed
   instrument shadows; theory-repair note before any L2.
3. Exactly 1 MISS → **P3L: QUALIFIED** — the missing piece named,
   follow-up charter; the other pieces stand as measured.
4. All HOLD → **P4L: instrument certified; R2/R3/R4's shadows MEASURED**
   — L2 (realizable removal ladder: split-half averaging / frame
   refreshment / label-free projection vs the oracle bounds; and the
   taxometric decision instrument on unknown-geometry worlds) becomes
   the next registration.

### Deliverables

The six: script; `results/m4_l1_typed_world/` (manifest.json,
gates.json, per-cell CSVs, decision.json — gitignored);
`reports/SUICA_M4_L1_TYPED_WORLD_REPORT.md` (Part 0 with predictions
first, timestamped before arms); outcome appended here; ledger row; ONE
commit (`feat(m4-l): L1 — ...`), never amended, not pushed by the agent.
Budget: card-space; target < 25 min wall; stop-and-report at 2× any
Part-0 stage estimate.

---

## L2 charter + planner derivation (dated append, 2026-08-10 — BEFORE L1
adjudication; the L2 registration follows L1 per L1's routing)

L1's arms are the pure poles (ISO / ALIGNED). Real worlds live on the
continuum between them. This iteration's planner question, solved by
derivation while L1 runs: **what are the laws on the η-continuum, and
what can be done LABEL-FREE?** Let the identity decompose as
b = √(1−η)·b_iso + √η·b_aligned (η = the aligned fraction of identity
energy; b_aligned isotropic WITHIN S).

**Derivation 1 (the interpolated projection gain).** Distance-based
grouping suffers the identity energy retained in the working space.
Raw space: E_raw = σ_b². After projection onto S (dim k_τ of m):
E_proj = η·σ_b² + (1−η)·σ_b²·(k_τ/m). Hence the projection gain

    G(η) = E_raw / E_proj = 1 / (η + (1−η)·k_τ/m)

— at η = 0 it is m/k_τ (L1's ISO prediction, = 16 at 48/3); at η = 1 it
is exactly 1 (no gain, the ALIGNED pole). One-parameter law, registered
as L2's quantitative prediction curve.

**Derivation 2 (the floor generalizes).** Along a boundary normal
u ∈ S, post-projection identity sd is
σ_u²(η) = η·σ_b²/k_τ + (1−η)·σ_b²/m, so the irreducible error floor per
boundary is ≈ Φ(−Δ/(2σ_u(η))) — continuous in η, reducing to L1's
ALIGNED floor at η = 1 and to a negligible floor at η = 0. Consequence
worth stating plainly: **the owner's conjecture degrades GRACEFULLY —
identity removal by projection buys G(η), never less than 1, and the
un-buyable remainder is exactly the aligned fraction.**

**Derivation 3 (label-free estimability of η — the spectral taxometer).**
The pooled card covariance separates the poles by WHERE identity energy
sits: ISO identity raises the BULK (all m dims, +(1−η)σ_b²/m per dim);
ALIGNED identity feeds the same k_τ SPIKES as the types. Since the
per-card noise variance is independently estimable from split-half
disagreement (the K2a-validated machinery), the bulk excess over noise
identifies (1−η)σ_b², label-free. The aligned share then needs one more
number (total σ_b²), obtainable within-group AFTER provisional grouping
— which motivates the CROSS-FITTED design below. Detectability caveat,
stated as the registered form: the type spikes are label-free
recoverable only above the spiked-covariance (BBP-type) threshold —
type-axis variance must exceed ~√(m/N) × bulk variance; L2's Part 0
computes the exact constant for its dims and MUST verify every arm sits
on the intended side (rule 3/17).

**Derivation 4 (the audit needs cross-fitting).** Auditing completeness
on ESTIMATED groups is biased optimistic: assignment to nearest centroid
truncates within-group variance at the boundaries (overfitting absorbs
identity into the partition), so the same-data audit UNDERSTATES
surviving identity. The honest instrument is cross-fitted: cluster on
one half of occasions/persons, audit on the held-out half. L2 validates
the cross-fitted audit against the oracle-group audit (L1's V-4
calibration) and measures the same-data bias as a named quantity.

**L2's charter (registration to be written after L1 adjudicates, per
routing):** arms on the η-continuum (η ∈ {0, .25, .5, .75, 1} at 2–3
identity shares), label-free instruments only — estimated-S projection
(spiked-PCA), the spectral taxometer η̂, the cross-fitted completeness
audit — against L1's oracle companions as upper bounds. Leans: (i) G(η)
curve containment; (ii) floor curve containment; (iii) η̂ operating
characteristic (monotone, calibrated at the poles L1 certified); (iv)
cross-fitted audit tracks oracle audit while same-data audit shows the
predicted optimistic bias. Rule-16 enumeration and the full gate set as
always; margins from L1's realized variances.

### OUTCOME (appended 2026-08-09, after run; append-only)

**`SHADOWS_FAIL__V1HV2HV3MV4M__P2L` — V-1 HOLD · V-2 HOLD · V-3 MISS · V-4 MISS
→ P2L.** Report: `reports/SUICA_M4_L1_TYPED_WORLD_REPORT.md`. Script:
`scripts/run_suica_m4_l1_typed_world.py`. Δ = 5.928950380606693 (RN-10
bisection to predicted ρ_id=0 oracle ARI 0.99; the ladder did NOT fire —
no cell saturated). All six Part-0 gates passed; K2a anchor cell
`phi0.9_occ8_intzero` re-derived BIT-EXACTLY (residual 0.0, 2048 rows ×
35 columns); reconstruction residual 2.22e-16; ISO and ALIGNED at
ρ_id=0 bit-identical (0.0); oracle removal reproduces the ρ_id=0 card to
3.12e-16. 14 cells × 8 worlds × 512 authors; 51.62 s adjudicated compute.

**The measured ARI ordering reproduces the Part-0 prediction cell for
cell**, both instruments. The identity tax (R4) is real and geometry-
dependent: ISO loses 0.0994690680231981 of ARI across ρ_id 0→.75,
ALIGNED loses 0.8102662 — **8.1×** at matched total identity energy. The
dichotomy (R2) is measured at full strength: pooled ALIGNED−ISO at
ρ_id ∈ {.35,.55} = **−0.4128850283258326** [−0.4305262035508016,
−0.3953012811449916]. The ALIGNED floor Φ(−Δ/(2σ_{b,u})) predicts the
noise-free per-boundary rate to 2.3% (ρ=.35: 0.027750651041666664 vs
0.027118834519294123, contained) and 6.7% (ρ=.55: 0.107421875 vs
0.10068281660598428, not contained) — 1/2 cells, clause HOLDs.

**The two misses.**

- **V-3 fails on the oracle-PROJECTION clause alone** — ISO restoration
  fraction **0.13468401332369143** [−0.0258742, 0.2774669] against the
  registered ½ bar (ALIGNED 0.003369541289360893 ≤ ¼ ✓; removal ✓ but a
  designed identity). This miss was **PRE-REGISTERED in Part 0 §0.6 with
  its derivation and a predicted value of 0.15265055004035916**. A
  post-hoc unregistered diagnostic at ρ_id=.75 — where R2's ambient
  energy condition Δ² ≳ σ_b² is violated (σ_b²/Δ² = 1.125) — gives a
  restoration fraction of **0.1495074669143142**, i.e. FLAT, not rising.
  **Reading:** R2's m/k_τ factor is an ENERGY statement about the
  k-means objective; ARI at these separations is owned by the
  boundary-normal Bayes error, which projection provably cannot change
  because the boundary normal lies inside S. The ISO half of R2 needs
  its estimand re-stated as a consistency THRESHOLD, not achieved ARI.
- **V-4 fails on tracking, 5/8 < 6** — a calibration OFFSET, not a
  structural failure. The meter's numerator error is a near-constant
  +0.0011…+0.0023, traced entirely to the realized two-split state
  coefficient **B̂ = 0.245861** (design 0.25, −1.66%), which propagates
  as +0.002754 into Â. The designed null at ρ_id=0 HOLDS as an
  equivalence (Ŝ_id = 0.010918432922059714 [0.0006823, 0.0198269] inside
  the pilot margin 0.029389351349840543) though its CI excludes 0. The
  declared second reading (contiguous-only, design B) also tracks 5/8
  with the offset reversed in sign, so the MISS is robust to the reading.
  Rule 13 triggered on both razor-thin calls; both STABLE at B = 20 000.

**Routing robustness:** P2L under the registered readings and under both
declared second readings (V-2 on the instrument prediction → 3 misses →
still P2L; V-4 on the contiguous meter → still 5/8).

**Registration defects recorded (#27–#30):** #27 G2L's (0.05,0.95) band
clause and V-1's 0.95 bar are **jointly unsatisfiable on the ISO arm**,
PROVED — the identity-only boundary z is Δ-free (ISO 5.65685/√(ρ/(1−ρ)),
ALIGNED 1.41421/√(ρ/(1−ρ))), so the ISO arm's identity-only floor never
exceeds 5.45e-4 on the registered grid; handled by RN-10's pre-declared
guard, band clause 2/4 (both ALIGNED), both ISO failures **from above**
and unsaturated. #28 V-3(d) is a designed identity, not a test. #29 the
floor clause had no registered comparand (RN-7 pinned the noise-free
TRUE-card rate before arms). #30 V-3(c)'s bar tests a proposition R2 does
not make.

**Consequence for L2 (theory-repair note owed first, per P2L).** Decouple
Δ from σ_b so the consistency threshold can be crossed; re-state R2's ISO
half as a threshold claim and test it there; give the completeness audit
its one calibration constant (per-world B̂, or bias-corrected target with
the B̂ error in the CI) — either moves tracking from 5/8 to 8/8 on these
numbers. The taxometric decision instrument L2 was to open is NOT damaged:
its discriminating signal (the ALIGNED floor and the 8.1× tax) is exactly
the part that measured cleanly.

### Planner adjudication (2026-08-10, appended after the run)

Scored as executed: **V-1 HOLD, V-2 HOLD, V-3 MISS (on sub-clause (c)
alone), V-4 MISS (5/8, one diagnosed constant), → P2L.** The
theory-repair note P2L demands is IDT appendix Q, written today. What
the leg actually established is stronger than the verdict string:

- The 10-cell predicted ordering reproduced **cell for cell**, both
  instruments. The geometry's ordinal content is exact.
- The floor machinery is REAL and quantitative (ρ.35: measured
  0.027750651041666664 vs predicted 0.027118834519294123, contained;
  ρ.55 missed by ~7% relative — narrow).
- **The tax ratio at matched identity energy is 8.145911689523754×**
  (ALIGNED vs ISO ARI drop across the sweep) — the dichotomy's signal,
  and its z-ratio is exactly √(m/k_τ) = 4 by the executor's Δ-free
  boundary algebra. The m/k_τ factor is real; it lives in the FLOOR
  RATIO between geometries, not where my registration put it.
- **V-3(c) failed because the registered bar tested a proposition R2
  does not make** (defect #30): oracle projection restored only 13.5%
  [−2.6%, 27.7%] of the ISO deficit at ρ.55 and stayed flat (15.0%) at
  ρ.75 — because ARI above the localization threshold is owned by the
  boundary-normal Bayes error, and the boundary normal lies INSIDE S:
  projection onto S provably cannot change identity variance along it.
  R2's m/k_τ energy gain governs centroid LOCALIZATION, not
  above-threshold assignment error. Appendix Q restates it.
- **V-4's miss is one calibration constant**: B̂ = 0.245861 identical to
  six decimals in every ambient cell vs true 0.25; (0.25 − B̂)·c_cont =
  +0.002754 reproduces the offset; ALIGNED cells fail tracking because
  their CIs are 1.5–2.4× TIGHTER, not because their bias is larger. The
  meter works; it needs the constant. L2 implements the calibrated form.

**Planner registration defects recorded (#27–#30):** #27 (PROVED by the
executor) — G2L's mid-band clause and V-1's bar are JOINTLY unsatisfiable
on the ISO arm because ρ_id ties σ_b to Δ; individually each clause was
satisfiable. **Standing rule 18 (added 2026-08-10, paid for by #27):
rule-11 satisfiability is checked JOINTLY across all clauses sharing
generative knobs, not per-clause.** #28 — V-3(d) was a designed identity
(removal ≡ ρ_id=0 card at 3.1e-16), a construction certificate scored as
a lean (rule-3 family, vacuous-pass). #29 — a clause with no registered
comparand (executor pinned it pre-arms under rule 9). #30 — a lean bar on
a different quantity than the theorem's own (energy threshold vs achieved
ARI). **Standing rule 19 (added 2026-08-10, paid for by #30): every lean
bar is derived from the theorem's OWN quantity and scale, with the
registration stating which theorem-quantity the bar shadows; a bar on a
different quantity is a registration defect regardless of outcome.**

Consequences executed today: IDT appendix Q (the floor-invariance lemma,
the relocated dichotomy, the sharpened form of the owner's conjecture:
the removal-vs-projection gap IS the boundary-normal identity variance,
unremovable without labels); L2 registered below with rule-18/19
compliance and the calibrated audit.

---

## M4-L2 — The threshold, the continuum, and the label-free instruments (registered on the repaired theory)

**REGISTERED 2026-08-10, BEFORE RUN.** Planner: this document's author.
Executor: dispatched agent. Built on appendix Q's restated R2 and the
L2-charter derivations above; every lean states which theorem-quantity
its bar shadows (rule 19), and satisfiability is checked jointly across
knob-sharing clauses (rule 18).

### Questions

(A) **Threshold (R2 restated):** does the m/k_τ projection gain exist
where R2 actually puts it — the centroid-LOCALIZATION threshold?
Requires decoupling Δ from σ_b (L1's ρ_id parameterization tied them —
defect #27's root). (B) **Continuum:** does the projection-invariant
floor follow σ_u²(η) = η·σ_b²/k_τ + (1−η)·σ_b²/m across the η grid?
(C) **Label-free instruments:** estimated-S projection vs oracle-S in
the threshold regime; the spectral taxometer η̂; the CALIBRATED
cross-fitted completeness audit (per-world B̂), including the same-data
optimistic bias as a named quantity.

### Machinery and arms (master_seed 20260823; 8 worlds × 512 authors/cell)

L1's typed world with (Δ, σ_b) as INDEPENDENT knobs (rule-18 lesson) and
the η-mixture identity b = √(1−η)·b_iso + √η·b_aligned.

- **T-arms (threshold, question A):** ISO geometry, σ_b fixed at L1's
  ρ.55-equivalent energy; Δ swept over a 5-point ladder computed in
  Part 0 to bracket the AMBIENT localization break (pilot: ambient ARI
  crosses 0.5) while keeping the ORACLE-S floor negligible (< 0.005,
  Part-0 verified — so any ambient failure is localization, not Bayes).
  Instruments per cell: ambient Lloyd, oracle-S projection + Lloyd,
  estimated-S (top-k_τ spiked PCA) + Lloyd.
- **C-arms (continuum, question B):** η ∈ {0, 0.25, 0.5, 0.75, 1} at
  L1's Δ and two identity energies (L1's ρ.35/ρ.55 equivalents);
  measured per-boundary error rates vs the Part-0 floor curve.
- **Audit arms:** the C-arms double for the calibrated audit — per-world
  B̂ calibration; PRIMARY = cross-fitted (cluster on one occasion-half,
  audit on the other), SECOND = same-data (bias measurement).

### Part 0 gates

- **G0M** — construction residuals ≤ 1e-12; L1 anchors bit-exact
  (ISO/ALIGNED ρ.55 ARIs, floors, tax ratio 8.145911689523754, audit
  offset +0.002754); realized (Δ, σ_b, η) tolerances.
- **G1M (rules 10+3)** — non-degeneracy and liveness per arm; the
  T-arm ladder brackets the break in the pilot (ambient ARI spans
  (<0.3, >0.8) across the ladder) — else the pre-declared ladder shift
  fires once, disclosed (rule 17).
- **G2M (rule 18)** — JOINT satisfiability over all clauses sharing
  (Δ, σ_b, η), enumerated in the report; directions stated; 4-world
  pilot MDEs; rule-13 spec (B=2000, seed=master, ≥10×B at boundaries).
- **G3M (Part-0 predictions, rule 19 fidelity table)** — one row per
  lean: the theorem quantity, its predicted value/curve, the bar, and
  the derivation. Threshold window edges from the energy criterion; the
  floor curve σ_u²(η); the BBP-form detectability constant for
  estimated-S at these dims; the audit targets = designed shares with
  the B̂-corrected estimator.
- **G4M (hygiene + rule 16)** — full-object enumeration (cells → leans
  → routing); round-trip; chunked; rule-12 header; rule-14 self-check.

### Leans (paired world-block bootstrap B=2000; leans separate)

- **W-1 (localization gain exists) [prior .65; shadows: R2-restated's
  threshold claim]** — in ≥2 T-cells inside the bracketed window:
  ambient ARI < 0.3 while oracle-S ARI > 0.8 (per-cell CIs on the
  stated sides); and NO T-cell shows the reverse ordering.
- **W-2 (label-free reaches oracle) [prior .60; shadows: spiked-PCA
  consistency above the BBP-form threshold]** — estimated-S ARI within
  CI of oracle-S ARI in every T-cell on the detectable side of the
  Part-0 constant; the non-detectable side (if the ladder enters it)
  reported, not gated.
- **W-3 (the floor curve) [prior .75; shadows: the projection-invariant
  floor σ_u²(η)]** — C-arms: measured per-boundary error within CI of
  the floor curve in ≥7/10 cells, AND the η-ordering exact at both
  energies.
- **W-4 (calibrated audit) [prior .75; shadows: R3's completeness meter
  after the B̂ fix]** — cross-fitted calibrated audit tracks designed
  shares in ≥8/10 C-cells (CI containment); same-data audit shows the
  predicted-sign optimistic bias (pooled CI excluding 0 on the
  optimistic side) — the bias is a MEASURED named quantity, not a gate
  on its size.

### Routing (rule 16; precedence top-down)

1. G1M bracket fails after its one ladder shift → **P1M: STOP**
   (unrealizable threshold window at these dims; report).
2. W-1 MISS → **P2M:** R2-restated fails at its OWN quantity —
   the localization claim dies too; appendix Q gains a second dated
   correction; the ISO half of the dichotomy is then ONLY the floor
   ratio (still real: 8.15× tax). No further projection claims.
3. W-1 HOLD, ≥2 of {W-2, W-3, W-4} MISS → **P3M:** threshold confirmed
   but instruments unreliable — instrument-repair leg before any
   taxometric use.
4. W-1 HOLD, ≤1 of the rest MISS → **P4M: QUALIFIED-or-CLEAN** — the
   restated R2 is MEASURED; the taxometer η̂ and calibrated audit are
   certified to the extent held; L3 (the taxometric decision on
   unknown-geometry worlds, and the realizable removal ladder against
   the floor bound) becomes the next registration. Any single miss is
   named and chartered.

### Deliverables

The six: `scripts/run_suica_m4_l2_threshold_continuum.py`;
`results/m4_l2_threshold_continuum/`;
`reports/SUICA_M4_L2_THRESHOLD_CONTINUUM_REPORT.md` (Part 0 with the
rule-19 fidelity table first); outcome appended here; ledger row; ONE
commit (`feat(m4-l): L2 — ...`), never amended, not pushed by the agent.
Budget: card-space; target < 25 min wall; stop-and-report at 2× any
Part-0 stage estimate.

### OUTCOME (appended 2026-08-09, after run; append-only)

**`LOCALIZATION_FAILS__W1MW2MW3HW4M__P2M` — W-1 MISS · W-2 MISS · W-3 HOLD ·
W-4 MISS → P2M.** Report: `reports/SUICA_M4_L2_THRESHOLD_CONTINUUM_REPORT.md`.
Script: `scripts/run_suica_m4_l2_threshold_continuum.py`. All five Part-0 gates
passed; every L1 anchor re-derived BIT-EXACTLY by calling L1's own path (ISO/
ALIGNED ρ.55 ARIs 0.9664213102607500 / 0.4205650587388517, both floors, tax
ratio 8.145911689523754 from drops 0.0994690680231981 / 0.8102662439562028; the
audit offset reproduced as +0.0027534823998187526 from c_cont = 0.66525305625).
Reconstruction residual 2.220446049250313e-16; the η-mixture's designed energy
exact to 2.9% (η=1 has only k_τ=3 effective dims at 512 authors). The Δ ladder
solved on a 34-point prediction-stream grid to ambient-ARI targets
(.10,.25,.50,.75,.90) → Δ = 2.210909967446789 / 2.6016054931828423 /
3.1758671500715328 / 4.066041381764518 / 5.036879557726254; **the bracket HOLDS**
(pilot ambient 0.0974 → 0.9207) so the one pre-declared ladder shift did NOT
fire and P1M was not reached. 15 cells × 8 worlds × 512 authors;
139.226 s adjudicated compute (part0 93.648 written to disk BEFORE arms and
enforced in code, arms 45.443, finalize 0.134) against a <25 min budget.

**Standing rule 18's first application paid for itself immediately.** G2M
reduced all nine knob-sharing T-arm clauses to sets of Δ and intersected all 36
pairs BEFORE any arm ran. Result, PROVED on the grid: `{ambient ARI < 0.30}` =
Δ ∈ [0.7500, 2.6978] and `{oracle-S ARI > 0.80}` = Δ ∈ [4.2386, 9.0000] are
**disjoint by a factor 1.5711320335087848** — W-1's window is empty at every one
of the 34 grid points, so no ladder could have contained a single T-cell, let
alone the two the bar demands (**defect #31**). The mechanism is appendix Q.1
itself: the `{oracle-S > 0.80}` set is IDENTICAL to the `{Bayes ceiling > 0.80}`
set, because oracle-S sits ON the projection-invariant ceiling, and that ceiling
is set by K2a's slow-state channel, which contributes 2.3× the identity's own
boundary-normal variance at the ρ.55-equivalent energy and is free of both Δ and
σ_b. **Defect #32**: the registered design clause "oracle-S floor < 0.005
everywhere" is likewise jointly unsatisfiable with the bracket's low end under
both readings (identity-only needs Δ ≥ 3.1362; full Bayes needs Δ ≥ 5.7283,
above even the ladder's top rung).

**W-1 MISSES 0/5 as registered — and the localization gain is REAL and LARGE.**
The theorem-quantity second reading was pinned in Part 0 §0.5 before any
main-world number (as L1 §0.6 pinned V-3(c)'s miss), and it holds in 4/5 cells:
localization gap (oracle-S − ambient) = **+0.2571091862002618**
[+0.24085764314672325, +0.2769365050856173] at T0 (ambient 0.0782933783 →
oracle-S 0.3354025645, a **4.28×** ratio), +0.22879210873022762,
+0.12074128749937892, +0.030243778023137782, and +0.007370778741078776 (CI
covering 0) at T4. In T0/T1/T2 oracle-S sits ON the Bayes ceiling (CI contains
0) while ambient is decisively below it — exactly "ambient cannot localize, the
projected instrument can, up to Bayes". **R2-restated did not die at its own
quantity; the registered bar never tested it there.**

**W-2 MISSES 0/5, and defect #33 explains it (rule-19 class, same family as
#30).** All five rungs are on the detectable side (θ_min/√γ =
1.7734492790446281 … 9.173517917654014, √γ = 0.3535533905932738), but the BBP
law's own quantity is the SUBSPACE OVERLAP, and the same law predicts overlaps
of only 0.618/0.759/0.860/0.926/0.956 there — it forecasts the bar's failure at
the low rungs. Measured overlap 0.4326010144 / 0.6271676199 / 0.8004244064 /
0.9091396379 / 0.9510052040, relative shortfall 30.1%/17.4%/7.0%/1.8%/0.5%:
right shape, right magnitude, converging as θ grows.

**W-3 HOLDS — the leg's clean positive.** The η-continuum floor law
σ_u²(η) = η σ_b²/k_τ + (1−η) σ_b²/m is measured: containment 7/10 (exactly the
bar) and η-ordering EXACT at both energies. Two of the three failures are a
design-resolution artifact — the per-boundary rate is a Bernoulli mean over 1536
tests (resolution 6.510416666666667e-4) against η=0 predictions of 4.05e-14 and
3.53e-7 — so over the eight resolvable cells containment is **7/8**; the one
genuine miss is C_rho55eq_eta0.25 (measured 0.007893880208333334
[0.006429036458333334, 0.009358723958333332] vs 0.00982199837718639, 19.6%
low). The η=1 poles reproduce L1's independent ALIGNED measurements to 5.6% and
8.4% on fresh worlds under a fresh master seed (**defect #35** recorded for the
resolution clause).

**W-4: the bias half HOLDS decisively, the tracking half MISSES 2/10 — and
appendix Q.5's "one calibration constant" is REFUTED.** Derivation 4 is
measured: `same-data − cross-fitted` is negative in **10/10** cells, pooled
**−0.005767022729929317** [−0.006438551315473723, −0.005165302746842192], 9.1×
its own half-width and 5.4× the pilot MDE, growing monotonically with η at both
energies. But tracking fails, and the three diagnoses do not point at the
constant: (i) the calibration WORKS — B̂_cal = 0.2519…0.2521 against
w_slow² = 0.25, while the L1-style theoretical-constant reading reproduces L1's
own pathology on fresh worlds (B̂_theory = 0.2477…0.2481) — yet the calibrated
meter tracks **2/10** while the UNCALIBRATED one tracks **5/10**, because the
theory constant's downward bias partially cancels the meter's own low bias;
(ii) cross-fitting halves the occasion budget and collapses the two-split
contrast c_int − c_cont from L1's 0.0969198750000001 to 0.04499999999999993 (a
2.1537750000000058× conditioning loss), which the registration did not budget;
(iii) with TRUE groups on the identical audit half and the identical calibrated
meter, tracking is **10/10** — the meter is sound; the residual is
partition-borne, grows monotonically from −0.011 (η=0) to −0.030 (η=1) in share
units against a ~0.015 CI half-width, and is caused by labels fitted on
occasions {0,1,2,3} carrying slow-state structure (ar_cross_cov 0.66525305625 at
φ=.90) into the audit half. **Defect #34** recorded: the registration had no
clause able to distinguish "the constant is wrong" from "the partition is
wrong".

Rule 13: 1 clause triggered (C_rho35eq_eta0.25's tracking, distance
0.0007948328539089432 vs MC endpoint sd 0.0004827646302271462), 0 BOUNDARY,
STABLE at B = 20000. Routing is invariant under every declared second reading:
W-1's second reading is routing-inert by construction; W-2 under the overlap
companion is still 0/5; W-3 under the resolution-restricted count is 7/8, still
HOLD; W-4 under the theoretical-constant meter is 5/10, still < 8 → MISS. P2M
under all of them.

**Registration defects recorded (#31–#35):** #31 W-1's two conditions jointly
unsatisfiable, PROVED on the Δ axis before the arms (gap factor 1.571); #32 the
"floor < 0.005 everywhere" design clause jointly unsatisfiable with the
bracket's low end; #33 W-2's bar on an ARI where the BBP law owns an overlap
(rule-19 class); #34 W-4's bar unable to separate the constant from the
partition; #35 W-3's containment clause unresolvable at η=0 (1536 Bernoulli
tests vs a 3.5e-7 prediction).

**The brief to the planner, in one line each.** (1) P2M is executed as
registered, but its stated reading ("the localization claim dies too") is NOT
what was measured — the mechanism was measured at 4.28× in three independent
cells. (2) What died is the gain's USABILITY at these dimensions, and that
deserves appendix Q's second correction as a **localization window lemma**: the
usable window [Δ*_ambient, Δ*_Bayes] has width governed by (d/n)^(1/4) against
the required Bayes z, and at d=64, n=512, G=4, k_τ=3 with K2a's state channel it
is EMPTY for any bar demanding a high absolute ARI on the projected side;
widening it needs more authors, more dimensions, or a quieter state channel —
all L3 knobs. (3) W-3 makes derivation 2 a measured object. (4) Q.5 needs its
own correction: R3's meter is correct, and what is uncalibrated is its behaviour
on an ESTIMATED partition — a different object. (5) L3 budget note: if the audit
must be cross-fitted AND well-conditioned, n_occ must rise (16 restores L1's
contrast on each half) or the split must be over PERSONS rather than occasions.

### Planner adjudication (2026-08-10, appended after the run)

Scored as executed: **W-1 MISS, W-2 MISS, W-3 HOLD, W-4 MISS → P2M.**
Adjudicated with the grade the evidence licenses:

- **W-3 is the leg's positive yield: the η-continuum floor curve is
  MEASURED** (7/10 containment at the bar; exact η-ordering at both
  energies; the two resolution-void cells are a quantum fact, defect
  #35). Derivation 2 of the L2 charter is now empirical.
- **W-1 missed on a window my own conditions made EMPTY** — defect #31,
  PROVED before arms by the rule-18 joint check (disjoint by 1.571×):
  {oracle-S > 0.8} is IDENTICAL to {Bayes ceiling > 0.8} because
  oracle-S sits on Q.1's projection-invariant ceiling, which is set by
  the slow-state channel (2.3× the identity's boundary-normal variance)
  and is Δ-free. The MECHANISM was measured and is real: localization
  gap +0.2571091862002618 [+0.2409, +0.2769] at T0 (4.28×), declining
  monotonically to ~0 at T4, oracle-S ON the ceiling in T0–T2.
  **P2M's registered reading ("the localization claim dies") is
  corrected in appendix R to the localization-window lemma:** the gain
  is a continuous measured mechanism; the QUALITATIVE window
  [Δ*_ambient, Δ*_Bayes] has dimension-governed width (executor's
  (d/n)^¼ scaling recorded as a conjecture) and is empty at these dims.
  No further projection claims beyond the measured gap.
- **W-4's miss REFUTES appendix Q.5** — the honest entry: calibrating
  B̂ made tracking WORSE (2/10 vs L1's 5/10 uncalibrated); the same
  calibrated meter on TRUE groups tracks 10/10; cross-fitting cost the
  two-split contrast a 2.1538× conditioning loss. The audit's residual
  error is PARTITION-borne, not constant-borne. The completeness meter
  is henceforth certified ON TRUE/GIVEN PARTITIONS ONLY; its
  estimated-partition form needs the propagation instrument (L3).
  The same-data optimistic bias itself was confirmed clean (pooled
  −0.005767 [−0.006439, −0.005165], 10/10).
- **W-2**: defect #33 (rule-19 class, again mine) — the bar sat on ARI
  equality where the BBP law owns the OVERLAP; the overlap companion
  approaches prediction with Δ (shortfall 30.1% → 0.5%). The
  estimated-S-equals-oracle-S ARI claim is withdrawn at these dims.

**Planner registration defects recorded (#31–#35):** #31 W-1's joint
emptiness; #32 the floor-everywhere clause vs the bracket's low end
(same family); #33 W-2's wrong-quantity bar; #34 W-4's bar cannot
separate constant-wrong from partition-wrong (confound in the lean's
design); #35 W-3's η=0 cells unresolvable at the run's quantum.
**Standing rule 20 (added 2026-08-10, paid for by #31+#32): when the
rule-18 joint check finds ANY lean's condition-set empty, the leg STOPS
before arms as a registration defect (rule-10's analogue at the
adjudication layer), unless the registration pre-declared empty-set as
an adjudicable outcome.** (L2's registration did not; the executor
proceeded correctly under its text; the rule closes the gap.)

Theory consequences: IDT appendix R (the window lemma; Q.5's refutation;
the floor curve's promotion to MEASURED). L3 registered below.

---

## M4-L3 — The taxometer, and the partition-propagated completeness meter

**REGISTERED 2026-08-10, BEFORE RUN.** Planner: this document's author.
Executor: dispatched agent. Rule 20 is in force (first application):
any empty joint condition-set stops the leg pre-arms.

### Questions

(A) **The taxometer:** in an unknown-geometry world, can η be READ
label-free? η̂ built from the bulk-excess route (per-card noise variance
from the validated split-half machinery; bulk excess over noise
identifies (1−η)σ_b²) combined with the within-provisional-group
spectrum route; second reading: the alignment angle between
within-group top-space and between-group axes. (B) **The
partition-propagated meter:** model the estimated-partition audit error
as a function of partition quality and produce a corrected meter —
audit deviation regressed on (1−ARI) across cells with the oracle
anchor at ARI = 1 — then validate the corrected meter's tracking.

### Arms (master_seed 20260824; 8 worlds × 512 authors/cell; card-space)

The L2 C-grid reused at fresh seeds: η ∈ {0, .25, .5, .75, 1} × two
identity energies (ρ.35/ρ.55 equivalents), L1's Δ. Per cell: η̂ (both
routes), provisional grouping (both L1 instruments), the calibrated
meter on estimated AND true partitions, cross-fitted and same-data.

### Part 0 gates

- **G0N** — anchors bit-exact from L2 (floor-curve cells, audit tables,
  pooled bias −0.005767022729929317, conditioning ratio 2.1537750000000058,
  B̂_cal band); construction residuals; realized-knob tolerances.
- **G1N (rules 10+3+17)** — non-degeneracy; liveness; realizability of
  the η̂ routes (pilot: bulk-excess estimable with CI half-width < 0.3
  at every grid point — else the pre-declared fallback drops the
  affected route, disclosed).
- **G2N (rules 18+20)** — JOINT satisfiability over every lean's
  condition-set; ANY empty set → STOP pre-arms (rule 20).
- **G3N (rule 19 fidelity table)** — per lean: the theorem/derivation
  quantity, the predicted curve, the bar, the why. η̂'s quantity is η
  itself (bar = half the grid spacing, 0.125); the propagation model's
  quantity is the audit deviation vs (1−ARI) slope with the oracle
  anchor; pilot MDEs; rule-13 spec (B=2000, seed=master, ≥10×B at
  boundaries).
- **G4N (hygiene + rule 16)** — full-object enumeration; round-trip;
  chunked < 600 s; rule-12 header; rule-14 self-check (η̂-vs-η,
  share-vs-share, deviation-vs-ARI — no cross-scale gate).

### Leans

- **X-1 (taxometer orders) [prior .70; shadows: η itself]** — η̂
  (PRIMARY route) strictly monotone in designed η at both energies
  (Spearman = 1 over each 5-point grid) with pole calibration: η̂ CI
  contains 0 at η=0 and 1 at η=1 (all four pole cells).
- **X-2 (taxometer calibrated) [prior .55; shadows: η]** — |η̂ − η| ≤
  0.125 in ≥8/10 cells (grid half-spacing bar).
- **X-3 (propagation model) [prior .60; shadows: the deviation-(1−ARI)
  relation]** — pooled fit R² ≥ 0.7 with the oracle anchor inside the
  fit's CI at ARI=1, AND the corrected meter tracks designed shares in
  ≥8/10 cells on ESTIMATED partitions (L2's calibrated meter managed
  2/10).
- **X-4 (bias-sign consistency) [prior .80]** — the same-data optimistic
  bias persists under the corrected meter (pooled CI excluding 0,
  optimistic side).

### Routing (rule 16; precedence top-down)

1. Rule-20 stop (any empty set) → **P1N: registration defect, no arms.**
2. X-1 MISS → **P2N: the taxometer is dead at these dims** — the
  line's label-free geometry reading falls back to the floor/audit
  pair; L-line synthesis proceeds without a taxometer.
3. X-1 HOLD, X-3 MISS → **P3N: meter stays certified on true/given
  partitions only** — the estimated-partition audit is retired as an
  instrument (not patched again); synthesis proceeds with that scope.
4. X-1 HOLD, X-3 HOLD (X-2/X-4 any) → **P4N: the line's instrument set
  is complete** — the planner writes the M4-L synthesis and IDT
  appendix S (the answer to the owner's reverse question, final form),
  with any X-2/X-4 miss named and scoped.

### Deliverables

The six: `scripts/run_suica_m4_l3_taxometer_meter.py`;
`results/m4_l3_taxometer_meter/`;
`reports/SUICA_M4_L3_TAXOMETER_METER_REPORT.md` (Part 0 with the
fidelity table first); outcome appended here; ledger row; ONE commit
(`feat(m4-l): L3 — ...`), never amended, not pushed by the agent.
Budget: card-space; target < 25 min wall; stop-and-report at 2× any
Part-0 stage estimate.

### OUTCOME (appended 2026-08-09, after run; append-only)

**`TAXOMETER_DEAD__X1MX2HX3MX4H__P2N` — X-1 MISS · X-2 HOLD · X-3 MISS · X-4 HOLD
→ P2N.** Report: `reports/SUICA_M4_L3_TAXOMETER_METER_REPORT.md`. Script:
`scripts/run_suica_m4_l3_taxometer_meter.py`. 10 cells × 8 worlds × 512 authors;
230.416 s adjudicated compute (part0 207.345 written to disk BEFORE arms and
enforced in code by `require_part0()`, arms 22.385, finalize 0.686) against a
<25 min budget.

**Standing rule 20's first application: NO STOP.** The rule-18 joint check
reduced every lean's clauses to sets of design points over the (Δ, σ_b²) plane
(3 Δ-factors × 3 energy-factors, 8 prediction replicates each — the same n the
arms use) and intersected them within each lean BEFORE any arm ran. Condition-set
sizes: **X-1 6/9, X-2 9/9, X-3 2/9, X-4 9/9** — none empty, so the leg proceeded.
Two facts were put on the record pre-arms: **X-3's registered point (`d1_e1`) is
OUTSIDE its own condition-set** (clause X-3c reaches 8/10 only at Δ×1.4), a
pre-registered forecast that X-3 was the lean at risk; and X-1's η=0 pole clause
fails at every high-energy point. Both forecasts were borne out.

**X-1 MISSES on two of four pole cells, and P2M's pattern repeats a third time.**
η̂ (PRIMARY: bulk-excess with the panel-exact state-innovation whitener,
provisional grouping by L1's Lloyd instrument) is **strictly monotone with
Spearman = 1 at BOTH energies under every declared reading** (spectral grouping,
true partition, oracle whitener, alignment angle). Both η=1 poles calibrate
(CIs [0.896515, 1.117619] and [0.940217, 1.053524] contain 1). Both η=0 poles
fail: η̂ = **0.092384** [0.031550, 0.155382] and **0.049464** [0.008869,
0.091537], CIs excluding 0 from above by 0.0316 and 0.0089. Rule 13 did not
trigger on either. **Defect #36** recorded (rule-19 class, third of the #30/#33
family): X-1's pole clause demands a tolerance of 0 on the same quantity for
which X-2 declares 0.125 to be the grid's own resolution — a nil-significance
test standing in for an equivalence claim, at a design whose η=0 CI half-width
is 0.041–0.062 against an identified O(0.05) finite-sample pole bias.

**X-2 HOLDS 10/10 against a .55 prior — the leg's clean positive.** |η̂ − η| ≤
0.125 in every cell (max 0.092384, median 0.024), and 10/10 also under the
true-partition and oracle-whitener readings. **The label-free taxometer is
calibrated everywhere at the resolution the registered grid defines.**

**Derivation 3 needs an amendment, and the leg measures its price.** Taken
LITERALLY (flat card-space bulk) the bulk-excess estimator reads η̂ =
−0.576…−0.256 (ρ.35eq) and −0.077…0.383 (ρ.55eq): `M Mᵀ = A_SCALE² L
diag(G_PROFILE²) Lᵀ` spreads every isotropic latent channel over a 2.3884×
range, so the card bulk is not flat. Whitening by the state-channel shape is
what makes derivation 3 true in card space. The three shape estimates cost:
pure oracle → all four poles calibrate; panel-exact state innovations (the
matrix analogue of L2's accepted MN-6) → η=0 bias +0.049/+0.092, condition ratio
2.698127; strictly data-only (`B̂_mat`) → unusable, η̂ = 0.594/0.787 at η=0,
condition ratio 15.598213 because 1/(c_int − c_cont) = 10.32 amplifies a
d/n = 0.125 covariance error.

**ROUTING IS NOT INVARIANT under one declared second reading.** Under the
PURE-ORACLE whitener all four poles calibrate ([−0.028313, 0.112146],
[−0.014991, 0.077757], [0.890498, 1.125840], [0.937830, 1.056065]) → X-1 HOLD →
**P3N, not P2N**. The η=0 pole miss is therefore roughly half
whitener-estimation-borne and half intrinsic to `Â_mat`'s own sampling error.
Under every other declared reading (spectral grouping, true partition, alignment
angle, √ and misassignment-rate propagation forms) the route stays P2N.

**X-3: the propagation model is a good DESCRIPTION and an inadequate
CORRECTION.** Sub-clauses (a) and (b) HOLD: pooled fit R² =
**0.891596154639982** [0.826187803453253, 0.9273340435266129], slope
−0.04718966206694972 [−0.05492139609682376, −0.03718986553430942], intercept
+0.0012162320427350467 [−0.017136477786541676, 0.02306420791159516], and the
oracle anchor **+0.004428032310894993** [−0.014988701299138155,
0.028277087946167102] lies INSIDE the intercept CI. Sub-clause (c) MISSES 3/10 —
**and the mechanism is the opposite of L2's**: the UNCORRECTED cross-fitted
meter tracks 10/10 here, and the corrected one tracks 3/10, because the
correction removes the world-common `B̂_cal` offset and thereby shrinks the CI
by 6–39× (half-widths 0.020 → 0.0005–0.0028). At that precision the fit's own
residual (≤ 0.004800 in share units, 3.6% of the ρ.35eq target) is resolvable in
seven cells. **Defect #37**: clause (c) has no precision budget — it gets HARDER
the better the correction works, and conflates "the correction is right" with
"the residual is below the resolution the correction itself creates". The
predicted concavity is confirmed: √(1−ARI) gives R² = 0.9780973208413469 and
(1 − Hungarian accuracy) gives 0.8576211182723725.

**X-4 HOLDS decisively; derivation 4's sign is now measured a third time and
survives correction.** Pooled same-data optimistic bias UNDER THE CORRECTED
METER = **−0.007639458871208268** [−0.008786900570630973, −0.006390021005151330],
negative in 10/10 cells, 6.37× its own half-width and 1.70× the pilot MDE; the
RAW value is −0.007588169891501556 [−0.009114406065434490, −0.006043047568769170]
— the correction moves it by 0.00005, exactly as Part 0 predicted.

**All five Part-0 gates passed.** Every L2 anchor re-derived BIT-EXACTLY: pooled
same-data bias −0.005767022729929317 (residual exactly 0.0, recomputed from all
ten persisted L2 C-cell files); conditioning ratio 2.1537750000000058 re-derived
from k2a's own AR algebra (0.0969198750000001 / 0.04499999999999993); the
`B̂_cal` band 0.2519–0.2521 at the quoted 4-decimal precision (realized
0.2519088212714649 – 0.2521478508144498); and two full L2 C-cells
(`C_rho55eq_eta0`, `C_rho55eq_eta1`, eight columns × eight worlds each,
carrying the floor-curve and audit-table anchors) re-measured row-by-row through
L2's OWN path on L2's OWN worlds. Reconstruction residual
2.220446049250313e-16; realized latent Δ exact to 1.7763568394002505e-15. The
rule-17 fallback did NOT fire (max pilot η̂ CI half-width 0.1405 < 0.3). Rule 13:
1 clause triggered, 0 BOUNDARY, STABLE at B = 20 000.

**Free confirmation: R.1's η-floor law reproduces on fresh worlds under a fresh
master seed** — measured vs predicted per-boundary rate 0.004069/0.004188,
0.014567/0.013953, 0.027832/0.027122 (ρ.35eq) and 0.009684/0.009785,
0.041911/0.040059, 0.073405/0.072239, 0.101481/0.100688 (ρ.55eq), η-ordering
exact at both energies.

**Anomalies disclosed with timing.** (1) A 3.2 s pre-Part-0 feasibility probe on
the PREDICTION STREAM ONLY (no world of any index generated, no artifact
written, no hypothesis channel touched) established that the strictly data-only
whitener is unusable and fixed the PRIMARY route before any pilot or main world
existed. (2) **Defect #38 (executor-side)**: the registration says "regressed on
(1−ARI) across cells" without stating the observation grain; the first draft used
(cell, world), where per-panel `B̂_cal` noise attenuates R² from 0.892 to
0.040428064883860304 — corrected pre-arms to the registration's own cell grain,
both grains reported. (3) G0N's realized-energy tolerance was moved off L2's flat
3% realized-luck bar onto the design-derived 4σ band 4√(2[(1−η)²/m + η²/k_τ]/n)
(0.0361 at η=0 rising to 0.1443 at η=1, since the aligned component has only
k_τ=3 effective dimensions); realized max 0.05969940197215595, band violations 0.
(4) The FIRST Part-0 execution reported a rule-20 STOP on X-4; the cause was two
executor mis-implementations of the registered clauses — X-4's clause had used
the marginal sd of `sd_dev` for a PAIRED difference whose own sd is ~15× smaller,
and X-3's intercept clause a cell-level sd instead of the refitted intercept's
spread. Both were corrected before any arm ran (PN-9 now evaluates every clause
with the finalize stage's own paired structure) and the defective first verdict
is disclosed rather than silently overwritten.

**The brief to the planner, one line each.** (1) P2N is executed as registered,
but "the taxometer is dead at these dims" is NOT what was measured: ordering
exact under every reading, calibration inside the registered tolerance 10/10,
both η=1 poles contained — only the η=0 pole, where the estimator's bias exceeds
its own precision (defect #36). (2) Derivation 3 owes an amendment: the card
bulk is `G_PROFILE`-shaped, and the state-channel-shape whitening is not a
refinement but a precondition. (3) The propagation model earns R² 0.892 with its
ARI=1 value on R.3's certified anchor, and still fails as a corrector because it
sharpens the meter faster than it fixes it (defect #37). (4) Derivation 4's sign
is now three-times measured and correction-proof. (5) Scoping fact: `B̂_cal`
moved from 0.2519–0.2521 (L2's worlds) to 0.247053–0.247370 (L3's), flipping the
true-partition deviation from −0.0075 to +0.0044 — the meter's per-world offset
is the same size as the partition-borne error the line has chased since W-4, so
further meter work needs more world blocks, not more estimator repair.

### Planner adjudication (2026-08-10, appended after the run) — THE LINE CLOSES

Scored as executed: **X-1 MISS (pole clause only), X-2 HOLD 10/10, X-3
MISS (tracking clause only), X-4 HOLD → P2N on the primary reading**
(the oracle-whitener second reading gives P3N; primary controls, both
disclosed). Adjudicated with the grade the evidence licenses:

- **The taxometer is ALIVE as a measuring device.** |η̂ − η| ≤ 0.125 in
  10/10 cells (max 0.0924, median 0.0241); ordering Spearman = 1.0 at
  both energies under EVERY reading; both η=1 poles calibrate. What
  died is my pole clause — tolerance ZERO on a quantity X-2 declares
  resolvable to 0.125, a nil-significance test standing in for an
  equivalence claim (defect #36, third of the #30/#33 family). η̂ is
  certified as a calibrated geometry reader with a known +O(0.05) pole
  bias and the whitening-shape PRECONDITION (Derivation 3 amended: the
  bulk is G_PROFILE-shaped; flat-bulk reads −0.576 at η=0; the
  data-only whitener is unusable at condition ratio 15.6 — the
  panel-exact innovation form is the route).
- **The propagation law HOLDS** (deviation ~ (1−ARI): R² = 0.892
  [0.826, 0.927], slope −0.0472, oracle anchor inside) and the
  corrected meter is ACCURATE — residuals ≤ 0.0048 (3.6% of target) —
  but fails my tracking bar BECAUSE the correction sharpened CIs
  6–39×, exposing sub-budget residuals (defect #37: no precision
  budget). **Standing rule 21 (added 2026-08-10, paid for by #37):
  CI-containment bars on instrument validations carry a registered
  absolute-error budget; an instrument may not fail validation because
  its precision exposes a residual smaller than the budget.**
- **The meter chase ends by measurement, not fatigue:** B̂_cal moved
  0.252 → 0.247 between L2's and L3's world batches, flipping the
  true-partition deviation sign — the per-world constant fluctuates at
  the same ~0.005 scale as the partition-borne error and the corrected
  meter's residual. **The completeness meter's noise floor is ~0.005
  absolute; it is certified at that precision, on any partition, with
  the propagation correction.** Nothing below that scale is chased.
- **Free confirmation:** the η-floor law reproduced on fresh worlds
  under a fresh master seed (third independent confirmation; ordering
  exact at both energies).

Defects #36–#38 recorded (#38: regression grain unstated — executor
pinned it pre-arms). Routing P2N's registered fallback ("synthesis
without a taxometer") is OVERRIDDEN by the graded adjudication above in
one respect only: the synthesis proceeds WITH the taxometer, scoped to
its certified precision and preconditions. **The M4-L line closes at
three legs.** Consequences executed today: IDT appendix S (the final
answer to the reverse question); `docs/SUICA_M4_L_TYPOLOGY_LINE_SYNTHESIS.md`;
the defense phase (pre-authorized by the owner's standing instruction,
shelved during this line) resumes at D1.
