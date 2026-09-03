# GRAIN M4-W — the disjoint-cohort transport line

Opened 2026-08-18, directly after the U-line's full close. The U-line left
one interval open (whether the common mass outlasts the distinctive tail —
7/7 point estimates one side, no CI excluding zero) and ruled same-corpus
escalation out as significance-chasing. The recon that opened this line:
the PANDORA comments file contains **10,296 distinct authors**, of whom
**8,895 are OUTSIDE the 1401 Big5 cohort** — a disjoint cohort on the same
platform, 14.6M comments, with the full event structure the selection
machinery needs (MBTI9K and TWITTER corpora were reconned and REJECTED:
no per-comment venue/timestamp columns).

Transport semantics (Q-line vocabulary): a LAW transports when its
re-instantiated measurement on a fresh cohort reproduces it; an INSTRUMENT
transports when the frozen artifact does. W1 runs the law arms as primary
and the instrument arms as sensitivity.

Governance: label-free THROUGHOUT the line's opening phase —
`author_profiles.csv` is never opened (the disjoint authors' MBTI labels
stay closed); the ID-leak scan universe widens to ALL 10,296 author names;
the cohort listing itself is a gitignored artifact built by predicate;
aggregates only; EXPLORATORY corpus-level; no person claims; no
cross-corpus author linkage (none exists — one file, disjoint subsets).

## Line charter

- **W1 (registered below):** slow-time law transport — the decay-rate
  contrast Λ at ~9× authors (closing the U-line's open middle) + the
  sealed-prediction transport table (floor share, D, λ ordering).
- **W2 (named, not registered):** fast-time transport — U1's order
  channel (rho, dwell dominance, cross-thread share) and the T-line
  branch-code transport (T1's open item).
- **W3 (named, not registered):** the label side, only after W1/W2 close
  label-free and only under a fresh stamped design (the disjoint cohort's
  labels are MBTI axes, not Big5 — a different join class needing its own
  governance review).

---

## W1 — slow-time law transport and the closing of the middle (registered BEFORE run, 2026-08-18)

### 4W header

- **Object.** The U-line's slow-time laws as SEALED PREDICTIONS carried to
  a disjoint author cohort: (1) the decay-rate contrast Λ (the open
  middle); (2) DRIFT_WITH_CORE's floor share and D; (3) the layer ordering
  (λ_taste smallest). When × transport; author structure only.
- **Fixed (pinned BY FORMULA per #81, with provenance).** Block
  construction, bins, epoch-matched exact stratified cross baseline,
  within-quarter permutation, paired cluster bootstrap — all from the
  committed U2/U2b/U2c runners (`scripts/run_suica_m4_u2_persistence_curve.py`,
  `scripts/run_suica_m4_u2b_persistence_budget.py`,
  `scripts/run_suica_m4_u2c_decay_rate_contrast.py`), reused by
  import-by-file. Key formulas quoted: block features √(count/K), L2, over
  the arm's vocabulary; E(b) = mean self cosine − epoch-matched exact
  stratified cross mean (quarter = 91.3 d from corpus min);
  λ_row from OLS of log E_row(b) on realized per-bin mean gap (years),
  five verdict bins, 3y+ descriptive only; Λ = λ_distinct − λ_common;
  intersection predicate per block: m ≤ common-events ≤ K − m; K = 50.
- **Varied.** The cohort (disjoint 8,895); the vocabulary instantiation
  (law arm: SR0's RULE re-instantiated on this cohort → floor 89,
  1,443 communities; instrument arm: the frozen 1191 vocabulary and the
  frozen 32-community Common set applied as-is); typology ablation.
- **What falsifies.** Λ CI strictly below 0 (DISTINCT_SLOWER) refutes the
  U-line's provisional sign on fresh authors; transport-table BREAKS
  refute cohort-generality of the corresponding law.
- **Layer.** R.

### Census (planner arithmetic #43; predicates exact per #77/#78, executed 2026-08-18)

| quantity (exact predicate) | value |
|---|---|
| disjoint events (parseable author+subreddit+created_utc) | 14,634,702 |
| authors_seen | 8,895 |
| law-arm vocabulary (floor ceil(0.01·8895) = 89 users) | **1,443**; coverage 0.7587 |
| typology names in law vocabulary / in Common(0.5) | 22 / **8** |
| law-arm Common(0.5) | **71 communities** |
| pool (≥ 4 K=50 in-vocab blocks) | **6,111**; 213,489 blocks; 18 quarters |
| self pairs 2–3y / 3y+ | 2,591,663 / 1,004,737 |
| **gate (2–3y intersection, m=10): pairs / authors** | **1,211,631 / 3,241** |
| gate at m=5 | 1,790,865 / 3,746 |
| Λ width projection (#79b/#80b: √N scaling of U2c's REALIZED 0.1206 at 424 authors; m=10 primary avoids the m=5 per-unit-noise inflation) | **≈ ±0.044** vs sealed point +0.074 |

Anchor gates (#78): the executor re-executes these exact predicates and
must reproduce every census number above (STOP on mismatch). Sufficiency
targets (trivially cleared, stated for form): ≥ 400 authors and ≥ 100k
pairs in the 2–3y intersection.

### Sealed predictions (source values from committed U-line artifacts)

| quantity | source (cohort 849/424) | transport question |
|---|---|---|
| Λ | +0.0741 [−0.0558, +0.1854] | does the CI now EXCLUDE zero, and which side? |
| floor share E(2–3y)/E(0–90d), full row | 0.5348 [0.4203, 0.6320] | REPRODUCES / SHIFTS / BREAKS |
| D = E(0–90d) − E(2–3y), full row | 0.3058 [0.2364, 0.3855] | same |
| λ_full | 0.2943 [0.1895, 0.4405] | same |
| ordering: λ_taste smallest of four rows | held in both U2c arms | ORDER HOLDS / ORDER BREAKS |

Transport classification (effect-size keyed #75): REPRODUCES = target CI ∩
source CI nonempty AND same sign/cell; SHIFTS = disjoint CIs, same
sign/cell; BREAKS = different sign/cell. The table is SECONDARY; flags per
#73; it never routes the verdict.

### Verdict estimand and cells

Λ on the law arm at q = 0.5 (re-derived Common = 71 communities), m = 10,
paired cluster bootstrap B = 1000 (identical author draws across rows;
replicates with non-positive E dropped and counted). Null machinery
(#80a-compliant, pinned): per-bin E null centers by within-quarter
permutation (B = 499, expected 0); the Λ permutation null is posed on the
LINEAR slope contrast (domain-safe form — the log fit runs on real data
only, where all realized E > 0); the verdict criterion is the bootstrap CI.

1. **SIGN_UNRESOLVED** — Λ CI straddles 0 (interval statement; with the
   projected ±0.044 this now costs the provisional sign real credibility).
2. **COMMON_STANDING** — Λ CI > 0: the middle CLOSES; the T-line
   proposition acquires its When clause on fresh authors.
3. **DISTINCT_SLOWER** — Λ CI < 0: the provisional sign REFUTED on fresh
   authors; the U-line reading is re-examined before any theory move.

### Arms

| arm | role |
|---|---|
| LAW: re-derived vocab 1443, Common(0.5)=71, m=10, pool 6,111 | PRIMARY (verdict + transport table) |
| INSTRUMENT: frozen 1191 vocab + frozen 32-community Common | sensitivity; #73 flags vs law arm |
| clean_no_explicit_personality (23 names removed from the law vocab; 22 present; re-split, rerun Λ + floor share) | co-reported; #73 flags — the typology-enriched cohort (8/71 in Common) makes this arm's divergence informative |
| m = 5 | sensitivity |
| taste row (per-fold PPMI+SVD d=64, 5 folds, training-first-half-only, purity asserted) | the ordering prediction |
| 3y+ | descriptive only |

### Registered leans

- Λ > 0 with point ≈ +0.07/y (the sealed U2c point; disclosed-knowledge
  lean, not a fresh prediction).
- Floor share and D REPRODUCE (the DRIFT_WITH_CORE law is cohort-general).
- λ_taste smallest (ORDER HOLDS).
- Clean-arm lean (weak, informative either way): Λ survives the typology
  ablation within its CI class — the U-line's identity-robust pattern; a
  large clean-arm shift would localize the slow common mass in the
  typology communities themselves.

### Deliverables and discipline

Standard six: `scripts/run_suica_m4_w1_slow_transport.py` + tests
(cache-builder anchors; census-predicate reproduction; formula-level
contract tests against the imported U2c estimator on a shared toy;
permutation/bootstrap pairing invariants; ID-leak helper over the WIDENED
universe — all 10,296 names); gitignored `results/m4_w1_slow_transport/`
(includes the disjoint events cache and the cohort listing — never
committed); report `reports/SUICA_M4_W1_SLOW_TRANSPORT_REPORT.md`
(rule-24 tables; verdict + transport table; null centers; projection vs
realized; the eq-12 projection caution; three-year scoping language;
config block); outcome appended here; one CLAIMS_LEDGER row (EXPLORATORY,
label-free, corpus-level, disjoint-cohort transport named); ONE commit
`feat(m4-w): W1 — slow-time law transport — <VERDICT>`, never amended,
never pushed by the executor. Suite green (1197 + new) before commit.
`author_profiles.csv` is NEVER opened. SEED = 20260818; B_perm = 499;
B_boot = 1000.

---

## W1 outcome (executor, 2026-08-18)

**`COMMON_STANDING`. The U-line's open middle CLOSES.**

`Λ = λ_distinct − λ_common = +0.0911/y [+0.0628, +0.1490]` on the LAW arm at
the registered q = 0.5 / m = 10 (125,555 blocks; 1,211,631 self pairs and
3,241 contributing authors in the 2–3y bin; paired cluster bootstrap
B = 1000, all 1,000 replicates retained). The bootstrap interval — the
registered verdict criterion — lies strictly above zero on a cohort that
shares no author with the one that produced the provisional sign. The
DISTINCTIVE carrier decays faster than the COMMON carrier; over the observed
three-year window the common mass is the standing part.

Realized half-width **0.0431** against the registered projection **±0.044**
(ratio 0.979) — the #79b/#80b √N scaling was accurate to 2%.

### Census (#78) — every pinned predicate re-executed, 16 of 16 PASS

Disjoint events 14,634,702; authors_seen 8,895; floor ceil(0.01·8895) = 89 →
1,443 communities at coverage 0.7587; Common(0.5) = 71 communities carrying
0.5022 of the in-vocabulary mass; typology names 22 in vocabulary / 8 in
Common; pool 6,111 authors and 213,489 blocks over 18 quarters; self pairs
2–3y 2,591,663 (3y+ 1,004,737); gate m=10 1,211,631 pairs / 3,241 authors;
gate m=5 1,790,865 / 3,746. Nothing drifted; the leg never needed its STOP.

### The four-row rate table (LAW arm)

| row | λ (1/y) | 95% CI | R² | F = E(2–3y)/E(0–90d) | D |
|---|---|---|---|---|---|
| full (1,443) | 0.1236 | [0.0591, 0.2520] | 0.925 | 0.7360 | 0.1408 |
| common (71) | 0.0901 | [0.0362, 0.2069] | 0.942 | 0.8014 | 0.1117 |
| distinctive (1,372) | 0.1812 | [0.1045, 0.3223] | 0.923 | 0.6392 | 0.1734 |
| taste (PPMI+SVD d=64) | 0.1010 | [0.0854, 0.1345] | 0.982 | 0.7838 | 0.0610 |

Nulls (#80a-compliant): per-bin within-quarter permutation centres are
≤ 5e-05 in absolute value, i.e. zero to four decimals as the construction
requires. The Λ null is posed on the LINEAR slope contrast — realized
+0.0236 against a null centre of +0.00006 and a null band
[−0.00083, +0.00086], all 499 replicates finite. The LOG-form null retains
only 37 of 499 replicates (a relabelling drives E(b) through zero), which is
why the registration put the null on the linear form; neither null routes.

### Arms — all four land in the same cell

| arm | Λ | 95% CI | cell |
|---|---|---|---|
| LAW (q=0.5, m=10, Common 71) | +0.0911 | [+0.0628, +0.1490] | `COMMON_STANDING` |
| m = 5 sensitivity | +0.0953 | [+0.0648, +0.1509] | `COMMON_STANDING` |
| INSTRUMENT (frozen 1191 / Common 32) | +0.0644 | [+0.0168, +0.1107] | `COMMON_STANDING` |
| clean_no_explicit_personality (22 removed) | +0.0780 | [+0.0484, +0.1397] | `COMMON_STANDING` |

The instrument arm is the informative one for transport semantics: 1,187 of
the 1,191 frozen community names exist in the disjoint stream, but the frozen
32-community Common set holds only **0.3473** of that cohort's mass where it
held 0.50 on the source. The frozen artifact does NOT carry its own mass
split — and the contrast survives anyway, weaker and still strictly positive.
The LAW transports; the INSTRUMENT transports too, at a cost in effect size.

The clean arm matters here because the cohort is typology-enriched: removing
the 22 explicit-typology communities (leaving 1,421, re-blocked, re-pooled,
mass split re-derived to Common(0.5) = 84) moves Λ from +0.0911 to +0.0780,
same cell, overlapping intervals. The slow common mass is NOT localized in
the typology forums.

### Sealed-prediction transport table (SECONDARY — never routed)

| quantity | source | target | classification |
|---|---|---|---|
| Λ | +0.0741 [−0.0558, +0.1854] | +0.0911 [+0.0628, +0.1490] | **BREAKS** (cell) |
| floor share, full row | 0.5348 [0.4203, 0.6320] | 0.7360 [0.5453, 0.8568] | **REPRODUCES** |
| D, full row | 0.3058 [0.2364, 0.3855] | 0.1408 [0.0850, 0.2224] | **SHIFTS** |
| λ_full | 0.2943 [0.1895, 0.4405] | 0.1236 [0.0591, 0.2520] | **REPRODUCES** |
| ordering: λ_taste smallest | ORDER HOLDS | ORDER BREAKS | **BREAKS** |

Two BREAKs, and both need reading rather than reciting:

1. **Λ "BREAKS" as a PRECISION GAIN, not a contradiction.** The intervals
   OVERLAP and the sealed source point +0.0741 lies INSIDE the target
   interval. The source half-width was 0.1206 and could not exclude zero;
   the target's is 0.0431 and does. The registered rule keys on the CELL, so
   it lands in BREAKS — the substance is confirmation of the source's
   direction at a width the source never had. The planner should decide
   whether the #75 classification wants a fourth label for this case.
2. **The ordering BREAKS narrowly.** λ_taste = 0.1010 is rank 2 of 4,
   0.0109/y above the slowest row (`common` at 0.0901), and the taste row's
   own interval [0.0854, 0.1345] CONTAINS the common row's point. Order
   slowest-first: common 0.0901 < taste 0.1010 < full 0.1236 < distinct
   0.1812.

Floor share and D split classification for one reason: the disjoint cohort's
curve is FLATTER at both ends. The sealed values imply a source curve running
0.6574 → 0.3516; the target full row runs 0.5332 → 0.3925. Lower start,
higher finish — the RATIO reproduces while the DIFFERENCE shifts down. Part
of that gap is the disclosed provenance asymmetry (the sealed floor share and
D come from U2's 849-author POOL arm while Λ and λ_full come from U2c's
INTERSECTION set), and this leg does not separate the two contributions.

### Registered leans

- **L1 (Λ > 0 at ≈ +0.07/y): HELD** — realized +0.0911, cell
  `COMMON_STANDING`.
- **L2 (floor share and D REPRODUCE): PARTIAL** — floor share REPRODUCES, D
  SHIFTS.
- **L3 (λ_taste smallest): MISSED**, narrowly — rank 2, inside its own CI of
  the winner.
- **L4 (clean arm survives within CI class): HELD.**

### Governance

Label-free throughout — `author_profiles.csv` was never opened, and the
disjoint authors' MBTI labels stay closed. The ID-leak universe was widened
to all 10,296 author names as registered; the scan returns **0 NEW hits**.
Three raw hits exist in `docs/CLAIMS_LEDGER.md` and all three are DICTIONARY
COLLISIONS — candidate author names that are ordinary English words or bare
digit runs, occurring in prose committed long before this line opened. They
are separated from leaks mechanically, never by hand: a hit counts as
pre-existing only when the same scanner produces the identical hit, same file
and same line, on that file's version at HEAD. W1's own outputs have no HEAD
version and therefore zero tolerance, and they are clean. **This is a
governance finding the planner should rule on: at 10,296 candidates the
substring scan has a false-positive floor, and the widened universe will keep
paying it on every future leg.**

The cohort-composition caveat binds every claim above: the disjoint cohort is
PANDORA's MBTI-labelled population, typology-enriched by construction (8 of
71 Common communities are typology names). A law that transports here has
been carried to a different KIND of author population on the same platform,
not to a fresh sample of the same one.

Artifacts: `results/m4_w1_slow_transport/` (gitignored — includes the events
cache and the disjoint cohort listing, never committed). Report:
`reports/SUICA_M4_W1_SLOW_TRANSPORT_REPORT.md`. Runtime ≈ 24 minutes;
17 estimator calls; SEED 20260818, B_perm 499, B_boot 1000.

## W1 planner adjudication (2026-08-18)

**Verdict ACCEPTED: `COMMON_STANDING` — the U-line's open middle CLOSES on
fresh authors.** Λ = +0.0911/y [+0.0628, +0.1490] on 3,241 authors sharing
nobody with the source cohort; all four arms in the same cell; the width
projection realized at ratio 0.979 (the #79b/#80b machinery is now
calibrated on two consecutive exact hits); anchor gates 16/16 first try.

**The fourth transport class is ADOPTED (executor's request granted).**
**RESOLVES** = same sign; the source CI straddled a cell boundary that the
target CI excludes; the source point lies inside the target CI. It sits
between REPRODUCES and SHIFTS in the #75 scheme. Λ's row is reclassified
BREAKS → RESOLVES at adjudication (the committed report stands as written;
this note carries the reclassification). A transported precision gain is
not a contradiction and must never be filed as one.

**The When clause lands — in the TWO-REGIME form.** The strict ordering
(λ_taste smallest) did NOT transport: on the disjoint cohort the slow end
is {common 0.0901, taste 0.1010} — statistically tied (taste's CI contains
common's point) — and the fast end is the distinctive tail (0.1812),
cleanly separated. The licensed proposition, revised to what BOTH cohorts
support: **the distinctive is the ink — it fades fastest; what stands is
the common scaffold and the taste direction written through it.** The
three-layer strict ordering is retired in favor of the two-regime
statement; the ordering break carries its #73 flag.

**Law vs instrument — a textbook dissociation.** The LAW transports fully
(re-instantiated vocabulary and split → strict positive Λ). The frozen
INSTRUMENT transports at reduced effect (+0.0644, still positive) and
demonstrably does not carry its own mass split (the 32-community Common
holds 0.3473 of this cohort's mass against 0.50 at source). Q-line
vocabulary applies verbatim: transport the rule, not the artifact. The
clean arm defuses the cohort-composition worry: the slow common mass is
NOT the typology forums (Λ +0.0780 with them removed).

**Cohort-level persistence differences are real and now on record.** All
λ levels are lower on the disjoint cohort (λ_full 0.1236 vs 0.2943): the
disjoint signature is more persistent across the board; the floor-share
LAW reproduces while the absolute drop D shifts. Caveat carried: the
sealed floor-share/D rows were sourced from U2's pool arm while Λ/λ_full
came from U2c's intersection set (planner registration imprecision,
#81-family, recorded here without a number — it moves no classification
under the registered rule); and this cohort is typology-enriched by
construction.

**Defect #83 (planner, purchased; registry thirtieth note).** The widened
ID-gate ("0 hits over 10,296 names, blocking") was registered WITHOUT
pre-executing it against the already-committed tree: three dictionary
collisions (ordinary-word/digit usernames) pre-exist in old ledger prose,
so the gate as registered was unsatisfiable. The executor's mechanical
separation is ADOPTED AS STANDING POLICY: a raw hit is pre-existing only
if the same scanner reproduces the identical hit (same file, same line)
on that file's HEAD version; leg-authored files have no HEAD version and
carry zero tolerance; the verdict gate is NEW hits = 0. Convention: an
ID-gate over a widened universe is pre-executed against HEAD at
registration, and the pre-existing-hit policy is pinned with the gate.

**Lean record.** L1 HELD (disclosed-knowledge). L2 PARTIAL (floor share
reproduces; D shifts with the flatter-curve reading). L3 MISSED narrowly
(the ordering break above). L4 HELD.

**Line state.** W1 closed. W2 (fast-time + branch-code transport) is
registered next — with one design correction the planner owes to W1's
reading: **AUC-family transport must be SIZE-MATCHED** (pooled AUCs depend
on gallery size; the sealed T1/U1 values were measured at N = 1304/984,
so the primary transport arms draw seeded same-size subpools, with the
full-pool runs as sensitivity). W3 (labels, MBTI-axis class) stays gated
on its own governance review.

---

## W2 — fast-time and branch-code transport (registered BEFORE run, 2026-08-18)

The label-free transport phase's second and closing leg: U1's order channel
and T1's hierarchical selection identity as sealed predictions on the
disjoint cohort. The W1 adjudication's design correction binds: **AUC-family
transport is SIZE-MATCHED** — pooled AUCs depend on gallery size, and the
sealed values were measured at N = 984 (U1) and N = 1304 (T1), so the
primary arms draw seeded same-size subpools; full-pool runs are sensitivity.

### 4W header

- **Object.** When × transport (fast time): does the order channel
  re-instantiate (rho, dwell dominance, cross-thread retention)? And
  Where × Who × transport: does the branch code re-instantiate (flat /
  path / residual AUCs, stable depths)? Sealed predictions from committed
  U1/T1 artifacts; author structure only.
- **Fixed (by formula/import per #81).** The order machinery from
  `scripts/run_suica_m4_u1_order_identity.py` (bigram Hellinger sphere,
  α = 0.5; per-fold spherical k-means C = 24 + OOV on training authors'
  early halves, seed SEED+1000·fold; exact-bag within-half shuffles
  B = 499; rho = (AUC_real − null mean)/(1 − null mean); cluster bootstrap
  B = 1000). The branch machinery from
  `suica_core/hierarchical_selection_identity.py` via the T1 driver
  pattern (cross_fitted_hierarchical_identity; max_depth 6, min_leaf 30,
  folds 5, √freq Hellinger sphere input, bootstrap 1000, permutations 499;
  triple gate for stable depths).
- **Varied.** Cohort (disjoint); vocabulary (law arm 1,443); seeded
  size-matched subpools vs full pools.
- **What falsifies.** rho landing outside U1's ORDER_CHANNEL cell breaks
  the order law on fresh authors; a branch-code family collapse (flat or
  residual to chance, or no stable depth) breaks T1's law.
- **Layer.** R.

### Census (planner arithmetic #43; predicates exact per #77/#78, executed 2026-08-18 on the W1 cache)

| quantity (exact predicate) | value |
|---|---|
| law vocabulary re-verified (floor 89) | 1,443 |
| U1-pool: ≥ 50 law-vocab events per half, halves by FULL-STREAM median (boundary ≤ to early) | **7,247** |
| T1-pool: ≥ 40 total events (SR0 split-half rule) | **8,625** |
| size-match feasibility (984 / 1304) | both TRUE |
| tie rate, in-law-vocab spliced adjacency | **0.00882** (source 0.11522 — an attenuation source absent here; named cohort-instrument difference) |
| session ≤ 1 h fraction (same denominator) | 0.56147 |
| cross-thread share of all full-stream adjacencies | 0.73174 |

Anchor gates (#78): the executor re-executes and must reproduce these
exactly (STOP on mismatch). ID-gate per #83: pre-executed at W1 — raw 3
pre-existing dictionary collisions (HEAD-identical policy pinned), NEW
hits = 0 is the blocking form.

### Sealed predictions (from committed U1/T1 artifacts)

| family | quantity | source | class rule |
|---|---|---|---|
| order | rho (primary arm) | 0.2893 [0.2695, 0.3114] | four-class (#75 + RESOLVES) |
| order | stay-scalar arm rho | 0.1803 [0.1375, 0.2234] | four-class |
| order | cross-thread arm rho | 0.1626 [0.1373, 0.1895] | four-class |
| order | dwell same-state adjacency share | 0.7122 | descriptive |
| branch | flat AUC | 0.9837 | four-class (CIs from T1 artifacts; if a source CI is absent, classify by point-in-target-CI and flag the missing interval) |
| branch | path AUC | 0.7461 | same |
| branch | terminal residual AUC | 0.9552 | same |
| branch | stable depths | {1,2,3,4,5} | DEPTHS_REPRODUCE (same set) / DEPTHS_SHIFT (symmetric difference ≤ 1) / DEPTHS_BREAK |
| branch clean arm | flat/path/residual, depths | 0.9661 / 0.7317 / 0.9417, {1–4} | same; #73 flags |

### Design

- **Order family (VERDICT ROUTER).** Law vocabulary; primary pool = seeded
  draw of N = 984 from the 7,247 (numpy default_rng(SEED), uniform without
  replacement, sorted); 5-fold; U1's full pipeline (raw-adjacency primary
  arm plus stay-scalar and cross-thread arms with their own nulls).
  Full-pool 7,247 as sensitivity. rho's transport class routes:
  1. **ORDER_BREAKS** — target rho lands outside U1's ORDER_CHANNEL cell
     (point below 0.10 or above 0.33 with CI support, per U1's registered
     boundaries; includes the no-channel case) — NULL-first cell.
  2. **ORDER_SHIFTS** — same cell, disjoint CIs.
  3. **ORDER_TRANSPORTS** — REPRODUCES or RESOLVES class.
- **Branch family (co-primary; #73 flags on the verdict line, never
  routing).** Primary pool = seeded draw of N = 1304 from the 8,625
  (independent draw, same SEED stream documented); halves by SR0's
  per-author median rule; T1's full+clean arms at d6/l30 with the triple
  gate; SEED = 20260818. Full-pool run NOT registered (compute-priced;
  named as future sensitivity).
- Registered descriptives: null location vs bag ceiling (order); tie-rate
  and session differences as cohort-instrument notes; per-depth gain
  tables (branch).

### Registered leans

- Order: **ORDER_TRANSPORTS** with target rho in [0.25, 0.40] — the tie
  attenuation absent here (0.9% vs 11.5%) leans rho slightly HIGH of
  source; dwell dominance transports (stay-arm rho detected, > 50% of
  primary in share terms, descriptive).
- Branch: flat and residual REPRODUCE; path SHIFTS plausibly (path AUC is
  the most resolution-sensitive object); depths REPRODUCE or SHIFT by one.
- Clean arm: same-cell transport (the U/W identity-robust pattern).

### Deliverables and discipline

Standard six: `scripts/run_suica_m4_w2_fast_branch_transport.py` + tests
(seeded-subpool determinism; U1-formula agreement on a toy; suica_core
import contract; exact-bag invariance re-asserted on this cohort;
census-predicate reproduction; ID-leak with the #83 policy); gitignored
`results/m4_w2_fast_branch_transport/`; report
`reports/SUICA_M4_W2_FAST_BRANCH_TRANSPORT_REPORT.md` (rule-24 tables;
both families' transport tables with classes and flags; null locations;
size-matching stated on every AUC row; cohort-composition caveat;
boundaries with the eq-12 projection caution); outcome appended here; one
CLAIMS_LEDGER row; ONE commit
`feat(m4-w): W2 — fast-time and branch-code transport — <VERDICT>`; suite
green (1236 + new). Label-free; `author_profiles.csv` never opened;
SEED = 20260818; B_perm = 499; B_boot = 1000.

---

## W2 outcome (executor, 2026-08-18)

**`ORDER_SHIFTS` (cell 2 of 3). The fast-time order channel RE-INSTANTIATES
on the disjoint cohort — in the same cell, at a lower level.**

Size-matched primary `rho = 0.2207 [0.2023, 0.2401]` on 984 authors drawn
from the censused 7,247, against U1's sealed `0.2893 [0.2695, 0.3114]` on its
own 984. Both points sit inside U1's `ORDER_CHANNEL` cell (0.10–0.33, CI
strictly above zero); the intervals do not meet. Same law, different
magnitude. The exact-bag shuffle p is at its floor (0.002 = 1/500) on all
three arms, and bag invariance is bit-exact per fold.

### Anchor gates (#78) — 10 of 10 PASS, first execution, nothing drifted

Disjoint events 14,634,702; authors 8,895; floor ceil(0.01 × 8895) = 89 →
**1,443** communities, and the recomputed list is BIT-IDENTICAL to the one W1
cached, not merely the same length. U1-pool **7,247**; T1-pool **8,625**; tie
rate on in-law-vocab spliced adjacency **0.00882**; session share **0.56147**;
cross-thread share of all adjacencies **0.73174**. The leg never needed its
STOP.

### The size-matched draws

One `numpy.random.default_rng(20260818)` stream, two draws, ORDER first and
BRANCH second: 984 of 7,247 (digest `e44097a2b36ad6ea`), then 1,304 of 8,625
(digest `b76122f923db8740`), both uniform without replacement and sorted
before use. The draw ORDER is part of the pin — swapping it changes both
subpools. 138 authors fall in both draws, which is harmless: the families are
separate instruments, not a joint test.

### Order family (VERDICT ROUTER) — two of three arms REPRODUCE

| arm (N = 984) | target rho | source rho | class |
|---|---|---|---|
| raw adjacency (PRIMARY, routes) | 0.2207 [0.2023, 0.2401] | 0.2893 [0.2695, 0.3114] | **SHIFTS** |
| stay-rate scalar | 0.1778 [0.1344, 0.2191] | 0.1803 [0.1375, 0.2234] | **REPRODUCES** |
| cross-thread only | 0.1383 [0.1192, 0.1593] | 0.1626 [0.1373, 0.1895] | **REPRODUCES** |

The two channel-restricted arms land on top of their seals (deltas −0.0025
and −0.0243, both overlapping); only the FULL raw-adjacency arm moves enough
to separate. Null locations sit on the same side of the unigram bag ceiling
as at source (primary −0.0254 here against −0.0211 there).

### Full-pool sensitivity — the size correction, made visible

At the full censused gallery of 7,247 the same arm gives
`rho = 0.1762 [0.1700, 0.1831]` — same cell, but **0.0446 below** the
size-matched value across a 7.36× gallery change. Had this leg compared the
full pool against U1's 984-author seal it would have manufactured a transport
failure out of gallery size alone. The W1 adjudication's correction is
vindicated on its first use.

### Branch family (co-primary, #73 flags, never routing)

| arm | quantity | target | source | class |
|---|---|---|---|---|
| full | flat AUC | 0.9817 [0.9772, 0.9852] | 0.9837 | **REPRODUCES** |
| full | path AUC | 0.7154 [0.7028, 0.7425] | 0.7461 | **SHIFTS** ⚑ |
| full | residual AUC | 0.9722 [0.9617, 0.9761] | 0.9552 | **SHIFTS** ⚑ |
| clean | flat AUC | 0.9562 [0.9480, 0.9638] | 0.9661 | **SHIFTS** ⚑ |
| clean | path AUC | 0.6929 [0.6918, 0.7273] | 0.7317 | **SHIFTS** ⚑ |
| clean | residual AUC | 0.9486 [0.9374, 0.9580] | 0.9417 | **REPRODUCES** |
| full | stable depths | {1,2,3,4,5,6} | {1,2,3,4,5} | **DEPTHS_SHIFT** ⚑ |
| clean | stable depths | {1,2,3,4} | {1,2,3,4} | **DEPTHS_REPRODUCE** |

No branch quantity comes anywhere near chance: every AUC interval sits far
above 0.50, so the registered falsifier (a branch-code family collapse) does
NOT fire. The identity signal is intact; what moves is its DISTRIBUTION
between the frozen path and the terminal residual. Path drops (−0.0307) while
residual RISES (+0.0170): on the disjoint cohort the tree's frozen route
carries LESS of the identity and the within-leaf remainder carries MORE. T1's
own headline — that the tree does not exhaust the identity tail — is not
merely reproduced but strengthened.

Stable depths go one level DEEPER on the full arm. T1's depth 6 was
underresolved (N = 26, gain CI crossing zero); here depth 6 carries N = 257
and clears all three gates. The clean arm's depth set is identical to T1's.

**Missing source intervals (registered rule, applied to all six AUC rows).**
T1's committed artifacts publish per-depth gain CIs but NO interval for flat,
path or residual. Every branch AUC row therefore fell to the registered
fallback — classify by point-in-TARGET-CI — and every one carries the
missing-interval flag. The target intervals are this leg's own instrument
(RD-W2-2): an author-level cluster bootstrap, B = 300, re-running the SAME
pinned `cross_fitted_hierarchical_identity` on each resample with its
permutation/bootstrap counts at the minimum (a contract test shows those
counts leave the AUCs bit-identical). Measured bias runs −0.0004 to +0.0159;
the bias-corrected interval is carried per row, and it flips exactly one
classification (`clean.path`, which is REPRODUCES under the percentile
interval's lower edge 0.6918 and SHIFTS under the recentred one). That row
should be read as borderline, not as a finding.

### Cohort-instrument notes

The registration's named difference is real and large: the tie rate is
**0.00882** here against **0.11522** at source (13.1× lower). The session
share is 0.56147 against a source test-fold share of 0.6700, and the
same-state dwell share falls to **0.6141** from the sealed **0.7122** —
dwell is still the majority of adjacencies, but less dominant. OOV occupancy
rises (0.2801 vs 0.2071) and realized transitions per author fall (median
839.5 vs 1224.5). This cohort's fast-time instrument is simply coarser.

### Registered leans

- **L1 (ORDER_TRANSPORTS with rho in [0.25, 0.40], leaning HIGH): MISSED,
  and in the wrong DIRECTION.** The tie-attenuation argument predicted rho
  above source because ties cannot carry order; the realized rho is 0.0686
  BELOW source. The argument's premise held (ties are 13× rarer) and its
  conclusion failed. Recorded as an anomaly, not explained away.
- **L2 (dwell dominance transports): HELD** — stay-arm rho detected and
  strictly positive, dwell share 0.6141 > 0.5, though the share itself fell.
- **L3 (flat and residual REPRODUCE, path SHIFTS plausibly): PARTIAL** —
  flat REPRODUCES and path SHIFTS as leaned, but residual SHIFTS too, and
  UPWARD, which no lean anticipated.
- **L4 (depths REPRODUCE or SHIFT by one): HELD** — one deeper on full,
  identical on clean.
- **L5 (clean arm same cell): HELD** — all three clean quantities remain far
  above chance.

### Governance

Label-free throughout: `author_profiles.csv` was never opened. The ID-leak
gate over all 10,296 names returns **0 NEW hits** under the #83
HEAD-identical policy; 4 raw hits are dictionary collisions reproduced
identically at HEAD (three in `docs/CLAIMS_LEDGER.md`, one in this plan's own
W2 registration section). The scan is run TWICE and the numbers printed in
the report are verified to be the numbers of the report that carries them
(fixed point). The cohort-composition caveat binds: 22 explicit-typology
communities sit in the 1,443-name vocabulary, and the clean arm is the handle
on that.

Artifacts: `results/m4_w2_fast_branch_transport/` (gitignored). Report:
`reports/SUICA_M4_W2_FAST_BRANCH_TRANSPORT_REPORT.md`. Runtime ≈ 65 minutes
(order size-matched 358 s, order full-pool 2,313 s, branch arms 52 s, branch
AUC bootstrap 1,139 s). SEED 20260818, B_shuffle 499, B_boot 1000.

## W2 planner adjudication + LABEL-FREE PHASE CLOSE (2026-08-19)

**Verdict ACCEPTED: `ORDER_SHIFTS`.** The order channel re-instantiates on
fresh authors IN CELL: raw arm 0.2207 [0.2023, 0.2401] against the sealed
0.2893 (disjoint CIs, same ORDER_CHANNEL cell); the stay-rate scalar
REPRODUCES nearly exactly (0.1778 vs 0.1803) and cross-thread REPRODUCES
(0.1383 vs 0.1626). Two of three arms reproduce; the full-arm level is
cohort-owned. Null locations on the registered side of the bag ceiling;
exact-bag invariance bit-exact per fold; anchors 10/10 first try including
a bit-identical law-vocabulary recomputation.

**The size-matching correction was vindicated on first use.** Full-pool
rho 0.1762 at gallery 7,247 vs 0.2207 size-matched — a 0.0446 artifact of
gallery size alone (7.36×). Comparing the full pool against the sealed
984-author value would have manufactured a transport failure. Standing
guidance reinforced: AUC-family quantities are compared ONLY size-matched.

**Branch family accepted with its five #73 flags — and T1's open item
CLOSES.** No branch quantity approaches chance (the registered falsifier
does not fire): flat REPRODUCES (0.9817 vs 0.9837); the identity
DISTRIBUTION redistributes — path down (0.7154), residual UP (0.9722 vs
0.9552), depth 6 newly resolved at N = 257 where T1 had N = 26. This
STRENGTHENS T1's `TREE_DOES_NOT_EXHAUST_IDENTITY_TAIL` on fresh authors:
the tree code transports; the continuous tail carries even more here. The
clean arm's depth set reproduces identically ({1–4}). RD-W2-2 (T1
published no headline CIs; the registered point-in-target-CI fallback ran
on all six rows with flags; executor-supplied target CIs under a
bit-identity contract test) is accepted; clean.path is read as borderline,
as disclosed. RD-W2-1 (sensitivity-row bootstrap reduction) accepted.

**Lean record.** L1 MISSED IN THE WRONG DIRECTION: the tie-attenuation
premise held (0.9% vs 11.5%) but rho landed 0.0686 LOW — single-factor
attenuation reasoning does not predict cross-cohort levels. Guidance (not
a numbered defect; leans are allowed to miss and this miss is the
information): cross-cohort LEVEL leans from single-factor instrument
reasoning are retired; levels are cohort-owned. L2/L4/L5 held; L3 partial.
Dwell share 0.6141 vs 0.7122 — still majority, less dominant; this
cohort's fast-time instrument is coarser across the board. **Zero new
planner defects this leg** — the first since the U-line opened.

**THE LABEL-FREE TRANSPORT PHASE CLOSES (W1 + W2).** The phase's law, now
supported five ways: **structure and ratio laws transport; levels belong
to the cohort.** (Λ's sign closes and the floor-share ratio reproduces
while D and all λ levels shift — W1; the rho cell and the stay scalar
reproduce while the raw level shifts — W2; flat reproduces while
path/residual redistribute — W2.) Program state after this close: the
T-line is FULLY closed (branch transport was its last open item), the
U-line is fully closed (W1 closed its middle), and the W-line's label-free
phase is complete. W3 (the MBTI-axis label class on the disjoint cohort)
remains GATED on its own governance review — it is a new join class and
opens only under a fresh stamped design after that review. Named-open
elsewhere: R3 reconciliation derivation (theory-side), long-horizon floor
(external data), non-dwell order residual, finer-state order.
