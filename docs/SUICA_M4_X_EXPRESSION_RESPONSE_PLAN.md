# GRAIN M4-X — the expression-response line (Where × What, What × When on metadata)

Opened 2026-08-19, after the R-line's full close. The 4W empirical state
(`SUICA_4W_EMPIRICAL_STATE.md`) records crossings #2 (Where × What,
condition response) and #3 (What × When, state trajectories) as the last
crossings with no real-data measurement. This line opens them WITHOUT
reading a single text body: the comments file carries
`word_count_quoteless` — the author's OWN-WORD count per comment, quotes
removed — a crude but honest expression-volume scalar (What-adjacent:
how much one says, not what one says; the boundary is permanent and
named on every claim).

Governance: label-free throughout the opening phase (`author_profiles.csv`
never opened); NO text bodies read (metadata columns only); ID-leak
universe = all 10,296 names under the #83 HEAD-identical policy;
aggregates only; EXPLORATORY corpus-level; no person claims; no
psychological naming (verbosity is a technical expression-volume object).

## Line charter

- **X1 (registered below):** the venue response of expression volume —
  crossing #2's first real-data projection: do communities condition an
  author's verbosity, and is the author×community response profile
  REPRODUCIBLE (persistent across the author's own time halves)?
- **X2 (named, not registered):** What × When paths — burst/order/dwell
  structure of expression volume in event time (the U1 machinery pointed
  at y instead of state identity).
- **X3 (named, not registered):** the stage-E trait join (verbosity level
  / response profile vs Big5) — the literature's classic claims meet the
  dissociation family; only under fresh stamps, only after X1/X2.

---

## X1 — the venue response of expression volume (registered BEFORE run, 2026-08-19)

### 4W header

- **Which 4W object.** Crossing #2 (Where × What): the condition-response
  operator R_v's first real-data projection — the persistence of the
  author×community interaction profile of y = log(1 + wcq). The theory's
  projection caution binds: profile persistence is ONE static projection
  of the response operator; a null is a statement about this projection.
- **What is fixed.** y = log(1 + word_count_quoteless) (pinned #70;
  word_count as sensitivity); halves by the author's FULL-STREAM median
  created_utc (≤ to early); communities = raw subreddit names (no
  vocabulary floor — the estimand needs no cross-author support floor;
  a W1-law-vocabulary restriction is a sensitivity arm); cell (u, c, h)
  eligible at n_min comments; author eligible at ≥ k_min shared
  communities whose BOTH halves are eligible.
- **What varies.** Cohort (disjoint primary; Big5 replication); n_min ∈
  {10 primary, 5}; k_min = 3; raw vs law-vocabulary community universe.
- **What falsifies.** R inside its permutation band AND the interaction
  covariance share CI including 0 → NO_REPRODUCIBLE_RESPONSE for this
  projection.
- **Layer.** R.

### Census (planner arithmetic #43; predicates exact per #77/#78, run 2026-08-19)

| quantity (exact predicate) | value |
|---|---|
| rows parseable (author+subreddit+created_utc+wcq) | 17,640,062 of 17,640,062 |
| wcq = 0 share | 0.0000 |
| authors | 10,296 (1,401 Big5 + 8,895 disjoint) |
| pool, disjoint, n=10/k=3 | **4,342** (PRIMARY) |
| pool, disjoint, n=5/k=3 | 5,842 (sensitivity) |
| pool, Big5, n=10/k=3 | **615** (replication arm) |
| median shared eligible communities (n=5) | 4 (disjoint) / 3 (Big5) |
| within-cell sd of y (median, n=5 early cells) | 0.9070 |
| R width projection (#79b: sd of a 3-point per-author correlation ≈ 0.5–0.7; SE ≈ 0.6/√4342) | ≈ ±0.018 at the primary pool — detection from R ≈ 0.02 |

Antecedent verification (#84): the per-author correlation is defined at
k_min = 3 (df = 1, noisy per author, powerful in the mean over 4,342);
no saturation risk (within-cell noise bounds every per-author correlation
away from 1 — headroom exists by construction and is REPORTED from the
realized distribution); the null's coordinate is the mechanism's own
(community-label permutation); all design parameters named here.

### Estimands (transforms pinned #70; own nulls #68/#66)

Within each half, on the eligible cell grid: double-centered deviations
v_{u,c,h} = ȳ_{u,c,h} − ā_{u,h} − b̄_{c,h} + ḡ_h (author and community
means over eligible cells, unweighted; the incomplete-design caveat is
named and the permutation null preserves the design exactly).

1. **R (response-profile persistence):** mean over eligible authors of
   corr_c(v_{u,·,early}, v_{u,·,late}) over the author's shared eligible
   communities. Own null: community labels permuted within
   (author, half) among that author's eligible cells, B = 499; band =
   2.5/97.5 percentiles; cluster bootstrap over authors B = 1000 for the
   CI on R.
2. **The reproducible variance budget** (cross-half covariance trick —
   attenuation-free): shares of comment-level Var(y) from
   (a) author: cov over authors of half-means; (b) community: cov over
   communities of half-means (size-weighted; unweighted sensitivity);
   (c) interaction: cov over eligible cells of (v_early, v_late),
   unweighted (weighted sensitivity); residual = remainder. Same
   permutation null for the interaction share; cluster bootstrap CIs.
3. Descriptives (non-verdict): per-cohort budgets; the realized
   per-author correlation distribution (the headroom report); community
   count sensitivity.

### Cells (NULL-first #55; effect-size keyed #75 on the interaction share, with R co-reported)

1. **NO_REPRODUCIBLE_RESPONSE** — R inside its band AND interaction
   share CI includes 0.
2. **RESPONSE_TRACE** — detected; interaction share < 0.02 of total
   variance.
3. **IDIOSYNCRATIC_RESPONSE** — share ∈ [0.02, 0.10].
4. **RESPONSE_MAJOR** — share > 0.10.

Boundary rationale: author-main verbosity shares in the literature run
~0.2–0.4 on log scales and community norms ~0.05–0.15; a persistent
interaction at 2–10% of total variance would be a real second-order
channel. CI membership decides; straddles reported as straddles.
Big5-replication and law-vocabulary divergences carry #73 flags; the
primary routes.

### Registered leans

- Interaction DETECTED and small: R ∈ (0.05, 0.30], share ∈
  (0.005, 0.05] — the RESPONSE_TRACE/IDIOSYNCRATIC boundary region.
  Rationale: on the log scale, "personal baseline × venue norm"
  (multiplicative) is ADDITIVE and lands entirely in the mains; the
  interaction asks for more than that — genuine per-author venue
  idiosyncrasy — and the dissociation-family record counsels modesty
  about second-order channels.
- Descriptive leans: author main share ∈ [0.15, 0.45]; community main ∈
  [0.02, 0.15].
- Replication lean: Big5 cohort same cell (#73 if not).

### Deliverables and discipline

Standard six: `scripts/run_suica_m4_x1_venue_response.py` + tests
(double-centering correctness on a hand toy; permutation preserves the
design; covariance-budget recovery on a synthetic two-way world with
planted shares — the #76 operating point: planted shares at realistic
magnitudes {author .3, community .08, interaction .02} and a null world
{interaction 0}, both must be recovered before the real arm, A1 stop on
failure; headroom report; ID-leak helper); gitignored
`results/m4_x1_venue_response/` (fresh metadata cache with anchors:
17,640,062 parseable rows, 10,296 authors); report
`reports/SUICA_M4_X1_VENUE_RESPONSE_REPORT.md` (rule-24 tables; budget
table with CIs and null bands per cohort; R with band and CI; headroom
distribution; boundaries: metadata-only, expression VOLUME not content,
the projection caution, no psychological naming); outcome appended here;
one CLAIMS_LEDGER row (EXPLORATORY, label-free, corpus-level,
metadata-only named); ONE commit
`feat(m4-x): X1 — the venue response of expression volume — <VERDICT>`;
suite green (1304 + new); ID-leak scan under #83 (universe 10,296).
SEED = 20260819; B_perm = 499; B_boot = 1000. `author_profiles.csv`
never opened; no text bodies read.

---

## X1 outcome (executor, 2026-08-19)

**VERDICT — `A1_STOP__SYNTHETIC_GATE_FAILED`.** Part 0 failed on BOTH of its
clauses, so the A1 stop fired before any real estimand was computed. There is
no corpus value of R, of the variance budget, or of the headroom distribution
anywhere in this leg's artifacts. The registered cells are UNSCORED —
`NO_REPRODUCIBLE_RESPONSE` is *not* the outcome; the leg reached no reading of
crossing #2 in either direction.

Harness `scripts/run_suica_m4_x1_venue_response.py`; tests
`tests/test_m4_x1_venue_response.py`; artifacts
`results/m4_x1_venue_response/` (gitignored); report
`reports/SUICA_M4_X1_VENUE_RESPONSE_REPORT.md`. Runtime 39.8 s end to end
(17.6M-row metadata stream 17.5 s). SEED 20260819, B_perm 499, B_boot 1000.

### The census reproduced exactly (#78) — all ten pins

17,640,062 parseable rows of 17,640,062 streamed; wcq = 0 share 0.0000 (119
rows); 10,296 authors = 1,401 Big5 + 8,895 disjoint; pools **4,342** (disjoint
n=10/k=3, PRIMARY), **5,842** (disjoint n=5/k=3), **615** (Big5 n=10/k=3); law
vocabulary floor 89 = ceil(0.01 x 8,895) giving exactly **1,443** communities.
Two inherited cross-checks also land on the nose: 14,634,702 disjoint events
(W1's cache) and 3,005,360 Big5 events (U2's `ANCHOR_EVENTS`). The eligibility
predicate is therefore not in doubt.

Two NON-GATING descriptives in the census table did not reproduce and are
recorded as anomalies: the within-cell sd of y (registered 0.9070, observed
0.9035 over disjoint n=5 early cells, ddof=1), and the median shared eligible
communities (registered 4 disjoint / 3 Big5; observed 5.0 / 4.0 over authors
with at least one shared pair and 7.0 / 8.0 over eligible authors — off by one
under the first reading in both cohorts). Neither gates anything.

### Part 0 — what the gate found (all numbers synthetic; no real y)

The world is built at the primary arm's REALIZED design density by reusing the
real skeleton (4,342 authors, 5,369 communities, 42,141 occupied slots, the
real per-cell comment counts) and planting entirely synthetic y. Tolerance
`max(0.01, 3 x replicate sd)` over 8 replicates; the 0.01 floor is half the
width of the registration's own TRACE/IDIOSYNCRATIC boundary.

| planted | value | recovered (8 reps) | tolerance | bias | status |
|---|---|---|---|---|---|
| author main | 0.3000 | 0.3261 | 0.0217 | +0.0261 | FAIL |
| community main | 0.0800 | 0.1351 | 0.0268 | +0.0551 | FAIL |
| interaction | 0.0200 | 0.0613 | 0.0100 | +0.0413 | FAIL |

Null world (interaction planted at exactly 0.0000): the estimator returns
0.0458 +- 0.0014 over 8 replicates; on the scored replicate 0.0482 with a
cluster-bootstrap CI [0.0566, 0.0738] that excludes 0 **and does not even
cover its own point estimate**, and R = 0.4524 against a permutation band
[0.2184, 0.2439]. Both honesty clauses fail.

Ablations localize the artifact exactly. Author main ONLY (0.3000, nothing
else): interaction reads 0.0383, R reads 0.4244. Community main ONLY (0.0800,
nothing else): interaction reads 0.0078, R reads 0.0006. The two leaks are
additive (0.0383 + 0.0078 = 0.0461 vs the null world's 0.0458), and the AUTHOR
main through thin communities is the dominant one.

### The mechanism, stated once

Double-centering removes the mains EXACTLY only on a complete grid. On an
incomplete one, `v` retains two residues: a per-COMMUNITY term (the community's
own author-set mean of the author main) and a per-AUTHOR term (the author's own
community-set mean of the community main). Both are the same in EARLY and LATE
by construction, because a main effect does not change across halves — so the
cross-half covariance trick, which is attenuation-free against noise, is
defenceless against them. The per-author term is removed by the within-author
centering inside the correlation, which is why the community main leaks 0.0078
into the share but 0.0006 into R; the per-community term survives both, which
is why the author main leaks into R at 0.4244.

The registered permutation null does not absorb this. Permuting community
labels within (author, half) moves the community MAIN together with the
interaction, injecting within-author variance the real data does not carry, so
the band ([0.0423, 0.0440]) sits BELOW the artifact it is meant to calibrate
(0.0482) — anti-conservative, not conservative.

### Defect candidates recorded for the planner (nothing below was run)

1. **The #84 antecedent verification checked the wrong antecedent.** It
   verified the ceiling (no saturation) and the null's coordinate, and asserted
   that "the estimand needs no cross-author support floor". It never asked
   whether the estimator's ZERO is zero on the realized design. It is not.
2. **The routing statistic is zero-referenced but its zero moves.** Cell 1
   asks whether the share's CI includes 0; on an incomplete grid the
   no-response value is the leak, and the author cluster bootstrap makes the
   grid MORE incomplete (about 63% distinct authors per replicate), so the CI
   tracks the leak rather than the estimand.
3. **The permutation null is not a clean mechanism coordinate here.** A null
   that resamples the interaction while HOLDING both mains fixed would be the
   matching coordinate.
4. **The headroom clause is stated about the wrong object.** At k_min = 3 a
   per-author Pearson correlation over three points is NOT bounded away from 1
   (4.70% of authors above 0.99 on the planted world, extremes rounding to
   +-1.0000). What is bounded is the MEAN over thousands of authors.
5. **Support is a PRIMARY question, not a sensitivity.** The primary design has
   a median of 2.0 eligible authors per community and 5.77% of slots in
   single-author communities; the law-vocabulary arm the registration demoted
   to a sensitivity has 9.0 and 0.32%. The leak scales with exactly that
   sparsity, so a cross-author support floor is a candidate PRIMARY — the
   opposite of what the registration reasoned.

### What still stands

The 4W header, the census, the eligibility predicate, the half rule and the
cell/lean boundaries are all intact and re-usable: only the ESTIMATOR's
calibration failed. The metadata-only boundary held throughout — no text body
was read, `author_profiles.csv` was never opened, and the ID-leak scan over all
10,296 author names cleared with 0 NEW hits (3 pre-existing dictionary
collisions in `CLAIMS_LEDGER.md`, carried unchanged from HEAD under the W1
policy).

## X1 planner adjudication (2026-08-19)

**Registered outcome ACCEPTED: `A1_STOP__SYNTHETIC_GATE_FAILED` — no
cell, and expressly NOT `NO_REPRODUCIBLE_RESPONSE`.** The gate
architecture did precisely what #76/#84 built it to do: it caught a real
estimator defect on the realized design BEFORE any real estimand ran
(39.8 s leg; ten anchors exact; the real skeleton reused with wholly
synthetic signal). The executor's restraint is noted with approval
twice: Part 0 was NOT run on the better-conditioned law-vocabulary
skeleton (choosing an arm by gate performance after a failure is a
planner re-registration, not an executor pivot), and the #83 scan
caught a PANDORA username ("independent") inside the executor's own
docstring — the widened-universe policy working end to end.

**The mechanism, owned.** Double-centering removes mains EXACTLY only
on a complete grid. The registered primary grid has fill 0.0018 with a
median of 2.0 eligible authors per community and 5.77% single-author
slots; v therefore retains per-community residues of the author main
and per-author residues of the community main — IDENTICAL across
halves — and the "attenuation-free" cross-half covariance has no
defence: the null world read share 0.0458 (2.3× the registered TRACE
boundary) and R = 0.45, with the registered permutation null sitting
BELOW the artifact (it moves the community main together with the
interaction). Ablations localize the leak to the author main.

**Defect #85 (planner, purchased; registry thirty-second note).** Three
parts. (a) The estimator's ZERO POINT was never verified on the
REALIZED design geometry — #84's antecedent checks covered the ceiling
and the null's coordinate but not whether zero reads zero on this
skeleton; (b) the zero-reference is not bootstrap-stable (the author
cluster bootstrap makes the grid ~37% more incomplete and moves the
zero); (c) the registration's "the estimand needs no cross-author
support floor" was exactly backwards — the ESTIMAND needs no semantic
floor, but the ESTIMATOR needs cross-author SUPPORT, so support
conditioning (fill, units-per-condition, singleton share, connectivity)
is a PRIMARY design parameter. Convention: every estimator whose
exactness depends on design completeness carries a zero-point gate ON
THE REALIZED SKELETON (real incidence, synthetic signal), verified
under bootstrap resampling as well; support conditioning is censused
and pinned as primary design; nulls hold fixed everything the estimand
claims to have removed. Minor (#77 family, non-gating): two census
descriptives did not reproduce under ambiguous predicate readings
(within-cell sd 0.9035 vs 0.9070; the shared-communities median) — the
gated pool counts matched to the unit.

**X1b is registered below** (the SR4/U2c completion pattern): design
repaired (law vocabulary + support floor, censused on X1's committed
cache) AND estimator repaired (exact two-way fixed effects on the
connected component), with the gate that failed here required to PASS
there before any real number.

---

## X1b — the venue response, repaired design and exact estimator (registered BEFORE run, 2026-08-19)

### What changes from X1 (everything else inherits)

1. **Design (support conditioning as primary, #85c).** Community
   universe = the LAW vocabulary (1,443; floor 89, reproduced on X1's
   cache); pinned predicate order with NO fixed-point iteration:
   (1) cells n ≥ 10, disjoint authors, law communities; (2) shared
   pairs (both halves eligible); (3) community support floor s = 5
   authors with a shared pair; (4) authors with ≥ 3 surviving shared
   communities; (5) largest connected component of the bipartite
   graph. Census (planner, on X1's committed cell cache, 2026-08-19):

   | s | authors | communities | shared pairs | fill | med authors/comm | singletons | LCC |
   |---|---|---|---|---|---|---|---|
   | 3 | 3,686 | 1,145 | 32,415 | 0.0077 | 11.0 | 0.0009 | 1.000 |
   | **5 (PRIMARY)** | **3,665** | **1,000** | **31,899** | 0.0087 | **12.5** | **0.0000** | **1.000** |
   | 8 (sensitivity) | 3,595 | 780 | 30,561 | 0.0109 | 16.0 | 0.0000 | 1.000 |

2. **Estimator (exact zero on any connected design).** Per half,
   two-way fixed effects on the eligible shared cells by alternating
   projections (within-transform demeaning iterated to max-change
   ≤ 1e-10; exact main removal is guaranteed on the connected
   component, and the LCC covers 100% of authors at every s); v = the
   FE residual of the cell mean. R and the interaction covariance
   share defined on v exactly as in X1; the author and community main
   shares from the same cross-half covariances as in X1 (mains are
   estimated pre-FE; only the interaction object changes).
3. **The gate that failed must now pass (Part 0, A1 stop).** The SAME
   realized-skeleton battery, now on the X1b skeleton: planted
   {author .30, community .08, interaction .02}, the NULL world
   (interaction 0), author-only and community-only ablation worlds,
   8 replicates; PASS = recovery within max(0.01, 3·replicate sd),
   null non-detection (share CI covers 0 AND its point within the
   band; R inside band), ablation leakage < 0.005, AND the
   bootstrap-zero check (#85b: the null world's cluster-bootstrap CI
   covers 0). Any clause fails → A1 stop, honest report.
4. **Replication arm.** Big5 cohort under the identical predicate
   chain; its pool is censused IN-LEG with a #69 floor of 300 authors
   — below the floor the arm reports as underpowered-descriptive
   rather than flagging #73.

Cells, leans, sensitivities (s = 8; n = 5; word_count), boundaries,
governance, and discipline inherit X1's registration verbatim
(metadata-only; no bodies; no profiles; #83 universe 10,296;
EXPLORATORY; the projection caution). Deliverables: standard six,
`scripts/run_suica_m4_x1b_venue_response_fe.py` + tests (FE exactness
on a hand toy AND on the realized skeleton; gate battery; predicate-
chain reproduction of the census table above as blocking anchors),
gitignored `results/m4_x1b_venue_response_fe/`, report
`reports/SUICA_M4_X1B_VENUE_RESPONSE_FE_REPORT.md`, outcome appended
here, one CLAIMS_LEDGER row, ONE commit
`feat(m4-x): X1b — the venue response, exact estimator — <VERDICT>`;
suite green (1359 + new). SEED = 20260819; B_perm = 499;
B_boot = 1000.

---

## X1b outcome (executor, 2026-08-19)

**VERDICT — `A1_STOP__SYNTHETIC_GATE_FAILED`.** Part 0 passed **8 of its 9
clauses** and failed one, so the A1 stop fired again and no real estimand was
computed. There is no corpus value of R, of the variance budget or of the
headroom distribution anywhere in this leg's artifacts. The four cells and all
five leans are UNSCORED; `NO_REPRODUCIBLE_RESPONSE` is expressly NOT the
outcome.

The registered repair itself **succeeded completely.** What failed is the one
object the X1b registration explicitly declined to change.

Harness `scripts/run_suica_m4_x1b_venue_response_fe.py`; tests
`tests/test_m4_x1b_venue_response_fe.py`; artifacts
`results/m4_x1b_venue_response_fe/` (gitignored); report
`reports/SUICA_M4_X1B_VENUE_RESPONSE_FE_REPORT.md`. Runtime about 45 s end to end,
entirely off X1's committed 13.6 MB cell cache — the comments file was not
re-streamed. SEED 20260819, B_perm 499, B_boot 1000.

### The design repair — the planner's census reproduced to the unit (#78)

The pinned chain (no fixed-point iteration) reproduces every blocking anchor:

| s | authors | communities | shared pairs | fill | med authors/comm | singleton communities | LCC |
|---|---|---|---|---|---|---|---|
| 3 | 3,686 | 1,145 | 32,415 | 0.0077 | 11.0 | 1 (0.0009) | 1.000 |
| **5 (PRIMARY)** | **3,665** | **1,000** | **31,899** | 0.0087 | **12.5** | **0** | **1.000** |
| 8 | 3,595 | 780 | 30,561 | 0.0109 | 16.0 | 0 | 1.000 |

The inherited anchors also hold: 17,640,062 parseable rows, 10,296 authors
(1,401 + 8,895), vocabulary floor 89 giving exactly 1,443 communities. The
published fill / median / singleton cross-checks agree at every floor.
Attrition at s = 5: 140,026 eligible cells → 37,414 shared pairs → 36,612
after the support floor → 31,899 after the k ≥ 3 floor → 31,899 in the LCC
(the graph was **already connected** before step 5, so the exactness argument
is valid, not merely assumed). In-leg arm censuses: s = 8 gives 3,595 authors;
n_min = 5 gives 5,094 authors / 1,243 communities / 56,460 pairs; the
word_count arm shares the primary design exactly; the Big5 replication gives
**408 authors**, above the in-leg #69 floor of 300, so it would have carried a
#73 comparison had a cell been reached.

### The estimator repair — exact, and demonstrated on the SAME null world

Alternating projections converge in 28 sweeps per half; the residual AUTHOR
means are ≤ 3.9e-11 and the residual COMMUNITY means ≤ 2.8e-17 on the realized
incomplete grid — the thing a single double-centering cannot do.

| skeleton | estimator | interaction share, null world (mean ± sd of 8) | R |
|---|---|---|---|
| X1's registered skeleton (4,342 × 5,369) | X1 double-centering | **0.0458 ± 0.0014** | 0.4464 |
| X1's registered skeleton | X1b exact FE | 0.0000 ± 0.0001 | 0.0017 |
| X1b skeleton (3,665 × 1,000) | X1 double-centering | 0.0183 ± 0.0014 | 0.2436 |
| **X1b skeleton** | **X1b exact FE** | **−0.0000 ± 0.0001** | **0.0000** |

The top-left cell reproduces X1's committed 0.0458 exactly, which validates
the comparison. Two things follow. (a) The leak was the PROJECTION, not the
grid: the FE reads zero even on X1's own sparse skeleton, while support
conditioning alone would only have taken double-centering from 0.0458 to
0.0183 — still nine times the TRACE boundary's half-width. (b) On the scored
replicate with full inference on both sides, the old estimator gives
0.0176 with CI [0.0241, 0.0286] (covering neither 0 nor its own point) and
R = 0.2442 against band [0.0636, 0.0965]; the new one gives −0.0001 with CI
[−0.0003, 0.0002] and R = 0.0078 against band [−0.0166, 0.0166]. Support
conditioning is still worth its place — it is what makes the mains estimable
and the graph connected — but the registration's estimator clause is what
purchased the zero.

### Part 0 — the nine clauses

| # | clause | value | status |
|---|---|---|---|
| 1 | recovery — author main | 0.3220 for 0.3000, tol 0.0179 | **FAIL** |
| 2 | recovery — community main | 0.1072 for 0.0800, tol 0.0424 | PASS |
| 3 | recovery — interaction | 0.0170 for 0.0200, tol 0.0100 | PASS |
| 4 | null world — share CI covers 0 | [−0.0003, 0.0002] | PASS |
| 5 | null world — share inside its band | −0.0001 in [−0.0011, 0.0010] | PASS |
| 6 | null world — R inside its band | 0.0078 in [−0.0166, 0.0166] | PASS |
| 7 | ablation — author-only leakage | 0.0001 < 0.0050 | PASS |
| 8 | ablation — community-only leakage | 0.0001 < 0.0050 | PASS |
| 9 | #85b bootstrap-zero — null CI covers 0 | yes, and it covers its own point | PASS |

The X1 pathologies are all gone: the CI now covers its own point estimate, the
permutation band now brackets the artifact instead of sitting below it, and
both ablations are three orders of magnitude inside their clause (author-only
0.0001 against X1's 0.0383; community-only 0.0001 against 0.0078).

### The one failing clause, and its mechanism

The author main share is estimated PRE-FE — from the cross-half covariance of
author half-means, X1's definition, which the X1b registration explicitly kept
("mains are estimated pre-FE; only the interaction object changes"). That
estimator's target is not Var(a). An author's half-mean is a_u PLUS that
author's own average of the community main and of the interaction over the
communities they occupy, and that composition average is the same in both
halves up to cell-size reweighting, so the cross-half covariance keeps it. The
size of the term is a design constant: E[1/k_u] = 0.1909 over the 3,665
eligible authors predicts a bias of 0.1909 × (0.08 + 0.02) = **0.0191**
against an observed **+0.0220**. The support floor raised authors-per-community
from X1's 2.0 to 12.5 and thereby fixed the community main (0.1072 for 0.0800
now sits inside its tolerance, against X1's 0.1351); it does not raise
communities-per-author, so it cannot touch the author main.

A second, smaller and opposite bias is now on the record: the two-way
projection removes A + C − 1 = 4,664 dimensions from 31,899 cells, so a white
planted interaction retains 0.8538 of itself. That predicts 0.0171 for a
planted 0.0200 against the observed 0.0170 — the interaction estimand is
CONSERVATIVE by a computable factor, and passes its clause with room to spare.

### Defect candidates recorded for the planner (nothing below was run)

1. **The main-share estimators are MARGINAL, not main.** Their target is
   Var(a) plus a composition term whose divisor is communities-per-author, so
   no support conditioning can move them. Candidates: read the mains off the
   SAME two-way projection that now carries the interaction (effect variances
   with their sampling variance removed), or re-state the clause so the mains
   are scored against their own marginal targets. Neither was run.
2. **Routing clauses and descriptive clauses are gated together.** The cells
   route on the interaction share and R, and both now behave exactly; the leg
   nevertheless stops on a co-reported descriptive component. A planner may
   wish to separate the two so a descriptive bias downgrades a number instead
   of stopping a leg — or may deliberately keep them together, since a budget
   that over-reads its largest component is not a budget.
3. **The FE's degrees of freedom are worth pinning.** Either correct the
   interaction share for the (A + C − 1)/M projection loss or register the
   estimand explicitly as a LOWER bound.

### Governance

Metadata only; `word_count_quoteless` and `word_count` are the sole
text-derived quantities; no body was read; `author_profiles.csv` was never
opened. The ID-leak scan over all 10,296 author names cleared with **0 NEW
hits** (4 pre-existing dictionary collisions carried unchanged from HEAD:
three in `CLAIMS_LEDGER.md`, one in this plan at the line where the X1
adjudication quotes the colliding word). The new boundary is named on every
claim: the analysis universe is now WELL-SHARED VENUES by construction.

## X1b planner adjudication (2026-08-19)

**Registered outcome ACCEPTED: `A1_STOP__SYNTHETIC_GATE_FAILED` — as
routed, 8 of 9 clauses passing. No cell; the corpus remains unread on
crossing #2.** But the situation is categorically different from X1's
stop, and the adjudication must say so precisely:

**The repair WORKED for everything that routes.** The exact-FE
estimator reads the null world at −0.0000 ± 0.0001 with R = 0.0000 —
and reads ZERO even on X1's own registered skeleton (0.0458 → 0.0000),
proving the leak was THE PROJECTION, NOT THE GRID (support conditioning
alone would only have bought 0.0458 → 0.0183). Recovery, both
ablations, both null clauses, and the new #85b bootstrap-zero clause
all pass. The verdict-routing machinery (interaction share + R) is now
CERTIFIED EXACT on the realized skeleton, twice over.

**The one failing clause is a DESCRIPTIVE object with a solved
mechanism.** The author-main share was registered PRE-FE (X1's
definition, deliberately kept): its estimand is MARGINAL, not main —
an author's half-mean carries the author's own composition average of
the community main and the interaction, persistent across halves by
construction. The bias is design-predicted: E[1/k_u]·(0.08+0.02) =
+0.0191 against +0.0220 observed; the mirror contamination shows in
the null world's community main (0.1139 vs 0.0800). No support floor
can move it (floors raise authors-per-community, not
communities-per-author). Also quantified: the FE projection removes
A+C−1 of P dimensions, making the interaction share conservative by
exactly P/(P−A−C+1)⁻¹ = 0.8538 (predicted 0.0171, observed 0.0170).

**Defect #86 (planner, purchased; registry thirty-third note).** Three
parts, from the executor's candidates, all adopted. (a) GATE CLAUSE
GRANULARITY: the gate bundled verdict-routing clauses with descriptive
clauses, so a descriptive bias with a solved mechanism stopped a leg
whose routing machinery is exact. Convention: gates separate ROUTING
clauses (A1-stopping) from DESCRIPTIVE clauses (failure → the
descriptive is reported WITH its dual-stamped bias annotation per #67,
never a stop). (b) ESTIMAND NAMING (#59/#81 family): "author main
share" was in fact the MARGINAL author share; budget components name
their estimand exactly, and marginal estimands are gate-scored against
MARGINAL targets (planted value + the design-composition term,
computable from the skeleton). (c) DF PINNING: a projection estimator's
dimension loss is pinned; the df-corrected share routes, the raw share
is co-reported (#67 dual stamp). Also noted: the #83 baseline is now 4
pre-existing collisions — the fourth introduced by the PLANNER's own
adjudication commit quoting the flagged word; the policy absorbed it
correctly, and the lesson (do not quote flagged usernames into
committed prose, even common-word ones) is recorded here without a
number.

**Executor discipline noted:** seeds deliberately NOT re-chosen after
the failure (gate-shopping refused a second time); the FE stopping rule
tightened so BOTH residual group means are ≤ 1e-10 on exit; the
bootstrap-FE weighted-projection equivalence proven by contract test.

---

## X1c — the venue response, clause-separated gate (registered BEFORE run, 2026-08-19)

Third and final registration of this estimand. Everything inherits
X1b (design chain at s = 5: 3,665 × 1,000, 31,899 shared pairs,
LCC 1.000; exact-FE estimator; arms; cells; leans; boundaries;
governance) except the following #86-compliant changes:

1. **Gate restructure (#86a).** ROUTING clauses (A1-stopping):
   df-corrected interaction-share recovery within max(0.01, 3·rep sd);
   null-world share CI covers 0 with point inside the permutation band;
   null-world R inside its band; both ablation leakages < 0.005; the
   #85b bootstrap-zero clause. DESCRIPTIVE clauses (report-gated):
   marginal author and community shares scored against their MARGINAL
   TARGETS — planted value plus the design-composition term, both
   derived from the skeleton with pinned formulas (the executor pins
   and prints the derivation); a descriptive failure annotates, never
   stops.
2. **Estimand naming and correction (#86b/c).** The verdict routes on
   the DF-CORRECTED reproducible interaction share:
   share_corr = share_raw · P/(P − A − C + 1) (numerics pinned from
   the realized skeleton). Raw co-reported (#67). Cell boundaries
   unchanged (0.02 / 0.10 on share_corr). R unchanged. The budget
   table reports the MARGINAL author/community shares under that name
   with their design-composition bias annotations, plus the
   FE-interaction rows.
3. Everything else — B_perm = 499, B_boot = 1000, SEED = 20260819,
   seeds inherited, standard six, ONE commit
   `feat(m4-x): X1c — the venue response, clause-separated gate —
   <VERDICT>`, suite green (1410 + new), #83 scan (baseline 4
   pre-existing) — verbatim from X1b.

## X1c outcome (executor, 2026-08-19)

**VERDICT — `RESPONSE_TRACE`.** Part 0's five ROUTING clauses all passed, the
two DESCRIPTIVE clauses passed as well, the A1 stop did not fire, and the real
arms ran. Crossing #2 has its first corpus reading: on the primary arm the
DF-CORRECTED reproducible author × community interaction carries **0.0190** of
comment-level Var(y), cluster-bootstrap CI **[0.0171, 0.0193]**, permutation
band [−0.0016, 0.0017]; raw **0.0162** [0.0146, 0.0165] under the #67 dual
stamp. **R = 0.2788** [0.2598, 0.2934] against a permutation band
[−0.0161, 0.0152]. The interaction is DETECTED and it is SMALL — the whole CI
sits below the 0.02 TRACE/IDIOSYNCRATIC boundary, so the cell is not a
straddle.

Harness `scripts/run_suica_m4_x1c_venue_response.py`; tests
`tests/test_m4_x1c_venue_response.py` (45); artifacts
`results/m4_x1c_venue_response/` (gitignored); report
`reports/SUICA_M4_X1C_VENUE_RESPONSE_REPORT.md`. Runtime about 121 s end to
end, entirely off X1's committed 13.6 MB cell cache — the 17.6M-row comments
file was not re-streamed for the third leg running. SEED 20260819, B_perm 499,
B_boot 1000, seeds inherited and not re-chosen.

### Anchors (#78) — all nine, exact

17,640,062 parseable rows; 10,296 authors = 1,401 Big5 + 8,895 disjoint;
vocabulary floor 89 giving exactly 1,443 law communities; the predicate chain
at s = 3 / 5 / 8 giving 3,686 / 1,145 / 32,415, **3,665 / 1,000 / 31,899**
and 3,595 / 780 / 30,561, with **0 singleton communities and LCC author
coverage 1.000** at the primary floor. Attrition at s = 5: 140,026 eligible
cells → 37,414 shared pairs → 36,612 → 31,899 → 31,899 (already connected
before the LCC step). The alternating projection converges in 31 sweeps per
half on the corpus arm with residual author means ≤ 4.9e-11 and residual
community means ≤ 6.3e-17.

### The gate, separated (#86a)

| family | clause | value | status |
|---|---|---|---|
| ROUTING | (i) recovery, DF-CORRECTED interaction | 0.0199 for 0.0200, gap −0.0001, tol 0.0100 | PASS |
| ROUTING | (ii) null world — share CI covers 0 AND point inside band | CI [−0.0004, 0.0002]; −0.0001 in [−0.0012, 0.0012] | PASS |
| ROUTING | (iii) null world — R inside band | 0.0078 in [−0.0166, 0.0166] | PASS |
| ROUTING | (iv) ablations — author-only and community-only leakage | 0.0001 and 0.0001, both < 0.0050 | PASS |
| ROUTING | (v) #85b bootstrap-zero | the null CI covers 0 and its own point | PASS |
| DESCRIPTIVE | marginal AUTHOR share vs its MARGINAL TARGET | 0.3220 for a target of 0.3267, gap −0.0047, tol 0.0179 | PASS |
| DESCRIPTIVE | marginal COMMUNITY share vs its MARGINAL TARGET | 0.1072 for a target of 0.1077, gap −0.0006, tol 0.0424 | PASS |

**The df correction, pinned.** P = 31,899, A = 3,665, C = 1,000, rank removed
A + C − 1 = 4,664, residual df 27,235, retained fraction 0.8538, factor
**1.1712502**. It converts a raw recovery of 0.0170 (gap −0.0030 against the
planted 0.0200) into 0.0199 (gap −0.0001). X1b's prediction of the projection
loss is confirmed to the fourth decimal by an estimator that now corrects for
it.

**The marginal targets, derived and printed.** The estimator of a main share
is a cross-half covariance of size-weighted half-means, so it targets the
planted component PLUS the unit's own composition average of everything else
that persists. With `w[u,c,h] = n[u,c,h] / Σ_c' n[u,c',h]`,
`t[u,c,h] = n[u,c,h] / Σ_u' n[u',c,h]`, `κ_u = Σ_c w[u,c,e] w[u,c,l]`,
`λ_c = Σ_u t[u,c,e] t[u,c,l]` and `Ŵ_c` the normalized `n[c,e] + n[c,l]` size
weights the budget already uses:

```
T_a = v_a (1 − 1/A) + (v_c + v_g)·mean_u[κ_u]
      − v_c·Σ_c w̄[c,e] w̄[c,l] − v_g·mean_u[κ_u]/A
T_c = v_c (1 − Σ_c Ŵ_c²) + v_a·(Σ_c Ŵ_c λ_c − Σ_u φ[u,e] φ[u,l])
      + v_g·(Σ_c Ŵ_c λ_c − Σ_c Ŵ_c² λ_c)
```

On the realized skeleton mean_u[κ_u] = **0.2905**, which is **52% larger** than
the 1/k_u heuristic X1b used (0.1909) — the gap is entirely cell-size
inequality, and it is why X1b's first-order prediction (+0.0191) undershot the
observed bias (+0.0220) while the exact term (+0.0290 composition − 0.0024
mean-removal = +0.0267) brackets it. Author target 0.3267, community target
0.1077 (size-weighted mean_c[λ_c] = 0.0937). Scored against the PLANTED
components instead — X1b's clause — the author gap is +0.0220 against a
tolerance of 0.0179 and that clause still FAILS, exactly as before: #86a is
what let the leg through, and the estimator was not quietly changed to make it
pass. The NULL world scores its own targets as a free cross-check and passes
both (author 0.3220 for 0.3209; community 0.1139 for 0.1059). The derivation
was verified against simulation on both a synthetic design (contract test) and
the realized skeleton (48 replicates: author 0.3262 ± 0.0008 se against a
target of 0.3267, community 0.1084 ± 0.0020 against 0.1077).

### The corpus reading

PRIMARY arm: 3,665 disjoint authors × 1,000 law communities, 31,899 shared
pairs, 63,798 eligible cells, 6,811,576 comments, comment-level
Var(y) = 1.2423.

| quantity | value | CI | band |
|---|---|---|---|
| interaction share, DF-CORRECTED (routes) | **0.0190** | [0.0171, 0.0193] | [−0.0016, 0.0017] |
| interaction share, raw (#67) | 0.0162 | [0.0146, 0.0165] | [−0.0014, 0.0015] |
| R | **0.2788** | [0.2598, 0.2934] | [−0.0161, 0.0152] |
| MARGINAL author share | 0.1804 | [0.1729, 0.1882] | — |
| MARGINAL community share | 0.0928 | [0.0898, 0.1075] | — |
| residual | 0.7106 | [0.6931, 0.7180] | — |

Both of cell 1's conditions are refused: R is far outside its band and the
share's CI excludes 0. The mains are reported under their true name — this
arm's composition weights are κ̄ = 0.2905 and λ̄ = 0.0937, so inverting the
2 × 2 mixing as an ANNOTATION ONLY (not a registered estimand, nothing routes
on it) implies an author main near 0.1526 and a community main near 0.0767.

| arm | corrected share [CI] | R | cell | #73 |
|---|---|---|---|---|
| PRIMARY (n=10, s=5, wcq) | 0.0190 [0.0171, 0.0193] | 0.2788 | RESPONSE_TRACE | — |
| s = 8 | 0.0189 [0.0172, 0.0193] | 0.2738 | RESPONSE_TRACE | none |
| n_min = 5 | 0.0208 [0.0192, 0.0214] | 0.2192 | IDIOSYNCRATIC_RESPONSE (STRADDLE) | **#73** |
| y = word_count | 0.0188 [0.0170, 0.0191] | 0.2777 | RESPONSE_TRACE | none |
| Big5 replication (408 authors) | 0.0177 [0.0131, 0.0189] | 0.2823 | RESPONSE_TRACE | none |

The five arms span 0.0031 of total variance. The single #73 is a BOUNDARY
CROSSING: the n_min = 5 arm's CI [0.0192, 0.0214] touches both cells and is
reported as a straddle, with its point at 0.0208 placing it in
IDIOSYNCRATIC_RESPONSE. The Big5 replication carries 408 authors, above the
in-leg #69 floor of 300, and lands in the primary cell — the replication lean
is scored and HELD rather than waived.

**All five registered leans HELD** (six rows, because the share lean is
reported on both scales): R = 0.2788 in (0.05, 0.30]; the interaction share in
(0.005, 0.05] on the RAW scale the lean was written for (0.0162) and also on
the corrected scale the verdict routes on (0.0190); the marginal author share
0.1804 in [0.15, 0.45]; the marginal community share 0.0928 in [0.02, 0.15];
the Big5 replication in the primary cell. The registration's own reasoning —
"personal baseline × venue norm is additive in log and lands in the mains, so
the interaction asks for more" — is vindicated: the interaction is real and it
is second-order.

**Headroom, about the MEAN (#84 as restated by #85).** 3,665 authors scored,
0 undefined; realized per-author correlation mean 0.2788, sd 0.5021, median
0.3501, 2.76% above 0.99, 9.55% above 0.90, 73.64% positive, extremes at
±1.0000. Per-author saturation is acknowledged and is NOT a ceiling: at k = 3
a three-point correlation reaches ±1 whenever three points line up. The
bounded object is the mean, and it separates cleanly from Part 0's synthetic
null world (0.2788 against 0.0078).

### Honest anomalies

1. **The cluster-bootstrap interval is not symmetric about its point**; every
   arm's point sits near the upper edge (primary: point 0.0162 raw against a
   bootstrap mean of 0.0155). The mechanism is the resample's sparsity — a
   bootstrap author multiset holds about 63% of the distinct authors, so each
   replicate's projection removes a larger fraction of its own design than the
   full design's, while the pinned factor stays at the realized skeleton's
   value (which is what the registration fixes). The direction is CALIBRATED,
   not argued: Part 0's planted world has a known truth of 0.0200 and shows
   the same asymmetry — point 0.0204, interval [0.0188, 0.0206] — and still
   covers it. The corpus intervals are therefore conservative on the low side
   only, and no cell assignment is at risk.
2. **The estimand lives on the cell boundary.** Four arms sit at 0.0177–0.0190
   and one at 0.0208, against a registered boundary of 0.0200. The cells did
   their job, but the leg's headline is a few thousandths of total variance
   from a different label.
3. Non-gating, inherited from X1: the two census descriptives that did not
   reproduce in X1 (within-cell sd; median shared communities) were not
   re-examined here.

### Deviation

X1b's X1-versus-X1b estimator-comparison grid was NOT re-run. It was a
one-time demonstration of the repair, already adjudicated, and re-running it
would add nothing to a leg whose purpose is the corpus reading; X1b's
artifact holds it. Nothing else in the registration was skipped, and nothing
was added to the routing path.

### Defect candidates recorded for the planner (nothing below was run)

1. **The df factor is pinned at the realized design and does not follow the
   bootstrap.** The registration is explicit that P/A/C come from the realized
   skeleton, and that is what was done; the consequence is anomaly 1. A future
   registration can either correct each replicate by its OWN residual-df
   fraction, or declare the interval one-sidedly conservative — which is what
   this report does, with the planted world as its calibration.
2. **The mains are now correctly NAMED but still not ESTIMATED.** #86b gave
   the marginal estimators exact targets; it did not give the leg an estimator
   of Var(a) or Var(b). The 2 × 2 inversion printed here is arithmetic on the
   design, not an estimator with a null or a CI. Any leg that wants the mains
   as objects — X3's trait join is the obvious one — needs them read off the
   same two-way projection that now carries the interaction, with their
   sampling variance removed, and registered with their own gate.
3. **Cell boundaries drawn before the estimand's scale was known.** The
   0.02 / 0.10 boundaries were set in X1 on literature priors; the corpus
   answer landed within 0.001 of the first one, and one arm crossed it. A
   planner may wish to register a boundary REGION (report both cells when the
   CI touches both) rather than a point, so that "straddle" is a first-class
   outcome and not an annotation on a cell.

### Governance

Metadata only; `word_count_quoteless` and `word_count` are the sole
text-derived quantities; no body was read; `author_profiles.csv` was never
opened. The ID-leak scan over all 10,296 author names cleared with **0 NEW
hits** (4 pre-existing dictionary collisions carried unchanged from HEAD:
three in `CLAIMS_LEDGER.md`, one in this plan). Every boundary is carried
forward, plus the one X1c adds: the routing statistic is DF-CORRECTED and the
two main shares are MARGINAL, named and annotated as such on every table.

## X1c planner adjudication + the X1 arc CLOSES (2026-08-19)

**Verdict ACCEPTED: `RESPONSE_TRACE`, with the boundary-region
annotation adopted as part of the verdict.** Routing 5/5 (df-corrected
recovery at gap −0.0001), descriptive 2/2 against the exact marginal
targets — and the executor's integrity demonstration is noted with
approval: scored against X1b's planted-component clause the author gap
STILL fails (+0.0220), so the gate structure, not a quietly changed
estimator, is what let the leg through. Both readings printed side by
side. The marginal-target derivation went past the planner's
first-order heuristic (κ̄ = 0.2905 vs 1/k̄ = 0.1909 — cell-size
inequality explains X1b's undershoot) and was verified by simulation
twice.

**Crossing #2's first corpus reading, formalized.** Expression volume
is venue-conditioned in three layers on this corpus: the personal
baseline (marginal share 0.1804; 2×2-inverted true main ≈ 0.153), the
venue norm (0.0928; ≈ 0.077), and a SMALL, REAL, PERSISTENT
idiosyncratic response: **df-corrected reproducible interaction share
0.0190 [0.0171, 0.0193] against a ±0.0017 band; profile persistence
R = 0.2788 [0.2598, 0.2934] against ±0.016; 73.6% of authors
positive; REPRODUCED on the Big5 cohort (0.0177, same cell, lean
held).** The response operator's first projection is NON-EMPTY — the
first crossing where the personal second-order channel is present
rather than silent. All five registered leans HELD, on both the raw
scale the lean was written for and the corrected scale the verdict
routes on. The registration's reasoning stands: multiplicative
baseline × norm lands in the mains; the interaction asked for more,
and the corpus has more — about 1.9% of variance worth.

**The honest fragility, carried into the verdict.** The estimand lives
ON the registered boundary: arms span 0.0177–0.0208 against the 0.0200
line set from literature priors; one #73 straddle (n_min = 5 at 0.0208
[0.0192, 0.0214]). The DETECTION is unambiguous (CI 10× outside the
band); the LABEL is boundary-sensitive. Licensed sentence: "a real,
persistent idiosyncratic venue response at the TRACE/IDIOSYNCRATIC
boundary (≈ 2% of variance)" — never the bare cell name without the
boundary note.

**Defect #87 (planner, purchased; registry thirty-fourth note).** Two
parts. (a) Cell boundaries were registered as POINTS from literature
priors with no boundary-region provision, and the corpus landed within
0.001 of one — "straddle" had to work as an annotation where it
deserves first-class status. Convention: every effect-size
classification registers a BOUNDARY REGION (default: ± the projected
CI half-width) inside which the verdict IS the region statement
("AT the boundary"), first-class, not an annotation. (b) The pinned df
factor is not recomputed per bootstrap replicate, making the interval
one-sidedly conservative — the executor calibrated this against the
planted world (truth covered; low side only) rather than arguing it.
Convention: projection-dependent corrections are recomputed inside
every resample, or the one-sided conservatism is declared WITH a
planted-world calibration. Also adopted from the executor's
candidates: any leg wanting the MAINS as objects (X3's trait join)
must first build main ESTIMATORS (the marginals are now correctly
named but still not estimators of Var(a)/Var(b)) under their own gate.

**The X1 arc closes: X1 (A1 stop, projection leak caught) → X1b
(A1 stop, routing machinery certified, descriptive mechanism solved) →
X1c (reading).** Three registrations, two stops, zero real-data
contamination, one boundary-honest reading — the gate discipline's
full showcase, with defects #85–#87 as its tuition. The X-line
continues: X2 (What × When paths of expression volume) and X3 (the
stage-E trait join, now with a validated coordinate and the
main-estimator prerequisite) are named, not queued.

---

## X2 — the path of expression volume (registered BEFORE run, 2026-08-19)

Crossing #3 (What × When): the first real-data projection of expression
STATE TRAJECTORIES — does expression volume have reproducible path
structure in event time, and is the path parameter an AUTHOR attribute?
Metadata-only, as the line charter binds.

### 4W header

- **Object.** The lag-1 event-time autocorrelation of y = log1p(wcq)
  within an author-half — one static projection of the What × When
  trajectory kernel; the eq-12 projection caution binds (a flat r1 does
  not falsify dynamics; it is a statement about this projection).
- **Fixed.** y pinned (#70); per-author stable order by created_utc
  (ties keep stream order); halves by full-stream median (≤ early);
  adjacency = consecutive events WITHIN a half (never across the
  boundary); r1 per (author, half) requires sd > 0 (censused: zero
  degenerate halves).
- **Varied.** Raw adjacency vs cross-thread-only (link_id differs) vs
  venue-residualized (y minus the global community-half mean b̄_{c,h}
  computed over ALL events in that community-half); cohort (disjoint
  primary, Big5 replication); pool floor 50 vs 100 events/half.
- **What falsifies.** Ownership ρ_own CI including 0 → the path
  parameter is not an author attribute at this power (PATH_NOT_OWNED).
- **Layer.** R.

### Census (planner arithmetic #43; #77/#78 exact; #57 — NO estimand
value was evaluated: r1 was never computed; degenerate halves counted
by sd = 0 alone)

| quantity (exact predicate) | disjoint | Big5 |
|---|---|---|
| pool: ≥ 50 events in EACH half (all events) | **8,008** | **1,116** |
| cross-thread share of within-half adjacencies | 0.73159 | 0.62054 |
| degenerate halves (sd = 0) | 0 | 0 |
| median adjacencies per half | 348.0 | 491.75 |
| ownership width projection (analytic 1/√N) | ±0.0112 | ±0.0299 |

Inherited anchors: 17,640,062 parseable rows; 10,296 authors
(1,401 + 8,895). Per-half r1 sampling noise (analytic): ≈ 1/√348 ≈
0.054 — named as the attenuation floor on ρ_own (the ownership
correlation reads the attenuated per-half parameter; no disattenuation
is registered — raw ρ_own routes).

### Estimands (own nulls #68/#66; transforms pinned #70)

1. **Path presence (co-reports):** mean over (author, half) of r1,
   against the MARGINAL-PRESERVING within-half permutation null
   (each half's y sequence permuted; the per-half marginal is EXACTLY
   invariant — contract test, the U1 exact-bag pattern on a scalar);
   B = 499; band = 2.5/97.5 percentiles.
2. **Ownership ρ_own (ROUTES):** Pearson over pool authors of
   (r1_early, r1_late). Own null: author pairing permuted between
   halves, B = 499. Cluster bootstrap over authors B = 1000 for the CI.
3. Arms as above; each arm carries its own presence + ownership pair.

### Gate (Part 0, #85/#86-compliant, on the REALIZED skeleton — real
per-author sequence lengths and half splits, wholly synthetic y)

ROUTING clauses (A1-stopping): (i) ownership recovery on a planted
world — per-author AR(1) with author-owned φ_u at a #76-declared
operating point (across-author ownership ρ_true = 0.5; 8 replicates;
tol max(0.02, 3·rep sd)); (ii) the COMMON-PATH null world (every author
the same φ̄ > 0: presence WITHOUT ownership — ρ_own CI must cover 0,
point inside band; the design's decisive honesty check); (iii) the iid
world (no path at all: mean r1 inside band AND ρ_own inside band);
(iv) marginal-preservation contract (bit-exact per-half marginals under
the shuffle); (v) #85b bootstrap-zero on the common-path world.
DESCRIPTIVE clauses (annotate, never stop): mean-r1 recovery against
the planted φ̄ mapping on both non-null worlds.

### Cells (NULL-first #55; effect-size keyed #75; #87 boundary REGIONS
declared: half-width = the projected CI half-width, 0.011 primary)

On ρ_own (primary arm): 1 **PATH_NOT_OWNED** — CI includes 0 or point
below 0.15 − 0.011; **AT_BOUNDARY(0.15)** — point inside
[0.139, 0.161] (first-class verdict); 2 **WEAKLY_OWNED** —
(0.161, 0.489); **AT_BOUNDARY(0.50)** — [0.489, 0.511]; 3
**STRONGLY_OWNED** — above 0.511 (the U3 reliability-gate level:
trait-join eligible). Presence co-reports with its own band. Arm
divergences carry #73 flags; the primary routes.

### Registered leans

- Presence: mean r1 positive and clearly detected (session structure
  alone should produce it) — magnitude deliberately un-leaned (#57:
  no pilot was taken).
- Ownership: ρ_own ∈ (0.30, 0.60] — WEAKLY_OWNED toward the upper
  boundary; rationale: the U3 slow-coordinate reliabilities ran
  0.77–0.94, but r1's per-half sampling noise (≈ 0.054) attenuates,
  and rhythm is plausibly less stable than level.
- Cross-thread arm retains most of the ownership (rhythm is not just
  reply mechanics) — held moderately; the U1 precedent (43.8%
  mechanics) prices the alternative.
- Venue-residualized arm retains most (rhythm is not venue-routing).
- Big5 replication: same cell (#73 if divergent).

### Deliverables and discipline

Standard six: `scripts/run_suica_m4_x2_volume_path.py` + tests
(marginal-preservation bit-exactness; AR-world construction and the
common-path null's honesty on a toy; ownership/pairing-permutation
correctness; anchors; #83 helper); gitignored
`results/m4_x2_volume_path/` (fresh event cache: author, created_utc,
link hash, y — anchors above); report
`reports/SUICA_M4_X2_VOLUME_PATH_REPORT.md` (rule-24; gate table with
ROUTING/DESCRIPTIVE sections; presence + ownership per arm with bands
and CIs; boundaries: metadata-only, expression VOLUME not content, the
projection caution, no psychological naming, cohort caveat); outcome
appended here; one CLAIMS_LEDGER row; ONE commit
`feat(m4-x): X2 — the path of expression volume — <VERDICT>`; suite
green (1455 + new); #83 scan (baseline 4 pre-existing).
SEED = 20260819; B_perm = 499; B_boot = 1000.

---

## X2 outcome (executor, 2026-08-19)

**`WEAKLY_OWNED`.** All five ROUTING clauses of Part 0 passed and both
DESCRIPTIVE clauses annotated cleanly, so the A1 stop did not fire and
crossing #3 has its first corpus number. Runtime about 640 s (one stream pass
over the 17,640,062-row comments file, then everything vectorised off a
fresh gitignored event cache).

### The census reproduced the planner's table to the unit (#78)

| registered predicate | registered | observed |
|---|---|---|
| rows parseable | 17,640,062 | 17,640,062 |
| authors (Big5 + disjoint) | 10,296 (1,401 + 8,895) | 10,296 (1,401 + 8,895) |
| pool ≥ 50 events in EACH half, disjoint | 8,008 | 8,008 |
| pool ≥ 50 events in EACH half, Big5 | 1,116 | 1,116 |
| cross-thread share of within-half adjacencies, disjoint | 0.73159 | 0.73159 |
| cross-thread share of within-half adjacencies, Big5 | 0.62054 | 0.62054 |
| degenerate halves (sd = 0) | 0 / 0 | 0 / 0 |
| median adjacencies per half | 348.0 / 491.75 | 348.0 / 491.75 |

Definition pinned by the anchor: "median adjacencies per half" is the
MEDIAN OVER AUTHORS of the MEAN of that author's two half adjacency
counts. The Big5 quarter-fraction 491.75 is reachable no other way, and
that is what fixed the reading. In-leg census: the floor-100
sensitivity pool is **6,863** disjoint authors; the primary arm carries
16,016 cells, 14,586,457 events and 14,570,441 within-half adjacencies;
the cross-thread arm keeps 10,659,581 of them. Zero cells were
undefined and zero authors were dropped on any arm.

### Part 0 — the gate on the realized skeleton (wholly synthetic y)

| # | ROUTING clause | observed | status |
|---|---|---|---|
| i | ownership recovery, author-owned AR(1) at planted ρ_own = 0.50 | 0.4977 over 8 replicates (sd 0.0119), gap −0.0023, tolerance 0.0357 | **PASS** |
| ii | COMMON-PATH null world: presence WITHOUT ownership | ρ_own +0.0085, CI [−0.0197, 0.0384], band [−0.0203, 0.0217] | **PASS** |
| iii | iid world | mean r1 −0.004565 in [−0.005659, −0.003678]; ρ_own +0.0131 in [−0.0225, 0.0234] | **PASS** |
| iv | marginal preservation, BIT-EXACT | 16,016 cells; multisets bit-exact; the index array is a permutation; no index leaves its own cell | **PASS** |
| v | #85b bootstrap-zero on the common-path world | CI [−0.0197, 0.0384] covers 0 and covers its own point | **PASS** |

| # | DESCRIPTIVE clause | observed |
|---|---|---|
| D1 | mean-r1 recovery, AR(1) world | +0.1927 observed vs +0.1923 predicted (Marriott–Pope) |
| D2 | mean-r1 recovery, common-path world | +0.1918 observed vs +0.1925 predicted |

The **ownership mapping** was derived, printed and solved rather than
tuned. With g(φ) ≈ φ and Bartlett's Var(r1) = (1 − φ²)/n, the per-half
errors carry no cross-half covariance, so ρ_own = V / √((V+A_e)(V+A_l))
with A_h = (1 − φ̄² − V)·E_u[1/(n_{u,h} − 1)]. On the realized skeleton
m_e = 0.00465751, m_l = 0.00467906, and the bisection solution at
φ̄ = 0.20, ρ = 0.50 is V = 0.00446072, i.e. **sd(φ_u) = 0.066789** —
the readable special case V = A (plant exactly as much across-author φ
variance as the estimator's own sampling variance). The COMMON-PATH
world is the design's decisive honesty check and it behaved: presence
0.1918 — as strong as the owned world's 0.1927 — with ρ_own +0.0085 and
a CI covering zero. A world with a path but no owner was **not** called
owned.

### The corpus reading

| arm | authors | adjacencies | presence mean r1 | band | ρ_own | CI | pairing band | cell |
|---|---|---|---|---|---|---|---|---|
| **PRIMARY** raw adjacency, disjoint | 8,008 | 14,570,441 | **+0.1111** | [−0.005635, −0.003648] | **0.2589** | [0.2273, 0.2921] | [−0.0228, 0.0213] | `WEAKLY_OWNED` |
| cross-thread only | 8,008 | 10,659,581 | +0.0545 | [−0.006454, −0.003859] | 0.1062 | [0.0713, 0.1394] | [−0.0200, 0.0209] | `PATH_NOT_OWNED` **#73** + **#87 straddle** |
| venue-residualized | 8,008 | 14,570,441 | +0.0917 | [−0.005820, −0.003643] | 0.2204 | [0.1864, 0.2534] | [−0.0225, 0.0213] | `WEAKLY_OWNED` |
| Big5 replication | 1,116 | 2,989,174 | +0.1773 | [−0.006953, −0.001800] | 0.6367 | [0.5894, 0.6785] | [−0.0572, 0.0549] | `STRONGLY_OWNED` **#73** |
| floor-100 sensitivity | 6,863 | 14,402,480 | +0.1150 | [−0.003999, −0.002226] | 0.3341 | [0.2977, 0.3653] | [−0.0246, 0.0251] | `WEAKLY_OWNED` |

**PRESENCE is unambiguous on every arm** — the primary's mean r1 sits
about 58 band-widths above the upper edge of its own
marginal-preserving band. **OWNERSHIP routes at 0.2589**, a CI that
clears the 0.161 region edge by 0.066 and the 0.489 edge by 0.197: the
primary cell is `WEAKLY_OWNED` and it is **not** a straddle. Retention
ratios: venue-residualized **0.8516**, cross-thread **0.4104**.

### Leans: two held, three broke

| lean | registered | observed | status |
|---|---|---|---|
| presence positive and detected (magnitude un-leaned, #57) | > 0 and outside the band | +0.1111 vs [−0.005635, −0.003648] | **HELD** |
| ρ_own ∈ (0.30, 0.60] | (0.30, 0.60] | 0.2589 | **BROKEN** (low) |
| cross-thread retains most | ≥ 0.50 | 0.4104 | **BROKEN** |
| venue-residualized retains most | ≥ 0.50 | 0.8516 | **HELD** |
| Big5 same cell | `WEAKLY_OWNED` | `STRONGLY_OWNED` | **BROKEN** (high) |

The ownership lean broke LOW and the Big5 lean broke HIGH, which is one
fact seen twice — see the attenuation arithmetic below.

### Attenuation arithmetic (ANNOTATION; nothing routes on it)

The registration named the attenuation and refused to correct for it.
Inverting the SAME first-order model Part 0 plants — A from each arm's
own pair counts and its own mean r1, then V from its own ρ_own — reads:

| arm | median pairs/cell | A | implied Var(φ) | implied sd(φ) |
|---|---|---|---|---|
| primary | 348.0 | 0.004600 | 0.001610 | 0.0401 |
| cross-thread | 266.0 | 0.006169 | 0.000740 | 0.0272 |
| venue-residualized | 348.0 | 0.004618 | 0.001309 | 0.0362 |
| Big5 | 492.0 | 0.004154 | 0.007303 | 0.0855 |
| floor-100 | 447.0 | 0.003034 | 0.001524 | 0.0390 |

Three readings follow, and they are the leg's real content beyond the
cell. (1) The **floor-100 arm is the primary at a better precision**:
0.3341 against 0.2589, but the implied Var(φ) agrees to 5%
(0.001524 vs 0.001610). Two ρ, one object. (2) The **Big5 divergence is
NOT attenuation**: its A is if anything smaller, and its implied Var(φ)
is **4.5×** the primary's, with the higher presence to match
(0.1773 vs 0.1111). The label-carrying cohort genuinely has more
across-author dispersion in this path parameter than the disjoint
cohort does. (3) The **cross-thread loss is not only arithmetic**: its
implied Var(φ) is 46% of the primary's on top of a larger A, so a real
part of the owned rhythm is reply-chain mechanics — U1's 43.8%
precedent priced this and the lean paid.

### Honest anomalies

- The presence permutation band sits **entirely below zero**. A uniform
  permutation of a finite sequence has a small negative expected lag-1
  correlation; the primary's null mean is −0.004680 against the
  analytic offset −E[1/pairs] = −0.004668. It is a bias-offset band,
  not a zero band. Detection is untouched.
- The Big5 arm reads **above the trait-join-eligible level** (0.6367,
  CI entirely above 0.511) while the primary does not. Whatever X3
  intends to join, it will join it on the cohort that reads
  `STRONGLY_OWNED`, and the two cohorts do not agree.
- The cross-thread cell is a **#87 straddle**: its CI crosses 0.139, so
  `PATH_NOT_OWNED` there is boundary-sensitive and should be read as
  "at or below the low boundary", not as a clean null.
- The cluster-bootstrap interval is asymmetric about its point on the
  null world (+0.0085 inside [−0.0197, 0.0384]) — X1c's asymmetry,
  unchanged, conservative on the side that matters, and far from every
  region edge on the primary.

### Boundaries carried

Metadata only, expression VOLUME and not content (permanent — no body
was read); the What × When projection caution (r1 is ONE static
projection of the trajectory kernel, so 0.2589 is a lower bound on the
kernel's ownership and a flat r1 would not have falsified dynamics); no
psychological naming; EXPLORATORY corpus-level, no person claims; the
disjoint cohort is TYPOLOGY-ENRICHED by construction; the attenuation
note (raw ρ_own routes, no disattenuation); and NEW — the pool floors
at 50 events in EACH half, so the leg speaks only for authors active on
both sides of their own median.

### Deviations and defect candidates (none run)

- **Deviation (recorded):** the leg builds its OWN event cache rather
  than reusing X1's. X1's cache aggregates over (author, community,
  half) and carries neither event order nor `link_id`, so it cannot
  serve a path estimand. One stream pass, five metadata columns, no
  bodies.
- **Deviation (recorded):** two unregistered ANNOTATIONS are printed —
  a cluster-bootstrap interval on the presence mean, and the
  attenuation arithmetic above. Both are labelled in the report; the
  verdict touches neither.
- **Defect candidate (#87 follow-on):** the registered region
  half-width 0.011 was the ANALYTIC 1/√N projection, and the realized
  cluster-bootstrap half-widths are 0.032 (primary) and 0.045 (Big5) —
  three to four times wider. A region set from the analytic width
  under-covers the interval it is meant to price; future legs should
  set the region from the REALIZED interval or say why not.
- **Defect candidate:** presence and ownership are reported per arm,
  but the arms' pool sizes differ (8,008 / 6,863 / 1,116), so the
  cross-arm comparison is partly a precision comparison. A leg that
  wants arms comparable should match precision, not just floor it.
- **Defect candidate:** the lag-1 projection is one coordinate. The
  Big5/disjoint dispersion gap (4.5×) is the kind of thing that could
  be a genuine cohort difference or a selection artifact of how the
  1,401 were sampled; neither this leg nor X1 can separate them.

## X2 planner adjudication (2026-08-19)

**Verdict ACCEPTED: `WEAKLY_OWNED`** — ρ_own = 0.2589 [0.2273, 0.2921],
clean of both boundary regions (clears the 0.161 edge by 0.066); the
gate passed FIRST TRY, all five routing clauses, with the decisive
common-path clause doing exactly its job (presence 0.1918 in the
unowned world, as strong as the owned world's 0.1927, and the gate
refused to call it owned). The ownership mapping was derived and
bisection-solved, not tuned — and the executor's Marriott–Pope
small-sample handling of the r1 bias (the presence null band sits at
the analytic −E[1/pairs], an offset band, not a zero band) is exactly
the standard the line now runs at. The X1-arc tuition (#85–#87) bought
X2 a clean first gate.

**Crossing #3's first reading, formalized.** Expression volume has REAL
path structure in event time (presence +0.1111, ~58 band-widths out),
and the rhythm parameter is an author attribute — weakly, on the
disjoint cohort. Retention decomposition: **85.2% of the ownership
survives venue-residualization** (the rhythm is intrinsic, not venue
routing) and **41.0% survives the cross-thread restriction** — with
implied Var(φ) at 46%, a real part of the owned rhythm IS reply-chain
mechanics (U1's 43.8% precedent priced it; the lean broke exactly
there and pays the same lesson twice: fast-time channels are
mechanics-heavy). Lean record: 2 held, 3 broke (ρ_own low of the lean;
cross-thread retention; Big5 cell) — honest, and each break carries
information.

**The flag for X3 (the leg's most consequential row).** The Big5
cohort reads ρ_own = 0.6367 [0.5894, 0.6785] — STRONGLY_OWNED, above
the trait-join-eligible 0.511 line — while the primary reads 0.2589,
and the attenuation arithmetic shows this is NOT precision: implied
Var(φ) is 4.5× the disjoint cohort's, with higher presence too. The
label-carrying cohort genuinely differs on this object. Carried into
X3 with the standing caveat the executor named: neither X1 nor X2 can
separate a genuine cohort difference from a selection artifact of how
the 1,401 were chosen (test-takers may be atypical posters); every X3
claim inherits this caveat verbatim.

**Defect #88 (planner, purchased; registry thirty-fifth note).** Two
parts. (a) The #87 boundary-region half-width was priced from the
ANALYTIC null sd (1/√N = 0.011) while the realized bootstrap
half-widths run 0.032–0.045 — the region under-covers the interval it
prices; the convention is corrected: region widths are priced from the
PROJECTED BOOTSTRAP width of the non-null estimate (#79b machinery),
never from the null sd. (b) Arms with different pool sizes make
cross-arm comparisons partly precision comparisons — the W2
size-matching lesson extends WITHIN legs: cross-arm ownership
comparisons use the implied-variance scale (as the executor's
annotation did) or size-matched subpools. The two unregistered
annotations (presence bootstrap; attenuation arithmetic) are accepted
as labelled.

**Program milestone: every 4W crossing now has a real-data reading.**
Crossing #1 (T/W lines), #2 (X1c), #3 (this leg), #4 (U-line). The 4W
empirical-state document receives its completion addendum with this
commit.

---

## X4 — the three-level decomposition and the ergodicity contrast (registered BEFORE run, 2026-08-19)

**Provenance: the OWNER's conference input, relayed 2026-08-19** — the
three-designs schema for personality measurement: (1) one-shot group
measurement yields BETWEEN-person relations; (2) repeated measurement of
many yields the AVERAGE WITHIN-person relation; (3) repeated measurement
of one yields THAT PERSON's relation. X1c/X2 already instantiated levels
2–3 for single-variable structures (venue norms vs idiosyncratic
response; presence vs ownership); X4 measures ONE two-variable relation
at ALL THREE levels and registers the level comparison — the ergodicity
contrast — as the verdict. The relation: **event commonness × expression
volume**, metadata-only.

### 4W header

- **Object.** The Where × What relation "how common a venue is × how
  much one writes there," decomposed into the owner's three levels;
  verdict = whether level 1 (between-person) equals level 2 (average
  within-person). Non-ergodicity here is the two-variable form of the
  program's dissociation family (levels do not reduce to one another).
- **Fixed.** x = log10(community share of all 17,640,062 parseable
  events) — a GLOBAL fixed community attribute (46,214 communities,
  range [−7.25, −1.14]); y = log1p(wcq) (#70); X2's universe, order,
  halves; pool = X2's ≥ 50 events/half AND within-author sd(x) > 0 in
  each half.
- **Varied.** Cohort (disjoint primary; Big5 replication, #73 — X2's
  cohort-difference flag rides along); floor-100 sensitivity.
- **What falsifies.** Δ_erg CI including 0 → the levels are
  indistinguishable at this power (scoped by realized width).
- **Layer.** R.

### Census (planner arithmetic #43; #77/#78 exact; #57 — NO estimand
value evaluated: no x–y slope, correlation, or covariance was computed)

| quantity (exact predicate) | disjoint | Big5 |
|---|---|---|
| pool (X2 pool ∧ sd(x) > 0 both halves) | **8,004** | **1,112** |
| within-author sd(x), q25/med/q75 | 0.805 / 0.965 / 1.111 | 0.660 / 0.856 / 1.013 |
| distinct communities per author, median | 64 | 55 |
| top-1 community share, med / q90 | 0.259 / 0.577 | 0.329 / 0.743 |

### Estimands (own nulls #68/#66; #79 ratio-free; transforms pinned)

1. **Level 1, β_between:** OLS slope of person-mean ȳ_u on person-mean
   x̄_u across pool authors (correlation co-reported). Author-cluster
   bootstrap B = 1000.
2. **Level 2, β_within:** the within-centered pooled slope per half
   (y − ȳ_{u,h} on x − x̄_{u,h}, over events), averaged over the two
   halves. Same bootstrap.
3. **Level 3:** per-(author, half) slopes β_{u,h}; ownership
   ρ_own(β) = Pearson over authors of (β_early, β_late) with the
   pairing-permutation null (X2 machinery, B = 499); true dispersion
   Var(β) by cross-half covariance; headroom about the mean.
4. **The ergodicity contrast (ROUTES): Δ_erg = β_between − β_within**,
   paired author-cluster bootstrap CI (both slopes recomputed per
   resample).

### Gate (Part 0, realized skeleton — real x-sequences and halves,
wholly synthetic y; #85/#86 clause separation; A1 on routing failures)

ROUTING: (i) ERGODIC world (one β for all, no person-level confound —
Δ_erg CI must cover 0); (ii) NON-ERGODIC world (person intercepts a_u
correlated with x̄_u planting a derived, printed Δ ≠ 0 — recovery
within max(0.02·|Δ|-scale, 3·rep sd)); (iii) OWNED-SLOPES world
(Var(β) > 0 at an ownership-0.5 operating point (#76) — ρ_own
recovery); (iv) NULL world (β ≡ 0, no relation: every level must read
0 within band); (v) #85b bootstrap-zero on (iv); (vi) **REGION PRICING
(#88a executed in-leg): the #87 boundary-region half-widths for BOTH
routed objects are set to Part 0's realized bootstrap half-widths on
the matched planted worlds, pinned in the artifacts BEFORE any real
number.** DESCRIPTIVE (annotate, never stop): level-value recoveries.

### Cells (NULL-first #55; #75; regions priced per (vi))

On Δ_erg (primary arm): 1 **LEVELS_INDISTINGUISHABLE** — CI includes 0
(scoped by realized width; no equivalence claim). 2
**NONERGODIC_SAME_SIGN** — CI excludes 0, β_between and β_within share
sign. 3 **NONERGODIC_SIGN_FLIP** — CI excludes 0, signs differ with
both slopes detected. Co-primary (X2's scheme with priced regions):
ρ_own(β) in NOT/AT/WEAKLY/AT/STRONGLY. Arm divergences #73; primary
routes.

### Registered leans

- β_within < 0 (commoner venue → shorter comments) — held moderately;
  β_between < 0 likewise.
- Δ_erg ≠ 0, same sign, |β_between| > |β_within| (composition
  amplifies the between-person relation) — NONERGODIC_SAME_SIGN, held
  weakly; the schema's whole point is that level 1 need not equal
  level 2, and the dissociation family leans that way.
- ρ_own(β): WEAKLY_OWNED (X1c's idiosyncratic response is the
  categorical cousin at R = 0.279).
- Big5 replication: same Δ_erg cell; ownership possibly higher (X2's
  cohort flag) — #73 either way.

### Deliverables and discipline

Standard six: `scripts/run_suica_m4_x4_three_levels.py` + tests
(within-centering exactness; the ergodic world's Δ ≈ 0 honesty; the
non-ergodic world's derived Δ; slope-ownership machinery on a toy;
region-pricing pinning order — priced before real, asserted from
timestamps; anchors; #83 helper); gitignored
`results/m4_x4_three_levels/` (event cache with x attached); report
`reports/SUICA_M4_X4_THREE_LEVELS_REPORT.md` (rule-24; the three-level
table per arm; Δ_erg with CI; ownership with priced regions; gate
table ROUTING/DESCRIPTIVE; the owner-schema mapping stated; boundaries:
metadata-only, volume-not-content, projection caution, no psychological
naming, cohort caveats); outcome appended here; one CLAIMS_LEDGER row;
ONE commit `feat(m4-x): X4 — the three-level decomposition — <VERDICT>`;
suite green (1541 + new); #83 scan (baseline 4). SEED = 20260819;
B_perm = 499; B_boot = 1000.

---

## X4 outcome (executor, 2026-08-19)

**`NONERGODIC_SIGN_FLIP`** (co-primary ownership: **`NOT_OWNED`**). All six
ROUTING clauses of Part 0 passed and all six DESCRIPTIVE clauses annotated, so
the A1 stop did not fire and the owner's three-level schema has its first
corpus instance. Runtime about 40 s after the cache build (one stream pass
over the 17,640,062-row comments file, four columns, then everything
vectorised off a fresh gitignored event cache with x attached).

**The headline, in one line: the between-person relation and the
average-within-person relation have OPPOSITE SIGNS on this corpus.** Across
people, someone who spends their life in commoner venues writes MORE
(`beta_between = +0.069062` [+0.048843, +0.090518]); within a person, moving
to a commoner venue makes them write LESS (`beta_within = −0.007316`
[−0.011278, −0.003048]). `Delta_erg = +0.076377` [+0.055630, +0.097254], an
interval nowhere near zero and 3.5× the width of the priced null-world region.
Both slopes are detected against their own intervals, which is what the
registered cell 3 requires.

### The census reproduced the planner's table to the unit (#78, 21 predicates)

| registered predicate | registered | observed |
|---|---|---|
| rows parseable | 17,640,062 | 17,640,062 |
| authors (Big5 + disjoint) | 10,296 (1,401 + 8,895) | 10,296 (1,401 + 8,895) |
| communities | 46,214 | 46,214 |
| x = log10(share) range (2 dp) | [−7.25, −1.14] | [−7.25, −1.14] |
| pool (≥ 50 each half ∧ sd(x) > 0 each half) | 8,004 / 1,112 | 8,004 / 1,112 |
| within-author sd(x) q25/med/q75, disjoint | 0.805 / 0.965 / 1.111 | 0.805 / 0.965 / 1.111 |
| within-author sd(x) q25/med/q75, Big5 | 0.660 / 0.856 / 1.013 | 0.660 / 0.856 / 1.013 |
| distinct communities per author, median | 64 / 55 | 64 / 55 |
| top-1 community share, med / q90 | 0.259 / 0.577 and 0.329 / 0.743 | same |

TWO DEFINITIONS WERE PINNED BY THE ANCHORS, and both are worth recording
because neither is the obvious reading.

1. **"within-author sd(x)" is the MEAN OF THE AUTHOR'S TWO HALF SDs**
   (population sd), not the sd over all of the author's events. Pooling both
   halves reads 0.834 / 0.999 / 1.148 instead — a different quantity, because
   an author who drifts between communities across their own median adds
   BETWEEN-half spread that no within-half slope can use.
2. **The `sd(x) > 0` pool clause is float-path sensitive, and the anchors pin
   the path.** `np.std` on each half's contiguous x slice (numpy's pairwise
   two-pass) gives exactly 8,004 / 1,112; a grouped sequential accumulation
   gives 8,008 / 1,114; the mathematically exact test `min(x) == max(x)` gives
   8,000 / 1,103. The anchored path therefore admits 13 halves whose x is
   mathematically CONSTANT. The leg reproduces the anchored path exactly and
   then handles those 13 where they belong: their contribution to the level-1
   and level-2 pooled sums is exactly zero and is set to zero, and the level-3
   estimator returns NaN for them and counts them (4 authors dropped on the
   primary arm, 9 on Big5, 1 on floor-100).

In-leg census: the floor-100 sensitivity pool is **6,859** disjoint authors;
the primary arm carries 8,004 authors, 16,008 cells and 14,581,265 events; the
Big5 arm 1,112 / 2,224 / 2,984,569.

### Part 0 — six ROUTING clauses, all PASS

| # | world | required | observed | status |
|---|---|---|---|---|
| i | ERGODIC (one β, intercepts unrelated to x̄_u) | \|mean Δ\| ≤ max(0.0050, 3·rep sd) = 0.026973 and ≥ 6/8 CIs cover 0 | Δ = +0.002027 (sd 0.008991), **8/8** CIs cover 0 | PASS |
| ii | NON-ERGODIC, derived Δ = γ = +0.25 | \|gap\| ≤ max(2%·0.25, 3·rep sd) = 0.028108 | Δ = +0.256506 (sd 0.009369), gap +0.006506 | PASS |
| iii | OWNED-SLOPES at the derived ρ = 0.50 | \|gap\| ≤ max(0.02, 3·rep sd) = 0.4948 | ρ_own = +0.7790 (sd 0.1649), gap +0.2790 = 1.69 rep sd | PASS (uninformative — see below) |
| iv | NULL (β ≡ 0) — all four objects read 0 | ≥ 6/8 on each of five coverage counts | β_b +0.002927 (8/8), β_w +0.000141 (8/8), Δ +0.002786 (8/8), ρ_own +0.0024 (6/8 CIs, 7/8 in band) | PASS |
| v | #85b bootstrap-zero on the NULL world | ≥ 6/8 both counts | Δ 8/8, ρ_own point-coverage 8/8 | PASS |
| vi | #88a region pricing, pinned before any real number | three finite positive half-widths, artifact first | Δ ±0.021532; ρ ±0.3827 at 0.15, ±0.1837 at 0.50 | PASS |

Clause (i) records a RECORDED REFINEMENT: the registration says "the Δ CI must
cover 0 (8 replicates)", and with eight separately drawn 95% intervals the
honest reading is a COVERAGE floor rather than "all eight or the leg dies"
(Binomial(8, 0.95): a correct estimator fails an 8-of-8 requirement 34% of the
time and a 6-of-8 requirement 0.58% of the time). The realized count was 8/8,
so the refinement changed nothing about this run's outcome.

The DERIVED Δ of clause (ii) is analytic and parameter-free: within-(u,h)
centering annihilates a_u so E[β_within] = β, while ȳ_u = (β+γ)x̄_u + w_u + ē_u
gives E[β_between] = β + γ, hence **Δ_erg = γ exactly**. The contract test
asserts the identity to machine precision with the noise switched off, not
merely on average. That same world also plants a SIGN FLIP (β_within = −0.10,
β_between = +0.15) and D2 recovered it: −0.1000 / +0.1565.

### The three levels

| arm | authors | β_between (CI) | β_within (CI) | Δ_erg (CI) | cell |
|---|---|---|---|---|---|
| PRIMARY, disjoint | 8,004 | +0.069062 [+0.048843, +0.090518] | −0.007316 [−0.011278, −0.003048] | **+0.076377 [+0.055630, +0.097254]** | `NONERGODIC_SIGN_FLIP` |
| Big5 replication | 1,112 | +0.051346 [−0.005921, +0.107078] | −0.009454 [−0.021970, +0.001675] | +0.060799 [+0.002037, +0.116844] | `NONERGODIC_SIGN_UNRESOLVED` |
| floor-100 sensitivity | 6,859 | +0.063888 [+0.039694, +0.087040] | −0.007804 [−0.011817, −0.003275] | +0.071692 [+0.046891, +0.094870] | `NONERGODIC_SIGN_FLIP` |

The Big5 arm lands in a CELL THE REGISTRATION DID NOT NAME. Its Δ excludes 0
and its two slopes have opposite point signs, but neither slope's own interval
excludes 0, so the flip is not established at n = 1,112. Rather than force it
into cell 2 or cell 3 the runner adds a completeness fallback,
`NONERGODIC_SIGN_UNRESOLVED`, and flags the divergence under #73. The primary
routes. Read plainly: the Big5 cohort agrees in DIRECTION and magnitude with
the primary and simply cannot resolve it — which is exactly what a 7×-smaller
cohort should look like, and is a weaker kind of #73 flag than a real
disagreement.

Level 2 is the mean of two half slopes that DISAGREE: −0.014776 early,
+0.000144 late (Big5 −0.022666 / +0.003759; floor-100 −0.015400 / −0.000209).
The registered estimand is their average and that is what routes, but level 2
should not be read as a stationary quantity on this corpus. Nothing in this
leg separates calendar drift, cohort composition, or the half split itself.

### Level 3, and the leg's real methodological finding

The level-3 object as registered is **ONE-AUTHOR FRAGILE**, and the leg's own
#88a pricing is what exposed it. `den = Σ(x − x̄_cell)²` is unbounded below in
this pool: the median half has den ≈ 324, but the smallest non-degenerate late
half has den = 4.66e−06, so that ONE author's per-half slope carries a
sampling variance of 214,404 — about 96% of the whole arm's
A_late = 27.95 = σ²·E[1/den]. Consequences, all of them recorded:

- the #76 mapping lands on a pathological operating point (V = 9.34,
  sd(β_u) = 3.06) and clause (iii) recovers 0.779 instead of 0.500, PASSING
  only because its own replicate sd is 0.165 and the tolerance is 3× that. The
  clause is recorded as PASSED **and** as UNINFORMATIVE;
- the priced ρ regions come out ±0.3827 at 0.15 and ±0.1837 at 0.50, which
  OVERLAP: `WEAKLY_OWNED` is an EMPTY interval at this pricing. The
  registration's own ownership lean therefore could not have held whatever the
  data said, and is recorded BROKEN with that caveat attached. The routed cell
  is unaffected — `NOT_OWNED` is decided NULL-first, by whether the interval
  covers 0, never by a region edge;
- `Var(β)` by cross-half covariance comes out NEGATIVE on the primary
  (−0.00972847) and Big5 (−0.00047770) arms, so sd_true and the headroom are
  undefined and print as n/a;
- ρ_own reads −0.0052 with a wildly asymmetric CI [−0.0156, +0.2740] — the
  bootstrap resamples that happen to omit the one author read ≈ +0.27.

The x-only PRECISION FLOOR (den ≥ 1, i.e. the per-half slope's sampling sd
does not exceed the event-noise sd) was pinned in the config before any real
number and carried as a NON-ROUTING sensitivity on every arm. It excludes 14
of 8,000 primary authors (0.18%) and changes the reading completely:

| arm | authors above the floor | ρ_own (CI) | Var(β) | sd_true | headroom about the mean |
|---|---|---|---|---|---|
| PRIMARY | 7,986 | **+0.2768 [+0.2416, +0.3103]** | +0.00798683 | 0.089369 | [−0.1681, +0.1822] |
| Big5 | 1,096 | **+0.3338 [+0.2590, +0.4184]** | +0.01745042 | 0.132100 | [−0.2555, +0.2623] |
| floor-100 | 6,851 | +0.3246 [+0.2941, +0.3554] | +0.00778750 | 0.088247 | [−0.1692, +0.1767] |

Part 0's D6 clause shows the machinery is sound and the pool's den tail is not:
on the floored object the SAME planted worlds recover +0.5051 (sd 0.0138,
half-width 0.0293) against a planted 0.50, and +0.1468 (half-width 0.0357)
against a planted 0.15 — half-widths in the same range X2 realized (0.032 /
0.045), which is what #88a expected a priced region to look like.

**This is an executor annotation and it routes nothing**, but it should not be
lost: at 0.2768 the floored primary sits within a whisker of X2's ρ_own = 0.259
and X1c's R = 0.279. Three different second-order personal channels on the
same corpus — venue response profile, expression path, and now the venue
GRADIENT of expression volume — all land in the same narrow band. Whether that
is one channel seen three ways or a coincidence of three ~0.27s is X3's
problem, not this leg's.

### Registered leans

| lean | status |
|---|---|
| β_within < 0 | **HELD** (−0.007316, CI excludes 0) |
| β_between < 0 | **BROKEN** — it is POSITIVE, +0.069062, and that reversal is the leg's finding |
| Δ_erg NONERGODIC_SAME_SIGN with \|β_between\| > \|β_within\| | **BROKEN** on the cell (a SIGN FLIP, not same-sign); the magnitude half of the lean held, 0.069 vs 0.007 |
| ρ_own WEAKLY_OWNED | **BROKEN** (`NOT_OWNED`) — against a ladder whose WEAKLY_OWNED interval was empty; the floored annotation reads +0.2768 |
| Big5 same Δ cell | **BROKEN** (`NONERGODIC_SIGN_UNRESOLVED`, a power difference not a disagreement) |
| Big5 ownership possibly higher | **HELD** (+0.3338 vs +0.2768 floored; the routed pair is noise) |

### Governance

Four metadata columns streamed once; no body ever read; `author_profiles.csv`
never opened; the leg is label-free end to end. ID-leak scan over all 10,296
author names PASSED with 0 NEW hits (4 pre-existing dictionary collisions
carried unchanged from HEAD). The executor's first draft tripped the scan on
two NEW hits — the word "independent" is a PANDORA username, the same
collision the X2 executor recorded — and the prose was rewritten rather than
the gate relaxed. #88a ordering asserted from artifact timestamps: the region
pricing was on disk 6 ms before the first real-arm number and a test asserts
the order from the artifacts, not from prose.

### Deviations and defect candidates (none run)

1. **DEVIATION (recorded, changed no outcome):** clause (i)'s "the CI must
   cover 0 (8 replicates)" implemented as a ≥ 6/8 coverage floor with the
   binomial arithmetic printed; realized 8/8.
2. **DEVIATION (recorded):** a fourth Δ cell, `NONERGODIC_SIGN_UNRESOLVED`,
   added for completeness because the registered cell 3 requires both slopes
   detected and the Big5 arm satisfies everything except that. Without it the
   Big5 arm would have been forced into a cell it does not belong in.
3. **DEVIATION (recorded, non-routing):** the x-only precision floor arm and
   the two floored Part 0 worlds (D5/D6) are unregistered additions. They exist
   because the registered level-3 object turned out to be one-author fragile
   and a report that said only `NOT_OWNED` would have been true and useless.
4. **Defect candidate #89:** a pool predicate stated as `sd(x) > 0` is not a
   predicate — three defensible float paths give three different pools
   (8,004/1,112, 8,008/1,114, 8,000/1,103). Registrations should state the
   ESTIMABILITY floor the clause is trying to enforce (here: a minimum
   `Σ(x − x̄)²`, which is what actually bounds the slope's variance), not a
   strict inequality on a computed sd.
5. **Defect candidate #90:** a `max(floor, 3 × replicate sd)` tolerance cannot
   distinguish "recovered" from "too unstable to say". Clause (iii) passed at a
   gap of 0.279 on a target of 0.500. A routing tolerance should carry a
   SECOND arm — a cap on the replicate sd itself, or a required ratio of gap to
   floor — so that a runaway spread fails rather than absolves.
6. **Defect candidate #91:** #87 boundary regions can OVERLAP once priced, which
   silently empties an interior cell. The pricing step should assert ladder
   coherence and, when the regions collide, report the ladder as unresolved
   rather than letting a lean be scored against a cell that could not occur.
7. Carried, unchanged: the X2 cohort-selection caveat — a Big5/disjoint
   difference cannot be separated from how the 1,401 were selected.

## X4 planner adjudication (2026-08-19)

**Verdict ACCEPTED: `NONERGODIC_SIGN_FLIP`** — Δ_erg = +0.0764
[+0.0556, +0.0973] on 8,004 authors, with β_between = +0.0691 and
β_within = −0.0073 of OPPOSITE SIGNS; the floor-100 arm reproduces the
cell; the Big5 arm is same-direction underpowered
(NONERGODIC_SIGN_UNRESOLVED, a #73 power flag, not a disagreement —
its Δ CI [+0.0020, +0.1168] sits inside the primary's story). The
executor's added fourth cell is adopted retroactively as the honest
handling of an underpowered replication. Gate 6/6 routing first try;
the ERGODIC world read Δ = +0.002 (8/8 CIs covering 0) — the
instrument demonstrably does NOT manufacture non-ergodicity.

**The reading, in the bridge's vocabulary.** Between persons, a life
spent in commoner venues goes with WRITING MORE; within a person,
moving to a commoner venue goes with WRITING LESS. The between-person
relation is not merely a biased estimate of the within-person one —
it has the WRONG SIGN. 三枝・下司 (2025) の警鐘のこの関係における
実測形:個人間関係から個人内関係を読むと、大きさ以前に**向き**を
誤る。The plausible mechanism (named, not tested): composition —
who lives where — dominates between persons, while the within-person
response reflects venue norms; selection and response point in
opposite directions here.

**Ownership: NOT_OWNED as routed, with the instructive fracture on
record.** The registered level-3 object carried no estimability
floor: one author-half with Σ(x−x̄)² = 4.7e−06 contributed 96% of a
variance term, the priced ρ regions ballooned (±0.38) until
WEAKLY_OWNED became an EMPTY interval (the leaned cell could not
occur), and cross-half Var(β) went negative. NULL-first routing kept
the verdict sound. The executor's non-routing precision floor
(den ≥ 1) reads **ρ_own = +0.2768 [+0.2416, +0.3103]** — and the
planner notes for the record what now stands beside it: X1c's
response-profile persistence 0.279, X2's rhythm ownership 0.259.
**Three different level-3 expression parameters, one narrow band
around ≈ 0.27 on the disjoint cohort.** Named (not claimed): a
possible level-3 reliability plateau for expression metadata at this
span — a future look, priced, not queued. Big5 ownership again higher
(+0.334), the X2 cohort flag repeating.

**Defects #89–#91 (planner, purchased; registry thirty-sixth note).**
(a) #89: a pool clause stated as "sd(x) > 0" is not an estimability
predicate — it admitted a 4.7e−06 denominator, and it is float-path
sensitive (8,004 / 8,008 / 8,000 by computation path; the anchored
path reproduced, but predicates must state an estimability FLOOR in
the estimand's own denominator units and pin the computation path).
(b) #90: a tolerance of max(floor, 3·rep sd) cannot distinguish
"recovered" from "too unstable to say" — clause (iii) passed
uninformatively on a 1.69-rep-sd gap; recovery clauses carry a
registered precision CEILING or report UNINFORMATIVE as a first-class
status. (c) #91: priced #87 regions can overlap and silently EMPTY an
interior cell — region pricing must assert LADDER COHERENCE (regions
disjoint and every interior cell non-empty), else the classification
collapses to the coarser partition EXPLICITLY. Lean record: 2 held,
4 broken — including the leaned ρ cell broken against an interval
that could not occur (#91's teeth).

**Line state.** X4 closes. The bridge (§2) receives the sign-flip in
the paper's vocabulary with this commit. Next per the bridge build-out:
X5 (the ergodicity atlas) with #89–#91 baked into its registration;
X-M/X3 queued behind it.

---

## X5 — the ergodicity atlas (registered BEFORE run, 2026-08-19)

The bridge's build-out item 2: not one relation but a MAP — which kinds
of metadata relations carry large level-conflation errors, and where do
individual parameters live. Five relations, each measured
THREE-LEVEL-COMPLETE (bridge §6) with X4's machinery, under the
#89–#91 conventions baked in from registration.

### 4W header

- **Object.** Five Where/What/When metadata relations at all three
  measurement series; the atlas's verdict is the SUMMARY SHAPE of the
  per-relation Δ_erg cells. Provenance: the owner's support program
  (bridge §5.2).
- **Relations (transforms pinned #70).** y = log1p(wcq) and
  x as follows unless stated: **R1** commonness × volume (X4's
  relation: Δ_erg IMPORTED from X4's committed artifacts with a
  bit-check; ρ_own RECOMPUTED under the estimability floor — X4's
  annotated +0.2768 is the check value); **R2** gap × volume
  (x = log10 seconds since the author's previous comment; first event
  dropped; nonpositive gaps dropped); **R3** volume × score
  (x = log1p(wcq), y = slog(score) = sign·log1p(|score|));
  **R4** commonness × score (y = slog(score)); **R5** commonness × gap
  (y = log10 gap seconds).
- **Fixed.** X2/X4 universe, order, halves; the ESTIMABILITY FLOOR
  (#89): per (author, half, relation), den = Σ(x − x̄)² ≥ 1 computed by
  the pinned two-pass float64 path; pool = ≥ 50 usable events per half
  AND the floor in each half. The level-3 estimator is the FLOORED
  per-author slope THROUGHOUT (X4's lesson made primary).
- **What falsifies.** All five relations reading LEVELS_INDISTINGUISHABLE
  (the atlas uniform-ergodic) would refute the conflation-warning's
  relevance to this corpus family.
- **Layer.** R.

### Census (planner #43; #77/#78 exact; #57 clean; #89 floor applied)

| relation | disjoint pool | Big5 pool | within-sd(x) med (disjoint) |
|---|---|---|---|
| R2 gap × volume | 7,989 | 1,100 | 1.138 |
| R3 volume × score | 8,008 | 1,116 | 1.031 |
| R4 commonness × score | 7,986 | 1,096 | 0.966 |
| R5 commonness × gap | 7,966 | 1,081 | 0.966 |

score missing 147 of 17,640,062 parseable; R1 pools are X4's (8,004 /
1,112). Anchors: all of the above plus X4's inherited set.

### Estimands and cells

Per relation × cohort: β_between, β_within (per-half, averaged),
Δ_erg with paired cluster bootstrap B = 1000; per-relation Δ cells as
X4's four (LEVELS_INDISTINGUISHABLE / NONERGODIC_SAME_SIGN /
NONERGODIC_SIGN_FLIP / NONERGODIC_SIGN_UNRESOLVED), regions priced per
#88a from the matched gate worlds. ρ_own per relation (FLOORED), with
the #91 LADDER-COHERENCE assertion: after pricing, regions must be
disjoint and every interior cell non-empty, ELSE the ownership
classification COLLAPSES EXPLICITLY to the binary
OWNED (CI > 0) / NOT_OWNED — no silent empty cells.

**Atlas summary (ROUTES, NULL-first #55):**
1. **ATLAS_UNIFORM_ERGODIC** — all five relations
   LEVELS_INDISTINGUISHABLE.
2. **ATLAS_UNIFORM_NONERGODIC** — all five in nonergodic cells.
3. **ATLAS_HETEROGENEOUS** — at least two different Δ cells across
   relations (the map has structure).

### Gate (Part 0 per relation, realized skeletons, #90 ceilings)

For EACH of R2–R5 (R1 inherits X4's passed gate): ERGODIC world
(Δ CI covers 0, ≥ 6/8), NON-ERGODIC world (derived printed Δ,
recovery), NULL world (all read 0), #85b bootstrap-zero. The OWNED
world once on R2's skeleton (machinery shared; floored estimator).
**#90 ceilings:** every routing recovery clause carries the assertion
rep-sd ≤ the tolerance floor; a clause whose rep-sd exceeds it is
UNINFORMATIVE and, being a routing clause, STOPS the leg (A1) — an
uninformative pass can certify nothing. DESCRIPTIVE clauses annotate.
Region pricing (#88a) per relation from its matched worlds, pinned
before any real number, with the #91 coherence check.

### Registered leans (directional, weakly held; the atlas exists
because these are guesses)

- R2: β_within > 0 (rest precedes longer comments); Δ SAME_SIGN.
- R3: both positive, small; SAME_SIGN or INDISTINGUISHABLE.
- R4: β_within > 0 (bigger venue, more eyes, more score); SAME_SIGN.
- R5: unleaned (stated: no defensible prior).
- Summary: **ATLAS_HETEROGENEOUS** (R1's SIGN_FLIP already in hand
  vs the SAME_SIGN leans above).
- Ownership: the ≈ 0.27 band (bridge §7) appears for at least two of
  the four new relations on the disjoint cohort — the level-3
  plateau's first test as a PREDICTION rather than an observation.

### Deliverables and discipline

Standard six: `scripts/run_suica_m4_x5_ergodicity_atlas.py` + tests
(floored-estimator correctness incl. the #89 two-pass path; per-relation
gate battery; #90 ceiling logic both directions; #91 coherence check
and the explicit collapse; R1 import bit-check; anchors; #83 helper);
gitignored `results/m4_x5_ergodicity_atlas/`; report
`reports/SUICA_M4_X5_ERGODICITY_ATLAS_REPORT.md` (rule-24; THE ATLAS
TABLE — 5 relations × {β_b, β_w, Δ, cell, ρ_own, ownership cell} ×
2 cohorts; gate tables; the bridge-vocabulary summary paragraph;
boundaries incl. volume-not-content, projection caution, cohort
caveats); outcome appended here; one CLAIMS_LEDGER row; ONE commit
`feat(m4-x): X5 — the ergodicity atlas — <VERDICT>`; suite green
(1604 + new); #83 scan (baseline 4). SEED = 20260819; B_perm = 499;
B_boot = 1000.

---

## X5 outcome (executor, 2026-08-19)

**`ATLAS_HETEROGENEOUS`** — four different Δ cells across the five relations
on the disjoint cohort. Every ROUTING clause of every per-relation Part 0
passed, every #90 ceiling read INFORMATIVE, every priced ladder read
COHERENT, and the 22-predicate census reproduced the planner's table to the
unit. Runtime 102 s off the cache, plus a 45 s cache build (one 26 s stream
pass over the 17,640,062-row comments file for five metadata columns).

**The headline: the level-conflation error is NOT a property of the corpus,
it is a property of the RELATION.** On one event stream, at one span, with
one estimator, the same contrast reads a sign flip, two scoped nulls, an
unresolved flip and a same-sign 3.5× overstatement, depending only on which
pair of metadata channels is crossed.

### THE ATLAS — disjoint cohort (PRIMARY)

| relation | authors | β_between | β_within | Δ_erg [95% CI] | Δ cell | ρ_own [95% CI] | ownership |
|---|---|---|---|---|---|---|---|
| **R1** commonness × volume | 7,986 (Δ on X4's 8,004) | +0.069062 | −0.007316 | **+0.076377** [+0.055630, +0.097254] | `NONERGODIC_SIGN_FLIP` | +0.2768 [+0.2441, +0.3092] | `WEAKLY_OWNED` |
| **R2** gap × volume | 7,989 | +0.078450 | +0.059092 | +0.019358 [−0.001034, +0.037833] | `LEVELS_INDISTINGUISHABLE` | +0.4151 [+0.3925, +0.4372] | `WEAKLY_OWNED` |
| **R3** volume × score | 8,008 | +0.006501 | +0.005172 | +0.001329 [−0.010809, +0.013378] | `LEVELS_INDISTINGUISHABLE` | +0.3585 [+0.3320, +0.3847] | `WEAKLY_OWNED` |
| **R4** commonness × score | 7,986 | −0.022737 | +0.001884 | −0.024622 [−0.035282, −0.013686] | `NONERGODIC_SIGN_UNRESOLVED` | +0.2863 [+0.2435, +0.3319] | `WEAKLY_OWNED` |
| **R5** commonness × gap | 7,966 | −0.256325 | −0.073531 | **−0.182794** [−0.205119, −0.158719] | `NONERGODIC_SAME_SIGN` | +0.2164 [+0.1849, +0.2480] | `WEAKLY_OWNED` |

### THE ATLAS — Big5 replication (#73)

| relation | authors | β_between | β_within | Δ_erg [95% CI] | Δ cell | ρ_own [95% CI] | ownership |
|---|---|---|---|---|---|---|---|
| R1 | 1,096 (Δ on 1,112) | +0.051346 | −0.009454 | +0.060799 [+0.002037, +0.116844] | `NONERGODIC_SIGN_UNRESOLVED` | +0.3338 [+0.2681, +0.4199] | `WEAKLY_OWNED` |
| R2 | 1,100 | +0.125773 | +0.083492 | +0.042281 [−0.014043, +0.093462] | `LEVELS_INDISTINGUISHABLE` | +0.5380 [+0.4800, +0.5916] | `STRONGLY_OWNED` |
| R3 | 1,116 | +0.000506 | +0.014898 | −0.014392 [−0.044014, +0.015019] | `LEVELS_INDISTINGUISHABLE` | +0.4356 [+0.3536, +0.5175] | `WEAKLY_OWNED` |
| R4 | 1,096 | +0.004780 | +0.007481 | −0.002702 [−0.030359, +0.027177] | `LEVELS_INDISTINGUISHABLE` | +0.3396 [+0.2674, +0.4108] | `WEAKLY_OWNED` |
| R5 | 1,081 | −0.347683 | −0.084064 | −0.263619 [−0.322152, −0.206946] | `NONERGODIC_SAME_SIGN` | +0.2398 [+0.0788, +0.3885] | `WEAKLY_OWNED` |

Big5 route: `ATLAS_HETEROGENEOUS` as well (three distinct cells). Two #73
divergences on Δ (R1 flip → unresolved, R4 unresolved → indistinguishable),
both power differences at n ≈ 1,100 rather than disagreements; ONE on
ownership (R2 weakly → strongly). Big5 ownership is HIGHER on all five
relations — the X2/X4 cohort flag repeating five times.

### The census reproduced the planner's table to the unit (#78, 22 predicates)

| predicate | registered | observed |
|---|---|---|
| rows parseable | 17,640,062 | 17,640,062 |
| authors (Big5 + disjoint) | 10,296 (1,401 + 8,895) | same |
| communities / x range | 46,214 / [−7.25, −1.14] | same |
| score missing | 147 | 147 |
| R1 pool by X4's `sd(x) > 0` path | 8,004 / 1,112 | 8,004 / 1,112 |
| R2 / R3 / R4 / R5 pools, disjoint | 7,989 / 8,008 / 7,986 / 7,966 | same |
| R2 / R3 / R4 / R5 pools, Big5 | 1,100 / 1,116 / 1,096 / 1,081 | same |
| within-sd(x) medians, disjoint | 1.138 / 1.031 / 0.966 / 0.966 | same |

Two arithmetic identities fell out and are worth recording. The candidate set
(≥ 50 events in each half, before any relation's own clause) is **9,124**
authors = 8,008 + 1,116, which is exactly R3's pool: with x = log1p(wcq) and
≥ 50 events a half, the #89 floor never binds. And R4's pool (7,986 / 1,096)
is exactly X4's own precision-floored author count, because `score` is missing
for only 147 of 17.6M rows — so R1 and R4 are pooled on the SAME people and
differ only in y.

### Part 0 — five per-relation batteries, all PASS

| relation | ERGODIC Δ (sd, cover) | NON-ERGODIC gap (rep sd) | #90 Δ ceiling | OWNED ρ (rep sd) | #90 ρ ceiling | priced Δ / ρ@0.15 / ρ@0.50 | #91 |
|---|---|---|---|---|---|---|---|
| R1 | inherited from X4 (+0.002027, 8/8) | inherited | inherited | inherited | inherited | ±0.021532 / ±0.0357 / ±0.0293 (X4's FLOORED worlds) | COHERENT |
| R2 | −0.001043 (0.015430, 6/8) | +0.002026 = 0.14 rep sd | 0.014128 ≤ 0.02 | +0.5042 (0.0101) | 0.0101 ≤ 0.02 | ±0.019237 / ±0.0284 / ±0.0211 | COHERENT |
| R3 | +0.003458 (0.019292, 7/8) | +0.005285 = 0.61 rep sd | 0.008675 ≤ 0.02 | +0.5013 (0.0126, annotates) | — | ±0.022525 / ±0.0290 / ±0.0218 | COHERENT |
| R4 | −0.001790 (0.011940, 8/8) | −0.001547 = 0.09 rep sd | 0.018061 ≤ 0.02 | +0.4994 (0.0090, annotates) | — | ±0.021514 / ±0.0388 / ±0.0306 | COHERENT |
| R5 | −0.006324 (0.016342, 6/8) | +0.000591 = 0.05 rep sd | 0.012910 ≤ 0.02 | +0.5047 (0.0172, annotates) | — | ±0.021404 / ±0.0377 / ±0.0292 | COHERENT |

The ownership RECOVERY clause routes once, on R2, as registered; the owned
worlds still ran on every relation because #88a prices each relation's ρ
regions from ITS OWN matched worlds, and there they annotate (D4). Every NULL
world read four zeros at ≥ 6/8 coverage and every #85b bootstrap-zero clause
passed.

### What the #89 floor bought — the three defects, closed

| object | X4, no floor | X5, floored |
|---|---|---|
| cross-half Var(β), R1 disjoint | **−0.009728** (negative; sd_true undefined) | **+0.007987** (defined; sd_true 0.0894) |
| owned-world ρ replicate sd (the #90 object) | 0.1649 → clause PASSED UNINFORMATIVELY | 0.0090–0.0172 → **INFORMATIVE everywhere** |
| priced ρ half-width at 0.50 (the #91 object) | ±0.1837, overlapping ±0.3827 at 0.15 | ±0.0211–0.0306 |
| ladders coherent | 0 of 1 (`WEAKLY_OWNED` was an EMPTY interval) | **5 of 5** |

The three defects the planner purchased from X4 were not merely honoured, they
were the leg's most useful machinery: with the floor in the POOL rather than in
a footnote, the level-3 object stopped being one-author fragile, the ceiling
clauses became able to certify, and the registered ownership ladder resolved
for the first time — every relation lands in a NON-EMPTY interior cell.

### The plateau PREDICTION — BROKEN, and instructively

| relation | ρ_own | in [0.25, 0.30]? |
|---|---|---|
| R2 gap × volume | +0.4151 | no |
| R3 volume × score | +0.3585 | no |
| R4 commonness × score | **+0.2863** | yes |
| R5 commonness × gap | +0.2164 | no |

One of four, against a registered ≥ 2. Including R1's imported +0.2768 the
five values run **0.2164 → 0.4151**, a spread of 0.199 on one cohort at one
span. **The ≈ 0.27 band was a coincidence of three parameters, not a plateau.**
Level-3 reliability is a property of the relation being owned — which is the
most useful negative result the atlas produces for build-out item 3, since a
catalogue whose entries were all the same number would not be worth compiling.

### Leans: 5 held, 3 broken, 1 registered unleaned

HELD — R2 β_within > 0 (+0.059092: rest precedes longer comments); R3 both
slopes positive and small (+0.006501 / +0.005172); R3's cell
(`LEVELS_INDISTINGUISHABLE`, which the lean admitted); R4 β_within > 0
(+0.001884); the summary route `ATLAS_HETEROGENEOUS`.
BROKEN — R2's `NONERGODIC_SAME_SIGN` (the Δ CI covers zero by 0.001, 2.7% of
its own width: a power-limited null, recorded as such); R4's
`NONERGODIC_SAME_SIGN` (Δ excludes zero with opposite point signs but
β_within's own interval does not, so it is X4's fourth cell again); the
plateau prediction. UNLEANED — R5, registered with no defensible prior, and it
produced the atlas's LARGEST conflation error.

### R5, the unleaned relation, is the atlas's largest error

Between people, someone whose life is spent in commoner venues comments with
SHORTER gaps (β_between = −0.256); within a person, moving to a commoner venue
also shortens the gap, but only 29% as steeply (β_within = −0.0735). The
level difference −0.183 [−0.205, −0.159] is **2.5× the within-person slope
itself**: reading the within-person relation off the between-person one would
overstate it by that factor. Mechanism named, not tested: composition (who
lives in big venues) again, now on the timing channel.

### The R1 import bit-check

X4's Δ and its CI are IMPORTED, not recomputed. The floored ownership was
recomputed on this leg's fresh five-column cache and reproduced X4's committed
+0.276816343318229 with an absolute difference of **0.0** on the same 7,986
authors — same events, same order, same pinned two-pass path. The atlas and
X4 are demonstrably measuring the same corpus.

### Honest anomalies (all in the artifacts)

The five relations are five PROJECTIONS of one stream and share channels
(R1/R4/R5 share x, R1/R2 share y, R3/R4 share y), so agreement between two of
them is partly shared measurement. `slog(score)` is platform feedback, never
quality. The gap channel drops 433,827 nonpositive gaps plus one first event
per author, and the drops concentrate in same-second bursts — the fastest
activity. R1 and R4 have level-2 halves that disagree in sign, so level 2 is
not stationary there. R4's Δ ceiling passed at 90% of its ceiling and R5's ρ
ceiling at 86% — thin margins, named. R2's `LEVELS_INDISTINGUISHABLE` is
decided by 2.7% of its CI width.

### Deviations, all recorded

1. **The Δ tolerance FLOOR is an absolute 0.02 in Δ units**, not X4's 2% of
   the planted scale (0.005). #90 turns the floor into a CEILING on the
   replicate spread, so it has to be a resolution statement in the estimand's
   own units rather than a fraction of a freely chosen planted scale. The
   planted world itself is X4's, untouched (β = −0.10, γ = +0.25, sd_a = 0.5,
   sd_e = 1.0), and the realized recovery tolerance is unchanged because the
   3 × rep-sd term binds either way.
2. **The owned worlds ran on all four new relations**, since #88a prices per
   relation from matched worlds; only R2's ownership RECOVERY clause routes.
3. **Atlas route 2 was refined** to "all five in the SAME nonergodic cell";
   the literal predicate ("all five nonergodic") is recorded next to it and
   did not bind here (it read false).
4. **The plateau band was operationalized** as the point estimate lying in
   [0.25, 0.30], pinned in the config before the run from bridge §7's three
   values; the CI-overlap count is co-reported (also 1 of 4).
5. R1's ownership pool (7,986 / 1,096) is the FLOORED subset of X4's Δ pool
   (8,004 / 1,112); the atlas row rests on two slightly different author sets
   and says so.
6. A `--stop-after-census` development flag exists and writes no estimand.

### Governance

Five metadata columns streamed once; no body ever read; `author_profiles.csv`
never opened; aggregates only; EXPLORATORY corpus-level with no person claim
and no psychological naming; the cohort-selection caveat carried unchanged;
ID-leak scan over all 10,296 author names PASSED with 0 NEW hits (4
pre-existing dictionary collisions carried from HEAD). Suite 1,604 + 70 new.

### Line state

X5 closes. The bridge receives build-out item 2 (the map) and, unexpectedly,
a correction to item 3's premise (there is no level-3 plateau). Queued behind
it, unchanged: X-M / X3.

## X5 planner adjudication (2026-08-19)

**Verdict ACCEPTED: `ATLAS_HETEROGENEOUS`** — four distinct Δ cells
across five relations on the primary cohort; every routing clause
passed, every #90 ceiling INFORMATIVE, every #91 ladder COHERENT; the
R1 import bit-checked at 0.0; 22/22 anchors.

**The map's reading (the leg's theory yield).** The conflation error is
a PROPERTY OF THE RELATION: relations routed through venue COMPOSITION
(commonness × anything) are non-ergodic — sign-flip on volume
(+0.076), sign-unresolved on score (−0.025, between-person NEGATIVE),
and a 2.5× same-sign exaggeration on gap (Δ = −0.183, the atlas's
largest error, on the one relation the planner declined to lean) —
while relations between two BEHAVIORS (volume × score, gap × volume)
are level-consistent at this power. Mechanism named (not tested),
consistent with X4's: who-lives-where dominates between persons;
within persons only the response to the venue remains. In the
bridge's vocabulary: 「どの関係で混同が危険か」への最初の答え —
**組成(Where)を経由する関係は危険、行動どうしの関係は比較的安全**。

**Ownership is universal and the plateau is DEAD.** All five slopes
are WEAKLY_OWNED on the disjoint cohort (0.216–0.415) and the Big5
cohort reads HIGHER ON ALL FIVE (the cohort flag now at seven
occurrences). The registered ≈0.27 plateau prediction BROKE (1 of 4 in
band; spread 0.199) — level-3 reliability is a property of the
RELATION, not a constant of the medium. A registered conjecture
killed within one leg of its birth is the discipline working on
theory, not just on instruments; build-out item 3's catalogue premise
is corrected BEFORE compilation (catalogue BY RELATION, and BY
PARAMETER CLASS — see the bridge §8 table this commit adds: LEVEL
parameters are strongly owned, 0.77–0.94; RESPONSE/DYNAMICS
parameters weakly, 0.22–0.42).

**Defect #92 (planner, purchased; registry thirty-seventh note).** The
X5 registration inherited X4's recovery tolerance in %-of-planted form
where #90's own logic demands a RESOLUTION statement in the estimand's
units — under the literal inherited floor every relation would have
A1-stopped for replicate spreads the #90 ceiling itself certifies as
informative. The executor's re-posing (absolute floor 0.02 in Δ units,
matching the priced region scale) is accepted and conventionalized:
gate floors are resolution statements in estimand units, tied to the
priced region half-width, stated at registration. Remaining
deviations (owned worlds on all four skeletons with only R2 routing;
the refined route-2 predicate recorded beside the literal; the
pinned plateau operationalization; the R1 pool distinction; the dev
flag) accepted as recorded. Hygiene note for a future leg, no number:
one contract test writes to /tmp instead of pytest tmp_path.

**Line state.** X5 closes; the bridge receives §8 (the map + the
ownership catalogue). Next per the build-out: X-M (the mains
estimator instrument) then X3 (the stage-E trait join, whose
coordinate menu is now rich: response profile 0.279, rhythm 0.259,
five relation slopes 0.22–0.42 — all Big5-higher — plus the strongly
owned level parameters).

---

## X-M — the mains estimator (instrument leg, registered BEFORE run, 2026-08-19)

The #87 prerequisite for X3: ESTIMATORS of the true author main Var(a)
and community main Var(b) on the incomplete two-way design — X1c
correctly renamed its budget rows MARGINAL; this leg builds the main
estimators proper, R1-class (certificates only, no theory verdict).

### Design

- **Skeleton.** X1c's primary design (law vocabulary, support floor
  s = 5: 3,665 authors × 1,000 communities, 31,899 shared pairs,
  LCC coverage 1.000 — #78 anchors, STOP on mismatch), n_min = 10
  cells, from X1's committed cell cache.
- **Estimators.** Per half, the exact two-way FE fit (X1b/X1c
  machinery, alternating projections to ≤ 1e-10 on both residual group
  means); coefficients IDENTIFIED by the pinned normalization: author
  and community coefficient vectors each centered to mean 0 over the
  connected component, per half. Then:
  Var̂(a) = cov over authors of (â_early, â_late);
  Var̂(b) = cov over communities of (b̂_early, b̂_late), size-weighted
  primary and unweighted secondary. Shares of comment-level Var(y).
  Any joint-estimation df interaction is DERIVED, printed, and
  contract-tested (the X1c pattern).
- **Gate (the leg IS the gate; #85/#86/#89/#90/#92 in force).** On the
  realized skeleton, wholly synthetic y, 8 replicates each: worlds
  {a-only}, {b-only}, {g-only}, {null}, {full: a .30 + b .08 + g .02}.
  ROUTING clauses: own-component recovery with floors stated as
  RESOLUTION 0.01 in share units (#92) and #90 ceilings (rep-sd ≤
  floor, else UNINFORMATIVE → A1); CROSS-LEAKAGE < 0.005 into every
  other component in every world (the #85 zero-point family, now for
  mains); bootstrap-zero on the null world (#85b); bootstrap-stability
  of the zero under author resampling.
- **Cells.** MAINS_CERTIFIED (all clauses) / INSTRUMENT_DEFECT
  (any routing failure; A1, honest report). Certified values on the
  REAL data are then reported once (the corpus main shares with CIs —
  the first true main budget, superseding X1c's marginal-annotated
  reading as the interpretable budget; X1c's rows remain valid under
  their marginal name).
- **Leans.** Certification passes (the machinery is X1b/X1c's, twice
  certified for the interaction; the new risk is coefficient-level
  normalization consistency). Real mains: author ≈ 0.15, community ≈
  0.077 (X1c's 2×2-inversion annotations — now tested as predictions).

### Deliverables and discipline

Standard six; `scripts/run_suica_m4_xm_mains_estimator.py` + tests
(normalization consistency; leakage battery; df derivation; anchors;
#83 helper — baseline 4); gitignored `results/m4_xm_mains_estimator/`;
report `reports/SUICA_M4_XM_MAINS_ESTIMATOR_REPORT.md`; outcome
appended here; one CLAIMS_LEDGER row; ONE commit
`feat(m4-x): X-M — the mains estimator — <VERDICT>`; suite green
(1674 + new); SEED = 20260819; B_perm = 499 where applicable;
B_boot = 1000. Metadata-only; label-free; profiles never opened.

---

## X-M outcome (executor, 2026-08-19)

**Verdict: `INSTRUMENT_DEFECT`. 9 of 10 ROUTING clauses passed; the A1
stop fired; NO real-data main was computed, reported or stored.** The
descriptive family passed 3/3. Runtime 18.9 s, entirely from X1's
committed cell cache; the comments CSV was never re-streamed.

### The instrument was built, and it works

The estimator is the exact two-way FE fit per half (X1b's alternating
projections, its tightened stopping rule; the residual this leg returns is
BIT-IDENTICAL to `fe_residual`'s, contract-tested), with the coefficients
kept and identified by the pinned normalization — fit, centre the author
coefficients into the intercept, centre the community coefficients likewise.
The normalization is confluent (its two steps touch disjoint vectors) and
the normalized triple is invariant to which side the sweep starts from, both
contract-tested; the raw split genuinely moves, so the test is not vacuous.

Everything the battery could certify, it certified:

- **author main recovery** — gap +0.0002 in `{a-only}` and +0.0012 in
  `{full}`, replicate sds 0.0036 / 0.0061, both under the #90 ceiling;
- **cross-leakage, all 6 zero-point rows below 0.005** — the largest is the
  interaction's bleed into the author coefficient in `{g-only}`, 0.00395;
- **`{null}` bootstrap-zero and bootstrap-stability** — both mains' CIs cover
  0 and their own points (author 0.00016 [−0.00011, 0.00044]);
- **the interaction echo** — X1c's estimand reproduces to 0.0199 against
  0.0200 in both worlds that plant it, which certifies that this leg's fit
  IS X1c's fit.

**The #87 diagnosis is now a measurement, not an inference.** Both
estimators ran on all five worlds. The FE mains pass all 6 zero-point rows;
the MARGINAL mains fail 3 of them — `{a-only}` community 0.0263, `{b-only}`
author 0.0225, `{g-only}` author 0.0057, against the same 0.005 bound. The
marginal rows were not a conservative version of the mains; they were
contaminated, and the projection is what removes the contamination.

### The df question, answered in the negative

Derived, printed and contract-tested in-leg: **the mains carry NO
residual-df shrinkage.** They are the projection's COORDINATES, not its
residual, and the cross-half form is attenuation-free, so neither
`(P − A − C + 1)/P` nor a noise term touches them. The only interaction is
the MEAN-REMOVAL (gauge) loss, and it collapses to ONE formula for all three
readings: for a weighted population covariance with normalized weights `p`,
`factor(p) = 1/(1 − Σ p_i²)`. Realized: author 1.000273 (Σp² = 0.000273),
community size-weighted 1.023667 (Σp² = 0.023120), community unweighted
1.001001 — against X1c's interaction factor 1.171250 for contrast. Corrected
routes, raw co-reported (#67); per #87(b) treatment one, the factor is
RECOMPUTED inside every bootstrap replicate from that replicate's own
weights, and the weighted-projection equivalence is contract-tested by
building the replicated design explicitly.

### Why it stopped, and the honest shape of the stop

The single non-passing clause is **R02, the size-weighted community main's
own-recovery clause in `{b-only}` — and it did not FAIL, it read
UNINFORMATIVE.** Its gap is +0.0036, inside any tolerance; its replicate sd
is 0.0157 against the #90 ceiling of 0.0100, so the tolerance it would have
been scored at is 0.0470. A clause that claims 0.01 resolution and delivers
0.047 has measured nothing, and #90 exists to refuse that pass.

**The spread is the WORLDS, not the estimator.** Scored PAIRED — each
replicate against its own realized planted target — the same estimator on
the same worlds reads mean error −0.000045 with sd 0.000496, while the
realized target itself moves with sd 0.0150: thirty times larger.

**The cause is structural and was visible at registration.** The
size-weighted community covariance concentrates on an EFFECTIVE **43.3**
communities, not the nominal 1,000, so the realized weighted variance of an
iid component moves draw-to-draw with closed-form sd `0.08·√(2·Σp²)` =
0.0172. Over 8 replicates, P(replicate sd ≤ ceiling) = **0.267**. No seed
would have changed that; the same diagnostic gives 1.000 for the UNWEIGHTED
community clause and 0.954 for the author clause.

**A clause that PASSED must be read with the same eye.** The identical
estimand in `{full}` drew a replicate sd of 0.0072 and cleared the ceiling
with 0.0028 to spare — at that same 0.267 probability. The same clause
reading INFORMATIVE in one world and UNINFORMATIVE in another, on the same
estimand and the same skeleton, is the sharpest available demonstration that
this ceiling is being asked a question it cannot answer at this replicate
budget.

### Consequences for X3

X3's #87 prerequisite is **NOT discharged**. The author main is certified at
the registered resolution; the community main is not, in either direction —
it was neither validated nor invalidated. A future registration can discharge
this cheaply: nothing about the estimator needs changing.

### Deviations, all recorded

1. **No permutation null was run.** `B_perm = 499` is registered "where
   applicable"; no X-M clause is defined against a permutation band.
2. **`{g-only}`'s seed offset (23) is new**, taken from X1b's series before
   the run and pinned in the config; the four inherited offsets are X1b's
   (13 / 19 / 7 / 0). No seed was re-chosen after seeing a result (#76).
3. **The mean-removal factor is recomputed per bootstrap replicate** —
   #87(b) treatment one, available here because the mains' correction is a
   function of the resample's own weights.
4. **A development prototype touched the real arm.** While the estimator was
   being written, a scratchpad script (never committed, never an artifact)
   evaluated the mains on the real skeleton to size the compute. No real
   main appears in the report, the artifacts, the ledger row, or this
   outcome. Recorded because concealing it would be the worse failure.

### Defect candidates for the planner (nothing below was run)

1. **A #90 ceiling on a WORLD-LIMITED clause cannot distinguish a bad
   estimator from a bad gate.** The replicate spread of an own-recovery
   clause is estimator error PLUS world-draw variability, and on a
   size-skewed design the second term dominates by two orders of magnitude.
   Convention to consider: own-recovery clauses are scored PAIRED (each
   replicate against its own realized target) so the ceiling constrains the
   ESTIMATOR, with the nominal-target reading co-reported.
2. **Replicate counts should be DERIVED from the estimand's effective
   sample, not inherited.** `sd ≈ V·√(2·Σ p_i²)` is computable from the
   DESIGN ALONE before any world is drawn, so a registration can state the
   replicate budget that makes each clause informative — and refuse to
   register one that cannot be. Here 8 replicates would have needed to be
   about 60 for R02 to clear its ceiling reliably.
3. **A size-weighted estimand needs its effective sample published beside
   its nominal one.** A covariance over 1,000 communities whose weights
   concentrate on 43 is not a covariance over 1,000 anything, and every
   table printing the nominal count invites the wrong reading of its
   precision.

### Governance

Metadata only; `word_count_quoteless` is the sole text-derived quantity; no
body was read; `author_profiles.csv` was never opened. The ID-leak scan over
all 10,296 author names cleared with **0 NEW hits** (4 pre-existing
dictionary collisions carried unchanged from HEAD). Every X-line boundary is
carried forward, plus the one this leg adds: it is an INSTRUMENT leg, and
nothing it certifies is evidence about people or venues.

## X-M planner adjudication (2026-08-19)

**Registered outcome ACCEPTED: `INSTRUMENT_DEFECT` as routed — and the
diagnosis is unambiguous: THE ESTIMATOR IS SOUND, THE GATE CLAUSE WAS
UNDERPOWERED.** 9/10 routing clauses passed. R02 (size-weighted
community own-recovery) read UNINFORMATIVE because the size-weighted
covariance concentrates on an EFFECTIVE 43.3 communities of the
nominal 1,000: the planted target's own draw noise (closed form
0.08·√(2Σp²) = 0.0172) exceeds the #90 ceiling (0.01), so
P(8-replicate sd ≤ ceiling) = 0.267 for ANY correct estimator — and
the honest anomaly proves it: R04 passed the same ceiling on the same
estimand at that same 0.267. Paired scoring (each replicate against
its own realized target) shows estimator error sd 0.000496 — 30×
inside the target noise. No seed could have changed this; the A1 fired
on gate arithmetic, not on the instrument.

**The leg's substantive gifts, banked.** (1) The df derivation: the
mains are projection COORDINATES, not residuals — the cross-half form
is attenuation-free with NO residual-df shrinkage; the only correction
is the mean-removal gauge factor 1/(1 − Σp²) (author 1.000273,
size-weighted community 1.023667), recomputed per bootstrap replicate.
(2) FE-vs-MARGINAL leakage MEASURED: the FE mains pass all six
zero-point rows while the marginal estimators fail three ({a-only} →
community 0.0263 etc.) — #87's diagnosis is now a number, not an
inference. (3) The author main IS certified at 0.01 resolution.

**Defect #93 (planner, purchased; registry thirty-eighth note).**
Three parts, from the executor's candidates, adopted. (a) A #90
ceiling on a WORLD-LIMITED clause cannot separate a bad estimator from
a bad gate — own-recovery clauses are scored PAIRED (error against the
replicate's own realized target) with the ceiling on the PAIRED sd;
the nominal reading is co-reported. (b) Replicate counts are DERIVED
pre-registration from the estimand's effective sample (the closed-form
target sd is design-only arithmetic the planner could have run —
~60 replicates would have made R02 reliable at nominal scoring).
(c) A size-weighted estimand publishes its EFFECTIVE sample beside its
nominal one. Also recorded WITH a convention (from the executor's
disclosure 4, appreciated for its honesty): development prototypes
never touch the real arm before certification — instrument-leg dev
work runs on synthetic worlds only; the disclosed scratchpad contact
produced no artifact, report, or ledger content, and the rule now
exists so the next leg does not rely on restraint alone.

**X3 remains gated:** the author main is certified; the size-weighted
community main is neither validated nor invalidated. X-Mb below
completes it — gate arithmetic only, the estimator untouched.

---

## X-Mb — the mains estimator, paired-scored gate (registered BEFORE run, 2026-08-19)

Completion of X-M under #93; the SR4/U2c/X1c pattern. Everything
inherits X-M (skeleton, estimators, normalization, worlds, seeds,
clauses R01/R03–R10, descriptives) except:

1. **Paired scoring (#93a)** for all four own-recovery clauses: error =
   estimate − the replicate's own realized planted component; ROUTING
   ceiling = paired rep-sd ≤ 0.01 (share units, #92) and |paired mean
   error| ≤ 0.01; the nominal (world-limited) reading co-reported with
   its derived reliability P(informative) from the closed form.
2. **Effective samples (#93c)** published for every weighted estimand
   row (nominal vs effective; R02's 1,000 vs 43.3 the first entry).
3. **Replicate budget (#93b)**: 8 paired replicates suffice under
   paired scoring (paired sd ≈ 0.0005 ≪ 0.01 by X-M's own artifact);
   the derivation is printed, not assumed.
4. If MAINS_CERTIFIED: the real arm runs ONCE (the first true main
   budget, with CIs, gauge factors applied, effective samples shown)
   and the X1c predictions (author ≈ 0.15 [0.10, 0.20], community ≈
   0.077 [0.05, 0.11]) are scored.
5. Dev-prototype rule (#93 note) in force: synthetic worlds only until
   certification inside the run.

Deliverables: standard six; ONE commit
`feat(m4-x): X-Mb — the mains estimator, paired-scored — <VERDICT>`;
suite green (1742 + new); #83 baseline 4; tests use pytest tmp_path.
SEED = 20260819; B_boot = 1000.

---

## X-Mb outcome (executor, 2026-08-19)

**Verdict: `MAINS_CERTIFIED`. All 10 ROUTING clauses passed under paired
scoring, the descriptive family passed 3/3, the certificate was stamped, and
the real arm ran ONCE through the gated path — the first true main budget of
this design.** Runtime 30.4 s, entirely from X1's committed cell cache; the
comments CSV was never re-streamed.

### The repair, and exactly how much of it there was

The estimator is X-M's, imported by file and contract-tested to return X-M's
own numbers on the same world; the skeleton, the pinned normalization, the
five worlds, the seed offsets (13 / 19 / 23 / 7 / 0), the 8 replicates, the
six inherited routing clauses R05–R10 and the three descriptives D01–D03 are
unchanged, character for character where they are text and value for value
where they are numbers. What changed is the arithmetic of four rows.

Scored PAIRED — each replicate against its own realized planted component,
computed by reconstructing that replicate's drawn `a`/`b`/`g` from the
builder's own stream and applying the estimator's own functional to them —
the four own-recovery clauses read:

| clause | paired mean error | paired sd | PAIRED | nominal gap | nominal sd | NOMINAL | P(informative) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R01 `{a-only}` author | −0.000304 | 0.001084 | PASS | +0.0002 | 0.0036 | PASS | 0.954 |
| R02 `{b-only}` community | −0.000046 | 0.000508 | PASS | +0.0036 | 0.0157 | **UNINFORMATIVE** | 0.267 |
| R03 `{full}` author | +0.003965 | 0.002020 | PASS | +0.0012 | 0.0061 | PASS | 0.954 |
| R04 `{full}` community | +0.000531 | 0.000456 | PASS | −0.0084 | 0.0072 | PASS | 0.267 |

**Exactly one clause moved: R02.** Its nominal replicate sd is 30.9× its
paired sd, and that ratio is the whole of #93a — the world's draw noise being
charged to the instrument, and then not being charged to it. The other three
clauses passed both ways, which is the right shape for a repair: a gate that
changed every answer would have been a different gate.

### One clause passes with a bias, and pairing is what makes it legible

R03's paired mean error is +0.003965 against a standard error of 0.000714 —
inside the registered 0.01 resolution, therefore a PASS, but five and a half
standard errors from zero and not noise. It has a name and a measured value
in the same battery: the interaction's bleed into the author coefficient
reads 0.00395 in `{g-only}`, the largest of the six zero-point rows. The two
agree to the fifth decimal. This is the df derivation's `popvar(M_a·g)` term
appearing twice — once as leakage in a world with no author effect, once as
bias in a world that has one. Nominal scoring hid it inside a replicate
spread six times its size; paired scoring puts it on the table. It is
reported, not repaired: it is inside the resolution the registration claims.

### Effective samples (#93c), published

| row | nominal | Σp² | effective | effective/nominal | gauge factor |
| --- | --- | --- | --- | --- | --- |
| R02 / R04 community main, size-weighted (PRIMARY) | 1,000 | 0.023120 | **43.3** | 0.0433 | 1.023667 |
| R01 / R03 author main | 3,665 | 0.000273 | 3,665.0 | 1.0000 | 1.000273 |
| community main, unweighted (secondary) | 1,000 | 0.001000 | 1,000.0 | 1.0000 | 1.001001 |
| *(contrast)* X1c interaction, residual df | 31,899 | — | 27,235 | 0.8538 | 1.171250 |

### The replicate budget (#93b) — and the correction the note needed

PAIRED side, closed form. X-M's own artifact measures the paired sd on all
four clauses (0.001084 / 0.000496 / 0.002019 / 0.000445); the widest is
0.002019. With `(R−1)s²/σ² ~ χ²_{R−1}`, 8 replicates put the sd ceiling
5.0× away (χ²₇ threshold 171.7, Chernoff bound on the breach probability
10^−30.9) and the mean ceiling 14 standard errors away. The registered budget
is adequate by a margin that is arithmetic, not hope.

NOMINAL side, simulated on this skeleton at four budgets:

| replicates | mean replicate sd | P(sd ≤ ceiling) |
| --- | --- | --- |
| 8 | 0.0141 | 0.267 |
| 16 | 0.0147 | 0.136 |
| 30 | 0.0150 | 0.056 |
| 60 | 0.0152 | 0.011 |

**#93b's worked example does not hold.** The adjudication reasoned that about
60 replicates would have made R02 reliable at nominal scoring; measured, the
probability FALLS with the budget, because a sample sd concentrates on the
quantity it estimates and that quantity — the target's own draw sd, 0.0172 —
is above the ceiling. Sixty replicates would have made R02 more reliably
UNINFORMATIVE. The counterfactual would hold for a ceiling read on the
standard ERROR of the replicate mean, which shrinks as 1/√R; it does not hold
for a ceiling on the replicate SD, which is what #90 specifies. For this
clause the paired repair was not one option among several — it was the only
one that works.

### The dev-prototype rule, enforced rather than promised

The real arm is unreachable except through `run_real_arm`, which refuses a
certificate that is not `MAINS_CERTIFIED`, that carries no stamp, or whose
stamp is not already on disk. `main` contains no other call to `full_budget`
or to the cluster bootstrap, which is contract-tested against the source. The
stamp was written at `02:57:38.137176Z`; the real arm began at
`02:57:38.138163Z`, 984,917 ns later, and `certification_order.json` asserts
the order from the artifacts' own timestamps. A test drives the assertion to
FAIL on reversed stamps, so the clause is a check and not decoration.

### The first true main budget

| component | share (corrected) | 95% CI | raw (#67) | gauge factor |
| --- | --- | --- | --- | --- |
| author main `Var(a)` | **0.1286** | [0.1230, 0.1357] | 0.1286 | 1.000273 |
| community main `Var(b)`, size-weighted (PRIMARY) | **0.0552** | [0.0523, 0.0596] | 0.0539 | 1.023667 |
| community main, unweighted (secondary) | 0.0786 | [0.0772, 0.0854] | 0.0786 | 1.001001 |
| interaction (X1c's estimand) | 0.0190 | — | 0.0162 | 1.171250 |
| residual | 0.7972 | — | — | — |

Cluster bootstrap over authors, B = 1,000, the FE refitted and the
coefficients re-normalized inside every replicate, the gauge factor
recomputed from each replicate's own weights (#87(b) treatment one). X1c's
MARGINAL rows on the same skeleton read author 0.1804 and community 0.0928;
the gap between those and the mains above is the contamination the projection
removes, now measured at corpus scale rather than on synthetic worlds.

### Both registered predictions MISS, and they miss together

| prediction | value | realized | CI | result | gap |
| --- | --- | --- | --- | --- | --- |
| author main | 0.15 | 0.1286 | [0.1230, 0.1357] | **MISS** | −0.0214 |
| community main | 0.077 | 0.0552 | [0.0523, 0.0596] | **MISS** | −0.0218 |

The two gaps differ by 0.00041. X1c's 2×2 inversion was an algebraic
annotation on contaminated rows, and it over-corrected both mains by about
two variance points in the same direction — a property of the inversion, not
of either main. A prediction that breaks the same way twice says more than
one that breaks once. The three registered LEANS all HELD (certification
PASSES; author main in [0.10, 0.20]; community main in [0.05, 0.11]), which
is the honest reading of the difference between a lean and a prediction.

An observation with no claim attached: the UNWEIGHTED community secondary
reads 0.0786, which is 0.0016 from the predicted 0.077, while the registered
size-weighted PRIMARY reads 0.0552. The primary was named before the run and
is what routes and what is scored. On a design whose size-weighted community
side has an effective sample of 43.3, the two weightings are not estimating a
common number, and which one a downstream leg wants is a question that has to
be asked deliberately rather than inherited.

### Consequences for X3

**X3's #87 prerequisite is DISCHARGED.** The author main is certified at
0.01 resolution in two worlds; the size-weighted community main is now
certified in both worlds that plant it. X3 may take `Var(a)` and `Var(b)` as
estimated objects on this design, with two conditions carried forward: the
size-weighted community estimand has an effective sample of 43.3, and the
{full}-world author reading carries a known +0.004 interaction bias inside
the resolution.

### Deviations, all recorded

1. **The paired block reuses the gate's own per-replicate estimates.** X-M's
   diagnostic paired block re-drew each world and recomputed the estimator;
   this leg reads the per-replicate values the scored block already stored
   and computes only the realized target, so the pairing is bit-exact against
   the replicates the nominal clause read. The realized-component
   reconstruction is contract-tested against a noiseless world (shares
   summing to 1) where the cell means ARE the drawn components, with a
   wrong-stream-order guard so the test cannot pass vacuously.
2. **Paired errors route on the corrected scale, raw co-reported (#67).** On
   a fixed skeleton the estimate and the realized target carry the same
   constant mean-removal factor, so the corrected paired error is exactly the
   raw one times that factor; the identity is contract-tested and the routing
   decision is scale-invariant here.
3. **#93b's worked example is corrected in-leg**, by measurement, above.
4. **No permutation null was run.** `B_perm = 499` is registered "where
   applicable"; no X-Mb clause is defined against a permutation band.
5. **No development prototype touched the real arm.** The #93 note is
   enforced in code, and the ordering is asserted from the artifacts.

### Defect candidates for the planner (nothing below was run)

1. **A gate ceiling should declare which VARIANCE it bounds.** #93a fixed
   own-recovery clauses by pairing, but the general shape of the mistake is a
   clause whose statistic mixes estimator error with design-side draw noise.
   A convention could require every ceiling to name the component it
   constrains (estimator / world / both), so the arithmetic is checkable
   before the run rather than after the stop.
2. **A worked example inside an adopted defect deserves the same
   pre-registration arithmetic as a clause.** #93b's illustration was not
   derived and does not hold for the ceiling it illustrates. Either derive
   the number in the note, or mark it explicitly as an unchecked sketch.
3. **A registered PRIMARY weighting should be re-justified when its effective
   sample is small.** The size-weighted community main and its unweighted
   secondary differ by 0.023 here — more than either main's CI width — and
   the effective-sample disclosure (#93c) is what makes that difference
   readable. A convention could require a weighted primary whose effective
   sample falls below some fraction of nominal to state, at registration,
   why the weighting is the estimand of interest.

### Governance

Metadata only; `word_count_quoteless` is the sole text-derived quantity; no
body was read; `author_profiles.csv` was never opened. The ID-leak scan over
all 10,296 author names cleared with **0 NEW hits** (4 pre-existing
dictionary collisions carried unchanged from HEAD). Every X-line boundary is
carried forward, plus the two this leg adds: a certified instrument is still
only an instrument, and the gate was repaired rather than relaxed — paired
scoring makes the ceiling stricter on the estimator while removing a term the
clause was never entitled to charge it for.

## X-Mb planner adjudication (2026-08-19)

**Verdict ACCEPTED: `MAINS_CERTIFIED`** — 10/10 routing under paired
scoring (R02's paired sd 0.000508, 30.9× inside its nominal spread,
exactly as the X-M diagnosis predicted); the certification-before-real
order machine-asserted (984,917 ns); anchors exact.

**The first true main budget.** Author main **0.1286**
[0.1230, 0.1357]; community main (size-weighted primary) **0.0552**
[0.0523, 0.0596], unweighted 0.0786; against X1c's marginals
0.1804 / 0.0928. **Both registered predictions MISSED — informatively:**
X1c's 2×2 inversion over-corrected both mains by the SAME ~0.021
(gaps differ by 0.0004), so the inversion is a decent but biased
approximation; the true mains are now measured, and the marginal-vs-
main gap itself (≈ 0.05 author, ≈ 0.04 community) is the measured
composition content of the marginals. Leans held (both values inside
their intervals).

**Two honest anomalies adjudicated.** (1) R03's paired pass carries a
KNOWN bias +0.00397 = the interaction→author term (matches the
{g-only} leakage to the fifth decimal) — inside resolution, correctly
passed, and now a NAMED term future mains legs must carry. (2) The
planner's #93b worked example was WRONG: sweeping the nominal clause
shows P(informative) FALLS with replicate count (0.267 → 0.011 at 60)
because the replicate sd concentrates on the target's own draw sd —
the counterfactual holds only for a ceiling on the SE of the mean. The
#93b convention stands (budgets derived from closed forms); the
derivation must target the statistic the ceiling actually reads.
Dated correction recorded here; no new number. (3) Observation only:
the unweighted community secondary lands 0.0016 from the old
prediction while the weighted primary misses — weighting moves this
estimand more than either CI width; the primary was named pre-run and
routes.

**X3's #87 prerequisite is DISCHARGED**, carrying two conditions into
X3's design: the size-weighted community main has effective sample
43.3, and full-world author readings carry the +0.004 interaction
term inside resolution.

---

## X3 — the trait join of expression coordinates (registered BEFORE run, 2026-08-19)

The X-line's stage-E label leg: the literature's classic claims
(verbosity × personality) meet the dissociation family, on the
coordinates this line certified. SR1-class stamp machinery (U3's
implementation inherited verbatim by import-by-file); PANDORA Big5
labels re-joined ONCE; the label event recorded release-side and in
the private audit ledger at adjudication (planner-side).

### 4W header

- **Object.** Do the certified expression coordinates couple to Big5 —
  and if the LEVEL does, does the coupling survive venue adjustment
  (the composition question the atlas raises)? Crossing #2/#3 × Who.
- **Coordinates (label-free, frozen stage B; four registered).**
  (1) RAW LEVEL: person-mean y over the analysis pool (the
  literature's object); (2) ADJUSTED LEVEL: the X-Mb author-main
  coordinate â_u (venue-composition-free, certified this leg);
  (3) RHYTHM: X2's r1_u (Big5-cohort ownership 0.637); (4) R2 SLOPE:
  the gap×volume within-person slope (Big5 ownership 0.538, floored
  per #89). Reliability gate (U3's, label-free, BEFORE the stamp):
  split-half r ≥ 0.5 on the Big5 pool admits a coordinate; failures
  reported and excluded; if ALL fail the leg stops clean
  (COORDINATES_UNRELIABLE). Expectations: levels ≈ 0.9+, rhythm 0.637,
  slope 0.538 — all four expected admitted.
- **Fixed.** The Big5 pool (X2's 1,116 with per-coordinate
  eligibility); SR1's trait geometry BY FORMULA (#81: z-scored 5-dim
  Euclidean distance, the U3 pin); the bag distance (Hellinger
  full-signature cosine on the SR0-class vocabulary over this pool)
  for the increment partials; declared label reliability 0.8 (#57,
  secondary readings only).
- **What falsifies.** All admitted coordinates' raw Mantel r inside
  their own permutation bands → EXPRESSION_TRAIT_SILENT (the
  dissociation family's sixth test).
- **Layer.** P (label-bearing).

### Estimands (U3's machinery; own nulls; #79 ratio-free)

Per admitted coordinate: raw Mantel r (|c_u − c_v| vs trait distance),
permutation B = 999; partial Mantel | bag distance (SLS residual
permutation, B = 999, linearity declared per #82); activity-controlled
sensitivity. THE REGISTERED CONTRAST (routes alongside detection): if
RAW LEVEL detects, does ADJUSTED LEVEL retain it? Retention =
r_adjusted/r_raw as a descriptive with CIs on both rs (no ratio
null — #79); the CELL structure below keys on detection patterns.
Projection: minimal detectable r ≈ 0.019 at N ≈ 1,116 (z ∝ r√N from
SR1's realized 5.42 at 1,306; assumption declared).

### Cells (NULL-first #55; #75; the U3 scheme extended)

1. **EXPRESSION_TRAIT_SILENT** — no admitted coordinate detects raw
   (scoped: silent beyond r ≈ 0.019, with realized band widths).
2. **LEVEL_ONLY_COMPOSITION** — raw level detects; adjusted level does
   NOT; dynamics silent (the coupling rides venue composition — the
   program's "personality lives in Where" family extended to
   expression).
3. **LEVEL_INTRINSIC** — both levels detect (coupling survives venue
   adjustment).
4. **DYNAMICS_COUPLED** — rhythm or slope detects raw (any level
   pattern); the surprise cell.
   Bag-partial results refine any detecting cell with the
   REDUNDANT/INCREMENTAL suffix (U3's scheme). Per-trait table
   (5 traits × admitted coordinates) is SECONDARY, Bonferroni-guarded,
   never routing.

### Registered leans

- Primary: **EXPRESSION_TRAIT_SILENT to LEVEL_ONLY_COMPOSITION** —
  the dissociation family's record (five silences) against the
  literature's verbosity-E claim; if anything speaks it is the RAW
  level, and the adjusted-level contrast then decides whether the
  signal is composition (the family's prediction) or intrinsic.
- |raw r| ≤ 0.03 for every dynamics coordinate.
- The cohort caveat rides every claim (Big5-cohort ownership is
  higher on all measured coordinates — X2/X5's sevenfold flag — and
  the cohort's selection cannot be separated from it).

### Deliverables and discipline

Standard six; `scripts/run_suica_m4_x3_trait_join.py` + tests (stamp
order; gate exclusion; Mantel/SLS on toys with planted structure and
null worlds; coordinate determinism; import provenance; #83 helper
with baseline 4; pytest tmp_path only); gitignored
`results/m4_x3_trait_join/`; report
`reports/SUICA_M4_X3_TRAIT_JOIN_REPORT.md` (rule-24; stamp chain;
gate results; per-coordinate raw/partial table with bands; the
level-contrast reading; per-trait secondary table; boundaries:
metadata-only, no psychological naming of technical objects REGARDLESS
of outcome, EXPLORATORY corpus-level, no person claims, the label
event named, the cohort caveat); outcome appended here; one
CLAIMS_LEDGER row naming the label event (SR1/U3 class); ONE commit
`feat(m4-x): X3 — the trait join of expression coordinates —
<VERDICT>`; suite green (1805 + new). Stage discipline: config stamp
(no label) → stage B coordinates + gate + freeze/hash → single stage-E
join (first_join logged; stamp < freeze < join proven). A1: instrument
failures before the stamp mean no stamp. SEED = 20260819; B = 999.
The planner records the label event in the private-repo audit ledger
at adjudication.

---

## X3 outcome (executor, 2026-08-19)

**VERDICT: `EXPRESSION_TRAIT_SILENT` (cell 1, no suffix — nothing detects, so
nothing takes REDUNDANT/INCREMENTAL).** Tier EXPLORATORY, corpus-level,
layer P. Harness `scripts/run_suica_m4_x3_trait_join.py`; report
`reports/SUICA_M4_X3_TRAIT_JOIN_REPORT.md`; gitignored artifacts
`results/m4_x3_trait_join/`.

**All four coordinates were admitted, and all four are silent.** The
dissociation family's sixth test returns its sixth silence — this time
against the literature's oldest and most-cited metadata claim.

### The stamp chain

`config_stamped` < `coordinates_frozen` < `first_join`, proven from the
artifact timestamps by G-X3 and machine-asserted in the contract tests
(the proof is driven through all six order violations, the joint-quantity
lie and the label-before-stamp lie). Joint quantities before the stamp: 0.
Labels opened before the stamp: no. Both hashes match on recomputation.
`author_profiles.csv` was opened ONCE, in stage E, through U3's certified
reader, for `author` + the five Big5 columns; label completeness on the
analysis pool was **1,116 of 1,116 (1.0000)**.

A1 was honoured structurally: the whole instrument — anchors, coordinates,
gate and four value reproductions — lives in part 0, BEFORE the stamp, so
the registration's clean COORDINATES_UNRELIABLE exit could never have
produced a stamp. Stage B then re-executed the coordinates from the same
label-free sources and the split-half reliabilities came back BIT-IDENTICAL
to the ones the stamp pinned (a determinism gate that did not exist in U3).

### Anchors and the four value reproductions (all PASS)

Inherited census exact: 17,640,062 parseable rows / 10,296 authors /
1,401 Big5 / 8,895 disjoint / law vocabulary 1,443 at floor 89 /
9,124 candidates = 8,008 disjoint + **1,116 Big5** / R2 pool 7,989 + **1,100**.

The coordinates are the certifying legs' own estimators, and they prove it:

| reproduced from | committed | recomputed here |
|---|---|---|
| X2 rhythm ownership, Big5 arm | 0.6366996180212687 | identical to < 1e-9 |
| X5 R2 slope ownership, Big5 arm | 0.538039302428076 | identical to < 1e-9 |
| X5 R2 mean floored slope, Big5 arm | 0.05203279700207156 | exact |
| X-Mb certified author main, disjoint s=5 | 0.12858739914097542 | exact |
| X-Mb disjoint chain census s=5 | 3,665 / 1,000 / 31,899 | exact |

Three caches feed one coordinate table, so their author universes were
checked rather than assumed (identical name lists, identical candidate pools,
identical Big5 masks — BLOCKING).

### The reliability gate (label-free, before the stamp)

| coordinate | n eligible | split-half r | expected | gate |
|---|---|---|---|---|
| RAW LEVEL (person-mean y) | 1,116 | 0.8859 | ~0.9 | ADMIT |
| ADJUSTED LEVEL (a_hat_u) | **408** | 0.9015 | ~0.9 | ADMIT |
| RHYTHM (r1_u) | 1,116 | 0.6367 | 0.637 | ADMIT |
| R2 SLOPE (floored) | 1,100 | 0.5380 | 0.538 | ADMIT |

All four expectations landed; the rhythm and slope reliabilities ARE X2's and
X5's routing statistics, to the tenth decimal.

**THE #87 CENSUS THE REGISTRATION LEFT OPEN.** The Big5-cohort analog of the
X1c predicate chain, censused label-free in part 0 and pinned into the stamped
config: s = 3 → 446 authors / 431 communities / 3,889 shared pairs;
**s = 5 (PRIMARY, X-Mb's certified support) → 408 / 240 / 3,261**; s = 8 →
351 / 133 / 2,659. On the Big5 chain the FE budget reads author main 0.1376,
community main 0.0458, interaction 0.0177, residual 0.7989 — the author main
is close to the disjoint cohort's certified 0.1286, the community main lower.
The chain costs 63% of the pool, which is the leg's largest honest anomaly.

### The four rows

| coordinate | N | raw r | band | p | bootstrap CI | partial \| bag | + activity |
|---|---|---|---|---|---|---|---|
| RAW LEVEL | 1,116 | **+0.0006** | [-0.0207, +0.0216] | 0.948 | [-0.0199, +0.0241] | -0.0020 (p 0.866) | -0.0020 (p 0.854) |
| ADJUSTED LEVEL | 408 | **-0.0073** | [-0.0370, +0.0375] | 0.706 | [-0.0467, +0.0285] | -0.0078 (p 0.700) | -0.0079 (p 0.682) |
| RHYTHM | 1,116 | **+0.0023** | [-0.0200, +0.0242] | 0.841 | [-0.0202, +0.0259] | +0.0008 (p 0.953) | +0.0002 (p 0.985) |
| R2 SLOPE | 1,100 | **-0.0172** | [-0.0293, +0.0238] | 0.193 | [-0.0395, +0.0059] | -0.0189 (p 0.147) | -0.0197 (p 0.123) |

Every raw r is inside its own permutation band; every bootstrap CI covers
zero; the bag-partial and the activity-controlled row move nothing.

**The silence is not a dead harness.** On the SAME pairs, with the SAME trait
matrix and the SAME Mantel machinery, the Where channel is live:
Mantel(bag-distance, trait-distance) = 0.0454 on the raw level's pool
(0.0449–0.0528 across the four), against SR1's independently established
0.049 on a different pool of 1,306. A descriptive with no null of its own,
but it is the reading under which these bands mean what they say: the
instrument finds the coupling it is known to find, in these very pairs, and
finds none through any expression coordinate.

### The level contrast

The registered contrast never activated: the raw level does not detect, so
there is nothing for the adjusted level to retain. Reported anyway, with the
disclosed second reading that separates the adjustment from the chain's
eligibility loss:

| row | N | raw r | band | p | CI |
|---|---|---|---|---|---|
| RAW LEVEL, own support | 1,116 | +0.0006 | [-0.0207, +0.0216] | 0.948 | [-0.0199, +0.0241] |
| ADJUSTED LEVEL | 408 | -0.0073 | [-0.0370, +0.0375] | 0.706 | [-0.0467, +0.0285] |
| RAW LEVEL on the adjusted support (2nd reading) | 408 | -0.0080 | [-0.0364, +0.0368] | 0.659 | [-0.0455, +0.0326] |

The third row is an EXECUTOR ADDITION, disclosed and routing nothing. It
earns its place: raw and adjusted land within 0.0007 of each other ON THE SAME
408 authors, so whatever the venue adjustment does to this coordinate, it does
not move its (absent) trait coupling. The descriptive retention ratio
(-11.76) is printed and explicitly NOT read — its denominator is inside its
own band, so the ratio has no scale (#79 already forbids a null on it).

### Projection against realized power

| coordinate | N | projected mdr | realized 1.96 x null sd | ratio |
|---|---|---|---|---|
| RAW LEVEL | 1,116 | 0.0192 | 0.0218 | 1.135 |
| ADJUSTED LEVEL | 408 | 0.0317 | 0.0369 | 1.165 |
| RHYTHM | 1,116 | 0.0192 | 0.0227 | 1.187 |
| R2 SLOPE | 1,100 | 0.0193 | 0.0261 | 1.354 |

The registration's 0.019 at N ~ 1,116 reproduces as an arithmetic projection,
but the REALIZED bands are 1.14x to 1.35x wider — the `z ~ r*sqrt(N)`
transport of SR1's power to a different coordinate family is optimistic by
about a sixth to a third. **The scoped silence is therefore scoped at the
REALIZED widths, not at 0.019**: silent beyond |r| ~ 0.022 on the two
full-pool coordinates, ~0.026 on the slope and ~0.037 on the adjusted level.

### Secondary: the per-trait table

0 of 20 cells survive the Bonferroni guard (alpha = 0.00250). The three
largest, all failing it: ADJUSTED LEVEL x agreeableness 0.141 (p = 0.0042,
n = 408); RHYTHM x neuroticism -0.087 (p = 0.0035, n = 1,116); RAW LEVEL x
openness 0.086 (p = 0.0038, n = 1,116). Note what these are NOT: a marginal
Pearson against one trait and a Mantel r against the five-dimensional trait
geometry answer different questions, and only the Mantel row routes. The
agreeableness cell is the one a future leg would have to explain if it
wanted to reopen the level question — on 408 authors, at a p the guard
rejects, it explains nothing here.

### Leans

- **Primary HELD** at the null end: registered `EXPRESSION_TRAIT_SILENT to
  LEVEL_ONLY_COMPOSITION`, realized `EXPRESSION_TRAIT_SILENT`. The
  composition branch never had to be adjudicated.
- **Dynamics point HELD**: |raw r| = 0.0023 (rhythm) and 0.0172 (slope), both
  inside the registered 0.03. The slope is the leg's largest |r| and still
  well inside.
- **Cohort caveat carried on every claim** (the sevenfold Big5-ownership flag;
  the cohort's selection is inseparable from it).

### Honest anomalies

1. **The adjusted level costs two thirds of the pool** (408 of 1,116). It is
   the least powerful row in the leg and the level contrast's two registered
   rows do not stand on the same support — which is exactly why the second
   reading was added.
2. **Seven pool authors carry the zero bag signature** (no event in the 1,443
   law communities), so their bag distance is exactly 1 to everyone. Reported,
   not dropped; 7 of 1,116 cannot move a band.
3. **The retention ratio is uninterpretable** under a silent verdict and is
   printed only for completeness.
4. **The slope row is the closest thing to a signal and is still inside**
   (p = 0.193), and controlling the bag and activity moves it AWAY from zero
   (-0.0172 -> -0.0189 -> -0.0197) rather than towards it. That is the shape a
   real but sub-threshold coupling would also make. Named, not claimed.
5. **The realized bands are wider than the projection said** (ratios above).

### Governance

Metadata only; no text body read anywhere. Label event named in the config,
the report and the ledger: PANDORA Big5 re-join under an SR1-class stamp, one
open, five columns, aggregates only, no per-author trait value in any
committed file. #83 scan over the 10,296-name universe: **0 NEW hits**, 4
pre-existing dictionary collisions carried unchanged from HEAD — and the scan
is run TWICE, once at the gate and once again on the freshly written report,
so no committed file is ever scanned only in a stale version. No
psychological naming of any coordinate, as registered, regardless of outcome.

### What this closes and what it does not

Closes: crossing #2/#3 x Who on the X line's certified coordinates. The
"personality lives in Where" family is not extended to expression, because
expression does not couple in the first place — the composition question the
atlas raised cannot be answered by a coordinate that has nothing to
decompose. **The X line's own summary now reads: expression volume has strong
author ownership (X1c/X2/X4/X5) and no measured trait content.**

Does not close: the eq-12 caution stands — four first-order projections, on
this cohort, measured this way. A coordinate family with genuine trait content
(the bag channel) is live in the same pairs, so the silence is about these
coordinates and not about the pool.

## X3 planner adjudication + THE X-LINE CLOSES (2026-08-21)

**Verdict ACCEPTED: `EXPRESSION_TRAIT_SILENT` — the dissociation
family's SIXTH confirmation, now on expression coordinates.** All four
coordinates admitted (reliabilities 0.886 / 0.902 / 0.637 / 0.538,
reproducing the registered expectations to the last digit where
committed values existed) and all four silent — raw rs +0.0006 to
−0.0172, every one inside its band — while the LIVE ANCHOR carries the
weight: Mantel(bag, trait) = 0.0454 on the same pairs (SR1's 0.049
reappears under a different same-PANDORA pool/procedure; this is not an
independent-corpus replication). The registered multivariate coupling is
not detected in these coordinates. The literature's verbosity ×
personality claim finds no support in this corpus under this primary test;
the scoped non-detection is
stated at the REALIZED widths (|r| ≈ 0.022 full-pool, 0.026 slope,
0.037 adjusted — the z∝r√N projection ran optimistic 1.14–1.35×, a
dated calibration note, no number). The level contrast never activated:
the composition question dissolved — a coordinate with nothing to
decompose. The Big5-chain FE budget censused in-leg (author 0.1376,
community 0.0458, interaction 0.0177 at 408/240/3,261) sits beside the
disjoint budget (0.1286 / 0.0552 / 0.0190) — the mains transport-close
across cohorts, an observation banked for free.

**Adopted from the executor:** the INSTRUMENT-BEFORE-STAMP structure
(the entire instrument — anchors, coordinates, gate — executes in
Part 0 before the stamp, with stage B re-executing bit-identically) is
the standing stage pattern for all future label legs; A5 (the slope
row's controls moving it away from zero, the shape a sub-threshold
coupling would make) is named-not-claimed and priced for any future
higher-power look; all disclosed deviations accepted. Label event
recorded release-side (ledger row) and in the private audit ledger
(planner commit 4a303c1). Leans held at the null end. Zero new
planner defects.

**THE X-LINE CLOSES** — nine legs (X1/X1b/X1c, X2, X4, X5, X-M/X-Mb,
X3), two A1 stops with zero real-data contamination, one label event,
defects #85–#93 purchased, and the line's yield in one paragraph:
**expression volume is venue-conditioned in three layers and author-owned
in every parameter measured. Three commonness-involving projections show
the clearest cross-level transport gaps; two behavior-behavior projections
show no resolved gap at this power. None of the four registered expression
coordinates detects multivariate Big5 geometry at its realized resolution.**
The support mission's
honest conclusion for the bridge: this system offers the three-level
readout and the level-3 reliability instruments; it does not offer
trait proxies, and says so with certified silence.

## Post-closure external review — planner adjudication (2026-08-22)

Codex reviewed the X-line's closing documents and corrected the
planner's INFERENCE-SCOPE overclaims in four commits (84e7d9d,
fa8988a, b0b38c2, 2e83217). **All four are ACCEPTED IN FULL, and
defects #94/#95 — recorded by Codex against the planner's prose — are
RATIFIED as the planner's own**, the registry's first entries at the
language layer rather than the estimator layer:

- **#94 (mine):** common physical units were promoted to psychometric
  measurement invariance (units license COMMENSURABILITY only), and a
  difference between two registered linear slopes was promoted to
  process ERGODICITY (Δ_erg is retained as a historical artifact name
  but reads as a PROJECTION-SPECIFIC CROSS-LEVEL TRANSPORT GAP). X5's
  own data carried the counterwarning I under-read: R1/R4 early/late
  within-slopes disagree.
- **#95 (mine):** the projected r = 0.019 was quoted where realized
  bands (0.022–0.037, bootstrap to 0.047) belong; same-corpus reruns
  were called "independent replications" (they are ANALYTICAL
  RECURRENCES); and the registered multivariate null was generalized
  to "trait-mute everywhere" over unconfirmed univariate candidates
  up to |r| = 0.141.

**The Cross-Level Inference Contract
(`docs/SUICA_CROSS_LEVEL_INFERENCE_CONTRACT.md`) is ADOPTED as
BINDING program-wide**, including its claim gates, the
recurrence-vs-replication distinction, and the source-boundary
handling of the professor's "Level 5" (an oral-discussion ideal
limit, never attributed to 三枝・下司 2025).

**The X5-T temporal transport audit is ACCEPTED** (verdict
POST_CLOSURE_TEMPORAL_DIAGNOSTIC_COMPLETE; protocol committed before
execution; reproduction gate at ≤ 2.4e-16 on all ten X5 rows before
any multisegment estimate). Substantive reading adopted with its own
scoping: at K = 8 the disjoint cohort's within-slopes are temporally
equivalent at 0.02 for four of five relations (R1 unresolved — its
within-slope moves from −0.024 toward ≈ 0 across segments, the
concrete form of #94's counterwarning); the Big5 cohort shows
finest-resolution heterogeneity on R1/R2 with the power caveat; K
views are nested, not independent. The historical X4/X5 verdict names
stand as artifact names; their prose now reads through the contract.

Corrected prose is live in the bridge and the JPA feed (Codex's
84e7d9d); the 4W empirical state receives its correction addendum
with this commit; earlier append-only adjudication texts stand as
history and are read through this note. The planner notes with
appreciation that the review arrived through the same discipline it
audits: registered protocol, reproduction gates, one commit per unit,
and defect entries with conventions.
