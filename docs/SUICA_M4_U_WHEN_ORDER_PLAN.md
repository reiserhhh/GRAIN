# GRAIN M4-U — the When line (order, path, and slow time)

Opened 2026-08-18 by the planner, directly continuing the owner's 4W program
(`docs/SUICA_4W_PERSON_INTERPRETER_THEORY.md`) after the T-line measurement
phase closed (`docs/SUICA_M4_T_LINE_SYNTHESIS.md`, commit 2245b97).

## Line charter

The T-line answered the Where × Who crossing on **bags** (static per-author
selection distributions): identity is metric-borne, taste-dominated, and the
proposition "identity lives in the distinctive; personality lives in the
common" closed the measurement phase. Every T-line object was a static mean.

The 4W theory's §2.4 states that When studies **order mechanisms** — the
higher-order path information that "appears only in motion and vanishes in
static means" — whose general object is the author transition kernel
K_u (eq 12), of which any fixed-window statistic is only a projection. The
theory's own caution is binding here: an unstable or null projection does NOT
falsify the dynamics; it only shows that projection missed.

The U-line therefore asks, in order of technical readiness:

- **U1 (registered below)**: does the ORDER of community selections carry
  reproducible author structure beyond the bag? (fast time, event order)
- **U2 (named, not registered)**: the personal persistence curve — self-
  similarity of the selection signature as a function of calendar gap, with an
  epoch-matched cross-author baseline; fixed point vs drift vs floor.
  (slow time)
- **U3 (named, not registered)**: Who × When trait join — do order/dwell
  coordinates couple to Big5 beyond the bag channel? Label join only inside a
  stamped stage-E design; not before U1 closes label-free.

Governance: the whole line is corpus-level EXPLORATORY. U1 and U2 are
label-free (no Big5/MBTI value is read). R-G1..R-G8 bind: no person claims,
no cross-corpus author linkage, no user text or user IDs in committed
artifacts (ID-leak scan against the 1401 cohort IDs is a blocking gate),
aggregates only. The native corpus stays paused (owner 2026-07-12).

---

## U1 — order-borne selection identity (registered BEFORE run)

### 4W header (registration section, mandatory since the T-line)

- **Which 4W object.** When — the first-order projection of the author
  transition kernel K_u (theory eq 12) over coarse Where states, plus dwell
  (stay) statistics. The claim target is crossing #4 (Who × When): whether
  order carries author structure that the Where bag cannot see. Identity
  information only; no psychological naming (§5.4 binds: technical object).
- **What is fixed / what is varied.** Fixed: SR0 cohort and vocabulary
  (1191 communities at floor 15, reproduction bit-verified 2026-08-18);
  per-author median-timestamp early/late halves (T6'' pattern); Hellinger
  conventions; 5-fold author cross-fitting with per-fold training-early-only
  state maps. Varied: real event order vs within-half order-shuffled events
  (the exact-bag null); state resolution C ∈ {12, 24, 48}; raw vs
  session-restricted vs cross-thread adjacency; full vs
  clean_no_explicit_personality vocabulary.
- **What falsifies.** If the order excess ρ (defined below) is
  indistinguishable from its own shuffle null in the primary arm at every
  registered C, the verdict is NO_ORDER_CHANNEL **for this projection** —
  explicitly not a falsification of eq-12 dynamics (the theory's projection
  caution is quoted into the verdict whatever it is).
- **V/R/P layer.** R (representation/measurement) for the real-data claim;
  V for the synthetic authentication worlds.

### Question and estimand

Within-author event order is destroyed by shuffling each half's event
sequence while preserving that half's bag EXACTLY. Any drop in same-author
discrimination from order-sensitive features is therefore pure order
information. The composite discrimination statistic states its own null
(convention #68): the shuffle null does NOT sit at 0.5 — shuffled bigram
features still discriminate authors through their marginals, so the null
walks near the bag ceiling. The registered primary effect size is therefore
ceiling-aware:

    rho = (AUC_real − mean AUC_shuffle) / (1 − mean AUC_shuffle)

the fraction of the discrimination error remaining at the bag-implied null
that real order removes. Raw excess AUC is reported alongside; cells key on
rho (convention #75; effect size primary, detection secondary).

### Census (planner arithmetic, convention #43, run 2026-08-18)

Streamed 17,640,062 comment rows restricted to the 1401-author SR0 cohort
(no bodies read; scratchpad only). Vocabulary reproduction: floor
ceil(0.01·1401) = 15 → **1191 communities — bit-identical to SR0**.

| quantity | value |
|---|---|
| cohort authors with events | 1401 |
| vocabulary coverage of event stream | 0.7814 (OOV share 0.2186) |
| per-author vocab events, median (q25) | 379 (91) |
| early-half vocab events, median | 188 |
| late-half vocab events, median | 185 |
| adjacent-pair timestamp tie rate | 0.1152 |
| adjacent gaps ≤ 1 h | 0.6609 |
| adjacent gaps ≤ 24 h | 0.9207 |
| median of author-median gap | 2.83 h |
| calendar span, median (q25–q75) | 1157 d (642–1536) |
| pool: both halves ≥ 50 vocab events | **984** |
| pool: both halves ≥ 100 vocab events | 821 |
| pool: both halves ≥ 30 vocab events | 1100 |

Registered pools (convention #69 — pool sizes, not caliper widths): primary
pool = **984 authors** (≥ 50 vocab events in each half); sensitivity pool =
821 (≥ 100). The SR0 split-half rule (≥ 40 total) gave N=1304; U1's stricter
per-half floor is required because bigram features live in a squared space.

### Design

**Event stream.** Columns author, subreddit, created_utc, link_id only.
Events sorted per author by created_utc with a STABLE sort; ties (11.52% of
adjacent pairs) keep stream order — pinned; ties attenuate real order signal
and cannot create it (the shuffle null re-randomizes ties too). Halves by the
author's own median timestamp (SR0 rule, boundary events to early on ≤).

**States.** Per fold, communities are clustered into C states using ONLY
training authors' early halves: spherical k-means (cosine) on √(author-count)
community columns, C primary 24, sensitivity {12, 48} (endpoints named
against vocabulary size 1191, convention #74), seed SEED+1000·fold, n_init
10. Out-of-vocabulary events map to one extra OOV state (index C), so
adjacency is real — no splicing. Communities with zero training-early mass
also map to OOV. State count C+1; feature dimension (C+1)².

**Features (transforms pinned with the verdict, convention #70).**
Primary: the bigram Hellinger sphere — joint next-event counts
n_u(i, j) + α smoothing (α = 0.5) → joint probabilities P_u(i, j) →
elementwise √ → L2 normalization. This is the occupancy-weighted conditional
representation (√(P(i)·P(j|i))), exactly parallel to the T-line's unigram
Hellinger sphere. Secondary lens: unweighted conditional rows
(√ of row-normalized P(j|i), concatenated) — upweights rare rows.
Scalar arm: stay rate s_u = P(next state = current state), scored by
−|s_u^early − s_v^late|; its shuffle null is the bag-concentration pseudo-stay
Σ_j π_j² and is stated as such (#68).

**Discrimination.** Pooled same-author-vs-different AUC on cosine between
early features of u and late features of v, computed within each fold's test
authors under that fold's state map (purity: zero test-author mass in any
map — asserted), then unweighted mean over the 5 fold AUCs.

**The exact-bag null.** B = 499 within-half order shuffles (each half's event
sequence permuted independently, per author, batched and vectorized — the
T2 paid lesson binds: no per-user Python loops × B). Contract test: the
per-half bag (bincount over states) is EXACTLY invariant under the shuffle.
Null band = 2.5/97.5 percentiles of the B pooled AUCs; the null's location is
reported as the bag-implied pseudo-order AUC (#68). Power note (convention
#66): the null-sd method is the empirical percentile over B = 499; the
minimal detectable rho implied by the realized band width is reported, and
any equivalence statement carries the band-width-vs-ε projection (#71,
ε = 0.05 on rho).

**Uncertainty.** Cluster bootstrap over authors, B = 1000, for the CI on rho
(bands carry uncertainty; bias corrections, if any, correct predictions —
convention #67 dual stamping n/a here, no G2 object).

**Arms.**

| arm | role |
|---|---|
| raw adjacency, full vocab, C=24 | PRIMARY (verdict) |
| session-restricted (gap ≤ 3600 s) | secondary — fast-path order |
| cross-thread only (link_id differs) | co-primary decomposition — free-selection order with reply-chain mechanics removed (the theory's opportunity-structure control for crossing #4) |
| C ∈ {12, 48} | sensitivity |
| conditional-rows lens | secondary lens |
| stay-rate scalar | secondary channel |
| ≥100-events pool (821) | sensitivity |
| clean_no_explicit_personality (23 names removed, restated map) | governance echo; divergence from full arm flagged per #73 |

Descriptives registered as non-verdict-moving: same-thread share of
same-state stays (reply-chain mechanics share); OOV state occupancy;
realized transitions per author.

### Synthetic authentication (Part 0, gate before the real arm)

Three worlds, N = 900 synthetic authors, event counts drawn to match the
census median (379), C_true = 24:

- **W_null** — each author draws events iid from an author-specific Dirichlet
  bag. Target: rho CI includes 0 and |rho| < 0.05 (estimator honesty — must
  NOT detect).
- **W_sticky** — stay probability s_u ∈ {0.1, 0.25, 0.5} (3 knob points),
  otherwise iid from the bag. The chain P(i→j) = s·1[i=j] + (1−s)·π_j has
  stationary distribution EXACTLY π (identity proven in-registration:
  Σ_i π_i P(i→j) = s·π_j + (1−s)·π_j = π_j). Target: rho > 0 detected,
  monotone in s.
- **W_transition** — author-specific Metropolis-Hastings chains with target
  stationary π_u and author-specific proposal kernels (block-structured
  preferences); MH guarantees the stationary exactly. Contract test: empirical
  stationary within tolerance; acceptance formula unit-tested. Target:
  rho > 0 detected, monotone in the proposal-structure strength (3 points).

A world where the shuffle-null location fails to sit near the bag AUC fails
the gate (null mechanics broken). Part 0 must pass before the real arm runs;
a failed gate STOPS the leg before any real-data stamp (A1 hardening binds).

### Classification cells (NULL-first, convention #55; effect-size keyed, #75)

On the PRIMARY arm's rho with 95% cluster-bootstrap CI:

1. **NO_ORDER_CHANNEL** — CI lower ≤ 0 in the primary arm at every C; if
   additionally CI upper < 0.05, the scoped statement "no order channel
   beyond rho 0.05 in this projection" attaches with the band-width
   projection (#71).
2. **ORDER_TRACE** — CI lower > 0 and point rho < 0.10.
3. **ORDER_CHANNEL** — point rho ∈ [0.10, 0.33].
4. **ORDER_MAJOR** — point rho > 0.33 (order removes over a third of the
   remaining discrimination error).

CI-boundary straddles are reported as straddles; the verdict then takes the
interval statement, not the point (T-line #75 lesson). Full-vs-clean or
raw-vs-cross-thread divergence takes a #73 flag: primary arm routes the
verdict; divergences are named, never averaged away.

### Registered leans (before run)

- Primary lean: an order channel EXISTS but is small —
  rho in (0.02, 0.15], i.e., ORDER_TRACE to lower ORDER_CHANNEL. Rationale:
  66% of adjacencies are within 1 h (session bursts are real and plausibly
  author-stable), but the T-line showed the bag already carries identity to
  0.9837, and the dissociation lesson warns against expecting new channels to
  be large.
- Secondary lean: the stay-rate scalar alone detects (its own rho > 0) —
  burstiness is author-owned.
- Decomposition lean: the cross-thread arm retains most of the primary rho
  (free-selection order, not reply mechanics, carries the channel) — held
  weakly; a mechanics-dominated result is the informative surprise.

### Deliverables (standard six) and discipline

Script `scripts/run_suica_m4_u1_order_identity.py` + tests
`tests/test_m4_u1_order_identity.py`; gitignored
`results/m4_u1_order_identity/`; report
`reports/SUICA_M4_U1_ORDER_IDENTITY_REPORT.md` with rule-24 generated tables;
outcome appended to this plan doc; one CLAIMS_LEDGER.md row (EXPLORATORY,
label-free, corpus-level); exactly ONE commit
`feat(m4-u): U1 — order-borne selection identity — <VERDICT>`, never amended,
never pushed by the executor. Suite must be green (1021 + new) before commit.
Blocking gates: fold-purity assertion, exact-bag shuffle invariance test,
MH stationarity contract, ID-leak scan (0 of 1401 cohort IDs in any committed
artifact), timestamps in prose verified against artifacts.

Config pins: SEED = 20260818; folds = 5 (KFold shuffle=True, random_state =
SEED); per-fold cluster seed SEED+1000·fold; α = 0.5; B_shuffle = 499;
B_bootstrap = 1000; C primary 24; pools 984/821; data =
`/Volumes/mobile3/projects/project persona/data_sets/PANDORA_official/all_comments_since_2015.csv`
(columns author, subreddit, created_utc, link_id only; no bodies);
cohort = `results/m4_sr0_recon/cohort_authors.csv`; vocabulary reconstruction
via the T1 pattern (import by file, provenance stated).

---

## U1 outcome (executor, 2026-08-18)

Append-only. The registration text above is unchanged. Artifact stamps:
Part 0 completed `2026-08-17T20:48:42Z`, real-arm window
`2026-08-17T20:48:44Z` – `2026-08-17T21:09:02Z` (UTC; local date 2026-08-18).
Report: `reports/SUICA_M4_U1_ORDER_IDENTITY_REPORT.md`. Harness:
`scripts/run_suica_m4_u1_order_identity.py`. Tests:
`tests/test_m4_u1_order_identity.py`.

### Verdict

**`ORDER_CHANNEL`** (cell 3). Primary arm rho = **0.2893**, 95%
cluster-bootstrap CI **[0.2695, 0.3114]**, raw excess AUC 0.0175
[0.0152, 0.0195], permutation p = 0.0020 over B = 499. The CI straddles no
cell boundary, so point and interval agree and no interval-statement
substitution is needed. No arm diverges in cell membership, so **no #73 flag
is raised anywhere in this leg**.

The theory's projection caution is carried into the verdict as registered:
this is one first-order projection of K_u over coarse Where states, and a
positive reading here is not a claim about K_u itself.

### The null's own location (#68)

The null does not sit at 0.5 and did not: the primary shuffle null is
**0.9396** (band [0.9393, 0.9399], sd 0.00015) against a same-folds,
same-states unigram bag AUC of **0.9607** — a gap of 0.0211, inside the
registered 0.05 tolerance. The mechanism is an identity, not an accident:
for product-form counts the bigram Hellinger cosine is the SQUARE of the
unigram cosine, a monotone map, so the shuffled features inherit the bag's
ranking and are degraded only by the finite-sample noise of L−1 pairs.
Realized minimal detectable rho from the band: **0.0050**, an order of
magnitude inside the registered epsilon of 0.05.

### The reading the estimand forces

**Real AUC 0.9571 sits 0.0036 BELOW the static bag AUC 0.9607.** Order
recovers nearly all of the bigram representation's sparsity cost but does
not surpass the bag. rho is an order statement INSIDE this projection; it is
not a claim that order out-identifies the static selection bag. Any prose
that upgrades U1 to "order beats the bag" is wrong.

### Decomposition (all arms scored against their own exact-bag nulls)

| arm | rho | 95% CI | share of primary |
|---|---|---|---|
| PRIMARY raw adjacency, C=24, full vocab, N=984 | 0.2893 | [0.2695, 0.3114] | 100% |
| CO-PRIMARY cross-thread only | 0.1626 | [0.1373, 0.1895] | 56.2% |
| session-restricted (<= 3600 s) | 0.2986 | [0.2802, 0.3205] | 103.2% |
| C = 12 | 0.2269 | [0.2046, 0.2502] | 78.4% |
| C = 48 | 0.2430 | [0.2283, 0.2598] | 84.0% |
| conditional-rows lens | 0.1017 | [0.0545, 0.1473] | 35.2% |
| stay-rate scalar | 0.1803 | [0.1375, 0.2234] | 62.3% |
| pool >= 100 (N=821) | 0.2918 | [0.2687, 0.3150] | 100.8% |
| clean_no_explicit_personality (N=888) | 0.2666 | [0.2441, 0.2902] | 92.2% |

- **Dwell-dominated.** 71.2% of adjacent pairs are same-state; realized stay
  rate 0.6734 against a bag-implied pseudo-stay of 0.4086. ONE SCALAR per
  half recovers 62.3% of the channel and identifies authors at AUC 0.7388
  against its own null of 0.6814.
- **Reply-chain mechanics are a large minority, not a majority.** Removing
  within-thread adjacency costs 43.8% of rho; 53.1% of same-state stays sit
  inside one thread. The registered decomposition lean holds, but only at
  56% — "free-selection order" must never be stated without this number.
- **Fast time carries it.** Within-hour adjacency (67.0% of pairs) reads
  103.2% of the primary rho.
- **Occupancy weighting is doing the work.** The unweighted conditional-rows
  lens collapses to 0.1017, its CI straddling the 0.10 boundary. The channel
  lives in the dominant states' dwell-and-switch mass, not in rare
  transitions.
- **Governance contrast worth carrying forward.** The 23-community ablation
  costs 7.8% of rho here, against 41% of the S-line's trait coupling.

### Registered leans

- Primary lean rho in (0.02, 0.15]: **EXCEEDED** — 0.2893, ~1.9x the top of
  the leaned interval, CI entirely above it.
- Secondary lean (stay-rate scalar detects alone): **HELD**, 0.1803.
- Decomposition lean (cross-thread retains most): **HELD**, 56.2%.

### Part 0

**PASS**, before any real arm. W_null across 8 replicate worlds: mean rho
5.07e-05, 95% t-interval [-0.0108, 0.0109], between-replicate sd 0.0133,
every replicate inside |rho| < 0.05. W_sticky 0.0565 / 0.1847 / 0.4305 and
W_transition 0.5498 / 0.7853 / 0.9716 — both positive and monotone. MH
stationary drift 2.8e-17, detailed-balance violation 1.7e-18. Exact-bag
invariance held in every world and every real arm.

### Disclosures

- **A-U1-1 (pre-verdict, synthetic only).** The first synthetic
  parameterisation (Dirichlet 0.4) saturated at bag AUC 0.99999, leaving no
  discrimination error for order to remove; the gate FAILED and stopped the
  leg exactly as A1 requires. The concentration was re-set to Dirichlet(4.5)
  — the grid point putting the synthetic bag AUC nearest a declared 0.96
  target — and the gate re-run. No real arm ran before the gate passed.
- **RD-U1-1.** The W_null target's two clauses ("CI includes 0" AND
  "|rho| < 0.05") cannot both be evaluated on ONE draw at the pinned N = 900:
  the bootstrap CI half-width (~0.018) is the size of rho's own between-draw
  scatter (0.0133), and the first seed missed zero by 0.0008 while sitting
  2.7x inside the magnitude clause. Resolved by making the test STRICTER:
  8 replicate worlds, magnitude required of every one, "must not detect"
  decided on the across-replicate mean. Both clauses survive in substance.
- **RD-U1-2.** The synthetic null-location check conflates a broken null with
  an under-sampled world. The null−bag gap is the finite-sample penalty of
  estimating a (C+1)^2-cell object from L−1 pairs; at the registration's own
  pinned synthetic event count that is 0.33 pairs per cell, against 2.4 at
  the real arm's budget. Split: synthetic worlds must retain >= 40% of the
  bag's lift over 0.5 (all did, none near the threshold), and the literal
  |null − bag| <= 0.05 check is enforced on the real primary arm, where it
  passes at 0.0211.

### Census and gates

Vocabulary reproduced bit-identically (17,640,062 rows streamed, 1401 authors,
floor 15, **1191 communities**); pools **984 / 821** exactly as censused; 23
typology communities removed for the clean arm (re-derived pool 888).
Adjacent-pair tie rate reproduces the census at **0.11522** on in-vocabulary
events (0.10760 over all events — the census value is the in-vocabulary one).
Blocking gates all PASS: Part 0, exact-bag invariance, fold purity, vocabulary
and pool reproduction, real-arm null location, ID-leak scan (0 of 1401 cohort
IDs in any committed file). Binned-vs-exact AUC error 1.6e-05.

### What this hands U2 and U3

U2 (persistence curve) inherits a live, resolution-stable order channel and
should expect dwell to dominate its slow-time analogue. U3 must not open
labels against the primary arm as if it were free-selection order: the
mechanics-controlled cross-thread arm (rho 0.1626) is the honest object for
any Who x When trait join, and the stay-rate scalar is a cheap, already-
validated single coordinate to carry into a stage-E design.

## U1 planner adjudication (2026-08-18)

**Verdict ACCEPTED: `ORDER_CHANNEL`** — rho 0.2893 [0.2695, 0.3114],
p = 0.0020, all eight arms in the same cell, zero #73 flags, gate passed
before any real arm with the A1 stop demonstrably live (it fired on the
saturated first world). The CI straddles no boundary; the conditional-rows
lens straddle (0.0545–0.1473 across 0.10) is a lens-level note, not a
verdict-level one, and is carried as registered.

**The scoping is adopted as binding.** Real bigram AUC 0.9571 sits 0.0036
BELOW the same-folds unigram bag AUC 0.9607: order is REPRODUCIBLE but, at
this resolution, REDUNDANT for absolute identity — order recovers nearly all
of the bigram representation's sparsity cost and does not surpass the bag.
The sentence "order beats the bag" is forbidden U1 prose repo-wide; rho is a
within-projection statement. The 4W theory's projection caution binds the
positive result exactly as it would have bound a null one: this is ONE
first-order projection of K_u (eq 12), and ORDER_CHANNEL is a statement about
that projection.

**Theory reading (first positive When projection).** The channel is
dwell-dominated: 71.2% of adjacent pairs are same-state, realized stay 0.6734
against the bag-implied pseudo-stay 0.4086, and ONE scalar per half (the stay
rate) recovers 62.3% of the primary rho. This is the first label-free
measurement of the theory's crossing-#4 candidate "inertia" — kept strictly
technical per §5.4: dwell is an author-owned selection-process statistic, not
a psychological attribute. Mechanics decomposition honored: reply-chain
mechanics carry 43.8% of the channel (and 53.1% of same-state stays sit
inside one thread), so the executor's pre-scoping is ADOPTED AS BINDING:
**the cross-thread arm (rho 0.1626) is the canonical order object for U3's
Who × When trait join**; the primary arm may never feed a label join as if it
were free-selection order.

**Governance contrast recorded.** The 23-community typology ablation costs
7.8% of order-identity rho against 41% of the S-line's trait coupling on the
same ablation — the third instance of the identity-robust/trait-fragile
asymmetry (T-line dissociation family). Identity keeps not caring about the
criterion-adjacent communities; trait coupling keeps depending on them.

**Lean record.** The primary lean (0.02, 0.15] was EXCEEDED ~1.9× with the
CI entirely above its top — the order channel is materially larger than the
planner leaned; the stay-rate and cross-thread leans held. Recorded as a
prediction miss in the honest direction (underclaim), no repair needed.

**Defect #76 (planner, purchased; registry twenty-fifth note).** The
synthetic worlds were registered by MECHANISM with no OPERATING POINT: the
Dirichlet concentration was unpinned, and the first parameterisation
saturated at bag AUC 0.99999, making the registered W_sticky target
unreachable by ANY correct estimator; separately, the registered synthetic
null-location check was ill-posed at the worlds' own sampling density
(0.33 pairs/cell against 2.4 real). The gate architecture caught both — an
A1 stop and two disclosed re-posings (RD-U1-1, RD-U1-2), each STRICTER than
the registered form — but the registration should not have needed catching.
Convention purchased: (a) every synthetic authentication world registers a
target operating point (a realized discrimination density, e.g. target bag
AUC) on a declared grid, not just a mechanism; (b) null-location gate checks
are split — a mechanics-retention bound on the synthetic side, the literal
location bound on the real arm. Minor precision note (unnumbered): census
statistics pin their denominators (the 0.1152 tie rate is the in-vocabulary
denominator; all-events gives 0.1076).

**Named-open (not queued).**
- The non-dwell residual of the channel (37.7% of rho): transition structure
  proper vs higher-order dwell — a carrier refinement, priced by the T-line
  lesson that carrier chains go deep.
- The absolute-increment question (combined unigram+bigram representation):
  expected ≈ 0 given the redundancy anomaly; low priority.
- Finer-state order (25 states is coarse; an order channel below this
  resolution is invisible here by construction).

**Line state.** U1 CLOSED. U2 (the persistence curve — slow time) is next and
inherits: the live order channel, dwell dominance as its expected slow-time
analogue, and the census (spans median 1157 d). U3 remains gated on U2 and on
the cross-thread designation above.

---

## U2 — the personal persistence curve (registered BEFORE run, 2026-08-18)

### 4W header

- **Which 4W object.** When, slow time — the persistence of the Where
  signature: how the marginal selection distribution π_u holds or drifts over
  calendar time. A slow-time projection of K_u (eq 12); §2.4's "state
  durations and change points" at the coarsest scale. Identity structure
  only; no psychological naming (§5.4).
- **What is fixed / what is varied.** Fixed: SR0 vocabulary (1191);
  Hellinger unigram sphere (√freq, L2 — the T-line's identity-carrying
  metric); blocks of exactly K = 50 consecutive in-vocabulary events
  (disjoint, no sliding), so measurement noise is constant by construction.
  Varied: the calendar gap Δt between block midpoints, binned; personal vs
  epoch-matched cross-author pairing.
- **What falsifies.** A flat personal-excess curve (decay contrast D
  indistinguishable from its permutation null AND inside the equivalence
  band) yields FIXED_POINT — a statement that drift is undetected at this
  span and power, for this projection; NOT "no dynamics" (the eq-12
  projection caution is quoted into any verdict).
- **V/R/P layer.** R. No synthetic gate is required: the estimator is a
  binned mean contrast with a permutation null, not a fitted discriminator;
  its honesty checks are the permutation null itself plus contract tests
  (defect #76's operating-point convention does not bind — no world is
  simulated; where a functional form appears it is a refinement, never the
  verdict carrier).

### Question and estimand

Is the personal selection signature a standing wave or a moving one? For
authors' disjoint K-event blocks, define per gap bin b:

    E(b) = mean cos(h_i, h_j | same author, gap in b)
         − mean cos(h_i, h_j | different authors, epoch-matched, gap in b)

where h is the block's Hellinger vector and the cross term matches the
self-pairs' joint (quarter_i, quarter_j) histogram — absorbing global
platform drift (communities rising and dying) so E is PERSONAL persistence
in excess of shared-epoch similarity. Primary contrasts: existence
E(0–90d) > 0; decay D = E(0–90d) − E(2–3y). The endpoint is registered
against the available span (convention #74): 2–3y is the last bin with deep
support; 3y+ is a descriptive extension, never the verdict endpoint.

### Census (planner arithmetic, #43, run 2026-08-18 on the U1 events cache)

Cache: `results/m4_u1_order_identity/events_cache.npz` — 3,005,360 cohort
events, built by the committed U1 runner from the canonical CSV under
contract tests; the U2 runner must re-verify its anchors (event count,
1401 authors, vocabulary 1191) before use.

| quantity | value |
|---|---|
| authors with ≥ 4 blocks (K=50) | **849** (primary pool, #69) |
| authors with ≥ 2 / ≥ 8 blocks | 1028 / 690 |
| total blocks | 46,318 over 18 calendar quarters |
| self pairs 0–90d / 90–180d / 180–365d | 1,005,742 / 783,654 / 1,198,561 |
| self pairs 1–2y / 2–3y / 3y+ | 1,248,992 / 417,963 / 100,150 |
| authors contributing to 2–3y / 3y+ | 564 / 332 |
| min cross candidates per self quarter-pair cell | 115.4× (epoch matching feasible in every bin) |
| activity terciles (pool sizes) | 302 / 265 / 282 (edges 650 / 2050 vocab events) |

### Design pins (#70)

- Blocks: K = 50 consecutive in-vocabulary events, disjoint, stream-ordered
  (stable sort, ties keep stream order — U1's pin); trailing remainder
  dropped; block midpoint = mean of first and last event timestamps.
- Features: √(count/K) over the 1191 vocabulary, L2-normalized. No OOV state
  (bags are vocab-restricted — T-line convention; coverage 78.14% named).
- Bins: {0–90d, 90–180d, 180–365d, 1–2y, 2–3y} verdict bins + 3y+
  descriptive.
- Cross baseline: per bin, different-author block pairs sampled to match the
  self-pairs' joint quarter-pair histogram, up to 20 cross pairs per self
  pair per cell (min feasibility 115× censused); quarter = 91.3 days from
  the corpus minimum timestamp.
- Own null (#68, #66): B = 499 permutations reassigning block→author labels
  within each calendar quarter (epoch structure preserved exactly, identity
  destroyed); E(b) and D recomputed per permutation; band = 2.5/97.5
  percentiles; E's expected null location is 0 BY CONSTRUCTION of the
  contrast — stated, and verified by the null's realized center.
- Uncertainty: cluster bootstrap over authors, B = 1000, CIs on E(b), D, and
  the floor share E(2–3y)/E(0–90d).
- Attenuation: constant across bins by the fixed-K construction; E is an
  attenuated level (reliability < 1 at K = 50), D and the floor share are
  the transportable quantities; adjacent-block similarity is reported as the
  reliability descriptive.
- Refinement (not verdict-carrying): if D is detected, fit
  E(Δt) = E_inf + A·exp(−Δt/τ) on bin-midpoint means (cluster-bootstrap CI;
  τ capped at 10 y, cap-hit reported).

### Arms

| arm | role |
|---|---|
| full vocab, pool 849, verdict bins | PRIMARY |
| balanced panel (564 authors with 2–3y support) | sensitivity — composition shift control |
| activity terciles (302/265/282) | secondary — decay shape by activity |
| clean_no_explicit_personality (23 removed, blocks recomputed) | governance echo; #73 flag on divergence |
| 3y+ extension | descriptive only |
| K = 100 blocks | sensitivity — block-size stability |

### Classification cells (NULL-first #55, effect-size keyed #75)

1. **NO_PERSONAL_PERSISTENCE** — E(0–90d) CI includes 0 or is inside the
   permutation band. (Would contradict the T-line's early/late AUC; the cell
   must exist regardless.)
2. **FIXED_POINT** — E(0–90d) > 0; D CI includes 0 AND |D| < 0.2·E(0–90d)
   (equivalence band; its achievable width vs the realized D band is
   reported per #71).
3. **DRIFT_WITH_CORE** — D > 0 outside its null band and E(2–3y) CI > 0.
4. **FULL_DRIFT** — D > 0 and E(2–3y) CI includes 0.

Straddles reported as straddles; the verdict takes the interval statement.
Full-vs-clean and primary-vs-balanced divergences carry #73 flags; the
primary arm routes.

### Registered leans

- Primary lean: **DRIFT_WITH_CORE with a large floor** — floor share
  E(2–3y)/E(0–90d) in [0.6, 0.9]. Rationale: the T-line's early-to-late
  AUC 0.9837 over median 3.2-year spans and the S-line's split-half 0.681
  demand a large permanent core; personal taste drift over 3 years is
  plausible but epoch matching already absorbs the platform's own drift.
- Secondary lean: the decay, if detected, is concentrated in the low-activity
  tercile (thin signatures drift-or-noise more); held weakly.

### Deliverables and discipline

Standard six: `scripts/run_suica_m4_u2_persistence_curve.py` +
`tests/test_m4_u2_persistence_curve.py`; gitignored
`results/m4_u2_persistence_curve/`; report
`reports/SUICA_M4_U2_PERSISTENCE_CURVE_REPORT.md` (rule-24 generated tables;
own-null locations; the eq-12 projection caution in the verdict; config
block); outcome appended here; one CLAIMS_LEDGER row (EXPLORATORY,
label-free, corpus-level); exactly ONE commit
`feat(m4-u): U2 — the personal persistence curve — <VERDICT>`, never
amended, never pushed by the executor. Suite green (1060 + new) before
commit. Blocking gates: cache anchor verification, permutation-null center
check, ID-leak scan (0 of 1401 cohort IDs in committed files), exact-K block
invariants. SEED = 20260818; B_perm = 499; B_boot = 1000. Label-free: no
Big5/MBTI value is read.
---

## U2 outcome (executor, 2026-08-18)

Append-only. The registration text above is unchanged. Artifact stamps: run
window `2026-08-17T21:48:30.732133Z` – `2026-08-17T21:50:40.015368Z` (UTC; local date 2026-08-18). Report:
`reports/SUICA_M4_U2_PERSISTENCE_CURVE_REPORT.md`. Harness:
`scripts/run_suica_m4_u2_persistence_curve.py`. Tests:
`tests/test_m4_u2_persistence_curve.py`. Artifacts (gitignored):
`results/m4_u2_persistence_curve/`.

### Verdict

**`DRIFT_WITH_CORE`** (cell 3). The personal selection signature is a
**moving wave with a standing core**: it drifts, and it does not drift away.

| quantity | value | 95% CI | own null band |
|---|---|---|---|
| E(0–90d) — existence | **0.6573** | [0.5973, 0.7228] | [−0.00088, 0.00090] |
| E(2–3y) — verdict endpoint (#74) | **0.3515** | [0.2708, 0.4270] | [−0.00143, 0.00168] |
| D = E(0–90d) − E(2–3y) | **0.3058** | [0.2364, 0.3855] | [−0.00173, 0.00164] |
| floor share E(2–3y)/E(0–90d) | **0.5348** | [0.4203, 0.6320] | — |

Permutation p = 0.0020 (the floor at B = 499) for both existence and decay.
The intervals straddle no cell boundary, so point and interval agree and no
interval-statement substitution is needed. **All seven arms land in
`DRIFT_WITH_CORE`; zero #73 flags.**

The registered projection caution is carried into the verdict: this is one
SLOW-TIME projection of K_u (eq 12) onto the marginal selection distribution
π_u over 1191 communities. The decay is a statement about π_u on the
Hellinger unigram sphere over calendar time, not about any psychological
attribute (§5.4).

### The curve

| bin | self mean cos | epoch-matched cross | E(b) | 95% CI |
|---|---|---|---|---|
| 0–90d | 0.7105 | 0.0532 | 0.6573 | [0.5973, 0.7228] |
| 90–180d | 0.6487 | 0.0516 | 0.5971 | [0.5318, 0.6665] |
| 180–365d | 0.5950 | 0.0503 | 0.5447 | [0.4727, 0.6163] |
| 1–2y | 0.5033 | 0.0479 | 0.4554 | [0.3784, 0.5274] |
| 2–3y | 0.3959 | 0.0444 | 0.3515 | [0.2708, 0.4270] |
| 3y+ *(descriptive)* | 0.3091 | 0.0453 | 0.2638 | [0.1869, 0.3339] |

Epoch matching is doing visible work: the cross floor itself falls from
0.0532 to 0.0444 across the bins, and that platform-wide decline is
subtracted out before E is read.

### The null's own location (#68)

The registration's claim — E's expected null location is 0 BY CONSTRUCTION —
is an argument, and it was checked as a number. Realized within-quarter
permutation centers, all six bins: **−1.5e−05, +2.4e−05, −3.7e−05, −3.7e−05,
+1.6e−05, +8.1e−05**; D's center **−1.9e−05**. Every one is three to four
orders of magnitude inside its own 95% band and four orders inside the
effect. The null destroys identity while preserving each author's
per-quarter block count exactly, so the epoch structure the cross baseline
matches on is untouched.

### The equivalence band could have spoken (#71)

The FIXED_POINT margin is 0.2 × E(0–90d) = **0.1315**; the realized
half-width of D's CI is **0.0745**, i.e. **0.567× the margin**. The design
had the power to DECLARE equivalence had D been near zero. `FIXED_POINT` was
reachable and was not reached — the decay is not an artifact of a band too
wide to exclude it.

### Arms

| arm | authors | blocks | E(0–90d) | E(2–3y) | D | floor share | cell |
|---|---|---|---|---|---|---|---|
| PRIMARY full vocab, pool 849 | 849 | 45,731 | 0.6573 | 0.3515 | 0.3058 | 0.5348 | `DRIFT_WITH_CORE` |
| balanced panel (2–3y support) | 564 | 39,076 | 0.6402 | 0.3506 | 0.2896 | 0.5477 | `DRIFT_WITH_CORE` |
| activity tercile 1 (≤650 ev) | 302 | 2,347 | 0.7073 | 0.2817 | 0.4256 | 0.3982 | `DRIFT_WITH_CORE` |
| activity tercile 2 (650–2050) | 265 | 6,416 | 0.6981 | 0.3110 | 0.3871 | 0.4455 | `DRIFT_WITH_CORE` |
| activity tercile 3 (>2050 ev) | 282 | 36,968 | 0.6523 | 0.3501 | 0.3021 | 0.5368 | `DRIFT_WITH_CORE` |
| clean_no_explicit_personality | 779 | 39,634 | 0.6588 | 0.3418 | 0.3170 | 0.5188 | `DRIFT_WITH_CORE` |
| K = 100 blocks | 690 | 22,257 | 0.6994 | 0.3854 | 0.3140 | 0.5511 | `DRIFT_WITH_CORE` |

Three readings the arm table forces.

1. **Composition is not the decay.** The balanced panel — only the 564
   authors who actually contribute a 2–3y pair, so the same people appear in
   every bin — gives D 0.2896 against the primary's 0.3058 and a floor share
   0.5477 against 0.5348. The curve is not an artifact of who survives to the
   far bins.
2. **Attenuation transports as registered.** K = 100 raises the LEVEL
   (E(0–90d) 0.6994 against 0.6573, exactly the predicted reliability gain
   from doubling the block) while D (0.3140 vs 0.3058, +2.7%) and the floor
   share (0.5511 vs 0.5348, +3.0%) barely move. The fixed-K design's promise
   — E is an attenuated level, D and the floor share are the transportable
   quantities — is empirically kept.
3. **Governance echo, fourth instance.** Removing the 23 explicit-typology
   communities costs 13.3% of the blocks (45,731 → 39,634) and 8.2% of the
   pool, and moves D by **+3.7%** (0.3058 → 0.3170) with the cell unchanged.
   Identity structure keeps not caring about the criterion-adjacent
   communities. Note the sign: the clean arm drifts slightly MORE, so the
   typology communities were, if anything, a mild stabiliser.

### Leans

- **Primary lean: the CELL held, the MAGNITUDE missed — in the OVERCLAIM
  direction.** DRIFT_WITH_CORE was leaned and delivered, but the floor share
  is **0.5348 [0.4203, 0.6320]** against the registered [0.6, 0.9]. The
  interval's top edge only grazes the lean's bottom. The signature keeps 53%
  of its near-gap excess at three years, not the 60–90% the planner leaned
  on.

  **Why the lean missed is itself a finding.** The lean was calibrated from
  the T-line's early/late AUC 0.9837 and the S-line's split-half 0.681 —
  DISCRIMINABILITY statistics. But the cross floor here is 0.05: an author's
  blocks can lose nearly half their excess similarity and still be trivially
  separable from a random stranger's, because the stranger is so far away.
  **AUC over-predicts the persistence floor.** A high early/late AUC is
  compatible with substantial signature drift, and reasoning from
  discriminability to stability is a mistake this leg should retire.
- **Secondary lean: HELD, and monotone.** Decay by activity tercile
  0.4256 / 0.3871 / 0.3021 (low → high), floor share 0.3982 / 0.4455 /
  0.5368. Thin signatures both start higher (a narrow repertoire self-
  correlates) and fall further. The lean was held weakly and is now the
  clearest gradient in the leg.

### Refinement fit (never verdict-carrying, and it could not have been)

E(Δt) = E_inf + A·exp(−Δt/τ): E_inf **0.1310** [−1.0224, 0.3558],
A **0.5439** [0.3130, 1.7268], τ **976.3 d** [415.6, 3650.0] ≈ 2.67 y, cap
hit in 11.5% of bootstrap replicates. Three parameters on five points is
weakly identified: the E_inf interval runs NEGATIVE at its bottom, which is
unphysical for a floor, and an eighth of replicates prefer a straight line to
a decay-with-floor over this window. Read τ as an order of magnitude — years,
not months — and nothing finer. The floor the verdict rests on is the
measured E(2–3y) and its own interval, not this E_inf.

### Reliability descriptive

Adjacent-block same-author cosine, the shortest personal gap the design can
see: mean **0.7703** (median 0.8090, sd 0.1962) over 44,882 pairs from 849
authors at a mean gap of 17.4 d. The 0–90d self mean (0.7105) already sits
close to that ceiling, so the near bin is measuring near the instrument's
limit, not far below it.

### Gates and honesty checks

- **Cache anchor gate PASS**: 3,005,360 events / 1401 authors / 1191
  vocabulary, verified before any computation.
- **Census reproduction PASS on every registered pin**: 849 / 1028 / 690
  authors, 46,318 blocks, 18 quarters, self pairs per bin
  1,005,742 / 783,654 / 1,198,561 / 1,248,992 / 417,963 / 100,150 (exact),
  564 and 332 contributors, terciles 302 / 265 / 282 at edges 650 / 2050.
  The tercile convention needed for those sizes is recorded: quantiles of
  per-author BLOCK counts (13 and 41 = 650 and 2050 vocabulary events) with
  `x ≤ lo | lo < x ≤ hi | x > hi`.
- **Permutation-null center PASS** in all six bins and for D (above).
- **ID-leak scan PASS**: 0 of 1401 cohort IDs in the committed set.
- **No synthetic gate**, as registered (R layer, no world simulated; #76's
  operating-point convention does not bind).

### Disclosed re-posing RD-U2-1 (cross baseline computed exactly)

The registration specifies the epoch-matched cross term as a sample of up to
20 different-author pairs per self pair per cell. The runner instead computes
each (quarter-pair × bin) stratum's cross mean **exactly**, over all its
different-author pairs, and weights strata by the identical self-pair
histogram. This is the zero-variance limit of the registered sampler and has
the identical estimand, because the censused feasibility makes 20 × self <
available in every stratum. **The registered sampler was also run** on the
primary arm as an equivalence check: 95,094,317 cross pairs drawn of
95,101,240 requested (shortfall 7.3e−05, from the rejection-attempt cap),
per-bin |sampled − exact| at most **4.8e−05**. The re-posing is STRICTER
than the registered form.

### Census note not reproduced (non-blocking)

The registration's "min cross candidates per self quarter-pair cell 115.4×"
does not reproduce at either resolution the executor could construct:
**26.0×** over occupied (quarter-pair × bin) strata, **71.4×** at
quarter-pair resolution (median over strata 203.3×). The figure's denominator
is not named in the registration. The operative property is intact — both
resolutions clear the 20× the registered sampler needs, and zero self pairs
were dropped for want of a cross partner — so this is recorded, not
escalated. Future census lines should pin the denominator (the U1
precision note, repeated).

### Theory reading

**The first slow-time measurement in GRAIN, and it is neither a fixed point
nor a random walk.** U1 found the fast-time channel dwell-dominated; U2 finds
the slow-time marginal decaying to a large but non-flat residue: 53% of the
near-gap personal excess survives three years, with the drift and the core
both far outside anything the permutation null can produce.

The number that matters for the rest of the line is the DISSOCIATION between
persistence and discriminability. The T-line's early/late AUC 0.9837 and this
leg's floor share 0.5348 are the same corpus, the same metric, the same
authors — and they say different things because AUC is a ranking statistic
against a distant stranger while E is a magnitude. **Identity remains
recoverable while the signature substantially moves.** Any future leg that
infers stability from a discrimination score inherits this error.

Boundary the verdict must carry: **the curve has not flattened by the verdict
endpoint** (E(3y+) = 0.2638 sits below E(2–3y) = 0.3515, and the fit's
asymptote sits below both). `DRIFT_WITH_CORE` is a statement at a THREE-YEAR
horizon. The core is what survives three years, not a demonstrated permanent
floor, and extrapolating past the observed span is not licensed here.

### Line state

U2 CLOSED. The When line now has both a fast-time result (U1
`ORDER_CHANNEL`) and a slow-time one (U2 `DRIFT_WITH_CORE`). U3 (Who × When)
inherits: the cross-thread arm as the canonical order object (U1
adjudication), a three-year horizon over which the Where signature is known
to move by half its excess, and the standing warning that a trait join fitted
on one epoch is being fitted to a moving target.

## U2 planner adjudication (2026-08-18)

**Verdict ACCEPTED: `DRIFT_WITH_CORE`, scoped as a THREE-YEAR statement.**
D = 0.3058 [0.2364, 0.3855] against a null band of ±0.0017; floor share
0.5348 [0.4203, 0.6320]; all seven arms in the same cell, zero #73 flags,
permutation-null center verified at 0 to within 1e-04 in every bin. The
FIXED_POINT cell was demonstrably reachable (equivalence margin 0.1315 vs
realized D half-width 0.0745, ratio 0.567 — #71 satisfied in substance) and
was not reached. The executor's endpoint caution is adopted VERBATIM as
binding scope: E(3y+) = 0.2638 still sits below E(2–3y) = 0.3515 and the
exponential asymptote is weakly identified (E_inf CI dips negative), so
**the core is what survives three years, not a demonstrated permanent
floor** — "permanent core" is forbidden U2 prose; "three-year core" is the
licensed phrase.

**The lean record, and the leg's transportable finding.** The planner's
floor-share lean [0.6, 0.9] MISSED in the overclaim direction (realized
0.5348, CI top 0.6320 only grazing the lean's bottom). U1 missed under, U2
missed over; both stand as honest prediction records. The diagnosis is the
finding: **a discrimination score over-predicts geometric persistence.** The
T-line's early/late AUC 0.9837 and this floor share 0.5348 are the same
corpus and the same metric family; they coexist because the cross floor is
0.05 — the signature can lose ~46% of its personal excess over three years
while the person remains trivially closer to themselves than to anyone
else. Re-identification works not because people stand still but because
they move inside a space where others are far away. Stability claims may
never be inferred from discrimination scores (this sentence is now standing
guidance, the slow-time mirror of U1's "order is reproducible but redundant"
scoping).

**Secondary structure accepted.** Tercile gradient monotone (D 0.4256 /
0.3871 / 0.3021; floor share rising with activity 0.3982 / 0.4455 / 0.5368)
— the planner's weak secondary lean held. Attenuation transported as
designed under K = 100 (level up, D +2.7% relative). Clean arm: D +3.7%
with 13.3% of blocks removed — the FOURTH instance of the identity-robust /
trait-fragile asymmetry, with the sign indicating the typology communities
acted as a mild stabiliser of the signature, not a driver of its identity
content.

**Executor deviations adjudicated.**
- RD-U2-1 ACCEPTED AS EXEMPLARY: the exact stratified cross mean replaces
  the registered ≤20× sampler (same estimand, zero MC variance, STRICTER),
  and the registered sampler was still run as an equivalence check
  (max per-bin |sampled − exact| = 4.8e-05). This is the correct direction
  of deviation: tighter than registered, with the registered form used to
  certify the replacement.
- Bootstrap holding cross means fixed: accepted (cross side estimated from
  ~1e9 pairs; its sampling variance is negligible against the
  author-clustered self side); recorded in config.
- Tercile-variable inference: accepted; the registration should have pinned
  the tercile variable exactly (folded into #77 below).

**Defect #77 (planner, purchased; registry twenty-sixth note).** Census
statistics shipped without pinned computation twice in two legs: U1's tie
rate reproduced only on the in-vocabulary denominator (0.1152 vs 0.1076
all-events), and U2's cross-candidate feasibility "115.4×" did not
reproduce under the executor's stratum and pair-eligibility definitions
(26.0× over occupied quarter-pair × bin strata; 71.4× at quarter-pair
resolution) — the planner's census counted candidate pairs without
excluding same-author pairs or handling the diagonal quarter cells, and
named neither stratum nor eligibility; the tercile variable was likewise
left inferable. No verdict was at risk (feasibility cleared the sampler's
20× need in every definition; zero self pairs dropped), but the pattern
recurs. Convention: every registered census quantity carries its exact
computation — denominator, stratum, pair-eligibility — or is explicitly
marked "approximate, feasibility-only"; tercile/split variables are pinned
by name and formula.

**Named-open (not queued).**
- The long-horizon floor: does E(Δt) flatten anywhere? Corpus-bounded
  (span q75 ≈ 4.2 y); a second corpus with longer panels would be needed —
  priced with the second-corpus transport charter already standing.
- The persistence budget by carrier (which layer moves — the distinctive
  taste coordinate or the common mass): a derived-prediction test of the
  T-line proposition; REGISTERED NEXT as U2b.
- The drift's shape (sudden change-points vs gradual mixing): §2.4's
  change-point object, needs per-author trajectory modeling; priced, not
  queued.

**Line state.** U2 CLOSED. The When quadrant now has both time scales:
fast (U1, ORDER_CHANNEL, dwell-dominated) and slow (U2, DRIFT_WITH_CORE,
three-year core 53%). U2b (persistence budget by carrier) is next —
label-free, cheap, and it tests a prediction DERIVED from the T-line
proposition. U3 (stage-E trait join) remains gated: it now inherits BOTH
the cross-thread designation (U1) and the drift-aware caution that
slow-time coordinates are moving targets (U2).

---

## U2b — the persistence budget by carrier (registered BEFORE run, 2026-08-18)

A derived-prediction test: the T-line proposition ("identity lives in the
distinctive; personality lives in the common") plus U2's DRIFT_WITH_CORE
forces the question WHICH LAYER MOVES. If re-identification survives three
years while the signature loses ~46% of its personal excess, the T-line
reading predicts the distinctive layer is the standing part. This leg tests
that prediction label-free on U2's own blocks.

### 4W header

- **Which 4W object.** When × the T-line's carrier split — the persistence
  curve of U2, decomposed by sub-vocabulary carrier (common mass vs
  distinctive tail vs the low-rank taste coordinate). Slow-time projection
  of K_u; identity structure only, no psychological naming.
- **Fixed.** U2's blocks, bins, pools, permutation-null machinery, Hellinger
  conventions — inherited unchanged (inheritance is not exemption, #56: the
  U2 anchors are re-verified bit-close before any new row). The vocabulary
  split is corpus-global and fit-free, precedent: SR0's vocabulary itself.
- **Varied.** The carrier restriction (full / common / distinctive / taste);
  the split level q ∈ {0.3, 0.5, 0.7}; the eligibility floor m ∈ {5, 10, 15}.
- **What falsifies.** A Δfloor CI strictly below 0 (COMMON_STANDING) refutes
  the derived prediction and forces a When-clause revision of the T-line
  proposition; a null split (NO_LAYER_SPLIT) says persistence does not
  decompose along the common/distinctive axis at this resolution.
- **Layer.** R.

### Census (planner arithmetic, #43; computations pinned per #77)

Universe: the 2,348,361 in-vocabulary cohort events of the U1 cache (the
denominator); communities ranked by descending event count over that
universe; Common(q) = the smallest rank prefix with cumulative share ≥ q;
Distinctive(q) = its complement within the 1191 vocabulary. Blocks: U2's
exact construction (849 authors, 45,731 blocks — reproduced). Pair
eligibility at floor m: a same-author pair enters a restricted row iff BOTH
blocks hold ≥ m events of that sub-vocabulary (pair-level, per row).

| quantity (pinned definition above) | value |
|---|---|
| Common(0.5) | **32 communities** (realized share 0.5036) |
| Common(0.3) / Common(0.7) | 8 (0.3077) / 104 (0.7008) |
| per-block common events, q5/q25/median | 0 / 8 / 25 |
| per-block distinctive events, q5/q25/median | 0 / 6 / 25 |
| eligible pairs at m=10, 2–3y: common / distinct | 253,946 / 230,661 |
| distinct-eligible authors at m=10, 2–3y | 506 |

### Design pins (#70, #72)

- Primary configuration: q = 0.5, m = 10.
- **One pair set for all rows (#72):** the INTERSECTION set — pairs whose
  both blocks hold ≥ m events in BOTH sub-vocabularies. Full, common,
  distinctive, and taste rows are all computed on this identical pair set;
  no cross-row comparison ever uses non-aligned pairs. Registered pool
  targets (#69): the intersection set must retain ≥ 100,000 pairs and
  ≥ 400 contributing authors in the 2–3y bin; if unmet at q=0.5/m=10 the
  leg STOPS and reports (no silent re-split).
- Rows: (1) full vocabulary (anchor; additionally the UNRESTRICTED full
  curve is recomputed and bit-compared against U2's committed artifacts —
  G0 gate); (2) common-restricted (block vector renormalized on Common(q));
  (3) distinctive-restricted; (4) taste — per-fold PPMI+SVD d = 64 community
  embeddings (SR3 pattern: 5 folds, KFold shuffle SEED, embeddings fit on
  TRAINING authors' first-half events only, first half by the author-median
  rule; test-author purity asserted), block taste vector = Hellinger-weighted
  mean of its communities' embeddings, L2-normalized; fold curves pooled by
  unweighted mean (U1 convention).
- Estimands per row: E_row(b) with U2's epoch-matched exact stratified cross
  baseline (RD-U2-1 form) on the intersection pair set; floor share
  F_row = E_row(2–3y)/E_row(0–90d). Verdict contrast:
  **Δfloor = F_distinct − F_common** with cluster-bootstrap (B = 1000) CI.
  Secondary contrast: F_taste − F_full.
- Own nulls (#68/#66): U2's within-quarter permutation, B = 499, per row and
  for Δfloor; expected null locations 0 by construction, verified.
- Comparison rule (pinned): LEVEL differences across rows are never
  interpreted (restricted rows carry different, block-varying attenuation);
  only within-row floor shares and their contrasts are transportable.
  Registered descriptive: mean common-share of blocks by account era (the
  mix-shift trajectory), non-verdict-moving.
- Sensitivities: q ∈ {0.3, 0.7} and m ∈ {5, 15} on the primary contrast;
  #73 flags on any cell divergence, primary routes.

### Classification cells (NULL-first #55, effect-size keyed #75)

1. **NO_LAYER_SPLIT** — Δfloor CI includes 0 AND |Δfloor| < 0.10
   (equivalence band; realized CI width vs the 0.10 band reported per #71).
2. **DISTINCTIVE_STANDING** — Δfloor CI > 0 (the derived prediction).
3. **COMMON_STANDING** — Δfloor CI < 0 (forces the When-clause revision).
4. **UNRESOLVED_SPLIT** — CI straddles 0 with |point Δfloor| ≥ 0.10;
   verdict takes the interval statement.

### Registered leans

- L1 (the derived prediction, held moderately): DISTINCTIVE_STANDING with
  point Δfloor ∈ (0.05, 0.25].
- L2 (held weakly): the taste row posts the highest floor share of all four
  rows (F_taste ≥ F_full) — the low-rank generalizing coordinate is the
  slow part.
- Stated openly: COMMON_STANDING is the informative surprise — it would
  relocate the standing core to the layer the T-line assigned to
  personality, and U3's design would then target the stable common layer.

### Deliverables and discipline

Standard six: `scripts/run_suica_m4_u2b_persistence_budget.py` +
`tests/test_m4_u2b_persistence_budget.py`; gitignored
`results/m4_u2b_persistence_budget/`; report
`reports/SUICA_M4_U2B_PERSISTENCE_BUDGET_REPORT.md` (rule-24 generated
tables; null locations; #71 projection; the eq-12 projection caution;
config block); outcome appended here; one CLAIMS_LEDGER row (EXPLORATORY,
label-free, corpus-level); exactly ONE commit
`feat(m4-u): U2b — the persistence budget by carrier — <VERDICT>`, never
amended, never pushed by the executor. Suite green (1078 + new) before
commit. Blocking gates: cache anchors; U2 full-curve bit-comparison (G0);
intersection pool targets; permutation-null centers; fold purity (taste
row); ID-leak scan (0 of 1401). SEED = 20260818; B_perm = 499;
B_boot = 1000. Label-free.

## U2b outcome (executor, 2026-08-18)

Append-only. The registration text above is unchanged. Artifact stamps: run
window `2026-08-17T22:25:22.373510Z` – `2026-08-17T22:27:14.220465Z` (UTC;
local date 2026-08-18). Report:
`reports/SUICA_M4_U2B_PERSISTENCE_BUDGET_REPORT.md`. Harness:
`scripts/run_suica_m4_u2b_persistence_budget.py`. Tests:
`tests/test_m4_u2b_persistence_budget.py`. Artifacts (gitignored):
`results/m4_u2b_persistence_budget/`.

### Outcome

**`POOL_GATE_UNMET`. THE LEG STOPS AT THE REGISTERED #69 GATE AND ASSIGNS NO
CELL.** The registration pins that the intersection pair set at q = 0.5 /
m = 10 must retain ≥ 100,000 pairs and ≥ 400 contributing authors in the 2–3y
bin, and that the leg STOPS and reports if that is unmet with no silent
re-split. It is unmet on BOTH clauses.

| #69 clause | required | realized | ratio |
|---|---|---|---|
| 2–3y pairs, intersection set | 100,000 | **99,714** | 0.9971× |
| 2–3y contributing authors | 400 | **348** | 0.8700× |

Everything below is PROVISIONAL and non-verdict-carrying; it is computed and
reported so the planner can adjudicate the re-split it now owns from numbers
rather than from an empty report (disclosed re-posing RD-U2B-1, below).

### Why the gate missed, and it is not a construction error

Every registered census pin reproduces EXACTLY — all fourteen, including the
two the gate depends on:

| registered census quantity | registered | observed |
|---|---|---|
| universe (in-vocabulary cohort events) | 2,348,361 | **2,348,361** |
| Common(0.5) | 32 at 0.5036 | **32 at 0.5036** |
| Common(0.3) / Common(0.7) | 8 at 0.3077 / 104 at 0.7008 | **identical** |
| pool | 849 authors, 45,731 blocks | **identical** |
| per-block common q5/q25/median | 0 / 8 / 25 | **identical** |
| per-block distinctive q5/q25/median | 0 / 6 / 25 | **identical** |
| eligible pairs m=10, 2–3y: common / distinct | 253,946 / 230,661 | **identical** |
| distinct-eligible authors m=10, 2–3y | 506 | **identical** |

The construction here IS the planner's construction, verified on the two
MARGINAL eligibilities to the last pair. What the census did not carry is
their INTERSECTION, and the intersection is far smaller than either marginal:
99,714 pairs against 253,946 and 230,661 — **39.3% of the common row's 2–3y
pairs and 61.7% of U2's 564 2–3y contributors.** The mechanism is structural,
not incidental. A block holds exactly K = 50 in-vocabulary events, so
"≥ 10 events in BOTH sub-vocabularies" is the single per-block predicate
`10 ≤ common ≤ 40`; it discards every LOPSIDED block, the per-block common
count is U-shaped (q5 = 0 for both halves, i.e. ≥ 5% of blocks are pure
common and ≥ 5% pure distinctive), and lopsidedness is author-persistent, so
whole authors leave rather than a thin slice of each author's pairs. **This
is defect #77's third instance and its first with teeth: a registered census
quantity that was projected from marginals rather than computed.** The
previous two (U1's tie rate, U2's 115.4× feasibility) cost a footnote; this
one costs the leg's primary configuration.

One clean consequence worth pinning for the convention: eligibility on a
fixed-K block is a BLOCK property, not a pair property, so the intersection
pair set is exactly the same-author pair set of the eligible block subset.
That is what makes #72 hold by construction here — identical pair indices,
identical cross reservoir, identical permutation plans and identical
bootstrap author draws across all four rows, so Δfloor's interval is paired
replicate by replicate rather than aligned after the fact.

### The four rows — PROVISIONAL

All on the ONE intersection pair set (20,047 blocks, 99,714 self pairs in
2–3y). **LEVEL differences across rows are never interpreted** (registered
comparison rule): a restricted row carries its own block-varying attenuation.
Only within-row floor shares and their contrasts transport.

| row | 0–90d | 90–180d | 180–365d | 1–2y | 2–3y | 3y+ | D [CI] | F = E(2–3y)/E(0–90d) [CI] |
|---|---|---|---|---|---|---|---|---|
| full | 0.5218 | 0.4691 | 0.4262 | 0.3430 | 0.2672 | 0.2441 | 0.2546 [0.1564, 0.3600] | **0.5120** [0.3441, 0.6793] |
| common (32 communities) | 0.5879 | 0.5511 | 0.5152 | 0.4296 | 0.3405 | 0.3182 | 0.2474 [0.1258, 0.3858] | **0.5792** [0.3603, 0.7760] |
| distinctive (1159) | 0.4440 | 0.3753 | 0.3323 | 0.2578 | 0.2055 | 0.2012 | 0.2385 [0.1329, 0.3481] | **0.4628** [0.3000, 0.6655] |
| taste (d = 64, OOF) | 0.4278 | 0.4071 | 0.3810 | 0.3264 | 0.2956 | 0.2636 | 0.1322 [0.0754, 0.1943] | **0.6909** [0.5530, 0.8210] |

**Δfloor = F_distinct − F_common = −0.1163 [−0.2620, 0.0299]**, provisional
cell `UNRESOLVED_SPLIT` (|point| ≥ 0.10 with the interval straddling zero).
Secondary contrast F_taste − F_full = +0.1789 [−0.0373, 0.3749].

**The point estimate carries the sign of the informative surprise, and the
sign is unanimous across the whole registered grid.** Δfloor is NEGATIVE at
all five configurations — −0.1163 (q .5/m 10), −0.4406 (q .3), −0.1209
(q .7), −0.0980 (m 5), −0.1083 (m 15) — and two of the five have an interval
strictly below zero. The direction that keeps appearing is `COMMON_STANDING`:
the COMMON mass, not the distinctive tail, is the slower-decaying carrier at
this resolution. That is the opposite of the derived T-line prediction and it
is what the planner should be adjudicating toward, with the caveat that no
interval at the registered primary configuration excludes zero.

### The sensitivity grid, which is also the adjudication data

| q | m | Common(q) | 2–3y pairs | 2–3y authors | #69 | F_common | F_distinct | Δfloor [CI] | cell |
|---|---|---|---|---|---|---|---|---|---|
| **0.5** | **10** | 32 | 99,714 | 348 | **UNMET** | 0.5792 | 0.4628 | −0.1163 [−0.2620, 0.0299] | `UNRESOLVED_SPLIT` |
| 0.3 | 10 | 8 | 48,862 | 199 | UNMET | 0.8109 | 0.3703 | −0.4406 [−0.5648, −0.2011] | `COMMON_STANDING` |
| 0.7 | 10 | 104 | 108,716 | **401** | **PASS** | 0.5792 | 0.4583 | −0.1209 [−0.2256, −0.0007] | `COMMON_STANDING` |
| 0.5 | 5 | 32 | 179,107 | **424** | **PASS** | 0.5625 | 0.4646 | −0.0980 [−0.2144, 0.0424] | `NO_LAYER_SPLIT` |
| 0.5 | 15 | 32 | 42,230 | 259 | UNMET | 0.5899 | 0.4817 | −0.1083 [−0.2789, 0.0378] | `UNRESOLVED_SPLIT` |

Three #73 flags (q = 0.3, q = 0.7, m = 5 diverge in cell from the primary
configuration). **Two ALREADY-REGISTERED configurations clear both #69
targets — q = 0.7/m = 10 and q = 0.5/m = 5 — so the planner can adjudicate a
promotion rather than charter a new leg. The executor did not promote either;
that is the re-split the registration forbids doing silently.** Note what the
two say: q = 0.7 lands `COMMON_STANDING` (CI top −0.0007, grazing), m = 5
lands `NO_LAYER_SPLIT`. They do not agree on a cell, and the honest reading
is that this design resolves the SIGN more confidently than the CELL.

### The equivalence band could NOT have spoken (#71) — the leg's real defect

The `NO_LAYER_SPLIT` band is |Δfloor| < 0.10. The realized half-width of
Δfloor's 95% CI is **0.1459 — 1.459× the band.** Unlike U2, where the
FIXED_POINT cell was demonstrably reachable (0.567× its margin), **this
design could not have DECLARED a null split at any Δfloor**: a Δfloor of
exactly zero would still have read `UNRESOLVED_SPLIT`. The m = 5 row's
`NO_LAYER_SPLIT` is therefore the arithmetic of a point inside the band with
a wide interval, not a demonstration of equivalence. Any re-registration
should price the interval width against the 0.10 band BEFORE the run — the
pool gate and the band are the same power question asked twice, and the leg
failed both.

### Own nulls (#68), and an honest anomaly

Per-row per-bin within-quarter permutation centers on E(b) are at most
**2.7e−04** in any bin of any row (full row max 8.5e−05), three to four
orders of magnitude inside their own bands and four inside the effects —
U2's result reproduced on every carrier.

**The Δfloor null is the anomaly and it is structural.** A floor share is a
RATIO whose numerator and denominator the permutation both drives to zero, so
its null is heavy-tailed by construction: the realized Δfloor permutation
center is **+0.04479** with IQR **[−2.346, +2.651]** over 499 replicates
(100% finite). The center is where the registration's argument says it should
be — the claim is about LOCATION — but the 95% band is not a bound and is not
read as one. Registered nulls on ratio estimands should in future be posed on
the LOG ratio or on the numerator contrast; that is a convention gap, not a
run defect.

### Gates

- **Cache anchor gate PASS**: 3,005,360 events / 1401 authors / 1191
  vocabulary, verified before any computation.
- **G0 (#56) PASS, BITWISE**: U2's primary arm recomputed here through the
  imported estimator and compared field by field against
  `results/m4_u2_persistence_curve/arms.json` — **max absolute difference
  0.0 across 20 fields, every field bit-identical**, including the full
  curve, both CIs, the null bands, D, the floor share and the permutation
  p-values. The machinery is U2's object, imported by file, not a
  re-implementation.
- **Census reproduction PASS on all fourteen registered pins** (table above);
  zero non-reproducing quantities, in contrast to U1 and U2.
- **Taste-row fold purity PASS**: 5 folds, 679/679/679/679/680 training
  authors, zero train/test overlap, and the mass identity exact — the fitted
  count matrix's total equals the training authors' first-half in-vocabulary
  mass to within 1e−06 in every fold, with the test authors' mass entirely
  absent.
- **Sub-vocabulary count recovery PASS**: counts recovered from the Hellinger
  feature as K·f², maximum deviation from integrality 1.05e−05.
- **Pool gate #69 UNMET** — the leg's outcome.
- **ID-leak scan PASS**: 0 of 1401 cohort IDs in the committed set.
- **No synthetic gate**, as registered (R layer, no world simulated).

### Registered leans (both PROVISIONAL)

- **L1 MISSED, and missed in SIGN, not magnitude.** The lean was
  `DISTINCTIVE_STANDING` with Δfloor in (0.05, 0.25]; the realized point is
  **−0.1163**, the same magnitude with the opposite sign, and every
  configuration in the grid agrees on that sign. The derived prediction from
  the T-line proposition is not merely unconfirmed here — the data lean
  against it.
- **L2 HELD.** The taste row posts the highest floor share of all four:
  F_taste **0.6909** against F_full 0.5120, F_common 0.5792, F_distinct
  0.4628. The low-rank generalizing coordinate is the slow part, exactly as
  leaned. Per-fold floor shares 0.5547 / 0.4806 / 0.8310 / 0.7938 / 0.7992 —
  wide across folds, so the lean holds on the pooled estimate with visible
  fold-level dispersion.

### Registered descriptive — the mix-shift trajectory (non-verdict-moving)

Mean common-share of a block over all 45,731 pool blocks, overall 0.5070.

| calendar era | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|
| mean common share | 0.4517 | 0.4886 | 0.5407 | 0.5295 | 0.4393 |

| within-author tenure quintile | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| mean common share | 0.4987 | 0.5056 | 0.5223 | 0.5211 | 0.4875 |

Both readings of "account era" are reported because the registration does not
pin it. The calendar trajectory is a shallow inverted U with a range of
0.10; the personal-tenure trajectory is nearly flat (range 0.035). **The mix
shift is a platform-era effect, not a personal one**, which is reassuring for
the epoch-matched design: the cross baseline already absorbs the calendar
part.

### Disclosed re-posing RD-U2B-1 (the STOP is a stop at the VERDICT)

The registration says the leg "STOPS and reports". Implemented as a stop at
the VERDICT, not at the COMPUTATION: no cell is assigned, `verdict.json`
carries `outcome = POOL_GATE_UNMET` with `cell = null`, every estimate is
labelled PROVISIONAL in the report and in this section, and the registered
rows, contrasts and sensitivity grid are still computed so the adjudication
can happen once. **No re-split was performed and no configuration was
promoted.** If the planner intended a bare stop, the surplus here is
discardable; the reverse — adjudicating a pool failure with no numbers — is
not recoverable without a second run.

### Line state

U2b DELIVERED WITHOUT A VERDICT. The When line's carrier question is
UNRESOLVED at the registered primary configuration and the planner owns the
re-split. What the leg does deliver: the census defect that cost it (#77's
first consequential instance), a bit-exact G0 reproduction of U2's anchor
(#56 satisfied to 0.0), a unanimous negative SIGN on Δfloor across five
configurations pointing at the named informative surprise
(`COMMON_STANDING`), L2 held, and a #71 finding that this design's interval
is 1.459× its own equivalence band — so any promotion should re-price power,
not only the pool.


## U2b planner adjudication (2026-08-18)

**Registered outcome ACCEPTED: `POOL_GATE_UNMET` — no cell is assigned.**
The #69 gate did its job: intersection realization 99,714 pairs (0.997× the
target) and 348 authors (0.870×) at q=0.5/m=10. The executor's RD-U2B-1
(STOP implemented at the verdict, provisional numbers computed and
quarantined) is accepted: for a POOL/POWER gate this is the economical form
— it hands the adjudicator everything for one decision instead of pricing a
second blind run. The A1 principle is untouched: INSTRUMENT-validity gates
(G0, synthetic gates) remain hard stops before real data; pool gates stop at
the verdict with provisional numbers marked and quarantined. This
distinction is adopted as standing convention (recorded under #78).

**Why the gate missed — defect #78 (planner, purchased; registry
twenty-seventh note).** All fourteen registered census pins reproduce
exactly; the gate governed a quantity none of them computed. The pool
targets were set from MARGINAL eligibilities while the gate ran on their
INTERSECTION — and on a fixed-K block the both-carriers predicate
(10 ≤ common ≤ 40) removes lopsided blocks, whose owners leave wholesale
(61.7% of U2's 2–3y contributors survive). #77's third instance and its
first with teeth. Convention: **every registered gate quantity is itself
censused — the gate's exact predicate executed on the census data — never
derived from marginals or bounds**; pool/power gates stop at the verdict
with quarantined provisionals, instrument gates stop before real data.

**Defect #79 (planner, purchased; same note).** Two power/estimand defects
in one leg: (a) the 0.10 equivalence band was registered with no achievable-
width projection — realized Δfloor CI half-width 0.1459 = 1.459× the band,
so NO_LAYER_SPLIT was unreachable at any point value (the #71 convention
existed and was not executed at registration time); (b) the Δfloor
permutation null is a ratio of two permutation-nulled quantities and is
heavy-tailed by construction (IQR ±2.5 around a 0.045 center) — the location
check passes but the band is unreadable as a bound. Convention: every
equivalence cell carries a REGISTRATION-TIME width projection from prior
legs' realized dispersions; nulls and contrasts on ratio estimands are posed
on the log ratio or on a numerator/slope contrast, never on the raw ratio.

**The provisional evidence (quarantined, adjudication input only).** The
sign of Δfloor is negative in all five configurations (−0.098 to −0.441),
with two registered configurations clearing both #69 targets and both
carrying #73-flagged cells (COMMON_STANDING at q=0.7/m=10 with CI
[−0.2256, −0.0007]; NO_LAYER_SPLIT at q=0.5/m=5 — the latter now known
to be an arithmetic artifact of the unreachable band, per #79a). The
planner's derived prediction L1 MISSED IN SIGN. L2 HELD: the taste
coordinate posts the highest floor share of all rows (0.6909).

**No post-hoc promotion.** Promoting a passing sensitivity configuration to
primary after seeing its cell would manufacture a verdict at the noise edge
(the q=0.7 CI clears zero by 7e-4). Instead the completion leg U2c is
REGISTERED BELOW with full disclosure: the provisional sign is known; U2c
exists to issue a verdict from a registered-before-run primary under
corrected gate arithmetic (#78) and a re-posed, ratio-free contrast (#79),
exactly the SR4 pattern (a completion micro-leg whose registration names
what is already known).

**The emerging three-layer time structure (named, to be tested by U2c and
carried to the synthesis only if U2c confirms).** If the provisional sign
holds: the taste COORDINATE is slowest (F 0.69), the common mass is slower
than the distinctive tail (0.58 vs 0.46) — so identity's cross-sectional
carrier (the distinctive) is the FASTEST-churning layer, and re-identification
survives because the low-rank taste DIRECTION expressed through those
churning communities is stable. "Identity lives in the distinctive" would
then be a statement about WHERE the signal sits, not about what stands
still: the distinctive is the ink, the taste coordinate is the hand.

**Gates and discipline.** G0 bitwise (U2's twenty fields at 0.0 difference);
taste purity exact; suite 1078 → 1115; ID-leak 0/1401; one commit; ppmi_svd
replicated bit-identically from T2 with contract test (RN-SR3-1 pattern
honored). The mix-shift descriptive reads platform-era, not personal — and
the epoch-matched baseline already absorbs it.

---

## U2c — the decay-rate contrast (completion of U2b; registered BEFORE run, 2026-08-18)

**Disclosure first:** U2b's quarantined provisionals are KNOWN and point
COMMON_STANDING with sign unanimity. U2c is a verdict-issuing completion
under corrected arithmetic, not a prediction; the registration's honesty is
this disclosure (SR4 precedent).

### 4W header

- **Object.** Identical to U2b (which layer moves), re-posed per #79 on
  decay RATES instead of floor-share ratios.
- **Fixed.** U2b's split (Common(0.5) = 32 communities), blocks, bins,
  intersection pair-set construction (#72 by construction), exact stratified
  cross baseline, permutation and bootstrap machinery — inherited unchanged
  from the committed U2b runner (#56: anchors re-verified).
- **Varied.** Eligibility floor m = 5 (primary — the largest REALIZED
  passing pool); q = 0.7 / m = 10 confirmatory sensitivity; all other U2b
  grid points descriptive.
- **Falsifies.** A slope contrast CI strictly below 0 (DISTINCT_SLOWER)
  refutes the provisional sign; a straddling CI yields the interval
  statement (SIGN_UNRESOLVED — a live outcome per the projection below).
- **Layer.** R.

### Gate arithmetic (#78: the gate predicate was executed on census data)

The gate quantities are U2b's REALIZED intersection numbers (the gate's
exact predicate, already executed): q=0.5/m=5 → 179,107 pairs / 424 authors
in 2–3y (both targets met: ≥ 100,000 / ≥ 400); q=0.7/m=10 → 108,716 / 401
(met). The U2c runner re-verifies both against its own recomputation as a
blocking anchor; any mismatch STOPS the leg.

### Estimand and contrast (#79: ratio-free)

Per row (full / common / distinct / taste), fit by OLS on the five verdict
bins: log E_row(b) = log E0_row − λ_row · gap(b), gap in years at the row's
realized per-bin mean gaps on the intersection set. λ_row is the row's decay
rate (1/y), level-free by construction (attenuation moves log E0, not λ,
when gap-independent — U2's argument inherited). VERDICT CONTRAST:
**Λ = λ_distinct − λ_common** (positive Λ = the distinctive decays faster =
common standing), paired cluster bootstrap B = 1000 (identical author draws
across rows), CI on Λ. Replicates with any non-positive E(b) are dropped
with count reported (expected ≈ 0; all realized E(b) ≥ 0.20). Permutation
null (B = 499): within-quarter reassignment; the null is posed on Λ directly
(a slope contrast, not a ratio — well-behaved); realized center reported.
Secondary: λ_taste vs λ_full (L2's rate form).

### Registration-time width projection (#71/#79, executed now)

From U2b's realized quantities: implied rates λ_com ≈ 0.239/y,
λ_dist ≈ 0.338/y, Λ ≈ +0.10/y; the Δfloor CI half-width 0.146 maps to
≈ 0.12/y at m=10, shrinking with the m=5 pool (+22% authors) and the
five-bin slope fit to a projected half-width ≈ 0.08–0.10/y. **The
projection says detection is BORDERLINE: SIGN_UNRESOLVED is a live cell and
the verdict may be the interval statement.** This is stated before the run;
no equivalence cell is offered at all (none is reachable at this width —
posing one would repeat #79a).

### Cells (NULL-first #55, effect-size keyed #75)

1. **SIGN_UNRESOLVED** — Λ CI straddles 0; verdict is the interval
   statement (sign evidence carried as: U2c CI + U2b's quarantined sign
   unanimity, both reported, tier unchanged EXPLORATORY).
2. **COMMON_STANDING** — Λ CI > 0 (confirms the provisional sign; the
   T-line proposition acquires its When clause: the distinctive churns,
   the taste direction stands).
3. **DISTINCT_SLOWER** — Λ CI < 0 (refutes the provisionals; forces a
   re-examination of U2b's computation before any theory move).

Confirmatory q=0.7/m=10 divergence carries a #73 flag; primary routes.

### Leans (disclosed-knowledge)

- Λ > 0 with point ≈ +0.10/y (from the provisionals; this is a disclosure,
  not a prediction).
- λ_taste is the smallest rate of the four rows (L2's rate form; from
  F_taste 0.6909, λ_taste ≈ 0.16/y projected).

### Deliverables and discipline

Standard six; script `scripts/run_suica_m4_u2c_decay_rate_contrast.py` +
tests; gitignored `results/m4_u2c_decay_rate_contrast/`; report
`reports/SUICA_M4_U2C_DECAY_RATE_CONTRAST_REPORT.md`; outcome appended
here; one CLAIMS_LEDGER row; ONE commit
`feat(m4-u): U2c — the decay-rate contrast — <VERDICT>`; suite green
(1115 + new); blocking gates: cache anchors, U2b intersection re-verification,
G0 bit-comparison of the four-row E(b) table against U2b's committed
artifacts, permutation-null centers, taste-fold purity, ID-leak scan.
SEED = 20260818; B_perm = 499; B_boot = 1000. Label-free.


## U2c outcome (executor, 2026-08-18)

Append-only. The registration text above is unchanged. Artifact stamps: run
window `2026-08-17T22:53:29.632693Z` – `2026-08-17T22:55:04.435166Z` (UTC;
local date 2026-08-18). Report:
`reports/SUICA_M4_U2C_DECAY_RATE_CONTRAST_REPORT.md`. Harness:
`scripts/run_suica_m4_u2c_decay_rate_contrast.py`. Tests:
`tests/test_m4_u2c_decay_rate_contrast.py`. Artifacts (gitignored):
`results/m4_u2c_decay_rate_contrast/`.

### Outcome

**Cell 1, `SIGN_UNRESOLVED`. THE VERDICT IS THE INTERVAL STATEMENT.**

**Λ = λ_distinct − λ_common = +0.0741/y [−0.0558, 0.1854]** at the registered
primary q = 0.5 / m = 5 (179,107 self pairs, 424 contributing authors in the
2–3y bin). The registration named this cell live before the run and projected
a borderline half-width of 0.08–0.10/y; the realized half-width is **0.1206/y,
1.206× the top of that range**, and the interval straddles zero. Tier
unchanged: EXPLORATORY, label-free, corpus-level.

The registered sign evidence is therefore carried in the two pieces the
registration named: **this CI, together with U2b's quarantined sign
unanimity** — and U2c adds a third piece of the same kind, not a fourth kind.
Both of U2c's arms put the point on the same side of zero, and in both arms
λ_distinct exceeds λ_common:

| arm | pairs / authors (2–3y) | λ_full | λ_common | λ_distinct | λ_taste | Λ [CI] | half-width | cell |
|---|---|---|---|---|---|---|---|---|
| **primary q=0.5/m=5** | 179,107 / 424 | 0.2943 | **0.2535** | **0.3276** | 0.1736 | **+0.0741** [−0.0558, 0.1854] | 0.1206 | `SIGN_UNRESOLVED` |
| confirmatory q=0.7/m=10 | 108,716 / 401 | 0.2751 | 0.2315 | 0.3176 | 0.1515 | +0.0861 [−0.0109, 0.1609] | 0.0859 | `SIGN_UNRESOLVED` |

**Zero #73 flags** — the arms agree in cell. Note how close the confirmatory
arm comes: its lower bound is −0.0109, i.e. it misses `COMMON_STANDING` by
about 1.1e−02/y. That is a near-miss, not a result, and the primary routes.

### The four rates and their fits (primary arm)

R² is the log-linear fit's, descriptive only. All bootstrap replicates were
retained in every row: **zero drops for non-positive E(b) out of 1000**, as
the registration expected (realized E(b) ≥ 0.2033 everywhere in the fitted
window).

| row | λ (1/y) [CI] | R² | E0 | F = E(2–3y)/E(0–90d) | boot drops |
|---|---|---|---|---|---|
| full vocabulary | 0.2943 [0.1895, 0.4405] | 0.9939 | 0.5403 | 0.5094 | 0 / 1000 |
| common (32 communities) | 0.2535 [0.1408, 0.4244] | 0.9992 | 0.6122 | 0.5625 | 0 / 1000 |
| distinctive (1159) | 0.3276 [0.1854, 0.4970] | 0.9747 | 0.4280 | 0.4646 | 0 / 1000 |
| taste (d = 64, OOF) | 0.1736 [0.0970, 0.2737] | 0.9670 | 0.4419 | 0.6803 | 0 / 1000 |

Per-bin personal excess E(b) and the realized mean gaps the slope is fitted
at (the 3y+ column is DESCRIPTIVE and never entered the fit):

| row | 0–90d | 90–180d | 180–365d | 1–2y | 2–3y | *3y+ (unfitted)* |
|---|---|---|---|---|---|---|
| full | 0.5340 | 0.4820 | 0.4330 | 0.3459 | 0.2720 | *0.2434* |
| common | 0.5925 | 0.5554 | 0.5148 | 0.4259 | 0.3333 | *0.2962* |
| distinctive | 0.4376 | 0.3729 | 0.3272 | 0.2551 | 0.2033 | *0.2026* |
| taste | 0.4407 | 0.4183 | 0.3866 | 0.3295 | 0.2998 | *0.2756* |
| **mean gap (days)** | 43.18 | 133.31 | 266.62 | 517.59 | 875.16 | *1240.19* |
| **mean gap (years)** | 0.1182 | 0.3650 | 0.7300 | 1.4171 | 2.3961 | *3.3955* |
| **self pairs** | 453,665 | 356,709 | 538,528 | 541,856 | 179,107 | *49,230* |

### L2's rate form: HELD, and it is the cleanest thing in the leg

**λ_taste is the smallest of the four rows in BOTH arms** — 0.1736/y (primary)
and 0.1515/y (confirmatory) — against the registration's projected 0.16/y.
The secondary contrast λ_taste − λ_full = **−0.1207 [−0.2771, 0.0205]**
(confirmatory −0.1236 [−0.2798, 0.0300]); it is the UNPAIRED contrast (the
taste row's bootstrap draws are fold-stratified against the other rows'
pool-level draws) so its interval is conservative and it too straddles zero.
The lean is about the ORDERING, and the ordering holds in both arms.

### Nulls, and the second thing the registration's null argument missed

Per-row per-bin E(b) permutation centers are at most **3.7e−04** in any bin of
any row (full row max 5.8e−05), 818× to 4,699× inside their own effects —
U2's result reproduced on every carrier at both floors.

**The Λ null is the anomaly, and RD-U2C-1 is the disclosed handling.** The
registration posed the null directly on Λ on the ground that a slope contrast
is well behaved where a ratio is not. That is true of the SLOPE and false of
the LOGARITHM the registered slope is taken on: a within-quarter relabelling
drives E(b) to within a few 1e−4 of zero, the sign then flips freely across
bins, and log E(b) is undefined for most replicates. The registered null was
computed with the same positivity rule and **retains only 7 of 499
replicates** (confirmatory: 22 of 499) — center +0.0446, band unreadable.
Beside it the runner reports the positivity-free LINEAR slope contrast (OLS of
E(b) on gap_years, no logarithm), defined on 100% of replicates: center
**+6.09e−05** with band [−0.0029, 0.0026] against a realized −0.0162 on that
same scale. **No verdict rests on either null.** The convention gap is #79's
successor and belongs to the planner: *nulls on slope estimands taken on a
transformed scale inherit the transform's domain, so a log-slope null needs
either a positivity-safe estimator or a companion posed on the untransformed
slope.*

### The honest anomaly: the bigger pool bought a WIDER interval

The m = 5 pool did everything the projection said it would — **+21.8% authors
(424 vs U2b's 348) and +79.6% pairs (179,107 vs 99,714)** — and the interval
still came out **1.404× wider** than the confirmatory arm's, which runs on a
SMALLER pool (108,716 pairs / 401 authors). The per-row half-widths locate it
in the distinctive row: 0.1558 at m = 5 against 0.1153 at m = 10, a 1.352×
inflation, while full (1.065×), common (1.007×) and taste (1.059×) barely
move. Lowering the floor admits blocks holding as few as m events in a
sub-vocabulary, and a 1159-dimensional carrier vector built from five events
is a far noisier point on the sphere than one built from ten. **Pool size and
per-block precision trade against each other on a fixed-K block, and the
registration's width projection priced only the first.** The #71/#79
convention itself HELD — the number was stated before the run and is reported
against — but the model behind the number did not, and that is the reusable
lesson: a pool-size projection is not a width projection unless the
per-observation precision is held fixed across the configurations compared.

### Gates

- **Cache anchor gate PASS**: 3,005,360 events / 1401 authors / 1191
  vocabulary, verified before any computation.
- **GATE ANCHORS (#78) PASS, EXACTLY.** The gate's executed predicate was
  re-executed here on census data: q=0.5/m=5 → **179,107 pairs / 424 authors**
  (registered 179,107 / 424) over 27,485 eligible blocks; q=0.7/m=10 →
  **108,716 / 401** (registered 108,716 / 401) over 19,571 blocks. Both clear
  the #69 targets. This is the convention the planner adopted after U2b, run
  as a blocking anchor rather than a projection.
- **G0 (#56) PASS, BITWISE.** U2b's whole four-row table at q=0.5/m=10 —
  including all five taste folds — was recomputed here through the imported
  estimator and compared field by field against
  `results/m4_u2b_persistence_budget/rows.json`: **max absolute difference 0.0
  across 60 fields (15 per row × 4 rows), every field bit-identical**,
  including the curves, both CIs, the null bands, D, the floor shares and the
  permutation p-values.
- **Split census PASS**: universe 2,348,361; Common(0.5) = 32 at 0.5036;
  Common(0.7) = 104 at 0.7008; pool 849 authors / 45,731 blocks.
- **Taste-row fold purity PASS** at all three configurations: 15 folds
  (5 × 3), 679/679/679/679/680 training authors, zero train/test overlap, the
  mass identity exact to 1e−06 in every fold with the test authors' mass
  entirely absent.
- **Sub-vocabulary count recovery PASS**: counts recovered as K·f², maximum
  deviation from integrality 1.06e−05.
- **ID-leak scan PASS**: 0 of 1401 cohort IDs in the committed set.
- **No synthetic gate**, as registered (R layer, no world simulated).
- Suite 1115 → 1148 (33 new contract tests). SEED 20260818, B_perm 499, B_boot 1000; the run is
  deterministic (re-executed end to end, identical to the last digit).

### Registered leans

- **L1 (disclosed, not a prediction) — SIGN HELD IN POINT, UNRESOLVED IN
  INTERVAL.** Λ = +0.0741/y against the disclosed +0.10/y, the right sign and
  the right order of magnitude, with the interval straddling zero exactly as
  the registration warned it might.
- **L2 HELD.** λ_taste is the smallest rate of the four rows in both arms
  (0.1736 and 0.1515 against a projected 0.16).

### Line state

U2c DELIVERED A VERDICT, and the verdict is that the design cannot resolve the
sign at this resolution. What the leg does deliver: a ratio-free estimand that
behaves (four log-linear fits with R² between 0.913 and 0.999, zero dropped
bootstrap replicates, every fitted E(b) ≥ 0.2033); a bit-exact G0
reproduction of U2b's entire four-row table (60 fields at 0.0); the #78
convention executed as a blocking anchor and passing exactly; **L2 held in its
rate form in both arms — the low-rank taste coordinate is the slowest layer,
which is the one piece of the three-layer picture that U2c CONFIRMS**; and two
findings the planner owns — the log-slope null's domain problem (RD-U2C-1) and
the pool-size-is-not-width anomaly. The T-line proposition's When clause
remains OPEN: the distinctive tail decays faster than the common mass at the
point estimate of every configuration run so far (five in U2b, two here), and
no interval yet excludes zero.

## U2c planner adjudication + measurement-phase close (2026-08-18)

**Verdict ACCEPTED: `SIGN_UNRESOLVED` — the interval statement, exactly the
outcome the registration named as live.** Λ = +0.0741/y [−0.0558, +0.1854]
primary; confirmatory +0.0861/y [−0.0109, +0.1609] (misses COMMON_STANDING
by 0.011/y); zero #73 flags. The formalized evidence state: **seven of seven
configurations across U2b+U2c put the point estimate on the same side
(the distinctive decays faster), and no registered interval excludes
zero.** The T-line proposition's When clause therefore stays OPEN. Its
resolution path is the SECOND CORPUS (standing charter) — a same-corpus
power escalation (U2d) is PARKED WITH PRICE: escalating designs on the same
data until an interval clears zero is significance-chasing, not measurement.

**What DID close: the outer comparison of the three-layer structure.**
λ_taste is the smallest rate of the four rows in BOTH arms (0.1736 and
0.1515 /y, against full 0.294/0.275) — L2 confirmed in its rate form. The
low-rank taste direction is the slowest object measured in this line,
consistent with T3's identity budget (the generalizing coordinate) and U2's
core. The middle comparison (common vs distinct) is the open edge.

**Defect #80 (planner, purchased; registry twenty-eighth note).** Two parts,
both mine. (a) The registered null on Λ was posed on a LOG-scale slope
whose domain the permutation destroys — E(b) is driven through zero under
reassignment, so only 7/499 permutations were defined; #79's convention
moved the pathology from the ratio into the log. The executor's RD-U2C-1
(positivity-free linear-slope companion, 100% defined; no verdict rested on
either null) is the correct in-leg repair. Convention: a null on any
transformed-scale statistic verifies the transform's DOMAIN under the null
at registration and ships a domain-safe companion. (b) The width projection
modeled pool counts but not per-unit noise: m = 5 enlarged the pool exactly
as assumed (+21.8% authors) yet WIDENED the distinctive row's interval
1.35× — a 1159-dimensional carrier vector built from five events is a
noisier point on the sphere than one built from ten. Convention: width
projections model per-unit measurement noise as a function of eligibility
floors, never pool counts alone.

**Discipline noted.** Gate anchors re-executed exactly (#78's convention in
its first full use); G0 bitwise over 60 fields; suite 1115 → 1148; zero
dropped bootstrap replicates; the run re-executed end-to-end identically.

**MEASUREMENT-PHASE CLOSE.** The When line's label-free measurement phase
closes with U1 (ORDER_CHANNEL), U2 (DRIFT_WITH_CORE), U2b
(POOL_GATE_UNMET, the gate honored), U2c (SIGN_UNRESOLVED, the honest
interval). Synthesis: `docs/SUICA_M4_U_LINE_SYNTHESIS.md`. **U3 (the
Who × When trait join) is the REGISTERED-NEXT beat, not opened today**: its
object is fixed by this phase — the cross-thread stay rate (U1's binding
designation) and slow-time coordinates, joined to Big5 distances only
inside a config-before-joint stamped stage-E design (S-line SR1/SR2
machinery), with the drift-aware caution that slow-time coordinates are
moving targets. Also named-open: the long-horizon floor (corpus-bounded),
the non-dwell order residual, the distinct-vs-common closure (second
corpus).

---

## U3 — the Who × When trait join (registered BEFORE run, 2026-08-18)

The line's first LABEL leg. Everything the measurement phase fixed is
inherited as binding: the cross-thread designation (U1 adjudication — the
primary-arm stay rate is 43.8% reply-chain mechanics and may never feed a
label join as free-selection order), the drift-aware caution (slow-time
coordinates are moving targets), and §5.4 (When coordinates are technical
objects — no psychological naming regardless of outcome).

### 4W header

- **Object.** Crossing #4 (Who × When): do the When coordinates the
  measurement phase validated — inertia (cross-thread stay), signature
  tightness, personal drift — couple to Big5 similarity, and do they add
  anything BEYOND the Where bag channel (S-line SR1's r = 0.049)? The
  increment is the question; the raw coupling is its precondition.
- **Fixed.** The U2 block pool (849 authors, 100% Big5-complete); SR1's
  stamp machinery (config-before-joint; stage_part0 stamps and opens no
  label; first_join logged; G-U3 proves stamp < first_join); SR1's primary
  trait geometry (z-scored 5-dim Euclidean) and declared label reliability
  0.8 (#57 family — disattenuation is a SECONDARY reading only); the bag
  distance (Hellinger full-signature cosine distance on the same pool).
- **Varied.** Raw vs bag-controlled (partial) coupling; the three
  coordinates; activity-controlled sensitivity.
- **What falsifies.** The dissociation family predicts silence: a raw
  Mantel r inside its own permutation band for every coordinate is the
  registered lean, not a failure. What would surprise is a surviving
  partial coupling (When beyond Where).
- **Layer.** P (the first label-bearing projection of the When quadrant).

### Census (planner arithmetic #43; predicates exact per #77/#78)

Computed on the U1 events cache; label table touched ONLY for availability
counts (null-structure, no values — the SR1-established census operation).

| quantity (exact predicate) | value |
|---|---|
| U2 block pool (≥ 4 disjoint K=50 in-vocab blocks) | 849 |
| … with all five Big5 columns non-null | **849 (100%)** |
| stay-eligible (≥ 30 cross-thread adjacencies per half; halves by full-stream median) | 847 |
| tight-eligible (≥ 2 consecutive-block pairs, midpoint gap ≤ 90 d) | 763 |
| drift-eligible (≥ 3 pairs gap ≤ 180 d AND ≥ 3 pairs gap > 365 d) | 652 |
| census split-half reliabilities (proxy for stay: subreddit-level states, DECLARED PROXY; exact for tight/drift) | stay 0.7828 / tight 0.8872 / drift 0.9383 |
| coordinate dispersions (sd) | 0.2531 / 0.1426 / 0.1830 |

Predicate lesson recorded (#77 family, no number — it governs nothing
here): "≥ 50 in-vocab events per half" yields 1028 authors under an
in-vocab-median half rule vs U1's 984 under the full-stream-median rule;
U3's pools are coordinate-driven and inherit neither.

### Coordinates (label-free, frozen in stage B; transforms pinned #70)

1. **stay_ct (PRIMARY — U1's designated object).** One GLOBAL state map:
   spherical k-means, C = 24 + OOV state, fit on all 849 authors' EARLY
   halves (full-stream-median rule), seed 20260818, n_init 10 — declared
   corpus-global (precedent: the SR0 vocabulary). stay_ct = mean
   1[state_i = state_{i+1}] over adjacent full-stream event pairs with
   DIFFERENT link_code (cross-thread only).
2. **tight.** Mean Hellinger cosine over consecutive-block pairs with
   midpoint gap ≤ 90 d (U2 blocks).
3. **drift_pa.** Mean cosine (pairs ≤ 180 d) − mean cosine (pairs > 365 d)
   over the author's own block pairs.

**Reliability gate (label-free, BEFORE the stamp'.s joint stage):** each
coordinate enters stage E only if its split-half reliability on the ACTUAL
objects (stay_ct at C = 24 early/late halves; tight and drift odd/even
pair splits) is ≥ 0.5. A coordinate failing the gate is excluded
label-free and reported. If the PRIMARY coordinate fails, the leg STOPS
before any stamp (clean stop, no label touched, verdict
COORDINATE_UNRELIABLE). Census proxies project all three far above the
gate.

### Estimands (ratio-free #79; own nulls #68/#66)

Per admitted coordinate c, on the coordinate's eligible pool:
- **Raw coupling:** Mantel r between |c_u − c_v| and SR1's z-scored Big5
  Euclidean distance; permutation of author label-rows, B = 999 (the
  Mantel permutation IS the own null); two-sided band.
- **Increment (the U3 question):** partial Mantel r(c-dist, trait-dist |
  bag-dist), Smouse–Long–Sokal residual permutation, B = 999; bag-dist =
  Hellinger full-signature cosine distance on the same pairs. Sensitivity:
  additionally controlling |log n_u − log n_v| (activity; SR2 found
  observability trait-NULL on this corpus, named regardless).
- Disattenuation (SECONDARY reading only, SR1 pattern): observed r /
  sqrt(rel_c^SB · 0.8) with rel_c the measured split-half
  (Spearman–Brown), label reliability DECLARED 0.8.
- Projection (#79b, assumption declared): scaling SR1's realized z = 5.42
  at N = 1306, r = 0.049 by z ∝ r·√N gives minimal detectable
  r ≈ 0.022 at N = 849 (stay pool 847). Reported against realized bands.

### Multiplicity and cells (NULL-first #55, effect-size keyed #75)

The verdict keys on the PRIMARY coordinate (stay_ct). tight and drift_pa
are secondary rows; their detections carry a Bonferroni ×3 guard and flag
but never route the verdict.

1. **WHEN_TRAIT_SILENT** — stay_ct raw r inside its band; scoped statement
   "silent beyond r ≈ 0.022" attaches only with the realized-band width
   report (#71 executed form; no equivalence cell beyond this).
2. **WHEN_COUPLED_REDUNDANT** — stay_ct raw r outside its band AND partial
   r inside its band (the When coordinate reads the bag's trait channel
   through a scalar).
3. **WHEN_COUPLED_INCREMENTAL** — stay_ct partial r outside its band (When
   adds trait information beyond Where; the informative surprise).

Secondary-row cells reported in the same scheme with flags. SR1's
r = 0.049 is the declared effect-scale anchor for all rows.

### Registered leans

- Primary: **SILENT-to-REDUNDANT** for stay_ct — the dissociation family's
  fifth test (identity carriers keep being trait-silent; stay is an
  identity carrier). Point |raw r| ≤ 0.03.
- Secondary (weak): if any row detects raw, it is tight or drift (slow
  coordinates aggregate more signal than one fast scalar) — structural
  lean only, no trait-level lean (§5.4).

### Stage discipline and deliverables

Stages: Part 0 (anchors: cache, U2 pool predicate, census reproduction;
config hash + STAMP; no label opened) → stage B (global map, coordinates,
reliability gate, coordinate file frozen + hashed into the stamp chain) →
stage E (ONE join: author_profiles.csv Big5 columns opened once,
first_join logged, all registered quantities computed) → G-U3 (stamp <
first_join proof in the artifacts). A1 binds: instrument-gate failures
before the stamp mean NO stamp ever.

Standard six: `scripts/run_suica_m4_u3_when_trait_join.py` + tests
(stamp-order contract, gate exclusion logic, Mantel/partial on toy data
with known structure, map purity = label-free inputs only, ID-leak
helper); gitignored `results/m4_u3_when_trait_join/`; report
`reports/SUICA_M4_U3_WHEN_TRAIT_JOIN_REPORT.md` (rule-24 tables; the
label event named; boundaries: EXPLORATORY corpus-level, no person claims,
no psychological naming of coordinates, the eq-12 projection caution);
outcome appended here; one CLAIMS_LEDGER row naming the label event
(PANDORA Big5 re-join under stamp, SR1/SR2 class); ONE commit
`feat(m4-u): U3 — the Who × When trait join — <VERDICT>`; suite green
(1148 + new). SEED = 20260818; B = 999 both nulls; B_boot n/a (Mantel
bands are the uncertainty machinery; no bootstrap CI on r is registered).
The planner records the label event in the private-repo audit log at
adjudication (planner-side action, not the executor's).

---

## U3 outcome (executor, 2026-08-18)

**VERDICT: `WHEN_TRAIT_SILENT` (cell 1).** The primary coordinate's raw
Mantel r is **−0.0036**, band [−0.0231, +0.0213], p = 0.7430, over 847
authors and 358,281 pairs. The partial (bag-controlled) r is −0.0125, band
[−0.0217, +0.0202], p = 0.2570 — also inside its band, so the cell is
SILENT and not REDUNDANT. Both secondary rows land in the same cell with no
detection at any correction level. Harness
`scripts/run_suica_m4_u3_when_trait_join.py`, report
`reports/SUICA_M4_U3_WHEN_TRAIT_JOIN_REPORT.md`, artifacts (gitignored)
`results/m4_u3_when_trait_join/`.

### The stamp chain (G-U3, from the artifacts)

| event | utc | sha256 (16) |
|---|---|---|
| config stamped | 2026-08-18T06:56:01.975215+00:00 | `8c399494dfbba293` |
| coordinate table frozen | 2026-08-18T06:56:16.189523+00:00 | `4a4aa07100c8c473` |
| FIRST JOIN | 2026-08-18T06:56:16.267746+00:00 | — |

`stamp < freeze < first_join` = **true** (+14.2 s, then +0.1 s); joint
quantities before the stamp 0; labels opened before the stamp false; both
hashes re-verify; ID-leak scan 0 hits over 1,398 cohort names. The frozen
coordinate table's hash is **bit-identical to an earlier label-free
rehearsal of stages part0+B**, which is the determinism claim in its
strongest available form: the coordinates did not move when the labels
arrived.

### Part 0 anchors, and RD-U3-1

Cache anchors exact (3,005,360 / 1401 / 1191). Pool census exact:
849 / 847 / 763 / 652.

**RD-U3-1 (disclosed re-posing, decided label-free in part 0, before the
stamp).** The registered prose predicate "≥ 30 cross-thread adjacencies per
half" admits **848** authors from this cache while the registered census
names **847**: exactly ONE author sits at min(early, late) = 30 (the next
lowest is 27). Under the registration's own hierarchy — "planner arithmetic
#43; predicates exact per #77/#78" — the pool the census NAMES is the
registered object, so U3 adopted the STRICTLY TIGHTER `> 30`, which
reproduces 847 exactly and drops the boundary author. This is the same
direction as U1's RD-U1-1/RD-U1-2 (each stricter than the registered form).
The looser 848-author pool is recorded in `part0.json` and routes nothing.

### The label-free reliability gate (all three admitted)

| coordinate | split-half r | Spearman–Brown | gate (≥ 0.5) | pool | mean (sd) |
|---|---|---|---|---|---|
| `stay_ct` (PRIMARY) | 0.7685 | 0.8691 | ADMIT | 847 | 0.5413 (0.2037) |
| `tight` | **0.8872** | 0.9402 | ADMIT | 763 | 0.7718 (0.1427) |
| `drift_pa` | **0.9383** | 0.9681 | ADMIT | 652 | 0.2565 (0.1831) |

`tight` and `drift_pa` reproduce the registration's census reliabilities to
four decimals (0.8872 / 0.9383) — the exact-predicate census was exact.
`stay_ct`'s measured C = 24 value is 0.7685 against the census's DECLARED
PROXY 0.7828 (subreddit-level states), i.e. the proxy was honest and mildly
optimistic. Global map: C = 24 + OOV, 1181 of 1191 communities active, 10
zero-mass communities to OOV, cluster sizes 11–100.

### The joined estimands (stage E, one join)

| coordinate | raw r [band] p | partial \| bag [band] p | + activity | + bag² (2nd) | disatt. raw (2nd) |
|---|---|---|---|---|---|
| `stay_ct` | −0.0036 [−0.0231, +0.0213] 0.7430 | −0.0125 [−0.0217, +0.0202] 0.2570 | −0.0124 p 0.2490 | −0.0127 p 0.2380 | −0.0043 |
| `tight` | −0.0087 [−0.0251, +0.0254] 0.4930 | −0.0132 [−0.0252, +0.0237] 0.2950 | −0.0131 p 0.3280 | −0.0132 p 0.2910 | −0.0100 |
| `drift_pa` | −0.0042 [−0.0313, +0.0313] 0.7780 | −0.0111 [−0.0283, +0.0304] 0.4560 | −0.0110 p 0.4670 | −0.0116 p 0.4500 | −0.0048 |

Every point estimate is NEGATIVE and every one is inside its band. All nine
z-scores against their own nulls sit between −0.272 and −1.211. Big5
completeness on the 849 pool: **849/849 = 100%**, reproducing the census.

### THE POSITIVE CONTROL THAT MAKES THE SILENCE READABLE

On the SAME pairs, with the SAME trait matrix and the SAME Mantel
machinery, the WHERE channel is live: Mantel(bag-distance, trait-distance)
= **0.0490** on `stay_ct`'s pool (0.0494 on `tight`'s, 0.0559 on
`drift_pa`'s) against SR1's independently established **0.049** on a
different pool of 1306. A descriptive point estimate with no null run —
it is not a registered estimand of this leg and routes nothing — but it is
the reading under which the When rows' bands mean what they say. The
harness finds the coupling it is known to find, in these very pairs, and
finds none through any When coordinate. The coordinates are not
bag-degenerate either: corr(coordinate-distance, bag-distance) is 0.177 /
0.090 / 0.120, so the three When scalars are largely ORTHOGONAL to the bag
and still carry no trait signal of their own.

### Projection versus realized width (#71 executed form)

| coordinate | N | projected MDR at this N | realized 1.96 × null sd | ratio | band half-width |
|---|---|---|---|---|---|
| `stay_ct` | 847 | 0.0220 | 0.0220 | **1.000** | 0.0222 |
| `tight` | 763 | 0.0232 | 0.0254 | 1.094 | 0.0252 |
| `drift_pa` | 652 | 0.0251 | 0.0300 | 1.198 | 0.0313 |

The primary row's realized width matched the registered projection to
three decimals — the first width projection in the U line that did not
miss (contrast U2c's 1.21× and defect #80b). The scoped statement
"**silent beyond r ≈ 0.022**" is therefore carried at exactly the width it
was registered at, and attaches only to this table.

### Leans

- **Primary lean SILENT-to-REDUNDANT: HELD, on the SILENT side.** Point
  lean |raw r| ≤ 0.03: HELD with room to spare, |−0.0036| = 0.0036, an
  order of magnitude inside the leaned bound and 7.4% of SR1's anchor.
- **Secondary weak lean: NOT ACTIVATED.** No row detected raw, so the
  "if anything detects, it is a slow coordinate" prediction was never put
  to a test. Recorded as untested, not as held.

### Honest anomalies and defects the run surfaced

1. **The SLS partial controls the bag channel LINEARLY.** A shared
   NON-LINEAR dependence of both distances on the bag distance survives it
   — the contract tests exhibit exactly such a world (a uniform latent
   whose small-distance noise floor is shared by both matrices leaves a
   linear-residual partial of 0.26 where the truth is redundancy). An
   unregistered `bag + bag²` second reading was therefore computed and
   reported for every row; it moves the primary partial from −0.01246 to
   −0.01270, i.e. nothing. Offered to the planner as a candidate
   convention: **a partial that controls a covariate linearly must ship a
   non-linear companion, or declare the linearity as an assumption.**
2. **The registration's trait geometry is stated two ways.** The Fixed
   block inherits "SR1's primary trait geometry (z-scored 5-dim
   Euclidean)" while SR1's harness in fact used negative SQUARED Euclidean.
   U3 took the registration's literal estimand text (Euclidean DISTANCE)
   as primary and shipped the squared form as a second reading: −0.0036 vs
   −0.0036 on the primary row. The choice was worth nothing here, but the
   ambiguity should not have needed an executor's ruling.
3. **The 1-author census boundary (RD-U3-1 above).** Recorded as an
   instance of the #77/#78 family: an inequality written in prose and an
   inequality executed in arithmetic disagreed at the exact boundary.
4. **`freeze → first_join` is 0.1 s.** The ordering is proved from
   timestamps, and 0.1 s is a true ordering, but it is thin. The stronger
   evidence is structural and is the one to cite: the config hash and the
   coordinate hash both re-verify after the join, and the coordinate
   artifact is bit-identical to a label-free rehearsal.

### What this closes, and what it does not

The dissociation family's FIFTH test returns the family's answer. Identity
carriers keep being trait-silent: the T-line's tree code, the S-line's
γ = 0 arm, U1's order channel, and now all three When coordinates. The
When quadrant's first label-bearing projection of K_u is silent at a
corpus-level width of r ≈ 0.022 while the Where channel is live at 0.049
in the same pairs.

It does NOT close: anything about persons (corpus-level only); anything
about a psychological reading of stay/tight/drift (§5.4 binds regardless of
outcome — the null does not license "inertia is not a trait", only "this
scalar of this author's own stream does not covary with Big5 distance at
this width"); anything beyond ONE first-order projection of K_u per
coordinate (eq 12); and anything about coordinates measured over a fixed
window rather than each author's own span (the drift-aware caution).

## U3 planner adjudication + FULL LINE CLOSE (2026-08-18)

**Verdict ACCEPTED: `WHEN_TRAIT_SILENT`.** All nine registered estimates
(raw / bag-partial / activity-partial × three coordinates) are negative,
small, and inside their own permutation bands (primary raw r = −0.0036,
p = 0.74; |raw| = 7.4% of the 0.049 anchor). The silence is READABLE
because the executor carried the live-channel anchor: Mantel(bag-dist,
trait-dist) = 0.049014 on the SAME pairs and machinery — SR1's 0.049
independently reproduced — so the Where channel is live while every When
coordinate is mute. And the coordinates are not bag-degenerate
(corr with bag-dist 0.09–0.18): genuinely different scalars, genuinely no
Big5 signal. Scoped statement carried at exactly the registered width:
silent beyond r ≈ 0.022 — the projection ratio 0.9998 is the U line's
first width projection that did not miss (#80b's convention paying off on
its first use).

**The dissociation family's FIFTH confirmation, now covering When.**
Inertia (the order-identity carrier), tightness, and personal drift are
all trait-mute at corpus power. The theory's §5.4 restraint — dynamic
objects are "reproducible text-behavior processes," not personality
factors — is now empirically vindicated for Big5 on every dynamics
coordinate measured. Crossing #4's candidates exist as author structure
and have shown no Big5 content in this projection family.

**Stamp discipline.** G-U3 PASS with the stronger evidence noted: the
frozen coordinate artifact is bit-identical to a label-free rehearsal —
the coordinates did not move when labels arrived. The label event is
recorded in the release ledger row and in the private-repo audit ledger
(planner-side, private commit c6f0ed3).

**Executor deviations adjudicated.** RD-U3-1 ACCEPTED (the census-named
847 routes; the prose/arithmetic boundary disagreement at exactly one
author is the census-boundary family's third instance, folded into #81).
The two unregistered second readings (bag+bag² partial; squared-Euclidean
geometry) route nothing and are accepted as diagnostic notes.

**Defect #81 (planner, purchased; registry twenty-ninth note).**
Registration ambiguities that required executor rulings, twice in one
leg: (a) the trait geometry was pinned BY NAME ("SR1's z-scored Euclidean")
while SR1's harness actually used negative squared Euclidean — the literal
estimand text and the named source diverged (outcome-invariant here,
−0.003578 vs −0.003601, but the ruling should not have been needed);
(b) the prose predicate "≥ 30" vs the census arithmetic "> 30" disagree at
exactly one author. Convention: inherited machinery is pinned by QUOTED
FORMULA (with file+line provenance), never by name alone; eligibility
predicates and their census values are generated from the same code.

**Defect #82 (planner, purchased; same note).** The registered partial
(Smouse–Long–Sokal) controls the covariate LINEARLY; the executor's toy
suite exhibits a world with pure redundancy where the linear-residual
partial reads 0.26 — the REDUNDANT/INCREMENTAL cell boundary is
assumption-laden. Routes nothing here (everything silent at raw), but the
convention is purchased: a partial controlling a covariate linearly
declares linearity as an assumption or ships a nonlinear companion
(rank/spline residual) whenever the partial can route a verdict.

**THE WHEN LINE IS FULLY CLOSED** (measurement phase U1/U2/U2b/U2c +
label phase U3). The quadrant's answer, in one line: **the When channel
carries who you are, and — at this corpus's power, in every projection
measured — not what you are like.** Named-open unchanged: second-corpus
transport (owns the distinct-vs-common closure and now also the
transport of the When silences), long-horizon floor, non-dwell order
residual, R3 reconciliation.
