# GRAIN M4-S — The Selection Line

Line opened 2026-08-16, by the owner's direction: pursue the
selection conjecture on BOTH synthetic and real data. The
conjecture (the owner's, arriving from the identity era's mechanism
results): **the gauge reads frames — but WHICH frames a person
chooses is person-owned; selection-proximity should imply
personality-proximity.** This is the rind model's choice channel
returning with a mechanism: the context meter becomes a person
reader when the person picks the contexts.

The claim decomposes (adjudication standard for every S-leg):
**(a)** selection is a person-stable signature — already supported
at the program's strongest real-data tiers (choice-axis stability
5/5 holdout; H6 politics/news → openness, lockbox-confirmed
r=+0.096 q=.006); **(b)** selection-similarity ⇒ TRAIT-similarity —
NOT a theorem but a coupling strength, false in worlds where
selection is driven by non-trait identity. The synthetic track
measures (b) as a LAW OF THE COUPLING (with the γ=0 falsification
arm); the real track measures (b) as a corpus fact with the
direction geometry (T8) handled.

Tier: EXPLORATORY throughout. The ledger controls. Standing rules
1–33 and ALL conventions bind (#43 planner-arithmetic; #56
inheritance; #57 pilot second moments; #59/#60 estimand
non-degeneracy and shared components; #61 prediction bands; #62
channel coverage; #63 frame-controlled forms; #64 sign
conventions). Execution conventions as always; `suica_core/`, k2b,
P3b/v2 instruments READ-ONLY.

## REAL-DATA GOVERNANCE (binding on every SR-leg)

`docs/SUICA_REALTEXT_GOVERNANCE.md` R-G1..R-G8 in force: **no
person claims, ever** (corpus-level aggregate statistics only); NO
cross-corpus author linkage; committed artifacts and reports carry
NO user text, NO user IDs, aggregates only (`results/` stays
gitignored regardless); the Essays confirm-half LABELS remain
sealed and untouchable (one lockbox opening remains); the native
corpus stays paused (owner 2026-07-12; not unpaused by this line);
identity-flavored stability claims carry split-half (T6″-pattern)
forms. Data access per `docs/DATA_ACCESS.md`; sources read from the
private repo paths, never copied into this repo.

## Line charter

- **S1 — the choice-enabled generator** (synthetic; registered
  below): endogenous context selection with a coupling knob γ
  (trait-driven ↔ style-driven), certified.
- **S2 — the transfer law** (synthetic; register after S1): sealed
  r_transfer(γ) curve with the γ=0 null and T8 direction handling.
- **SR0 — the real-data reconnaissance** (registered below;
  artifact/data-manifest only, STRICT no-peeking): pin the
  selection-signature objects (per-user subreddit distributions /
  the 12 choice axes), gold labels, split-half feasibility,
  reliability ceilings, and SR1's power analysis — computing NO
  selection×trait statistic of any kind.
- **SR1 — the selection-geometry test on PANDORA** (register ONLY
  after SR0): the Mantel-style pairwise test, sealed, permutation
  null, direction geometry, R-G compliant.

---

## M4-S1 — the choice-enabled generator

**REGISTERED 2026-08-16, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. Instrument leg: certificates
only; the sharp law is S2's.

### The extension (rules 9/12 — minimal extraction with provenance; the v2 lineage)

`build_choice_world(author_seed, frame_seed, phi_slow, w_style,
beta, gamma)` inside `scripts/run_suica_m4_s1_choice_generator.py`,
extending the v2 builder (imported by file, hashes verified).
Mechanism: per-author preference over the family's 4 contexts,

    pi_a = softmax( beta · [ gamma · u(trait_a) + (1−gamma) · v(style_a) ] )

with u, v PINNED deterministic 4-dim projections (executor pins
them from the world's own loadings-derived coordinates, file:line,
decision rule: the first four principal author-coordinates of each
channel, orthonormalized; disclosed in Part 0). The author's
occasion-to-context exposure is drawn from pi_a (the selection
mechanism IS frame-exposure choice — mechanistically the point);
beta = 0 recovers uniform exposure. The executor maps the layout /
common-channel indexing exactly (P3's report already charted it)
and pins the minimal exposure-reallocation that preserves the
shared-frame objects per context; if author-specific exposure is
structurally impossible without editing k2b, STOP as
**INFEASIBLE_CHOICE** (an instrument finding, the P3 pattern).

### Certificates (the leg IS the battery; #61-compliant bands everywhere)

- **C-S1a (neutral anchor):** at beta = 0, field levels replicate
  the v2/M1c share-.25 row distributionally (2·√2·SEM per cell,
  the C1′ pattern); selection frequencies uniform within
  multinomial bands.
- **C-S1b (signature):** at beta = beta* (pinned in Part 0 so that
  realized selection entropy drops by a declared, detectable
  margin — arithmetic from pi, no worlds), realized per-author
  frequency vectors match pi_a within multinomial noise
  (per-author χ² within band), and the signature is
  SPLIT-HALF STABLE (occasion-split correlation of frequency
  vectors, band from multinomial arithmetic).
- **C-S1c (coupling placement):** ACROSS authors within worlds,
  corr(selection-similarity, trait-similarity) is POSITIVE at
  γ = 1 and ZERO at γ = 0 (equivalence band from permutation
  arithmetic) — with γ = 0 worlds still showing the SAME
  selection-signature stability (identity-driven selection: stable
  but trait-uninformative — the (b)-decomposition made physical).
  Generous certificate bands; the quantitative law is S2's.
- Non-degeneracy (#59): none of C-S1b/c is forced — exposure draws
  are stochastic and the couplings live in u, v, γ; the shared
  component of any A/B comparison is named (#60): author-stream
  objects only.

### Design

γ ∈ {0, 1} × 24 worlds each + beta = 0 anchor arm 8 worlds (56
worlds total; each world carries the full 985-author panel, so
author-pair statistics are abundant). share 0.25, φ 0.60,
w_style = 1.0 (the identity channel ON — γ = 0 needs a live
non-trait driver). Salts `m4s1-author` / `m4s1-frameA` /
`m4s1-pilot`, master_seed 20260816. Rule-29 predicates; #57 bands
(variances only, margin 1.25); projection gate (rule 25): C-S1c
detection power ≥ 0.8 at γ = 1 with false-fire ≤ 0.1 at γ = 0,
from pilot 4 worlds/arm; once-only escalation to 48 worlds/arm ON
THIS GATE. Stages < 600 s each; target < 45 min.

### Routing (rule 16 — disjoint, covering, entailed)

| # | condition | outcome |
|---|---|---|
| 1 | G0/import/hash failure | **STOP** |
| 2 | exposure reallocation structurally impossible | **INFEASIBLE_CHOICE** (instrument finding; redesign handback) |
| 3 | projection fails after escalation | **NON_PROJECTABLE** |
| 4 | all certificates PASS | **CHOICE_GENERATOR_CERTIFIED** — S2 registrable |
| 5 | any certificate fails | **INSTRUMENT_DEFECT(name)** — handback |

Deliverables: the six as always;
`reports/SUICA_M4_S1_CHOICE_GENERATOR_REPORT.md`; ONE commit
`feat(m4-s): S1 — the choice-enabled generator — <SLUG>`.

---

## M4-SR0 — the real-data reconnaissance (STRICT no-peeking)

**REGISTERED 2026-08-16, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. Artifact/data-manifest leg —
**it computes NO statistic linking selection to any label**, and
its script's Part 0 enumerates every quantity it is allowed to
produce (the no-peek gate is CODE, not intention).

### Mandate

1. **Sources pinned** (paths, sizes, columns; NO content beyond
   schema + row counts in the report): the raw PANDORA comment
   table (`data_sets/PANDORA_official/all_comments_since_2015.csv`
   — verify the subreddit and timestamp fields exist),
   `author_profiles.csv` (gold labels; verify the Big5 columns and
   the N=1401 gold cohort's identifiability BY COUNT ONLY), the
   prepared Big5 CSV (1401 rows), and the choice-channel lineage
   code in this repo (`scripts/suica_v2_lib.py` and the v2-era
   choice constructions; pin file:line of the 12-axis constructor
   or report NOT_FOUND honestly).
2. **The selection-signature object DEFINED** for SR1: per-user
   subreddit frequency vectors over a pinned vocabulary (frequency
   floor declared from counts alone), plus the 12-axis projection
   if its constructor is located; sparsity and coverage statistics
   (selection-side ONLY).
3. **Split-half feasibility:** timestamp-based halves per user
   (T6″-pattern); per-user comment counts distribution; the
   stability CEILING estimated from selection-side split-half
   agreement (labels untouched).
4. **SR1 power analysis:** N, pair counts, Mantel permutation
   plan, minimum detectable Mantel-r at the declared α — from
   selection-side reliabilities and label-side variances computed
   SEPARATELY (never jointly).
5. **R-G compliance block:** the report carries aggregates only;
   no user IDs, no text excerpts, no per-user rows; Essays and
   native corpus untouched; no cross-corpus linkage.

### Hard no-peek gate

The script computes selection-side objects and label-side
MARGINALS (variances, counts) in SEPARATE stages that never join;
any joint selection×label quantity is structurally absent from the
code (G-SR0 verifies by enumeration of produced artifacts; rule 24
tables generated). Violation = the leg is void.

### Routing

| # | condition | outcome |
|---|---|---|
| 1 | a pinned source missing/schema-broken | **SOURCE_GAP(name)** — the manifest reports what exists; SR1 blocked pending owner/planner |
| 2 | all sources pinned, signature object defined, power adequate | **SR1_READY** — SR1 registrable |
| 3 | sources pinned but power inadequate at N | **UNDERPOWERED_DESIGN** — the analysis says what N or reliability would suffice; SR1 blocked |

Deliverables: the six as always;
`reports/SUICA_M4_SR0_RECON_REPORT.md` (aggregates only); ONE
commit `feat(m4-s): SR0 — real-data reconnaissance — <SLUG>`.
Budget: no world generation; data reads chunked (the comment table
may be large — streamed counting, memory-bounded, stages < 600 s).

### Outcome of M4-S1 (executed 2026-08-15, append-only)

**`INSTRUMENT_DEFECT(C_S1a)` (rule-16 cell 5).** Modifiers:
COUPLING_PLACED_GAMMA1_POSITIVE_GAMMA0_NULL, SIGNATURE_STABLE_AT_BOTH_GAMMA. **The generator works and the coupling is placed;
the failing certificate is C-S1a, and the defect is in that certificate's
specification rather than in the apparatus.**

- **The structural question is settled by construction, not assumption.**
  emit_panel (k2b:359-381) hands author i the frame `common[ctx_index[i], o]` at
  EVERY occasion, so per-occasion choice is not expressible through it and
  INFEASIBLE_CHOICE was live. `emit_choice` transcribes emit_panel with **exactly
  one change** — the first index of the common term (k2b:377) becomes
  `choice[i, o]`. Certified in Part 0: with `choice[i,o] = ctx_index[i]` the
  panel is **bit-exact** (True), the
  truth panel is bit-exact (True) and the field
  agreement is identical (0.11118696412429328 vs
  0.11118696412429328). The shared-frame object per context is preserved:
  `common[k, o]` remains the one vector for context k at occasion o. Every v2
  object stays bit-identical (True) because the
  exposure draw uses its own rng stream taken after the builder returns.
- **u / v pins (RN-S1-2).** First four principal author-coordinates per channel:
  SVD of the author-centred channel matrix (`trait_pure` for u, `style` for v),
  right singular vectors 1..4, authors projected, each coordinate z-scored so
  beta means the same thing in both channels. **Sign convention (#64) pinned:**
  each component's sign fixed so its largest-magnitude loading is positive — SVD
  signs are otherwise arbitrary and would silently randomize pi across worlds.
  The panel has exactly 4 contexts (AskReddit, AskWomen, politics, worldnews),
  matching the 4-dim projection.
- **beta\* = 1.7, fixed by arithmetic with NO worlds.** Declared
  criteria: mean selection entropy <= 0.85·log k
  AND median per-author chi-square(3) power >=
  0.8 at the panel's median m =
  12.0; smallest beta on the pinned grid meeting
  both. Realized at beta*: entropy fraction 0.6198984954321972,
  median power 0.8069427664188857 — **the power
  criterion binds, not the entropy one** (entropy is far below its target at the
  selected beta).
- **C-S1a FAILS its level clause, PASSES its frequency clause.** Anchor level
  0.09262907190295885 [0.07652910892609367, 0.10785388737580495] against the routing reference
  0.11806906162144237 (R2's persisted v2 arm at the identical share .25 /
  phi .60 / w_style 1.0) — deviation **-0.02543998971848352** against a 2·√2·SEM
  band 0.024972074968206703 (z -2.8814248341250366). Frequencies uniform: mean
  uniformity chi-square 3.0881948266373147 vs crit95
  7.8147279032511765, 1.0 of worlds within.
- **The failure is robust under every reference and both band readings.**
  Deviation vs M1c's no-style row -0.03451883246292889;
  vs these same worlds' OWN layout exposure
  -0.03191225113985062 [-0.05123843785467711, -0.011515406750239305] (paired,
  CI excluding zero). Generous band 0.024972074968206703,
  strict band 0.017657923550317797; fails under both:
  True.
- **THE DEFECT IS IN THE CERTIFICATE, AND THE FIX IS ONE LINE.** C-S1a asks for
  uniform SELECTION (pi flat) and baseline EXPOSURE (the level the panel has when
  each author sits in one context) at the same beta = 0. Those are different
  neutralities. The deployed gauge pools authors by their NOMINAL context to
  estimate a per-context field, so spreading each author's occasions uniformly
  over four contexts destroys exactly the coherence it recovers — the level drop
  is the apparatus behaving correctly. Both readings are reported: **(A) uniform
  exposure** (the registration's explicit words) fails the level clause; **(B)
  beta = 0 as a no-op on the frame path** (`choice[i,o] = ctx_index[i]`) passes it
  EXACTLY — that is precisely what Part 0's bit-exactness certificate is.
  Handback: adopt (B) as the neutral anchor, or keep (A) and drop its level
  clause, which the registration's own mechanism predicts must shift. The anchor
  worlds' own layout-exposure level 0.06071682076310822
  sits within one SEM of the reference, so the worlds are sound.
- **C-S1b PASSES in both arms.** Realized frequencies track pi within multinomial
  noise: fraction of authors exceeding the 95th percentile of chi-square(3) =
  0.06460176991150442 at gamma 1 and
  0.06718289085545723 at gamma 0, against 0.05
  expected. Split-half stability 0.5629870740710835
  [0.5556645865206762, 0.5703783652507874] and
  0.5564373726309412 [0.5477653655574023, 0.5647096208307433].
- **C-S1c PASSES — and it is the certificate the line needed.** Mantel r between
  selection-similarity and trait-similarity across retained authors:
  **0.23983432331725474** [0.23709290162316124, 0.24284340944183067] at gamma = 1 and
  **-4.1883473632534314e-05** [-0.001165090846688084, 0.001053439915639846] at gamma = 0, against an
  equivalence band eps = 0.010125436066679785 from permutation arithmetic (permutation sd
  0.0022344399846873217). Separation
  0.23987620679088728. **The gamma = 0 arm's selection signature is just as
  stable as gamma = 1's (0.5564373726309412 vs
  0.5629870740710835) and carries no trait information whatsoever.**
  That is decomposition (b) made physical: **a perfectly stable selection
  signature can be completely uninformative about personality.** Stability is
  necessary and nowhere near sufficient — and this is the falsifier the real-data
  track will need, because SR1 could otherwise read (a) as evidence for (b).
- **Gates.** G0s1 PASS; G2s1 pilot predicates PASS; G3s1 PASS — detection power
  1.0 at gamma = 1,
  false-fire 0.0 at
  gamma = 0; escalation False.
- **Anomaly A-1 (before any number).** The pinned virtualenv was partially
  destroyed between legs: a temp reaper deleted files under `/private/tmp` at
  00:00, removing `pyvenv.cfg` and gutting `site-packages` — package directories
  survived but their `__init__.py` files did not, so numpy imported as an empty
  namespace package. Rebuilt from `requirements-lock-main.txt` and verified
  identical to the versions the previous legs ran under (numpy 2.4.4, pandas
  3.0.2, Python 3.12.12) BEFORE any S1 number existed.
- **One registration-defect candidate:** C-S1a's neutral anchor conflates uniform
  selection with baseline exposure (above). Non-blocking — it routes the leg to a
  handback exactly as the ladder intends, and the generator itself is certified on
  every other clause.
- Report: `reports/SUICA_M4_S1_CHOICE_GENERATOR_REPORT.md`.

### Planner adjudication of S1 (2026-08-16, appended after the run) — THE GENERATOR IS CERTIFIED BY DATED NOTE; THE FAILING CLAUSE WAS THE PLANNER'S CONFLATION

**INSTRUMENT_DEFECT(C_S1a) accepted as routed — and resolved by the
executor's dual reading. Defect #65 (mine):** C-S1a conflated two
neutralities — uniform SELECTION (π flat) and baseline EXPOSURE —
and demanded both at β = 0. Uniformizing exposure scatters each
author across frames, destroying exactly the frame-coherence the
gauge reads; the level drop (−0.03191225113985062
[−0.0512, −0.0115], paired against the same worlds' layout
exposure) is CORRECT behaviour predicted by appendix II's own
mechanism. Reading (B) — β = 0 as the frame-path no-op — is what
"neutral anchor" meant, and its evidence ALREADY EXISTS in this
leg's Part 0: panel, truth panel and field bit-exact at
choice = layout (0.09919789 = 0.09919789), every v2 object
bit-identical. **Dated note: C-S1a is re-typed to reading (B),
whose certificate is the leg's own Part-0 bit-exactness. With
C-S1b and C-S1c PASSED, the choice generator is CERTIFIED.** No
re-run is needed because no new quantity needs measuring — the
(A)-clause's failure is a planner specification defect, recorded,
not an instrument fault.

**Two findings worth carrying:** (i) the exposure-uniformization
level drop is a NEW mechanism-consistent reading (scattering frames
costs the gauge −0.032 at these knobs) — named THE SCATTER READING,
unregistered, adjudicating nothing, a candidate future tax channel;
(ii) C-S1c's γ = 0 arm is the line's treasure: **a selection
signature exactly as stable as the trait-driven one (split-half
0.556 vs 0.563) carrying ZERO trait information** (Mantel −4e-05
against ε = 0.0101) — decomposition (b) is now a measured physical
possibility, and SR1's real-data verdict must be read against it.

**Environment note:** the pinned venv was destroyed by the tmp
reaper between legs and rebuilt from the lockfile with identical
versions before any number (A-1) — the environment pin's value
demonstrated; future legs re-verify the interpreter as a matter of
course.

---

## M4-S2 — the transfer law (sealed shape of r(γ))

**REGISTERED 2026-08-16, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. The synthetic half of the
owner's conjecture, as a sealed quantitative law.

### The derived shape (planner theory; executed derivation and its error budget in Part 0)

Selection similarity is driven by the mixed vector γ·u + (1−γ)·v
with u ⟂ v (S1-certified independence); trait similarity by u
alone. Under the pinned pipeline the pairwise-similarity
correlation obeys

    r(γ) = r(1) · g(γ),  g(γ) = γ² / √(γ⁴ + (1−γ)⁴)

with r(1) the amplitude (attenuation through multinomial sampling
and the softmax pipeline — γ-independent, as C-S1b's equal
stabilities at both γ showed). Amplitude input: S1's persisted
r(1) = 0.23983432331725474 (G0 bit-verifies). Predicted interior
points (planner sanity, executor recomputes at full precision):
g(0.25) ≈ 0.1104, g(0.5) ≈ 0.7071, g(0.75) ≈ 0.9938 — a sharply
non-linear shape (near-flat top, collapsing knee) that no smooth
monotone guess reproduces by luck. Bands per #61: SE_pred (S1's
r(1) CI propagated) ⊕ SE_meas ⊕ SE_approx (the softmax/z-scoring
distortion budget, derived in Part 0 from the pipeline's own
arithmetic on probe-free objects; the R2b lesson on undersized
transport terms applies — the executor states the budget's
construction and its conservatism direction).

### Design

γ ∈ {0, 0.25, 0.5, 0.75, 1.0} × 16 worlds each (80 worlds; the
985-author panel gives abundant pairs). share 0.25, φ 0.60,
w_style 1.0, β = β* = 1.7 (S1's pinned value). Salts `m4s2-author`
/ `m4s2-frameA` / `m4s2-pilot`, master_seed 20260816. K2f
ordering: predictions for the three INTERIOR γ hashed BEFORE any
fresh world; the endpoints are re-measured as anchors (γ = 1
against S1's value distributionally; γ = 0 against the ε-null).

### Verdicts (rule 22; NULL-first; sides declared)

- **V-S2a [.75]:** r(0) NULL on fresh worlds (ε from permutation
  arithmetic as S1).
- **V-S2b [.55 for 3/3]:** the three interior measurements inside
  their sealed bands — 3/3 → **TRANSFER_LAW_SEALED**; 2/3 →
  **SHAPE_PARTIAL** (the miss names the distortion); ≤1/3 →
  **SHAPE_WRONG** (the analytic form dies; the curve is reported).
- **V-S2c (reading, the owner's direction question):** cosine-based
  vs distance-based selection similarity — the same Mantel
  quantities under both geometries, reported side by side (T8's
  expectation: cosine dominates; distance mixes magnitude).
- Anchor failures route **INSTRUMENT_DEFECT**, never a shape cell.

### Gates

G0s2 (S1 values bit-exact incl. r(1), β*, u/v pins, ε arithmetic);
G1s2 (C2-style battery; γ interior arms' π actually mixed — #59
non-degeneracy: interior g values are not forced by construction);
G2s2 pilot 4 worlds at γ ∈ {0.25, 0.75} (rule-29 predicates);
G3s2 projection (interior-band widths achievable at 16 worlds/arm;
once-only escalation to 32 ON THIS GATE); G4s2 (routing verified;
rule 24; stages < 600 s; target < 30 min).

Deliverables: the six as always;
`reports/SUICA_M4_S2_TRANSFER_LAW_REPORT.md`; ONE commit
`feat(m4-s): S2 — the transfer law — <SLUG>`. Dispatch order note:
SR0 (the real-data reconnaissance, already registered above) runs
BEFORE S2 — the real track takes its first leg; S2 follows.

### Outcome of M4-SR0 (executed 2026-08-15, append-only)

**`SR1_READY`.** **The first real-data leg of the identity era, and
no selection x label statistic was computed.** The no-peek gate is structural and
G-SR0 verifies it by enumeration: 9 artifacts,
0 unexpected, **0
joint selection x label quantities found**.

- **Sources all pinned; no SOURCE_GAP.** Comment table
  5389.8 MiB with `subreddit`
  (True), `created_utc`
  (True) and `author`
  (True) all present; `author_profiles.csv`
  10295 rows with all five Big5 columns
  (True); prepared Big5
  1401 rows / 1401
  unique user_id; `scripts/suica_v2_lib.py` present.
- **On the cohort count.** The registration's "1401 rows" is EXACT
  (True); a naive `wc -l` reports ~123k
  because the `text` column carries embedded newlines. Counted independently from
  `author_profiles.csv`, **1568** authors
  have all five Big5 values, and the prepared mainline cohort of
  1401 is a STRICT SUBSET of them
  (True, intersection
  1401). The registration's 1401
  is taken as canonical; the 1568 is
  reported as SR1 headroom, not as a discrepancy.
- **The 12-axis constructor: `FOUND (constructor); FITTED ARTIFACT ABSENT`.** It EXISTS and is LIVE, not a
  lost v2 artifact — `scripts/run_suica_e3_e4_choice_scale_class_react_v2.py` with `N_CLASSES = 12` (:37),
  `build_condition_classes()` (:43), KMeans (:63), the subreddit->axis map (:65),
  `choice_axis_scores()` (:81), axis naming (:126); the holdout refit that
  produced the 5/5 result is `scripts/run_suica_op6a_choice_axes_holdout_v3.py`. **But the FITTED artifact
  `results/suica_e3_e4_choice_class_v2_s128/condition_class_map.csv` is NOT on disk** (False)
  because `results/` is gitignored. Consequence for SR1: the primary signature
  object needs none of it, but the 12-axis projection is available only by
  REFITTING, and SR1 must not cite the existing 5/5 holdout as though the same
  fitted axes were in hand.
- **The selection-signature object, defined.** Floor declared BEFORE the stream
  from counts alone: a subreddit enters iff used by >=
  0.01 of cohort users =
  15 users. Streamed 17,640,062 rows,
  3,005,360 in cohort. **Vocabulary
  1191 of 15863 distinct
  subreddits; coverage 0.7813909148987143** of cohort comments.
  1306 users carry a signature (>= 20
  comments); mean 44.50076569678407 non-zero subreddits
  per user (sparsity 0.03736420293600678).
- **Split-half feasible, and selection is strongly person-stable.** Halves cut at
  each user's OWN median timestamp; 1271 users scored of
  1282 eligible. **Self cosine
  0.6814441717128207 (sd 0.2882601041061215) against permuted-other
  0.05115484595892177 — discrimination 0.630289325753899.** The
  number SR1 actually needs is the reliability of the PAIRWISE SIMILARITY object,
  since that is what a Mantel test correlates: half-split
  0.5788247760635731, **Spearman-Brown
  0.7332349793835083 — the SR1 ceiling** over
  404,550 pairs.
- **This is decomposition (a), measured on the corpus, and it is strong.** It is
  ALSO exactly what S1's gamma = 0 arm showed can be true while (b) is false. The
  two results must be read together: SR0 establishes that the corpus HAS a stable
  selection signature and says nothing whatever about whether that signature
  carries personality.
- **Label-side marginals (separate stage, never joined).** N =
  1401; per-trait sd: agreeableness 30.906133487248784, openness 27.687923217119188, conscientiousness 30.1921871108564, extraversion 30.146388899654326, neuroticism 32.24694596571011.
- **SR1 power.** N_effective 1306 (signature users, capped by the
  canonical cohort), 852,165 pairs, 999
  permutations planned at alpha 0.05 and power
  0.8; permutation null sd ~ 1/sqrt(N-1) =
  0.027681826617913324 (in the OBJECT count, not the pair
  count). **Minimum detectable OBSERVED Mantel r =
  0.07755299626311207**; corrected for attenuation with the
  measured selection reliability and an ASSUMED label reliability:
  1.0 -> 0.09056846224156084,
  0.8 -> 0.10125861909487652,
  0.5 -> 0.1280831476252909.
  Label reliability is NOT measured here (outside the mandate) and is carried as a
  declared parameter.
- **G-SR0, and a gate defect of my own, disclosed with timing.** The gate's FIRST
  implementation was a substring blacklist and it produced a demonstrable FALSE
  POSITIVE — it matched "cov" inside `coverage_comments_in_vocab`, a pure
  selection-side quantity. It was replaced, AFTER that failure and before the
  leg was declared clean, with a STRICTLY STRONGER test of the property the
  governance actually forbids: no leaf key may name both a selection object and a
  label object; `selection.json` may carry no label token at all;
  `labels.json` may carry no selection token at all; `power.json`, the declared
  meeting point, is instead held to carrying only scalars. A second iteration
  exempted the two compliance booleans (`label_table_opened`,
  `selection_artifact_opened`) whose whole purpose is to assert non-access — and
  ONLY when their value is verified `False`; a `True` would still fire. Final:
  PASS, 0 joint quantities, 0 unexpected artifacts.
- **R-G compliance, verified not asserted.** The report was scanned against all
  1401 cohort author ids: **0 leaks**. The
  `body` column was never read. No per-user rows, no text excerpts, no
  cross-corpus linkage; Essays untouched and its confirm-half labels sealed; the
  native corpus untouched and still paused. Sources read in place from the private
  paths; nothing copied into this repository. `cohort_authors.csv` is the only
  artifact carrying identifiers and lives in gitignored `results/`.
- **Anomalies.** A-1 interpreter re-verified as standing practice after the
  previous leg's venv loss (numpy 2.4.4 / pandas 3.0.2 / Python 3.12.12, matching
  every prior leg) BEFORE any number. A-2 `timeout(1)` absent on macOS. A-3 the
  missing fitted class map (above), a reconnaissance finding rather than an
  execution fault. A-4 the gate false positive (above), resolved before the leg
  was declared clean.
- **Registration-defect candidates: none.** The registration's source list,
  no-peek mandate and routing were all executable as written; the "1401 rows"
  description is exact.
- Report: `reports/SUICA_M4_SR0_RECON_REPORT.md` (aggregates only).

### Planner adjudication of SR0 (2026-08-16, appended after the run) — SR1 IS READY; THE (a)-ONLY READING RULE BINDS

**SR1_READY accepted — zero registration defects, zero ID leaks,
zero joint quantities.** The recon delivered: sources pinned (5.4
GiB comment table streamed at 17,640,062 rows without reading
`body`; the 1401 cohort EXACT, with the executor correcting its own
early wc-based miscount); the selection-signature object defined
(1191-subreddit vocabulary at the ≥15-user floor, coverage 0.7814,
1306 signature-carrying users); **selection proven strongly
person-stable on real data** (self-cosine 0.6814 vs permuted
0.0512; pairwise-similarity reliability 0.7332 by Spearman-Brown —
SR1's ceiling); power adequate (852,165 pairs; minimum detectable
observed Mantel r 0.0776). The 12-axis constructor is LIVE CODE
with its fitted map absent — SR1 refits if it uses axes and may
NOT cite the historical 5/5 as if those axes were in hand
(adopted). The executor's A-4 (its own no-peek gate's substring
false positive, replaced pre-clean-declaration by a strictly
stronger leaf-key test) is the gate-hardening pattern working on
the gate itself.

**Binding reading rule for SR1 (the executor's interpretive point,
adopted verbatim):** SR0 measured decomposition **(a) only**.
S1's γ = 0 arm stands as the physical counterexample — a perfectly
stable selection signature carrying zero trait information — so
SR0's stability is NOT evidence for the owner's conjecture, and no
SR1 text may read it as such.

---

## M4-SR1 — the selection-geometry test on PANDORA (sealed)

**REGISTERED 2026-08-16, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. THE REAL-DATA VERDICT LEG for
decomposition (b). Tier: EXPLORATORY, corpus-level; the R-G block
binds; **no person claims; aggregates only; quotable only with the
tier label.**

### Estimand (ONE primary; everything else second readings with no verdict weight)

Over the 1306 signature-carrying users: **Mantel correlation
between pairwise SELECTION similarity and pairwise TRAIT
similarity.**

- Selection similarity (primary): **Hellinger cosine** — cosine of
  sqrt-frequency signature vectors (compositional-data standard;
  pinned NOW, rule 9). Second geometries: raw-frequency cosine;
  Euclidean-distance-based similarity (the direction question,
  T8).
- Trait similarity (primary): **negative squared Euclidean over the
  five z-scored Big5** (the conjecture's plain "personalities are
  close"). Second: cosine of centred 5-profiles (shape-only).
- Statistic: Pearson over the 852,165 pairs; **null = 999
  permutations of users** (labels shuffled across users, pairs
  recomputed — the only valid exchangeability), seed master
  20260816; p one-sided positive per the conjecture's direction
  (rule 22: side declared).
- Disattenuation: observed r divided by √(0.7332 · rel_label) with
  rel_label a DECLARED parameter (0.8), quoted UNBUDGETED
  descriptive (rule 30 honesty — label reliability is unmeasured).

### Ordering (the seal, adapted)

No worlds exist to generate; the discipline is CONFIG-BEFORE-JOINT:
the full analysis config (estimand definitions, transforms,
permutation seed, cells, leans) is hashed and stamped BEFORE the
first joint selection×trait quantity is computed; SR0's no-peek
enumeration is the hand-off state; G-SR1 verifies the stamp
precedes the first join in the run log.

### Leans (sides declared; the S1-γ=0 reading rule in force)

**L-SR1 [POSITIVE_DETECTED .55 / POSITIVE_UNDETECTED .25 /
NULL_OR_NEGATIVE .20].** Prior grounds: the choice channel's
trait-LEVEL evidence (H6 lockbox r=+0.096; axes 5/5) suggests a
real but small pairwise coupling; the MDR 0.0776 sits near the
plausible magnitude — an honest coin-lean, not a confident one.

### Routing (rule 16 — disjoint, covering, entailed)

| # | condition | outcome |
|---|---|---|
| 1 | G0/stamp-order/no-peek violation | **STOP / VOID** |
| 2 | primary r > 0 AND perm p < .05 | **SELECTION_TRAIT_COUPLING_DETECTED** — decomposition (b) holds on this corpus at EXPLORATORY tier; magnitude + disattenuated value quoted UNBUDGETED; the owner's conjecture gains its first real-data pairwise confirmation |
| 3 | primary r > 0 AND p ≥ .05 | **COUPLING_BELOW_DETECTION** — the signature is real (SR0) and the coupling, if any, is below 0.0776 observed; read against S1's γ=0: stability without coupling is physical |
| 4 | primary r ≤ 0 | **NULL_OR_NEGATIVE_NAMED** — reported plainly; the second geometries checked for sign agreement |
| — | second readings (raw-freq; distance; profile-cosine; axis-space refit if used; split-half agreement; comment-count tertiles) | reported, adjudicating nothing; any sign DISAGREEMENT among geometries is flagged GEOMETRY_SPLIT (the owner's C-4 direction concern, on the record) |

### Gates

G0sr1: SR0's numbers bit-verified (ceiling 0.7332, MDR 0.0776,
N 1306, pairs 852165, vocabulary 1191, coverage 0.7814); the
cohort hand-off file consumed as-is; labels joined ONLY inside the
stamped stage. G1sr1: rule-29 predicates on similarity
distributions (cosines in [−1,1]); #57 (no pilot correlations; the
permutation machinery is the null). G2sr1: power stands on SR0's
analysis (rule 25 satisfied there; no new projection). G3sr1:
R-G compliance block re-run (ID-leak scan of the report against
all cohort IDs; aggregates only). Rule 24 generated tables; stages
< 600 s (the pair matrix at 1306 users is in-memory feasible;
permutations chunked); target < 40 min.

Deliverables: the six as always;
`reports/SUICA_M4_SR1_SELECTION_GEOMETRY_REPORT.md` (aggregates
only); ONE commit `feat(m4-s): SR1 — the selection-geometry test —
<SLUG>`. Sequencing note: SR1 dispatches before S2 (the real track
carries the line's point; S2's synthetic shape follows).

### Outcome of M4-SR1 (executed 2026-08-15, append-only)

**`SELECTION_TRAIT_COUPLING_DETECTED` (rule-16 cell 2).** Modifiers:
BELOW_SR0_MDR, AXIS_SPACE_NOT_RUN. **EXPLORATORY, corpus-level, no person claims.**
**Decomposition (b) holds on PANDORA: selection-similarity does carry
trait-similarity.**

- **The seal held.** Config hashed `11b36622c2099acae3d72b54b6c8502b5cc2a913831fce1bd1f64eb6efc11876` and stamped
  2026-08-15T17:13:13.916769+00:00 with **0 joint
  quantities in existence**; the first joint selection x trait quantity was
  computed at 2026-08-15T17:14:45.493576+00:00, **91.576807 s later**
  (True). The selection stage never opened the label
  table (False); SR0's no-peek enumeration was
  the hand-off state and its gate is re-verified PASS.
- **G0sr1: every SR0 number bit-verified** — ceiling 0.7332, MDR
  0.0776, N 1306, pairs 852165, vocabulary 1191, coverage
  0.7814, all matching. One executor slip caught and fixed BEFORE the first
  join: the signature threshold must apply to the IN-VOCABULARY comment total
  (SR0's own definition), not the overall total — the overall reading gives 1362
  users instead of 1306. Corrected, N and vocabulary then reproduced exactly.
- **PRIMARY: Mantel r = 0.048987613136188025, one-sided permutation p = 0.001**
  over 852,165 pairs from 1306 users. **0
  of 999 permutations reached the observed value.** Null mean
  -0.00010562885372651673, sd 0.009064019613144935, 95th pct 0.01540180833500572, max
  0.029409823942762026; **z = 5.416277113822429**.
- **A CORRECTION TO SR0's POWER ANALYSIS, WHICH THIS EXECUTOR WROTE, AND WHICH
  CUTS IN THIS LEG'S FAVOUR.** The observed r is 0.6312836744354127x
  SR0's declared MDR 0.0776 — i.e. BELOW it — yet p = 0.001.
  The reason: SR0's MDR used the standard 1/sqrt(N-1) heuristic for the Mantel
  permutation sd (0.027681826617913324), and the
  EMPIRICAL permutation null sd is 0.009064019613144935 — the
  heuristic **overstates the null spread by 3.0540342805269574x**.
  The corrected empirical MDR is 0.025393623364872876 and the observed
  r is 1.929130492025523x that. The registered test is the
  PERMUTATION null and always was, so the routing is unaffected — but SR0's power
  table was wrong, and it was wrong in the direction that flatters this leg. The
  `BELOW_SR0_MDR` modifier is carried for exactly this reason.
- **Disattenuated (UNBUDGETED, routes nothing): 0.06396318159111603**
  = observed / sqrt(0.7332 x 0.80). The selection reliability is MEASURED; the
  label reliability 0.80 is DECLARED, not measured, so this is an illustration of
  scale and never a result (rule 30).
- **No GEOMETRY_SPLIT** (False): all six selection x trait
  geometries agree in sign (+1). hellinger_cosine x negative_squared_euclidean = 0.048987613136187574; hellinger_cosine x centred_profile_cosine = 0.03207134928723893; raw_frequency_cosine x negative_squared_euclidean = 0.03703374254514458; raw_frequency_cosine x centred_profile_cosine = 0.025984074919015603; negative_euclidean_distance x negative_squared_euclidean = 0.04789480729045626; negative_euclidean_distance x centred_profile_cosine = 0.03201520943384964. The owner's C-4 direction concern is
  checked and clear — the reading does not depend on the direction convention.
- **Split-half agreement:** early 0.03985813511679325, late 0.04498873327209943, mean
  0.042423434194446344, agreeing in sign (True); the selection geometry's
  own early-vs-late self-consistency is 0.675573251251224.
- **Comment-count tertiles:** low (n=437, median 73 comments) = 0.0444626838820253; mid (n=434, median 481 comments) = 0.03199603525666885; high (n=435, median 2884 comments) = 0.06391622568311668 — positive in all three, and largest in the
  heaviest-posting tertile.
- **The 12-axis space: NOT RUN, on evidence.** A faithful refit runs TF-IDF over
  SLICE TEXT (constructor lines 49-55) and both inputs
  (`phase2_passB_slicetext_s128.parquet`, `tier_u_comments.parquet`) are absent
  from disk. Refitting would require reading comment text, which this line has
  never done; substituting a co-occurrence "axis" would be a different object and
  would invite the mis-citation the planner barred. Declared NOT_RUN with the
  reason rather than approximated.
- **G1sr1** predicates pass: cosines within [-1, 1]
  (True); selection similarity spans
  [0.0,
  1.0], trait similarity
  [-46.45520669416317,
  -0.0].
- **G3sr1 / R-G compliance:** the report was scanned against all
  1401 cohort IDs with the report present —
  **0 leaks**. Aggregates only, no per-user rows,
  no text excerpts, `body` never read, no cross-corpus linkage; Essays untouched
  with confirm-half labels sealed; native corpus untouched and still paused.
- **What this does and does not establish.** It establishes decomposition (b) at
  EXPLORATORY tier on one corpus: pairwise selection proximity carries pairwise
  trait proximity, robustly across geometries, halves and activity tertiles. The
  magnitude is small in absolute terms (0.048987613136188025), it is a CORPUS-LEVEL
  pairwise statistic and supports NO person-level claim, and the (a)-only reading
  rule still binds for SR0 — S1's gamma = 0 arm remains the standing proof that
  stability alone would not have implied this. **The owner's conjecture has its
  first real-data pairwise confirmation.**
- Rule 13: 0 events. **Registration-defect candidates: none** — the registration
  was executable exactly as written.
- Report: `reports/SUICA_M4_SR1_SELECTION_GEOMETRY_REPORT.md` (aggregates only).

### Planner adjudication of SR1 (2026-08-16, appended after the run) — DECOMPOSITION (b) HOLDS ON PANDORA; THE OWNER'S CONJECTURE GETS ITS FIRST REAL-DATA PAIRWISE CONFIRMATION

**SELECTION_TRAIT_COUPLING_DETECTED accepted — zero registration
defects, the seal proven (stamp 91.577 s before the first joint
quantity, zero prior joins, selection stage never opened the label
table), zero ID leaks with the scan re-run report-present.**
Primary: **Mantel r = 0.048987613136188025, one-sided permutation
p = 0.001 (0/999 permutations reached it; empirical null sd
0.009064019613144935; z = 5.416)** over 852,165 pairs of 1306
users. Robustness is the result's spine: **all six registered
geometries agree in sign** (GEOMETRY_SPLIT False — the C-4
direction concern does not bite on this corpus at this
granularity), both time halves (+0.0399 / +0.0450), and all three
activity tertiles (+0.0445 / +0.0320 / +0.0639). Disattenuated
0.0640 (UNBUDGETED; label reliability declared 0.80).
Axis-space honestly NOT_RUN (inputs absent; a substitute would be
a different object inviting the barred mis-citation).

**Defect #66 (mine — the #57 family, on power heuristics).** SR0's
power table used the 1/√(N−1) closed form for the Mantel null sd;
the empirical permutation null is 3.054× smaller (0.00906 vs
0.02768), so the declared MDR 0.0776 was overstated and the
observed 0.0490 sits at 1.93× the CORRECTED empirical MDR
(0.0254). The registered adjudicator was always the permutation
test, so routing is untouched — but the modifier BELOW_SR0_MDR is
carried, and the convention is enacted: **power analyses for
permutation tests pin the null-sd METHOD, preferring a small
pre-registered permutation calibration over closed-form
heuristics.** The executor corrected its own table and disclosed
the error's favourable direction unprompted.

**The reading, bounded exactly as the tier demands.** This is a
CORPUS-LEVEL pairwise aggregate at EXPLORATORY tier: small in
absolute magnitude, supporting NO person-level claim, quotable
only with its tier label and alongside the S1 γ=0 counterexample —
which is what makes the finding a finding: stability alone would
NOT have implied this (S1 proved a stable, trait-silent selection
signature is physical); the coupling is a measured fact about this
corpus, not a logical consequence of signature stability.
Decomposition (a) was SR0's; **decomposition (b) is now measured:
on PANDORA, people who choose similar frames measurably have
similar Big5 profiles, across every registered geometry.** Theory
consequence in appendix NN; S2 (the sealed synthetic shape
r(γ)) dispatches next to complete the line's synthetic half.

### Outcome of M4-S2 (executed 2026-08-15, append-only)

**`TRANSFER_LAW_SEALED` (rule-16 cell 4).** Modifiers:
GAMMA0_NULL_CONFIRMED_ON_FRESH_WORLDS, DIRECTION_READING_AGREES. **Interior hits 3/3 — but the
headline belongs to HOW they landed, not that they did.**

- **G0s2.** S1's amplitude bit-exact: r(1) = 0.23983432331725474 ==
  registration (True); β* = 1.7
  matches (True); S1's C-S1c PASS
  (True); S1 source hash verified. The measurement is S1's
  CERTIFIED instrument imported by file and called directly, so r(γ) here is the
  same object as S1's C-S1c Mantel (RN-S2-1).
- **g(γ) recomputed at full precision** and matching the planner's sanity values:
  g(0.25) = 0.11043152607484653 (sanity 0.1104),
  g(0.5) = 0.7071067811865475 (0.7071),
  g(0.75) = 0.9938837346736188 (0.9938).
- **Sealed:** g=0.25: g(γ)=0.11043152607484653 pred=0.02648527032905259 half=0.04363026050121361 (SE_pred 0.00016200230186118092, SE_meas 0.002324586047924536, SE_approx 0.021690319564518324); g=0.5: g(γ)=0.7071067811865475 pred=0.16958847637891772 half=0.21887836869808167 (SE_pred 0.0010373208655672403, SE_meas 0.002324586047924536, SE_approx 0.10940957607134302); g=0.75: g(γ)=0.9938837346736188 pred=0.2383674329614733 half=0.08727767214621368 (SE_pred 0.0014580207167506282, SE_meas 0.002324586047924536, SE_approx 0.043552479712535445). Hashed `ef870d43d6c73757b8de9336e8567571586ee0aaacb991a6cebc217b63a1af6a`, stamped 2026-08-15T17:27:40.278163+00:00 with
  **0 fresh worlds AND
  0 probe worlds in existence** — the cleanest
  K2f ordering the program has achieved, because SE_meas came from S1's persisted
  spread and SE_approx from standard-normal arithmetic, so no world of any kind
  was needed to build the band. Arms re-hashed to a match
  17.961275 s later.
- **Measured:** γ=0.0: -0.000556827079510771 [-0.001610303477140683, 0.0004102164964021282]; γ=0.25: 0.021687781588106775 [0.019706868338856466, 0.023929471306571363]; γ=0.5: 0.11011136328557458 [0.10764037282740947, 0.11285795106841882]; γ=0.75: 0.20669287217697138 [0.204047311871756, 0.20943216928148053]; γ=1.0: 0.237741075953029 [0.23415995152633748, 0.24081471764146747].
- **V-S2b: 3/3 inside the sealed analytic bands** — γ=0.25 pos -0.10995782940173755 INSIDE=True; γ=0.5 pos -0.2717359118085588 INSIDE=True; γ=0.75 pos -0.3629171127689849 INSIDE=True.
  **All three positions are NEGATIVE**, i.e. every measurement sits BELOW the
  analytic prediction, and the miss grows with the prediction's size.
- **THE LEG'S METHODOLOGICAL FINDING — REGISTRATION-DEFECT CANDIDATE, PINNED
  BEFORE ANY WORLD (RN-S2-6).** The registered SE_approx absorbs the softmax
  distortion into BAND WIDTH. But the budget's own probe-free arithmetic showed,
  before a single world existed, that the distortion is not uncertainty — it is a
  SYSTEMATIC, sign-stable bias: the analytic g OVERSHOOTS the pipeline ratio at
  every interior γ (analytic 0.11043152607484653 /
  0.7071067811865475 /
  0.9938837346736188 against pipeline
  0.07766211128618199 /
  0.44959247073676384 /
  0.8823526228138191), with MC spreads
  (0.013608651968474933 /
  0.02274388784050644 /
  0.0069168510746423885) an ORDER OF MAGNITUDE smaller
  than the bias. **A #61 band states UNCERTAINTY; a known bias belongs in the
  PREDICTION.** Absorbing it into width inflated the γ = 0.5 half-width
  (0.21887836869808167) past the
  prediction itself (0.16958847637891772), which
  makes V-S2b nearly unfalsifiable there.
- **So a SECOND prediction was stamped alongside the first** — equally
  probe-free, equally pre-world, using the pipeline ratio directly. The
  registered analytic prediction still ROUTES, unchanged. Result: γ=0.25: pipeline pred 0.01862603990771079 (half 0.005021709419392657), |err| 0.003061741680395986 vs analytic |err| 0.004797488740945814; γ=0.5: pipeline pred 0.1078277059876844 (half 0.0057682731266421915), |err| 0.0022836572978901792 vs analytic |err| 0.05947711309334314; γ=0.75: pipeline pred 0.2116184442197572 (half 0.005406852719451732), |err| 0.00492557204278582 vs analytic |err| 0.031674560784501926.
  **3/3 inside the pipeline bands, whose half-widths (~0.005) are 8x to
  40x TIGHTER than the analytic ones, with absolute errors ~13x smaller at
  γ = 0.5 (0.0023 against 0.0595).** The transfer law's FORM is right — monotone,
  sharply non-linear, correct ordering, near-flat top and collapsing knee — while
  its analytic COEFFICIENT overshoots; the softmax-corrected version is accurate
  to 0.002–0.005 across the interior.
- **The conservatism direction, as the registration requires:** the budget took
  the MAX absolute deviation across MC reps (not the mean) times
  1.5, because in R2b this executor's probe-based
  transport term undersized the realized value by 1.53x. The budget therefore
  errs WIDE, which makes the sealed test EASIER to pass — the direction that
  flatters a hit, and therefore the one that had to be disclosed.
- **V-S2a: r(0) NULL on FRESH worlds** — -0.000556827079510771
  [-0.001610303477140683, 0.0004102164964021282] against ε = 0.010125436066679785
  (True). S1's falsifier arm reproduces on worlds it never saw,
  which is what licenses reading the interior points as shape rather than
  artefact.
- **Anchors PASS (True), and they route INSTRUMENT_DEFECT rather than
  a shape cell.** γ = 1 re-measured 0.237741075953029 against S1's
  0.23983432331725474, deviation -0.0020932473642257254 within
  the 2√2·SEM band 0.006421905101506714 (z = -0.921937887620048).
- **V-S2c, the direction reading (adjudicates nothing):** cosine and distance
  geometries agree in sign at every γ (True) and
  cosine exceeds distance at γ = 1 (True) — T8's
  expectation, checked rather than assumed.
- **Gates.** G1s2 / #59: the interior arms' preferences are genuinely MIXED (mean
  max-π ≈ 0.586 at both γ = 0.25 and 0.75, against 0.25 for flat and 1.0 for
  collapsed), so the interior g values are not forced by construction. G2s2 PASS.
  G3s2 PASS: power 1.0 at each
  interior truth with false-fire
  0.0; escalation
  False.
- Rule 13: 0 events. Report: `reports/SUICA_M4_S2_TRANSFER_LAW_REPORT.md`.
- **One registration-defect candidate:** RN-S2-6, the SE_approx category error
  (systematic bias absorbed as band width). Non-blocking — the registered test
  routed exactly as written and the fix was stamped alongside it before any
  world.

### Planner adjudication of S2 (2026-08-16, appended after the run) — THE SHAPE IS SEALED, THE COEFFICIENT CORRECTED, THE LINE CLOSES

**TRANSFER_LAW_SEALED accepted, typed honestly.** The registered
3/3 (positions −0.110 / −0.272 / −0.363) is WEAK evidence exactly
where the band swallowed the prediction (γ = 0.5 half-width 0.219
vs prediction 0.170 — nearly unfalsifiable, as the executor said
first). The STRONG evidence is the co-stamped, equally probe-free
PIPELINE prediction: **3/3 inside bands 8–40× tighter** (errors
0.0031 / 0.0023 / 0.0049). The record therefore reads: **the
law's FORM is right — monotone, sharply non-linear, near-flat top,
collapsing knee, exactly g's silhouette — and the analytic
coefficient overshoots the softmax pipeline systematically
(0.1104/0.7071/0.9939 vs 0.0777/0.4496/0.8824); the LAW OF RECORD
is the pipeline-corrected shape, with the analytic g as its
idealization.** The γ = 0 falsifier reproduced on fresh worlds
(−0.00056 vs ε 0.0101) — the line's counterexample is now
twice-measured — and the γ = 1 anchor matched S1 at z = −0.92.
Direction reading: signs agree at every γ; T8's cosine-dominance
checked, not assumed. The stamp preceded EVERY world including
probes — the cleanest ordering of the program.

**Defect #67 (mine).** The registered SE_approx absorbed a KNOWN,
SIGN-STABLE bias (the softmax distortion, an order of magnitude
larger than its MC spread) into band WIDTH. **Convention: a #61
band states UNCERTAINTY; when the Part-0 error budget reveals a
systematic sign-stable deviation exceeding its MC spread, it is
applied as a PREDICTION CORRECTION (with its own uncertainty),
never absorbed into width — and both the idealized and corrected
predictions are stamped** (the executor's co-stamp is the
pattern's first instance and its proof of value).

**THE LINE CLOSES.** Five legs: S1 (generator certified; the γ = 0
treasure), SR0 (SR1_READY; (a) strong on real data), SR1
(**SELECTION_TRAIT_COUPLING_DETECTED — the owner's conjecture's
first real-data pairwise confirmation**), S2 (the shape sealed,
coefficient corrected). The conjecture is now theorized
END-TO-END: decomposition (a)/(b); a certified apparatus; a
physical falsifier (stability without coupling); a sealed
real-data detection (r = 0.0490, p = .001, robust across six
geometries, both halves, three tertiles); and a sealed synthetic
shape law. Synthesis:
`docs/SUICA_M4_S_SELECTION_LINE_SYNTHESIS.md`; theory appendix OO.
Named, not queued: the 12-axis refit (inputs absent, priced); the
real-data dose analogue (no γ knob exists in reality — the
coupling is one number per corpus); R3's identity-formalism
reconciliation (held from the R-line).
