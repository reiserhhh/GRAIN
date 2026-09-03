# GRAIN M4-T1: Hierarchical Selection Identity

Status: exploratory mechanism discovery. This line is label-free by design.

## Question

Does an author's context-selection profile contain a reproducible hierarchy of
conditional innovations? In particular, can a residual left at one group level
become stable author information after the population is split at the next
level?

The proposed object is

\[
\pi_u-\bar\pi = \sum_{l=1}^{L} d_l(u)+r_L(u),
\qquad
d_l(u)=m_l(u)-m_{l-1}(u),
\]

where \(m_l(u)\) is the centroid of the nested selection group containing
author \(u\) at level \(l\). The terminal residual \(r_L\) is not declared
noise: it is tested for remaining same-author information.

## Design

1. Use the existing SR1 early/late subreddit-frequency matrices. No comment
   body and no personality label is read by the primary experiment.
2. Split authors into five folds. In each fold, fit a binary Hellinger-space
   selection tree using only the early vectors of the other four folds.
3. Freeze the tree and route both early and late vectors of held-out authors.
4. At each depth, test whether the child selected by the early half predicts
   the late vector better than its parent centroid.
5. Build a conditional null by permuting late vectors only among held-out
   authors sharing the same early parent node. This preserves the coarser group
   and tests the additional tail branch.
6. Compare flat Hellinger matching, hierarchical-path matching, and terminal
   within-leaf residual matching.
7. Repeat after removing explicitly named personality/typology communities.
   This is a sensitivity arm, not a redefinition of the primary object.

## Primary readings

- cross-half predictive gain by depth;
- within-parent permutation p-value by depth;
- early/late branch and prefix agreement above the conditional null;
- same-author AUC from hierarchical paths versus flat Hellinger AUC;
- terminal residual same-author AUC within frozen leaves.

## Interpretation rules

- A positive held-out branch gain means that a residual at the parent level
  contains reproducible structure at the child level.
- A rare branch is not identity merely because it is rare. It must replicate
  across halves and held-out authors.
- Positive terminal-residual AUC means the fitted depth has not exhausted the
  stable author structure.
- Results establish selection-based author information, not personality. A
  psychological interpretation requires a later, separately registered
  external-connection study.
- Failure of the cleaned arm means the apparent identity hierarchy is specific
  to explicit typology communities or their ecology.

## Controls

- planted hierarchical-choice simulation;
- author-null simulation with identical population choice probabilities;
- held-out-author tree fitting;
- within-parent permutation rather than unrestricted pair permutation;
- explicit-personality-community ablation;
- no Big5/MBTI labels during model construction or primary scoring.

## Outputs

- `results/m4_t1_hierarchical_selection_identity/summary.json`
- `results/m4_t1_hierarchical_selection_identity/metrics_by_depth.csv`
- `results/m4_t1_hierarchical_selection_identity/per_user_depth.csv`
- `reports/SUICA_M4_T1_HIERARCHICAL_SELECTION_IDENTITY_REPORT.md`

## Completed reading (2026-08-17)

The mechanism test completed on held-out PANDORA authors. The primary full arm
retained five stable conditional levels; the arm excluding explicitly named
personality/typology communities retained four. At each retained level, all
three criteria agreed: the early child centroid improved late-half fit over
the parent, the late half locally replayed the early child above a
within-parent permutation null, and the early child reduced uncertainty about
the late child by positive permutation-corrected conditional mutual
information. In the cleaned arm, levels five and six still showed branch
replay and excess information, but their centroid-gain confidence intervals
crossed zero. They are therefore recorded as reproducible categorical code,
not confirmed predictive innovations.

The result supports a mixed representation:

\[
\pi_u-\bar\pi
=\sum_{l=1}^{L}d_l(u)+r_L(u),
\qquad
I(C_{u,l}^{early};C_{u,l}^{late}\mid P_{u,l})>0,
\]

where the discrete innovations are reproducible but do not exhaust identity.
Terminal within-leaf residual AUC remained above `.94` in both arms and above
`.94` across all resolution checks. Increasing tree depth therefore does not
turn the tail into a finite list of ever-smaller types. The current evidence
is more consistent with **nested conditional choices plus a continuous or
higher-order author coordinate**.

This is a label-free author-identity result. It does not identify personality,
and it does not license naming branches. The next scientific leg must separate
stable identity carried by topic/community history from psychologically valid
person-level structure using condition-matched controls and an independently
specified external-connection analysis.

---

## Planner adjudication of T1 and line adoption (2026-08-17, appended by the M-line planner)

The T-line is hereby adopted into the standing loop discipline
(rules 1–33, all conventions, six deliverables,
registration-committed-before-run). **T1's own ledger typing is
upheld exactly as written — EXPLORATORY, NOT PREREGISTERED — and
its numbers enter future legs only as G0 anchors requiring
bit-verification** from `results/m4_t1_hierarchical_selection_identity/`.
The work itself is strong: frozen held-out trees, a triple gate
(centroid gain + within-parent replay + permutation-corrected
conditional MI), planted and author-null controls both clean
(0.9407/0.4957 path AUC), resolution sensitivity, and the
personality-community ablation. `suica_core/` received a NEW module
only (git-verified additive; frozen operators untouched;
owner-directed); my legs continue to write under `scripts/` and
import the T1 core by file with provenance. The 4W theory document
(`SUICA_4W_PERSON_INTERPRETER_THEORY.md`) is adopted as the line's
map; its §11 four-questions template becomes a REGISTRATION SECTION
of every future leg (the 4W header).

---

## M4-T2 — the condition-matched residual audit, and the taste-transport falsifier

**REGISTERED 2026-08-17, BEFORE RUN.** Planner: the M-line planner;
executor: dispatched agent. The 4W doc's priority 1, with its
priority 4 (the Where-version of frame refresh) folded in as Arm B.
Label-free throughout: no Big5/MBTI value is read at any stage;
aggregates only; IDs confined to gitignored intermediates.

### 4W header (the §11 template)

- **Object:** Where × Who — the terminal within-leaf residual
  r_L(u) whose same-author AUC T1 measured at 0.9552 (full) /
  0.9417 (clean).
- **Fixed:** T1's frozen base tree and leaf assignments
  (`per_user_depth.csv`, config seed 20260817, d6/l30); the
  training-fold community embedding (Arm B); the SR0 vocabulary
  (1191, floor 15).
- **Varied:** stranger-match strata (observability calipers,
  Arm A); support disjointness (Arm B).
- **Falsifiers:** Arm A — the residual AUC collapses into the
  permutation null band under full observability matching
  (SUPPORT ARTIFACT); Arm B — no transport onto disjoint support
  (LOYALTY-ONLY identity).
- **Layer:** R (relational, label-free author-identity evidence).

### Question

T1 proved the tail exists; T2 asks WHAT CARRIES IT. Candidates
(4W doc §10.1): observability/support (volume, span, entropy,
breadth); specific community loyalty (self-history); a continuous
taste coordinate that GENERALIZES to unseen communities; deeper
structure. Arms A and B separate the first three.

### Arm A — matched-stranger audit (does the residual survive observability matching?)

Within each frozen leaf, the same-author statistic compares
score(early_u, late_u) against score(early_u, late_v) for strangers
v in the same leaf. **Matching construction (pinned):** the
stranger set for target u is restricted to v whose LATE-half
observables match u's LATE-half observables — cumulative caliper
ladder, all quantities selection-side:

1. L0 (anchor): unmatched — must reproduce T1's leaf-residual AUC
   within bootstrap error (G0 anchor, not a verdict);
2. L1: in-vocabulary comment volume, |log2 ratio| ≤ 0.5;
3. L2: + active time span, |Δ| ≤ 90 days;
4. L3: + selection entropy, |Δ| ≤ 0.5 bits;
5. L4: + community breadth (distinct in-vocab communities),
   ratio ∈ [0.7, 1.43].

Per level: AUC with user-bootstrap CI (B = 1000) and a
WITHIN-LEAF × WITHIN-STRATUM permutation null (late halves
permuted among matched candidates; B = 999; method pinned per
#66 — the null band is the permutation's own, no closed forms).
Admissible-pair attrition reported per level; a level whose
admissible stranger pool drops below 5 per target on median is
UNDERRESOLVED and says so.

### Arm B — taste transport onto disjoint support (the Where refresh)

**Embedding (pinned):** per T1 fold, PPMI + SVD (d = 64) of the
TRAINING authors' EARLY user×community matrix (sqrt counts);
community vectors frozen per fold; held-out users never touch the
fit (rule 12: file:line of the construction in Part 0). Per
held-out user: **centroid_E** = count-weighted embedding centroid
of the FULL early half; **centroid_Lnew** = centroid of LATE-half
communities ABSENT from that user's early half (≥ 3 such in-vocab
communities required; attrition reported). Same-author statistic:
cos(centroid_E(u), centroid_Lnew(u)) vs cos(centroid_E(u),
centroid_Lnew(v)) for leaf-and-L1-matched strangers; AUC,
bootstrap CI, within-leaf×stratum permutation null as Arm A.
**This is selection identity under support refreshment: the
person's taste predicting communities they had not yet joined.**

### Verdicts (rule 22; NULL-first per #55)

- **V-T2a (Arm A at L4):** COLLAPSED (AUC CI inside the permutation
  null band ± its own spread — equivalence first) / STRONG_SURVIVAL
  (CI entirely above the null band AND point ≥ 0.90) / PARTIAL
  (above the null band, below 0.90) / UNDERRESOLVED.
- **V-T2b (Arm B):** TRANSPORTS (CI above the null band) /
  LOYALTY_ONLY (inside) / UNDERRESOLVED.
- Readings (no gates): the AUC-by-ladder curve (which caliper bites
  hardest — the support decomposition); clean-arm replication of
  both arms (T1's 23-community ablation); per-leaf heterogeneity.

### Leans (sides declared)

**L-T2 [SURVIVES(strong or partial)+TRANSPORTS .35 /
SURVIVES+LOYALTY_ONLY .30 / COLLAPSED .20 / underresolved-mixed
.15].** Honest uncertainty: volume alone is enormously
author-informative in frequency space, and loyalty alone can carry
a 0.95 AUC; the interesting world is the one where both falsifiers
fail.

### Routing (rule 16 — disjoint, covering, entailed)

| # | condition | outcome |
|---|---|---|
| 1 | G0 anchor or reconstruction failure | **STOP** |
| 2 | V-T2a COLLAPSED | **SUPPORT_ARTIFACT_MAJOR** — T1's tail re-typed by dated note; the 4W doc's §5.2 gains a correction |
| 3 | V-T2a survival + V-T2b TRANSPORTS | **CONTINUOUS_TASTE_COORDINATE** — the tail is a generalizing person-level Where structure; priority 2 (tree+tail joint representation) becomes registrable on a certified object |
| 4 | V-T2a survival + V-T2b LOYALTY_ONLY | **HISTORY_IDENTITY** — stable but non-generalizing; typed as community-history; the psychological-candidate reading narrows sharply |
| 5 | any UNDERRESOLVED verdict (no higher cell) | **UNDERRESOLVED** (attrition and pool sizes reported; redesign named) |

### Gates

- **G0t2 (bit-exact):** T1's summary.json / metrics_by_depth.csv /
  per_user_depth.csv reproduce the adjudication-cited numbers
  (0.9552, 0.9417, 0.7461, 0.7317, depth gates, N 1304/1269);
  the L0 anchor reproduces the leaf-residual AUC within bootstrap
  error; SR0 vocabulary/floor at source. Frequency matrices
  reconstructed through the T1 core module (imported by file,
  ONE loader chain, provenance pinned) and verified against every
  persisted summary statistic that exists.
- **G1t2 (#59 non-degeneracy):** matched-stranger AUC is not
  forced (calipers restrict the comparison set, they do not
  equalize content); centroid_Lnew is nonempty for the retained
  cohort; the embedding's held-out purity (training-fold-only fit)
  verified by construction log.
- **G2t2 (projection, #66-compliant):** pilot permutation
  calibration on a 200-user subsample fixes the achievable null
  bandwidths at each ladder level BEFORE the full run; if the L4
  or Arm-B bands cannot separate 0.5 from 0.75 at the realized
  pool sizes, the leg declares UNDERRESOLVED-by-design and stops
  (N is corpus-fixed; there is no escalation to buy).
- **G3t2:** rule-29 predicates (AUC ∈ [0,1], null 0.5); rule 24
  generated tables; the R-G compliance block (label-free, no IDs
  in committed artifacts, ID-leak scan of the report); stages
  < 600 s; target < 45 min.

### Deliverables

The six as always: `scripts/run_suica_m4_t2_matched_residual.py`;
`results/m4_t2_matched_residual/` (gitignored);
`reports/SUICA_M4_T2_MATCHED_RESIDUAL_REPORT.md`; outcome append
HERE; one ledger row (EXPLORATORY, label-free); exactly ONE commit
`feat(m4-t): T2 — the condition-matched residual audit — <SLUG>`,
never amended, never pushed; suite green first (report the new
suite count — T1 added tests).

## M4-T2 outcome (2026-08-17, appended by the executing agent) — UNDERRESOLVED

**Routing cell 5 → `UNDERRESOLVED`.** Slug `underresolved`. Report:
`reports/SUICA_M4_T2_MATCHED_RESIDUAL_REPORT.md`; harness
`scripts/run_suica_m4_t2_matched_residual.py`; artifacts (gitignored)
`results/m4_t2_matched_residual/`. Label-free throughout; ID-leak scan of
the report against all 1401 SR0 cohort identifiers returned 0 hits (the
same scan over the plan doc, the ledger row, the harness and its tests
also returns 0).

### Gates

- **G0t2 PASS.** Every T1 anchor bit-matches (0.9552295265671575 /
  0.9416819726747061 / 0.7460863278537894 / 0.731654715213212; N 1304 /
  1269; vocabulary 1191 at floor 15; 23 removed). The L0 anchor reproduces
  T1's pooled leaf-residual AUC **exactly** (|diff| = 0.0). SR1 never
  persisted the half-split COUNT matrices Arm B needs, so they were
  re-derived from the comment stream under SR1's own split rule and
  verified against SR1's frequency matrices: max |Δfreq| = 0.0 in both
  halves, support patterns identical.
- **G1t2 PASS.** AUC moves across the ladder (0.8944–0.9552), so the
  calipers restrict rather than equalize; centroid_Lnew nonempty for 1081
  users; all five fold embeddings pure (train × held-out overlap = 0).
- **G2t2 UNDERRESOLVED_BY_DESIGN — fired BEFORE the full run.**
- **G3t2 PASS** on the required predicates (AUC ∈ [0,1]; the vectorised
  AUC agrees with the reference implementation to 1e-12 at all 12
  comparison sets). The `null 0.5` predicate is reported, not enforced —
  see RD-T2-1.

### G2t2 — the pilot, and why the leg is UNDERRESOLVED

200 targets, full realized pools, B = 299. Median admissible stranger pool
by level: L0 **25**, L1 **3**, L2 **0**, L3 **0**, L4 **0**. L3 and L4
produce permutation bands of literally zero width ([0.0333, 0.0333] and
[0.0500, 0.0500]) built on 15 and 10 surviving targets. They cannot
separate 0.5 from 0.75 — not because the band is wide, but because there
is no pool left to build a band from. Arm B's band ([0.4761, 0.5516],
width 0.0755, centred 0.5122) does separate.

**The registered UNDERRESOLVED-by-design condition was met, so no survival
and no collapse claim is available at L4.** The full ladder was still run,
because cell 5 obliges this leg to report attrition and pool sizes. The
gate cost nothing: the registered per-level rule (median pool < 5) voids
L1–L4 independently, so the post-hoc reading of L4 is UNDERRESOLVED too.

### Arm A — the ladder (full arm, B_boot = 1000, B_perm = 999)

| level | targets | median pool | AUC | CI95 | null band | level verdict |
|---|---|---|---|---|---|---|
| L0 | 1304 | 21.0 | 0.9552 | [0.9471, 0.9630] | [0.4399, 0.4652] | anchor, above null |
| L1 | 1125 | 3.0 | 0.9523 | [0.9440, 0.9600] | [0.4518, 0.4770] | UNDERRESOLVED (pool) |
| L2 | 528 | 0.0 | 0.9473 | [0.9322, 0.9622] | [0.3989, 0.4439] | UNDERRESOLVED (pool) |
| L3 | 143 | 0.0 | 0.9149 | [0.8675, 0.9592] | [0.2750, 0.3919] | UNDERRESOLVED (pool) |
| L4 | 96 | 0.0 | 0.8944 | [0.8268, 0.9551] | [0.1839, 0.2984] | UNDERRESOLVED (pool) |

**V-T2a: UNDERRESOLVED.** Arm A's reportable content is its attrition, not
its L4 AUC. Requiring only that a stranger match on in-vocabulary volume
takes the median pool from 21 to 3; adding the span caliper takes it to 0.
Within a frozen selection leaf, PANDORA users are so heterogeneous in how
much they are observed that condition-matched strangers essentially do not
exist. That is a real fact about the corpus, and it is also precisely why
the registered design cannot answer its own question.

### Arm B — taste transport onto disjoint support

**V-T2b: TRANSPORTS.** AUC **0.6031**, bootstrap CI95 **[0.5837, 0.6236]**,
permutation null band **[0.4870, 0.5149]** (mean 0.5010, sd 0.0073), 907
targets, 3914 negative comparisons. The CI sits entirely above the null
band. Attrition: 1081 / 1304 users (82.9%) had ≥ 3 late communities absent
from their early half (median 10 such communities); 64 leaves usable.
Embedding: PPMI + SVD d = 64 on TRAINING rows' early sqrt-count matrix,
per fold, all five folds verified pure.

**The early-half taste centroid identifies its owner through communities
that owner had not yet joined.** Selection identity is not only loyalty to
a fixed community set. The effect is modest and heterogeneous, though:
median pool is thin (2.0), and 24 of 46 scoreable leaves fall below 0.60
against a pooled 0.6031.

### Readings

- **Ladder curve.** AUC falls 0.9552 → 0.9523 → 0.9473 → 0.9149 → 0.8944.
  The volume caliper (L1) costs almost nothing in AUC (−0.0029) while
  costing 86% of the pool — the first and sharpest sign that the ladder
  spends resolution far faster than it buys control.
- **Clean arm (23-community ablation).** n_valid = 1269, matching T1's
  clean arm exactly. L0 = 0.9417 (T1's clean anchor, bit-matched); L4 =
  0.9257 [0.8880, 0.9602]; V-T2a clean UNDERRESOLVED (same pool collapse).
  **Arm B replicates: 0.5941 [0.5751, 0.6148] against null [0.4556,
  0.4858] — TRANSPORTS.** The transport finding does not depend on
  explicit personality/typology communities.
- **Per-leaf heterogeneity.** L0: 61 leaves, median 0.9698, min 0.7777,
  none below 0.60. Arm B: 46 leaves, median 0.5967, min 0.4306, max
  0.8333, 24 below 0.60.
- **UNREGISTERED width diagnostic (names the redesign, claims nothing).**
  Widening every caliper by a common factor, L4's median pool reaches the
  registered floor of 5 only at **8×** the registered widths.

### Anomalies (all pre-verdict)

- **A1** T1's per-fold seed is `SEED + 1000*fold`; the wrong reading gives
  0.9536 instead of 0.9552295265671575. Fixed before Part 0 was accepted.
- **A2** Part 0 persisted `volume_late` as the **all-subreddit** late
  comment count while the registration pins L1 to **in-vocabulary**
  volume (medians 338.5 vs 229.5). Repaired from the verified count
  matrix before the pilot; no verdict was computed under the wrong
  caliper. Disclosed, not silently corrected.
- **A3** The permutation null was vectorised. The registered null's
  combined value multiset is always the flat pool of admissible stranger
  scores, so its tie-averaged ranks precompute once and each replicate is
  one batched gather. Same statistic, not an approximation; checked
  against the reference implementation at every comparison set (G3t2).
- **A4** SR1's 4000-timestamp cap on the median split time is inherited
  deliberately — reproducing the frozen halves requires it. Named because
  it is invisible in SR1's own report.

### Registration-defect candidates (flagged, never repaired)

- **RD-T2-1 — `null 0.5` is not achievable for the registered statistic.**
  The comparison pools one positive per target against many negatives per
  target across heterogeneous targets, so the permutation null sits where
  that weighting puts it: 0.4525 at L0, and it walks down to 0.2410 at L4
  as pools collapse. #66's "the permutation's own band, no closed forms"
  is the instruction that survives contact with the data and the one this
  leg followed; the closed-form expectation should be struck. 7 of 12
  comparison sets have a null within 0.05 of 0.5.
- **RD-T2-2 — the ladder was registered with widths instead of a target
  pool size.** G2t2 caught the consequence, but it could only fire or not;
  it had no width to negotiate. A future ladder should register a target
  pool size and derive the calipers from it.

### What this leaves for the line

Arm A's question is still open and is *not* answered by "the residual
survives" — nothing here licenses that. The redesign is named in the
report (§9): match across leaves rather than within them; replace hard
calipers with a propensity score on the four observables so match quality
becomes measurable rather than binary; register a target pool size. Arm
B's TRANSPORTS result stands on its own, replicates under ablation, and is
the leg's substantive contribution — but with V-T2a undecided, the 4W
doc's priority 2 (tree + tail joint representation) does **not** become
registrable on a certified object, because the object is not yet certified.

### Planner adjudication of T2 (2026-08-17, appended after the run) — ARM A UNRESOLVABLE AS DESIGNED, ARM B DELIVERS THE FINDING: TASTE TRANSPORTS

**UNDERRESOLVED accepted as routed — and the standing scientific
result is Arm B's replicated TRANSPORTS.** The routing conjunction
(cell 3 required both arms) was the planner's; it under-describes a
leg in which one arm resolved cleanly: **taste transport AUC
0.6031 [0.5837, 0.6236] against its own permutation null
[0.4870, 0.5149], replicating under the 23-community ablation
(0.5941 [0.5751, 0.6148]), fold purity 0-overlap everywhere.** The
person's early selection centroid predicts which NEVER-BEFORE-USED
communities they adopt later — a generalizing continuous Where
coordinate EXISTS (modest, heterogeneous: per-leaf median 0.597,
24/46 leaves below 0.60). Arm A is honestly unresolvable on this
corpus at the registered calipers: the pilot gate fired BEFORE the
full run; the median-pool rule voids L1–L4 independently; the
reportable content is the attrition itself (volume matching alone
costs 86% of the pool while moving AUC by only 0.0029; L4 needs
8× the registered widths for a median pool of 5). The executor's
hardening of the separation criterion (a zero-width band must not
"pass") and its A2 disclosure (the volume caliper first measured on
all-subreddit rather than in-vocabulary counts, repaired pre-pilot)
are both the audit culture working.

**Defects (mine).** **#68 (RD-T2-1):** G3t2's "null 0.5" predicate
is wrong for a statistic pooling one positive per target against
many negatives — the true null walks with pool size (0.4525 at L0
to 0.2410 at L4); #66's permutation-band instruction is the one
that survives. **Convention: rule-29 predicates for composite
statistics state the statistic's OWN null (from its permutation
machinery), never an idealized constant.** **#69 (RD-T2-2):** the
ladder registered caliper WIDTHS rather than target POOL SIZES, so
the feasibility gate could only fire-or-not. **Convention: matching
designs register target pool sizes; realized caliper widths are
the reported quantity.** Routing-granularity note (no number):
independent arms get arm-level cells; conjunctions are reserved for
genuinely joint claims.

**What T1's 0.955 tail now reads as:** dominated by
loyalty/self-history plus support structure whose split THIS corpus
cannot adjudicate by matching — PLUS a real, modest, generalizing
taste component (0.60 on disjoint support, ablation-robust). The
matching question is not dead: it converts into a MEASUREMENT
question, which is T3.

---

## M4-T3 — the identity budget (one currency for the tail)

**REGISTERED 2026-08-17, BEFORE RUN.** Planner: the M-line planner;
executor: dispatched agent. The 4W doc's priority 2, reformulated
so Arm A's question is answered by direct measurement instead of
infeasible matching. Label-free throughout.

### 4W header

- **Object:** Where × Who — the decomposition of TOTAL selection
  identity into named carriers.
- **Fixed:** T1's frozen tree and leaves; T2's frozen per-fold
  embedding and observability vectors; the SR0 vocabulary.
- **Varied:** the REPRESENTATION handed to the same identity
  statistic (observability-only / tree-path / taste-embedding /
  flat), marginally and conditionally.
- **Falsifiers:** observability alone reproducing the flat ceiling
  (identity = support artifact); the taste embedding dying under
  observability-stratified conditioning (transport was
  support-carried); tree ⊕ embedding failing to approach the flat
  ceiling (the joint representation inadequate — priority 2
  refuted in this form).
- **Layer:** R, label-free.

### Question

Everything so far measured identity through different lenses with
different units. T3 puts four representations through ONE
statistic and one permutation machinery, marginally and
conditionally, to produce the IDENTITY BUDGET: how much
reproducible same-author information is carried by (i)
observability alone, (ii) the frozen tree path, (iii) the
continuous taste embedding beyond observability, (iv) nothing we
have named (the residual gap to the flat ceiling).

### Representations (all early→late, all persisted objects)

- **R_obs:** the 4-dim late-side observability vector of T2
  (volume, span, entropy, breadth; z-scored) — its same-author AUC
  is THE direct measurement of the support channel T2's Arm A
  could not isolate;
- **R_tree:** the frozen path code (depth-weighted prefix);
- **R_emb:** the full-early taste centroid in T2's frozen per-fold
  embedding (d = 64);
- **R_flat:** the Hellinger frequency vector (T1's flat ceiling,
  0.9837 anchor).

### Statistics (one machinery, #66/#68-compliant)

Same-author AUC per representation with user-bootstrap CIs
(B = 1000) and permutation nulls (B = 999) — each null from ITS OWN
machinery. CONDITIONAL readings by CORPUS-WIDE stratified
permutation (never within-leaf — the T2 lesson): AUC(R_emb |
observability-decile strata), AUC(R_tree | strata), AUC(R_obs |
leaf strata). Excess-bits companion per T1's conditional-MI
pattern for each marginal and conditional reading (the budget's
common currency; permutation-corrected).

### Verdicts (rule 22; NULL-first; sides declared)

- **V-T3a [support size]:** R_obs marginal — MAJOR (AUC CI above
  0.75) / MODERATE / MINOR (CI below 0.60). Lean: MODERATE .5 /
  MAJOR .3 / MINOR .2.
- **V-T3b [taste beyond support]:** R_emb under
  observability-stratified null — SURVIVES (CI above its
  stratified band) / DIES. Lean SURVIVES .65 (T2's transport was
  support-blind by construction).
- **V-T3c [joint adequacy, priority 2's core]:** AUC(R_tree ⊕
  R_emb concatenated, weights pinned by a declared rule fitted on
  training folds only) vs the R_flat ceiling — ADEQUATE (gap CI
  inside a declared band ε_gap = 0.03) / GAP_REMAINS (the
  unexplained residual is quantified and becomes the object).
  Lean GAP_REMAINS .6 — honest: 0.9837 is a high ceiling.
- Readings: the full budget table (marginal + conditional AUC +
  excess bits per representation); clean-arm replication; per-leaf
  heterogeneity for R_emb.

### Routing (rule 16 — arm-level cells, the T2 lesson)

| # | condition | outcome |
|---|---|---|
| 1 | G0/purity failure | **STOP** |
| 2 | V-T3a MAJOR | **SUPPORT_CHANNEL_MAJOR** (+ the other verdicts still reported) — T1's tail re-typed by dated note as substantially observability-carried |
| 3 | V-T3b SURVIVES | modifier **TASTE_BEYOND_SUPPORT** — the T2 transport finding upgraded: not support-carried |
| 4 | V-T3b DIES | modifier **TRANSPORT_WAS_SUPPORT** — T2's Arm B re-typed by dated note |
| 5 | V-T3c ADEQUATE | **JOINT_REPRESENTATION_ADEQUATE** — priority 2 lands: tree + continuous ≈ the flat ceiling |
| 6 | V-T3c GAP_REMAINS | **BUDGET_WITH_RESIDUAL** — the unexplained share is the next object, quantified |
| 7 | pool/purity underresolution anywhere | that verdict UNDERRESOLVED, others stand (arm-level, never conjunctive) |

### Gates

G0t3: T1/T2 anchors bit-verified (0.9837, 0.9552295265671575,
0.6031 and its null band, the embedding hashes, fold purity);
representations rebuilt through the persisted objects with
provenance. G1t3 (#59): none of the four AUCs is forced;
strata non-degenerate. G2t3 (#66/#68): pilot permutation
calibration (200 users) fixes every null band's achievable width
BEFORE the full run; corpus-wide strata sized to keep median
permutation pools ≥ 20 (the #69 convention: POOL TARGETS
registered, realized widths reported). G3t3: rule-29 with each
statistic's OWN null; rule 24; R-G block + ID-leak scan; stages
< 600 s; target < 45 min.

### Deliverables

The six as always: `scripts/run_suica_m4_t3_identity_budget.py`;
`results/m4_t3_identity_budget/` (gitignored);
`reports/SUICA_M4_T3_IDENTITY_BUDGET_REPORT.md`; outcome append
HERE; one ledger row (EXPLORATORY, label-free); exactly ONE commit
`feat(m4-t): T3 — the identity budget — <SLUG>`, never amended,
never pushed; suite green first (983 expected baseline).

## M4-T3 outcome (2026-08-17, appended by the executing agent) — BUDGET_WITH_RESIDUAL

**Routing: `BUDGET_WITH_RESIDUAL`, modifier `TASTE_BEYOND_SUPPORT`.** Slug
`budget-with-residual`. Report:
`reports/SUICA_M4_T3_IDENTITY_BUDGET_REPORT.md`; harness
`scripts/run_suica_m4_t3_identity_budget.py`; tests
`tests/test_m4_t3_identity_budget.py`; artifacts (gitignored)
`results/m4_t3_identity_budget/`. Label-free; ID-leak scan 0 hits.

### Gates

- **G0t3 PASS.** Every T1 and T2 anchor bit-matches (flat 0.9836592822513264
  / clean 0.9660999136733576; residual 0.9552295265671575; path
  0.7460863278537894; N 1304 / 1269; T2 Arm B 0.6031409031779736 with null
  [0.4869631462640095, 0.5149378939035671] on 907 targets). T2's fold
  embeddings re-verified held-out-pure. `R_flat` marginal rebuilds to
  0.9836592913058296 — |diff| 9.1e-9 from T1's persisted ceiling, i.e. the
  same estimator up to summation order.
- **G1t3 PASS.** AUC spread [0.5060, 0.9837]; nothing forced; strata
  non-degenerate.
- **G2t3 PASS.** All 12 readings separate at the pilot. The registered
  DECILE stratification survives at full width: 10 strata, median
  permutation pool **26** against the registered target of 20, chosen on the
  first rung of the ladder. Null bands are narrow (0.050–0.075) — the
  contrast with T2's collapse is the whole point of conditioning
  corpus-wide instead of within-leaf.
- **G3t3 PASS.** 24 predicates, each against the statistic's OWN null
  (#68, the defect T2 raised and this leg is the first to run under).

### THE IDENTITY BUDGET (full arm, B_boot = 1000, B_perm = 999)

| representation | stratification | AUC | CI95 | own null band | excess bits | pool |
|---|---|---|---|---|---|---|
| R_obs | marginal | 0.7294 | [0.7170, 0.7427] | [0.4879, 0.5136] | 1.734 | 260 |
| R_obs | obs stratum | 0.6018 | [0.5904, 0.6137] | [0.4879, 0.5092] | 0.655 | 26 |
| R_obs | leaf | 0.7153 | [0.7011, 0.7272] | [0.5139, 0.5412] | 1.074 | 21 |
| R_tree | marginal | 0.7523 | [0.7383, 0.7673] | [0.4861, 0.5153] | 1.540 | 260 |
| R_tree | obs stratum | 0.7244 | [0.7091, 0.7404] | [0.4877, 0.5162] | 1.093 | 26 |
| R_tree | leaf | **0.5060** | [0.4896, 0.5214] | [0.4918, 0.5204] | **0.008** | 21 |
| R_emb | marginal | 0.9449 | [0.9386, 0.9508] | [0.4862, 0.5136] | 4.878 | 260 |
| R_emb | obs stratum | **0.9311** | [0.9242, 0.9384] | [0.4905, 0.5143] | 2.823 | 26 |
| R_emb | leaf | 0.8871 | [0.8779, 0.8962] | [0.5211, 0.5469] | 2.400 | 21 |
| R_flat | marginal | 0.9837 | [0.9800, 0.9873] | [0.4871, 0.5135] | 5.957 | 260 |
| R_flat | obs stratum | 0.9783 | [0.9742, 0.9823] | [0.4899, 0.5127] | 3.286 | 26 |
| R_flat | leaf | 0.9580 | [0.9520, 0.9636] | [0.5248, 0.5489] | 3.033 | 21 |

Marginal bits as a share of the flat ceiling (5.957): taste 4.878 (82%),
observability 1.734 (29%), tree path 1.540 (26%). The carriers overlap, so
the shares do not sum; the ORDERING is the finding.

### Verdicts

- **V-T3a — the support channel: MODERATE.** `R_obs` marginal AUC **0.7294
  [0.7170, 0.7427]**, own null [0.4879, 0.5136], 1.734 excess bits.
  **This is the number T2's Arm A could not obtain by matching.** Four
  coordinates carrying no content whatsoever identify the author far above
  their own null — the support channel is real and substantial — but the
  CI does not reach the 0.75 MAJOR threshold. See RD-T3-1: the log1p
  variant reads 0.8447 [0.8355, 0.8538] and WOULD be MAJOR.
- **V-T3b — taste beyond support: SURVIVES.** `R_emb` under
  observability-stratified permutation: **0.9311 [0.9242, 0.9384]** against
  its own null [0.4905, 0.5143], 2.823 excess bits, median pool 26. The
  marginal is 0.9449, so stratifying on observability costs the taste
  coordinate **0.014** of AUC while costing `R_obs` itself **0.128**.
  **T2's TRANSPORTS result is upgraded: the taste coordinate is not
  support-carried.**
- **V-T3c — joint adequacy: GAP_REMAINS.** Weight fitted on training folds
  only: w* = **0.95** (training AUC 0.9565). Held-out joint **0.9450
  [0.9392, 0.9509]** against the flat ceiling **0.9837 [0.9798, 0.9871]**.
  **Gap 0.0387, paired-bootstrap CI [0.0330, 0.0447]** against the declared
  ε_gap = 0.03 — outside the band, but only just (RD-T3-2). Roughly 3.9 AUC
  points, or 5.957 − ~4.9 bits, of the tail remain unexplained by anything
  this line has named.

### Readings

- **Clean arm (23-community ablation) replicates every verdict.**
  n_valid 1269 = T1's exactly; `R_flat` marginal 0.9661 = T1's clean
  ceiling. V-T3a MODERATE (0.7233 [0.7105, 0.7361]); V-T3b SURVIVES
  (0.9107 [0.9015, 0.9193] vs null [0.4879, 0.5108]); V-T3c GAP_REMAINS
  (w* 0.9, joint 0.9272 vs flat 0.9661, gap 0.0389 [0.0320, 0.0458]).
- **Per-leaf heterogeneity for `R_emb`** (observability-stratified): 50
  strata scored, median 0.9418, min 0.8674, max 0.9835, IQR 0.0345 — far
  more homogeneous than T2's Arm B, where half the leaves sat below 0.60.
- **The stratifier:** PC1 of the z-scored late observability vector,
  explaining 0.5157 of its variance, loadings volume 0.410 / span 0.385 /
  entropy 0.538 / breadth 0.627.

### Post-verdict observations (marked as such)

- **The tree carries nothing beyond its own leaf.** `R_tree | leaf` reads
  0.5060 for 0.008 excess bits — indistinguishable from nothing. That is
  also a strong internal check on the conditioning machinery: a
  representation whose information IS the stratifier must fall to its null,
  and it does.
- **The joint is the embedding.** w* = 0.95 and the joint AUC 0.9449583
  sits 4.6e-5 from `R_emb` marginal alone (0.9449125) — adding the whole
  frozen hierarchy to the taste coordinate buys five ten-thousandths of an
  AUC point. **On this evidence the 4W doc's
  priority 2 is mostly tail, not tree**: the hierarchy is a coarse
  quantisation of the same geometry the embedding carries continuously, not
  an independent carrier to be added to it.
- **The conditioning is partial, and it cuts against V-T3b.** PC1 explains
  only 0.5157 of the observability variance, and `R_obs | obs_stratum`
  still reads 0.6018 above its null — residual observability survives inside
  a stratum. V-T3b therefore means "survives conditioning on the dominant
  observability axis", not "on all four coordinates". What holds the verdict
  up is the asymmetry (0.128 vs 0.014), not completeness of conditioning.

### Anomalies (all pre-verdict)

- **A1** The early-side observability vector did not exist — T2 persisted
  only the late side — and was built here from the comment stream under
  SR1's own split rule (4000-timestamp cap included) plus T2's verified
  early count matrix.
- **A2** `R_tree` is deliberately NOT T1's path statistic: T1's 0.7461 used
  an unweighted common prefix, while the registration asks for a
  depth-weighted code and V-T3c needs a concatenable vector. R_tree rebuilds
  to 0.7523. Relatives, not the same estimator; T1's number gates nothing
  here.
- **A3** The leaf-stratified nulls sit slightly above 0.5 (e.g. 0.5248–0.5489
  for `R_flat | leaf`) — the #68 weighting artifact, mild but present. Every
  comparison is against the reading's own band, never 0.5, which is exactly
  what #68 now requires.

### Registration-defect candidates (flagged, never repaired)

- **RD-T3-1 — "z-scored" underspecifies the transform, and V-T3a's cell
  boundary sits inside the ambiguity.** Raw z gives 0.7294 → MODERATE; the
  declared log1p sensitivity gives 0.8447 [0.8355, 0.8538] → MAJOR. Volume,
  span and breadth are heavy-tailed, so raw z-scores let outliers dominate
  the Euclidean geometry and UNDERSTATE the support channel. Same pattern
  clean-side (0.7233 vs 0.8262). A future registration must pin the
  transform.
- **RD-T3-2 — ε_gap was declared without a projection.** G2 projects pool
  sizes (#69) but nothing projects the equivalence band. The gap CI
  [0.0330, 0.0447] is 0.0117 wide and misses the ±0.03 band by 0.0030. A
  verdict this close to its own resolution should have had its band
  projected at G2, exactly as pool targets now are.
- **RD-T3-3 — "observability-decile strata" names one dimension for a
  four-dimensional object.** Deciles need a scalar; PC1 was declared and
  used, and it demonstrably does not exhaust the vector. A design meaning
  "condition on observability" should register the projection, or a
  multivariate scheme with its own pool target.

### What this leaves for the line

The tail is now budgeted rather than argued about. Observability is a real
and substantial carrier (MODERATE, and MAJOR under a defensible transform);
the taste coordinate is the dominant named carrier and is NOT
support-carried; the tree adds nothing the embedding does not already have;
and roughly 3.9 AUC points of ceiling remain unexplained. The residual is
now a quantified object, which is what cell 6 asks for. The obvious next
question is what lives in that residual — and the first thing to rule out
is that it is simply the flat vector's higher dimensionality (1191 vs 64),
which a d-sweep of the embedding would settle cheaply.

### Planner adjudication of T3 (2026-08-17, appended after the run) — THE BUDGET IS IN: TASTE DOMINATES, SUPPORT IS MODERATE, THE TREE IS SUBSUMED, A RESIDUAL REMAINS

**BUDGET_WITH_RESIDUAL + TASTE_BEYOND_SUPPORT accepted.** The
identity budget the 4W doc's eq. (13) needed now has measured
sizes, one machinery, own-null bands throughout (the first leg run
fully under #68):

- **Taste (R_emb) dominates:** 0.9449 marginal, 0.9311 under
  observability conditioning (4.878 / 2.823 excess bits; 82% of
  the 5.957-bit flat ceiling). T2's transport finding is UPGRADED:
  not support-carried. The dominance argument closes RD-T3-3's
  worry independently of conditioning completeness: **R_emb's
  CONDITIONAL 0.9311 exceeds R_obs's MARGINAL ceiling 0.7294 by
  0.20 with disjoint CIs — even total leakage of the support
  channel cannot produce the taste reading.**
- **Support (R_obs) is real and MODERATE:** 0.7294 [0.7170,
  0.7427], 1.734 bits — the direct measurement T2's Arm A could
  not deliver by matching.
- **The tree is subsumed by its own leaves:** within-leaf R_tree
  0.5060, 0.008 bits — as it must be; a clean internal-consistency
  check of the machinery.
- **GAP_REMAINS:** joint (tree ⊕ embedding, w* = 0.95) 0.9450 vs
  flat 0.9837; gap 0.0387 [0.0330, 0.0447] against ε 0.03. About
  3.9 AUC points of reproducible identity live outside the named
  representations — likely representation capacity (d = 64 + path
  vs 1191-dim flat), typed as the next open object, not queued.
- **All three verdicts replicate in the clean arm** (0.7233 /
  0.9107 SURVIVES / gap 0.0389). Per-leaf heterogeneity is tight
  (median 0.9418, min 0.8674) — far tighter than T2's Arm B.

**Defects (mine).** **#70 (RD-T3-1):** "z-scored" underspecified
the observability transform and V-T3a's MAJOR/MODERATE boundary
sits INSIDE the ambiguity (raw 0.7294 vs log1p 0.8447; clean
0.7233 vs 0.8262). The verdict stands as registered (raw was the
registered reading) with the sensitivity reported. **Convention: a
transform that can move a verdict across its boundary is pinned
WITH the verdict at registration; boundary-straddling sensitivity
forces the report to carry both readings.** **#71 (RD-T3-2):**
ε_gap = 0.03 was declared without a G2 projection (realized CI
width 0.0117 against a 0.0030 miss — the verdict is safe here, CI
entirely above ε, but the band was luck not design). **Convention
(the #61/#66 family): every ε-type band carries a G2 projection of
its achievable width.** RD-T3-3 resolved by the dominance argument
above; convention: multidimensional conditioning objects name
their reduction and its explained variance, and verdicts citing
conditioning state whether a dominance argument independently
holds.

**The theory reading (eq. 13, measured):** selection identity =
nested conditional innovations (real through 4–5 levels, then
subsumed) + a CONTINUOUS TASTE COORDINATE (dominant, generalizing
to unseen communities, support-independent by dominance) + a
moderate observability channel + ~4 points of unnamed residual.
The 4W doc's §5.2 extends accordingly (owner's text; noted here,
not edited). **The line's next question is no longer identity —
it is the CONNECTION: which budget component carries the S-line's
trait coupling?**

---

## M4-SR2 — the trait-coupling budget (which component carries r = 0.049?)

**REGISTERED 2026-08-17, BEFORE RUN.** Planner: the M-line
planner; executor: dispatched agent. The 4W doc's priority 5
(external connection), run exactly as its stage E demands: every
technical object (tree, embedding, observability vectors, budget)
was frozen LABEL-FREE in T1–T3; labels open only inside the
stamped joint stage. Governance: SR1's regime verbatim (R-G block,
aggregates only, ID-leak scans, EXPLORATORY corpus-level, no
person claims).

### 4W header

- **Object:** the (Where × Who) → trait connection — the
  DECOMPOSITION of SR1's detected coupling across the identity
  budget's components.
- **Fixed:** T1's tree; T2/T3's per-fold embeddings and
  observability vectors; SR1's primary trait similarity
  (neg-sq-Euclidean over five z-scored Big5) and its permutation
  machinery.
- **Varied:** the SELECTION-side representation (R_obs / R_tree /
  R_emb / R_flat), marginally and under observability-stratified
  permutation.
- **Falsifiers:** R_obs×trait DETECTED with R_emb×trait dying
  under conditioning → SR1's coupling re-typed as an
  activity/support correlate (a major correction to the S-line);
  R_emb×trait NULL marginal → the taste coordinate is
  trait-silent (the γ=0 world made real).
- **Layer:** R with external anchor; V/R/P typing enforced —
  corpus-level relations only.

### Estimands (one machinery: SR1's Mantel over the SR1 pair set, user-permutation nulls B = 999, #66/#68 own bands)

Per representation X ∈ {R_obs, R_tree, R_emb, R_flat}: Mantel
r(X-sim, trait-sim) with SR1's exact trait side. Similarities
pinned: R_obs — negative Euclidean on the registered (raw
z-scored) 4-dim vectors, with the log1p reading carried as the
#70-mandated sensitivity; R_tree — depth-weighted prefix overlap;
R_emb — cosine of taste centroids; R_flat — SR1's Hellinger cosine
(G0 anchor: must reproduce r = 0.048987613136188025 bit-exactly
under SR1's config). CONDITIONAL: r(R_emb-sim, trait-sim) and
r(R_flat-sim, trait-sim) under OBSERVABILITY-STRATIFIED user
permutation (T3's PC1 deciles, pool target ≥ 20 per #69 — the
null preserves any obs–trait coupling, so what survives is
selection–trait coupling beyond support).

### Verdicts (rule 22; NULL-first; arm-level cells per the T2 lesson)

- **V-SR2a [the confound check]:** R_obs×trait marginal —
  DETECTED (own-band exceeded, one-sided positive) / NULL. Lean
  DETECTED .5 (activity–trait couplings are plausible).
- **V-SR2b [the carrier question]:** R_emb×trait marginal
  DETECTED? and under observability-stratified null — SURVIVES /
  DIES / NULL_MARGINAL. Lean: detected-and-survives .5 /
  detected-but-dies .2 / null-marginal .3 (SR1's r was small; the
  taste coordinate may or may not be its carrier).
- **V-SR2c [the flat anchor under conditioning]:** SR1's own
  r under the stratified null — SURVIVES / DIES (does the S-line's
  headline survive the support confound at all — the audit SR1
  never ran).
- Readings: the full coupling-budget table (four representations ×
  {marginal r, band, p; conditional r, band, p}); clean-arm
  replication; the ordering of carriers.

### Routing (rule 16 — arm-level)

| # | condition | outcome |
|---|---|---|
| 1 | G0 (SR1 anchor bit-reproduction) failure | **STOP** |
| 2 | V-SR2c DIES | **SR1_RETYPED_SUPPORT_CONFOUND** — the S-line coupling re-typed by dated note; the owner's conjecture's real-data evidence re-scoped |
| 3 | V-SR2c SURVIVES and V-SR2b detected-and-survives | **TASTE_CARRIES_THE_COUPLING** — the generalizing coordinate is trait-coupled; the selection conjecture deepens to its strongest form |
| 4 | V-SR2c SURVIVES and V-SR2b dies-or-null | **COUPLING_BEYOND_TASTE** — the coupling survives support but is not carried by the d=64 taste coordinate; carrier named open |
| 5 | any UNDERRESOLVED (pool/width) | that verdict UNDERRESOLVED, others stand |

### Gates

G0sr2: SR1's primary reproduced bit-exactly under its persisted
config (r, p, N, pairs); T3's budget anchors verified; embedding
purity re-checked. G1sr2 (#59): none of the couplings forced;
strata non-degenerate. G2sr2 (#66/#68/#69/#71): pilot permutation
calibration (200 users) fixes every band's achievable width and
the ε-free verdict feasibility BEFORE the joint stage; pool
target 20. G3sr2: config-before-joint stamp (SR1's ordering
pattern — labels joined only after the hash; run-log proof);
rule-29 own-null predicates; rule 24; R-G + ID-leak scan; stages
< 600 s; target < 40 min.

### Deliverables

The six as always: `scripts/run_suica_m4_sr2_coupling_budget.py`;
`results/m4_sr2_coupling_budget/` (gitignored);
`reports/SUICA_M4_SR2_COUPLING_BUDGET_REPORT.md`; outcome append
HERE; one ledger row (EXPLORATORY, corpus-level); exactly ONE
commit `feat(m4-t): SR2 — the trait-coupling budget — <SLUG>`,
never amended, never pushed; suite green first (994 baseline).

## M4-SR2 outcome (2026-08-17, appended by the executing agent) — COUPLING_BEYOND_TASTE

**Routing: `COUPLING_BEYOND_TASTE` (cell 4), modifier
`SUPPORT_TRAIT_COUPLING_NULL`.** Slug `coupling-beyond-taste`. Report:
`reports/SUICA_M4_SR2_COUPLING_BUDGET_REPORT.md`; harness
`scripts/run_suica_m4_sr2_coupling_budget.py`; tests
`tests/test_m4_sr2_coupling_budget.py`; artifacts (gitignored)
`results/m4_sr2_coupling_budget/`. EXPLORATORY, corpus-level.
ID-leak scan 0 hits.

### Label governance — the T-line ordering paid off

| event | UTC |
|---|---|
| `config_stamped` (sha256 `abf198c6af3af6ae…`) | 2026-08-17T10:14:34.240101+00:00 |
| `first_join` | 2026-08-17T10:14:53.741310+00:00 |
| `labels_opened` (Big5 read) | 2026-08-17T10:14:53.771777+00:00 |

Stamp precedes join: true. Join precedes the first label read: true. Every
logged event before the join is label-free: true. **The G2 pilot is itself
label-free** — it calibrates against a synthetic standard-normal trait
surrogate, so the resolution question was settled before any real Big5
value was touched.

### Gates

- **G0sr2 PASS.** **SR1's primary is reproduced BIT-EXACTLY by calling
  SR1's own functions**: r = 0.048987613136188025, 852165 pairs, p = 0.001,
  null sd = 0.009064019613144935. T3 anchors match (0.7293990670964055,
  0.9311359256886822); T2 embedding purity re-verified.
- **G1sr2 PASS.** r spread [-0.0104, 0.0490], nothing forced; smallest
  observability stratum 130 against a pool target of 20.
- **G2sr2 PASS (label-free).** Projected full-scale null p95 across all
  eight readings 0.0114–0.0228, all below SR1's r = 0.049.
- **G3sr2 PASS.** 28 own-null predicates; stamp-before-join proven from the
  run log.

### THE COUPLING BUDGET (full arm, B_perm = 999)

| representation | null | pair set | Mantel r | own null band | p (1-sided) | z | detected |
|---|---|---|---|---|---|---|---|
| R_obs | marginal | full | **-0.010352** | [-0.02578, 0.02786] | 0.7680 | -0.75 | **False** |
| R_obs | stratified | full | -0.010352 | [-0.02523, 0.01148] | 0.5960 | -0.27 | False |
| R_obs (log, #70) | marginal | full | 0.012340 | [-0.01915, 0.02096] | 0.1200 | — | **False** |
| R_tree | marginal | within-fold | 0.007071 | [-0.01146, 0.01202] | 0.1350 | 1.14 | False |
| R_tree | stratified | within-fold | 0.007071 | [-0.00932, 0.01513] | 0.2530 | 0.63 | False |
| R_emb | marginal | within-fold | **0.023885** | [-0.02432, 0.02457] | 0.0310 | 2.01 | **True** |
| R_emb | stratified | within-fold | 0.023885 | [-0.01004, 0.03165] | 0.1110 | 1.21 | **False** |
| R_flat | marginal | within-fold | 0.047687 | [-0.01870, 0.02038] | 0.0010 | 4.82 | True |
| R_flat | marginal | full | **0.048988** | [-0.01756, 0.01728] | 0.0010 | 5.23 | True |
| R_flat | stratified | full | **0.048988** | [-0.00731, 0.02416] | 0.0010 | 5.13 | **True** |

Carrier ordering (marginal, within-fold — like for like): **R_flat 0.0477
(z 4.82) > R_emb 0.0239 (z 2.01) > R_tree 0.0071 (z 1.14) > R_obs -0.0087
(z -0.67)**.

### Verdicts

- **V-SR2a — the confound check: NULL.** R_obs × trait r = **-0.010352**,
  p = 0.768, own band [-0.02578, 0.02786]. The #70 log1p sensitivity agrees:
  r = 0.012340, p = 0.120, also undetected. **The support channel carries no
  trait coupling at all, in either transform.** The registration's first
  falsifier ("R_obs DETECTED with R_emb dying → re-type SR1 as an
  activity/support correlate") therefore never fires.
- **V-SR2b — the carrier question: detected-but-DIES.** R_emb × trait
  marginal r = **0.023885** (p = 0.031, band [-0.02432, 0.02457]); under the
  observability-stratified null r = 0.023885 against a band of
  [-0.01004, 0.03165], p = 0.111 — **inside**. The d = 64 taste coordinate
  carries roughly half of R_flat's coupling on the same pair set and none of
  it robustly.
- **V-SR2c — the audit SR1 never ran: SURVIVES.** SR1's own coupling under
  **user permutation within observability deciles**: r = **0.048988**, band
  [-0.00731, 0.02416], p = 0.0010, z = 5.13. **SR1's headline is not a
  support confound.**

### Readings

- **Clean arm (23-community ablation).** n_valid 1269 = T1's exactly.
  V-SR2a NULL (r = -0.013664, p = 0.851) and V-SR2c SURVIVES (r = 0.028701,
  stratified p = 0.026, z = 1.97) both replicate. V-SR2b reads NULL_MARGINAL
  clean (r = 0.018123, p = 0.076) rather than detected-but-DIES — see
  RD-SR2-3; both readings agree that the taste coordinate is not a
  surviving carrier.
- **THE ABLATION COSTS 41% OF SR1's HEADLINE.** 0.048988 → 0.028701.
  It still detects and still survives the stratified null, so the finding
  is not overturned — but a large minority of the S-line's headline
  coupling is carried by subreddits whose whole subject is personality
  typology, where a selection–trait coupling is close to tautological.
  **Flagged for adjudication; not resolved here.**
- **The within-fold mask is representative:** R_flat 0.047687 masked vs
  0.048988 full, a difference of 0.0013.
- **The stratified null is STRICTER, not weaker.** Within-stratum
  permutation preserves observability–selection structure, lifting the
  null's upper tail (R_emb marginal band [-0.024, 0.025] → stratified
  [-0.010, 0.032]). That is why R_emb dies, and it is correct behaviour.

### Post-verdict observation

**The trait coupling lives in the same residual T3 quantified.** T3 found
tree ⊕ embedding falls 0.0387 short of the flat ceiling and typed it
representation-capacity. SR2 finds the d = 64 taste coordinate captures
about half the marginal coupling and none of it robustly, while the full
1191-dim Hellinger vector carries all of it and survives conditioning. Two
independent lines now point at the same object: **the fine-grained
community structure that the 64-dimensional compression discards.** Note
also the dissociation this leg is the first to show — observability is a
substantial carrier of IDENTITY (T3: AUC 0.7294) and a null carrier of
TRAIT (here: r = -0.010). Those are different questions with different
answers.

### Anomalies (all pre-verdict, pre-label)

- **A1 — the first G0sr2 run FAILED and stopped the leg.** Two T3 anchor
  constants had been transcribed from rounded display output instead of the
  persisted values. The gate caught it; **no stamp was issued and no label
  was opened**. The harness was then changed so a failed G0 can never issue
  a stamp — on the first run the stamp was written before the PASS check,
  which is the wrong order for a governance artifact even though nothing
  joined. Both the error and the ordering fix precede the join.
- **A2 — the budget cannot run on SR1's pair set** (R_tree/R_emb are
  fold-local; cross-fold entries are undefined). Within-fold mask used,
  with R_obs/R_flat also on the full set. RD-SR2-1.
- **A3 — the four representations bring three different halves** (full /
  late / early), inherited from three frozen objects. RD-SR2-2.

### Registration-defect candidates

- **RD-SR2-1** — "SR1's exact pair set" is unavailable to two of the four
  representations; a future registration must register the mask or specify
  a pooled representation.
- **RD-SR2-2** — the coupling budget invites row-to-row comparison across
  representations that are not half-aligned; the registration should say so.
- **RD-SR2-3** — no convention exists for full-versus-clean disagreement
  INSIDE one verdict. Arm-level routing separates the three verdicts but
  says nothing about which arm owns a verdict when full and clean disagree.
  The full arm owns it here as the registered primary; the line should pin
  the rule.

### What this leaves for the line

SR1's coupling is real, is not a support artifact, and is not carried by
the generalizing taste coordinate — its carrier is named open, and the
evidence points at the flat representation's fine structure. Two questions
now dominate: (1) how much of the coupling is tautological, given that
removing 23 typology-named communities costs 41% of it; (2) whether an
embedding with more capacity than d = 64 recovers what the compression
loses, which would identify the carrier and would also test T3's
representation-capacity typing of its own residual.

### Planner adjudication of SR2 (2026-08-17, appended after the run) — THE COUPLING SURVIVES ITS AUDIT; THE CARRIER IS NOT THE TASTE CENTROID; THE DISSOCIATION IS THE FINDING

**COUPLING_BEYOND_TASTE accepted.** Three verdicts, one picture:

1. **V-SR2c SURVIVES** — SR1's headline r = 0.048987613136188025
   holds under observability-stratified permutation (p = 0.0010,
   z = 5.13): the S-line's real-data coupling is NOT a support
   confound. The audit SR1 never ran now exists and strengthens it.
2. **V-SR2a NULL** — observability carries NO trait coupling
   (r = −0.0104, p = 0.768; the #70 log1p sensitivity agrees at
   p = 0.120). The plausible confound is measured absent.
3. **V-SR2b detected-but-DIES** (clean: NULL_MARGINAL — both
   agree the taste coordinate is not a surviving carrier): the
   d = 64 taste centroid's coupling is small (0.0239, p = .031)
   and does not survive conditioning.

**The dissociation, stated as the T-line's central empirical
lesson:** the component that carries IDENTITY (the taste centroid,
82% of the identity bits; and observability, 0.73 AUC) is largely
TRAIT-SILENT, while the trait coupling rides fine-grained selection
structure that the identity-optimal compression discards (carrier
ordering: R_flat 0.0477/z 4.82 ≫ R_emb 0.0239/z 2.01 ≫ R_tree ≈
R_obs ≈ null). **S1's γ = 0 lesson — stable identity channels can
be trait-silent — has now appeared INSIDE the real corpus's
budget.** And the 23-community ablation costs 41% of the headline
(0.0490 → 0.0287, still detected at strat p = .026): a substantial
minority of the coupling is carried by explicitly
personality-adjacent communities; the majority is not.

**Defects (mine).** **#72 (RD-SR2-1/2):** the registration demanded
"SR1's exact pair set" for representations that are fold-local —
impossible for two of four — and thereby invited cross-row
comparisons over non-aligned pair sets. **Convention (rule 33's
cousin at the pair-set level): every coupling row declares its pair
set and half-alignment; comparisons are licensed only within
aligned sets, and the table marks alignment classes.** The
executor's within-fold mask (cost 0.0013, disclosed) was the right
in-leg repair. **#73 (RD-SR2-3):** no convention existed for
full-vs-clean divergence INSIDE one verdict (V-SR2b read
detected-but-DIES full, NULL_MARGINAL clean). **Convention: the
registered primary arm routes; the replication arm's divergence is
FLAGGED on the verdict line itself, never averaged away.** The
executor's A1 (a failed G0 stopping the leg BEFORE any stamp, then
hardening the harness so a failed G0 can never issue a stamp) is
adopted as the stamp-order convention for all sealed legs.

**What is now open, and precisely:** the coupling's carrier lives
between d = 64 and the 1191-dim flat vector — the same territory
as T3's 3.9-point identity gap. Localizing it is SR3.

---

## M4-SR3 — carrier localization (where between d=64 and flat does the coupling live?)

**REGISTERED 2026-08-17, BEFORE RUN.** Planner: the M-line planner;
executor: dispatched agent. Governance: SR2's regime verbatim
(labels open only inside the stamped joint stage; all
representations frozen label-free first; R-G block; ID scans).

### 4W header

- **Object:** the (Where × Who) → trait connection's CARRIER —
  its localization along the representation-capacity axis.
- **Fixed:** T2/T3's embedding construction recipe (per-fold PPMI
  + SVD, training-early-only) at d ∈ {64 (anchor), 128, 256, 512};
  the flat vector; SR1's trait side and machinery; T3's PC1
  deciles.
- **Varied:** embedding dimension only.
- **Falsifiers:** coupling flat across d then jumping only at full
  → the carrier is individual-community indicators (loyalty-
  adjacent, not a continuous coordinate); coupling rising smoothly
  to the flat value by d ≤ 512 → the carrier is fine-grained
  CONTINUOUS structure (the same object as T3's identity gap).
- **Layer:** R with external anchor; corpus-level only.

### Estimands (SR2's machinery verbatim; within-fold pair sets, alignment declared per #72)

Per d ∈ {64, 128, 256, 512}: Mantel r(emb_d-sim, trait-sim),
marginal and under T3-PC1-stratified permutation (B = 999, own
bands, pool target 20). Anchors: d = 64 must reproduce SR2's
0.023885 (marginal) bit-exactly; R_flat within-fold 0.047687 as
the ceiling row. IDENTITY companion (label-free stage, before the
stamp): same-author AUC per d (T3's machinery) — the identity
curve and the coupling curve on ONE dimension axis, so the
dissociation becomes a picture.

### Verdicts (rule 22; NULL-first; arm-level; #73 divergence flags)

- **V-SR3a [the coupling curve]:** the d at which stratified
  coupling first becomes DETECTED (own band), if any; monotonicity
  reading. Classification: CONTINUOUS_CARRIER (detected at some
  d ≤ 512 with r ≥ half the flat within-fold value) /
  INDICATOR_CARRIER (undetected through 512 while flat detects) /
  MIXED (detected but below half-flat) / UNDERRESOLVED.
- **V-SR3b [the identity curve, reading]:** does the identity gap
  close by d ≤ 512 (T3's 0.9450 → toward 0.9837)? Reported, no
  gate — it types T3's GAP_REMAINS residual on the same axis.
- Clean-arm replication of the coupling curve (#73 flags on any
  divergence).

### Leans

**L-SR3 [CONTINUOUS_CARRIER .40 / MIXED .30 / INDICATOR_CARRIER
.20 / underresolved .10].**

### Gates

G0sr3: SR2 anchors bit-exact (0.023885, 0.047687, 0.048987613136188025
full-set; T3's 0.9449/0.9837); embedding purity per fold at every
d. G1sr3 (#59): the curve is not forced (d=64 sits well below
flat on BOTH axes). G2sr3 (#66/#68/#69/#71): pilot calibration of
every band's achievable width; pool target 20. G3sr3:
config-before-joint stamp with the A1-hardened order (G0 PASS
precedes stamp issuance structurally); labels joined only after
the hash; identity companion computed BEFORE the stamp
(label-free); rule 24; R-G + ID scans; stages < 600 s; target
< 40 min.

### Deliverables

The six as always: `scripts/run_suica_m4_sr3_carrier_localization.py`;
`results/m4_sr3_carrier_localization/` (gitignored);
`reports/SUICA_M4_SR3_CARRIER_LOCALIZATION_REPORT.md`; outcome
append HERE; one ledger row (EXPLORATORY, corpus-level); exactly
ONE commit `feat(m4-t): SR3 — carrier localization — <SLUG>`,
never amended, never pushed; suite green first (1003 baseline).

## M4-SR3 outcome (2026-08-17, appended by the executing agent) — INDICATOR_CARRIER

**Routing: `INDICATOR_CARRIER`.** Slug `indicator-carrier`. Report:
`reports/SUICA_M4_SR3_CARRIER_LOCALIZATION_REPORT.md`; harness
`scripts/run_suica_m4_sr3_carrier_localization.py`; tests
`tests/test_m4_sr3_carrier_localization.py`; artifacts (gitignored)
`results/m4_sr3_carrier_localization/`. EXPLORATORY, corpus-level.
ID-leak scan 0 hits.

### Label governance (SR2's regime, with the A1 hardening now standing)

| event | UTC |
|---|---|
| `config_stamped` (sha256 `fa3bf734839b0b9e…`) | 2026-08-17T10:29:54.206498+00:00 |
| `first_join` | 2026-08-17T10:30:21.437455+00:00 |
| `labels_opened` | 2026-08-17T10:30:21.449356+00:00 |

Stamp<join true; join<labels true; all pre-join events label-free true;
**G0 PASS structurally precedes stamp issuance** true; **the entire
identity curve was computed before the stamp** true.

### Gates

- **G0sr3 PASS.** All anchors bit-exact: SR2's d=64 marginal coupling
  0.023884547516782918 and within-fold flat 0.04768658177503308 both
  reproduced by the sweep; SR1's full-set 0.048987613136188025; T3's
  identity AUCs 0.9449125076918007 (d=64) and 0.9836592913058296 (flat)
  both reproduced by the identity curve. Purity: all 20 (fold × d) cells
  pure, train×held-out overlap 0; available rank 1043 at every fold so
  d=512 is genuinely available (d_effective = d at all four dimensions).
- **G1sr3 PASS** (d=64 below flat on both axes). **G2sr3 PASS**
  (label-free; projected full p95 0.0154–0.0228, all below half-flat
  0.0238). **G3sr3 PASS** (25 own-null predicates + stamp order).

### The two curves

**Identity (label-free, pre-stamp):**

| representation | AUC | CI95 | excess bits | gap to flat |
|---|---|---|---|---|
| d=64 | 0.944913 | [0.9388, 0.9509] | 4.877 | 0.038747 |
| d=128 | 0.952439 | [0.9461, 0.9580] | 4.993 | 0.031221 |
| d=256 | 0.957391 | [0.9521, 0.9625] | 5.085 | 0.026269 |
| d=512 | 0.959635 | [0.9540, 0.9646] | 5.132 | 0.024024 |
| flat | 0.983659 | [0.9795, 0.9872] | — | — |

**Coupling (full arm, within-fold pair set, B=999):**

| representation | marginal r | p | det | stratified p | det |
|---|---|---|---|---|---|
| d=64 | 0.023885 | 0.0220 | True | 0.1260 | **False** |
| d=128 | 0.025248 | 0.0180 | True | 0.0950 | **False** |
| d=256 | 0.026100 | 0.0160 | True | 0.0760 | **False** |
| d=512 | 0.026660 | 0.0150 | True | 0.0630 | **False** |
| flat | 0.047687 | 0.0010 | True | 0.0010 (z 4.46) | **True** |

### Verdicts

- **V-SR3a: INDICATOR_CARRIER.** No embedding dimension up to 512 clears
  its own stratified band while the flat vector clears it at p = 0.0010,
  z = 4.46. The marginal curve *is* monotone non-decreasing (0.023885 →
  0.025248 → 0.026100 → 0.026660) and the stratified p-values fall
  monotonically (0.1260 → 0.0950 → 0.0760 → 0.0630) — the trend points the
  right way — but eight-fold more capacity plateaus at **56% of the
  within-fold flat value** and never arrives. **The carrier is not a
  low-rank continuous coordinate at any d ≤ 512.**
- **V-SR3b (reading, no gate): the identity gap narrows but does not
  close.** 0.038747 at d=64 → 0.024024 at d=512: **38.0% of T3's
  GAP_REMAINS residual closes**, and the CI at 512 still does not reach
  flat's. T3 typed that residual representation-capacity; this sweep says
  capacity is *part* of it and not all of it — measured on the same axis
  with the same machinery.

### Clean arm and #73 divergence flags

Clean classification **INDICATOR_CARRIER** — the headline does not diverge
(n_valid 1269 = T1's exactly). But **all four marginal readings diverge**:
every d detects marginally on the full arm (p 0.015–0.022) and none does
on the clean arm (p 0.087–0.125), while the flat vector detects on both.
Per #73 the primary (full) arm routes and the divergence is flagged, and
here **the flag is itself the finding: what little coupling the compressed
representation carries is largely the 23 typology-named communities** —
SR2's 41% ablation effect, now localized to the embedding.

### Anomalies (all pre-verdict)

- **A1 — one SVD per fold, sliced (pre-stamp, by design, asserted).** The
  frozen recipe returns `vt[:d].T * sv[:d]` from a single deterministic
  SVD, so d=128/256/512 are prefixes of the same factorisation as d=64.
  The harness asserts bit-equality against a genuine `ppmi_svd(counts, 64)`
  call at every fold before using any slice, and a unit test checks the
  same property across five widths. This is an optimisation with a proof,
  not a substitution.
- **A2 — the ceiling row is the WITHIN-FOLD flat value, not the full-set
  one (pre-verdict, #72).** Comparing a fold-local curve against a
  full-set ceiling is exactly the misalignment #72 exists to prevent, so
  the half-flat threshold is 0.5 × 0.04768658177503308 = 0.023843. Note
  SR2's d=64 marginal (0.023885) sits a hair above that line by
  coincidence; the classification turns on the STRATIFIED reading, where
  d=64 is undetected.
- **A3 — the two curves are different statistics on one axis
  (pre-verdict).** An AUC and a Mantel r are never subtracted here; the
  shared axis buys the SHAPE of each as capacity grows, nothing more.

### Registration-defect candidate

- **RD-SR3-1 — the sweep stops at 512 for no registered reason.** The
  available rank is 1043 at every fold, so d ∈ {768, 1043} were reachable
  at negligible extra cost, and the stratified p-values were still falling
  monotonically at 512 (0.0630). INDICATOR_CARRIER is the correct verdict
  *for the registered grid*, but the grid's top end was not justified by a
  projection the way pool sizes (#69) and bands (#71) now are. A capacity
  sweep should register its endpoint against the available rank.

### What this leaves for the line

The dissociation now has a shape. Both curves rise with capacity and both
stall well short of the flat vector: identity closes 38% of its gap,
coupling reaches 56% of the flat value and never clears the stricter band.
**The trait coupling — and a large part of T3's identity residual — rides
on specific-community membership rather than on a smooth taste geometry**,
and the clean arm sharpens that to: for the compressed representation, on
the typology-named communities in particular. The two open questions are
(1) whether d ∈ {768, 1043} closes either curve (RD-SR3-1), and (2)
whether the coupling that survives ablation is carried by identifiable
community *sets*, which a sparse-selection analysis on the flat vector
would answer directly.

### Planner adjudication of SR3 (2026-08-17, appended after the run) — NO LOW-RANK CARRIER; THE ENDPOINT QUESTION SHARPENS INTO A METRIC QUESTION

**INDICATOR_CARRIER accepted, with its scope stated exactly:** no
embedding dimension through 512 clears its stratified band (p
0.126 → 0.063, monotonically falling, never detecting) while the
flat vector clears at z = 4.46; the marginal coupling plateaus at
55.9% of the flat ceiling; the identity gap closes only 38.0%
(0.0387 → 0.0240, CIs disjoint from flat at 512). Clean arm
replicates the classification; the four #73 marginal-divergence
flags are consistent with SR2's 41% ablation cost. The A1
stamp-order hardening ran as standing convention
(g0_passed_before_stamp True; identity curve pre-stamp,
label-free), and the executor's process note (its own drafted
timestamps corrected against artifacts before commit) is rule-30
culture in prose.

**Defect #74 (RD-SR3-1, mine):** the sweep endpoint 512 was
registered against nothing — available rank is 1043 and the
stratified p was still falling at the cut. **Convention (the
#69/#71 family): a capacity sweep registers its endpoint against
the available rank, or states why it stops short.** The
classification therefore reads "no low-to-mid-rank continuous
carrier through 512"; the edge is SR4's.

**The sharper question the endpoint hides (planner derivation):**
at d = 1043 the truncation vanishes — the embedding becomes a
ROTATION of the full space, and what remains different from the
Hellinger flat row is the METRIC (PPMI weighting) alone. So the
endpoint completion is a three-way disambiguation, not a formality:
detection appearing by 1043 means a high-rank continuous carrier;
the PPMI-metric full-rank row failing while Hellinger detects
means the coupling is **METRIC-BORNE — living in the
common/popular community mass that PPMI reweighting suppresses**
(PPMI emphasizes distinctive co-occurrence; Hellinger keeps mass
on frequent communities); truncations failing while ppmi-full
detects means rank-borne after all. Each cell re-types the carrier
differently. SR4 below.

---

## M4-SR4 — the endpoint and the metric (completing the carrier curve to full rank)

**REGISTERED 2026-08-17, BEFORE RUN.** Planner: the M-line planner;
executor: dispatched agent. A micro-leg: SR3's machinery verbatim,
three new rows, one three-way classification. Governance: SR2/SR3's
regime verbatim (A1-hardened stamp; labels after hash; R-G; ID
scans).

### 4W header

- **Object:** the trait-coupling carrier's final type — rank-borne
  vs indicator vs METRIC-borne.
- **Fixed:** everything from SR3 (recipe, folds, pair sets, strata,
  trait side, bands machinery).
- **Varied:** three rows only — d = 768, d = 1043 (= available
  rank; the pure-rotation row), and **ppmi-full-metric** (the PPMI
  transform with NO truncation, cosine similarity — the metric
  isolated from capacity).
- **Falsifiers / cells:** each of the three carrier types has a
  row pattern that kills the other two.
- **Layer:** R with external anchor; corpus-level.

### Verdict (rule 22; NULL-first; #73 flags)

**V-SR4** on the stratified detections of the three rows plus
SR3's persisted curve:

| pattern | classification |
|---|---|
| detection at 768 or 1043 with r ≥ half-flat | **CONTINUOUS_CARRIER_HIGH_RANK** — SR3's cell amends by dated note |
| truncations (incl. 1043-rotation) undetected AND ppmi-full undetected AND Hellinger flat detected | **METRIC_BORNE** — the coupling lives in what PPMI suppresses (common-community mass); the carrier re-types |
| ppmi-full detected while 768/1043 truncations undetected | **RANK_ANOMALY_NAMED** (should be impossible — 1043 IS full rank; a detection difference between 1043 and ppmi-full isolates numerical/whitening effects; reported as an instrument finding) |
| all three undetected AND Hellinger flat detected AND 1043 ≈ ppmi-full (both null) | **METRIC_BORNE** (same cell as above — the operative contrast is PPMI-vs-Hellinger) |
| pool/width underresolution | **UNDERRESOLVED** |

Leans: METRIC_BORNE .45 / CONTINUOUS_CARRIER_HIGH_RANK .35 /
other .20 — the falling-p trend argues for late detection; the
38%-only identity-gap closure argues the flat advantage is not
mere capacity.

### Gates

G0sr4: SR3's full curve bit-verified (all ten coupling rows, both
identity endpoints, the anchors); rank availability 1043 confirmed.
G1sr4 (#59): the three rows are not forced. G2sr4 (#71): pilot
band projection for the three rows. G3sr4: A1-hardened stamp;
labels after hash; identity companion at 768/1043 (label-free,
pre-stamp) to extend V-SR3b's reading; #72 alignment (within-fold);
rule 24; stages < 600 s; target < 20 min.

### Deliverables

The six as always: `scripts/run_suica_m4_sr4_endpoint_metric.py`;
`results/m4_sr4_endpoint_metric/` (gitignored);
`reports/SUICA_M4_SR4_ENDPOINT_METRIC_REPORT.md`; outcome append
HERE; one ledger row; exactly ONE commit
`feat(m4-t): SR4 — the endpoint and the metric — <SLUG>`, never
amended, never pushed; suite green first (1012 baseline).

## M4-SR4 outcome (2026-08-17, appended by the executing agent) — METRIC_BORNE

**Routing: `METRIC_BORNE`.** Slug `metric-borne`. Report:
`reports/SUICA_M4_SR4_ENDPOINT_METRIC_REPORT.md`; harness
`scripts/run_suica_m4_sr4_endpoint_metric.py`; tests
`tests/test_m4_sr4_endpoint_metric.py`; artifacts (gitignored)
`results/m4_sr4_endpoint_metric/`. EXPLORATORY, corpus-level.
ID-leak scan 0 hits.

### Label governance

| event | UTC |
|---|---|
| `config_stamped` (sha256 `1f7ec6a6b0ab6970…`) | 2026-08-17T10:42:11.279493+00:00 |
| `first_join` | 2026-08-17T10:42:33.129163+00:00 |
| `labels_opened` | 2026-08-17T10:42:33.139959+00:00 |

stamp<join true; join<labels true; all pre-join events label-free true;
G0 PASS precedes stamp issuance true; identity companion pre-stamp true.

### Gates

- **G0sr4 PASS.** SR3's curve is **RECOMPUTED and bit-compared, not
  copied**: all **10/10 coupling rows** and **5/5 identity rows** reproduce
  exactly. n_valid 1304; all folds pure.
- **G1sr4 PASS**, **G2sr4 PASS** (label-free; projected full p95
  0.0154–0.0227, all below half-flat 0.023843), **G3sr4 PASS** (32 own-null
  predicates + stamp order).

### The rotation check (the leg's instrument premise)

| fold | n train | rank available | d=1043 effective | exactly full rank |
|---|---|---|---|---|
| 0–3 | 1043 | 1043 | 1043 | True |
| 4 | 1044 | 1044 | 1043 | False |

**Max |d=1043 − ppmi-full| = 5.329e-15.** The algebra holds numerically:
with PPMI = U S Vᵀ, freq·(V S) = (freq·PPMIᵀ)·U and right-multiplication by
an orthonormal-column matrix preserves inner products. The RANK_ANOMALY
cell is correctly unreachable. Fold 4's one-component shortfall is
controlled by the ppmi-full row, which never truncates anywhere.

### The completed coupling curve (full arm, within-fold, B=999)

| representation | marginal r | p | det | stratified p | z | det |
|---|---|---|---|---|---|---|
| d=64 | 0.023885 | 0.0220 | True | 0.1260 | 1.17 | False |
| d=128 | 0.025248 | 0.0180 | True | 0.0950 | 1.33 | False |
| d=256 | 0.026100 | 0.0160 | True | 0.0760 | 1.46 | False |
| d=512 | 0.026660 | 0.0150 | True | 0.0630 | 1.57 | False |
| **d=768** | **0.026824** | 0.0090 | True | **0.0520** | 1.59 | **False** |
| **d=1043** | **0.026841** | 0.0140 | True | **0.0520** | 1.65 | **False** |
| **ppmi-full** | **0.026841** | 0.0170 | True | **0.0560** | 1.66 | **False** |
| flat | 0.047687 | 0.0010 | True | **0.0010** | **4.45** | **True** |

### The completed identity curve (label-free, pre-stamp)

d=64 0.944913 / d=128 0.952439 / d=256 0.957391 / d=512 0.959635 /
**d=768 0.959913** / **d=1043 0.959946** / **ppmi-full 0.959946** /
flat 0.983659.

### V-SR4: METRIC_BORNE

No PPMI representation detects at ANY rank — including the untruncated one
— while the Hellinger flat row detects at p = 0.0010, z = 4.45.

**Capacity is provably exhausted, and this is the leg's point.** The
stratified coupling plateaus at 0.026841 from d=768 through full rank
through the unfactorised PPMI. There is no truncation left to remove, so
the remaining shortfall to flat cannot be capacity:

- **Coupling axis:** capacity buys **12.4%** of the d=64 shortfall;
  **87.6% is metric**. Full-rank PPMI reaches only **56.3%** of the flat
  value.
- **Identity axis:** capacity buys **38.8%** of T3's gap; **61.2% is
  metric** — 0.023713 of the original 0.038747 survives at zero truncation.

**Both axes, independently, return the same answer: the difference is
PPMI-vs-Hellinger geometry, not representation capacity. The coupling —
and most of T3's identity residual — lives in what PPMI suppresses, i.e.
the common-community mass that Hellinger's sqrt-frequency keeps and PPMI's
reweighting discounts.**

**Honest margin.** The binary detection is the weaker statement: the
high-rank rows sit at p = 0.052 / 0.052 / 0.056 — just short of the band,
not comfortably null. What does not depend on the threshold is the z
contrast, **1.66 for untruncated PPMI against 4.45 for Hellinger**, on
identical pairs with identical machinery.

### Candidate re-typing, flagged for adjudication (NOT asserted here)

T3 typed its 0.0387 GAP_REMAINS residual **representation-capacity**. On
the completed axis that typing does not survive: 61.2% of it persists at
zero truncation. **This is a candidate re-typing of a standing T3 result**
and is reported for the planner rather than applied by the executor.

### Clean arm and #73 flags

Clean classification **METRIC_BORNE** — same cell (n_valid 1269 = T1's
exactly). **Seven #73 divergence flags, all marginal**: every PPMI row
detects marginally on the full arm and none does on the clean arm, while
Hellinger flat detects on both. The typology-community dependence SR2 and
SR3 found now demonstrably persists all the way to full rank. Full arm
routes per #73.

### Anomalies (all pre-verdict)

- **A1 — the available rank is not uniform across folds (pre-stamp,
  disclosed).** Fold 4 trains on 1044 authors, so the registered d=1043
  row is exactly full rank in four folds and one component short in the
  fifth. The ppmi-full row is the clean control and agrees to 5.3e-15.
- **A2 — SR3's curve was recomputed rather than read (pre-verdict, by
  design).** A transcription check proves numbers were typed correctly;
  recomputation proves the machinery still produces them. 15/15 rows.
- **A3 — d=1043 and ppmi-full are one estimand by two routes
  (pre-verdict).** They are not independent evidence, and the report says
  so; their agreement is an instrument check, not a replication.

### Registration-defect candidate

- **RD-SR4-1 — the detection predicate is binary at a boundary the data
  sits on.** Three rows land at p = 0.052–0.056 against a 0.05 threshold.
  The verdict is unchanged either way *because the capacity argument does
  not use the threshold at all*, but a pattern table keyed on binary
  detections is fragile exactly where this corpus put its answer. Cells
  should key on effect-size contrasts (here z 1.66 vs 4.45) with detection
  as a secondary report, or declare a near-band zone.

### What this leaves for the line

The carrier question is answered as far as this instrument can answer it:
not capacity, not a low-rank continuous coordinate, but the METRIC —
common-community mass. The natural next step is a metric sweep rather than
a capacity sweep: interpolate between Hellinger and PPMI weighting (e.g. a
family of frequency-exponent reweightings) and find where the coupling
appears. That would name the carrier in one number instead of bounding it.

### Planner adjudication of SR4 (2026-08-17, appended after the run) — METRIC-BORNE, CAPACITY PROVABLY EXHAUSTED; THE MEASUREMENT PHASE CLOSES

**METRIC_BORNE accepted, with the strongest possible form of its
premise: capacity is exhausted BY PROOF** (full-rank PPMI ≡
unfactorised PPMI to 5.3e-15 — the rotation row did its job;
RANK_ANOMALY correctly unreachable; fold 4's rank 1044 controlled
by the never-truncating row). PPMI at ANY rank reaches 56.3% of
the Hellinger flat coupling and detects at no rank; **87.6% of the
coupling gap and 61.2% of the identity gap are METRIC, not
capacity.**

**Dated correction to the T3 adjudication (owed to the executor's
flag):** T3's GAP_REMAINS residual was typed
"representation-capacity". That typing DOES NOT SURVIVE the
completed axis — 61.2% of the gap persists at zero truncation. The
residual re-types as predominantly METRIC: structure the PPMI
weighting suppresses and the Hellinger geometry keeps.

**Defect #75 (RD-SR4-1, mine):** the pattern table keyed on binary
detection at a 0.05 boundary this corpus sits on (0.052–0.056 in
three rows). The verdict is unaffected — the capacity argument is
threshold-free (56.3% of flat at FULL rank) — but **convention:
classification cells key on effect-size contrasts with detection
secondary, or declare a near-band zone** (#74 note: the endpoint
convention now also anticipates non-uniform rank across folds).

**The T-line's final synthesis proposition (adjudicated wording,
carried to the synthesis doc and the 4W map):**

> **Identity lives in the distinctive; personality lives in the
> common.** The re-identification signal is carried by a low-rank,
> PPMI-friendly taste coordinate that generalizes to unseen
> communities (T2/T3); the Big5 coupling is carried by the
> Hellinger-kept COMMON-COMMUNITY MASS — how a person distributes
> over widely-shared opportunities — which the
> distinctiveness-emphasizing metric suppresses (SR2/SR3/SR4).
> Mechanically sensible: dimensional population-graded constructs
> should live in shared-opportunity allocation; individual identity
> in rare distinctive choices. EXPLORATORY, corpus-level, one
> cohort; no person claims.

**THE MEASUREMENT PHASE OF THE T-LINE CLOSES.** Seven legs under
the loop discipline (T2, T3, SR2, SR3, SR4 + the adopted T1 and
the 4W synthesis), defects #68–#75 each buying a convention, suite
970 → 1021, every leg label-governed with zero ID leaks. Synthesis:
`docs/SUICA_M4_T_LINE_SYNTHESIS.md`. Named, not queued: the
When-dynamics program (the 4W doc's §2.4, unposed); second-corpus
transport of the branch code; common-mass carrier refinement
(which shared communities, at cluster level — naming governance
applies); R3's identity-formalism reconciliation (held); the
owner's parked decisions.
