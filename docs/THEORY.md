# GRAIN Rind Theory Base v3 (皮モデル)

Created: 2026-07-05
Status: predictions committed BEFORE the regime test run; results section appended after.

## 1. The model

GRAIN treats a body of spontaneous text as a watermelon (suika): a person-stable
interior ("flesh") wrapped in theme/situation shells ("rind", 皮). The 2026-07
validation cycle established empirically what the rind is NOT: it is not a
nuisance that can be subtracted away. Statistical condition-centering was
falsified in every tested form (naive, partial-lambda, two-way-FE out-of-sample;
see CLAIMS_LEDGER P2/E1). The rind carries person signal because people choose
their rinds.

Formal object. For author u, slice i, rind r(i):

```text
y_ui = flesh_u + rind_{r(i)} + gamma_{u,r(i)} + e_ui        (style features)
r(i) ~ P(r | u)                                             (rind choice)
```

Three information channels, with measurability governed by the RIND REGIME:

| Channel | Definition | Free rind (Reddit) | Domain-fixed rind (X market) | Experiment-fixed rind (Essays prompt) |
|---|---|---|---|---|
| C1 choice | P(r \| u) vs population | rich (audited: axes retest 0.51-0.70, AUC 0.839/0.909; held-out confirmation in OP-6a) | micro only (within-domain: symbols, query groups) | zero by design (recoverable via the Rulebook A3 choice module) |
| C2 style-base | E[y_u] uncentered | attenuated by rind mixing | intermediate | maximal precision (within-population claim; see PRED-1b caveat) |
| C3 react | structure of gamma_{u,r} | **provisional**: person-specific signatures confirmed for 2 constructs under the stranger null (E6: first_person +0.15, directive +0.15); pending held-out replication before promotion | untested | design-dependent (future) |

C3 status note (2026-07-05 consistency pass): earlier versions of this table
said "retired"; the audited E6 matched-estimator result revives C3 narrowly
(2 style constructs, n=195 deep users, CI lower bounds 0.007/0.044). Treat as
provisional until replicated on users outside the deep-extraction cohort.

## 2. The design principle (the core theoretical commitment)

**Rind variance is controlled by DESIGN, not by STATISTICS.**

- Statistical control (centering) removes flesh together with rind because
  P(r|u) is informative — proven three independent ways (MCD benchmark 2026-06,
  P2 2026-07-04, E1 cross-fit audit 2026-07-05).
- Design control (fixing the rind experimentally, or sampling within one
  domain) removes rind variance from the observations themselves, leaving C2
  clean, at the price of silencing C1.

Consequence for instrument construction: a fixed-rind experiment yields a
rind-specialized scale (皮特化尺度) whose C2 precision-per-token exceeds
free-rind sampling; a free-rind sample yields C1, which no fixed-rind design
can provide. A complete GRAIN instrument therefore has two phases:
free-collection (C1) + fixed-prompt battery (C2), which is the corrected,
defensible form of the original "scale + projective hybrid" ambition.

## 3. Precision formula (what fixing the rind buys)

For a style construct with slice noise sigma2_e, between-rind variance
sigma2_r, and n slices spread over rind mix M:

```text
SE^2(style_base_u) ~ ( sigma2_e + H(M) * sigma2_r_effective ) / n
```

where H(M) grows with rind heterogeneity of the user's sample. Fixed rind sets
H ~ 0. Reliability-per-token is therefore predicted to ORDER by regime:
experiment-fixed >= domain-fixed >= free, for the same n.

## 4. Pre-committed regime predictions (test: run_suica_rind_regime_test_v3.py)

- **PRED-1 (within-corpus, cleanest)**: for the same PANDORA users, same slice
  counts, split-half reliability of style bases is higher when all slices come
  from ONE subreddit (rind fixed by selection) than when slices span >= 4
  subreddits (rind mixed). Pass rule: paired delta > 0 with 95% bootstrap CI
  above 0 for a majority of the 4 live style constructs.
- **PRED-1b (cross-corpus ordering, with occasion caveat)**: at a matched
  ~1,000-token budget, split-half reliability orders Essays >= X >= PANDORA-mixed.
  Reported descriptively; Essays confounds rind-fixing with single-occasion
  writing, so this is supporting evidence only.
- **PRED-2 (micro-choice)**: in the domain-fixed regime (X), within-domain
  choice (symbol distribution) is person-stable: same-user early/late AUC vs
  random pairs >= 0.65 (the migration-manual criterion).
- **PRED-3 (same flesh, different rinds)**: the inter-feature correlation
  structure of the 23 anchor-rate features (user level) is invariant across
  regimes: correlation of off-diagonal vectors >= 0.70 for every corpus pair.

Falsification consequences, committed now:
- PRED-1 fail -> the precision formula is wrong; fixed-rind experiments lose
  their principal justification; theory reverts to C1-only.
- PRED-2 fail -> C1 does not miniaturize; domain-fixed deployments (e.g., the
  planned X/market agent line) get style-base only.
- PRED-3 fail -> "flesh" is rind-specific; no universal base formula; GRAIN
  becomes a family of per-rind instruments with no cross-rind claims.

## 5. Governance for this test

- Essays users are split 50/50 by stable hash; only the dev half's TEXT is
  used here; Big5 label columns are never loaded; the confirm half stays
  untouched for P5. X is unlabeled (Tier X).
- v3 scoring formulas are frozen; nothing downstream of this test may alter
  them without triggering re-validation.

## 6. Results (2026-07-05 run; audit status in CLAIMS_LEDGER)

- **PRED-1: PASS 3/4 after audit** (n=918 paired users, same slice counts;
  round-3 audit applied parity, ordering, density, and time-span controls):
  span-matched deltas — first_person +0.063 [0.029,0.098], directive +0.165
  [0.074,0.260], novelty +0.182 [0.047,0.294]. The most rind-driven construct
  (novelty) gains most — consistent with the channel model. **tension's
  apparent gain was a temporal-clustering artifact** (the fixed arm was a
  chronological prefix; span-matched delta -0.033) — affect-state constructs
  do not benefit, consistent with tension being state-like, not trait-like,
  in text. **The design principle (control rind by design, not statistics) is
  empirically supported in its within-population form for style/stance
  constructs.** Any future run must span-match the arms.
- **PRED-1b: ordering NOT confirmed as stated.** At matched 6-slice budget:
  X (domain-fixed) is best or near-best everywhere (novelty 0.643, tension
  0.357); Essays (experiment-fixed prompt) is only ~equal to PANDORA-mixed.
  Amendment adopted into the formula: split-half reliability is
  POPULATION-RELATIVE — between-person variance differs across populations
  (Essays = homogeneous students; X/Reddit = heterogeneous internet users), so
  cross-population reliability comparisons do not test the rind principle.
  The principle holds within-population (PRED-1); planned fixed-rind
  experiments should therefore expect gains relative to free-rind sampling of
  the SAME population, not absolute reliability guarantees.
- **PRED-2: PASS with caveat** — X symbol-choice AUC 0.810 (criterion 0.65),
  but n=31 users only; needs a larger domain-fixed corpus before promotion.
- **PRED-3: PARTIAL (audit-corrected)** — the first run accidentally included
  a 24th, algebraically derived feature; on the true 23 lexicon rates the
  structure correlations are pandora-essays 0.620, essays-x 0.572, pandora-x
  0.530 (sparse-feature-robustness variant: pandora-x 0.494), and the X leg
  is weak (89.5% single-slice users). The flesh structure transports
  approximately, not exactly: cross-rind deployments need per-rind
  calibration (which is precisely the planned rind-specialized scale
  (皮特化尺度) route). Only the pandora-essays pair (0.620) is currently
  solid evidence.

### Formula v3.1 (amended)

```text
SE^2(style_base_u) ~ ( sigma2_e(pop) + H(M) * sigma2_rind + [occ structure] ) / n
reliability = f(design, population)  — never a design-only constant
C1 available iff rind (or micro-rind) is self-selected; C1 miniaturizes to
within-domain choice (PRED-2) under domain fixing.
```

## 7. v3.2 corrections (2026-07-07 no-lock falsification program;
## FORMAL_NOTES F1-F6 + wrong-world suite + C2 purity decomposition)

**7.1 The C2 row of section 1 was MISLABELED — corrected.** By the model's
own equation, the uncentered mean is ybar_u = flesh_u + m_u (+ noise), with
m_u = sum_r P(r|u) b_r the venue-mix term, person-stable exactly to the
extent choice is (FORMAL_NOTES F1/F2; wrong-world W-B: flesh-null world
shows raw retest 0.637 from choice alone, final frozen run). "C2 style-base
(E[y_u] uncentered)" is therefore a **C1+C2 COMPOSITE channel**, and the
flesh/choice split is a per-construct empirical quantity. Measured on
Tier-U (suica_c2_purity_decomposition_v1; estimators licensed by
W-B/W-B2/W-B2c/W-B3 — the class-disjoint estimand by W-B2c, and
mediated_total as an UPPER bound on mediation under coupling per W-B3):

| construct | rho_raw | rho_class_disjoint | choice-mediated share [95% CI] | corrected reading |
|---|---|---|---|---|
| first_person | 0.683 | 0.268 | 0.279 [0.208, 0.349] (cond-share) | flesh-dominant style trait — flesh share is a BAND ~0.56-0.72 (three estimators: cond-share 0.72, class-share 0.56, mediated_total lower-bound 0.60; F6.3 alarm fires on their disagreement — see 7.5) |
| directive | 0.409 | 0.100 | 0.471 [0.226, 0.662] | C1+C2 composite (estimators disagree 0.23-0.48; wide CI) |
| novelty | 0.382 | 0.033 | 0.587 [0.428, 0.723] | **VENUE-SIGNATURE construct (C1-dominant); not a style trait** (all three estimators agree: 0.587/0.588 point-coincident) |
| tension | 0.316 | 0.053 | undetermined (shared-matched < .10) | consistent with no-trait verdict |
| adversity | 0.129 | 0.028 | undetermined (cov_raw ~0.0004 — unreadable noise) | dead construct, unchanged |

Coherence readings (round-9 corrected wording — "consistent with", not
"explains", since a choice-mediated construct can still predict O via the
choice channel, cf. H6): (a) consistent with the lockbox H3 failure
(novelty's "stability" was never flesh), (b) novelty's top PRED-1 gain is
the confirmed artifact-side of the same fact (its variance IS rind), (c)
consistent with novelty's E9a MTMM failure (convergent 0.258 < max
discriminant 0.325). And the one flesh-dominant construct (first_person) is
exactly the one that survived external confirmation (H2, T4) — the channel
accounting closes.

**7.2 Design rule added (v4 battery gate; PROVISIONAL).** A style construct
may be used as a TRAIT predictor only if it passes a flesh-purity gate:
class-disjoint retest >= 0.15 AND choice-mediated share < 0.30 (point
estimate). Constructs failing the gate belong to the choice family and must
be modeled there (novelty: relabeled; directive: composite — report both
channels). Round-9 disclosures: the thresholds are post-hoc (set 2026-07-07
after seeing the decomposition, not in the matrix); first_person passes the
share arm on the point estimate only (CI [0.208, 0.349] spans 0.30); the
class-disjoint estimand's wrong-world license (W-B2c) was implemented in
round 9, after the first decomposition run. The gate binds prospectively
for v4 construction; it is not a retro-fitted confirmatory claim.

**7.3 F4 (centering theorem) with second-order amendment.** Centering is
mediator adjustment: it removes retest covariance by exactly
Var(m) + 2Cov(f, m) at first order UNDER pi ⊥ b (final frozen phase-diagram
cells: exogenous +0.010 vs predicted 0; kappa=0 -0.316 vs -0.304). Under
STRONG flesh-choice coupling the observed removal EXCEEDS the first-order
formula (-1.583 vs -1.294 at kappa=0.7): cross-fitting removes
self-estimation but not population-composition bias — the round-9 audit
confirmed the mechanism directly (oracle-b centering matches the identity
exactly, -1.332 vs -1.335, while estimated-b centering over-removes;
bhat-on-b regression slope 1.403 at kappa=0.7 vs 1.002 at kappa=0). FE
centering over-removes flesh whose venue composition is predictable —
centering is even MORE destructive than the first-order theorem states.
The same mechanism upward-biases the mediated_total estimator (W-B3
recorded FAIL: 0.541 vs target 0.426) => under coupling it is an upper
bound on mediation. (W-PHASE and W-B3 both recorded as FAILs against their
pre-committed point tolerances; sign and boundary structure confirmed in
all cells; exact composition-bias term = open derivation, OP-30.)

**7.4 F5 refinement (H(M) is loading-weighted) — POST-HOC amendment,
trigger disclosed.** The matrix pre-committed "low-entropy > high-entropy
for each construct; violation => the v3.1 continuous form is wrong". The
trigger FIRED: 2/5 constructs violated (first_person 0.693 vs 0.700 —
hairline; adversity 0.164 vs 0.191 — but adversity's cov_raw ~0.0004 makes
its gradient unreadable noise, disclosed rather than dropped). The
loading-weighted reading — the entropy penalty applies in proportion to a
construct's rind-loading (holds for directive/tension/novelty; absent for
flesh-dominant first_person) — is therefore a POST-HOC refinement
consistent with the formula's H(M)*sigma2_rind structure, adopted as the
v3.2 form and subject to fresh-data test (the prospective harness, OP-31).

**7.5 F6.3 misspecification alarm — fired and recorded.** The three
mediation estimators must agree under the additive model. They agree for
novelty (0.587 cond-share / 0.588 mediated_total) but DISAGREE for
first_person (0.279 / 0.398 / class-share 0.442) and directive (0.471 /
0.229). Known contributors: mediated_total's upward composition bias
(W-B3) and the class-vs-condition mediation gap (W-B2c: +0.106 in-world).
Residual disagreement is the F6.3 alarm operating as designed — flesh
shares are therefore reported as BANDS, and T1 (additivity) remains an
open attack surface (OP-29).
