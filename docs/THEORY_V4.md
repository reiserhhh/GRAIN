# GRAIN Rind Theory Base v4 (皮モデル第4版 — 統合改訂)

Created: 2026-07-07. Supersedes V3+its section-7 amendments as the READING
copy; V3 remains unchanged as the historical record (every claim's audit
trail lives in CLAIMS_LEDGER). v4 consolidates: the C2-compositeness
correction (no-lock program), the F1-F6 formal core, the flesh-purity gate,
the three-layer person model (OP-32), and the linguistic grounding
(LINGUISTIC_FOUNDATIONS). Nothing here exceeds its ledger status.

## 1. The model

For author u, slice i, rind (venue/topic shell) r(i):

```text
y_ui = f_u + b_{r(i)} + gamma_{u,r(i)} + e_ui      (style features)
r(i) ~ pi_u                                        (rind CHOICE — person-informative)
```

**Corrected channel table** (the v3 table mislabeled C2; per-construct
family assignments from results/suica_c2_purity_all19_v1):

| Channel | Definition | Estimand that isolates it | Status |
|---|---|---|---|
| C1 choice | pi_u vs population (12-class profile, log-ratio axes) | choice shares/axes directly | T3 held-out (OP-6a 5/5, shrinkage 0.027); **T4: politics axis -> Openness r=+0.096 (opening #1)** |
| C2 flesh | f_u — venue-invariant style base | CLASS-DISJOINT disjoint-occasion retest (licensed by W-B2c); NOT the raw mean | raw mean = C1+C2 COMPOSITE (F1/F2); flesh share is per-construct, measured; **T4: first_person -> Neuroticism r=+0.111 (opening #1) — the flesh-dominant flagship** |
| C3 react | structure of gamma_{u,r} (if-then signatures; CAPS lineage) | matched same-vs-stranger congruence on pair-shared classes | provisional (E6 revived, 2 constructs, n=195) |

Uncentered user means remain the DEPLOYMENT scores (they carry the most
person-stable signal); the correction is about ATTRIBUTION: which channel
the signal belongs to, and which constructs may be called traits.

## 2. The three-layer person model (v4 reporting architecture)

A person is reported in three layers, each with its own estimator family
and literature home (LINGUISTIC_FOUNDATIONS sections 1/3):

1. **位置 POSITION** — scores on the nomothetic construct families
   (F-family flesh traits; C-family venue signatures; composites reported
   two-channel). Population- and register-relative (Rulebook D1; OP-7a
   per-register norms for venue-coupled constructs).
2. **配置 CONFIGURATION** — if-then react signatures (C3) and choice
   profile shape: how the person's behavior is ARRANGED across situations
   (Mischel-Shoda CAPS lineage). Provisional tier.
3. **残差 RESIDUAL (個性)** — what remains after subtracting the group:
   the distinctive deviation profile (Furr normative/distinctive
   decomposition). Pilot measurement (OP-32-P, T1): distinctive-signature
   stability median r = 0.505 across >= 90-day-gap halves vs stranger null
   -0.018 (94.6% of users above the null band); distinctiveness magnitude
   itself moderately stable (r = 0.353). Complementary bound: ~32% of
   embedding-identifiable person-stable signal lies OUTSIDE the current
   battery (OP-9 M3) while identification parity holds (AUC 0.887/0.891) —
   the person is CODABLE without being exhaustively factorable
   (stylome/idiolect literature). The fingerprint thesis, operationalized:
   uniqueness = identification metrics; individuality = stable residual.

## 3. Design principles (each traceable to a falsification)

1. **Rind variance is controlled by DESIGN, not statistics.** Formal core:
   centering is mediator adjustment; it removes retest covariance by
   exactly Var(m) + 2Cov(f,m) at first order under pi ⊥ b, and MORE under
   flesh-choice coupling (population-composition bias — measured, OP-30
   open for the exact term). Boundary condition: exogenous assignment
   (fixed-prompt regimes) makes centering harmless. (F4; P2/E1 empirical;
   W-PHASE.)
2. **Flesh-purity gate (v4 battery admission; provisional thresholds).**
   A construct enters the F-family iff class-disjoint retest >= 0.15 AND
   choice-mediated share < 0.30. Rationale: the raw mean conflates
   channels (F1/F2); the gate prevents venue signatures from masquerading
   as traits (the opening-#1 H3 failure mode).
3. **Composite prediction over single edges.** Dev-measured frozen
   composites are the primary external predictors (opening-#1 lesson:
   literature-prior single edges underperform dev-measured aggregates).
4. **Population/register-relative measurement (D1).** Reliability and
   norms never transport as constants across populations, registers, or
   languages (PRED-1b amendment; OP-7a level shifts; JP protocol T3).
   **Cross-regime corollary (PRED-4, 2026-07-07, FAIL recorded)**: document
   type (free/assigned/domain-locked) is NOT controllable as a covariate —
   F4 bars adjustment whenever type is self-selected, and the tested
   statistical shortcut (restricting to FORM coordinates, hoping their
   between-person structure is regime-invariant) FAILED: CONTENT structure
   transports better in 2/3 corpus pairs (pandora-essays 0.75 vs 0.60;
   essays-x 0.74 vs 0.52). The within-corpus flesh/FORM finding and this
   between-corpus result are different objects (stability decomposition vs
   covariance geometry) — both stand. Licensed paths for cross-regime
   comparability: within-person multi-regime DESIGN (the 2-arm instrument;
   OP-35) plus regime-relative norms; nothing statistical substitutes.
5. **Estimator honesty by wrong-world licensing.** Every estimand used for
   a claim must pass its violation-world test (the licensing map lives in
   FALSIFIER_MATRIX; suites replicate frozen conventions exactly).

## 4. Formal core (proofs in THEORY_FORMAL_NOTES)

- F1/F2: raw mean = flesh + venue-mix; shared-set retest cannot purify.
- F3: disjoint sets kill the mix-mix term under (b independent, pi ⊥ b);
  cross terms and coupling residuals survive -> upper-bound reading.
- F4: centering = mediator adjustment (theorem + boundary + second order).
- F5: precision penalty H(M) is rind-loading-weighted (post-hoc v3.2
  amendment; prospectively testable).
- F6: overidentifying constraints — SB volume consistency; transport bound
  cross <= sqrt(w1 w2); estimator-agreement alarm (fired for first_person
  and directive -> shares reported as bands).

## 5. Falsification record (what died, what survived)

DIED (retired, with the killing evidence): condition-mean centering (3
independent ways + F4 theorem); affect word-rates as traits or occasion
states (E5; tension undetermined in purity too); attention weights as
measurement evidence (G11); novelty-as-style-trait (choice share 0.587,
class-disjoint 0.033 -> C-family); adjective pole enrichments as licensed
bridges (E11 v1 -> v2 -> fitted-null chain).

SURVIVED at their tiers: choice axes (T3+T4 for politics->O);
first_person flesh trait (T4 for ->N); register transport 19/19 (T2,
round-8 hardened); react signatures (provisional); distinctive residual
stability (T1 pilot); the estimator layer (P0/P0-B + 9-world suite,
2 adjudicated FAILs recorded as bounds, not hidden).

## 6. Measurement architecture v4

- **Scorer v4**: disjoint anchor lexicons (250 -> 227 words, 20 logged
  moves, battery-priority policy). The 5 frozen construct scores are
  BIT-IDENTICAL to v3 (selftest S2 max diff 0.0) — the validated chain
  carries over; what changes is the wider 23-anchor space (OP-23 fix:
  no manufactured between-category covariance). E7-family analyses should
  be re-run under v4 rates before any new state claim.
- **Families** (round-10 corrected, transported fit): F-family = {first_person,
  wcl_60 apostrophe-omission (strongest flesh, 0.365 class-disjoint), wcl_13
  negation, wcl_23 profanity} — **flesh lives in FORM** (mechanical/register
  habits, the idiolect literature's person-pure markers); C-family (8) =
  choice axes + content-vocabulary constructs — **rind carries CONTENT**;
  composites (6, two-channel reporting); assignment table in
  results/suica_c2_purity_all19_v1 and the registry. Empirical division of
  labor on Essays transport: the 4 FORM constructs alone carry N/C/A at
  near-full-battery level (0.150/0.119/0.117) while E requires the
  content/choice side (F_ONLY E = -0.001). Standing rule from the round-10
  defect: cluster/embedding-derived constructs are BOUND TO THEIR FIT —
  transport, never refit, when reusing a construct ID.
- **Residual layer**: OP-32 distinctive-profile machinery with stranger
  nulls; reported as layer 3.
- **JP branch**: scorer v1 at stage 1/3 (machinery-grade; するな lookahead
  fixed per round-8; stages 2-3 open).
- **Growth direction**: co-selection construct family (OP-33) to chase the
  ~32% unfactored share, under the same A/B + gate discipline.

## 7. External validity state and the remaining budget

Opening #1 (2026-07-07, T4): preregistered rule FAILED 2/7 — recorded in
full; H2 (first_person->N +0.111) and H6 (politics->O +0.096) confirmed;
no re-analysis. Opening #2 (the last) + untouched Essays confirm-half are
reserved for a v4 composite prereg: frozen dev-trained composites over the
F/C families, powered >= 90%, sealed publicly before any label contact.

## 8. Linguistic grounding

Component-by-component literature map in SUICA_LINGUISTIC_FOUNDATIONS_V1
(function-word stylistics; open-vocabulary LBA; Biber MD register theory;
CAPS; whole-trait density distributions; non-ergodicity; usage-based
individual grammars; stylome/idiolect; Furr distinctiveness). Honest gaps:
production mechanisms (constructs are behavioral residues, not mechanism
claims); dictionary psychometrics (OP-19); co-selection features (OP-33).
