# GRAIN Falsifier Matrix v1 — no-lock theory falsification program

Created: 2026-07-07. Purpose: enumerate every proposition the theory commits
to, and for each one name the falsifier that could kill it WITHOUT spending
lockbox budget. An empty falsifier cell = an unfalsifiable (or not-yet-
falsifiable) proposition — the "holes" this program exists to surface.
Opening #2 stays sealed throughout (user directive 2026-07-07).

Falsifier classes (iteration speed):
- **AN** analytic proof / derivation (minutes-days; finds internal
  contradictions, identifiability failures, boundary conditions)
- **SW** synthetic wrong-world simulation (minutes; does the machinery
  DISTINGUISH the theory's world from worlds where the assumption is false?)
- **LW** LLM-agent world (hours; construct-logic tests with language realism;
  never human-validity evidence)
- **DS** dev-data structural test (days; label-free or Tier-D-orientation,
  unlimited use)
- **PR** prospective registration (weeks; predictions sealed before data
  exists — "time's lockbox", infinitely re-armable, label-free for
  structural claims)
- **LB** lockbox (finite; ONLY for external-anchor validity; budget 1)

## Proposition inventory (from THEORY_BASE_V3 + ledger-carried claims)

| ID | Proposition | Status pre-program | Falsifiers available | Coverage BEFORE this program | This program adds |
|----|-------------|--------------------|----------------------|------------------------------|-------------------|
| T1 | Additive decomposition y_ui = flesh_u + rind_r + gamma_ur + e_ui | assumed, never attacked | SW (interactive/multiplicative worlds), AN | **EMPTY** | SW-F (misspecification world; deferred to v2 — logged hole) |
| T2 | Rind choice P(r\|u) is person-informative (self-selection) | supported (choice AUC 0.839/0.909; OP-6a T3) | SW, DS, PR | DS done | SW W-A (estimator honesty under choice-null) |
| T3 | C1: choice profile is a stable person channel | T3-grade (OP-6a) | SW, DS, PR | DS done | SW W-A; PR design (sealed future-window predictions) |
| T4 | C2: uncentered style mean measures a person trait ("flesh") | P1 raw SB up to 0.856 | AN, SW, DS | **partial — KNOWN UNQUANTIFIED CAVEAT**: raw mean = flesh + choice-mediated rind mix under T1 itself; P2a caveat never decomposed | **AN (variance accounting) + SW W-B + DS decomposition (rho_raw vs rho_same-rind vs rho_cross-rind); headline C2 numbers get a C1-share estimate** |
| T5 | C3: gamma_ur carries person-specific signatures (provisional, 2 constructs) | E6 revived (n=195) | SW, DS | DS done (matched estimator, round-5) | SW W-D (norm-only world FPR of the matched estimator) |
| T6 | Centering removes flesh WITH rind BECAUSE P(r\|u) informative (mechanism) | empirical (P2/E1, 3 ways) | **AN (proof)**, SW | empirical only — mechanism asserted, not derived | **AN: mediator-adjustment proof + boundary conditions (exogenous assignment => centering harmless/helpful) + SW phase diagram validating the derived sign boundary** |
| T7 | Fixed rind leaves C2 clean at the price of C1 | PRED-1 3/4 (T2-grade) | SW, DS, PR | DS done | covered by W-B machinery (fixed-rind estimand reads flesh only) |
| T8 | Complete instrument = free collection (C1) + fixed battery (C2) | design consequence | — (derived) | derived from T6/T7 | inherits their falsifiers |
| T9 | Precision formula: SE^2 ~ (sigma2_e(pop) + H(M) sigma2_rind + occ)/n; reliability falls with rind heterogeneity within population | v3.1 amended after PRED-1b | AN, DS | 2-arm version only (PRED-1); **continuous form never tested** | **DS: H(M) gradient test — retest by rind-entropy stratum, volume-controlled; pre-committed direction: monotone decrease** |
| T10 | States live below month scale; state space != trait space | E5/E7 (round-4/5 audited) | SW, DS, PR | DS done | SW W-C (timescale aliasing: does month-cut machinery FABRICATE month states from week states?) |
| T11 | Leak mask removes the label-echo channel | mask extended; sufficiency asserted | SW (text-level), DS | **EMPTY (sufficiency never bounded)** | SW W-E: leak-only world; residual leakage bound after masking |
| T12 | Style battery transports across dialogue registers | OP-7a 19/19 (round-8 audited) | DS, PR, (LB for clinical) | DS done | — (PR design covers future) |
| T13 | Affect vocab is slice-level residual (~95-97%), not occasion state | E5 (MixedLM, round-4) | SW, DS | DS done | W-C overlaps (state detection floor by timescale) |
| T14 | G11: embedding direction scores licensed only via MTMM (person-stable space makes any fitted direction "stable") | E9/E11 + shuffled-fit null (2026-07-07) | SW, DS | closed this week | — |
| T15 | PRED-2: C1 miniaturizes to within-domain micro-choice under domain fixing | X AUC 0.810, n=31 (caveat) | DS (bigger corpus), PR | thin | PR design (X pipeline accrues; sealed windows) |
| T16 | PRED-3: flesh correlation structure transports approximately across rinds | 0.620 solid for one pair only | DS, PR | partial | logged: needs second corpus pair; no new falsifier this round |

## Empty/weak rows surfaced (the holes, in priority order)

1. **T4 (C2 compositeness) — the largest**: by the theory's own equation,
   E[y_u] = flesh_u + sum_r P(r\|u) rind_r (+ mean gamma). The second term is
   person-stable whenever choice is (T3!), so raw style-base stability
   OVERSTATES flesh. The P2a caveat said this in words; no number exists.
   Consequence if the choice-mediated share is large: C2 and C1 are not
   separate channels as operationalized; headline "style base" claims (P1 SB
   0.856; even T4-confirmed H2) are channel-composites, and the THEORY table
   in section 1 mislabels the C2 row.
2. **T6 (mechanism never derived)**: "centering removes flesh with rind" is
   an empirical regularity with a verbal story. Without the derivation, the
   boundary conditions (when WOULD centering be safe?) are guesses; Essays-
   regime deployments need exactly that boundary.
3. **T11 (mask sufficiency unbounded)**: the leak mask is a hard gate in
   every pipeline; its failure modes were never measured.
4. **T9 (continuous H(M) untested)**: the precision formula survived one
   2-arm contrast after one amendment; its functional claim (monotone in
   heterogeneity) is untested and could fail again.
5. **T1 (additivity unattacked)**: logged as a v2 item (SW-F interactive
   world) — deferred, recorded as an open hole, not silently skipped.
6. **T5/T2 estimator honesty (W-D/W-A)**: the matched estimator and choice
   AUC have never been run inside null worlds end-to-end.

## Pre-committed criteria for this round (fixed before any run)

- **W-A (choice-null world)**: venue assignment trend-driven, no person
  coupling. Choice-profile AUC must land in [0.45, 0.55] and axis retest
  |r| median < 0.10. Outside => choice machinery fabricates signal (T2/T3
  estimator defect).
- **W-B (flesh-null world)**: person style flesh_u = 0 for all u; venues have
  fixed style offsets; choice person-stable. EXPECTED under the theory:
  rho_raw substantially > 0 (choice-mediated pseudo-stability — this is the
  T4 hole made visible), while the SAME-RIND estimand must read <= 0.10.
  If same-rind also inflates => the proposed purified estimand is broken too.
- **W-B2 (mixed world)**: flesh and rind effects both present at calibrated
  shares; the decomposition estimators must recover the planted flesh share
  within +/-0.10 across a 3x3 parameter grid.
- **W-C (timescale aliasing)**: true states AR(1) at 1-week scale, none at
  month scale. Month-cell machinery must NOT report month-scale state share
  above its null (fabrication check); detection curve reported.
- **W-D (norm-only react world)**: gamma_ur = class norms only (no person
  specificity). E6-style matched estimator increments must stay below the
  stranger-null band in >= 95% of simulated cohorts.
- **W-E (leak-only world)**: person signal injected ONLY via
  PERSONALITY_LEAK_RE-matchable content in text templates. After masking,
  downstream trait recovery must be <= null 95th percentile; residual
  leakage rate reported (bound).
- **DS decomposition (real Tier-U)**: report per construct: rho_raw,
  rho_same_rind (same condition early vs late, condition present both
  halves), rho_cross_rind (disjoint condition sets — P2 estimand), and
  C1-mediated share = (rho_raw - rho_same_rind) / rho_raw (clamped at 0),
  with user-bootstrap CIs. NO pass/fail — this is measurement of a
  theoretical quantity; interpretation rule fixed: share > 0.30 for a
  construct => the THEORY table must relabel that construct's C2 row as
  "C1+C2 composite"; share < 0.10 => flesh reading stands.
- **DS H(M) gradient**: within volume-matched strata, Spearman correlation
  of user rind-entropy vs |retest residual|... operationalized as: retest r
  by entropy tercile (volume-matched); pre-committed direction: r(tercile 1)
  > r(tercile 3). Violation => formula v3.1 wrong in its continuous form.

## Revision note (2026-07-07, round 9 — the matrix as first written was
## corrected by the program it launched; changes dated, nothing silent)

1. **W-A criterion re-specified**: the original band "AUC in [0.45,0.55]"
   presumed a tie-free statistic. The frozen E3 convention is strict
   pairwise (ties = losses) and DEFLATES under mass ties (0.039 observed —
   conservative direction). Corrected criterion: mid-tie AUC in [0.45,0.55]
   AND frozen strict AUC <= 0.55, at two sparsity levels. (The first suite
   draft's tie-unsafe rank AUC fabricated 0.947 and was replaced by the
   exact frozen convention — suite-defect log in the script.)
2. **W-B expectation corrected by F2**: this matrix originally expected
   "SAME-RIND retest <= 0.10" in the flesh-null world. The F2 derivation
   (written AFTER this matrix) proves shared-set retest CANNOT purify flesh
   — the correct expectation (implemented) is: raw AND shared-set inflate;
   DISJOINT-set reads ~0. Final frozen run: 0.637 / 0.430 / -0.071.
3. **DS share formula replaced**: original (rho_raw - rho_same_rind)/
   rho_raw presumed the same purification error; the implemented rule is
   1 - rho_cond_disjoint / rho_shared_matched (volume-matched arms), same
   0.30/0.10 thresholds. Under the ORIGINAL formula tension/adversity would
   also "relabel" — driven by their near-zero shared-set correlations
   (noise), which the implemented shared>=0.10 guard correctly refuses to
   interpret. Discrepancy disclosed; the implemented rule is the licensed
   one (W-B/W-B2/W-B2c/W-B3).
4. **W-B2c implemented in round 9** (was referenced but missing — audit
   catch): class-correlated b world; class-disjoint arm reads Var(f) within
   0.115, condition-disjoint over-reads by +0.106 => the class-disjoint
   estimand (basis of the novelty relabel) now HAS a wrong-world license.
5. **T9 trigger fired**: H(M) direction violated for 2/5 constructs;
   consequence executed as THEORY_BASE 7.4 (post-hoc loading-weighted
   amendment, prospectively testable), not silently absorbed.
6. **Pre-commitment provability**: matrix/scripts/results were uncommitted
   during the program (mtime chain only: matrix 04:50 -> suite 04:53 ->
   notes 05:03 -> decomposition 05:05 -> results 05:07, single-write).
   Discipline adopted going forward: COMMIT the matrix/criteria before
   running (the commit at the end of this round pins everything).

## Honest scope

SW/LW falsifiers test IDENTIFIABILITY and ESTIMATOR HONESTY (can the
machinery see the difference between the theory being true and false); DS
tests structural claims on the dev population; none of them substitute for
external-anchor validity (LB, budget 1, sealed). LW execution is deferred
this round (design retained; DeepSeek/local models available) — deferral
logged, not hidden.
