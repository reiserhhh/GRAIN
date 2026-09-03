# GRAIN M4-M — The Level-Law Completion Line

Line opened 2026-08-11. Question: **can the restored level law
(`field ≈ λ′ − κ′·V_person`, K2f, interpolation-grade) be completed —
its readability exponent identified on a decollinearized corpus, its
predictions sealed at extrapolated configurations, and its tax
coefficient tested for unity across grades?** This line executes K2f's
named follow-up ("a decollinearized corpus for the r-at-level question;
a D1-style seal for the new form") and appendix K's unification
candidate (the κ-invariant).

Tier: EXPLORATORY, label-free, synthetic throughout. The claims ledger
controls. Registrations are appended here BEFORE execution and never
edited after; outcomes are appended after adjudication. Route-index
entry at line synthesis.

## Standing rules (binding on every M-leg)

Rules 1–8 verbatim as in the K-line header
(`docs/SUICA_M4_K_IDENTITY_LINE_PLAN.md:13-30`); rules 9–21 as
canonically tabled in `docs/SUICA_DEFECT_REGISTRY.md` ("The rule set
the defects bought"); rules 22–24 as enacted in the D5 adjudication
(`docs/SUICA_DEFENSE_PHASE_PLAN.md:667-672`): 22 — every band,
tolerance or bound declares its sides; 23 — clauses declare they can do
their job (gate stages named, feasibility argued); 24 — rule 8 extends
to published prose, and report tables carrying artifact numbers are
GENERATED from artifacts, never typed (K2f adjudication convention).
The registry and the originating adjudication texts control over this
shorthand. Unnumbered conventions in force
(`docs/SUICA_DEFECT_REGISTRY.md` "Conventions in force"): round-trip
parsing, df-aware pilots, Part-0 bit-identity verification, chunked
foreground stages, aggregation provenance, salt embedding,
legacy-anchor parser naming; plus the K2f machinery convention — ONE
importlib loader chain per leg (RN-K2F-5).

Execution conventions (binding on dispatched agents): no background
jobs, no monitors — foreground chunked stages with explicit timeouts,
every stage under 600 s; Part 0 gates written into the report BEFORE
any arm runs; bit-exact anchor reproduction from persisted artifacts
before new arms; every anomaly self-reported with an explicit statement
of whether it was resolved before or after any hypothesis-relevant
number existed; exactly ONE commit per leg (`feat(m4-m): ...`), never
amended, never pushed by the agent; `results/` is gitignored —
committed deliverables are the script, the report, the plan-doc outcome
append, and the ledger row. `suica_core/` is frozen and READ-ONLY.

## Line charter (three legs, registered sequentially)

- **M1 — r-at-level on a decollinearized factorial** (registered below).
  The φ knob moves predicted attenuation at EXACTLY fixed V_person
  (`person_share_design` has no φ argument), so a share × φ factorial
  breaks the −0.964 collinearity that made K2f's exponent
  unidentifiable. Was the level law's flatness in r real, or an
  artifact of a corpus that never varied φ below 0.90?
- **M2 — the extrapolation seal** (register ONLY after M1 adjudication;
  blocked if M1 lands NON_IDENTIFIED_UNDERPOWERED). D1-style salted
  prospective seal of the M1-winner's predictions at ≥3 configurations
  OUTSIDE the trained envelope (share above 0.6634207990183637 and/or φ
  outside M1's ladder, with at least one joint-exterior corner), bands
  pre-declared from M1's LOO-RMSE, opened measure-first. 3/3 inside →
  the level law is graded PREDICTIVE (extrapolation); partial → the
  boundary of validity is named.
- **M3 — the one-κ question** (register only after M2 adjudication).
  Is the variance-tax coefficient ONE instrument constant? κ has
  appeared at ~0.72–0.75 in three independent fitting routes plus the
  level refit; M3 measures κ at response grade and level grade across
  new geometries within the frozen instrument's domain, with any
  cross-scale comparison carrying a pinned link (rule 14) or re-designed
  within-instrument. Verdict: common band (an instrument constant, the
  appendix-K species) vs a measured geometry-dependence law.

Line stop conditions: any leg's STOP cell routes back to the planner as
a registration defect before any hypothesis-relevant number exists; the
line closes early if M1 lands R_TERM_ABSENT_AT_LEVEL and M2 seals the
tax-only form (M3 still runs — κ unity is grade-independent).

---

## M4-M1 — r-at-level on a decollinearized factorial

**REGISTERED 2026-08-11, BEFORE RUN.** Planner: this document's author.
Executor: dispatched agent (implementation and execution only; this
registration text is binding).

### Question

K2f restored the level law as an intercept minus the tax — and proved,
in the same breath, that this corpus cannot say whether the intercept
hides λ·r^q: corr(r, V) = −0.9643543785903034 across the 26 rows,
because the legacy legs exercised φ only at {0.90, 0.98} and moved r
almost entirely through share, which also moves V. M1 asks the question
K2f could not: **at level, with V held fixed by design, does the field
depend on readability — and with what exponent?**

### Design mechanism (why this decollinearizes — verified against artifacts 2026-08-11)

- `person_share_design(share, int_share)` has NO φ argument:
  V(0.40) = 0.12000000000000004, V(0.45) = 0.13500000000000004
  (persisted in `results/m4_k2f_level_law/part0.json:fresh_arm` and
  K2f's ANCHORS). φ therefore moves r at EXACTLY fixed V.
- φ's r-leverage is live in the published corpus:
  r(0.30, 0.90) = 0.6758917867864564 vs r(0.30, 0.98) =
  0.645057248597175; r(0.50, 0.90) = 0.558364277337817 vs
  r(0.50, 0.98) = 0.5193517935368367
  (`results/m4_k2f_level_law/compiled_rows.csv`, round-trip). Higher φ
  → lower r; the ladder below extends φ DOWNWARD to raise r at fixed V.
- Share moves V (and r): the trained share envelope is
  [0.02, 0.6634207990183637]; M1's shares sit strictly inside it.

### Facts cited to motivate leans (rule 8 — verified 2026-08-11 against persisted artifacts at full precision)

- K2f winner F2: λ′ = 0.18021628978547316, q′ = −0.009622064624441264,
  κ′ = 0.750086268225045, p = 0.2064406330042716, LOO-RMSE =
  0.0061559195350209; q′ ci95 [−0.3792124136721057, 0.5313115708778163];
  κ′ ci95 [0.5202855978239498, 0.8612166024267973]
  (`results/m4_k2f_level_law/fits.json`).
- Sealed response-grade constants (D1 bundle, opened in D-open):
  λ = 0.17417497661611914, q = 1.8528700746510731,
  κ_hat = −0.7220359963712748. The sealed composite under-predicted all
  26 compiled levels (residuals −0.1611..−0.0597) — evidence that the
  level r-dependence is WEAKER than r^1.85.
- Response-grade q band: **q = 1.83 [1.71, 1.98]**
  (`docs/SUICA_IDENTITY_THEORY_V1.md:805,841`); cross-leg spread
  1.83–1.93 (fragility annex) is a SECOND reading, not the primary band.
- D-open S-4 anchor: r(0.40, 0.90) = 0.6185853753498524,
  measured b-only level 0.09350089316336324 (K2f script ANCHORS,
  bit-exact against `results/dopen_seal_opening/stage1_m4.json`).

### Machinery (rules 9/12 — source objects pinned)

Instrument: K2b's world family and field reader, loaded through K2f's
loader pattern with ONE loader chain (RN-K2F-5). Source objects:
`run_suica_m4_k2b_t4_branch.py` — `build_k2b_world` / `run_field_world`
(985-author K1-pinned panel, F2 m-multiset, 4 contexts);
`run_suica_m4_k2c_matched_pairs.py:predicted_attenuation` (2-arg, →
k2b:533-583); `run_suica_m4_k2e_double_matching.py:person_share_design`.
Arm configuration inherits K2F-FRESH's carrier verbatim (int_share = 0,
same w_int_arm, same panel pins) except (share, φ). Field statistic:
per-world `recovery_b_only`; cell level = mean over worlds (the
`_level_from_raw` aggregation, K2f lineage). New script:
`scripts/run_suica_m4_m1_r_at_level.py`. Seeds: master_seed 20260811;
main worlds salt `m4m1-world`, hash-derived per (cell, world-index
0..31); pilot salt `m4m1-pilot`, indices 0..3 — disjoint streams by
salt. Round-trip parsing everywhere.

### Design (pinned)

Factorial grid: share ∈ {0.10, 0.25, 0.40, 0.60} × φ ∈ {0.60, 0.70,
0.80, 0.90, 0.98} = 20 cells × 32 worlds = 640 fresh worlds. Shares
strictly inside the trained envelope; φ at 0.60–0.80 is an EXTENSION
beyond the exercised {0.90, 0.98} — disclosed as such, guarded by the
pilot (finiteness, non-degeneracy, liveness), and the law claim stays
scoped to the tested grid.

### Part 0 (ALL written into the report before any world runs)

- **G0m (anchors, bit-exact).** (i) predicted_attenuation(0.40, 0.90)
  == 0.6185853753498524; (ii) predicted_attenuation(0.45, 0.90) ==
  0.5889058864943755 and person_share_design(0.45, 0.0) ==
  0.13500000000000004 (K2f part0 fresh_arm); (iii) person_share_design
  (0.40, 0.0) == 0.12000000000000004; (iv) reload
  `results/m4_k2f_level_law/fits.json` and verify EVERY K2f number
  quoted in this registration bit-exactly (F2 params, LOO, both ci95s)
  — a mismatch is a planner citation defect: STOP, report, do not
  repair silently; (v) re-derive Dopen:M-4's level 0.09350089316336324
  from its raw CSV round-trip; (vi) grep the theory doc and verify the
  quoted response band [1.71, 1.98] appears verbatim — mismatch: STOP
  (rule 24).
- **G1m (design arithmetic — no worlds needed).** Compute the realized
  20-point (r, V) table from the pinned deterministic maps and write it
  into the report. Gates: (a) all shares inside
  [0.02, 0.6634207990183637]; (b) V max/min ≥ 2, with V derived from
  `person_share_design`, never assumed linear; (c) within-share r
  max/min ≥ 1.20 in at least TWO share levels; (d) cross-cell
  |corr(r, V)| ≤ 0.30, with corr(r^1.8528700746510731, V) also
  reported (the headline decollinearization numbers, against K2f's
  −0.9643543785903034); (e) no duplicate (r, V) design points.
  Fallback ladder (rule 17, pre-declared, arithmetic only): if (c) or
  (d) fails, extend φ once to {0.45, 0.60, 0.75, 0.90, 0.98} and
  recheck; still failing → **STOP_DESIGN_INFEASIBLE** (planner
  registration defect).
- **G2m (pilot, rule 17 — AFTER G0m/G1m, before main arms).** 4 corner
  cells (share ∈ {0.10, 0.60} × φ ∈ {0.60, 0.98}) × 4 worlds on the
  pilot salt. Checks: (i) all per-world fields finite, non-saturated;
  (ii) **liveness (rule 3)**: at share 0.60, the realized attenuation
  contrast between φ = 0.60 and φ = 0.98 exceeds 2× its pooled SE —
  using the k2b-side realized card-attenuation statistic if one is
  persisted per world (executor pins the source object file:line in
  Part 0; rule 12), else the pilot field contrast at the same corner
  (declared fallback: a liveness check on the outcome, catastrophic
  deadness only). Liveness failure → drop φ extension, fall back to
  exercised-range ladder {0.90, 0.92, 0.94, 0.96, 0.98}, re-run G1m +
  G3m-b; failure there → STOP_DESIGN_INFEASIBLE.
- **G3m (satisfiability/power, rules 11/18/21/23).**
  (a) Sides for every clause (rule 22) written in Part 0.
  (b) **Projected identification power** (the rule-11 check made real):
  σ_w = pooled per-world sd across the 16 pilot worlds, inflated by the
  df-aware factor sqrt(12 / χ²_{0.10, df=12}); then a parametric
  replication (B_proj = 500, seed = master) of the 20-cell experiment
  at truths (λ = 0.18021628978547316, κ = 0.750086268225045, ε = 0)
  × q_truth ∈ {1.0, 1.8528700746510731}, cell noise
  N(0, σ_w²/32), fitting form F1 with the full start grid; width proxy
  = quantile(q̂, .975) − quantile(q̂, .025). Gate: proxy ≤ 0.50 under
  BOTH truths (stricter than L-1's 0.60 to absorb proxy slack,
  disclosed). No projection at q_truth = 0 — structural
  non-identification there is cell R_TERM_ABSENT's subject, not a
  power failure (stated per rule 20's spirit: that branch is
  pre-declared adjudicable, not empty).
  (c) Stage estimates: part0 60 s, pilot 30 s, worlds 4 × 120 s
  (5 cells per chunk), fit 300 s, finalize 60 s; the 2×
  stop-and-report threshold convention applies.
- **G4m (hygiene).** Rule-16 truth table written verbatim into the
  report (below); rule 24: every report table carrying artifact
  numbers is generated from artifacts.

### Fits (pinned)

Fit on the 20 cell means (world-level fitting is minimizer-identical at
equal cell n; noted, not run). FOUR pre-declared forms, no others:

- F1: field = λ·r^q − κ·V (K2f lineage);
- F1e: field = λ·r^q − κ·V − ε, ε bounded [0, 0.05] (T4's nonnegative
  species floor; 0.05 = 3.3× the fragile ≤0.015 band ceiling; the ONLY
  bounded form, stated); near q ≈ 0 its (λ, ε) ridge is singular —
  disclosed; LOO pays for it;
- F2: field = λ·r^q − κ·V·r^p (K2f lineage);
- F3: field = (λ − κ·V)·r^q (K2f lineage).

Optimizer: K2f's OPT verbatim (scipy least_squares, trf, 2-point jac,
ftol/xtol/gtol 1e-14, max_nfev 20000, unbounded except F1e's ε).
Start grid (pinned; K2f's grid extended by q ∈ {−0.5, 0.0} on K2f's
own finding): λ ∈ {0.05, 0.17417497661611914, 0.5}; q ∈ {−0.5, 0.0,
0.5, 1.0, 1.8528700746510731, 3.0}; κ ∈ {0.0, 0.7220359963712748,
2.0}; p ∈ {0.0, 1.0, 1.8528700746510731}; ε ∈ {0.0, 0.01, 0.03}.
Selection by leave-one-CELL-out RMSE (20 refits per form, full grid +
full-data optimum). Parameter CIs: within-cell world-block bootstrap —
each draw resamples 32 world indices with replacement INDEPENDENTLY
per cell, recomputes the 20 cell means, refits from the full-data
optimum start; B = 2000, seed = master; K2f discard rules
(non-convergence, |param| ≥ 1e6), counts disclosed. Cells are never
dropped (the design is fixed; the uncertainty is world sampling within
cells — stated against K2f's row-resample, which faced a different
object). Rule 13: any verdict within Monte-Carlo error of its boundary
re-runs at B = 20000 and scores BOUNDARY if unstable. Tie rule
(pre-declared): if the top two forms' LOO-RMSE differ by < 5% of the
winner's, every verdict must agree across both, else that verdict
reports SPLIT with both values.

### Leans (sides declared, rule 22)

- **L-1 [prior .55] IDENTIFIED:** the LOO winner's q 95% bootstrap CI
  width ≤ 0.60. One-sided (smaller is better).
- **L-2 [conditional on L-1; priors below/overlap/above =
  .55/.35/.10]:** the winner's q CI against the response band
  [1.71, 1.98] — three outcomes, all named: entirely below / overlap /
  entirely above. Two-sided; every outcome informative. Registered
  lean: BELOW (the sealed composite under-predicts all 26 levels; the
  level picture at controlled V is intercept-like; the response-grade
  q was fitted on a pooled curve sharing K2f's collinearity).
- **L-3 [prior .70] TAX STABILITY:** the winner's κ CI overlaps K2f
  F2's κ′ ci95 [0.5202855978239498, 0.8612166024267973]. Two-sided
  overlap; disjoint-low and disjoint-high both named.
- **L-4 (reading, NO gate): (r, V)-sufficiency probe.** Within each
  share level, Spearman(residual, φ) across the 5 φ cells, per share,
  under the winner's fit; a monotone same-sign pattern in ≥3/4 share
  levels is the named finding "φ leaks past (r, V)" — T4's
  form-sufficiency questioned; reported, adjudicating nothing.

### Routing (rules 15/16 — the truth table, every realizable combination to exactly one outcome)

| # | condition | outcome |
|---|---|---|
| 1 | any Part-0/pilot gate fails after its declared ladder | **STOP_DESIGN_INFEASIBLE** (planner defect; no fit is run) |
| 2 | L-1 MISS AND winner λ CI contains 0 | **R_TERM_ABSENT_AT_LEVEL** — the tax-only level law is the COMPLETE level story on this family; level–response dissociation named; q-at-level closes as structurally unposed; M2 proceeds on the tax-only form |
| 3 | L-1 MISS AND winner λ CI excludes 0 | **NON_IDENTIFIED_UNDERPOWERED** — CI reported, no q claim; M2 blocked; leverage redesign named |
| 4 | L-1 HOLD AND L-2 below | **LEVEL_RESPONSE_DISSOCIATION** — q measured at level, below the response band; new named phenomenon; M2 seals the measured law |
| 5 | L-1 HOLD AND L-2 overlap | **SINGLE_EXPONENT_RESTORED** — T4's level form completed with the response exponent; M2 seals |
| 6 | L-1 HOLD AND L-2 above | **ABOVE_BAND_ANOMALY** — named; M2 seals the measured law; theory note required |
| — | L-3 disjoint (either side), any cell 2–6 | modifier **TAX_SHIFT_AT_LEVEL** — pre-registered anomaly fed into M3's charter |
| — | L-3 overlap, any cell 2–6 | modifier: κ's fourth independent appearance is counted |

L-2/L-3 are evaluated only in cells 2–6's L-1 HOLD branches where they
are defined (L-2 needs an identified q; in cell 2 the q question is
unposed and L-2 is recorded N/A; L-3 is evaluated in every cell 2–6
since κ is identified even where q is not — in cell 3, L-3 is reported
descriptively, adjudicating nothing). This enumeration is the full
partition (rule 16); the executor reproduces it in the report verbatim.

### Deliverables and budget

The six deliverables as always: (1)
`scripts/run_suica_m4_m1_r_at_level.py`; (2) artifacts under
`results/m4_m1_r_at_level/` (gitignored); (3)
`reports/SUICA_M4_M1_R_AT_LEVEL_REPORT.md` (tables generated from
artifacts); (4) outcome appended to THIS section (append-only); (5) one
`docs/CLAIMS_LEDGER.md` row (EXPLORATORY); (6) exactly ONE commit
`feat(m4-m): M1 — r-at-level on a decollinearized factorial — <SLUG>`,
never amended, never pushed. Suite (`python -m pytest -q -p
no:cacheprovider`) green before commit. Budget: ~640 worlds ≈ 0.6 s
each plus fits — target < 25 min wall, every stage < 600 s.

### Outcome (appended 2026-08-11 by the executing agent; append-only)

**`STOP_DESIGN_INFEASIBLE` — rule-16 cell 1 (a Part-0 gate fails after its
declared ladder; no fit is run).** Report:
`reports/SUICA_M4_M1_R_AT_LEVEL_REPORT.md`; harness
`scripts/run_suica_m4_m1_r_at_level.py`; artifacts `results/m4_m1_r_at_level/`.
**0 worlds were generated.** The leg stopped inside Part 0,
before the pilot, on this registration's own pre-declared fallback ladder.

**G0m PASSES completely — there is no citation defect.** Every anchor and every
K2f, D-open and theory-document number this registration quotes re-derives
bit-exactly: `predicted_attenuation(0.40, 0.90)`, `(0.45, 0.90)`,
`person_share_design(0.45, 0.0)` and `(0.40, 0.0)`; K2f F2's λ′, q′, κ′, p, its
LOO-RMSE at both persisted paths (`fits.json:L-1.best_loo_rmse` and
`loo.json:loo.F2.loo_rmse`) and both ci95s; D-open's M-4 level
0.09350089316336324 re-derived as the round-trip mean of its raw per-world
CSV; and `-0.9643543785903034` reproduced from `compiled_rows.csv` under Pearson,
which is what pins RN-M1-4's reading of "corr". The registration's facts are
accurate and its **mechanism is real**: φ does move `r` at exactly fixed `V`.

**G1m fails at gate (d), and the fallback ladder does not save it.** On the
registered base ladder `[0.6, 0.7, 0.8, 0.9, 0.98]`, gate (c) reached the
1.2 within-share `r` ratio in only **1** of 4 share
levels against a requirement of 2, and gate (d) measured
`corr(r, V) = -0.9407871367652862`. The pre-declared one-step extension to
`[0.45, 0.6, 0.75, 0.9, 0.98]` fired automatically inside Part 0 and did exactly its job on
(c) — **2**/4 levels, realized ratios
`[1.0525449708005403, 1.13540425278654, 1.2242247054603972, 1.3546050387988702]` — while (d) moved only to `-0.9107365249638539`,
**3.03578841654618× the 0.3 bar**. The failure is not an artifact
of the correlation convention: Spearman gives `-0.9306800811953776` and
`corr(r^q, V)` at the sealed response exponent gives `-0.9213071767159029`.
Gates (a), (b) and (e) pass (20 distinct design
points, `V` from 0.03000000000000001 to 0.18000000000000005, ratio 6.0).

**The bar is not merely missed — it is unreachable.** RN-M1-8's `diagnose` stage
(pure design arithmetic on the pinned deterministic maps; no world, no field, no
fit) searched the whole knob space. Over **every** 5-point distinct φ ladder in
(0.001, 0.999) at the registered shares, the infimum of `|corr(r, V)|` is
**`0.748768093111513` = 2.4958936437050436× the bar**, attained at the degenerate
ladder `[0.001, 0.011081, 0.021162, 0.988919, 0.999]` — the most extreme φ leverage this family can produce.
Freeing the shares as well, inside gate (a)'s envelope and subject to gate (b)'s
`V max/min ≥ 2.0`, the best of 4000 seeded draws
(474 rejected by (b)) is `0.5208187741410987` =
1.736062580470329× the bar. And the G2m liveness fallback ladder
`[0.9, 0.92, 0.94, 0.96, 0.98]`, had it ever been
reached, would have made (d) **worse** at
`-0.99515544292931` — every escape route
this registration wrote leads away from its own gate.

**Why, mechanically.** `V` is an EXACT linear function of share in this family
(`V/share = 0.3` at every share tested), and `r` is monotone
decreasing in share, so share drives `r` and `V` in lockstep. φ is the only knob
orthogonal to `V`, and its TOTAL leverage across the full open interval is
`0.05159009087311539` in `r` at share 0.1 rising to
`0.21722718146551878` at share 0.6 — against a
between-share `r` span of `0.2909602839380743` at fixed φ. Gate (b) REQUIRES a
share range wide enough that `V` varies by 2×, which is precisely what makes the
between-share `r` spread dominate; gate (d) requires the within-share φ-driven
spread to dominate instead. **In this world family both cannot hold at once.**
A second consequence, worth its own line: gate (c)'s 1.2 ratio is
reachable in only **2 of the 4 registered share
levels for any φ ladder whatsoever** (at shares `[0.1, 0.25]` the
full open φ interval tops out at ratios
1.06718684974016 and
1.1762868144261993), so "at least
2 share levels" was unknowingly demanding *all* the levels
that can ever comply.

**The defect (the planner owns it).** Rule-11 satisfiability, and rule-18 JOINT satisfiability across clauses sharing generative knobs: gate (b) and gate (d) share `share`, and the registration checked neither jointly nor arithmetically against the bar it wrote. The registration's own cited facts already contained the warning -- it quotes r(0.30, 0.90) = 0.6758917867864564 vs r(0.30, 0.98) = 0.645057248597175, a 4.78% move, and never asked what correlation a knob that small can buy against a share axis that moves r by 0.2910 at fixed phi.

**What was NOT run.** G2m's 16-world pilot, G3m(b)'s power projection (it needs
pilot σ_w), the 640 main worlds, and the fit — cell 1 reads
"no fit is run" and the ordering puts the pilot after the Part-0 gates. L-1,
L-2, L-3 and L-4 are all recorded **NOT EVALUATED**, with the reason in the
report's lean table. The four pre-declared forms, the start grid, the optimizer
pins, the bootstrap spec and every rule-22 side were nevertheless fixed in Part 0
before the stop and are persisted in `part0.json`, so a re-registration inherits
them unchanged rather than re-deriving them.

**Where the leverage would have to come from (a question for the planner, not an
executor's choice — nothing here is adopted).** As long as the only two knobs are
share and φ, `V` *is* share and the design has one effective axis plus a weak
second. Decollinearizing `r` from `V` at level needs either a knob that moves `r`
without moving share, or a `V` that is not a function of share alone.
`person_share_design(share, int_share)` sums the slow AND interaction shares, so
the `int_share` carrier — which M1 pinned to zero by inheriting K2F-FRESH
verbatim — moves `V` at fixed share. That is a second axis in exactly the place
the current design has none.

**Anomalies, with timing. No hypothesis-relevant number ever existed in this
leg** (0 worlds, no fit), so every anomaly below is
pre-hypothesis by construction. **A-1 (before any number of any kind):** the
dispatched environment did not exist — the only pandas on this machine belongs to
CPython 3.9.6, and the published machinery imports `datetime.UTC` (3.11+), so
`k2b`/`k2c`/`k2d`/`k2e` and the K2f harness all fail to import there; a CPython
3.12.12 venv was built outside the repo from `requirements-lock-main.txt`
verbatim (numpy `2.4.4`, pandas `3.0.2`, scipy `1.17.1` —
the lock's own pins) and the full suite was run green on it BEFORE any leg code
was written. **A-2 (before Part 0):** macOS ships no `timeout(1)`, so each stage
ran as its own foreground command under an explicit harness timeout. **A-3 (Part
0):** the rule-17 ladder fired — recorded because the firing is itself an event.
**A-4:** Part 0 ran more than once, every run before any world, deterministic and
identical each time; the ordering log was reset once so it carries the clean
final pass end to end. **A-5:** a first exploratory diagnostic called
`predicted_attenuation` inside an optimizer loop, overran its foreground timeout
and was killed; it was rewritten to precompute `r` on a fixed
100-point φ grid (2.072169780731201 s) and **no number from the
killed run appears anywhere in this leg**. **A-6 — rule 24 caught three of this
leg's own errors before commit:** the φ-leverage move at share 0.30 hand-typed as
"4.6%" where the anchors give `4.780124284524867`%; a leverage table whose headers
read "r at phi min/max" while the cells carry `r.min()`/`r.max()`, which are the
OPPOSITE φ endpoints since `r` decreases in φ; and a `V max/min ≥ 6.0` that
quoted the design's realized ratio where the gate's bar is
2.0. All three are now generated from artifacts.
**A-7:** no stage approached its 2× threshold — Part 0 `0.0369420051574707` s
against a 60 s estimate, `diagnose` `2.072169780731201` s against a 120 s
executor estimate.

**Line consequence.** M2 was registered as blocked only on
`NON_IDENTIFIED_UNDERPOWERED`; cell 1 is a different stop — nothing about the
level law was measured, so nothing about it was learned, and M2's charter is
untouched but unfed. M3 (the one-κ question) is independent of this leg and is
not blocked by it.

### Planner adjudication (2026-08-11, appended after the run) — THE STOP IS ACCEPTED; THE DEFECTS ARE MINE

**STOP_DESIGN_INFEASIBLE accepted as delivered.** The executor's conduct
was exemplary: G0m bit-exact on every quoted number (no citation
defect), ZERO worlds spent, the gate failure PROVEN unsatisfiable
rather than merely observed (infimum |corr(r, V)| over every 5-point φ
ladder at the registered shares = 0.748768093111513 = 2.496× the bar;
with shares freed under gate (b), best found 0.5208187741410987 — an
upper bound on that infimum, itself already 1.74× the bar), and rule 24
catching three of its own report cells pre-commit.

**Defect #43 (rule 11 violated in its letter).** G1m(d) was
deterministic arithmetic from pinned source objects — computable by the
planner at registration time with no pilot and no world. It was not
computed. The planner extrapolated φ's r-leverage from the two
published points (φ ∈ {0.90, 0.98}) into an unexercised region where
the attenuation map SATURATES: total φ leverage over the open interval
is 0.0516–0.2172 in r against a between-share span of 0.2910.

**Defect #44 (rule 18 violated).** Gates (b), (c), (d) share the
`share` knob and pull in opposite directions — (b) demands a wide share
range (V max/min ≥ 2), which is exactly what makes r track V; (d)
demands the within-share spread dominate. The joint check was not run;
the executor's diagnostic proves (b)+(d) jointly empty at ANY
admissible shares. Inside the same defect: gate (c)'s "≥ 2 of 4 share
levels" unknowingly named the only two levels that can EVER comply
(shares 0.10/0.25 top out at ratios 1.067/1.176 for any φ ladder).

**The deeper reading, and the rule it buys.** The failed bar sat on a
PROXY. Marginal corr(r, V) across cells is not what identification
needs: in a factorial, q is identified from within-share φ sweeps at
EXACTLY fixed V, and κ from between-share contrasts — the marginal
correlation can stay high while the conditional design information is
ample. The registration made the proxy a hard gate and left the
estimand-relevant object (the G3m-b projection) downstream of it, so
the leg died on a number the estimand does not consume. K2f's
follow-up phrase ("arms chosen to BREAK the r/V collinearity") seeded
the framing: "break the collinearity" was operationalized as "drive a
marginal correlation below a bar," where the operative content of
K2f's complaint was "restore identification" — K2f's collinearity
harmed it because that corpus ALSO had no within-stratum sweeps; the
factorial restores the sweeps, which is what suffices. **Rule 25
(enacted):** every design-feasibility gate is stated in the quantity
the leg's estimand requires (identification width, power, a registered
projection); marginal or proxy statistics of the design are REPORTED,
never gating. **Convention (planner-side):** when every input to a
registered gate is deterministic arithmetic from pinned source
objects, the planner RUNS the gate before committing the registration
and embeds the computed table in the registration text.

**Carried forward.** M1's Part-0 pins (four forms, optimizer, start
grid) inherit into M1b unchanged; the executor's leverage table is
M1b's design basis; leans and priors inherit UNCHANGED — zero worlds
ran, so no hypothesis-relevant information exists to update on.
Registry: #43, #44 and rule 25 appended by dated note. Appendix W is
untouched by this stop (it concerns the fit family, which never ran).

---

## M4-M1b — r-at-level, feasibility restated in the estimand's quantity

**REGISTERED 2026-08-11, BEFORE RUN.** Re-registration of M1 after its
cell-1 STOP. Planner: this document's author; executor: dispatched
agent (implementation and execution only; this text is binding).
Question, machinery, source objects, seeds policy, forms F1/F1e/F2/F3,
optimizer pins, start grid, LOO-cell selection, within-cell
world-block bootstrap (B = 2000; 20000 at rule-13 boundaries), tie
rule at 5%, leans L-1 [.55] / L-2 [below/overlap/above = .55/.35/.10,
conditional on L-1] / L-3 [.70] / L-4 (reading), and the rule-16 truth
table cells 2–6 with modifiers are INHERITED VERBATIM from M4-M1
above. Exactly the following changes.

### Design (pinned; planner arithmetic RUN at registration — rule 11 discharged, rule 25 compliant)

Grid: share ∈ {0.10, 0.25, 0.40, 0.60} × φ ∈ {0.05, 0.30, 0.60, 0.85,
0.98} = 20 cells × 32 worlds/cell. master_seed 20260811; salts
`m4m1b-world` / `m4m1b-pilot` (fresh streams; disjoint from M1's by
salt). φ at {0.05, 0.30, 0.60, 0.85} is a regime EXTENSION beyond the
exercised {0.90, 0.98} — guarded by G2m′, and the law claim stays
scoped to the tested grid.

Planner-computed design table (via the pinned maps, executor
reproduces bit-exactly in Part 0 — G0m′(vii)):

| share | V_person | r(φ=.05) | r(φ=.30) | r(φ=.60) | r(φ=.85) | r(φ=.98) | span |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.03000000000000001 | 0.8189581462487876 | 0.8155586799827954 | 0.8075174172340943 | 0.7908869485651705 | 0.7718092954224756 | 0.04714885082631204 |
| 0.25 | 0.07500000000000002 | 0.785015540293945 | 0.7761302864207245 | 0.7558507450373838 | 0.7168731389294273 | 0.6763691758553391 | 0.10864636443860598 |
| 0.40 | 0.12000000000000004 | 0.7411873080384952 | 0.726425348215848 | 0.6941115392115328 | 0.6367206581308248 | 0.5825497814736654 | 0.15863752656482977 |
| 0.60 | 0.18000000000000005 | 0.6573448847694047 | 0.6346912945232521 | 0.5883719155687073 | 0.5151304058057474 | 0.4541409476972356 | 0.20320393707216905 |

Descriptives (REPORTED, never gating — rule 25): marginal corr(r, V) =
−0.8495063312353189; corr(r^1.8528700746510731, V) =
−0.8649603255864755; against K2f's −0.9643543785903034. The
identification content lives in the within-share sweeps (V exactly
fixed by design), not in the marginal correlation — that sentence is
the rule-25 exemplar.

Pre-declared ALT ladder (fires ONLY on G2m′'s regime guard): φ ∈
{0.30, 0.55, 0.75, 0.90, 0.98}; planner table: spans
0.04374938456031985 / 0.09976111056538539 / 0.1438755667421826 /
0.1805503468260165 by share; corr(r, V) = −0.8915685583022667;
corr(r^q, V) = −0.9029258027968385; also passes G1m′ below.

### Gates

- **G0m′ (anchors, bit-exact).** (i)–(vi) as M1's G0m verbatim (all
  passed in M1; re-verify); (vii) reproduce BOTH planner design tables
  above bit-exactly from `predicted_attenuation` /
  `person_share_design`; (viii) verify the M1-STOP numbers cited in
  the adjudication above against `results/m4_m1_r_at_level/`
  (stop_diagnostic.json and the report tables): infimum
  0.748768093111513, freed-shares bound 0.5208187741410987, per-share
  full-interval spans 0.05159009087311539 / 0.11784317303319514 /
  0.17083747134975158 / 0.21722718146551878. Any mismatch → STOP
  (citation defect, not silently repaired).
- **G1m′ (arithmetic; planner-verified at registration, executor
  re-runs).** (a) shares inside [0.02, 0.6634207990183637] — PASS. (b)
  V max/min ≥ 2 — realized 6.0, PASS. (c′) within-share r-span ≥ 0.12
  at BOTH shares {0.40, 0.60} — realized 0.15863752656482977 /
  0.20320393707216905 (ALT: 0.1438755667421826 / 0.1805503468260165),
  PASS. (e) 20/20 distinct design points — PASS. There is NO
  marginal-correlation gate (rule 25).
- **G2m′ (pilot; AFTER G0m′/G1m′, before any main world).** Corners
  {0.10, 0.60} × {0.05, 0.98} × 4 worlds on `m4m1b-pilot`. (i) REGIME
  guard: finite, non-saturated, nonzero within-corner variance;
  failure at a φ-extension corner → ALT ladder once (re-run G1m′ and
  G3m′(b)), then STOP_DESIGN_INFEASIBLE. (ii) φ→r channel liveness:
  PRIMARY object = a per-world realized card-attenuation statistic if
  the k2b machinery persists one (executor pins the source object
  file:line in Part 0 — rule 12; requirement: contrast between φ .05
  and .98 at share .60 exceeds 2× its pooled SE). If no such statistic
  exists, liveness is certified by the pinned map's arithmetic (Δr =
  0.20320393707216905 at share .60) PLUS the projection — an
  OUTCOME-side field contrast is NOT a liveness gate here, because a
  flat field is cell-2 EVIDENCE, not channel death; M1's registration
  conflated these and M1b does not (rule 25 applied twice).
- **G3m′.** (a) sides per rule 22 as in M1. (b) **the feasibility
  gate — the only one:** σ_w = pooled per-world sd across the 16 pilot
  worlds, inflated by sqrt(12 / χ²_{0.10, df=12}); parametric
  replication B_proj = 500, seed = master, at truths
  (λ = 0.18021628978547316, κ = 0.750086268225045, ε = 0) × q_truth ∈
  {1.0, 1.8528700746510731}, cell noise N(0, σ_w²/32), fitting F1
  with the full start grid; width proxy = quantile(q̂, .975) −
  quantile(q̂, .025); PASS iff proxy ≤ 0.50 under BOTH truths.
  Escalation (pre-declared, once): on fail, recompute at 64
  worlds/cell (noise /√2); pass → the main grid runs at 64 worlds/cell
  (budget ×2, declared); fail again → STOP as **NON_PROJECTABLE**,
  with the int_share second axis (V moves at fixed share;
  species-disclosed, K2d dispatcher required) NAMED for a future
  registration, not adopted here. No projection at q_truth = 0
  (pre-declared adjudicable branch, as in M1). (c) Stage estimates:
  part0 60 s, pilot 40 s, worlds 4 × 150 s (5 cells per chunk; ×2 at
  escalation), fit 300 s, finalize 60 s; the 2× stop-and-report
  convention applies.
- **G4m′.** As M1's G4m: the inherited truth table reproduced verbatim
  in the report with cell 1 now reading "any G0m′/G1m′/G2m′/G3m′
  clause fails after its declared ladder → STOP (planner defect)";
  rule 24 — every table carrying artifact numbers generated from
  artifacts.

### Deliverables and budget

The six deliverables: `scripts/run_suica_m4_m1b_r_at_level.py`;
`results/m4_m1b_r_at_level/` (gitignored);
`reports/SUICA_M4_M1B_R_AT_LEVEL_REPORT.md` (generated tables);
outcome appended HERE (append-only); one `docs/CLAIMS_LEDGER.md` row
(EXPLORATORY); exactly ONE commit
`feat(m4-m): M1b — r-at-level, estimand-gated — <SLUG>`, never
amended, never pushed. Suite green before commit. Budget: 640 worlds
(1280 at escalation) ≈ 0.6 s each plus fits — target < 30 min wall,
every stage < 600 s.

### Outcome (appended 2026-08-11 by the executing agent; append-only)

**`NON_PROJECTABLE` — rule-16 cell 1, via G3m′(b) failing after its once-only
escalation.** Report: `reports/SUICA_M4_M1B_R_AT_LEVEL_REPORT.md`; harness
`scripts/run_suica_m4_m1b_r_at_level.py`; artifacts `results/m4_m1b_r_at_level/`.
**16 pilot worlds ran; 0 main worlds were
generated and no fit was run.**

**Rule 25 is validated in BOTH directions by this one leg, and that is the
headline.** It carried M1b PAST a marginal `corr(r, V)` of `-0.8495063312353189`
(`corr(r^q, V) = -0.8649603255864755`) — nearly 3× M1's withdrawn 0.30 bar, now
correctly REPORTED and gating nothing. It then STOPPED the leg on the quantity
the estimand actually consumes. Two different failures; only the second is
information.

**G0m′ PASSES on all eight clauses.** (i)–(vi) re-verified from M1. **(vii): both
of the planner's embedded design tables reproduce BIT-EXACTLY** — every `r`,
every `V`, all four spans on the main ladder, all four ALT spans, and all four
descriptives (`-0.8495063312353189`, `-0.8649603255864755`,
`-0.8915685583022667`,
`-0.9029258027968385`). The planner ran its own arithmetic before committing, as
the convention defect #43 bought requires, and it is correct to the last bit.
**(viii): the M1-STOP numbers cited in the adjudication verify against this
executor's own `results/m4_m1_r_at_level/`** — infimum `0.748768093111513`,
freed-shares bound `0.5208187741410987`, and all four full-interval spans.
**G1m′ PASSES**: (a), (b) (V ratio 6.0), (c′) — the absolute-span
repair — at `[0.15863752656482977, 0.20320393707216905]` against a `0.12` bar, and (e) 20/20
distinct points. Machinery inheritance was not asserted but PROVEN: the copied
harness was compared against the imported M1 module and all four forms fit a
fixed probe bit-identically, with identical start grids, optimizer dict and
inherited bars (RN-M1B-1).

**G2m′ PASSES, and its liveness clause is the registration's second
vindication.** All four corners finite, non-saturated, nonzero variance. The
φ→r channel is alive beyond argument: the realized card attenuation moves
`0.20325550047558588` between φ .05 and .98 at share .60 =
**94.8954999654606× its pooled SE**, landing `5.156340341683219e-05` from the
pinned map's predicted `Δr = 0.20320393707216905`. **The FIELD at that same
corner moves `-0.007269536568279722` = 0.7542598230697173× its pooled SE —
flat within noise.** M1's registration would have gated on exactly that number
as its declared fallback and the leg would have died a SECOND false death, on
evidence that the field does not respond to φ, which is the very thing the leg
exists to measure. M1b's text forbids it in advance ("a flat field is cell-2
EVIDENCE, not channel death"), written before the number existed. Read honestly
that flatness is weak evidence toward cell 2 (`R_TERM_ABSENT_AT_LEVEL`) — four
worlds against four at one share — and it adjudicates NOTHING here.

**G3m′(b) FAILS — the stop.** σ_w = `0.019489117988137468` pooled over the 16 pilot
worlds (df 12), df-inflated by `1.3797155080850578` to **`0.026889438327132725`**.
Projected 95% width of q̂ at 32 worlds/cell: `0.6446327208199195` at q_truth 1.0 and
`1.1702741415331803` at q_truth 1.8528700746510731, against the `0.5`
bar — both fail. The pre-declared once-only escalation FIRED. At 64
worlds/cell: `0.45036131116284384` (CLEARS) and `0.8082914682805795` (does not). The
registration requires BOTH. **Gate FAIL → NON_PROJECTABLE.** The failure is
PRECISION, not pathology: median q̂ tracks its truth at every configuration
({'1.0': 1.0021891019795262, '1.8528700746510731': 1.8442504925087377} at n=32) and zero replicates failed to converge.

**What n WOULD suffice — measured, not extrapolated.** Defect #43 was an
extrapolation where arithmetic was available; the handoff does not repeat it.
On a declared geometric ladder `[64, 128, 192, 256, 384, 512]`, running the binding truth until
it clears and confirming the other truth there: **192 worlds/cell** is
the smallest passing rung (widths `[0.7859406063487944, 0.5516920936367253, 0.45033528452170346]` at `[64, 128, 192]`),
with the non-binding truth confirming at `0.2531601642892628`. That is
**3840 worlds, 6.0× the registered base budget** — at
~0.6 s/world a wall-clock change, not a feasibility change. Precision caveat,
disclosed: the ladder re-drew the n=64 binding cell on a fresh stream and got
`0.7859406063487944` where the gate got `0.8082914682805795` (abs diff
`0.02235086193178515`), so the width proxy itself carries ~2.8%
Monte-Carlo error at B_proj = 500 and 192 should not be read as exact.

**This is NOT a registration defect and should not be recorded as one.** Every
clause was satisfiable, every bar was computed at registration, the ladder and
the escalation both fired exactly as written, and the gate returned a
well-defined verdict on a well-posed quantity. `NON_PROJECTABLE` is a
pre-declared outcome of a SOUND registration. **One judgement call is flagged
for the planner, not as a defect but as a choice worth revisiting:** the
two-truth conjunction is decided entirely by q_truth = 1.8528700746510731,
while the registered L-2 lean puts .55 on q being BELOW the response band. At
q_truth = 1.0 the escalated design already clears at
`0.45036131116284384`. Larger q means smaller r^q on r < 1, so the exponent's signal
shrinks and its interval widens — the gate is hardest exactly where the leg
thinks the truth is not. Whether that conjunction is intended conservatism or
over-strict is the planner's to settle; the executor scored it as written and
takes no position.

**Three routes, none adopted here.** (1) Buy the precision: 192
worlds/cell. (2) Re-state the gate if the two-truth conjunction is stricter than
intended — it would pass at 64 today under a lean-weighted or q-anchored
variant. (3) The int_share second axis, which this registration NAMES and
forbids adopting: it was therefore NOT probed, because exercising it requires
installing K2d's `int:` dispatcher on every reachable k2b instance (RN-K2F-5), a
machinery mutation this leg has no licence to make for a knob it may not adopt —
and under the planner-side convention #43 bought, that arithmetic is the
PLANNER's to run before registering a successor. It is deterministic and needs
no world.

**Anomalies, with timing.** The hypothesis-relevant boundary here is the PILOT:
before it no outcome-side number existed; after it the field contrast and σ_w
did; no lean was ever scored. **A-1/A-2 (before Part 0):** the M4-M1 environment
is reused verbatim — a CPython 3.12.12 venv outside the repo from
`requirements-lock-main.txt` (numpy `2.4.4`, pandas `3.0.2`, scipy
`1.17.1`), because this machine's only pandas belongs to CPython 3.9.6
which cannot import the published machinery; and macOS ships no `timeout(1)`, so
each stage ran as its own foreground command under an explicit harness timeout.
**A-3 (AT the pilot — the first hypothesis-relevant number):** the outcome-side
flatness above; it changed no gate because the registration had already removed
the field from the liveness clause BEFORE the number existed. **A-4 (after
σ_w):** the gate failed and the escalation fired, once, exactly as pre-declared.
**A-5:** RN-M1B-8 (the n-ladder) was added after σ_w existed — disclosed; it
consumes only σ_w and the pinned maps, no lean-relevant quantity, and adopts
nothing. **A-6:** the Monte-Carlo discrepancy at the one overlapping
configuration, disclosed above rather than smoothed. **A-6b — rule 24 caught a
claim in this leg's own prose pre-commit:** the liveness paragraph first said the
realized card contrast "agrees bit-exactly" with the pinned map's Δr; it does
not and could not (a measurement over 8 worlds against deterministic algebra),
they differ by `5.156340341683219e-05`, and the sentence is now generated from
both values. **A-7:** no stage approached its 2× threshold — Part 0
`1.9119329452514648` s vs 60, pilot `9.718364953994751` s vs 40, power
`82.15909004211426` s vs 120, diagnose `79.69169306755066` s vs 300.

**Line consequence.** M2 was registered as blocked on
`NON_IDENTIFIED_UNDERPOWERED`; this is a different stop and reaches it earlier —
the level law's exponent is not merely unidentified on the data, it is priced:
identification at the registered width costs 192 worlds/cell on this
design. M2's charter is untouched but still unfed. M3 (the one-κ question) is
independent of this leg and remains unblocked — and note that κ, unlike q, was
never the binding difficulty in any projection here.

### Planner adjudication of M1b (2026-08-11, appended after the run) — THE GATE WORKED; FUND THE MEASURED BUDGET

**NON_PROJECTABLE accepted as delivered. Zero registration defects** —
the first M-line leg on which the registration survived contact whole:
every clause satisfiable, every bar computed at registration, ladder
and escalation fired exactly as written, and the gate returned a
well-defined verdict on a well-posed quantity (σ_w =
0.026889438327132725 df-inflated; q-widths 1.1702741415331803 /
0.6446327208199195 at n=32 and 0.8082914682805795 /
0.45036131116284384 at n=64, truths 1.8528700746510731 / 1.0).

**Rule 25 validated in both directions by one leg, on the record.**
(i) It carried the leg PAST marginal corr(r, V) = −0.8495063312353189
— which M1's withdrawn proxy gate would have killed at Part 0. (ii) It
prevented a second false death at the pilot: the card channel moved
0.20325550047558588 = 94.90× pooled SE while the field at the same
corner moved −0.007269536568279722 = 0.754× SE — M1's declared
fallback would have gated on that flatness and killed the leg on what
is in fact WEAK CELL-2 EVIDENCE (the field not responding to φ is a
possible ANSWER, not channel death). Recorded as the rule's first
double exemplar.

**The executor's flagged judgement call is answered, and kept.** The
two-truth conjunction is decided by the q = 1.853 truth while L-2's
registered lean puts .55 on the truth being BELOW the band — the gate
is hardest exactly where the planner thinks the truth is not. That is
DELIBERATE and it stays in M1c: the instrument must be able to REFUTE
the lean, not merely confirm it; power against the disfavored
hypothesis is what makes the .55 falsifiable rather than
self-fulfilling.

**A planner note on the pilot whisper (adjudicating nothing).** At the
share-.60 corner the observed field contrast −0.0073 sits ≈5 pooled-SE
below the ≈ +0.041 that a (λ = 0.180, q = 1.853) LEVEL-truth predicts
across that corner's Δr = 0.203 — while the card channel underneath
moved at 94.9× SE. Four worlds per corner; the projection's own MC
error (the disclosed 2.8% proxy discrepancy) is the standing
demonstration of how unreliable this n is. Leans and priors therefore
inherit into M1c UNCHANGED; the note exists so that, whichever cell
the main run lands in, the record shows the first whisper and that it
moved nothing.

**Route: fund the measured budget.** The diagnostic ladder puts the
sufficient budget at **192 worlds/cell** (widths 0.45033528452170346
under q 1.8528700746510731; 0.2531601642892628 under 1.0) = 3840
worlds ≈ 6× base — feasible on this machine. Weakening the bar or
dropping the hard truth would be bar-shopping; the budget is what the
gate measured, so the budget is what the leg pays. M1c below. The MC
caveat (n=64 re-draw 0.7859 vs gate 0.8083) drives M1c's projection
precision: B_proj = 2000 with a pre-declared 10000-draw boundary
re-run.

---

## M4-M1c — r-at-level at the measured budget

**REGISTERED 2026-08-11, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. Everything INHERITED VERBATIM from
M4-M1b (which inherits M4-M1): question, machinery, source objects,
grid share {0.10, 0.25, 0.40, 0.60} × φ {0.05, 0.30, 0.60, 0.85,
0.98}, both planner design tables (G0m′(vii) unchanged), forms
F1/F1e/F2/F3, optimizer pins, start grid, LOO-cell selection,
within-cell world-block bootstrap B = 2000 (20000 at rule-13
boundaries), tie rule, leans L-1 [.55] / L-2 [.55/.35/.10 conditional]
/ L-3 [.70] / L-4 (reading), truth-table cells 2–6 with modifiers, and
L-1's 0.60 bar. Exactly the following changes.

- **Worlds/cell: 192** (the budget M1b's diagnostic measured). Salt
  `m4m1c-world`, master_seed 20260811, world indices 0..191 per cell;
  3840 worlds total.
- **No new pilot.** G2m″: M1b's pilot artifacts are the pinned noise,
  regime and liveness source — same instrument, same corners; G0m″
  verifies σ_w = 0.026889438327132725 and the pilot pass records
  bit-exactly from `results/m4_m1b_r_at_level/`. A SMOKE stage
  (before the remaining worlds, after Part 0): generate world index 0
  for each of the 20 cells and check ONLY per-world
  finiteness/saturation booleans — no aggregation, no level is read
  (pinned; these 20 worlds are retained in the main sample); any
  failure → STOP (a mid-grid regime break is a planner-scale
  surprise; the ALT ladder is NOT available here since it would need
  a fresh projection).
- **G3m″ feasibility confirmation (Part 0, before any world):**
  recompute the projection from M1b's persisted σ_w at n = 192,
  B_proj = 2000, seed = master, both truths; PASS iff both widths ≤
  0.50. Boundary rule (rule 13): if either width lands in
  [0.47, 0.53], re-decide at B_proj = 10000. On FAIL: once-only
  escalation to n = 256 (5120 worlds; recompute; pass → main at 256,
  budget declared; fail → STOP as NON_PROJECTABLE_AT_CEILING, handed
  back).
- **G0m″ anchors:** M1's (i)–(vi) and M1b's (vii)–(viii) re-verified,
  PLUS bit-exact verification of every M1b number cited in the
  adjudication above against `results/m4_m1b_r_at_level/` (σ_w, the
  four gate widths, the ladder widths 0.45033528452170346 /
  0.2531601642892628, cell sd 0.004753426045853251, the pilot
  contrasts 0.20325550047558588 / −0.007269536568279722 with their
  SE multiples).
- **Stages:** part0 (incl. projection) 240 s; smoke 30 s; worlds in 5
  chunks of 4 cells × indices 1..191 (~480 s each; 7 chunks of 3
  cells at escalation); fit 420 s; finalize 60 s; the 2×
  stop-and-report convention applies. Target < 80 min wall.
- **Deliverables:** `scripts/run_suica_m4_m1c_r_at_level.py`;
  `results/m4_m1c_r_at_level/` (gitignored);
  `reports/SUICA_M4_M1C_R_AT_LEVEL_REPORT.md` (generated tables);
  outcome appended HERE; one `docs/CLAIMS_LEDGER.md` row
  (EXPLORATORY); exactly ONE commit
  `feat(m4-m): M1c — r-at-level at the measured budget — <SLUG>`,
  never amended, never pushed; suite green before commit.

### Outcome (appended 2026-08-11 by the executing agent; append-only)

**`LEVEL_RESPONSE_DISSOCIATION` — rule-16 cell 4 (L-1 HOLD, L-2 below),
modifier `KAPPA_FOURTH_APPEARANCE`.** Report:
`reports/SUICA_M4_M1C_R_AT_LEVEL_REPORT.md`; harness
`scripts/run_suica_m4_m1c_r_at_level.py`; artifacts `results/m4_m1c_r_at_level/`.
**3840 worlds** (20 cells x 192),
the budget M1b's own diagnostic measured and the planner funded.

**The exponent is identified at level, and it is not the response exponent.**
Winner **F1e** (`field = lambda*r^q - kappa*V - epsilon, epsilon in [0, 0.05]`): q = **`-0.15040108849226472`**, 95% CI
`[-0.18322395953281184, -0.11871900002844447]`, width `0.06450495950436737` against L-1's `0.6` bar —
**L-1 HOLDS with 9.3x room**. The interval lies **entirely below** the response
band `[1.71, 1.98]` and **entirely below zero**: L-2 = **below**, the
registered lean (BELOW, prior .55) called it, and cell 4's
`LEVEL_RESPONSE_DISSOCIATION` is the outcome. lambda = `0.2249206339499495` CI
`[0.2226976852269149, 0.2267740781729326]` **excludes zero**, which is what keeps this out of cell 2.
kappa = `0.7601952008701406` CI `[0.7356727662590873, 0.7846243216827854]` sits inside K2f's
`[0.5202855978239498, 0.8612166024267973]` — L-3 **overlap**, **kappa's FOURTH independent
appearance**, and the tightest interval yet recorded for it.

**The sign is the finding, and it is visible in the raw cell means before any
fit.** At EXACTLY fixed V = `0.18000000000000005` (share 0.6), moving phi
across the ladder drops predicted attenuation r from `0.6573448847694047` to
`0.4541409476972356` while the measured b-only field recovery RISES from
`0.05410832013119198` to `0.063796931786496` — a gain of
4.03363828257993 pooled SEM. Less predicted card attenuation, MORE field
recovery, person-variance held exactly constant. The exponent does not merely
differ between grades; it is opposite in sign.

**All four forms agree on the picture** (q negative and small, kappa ~0.76,
lambda ~0.18-0.22): LOO-RMSE {'F1': 0.003198131708377386, 'F1e': 0.0031856515917748638, 'F2': 0.0034019365713125944, 'F3': 0.003877604046883495}; in-sample RMSE {'F1': 0.0026264051166751978, 'F1e': 0.002621078709438027, 'F2': 0.002591249722764473, 'F3': 0.0033903747201612703}.
Per-cell SEM ranged `0.0015046764572937737`-`0.0020066026535869932`; cell mean field
`0.05410832013119198`-`0.16512469544098618`.

**Gates.** G0m'' PASS on all nine clauses — **39 numbered citation
checks bit-exact**, including (vii) the planner's design table, (viii) the
M1-STOP numbers against `results/m4_m1_r_at_level/`, and (ix) EVERY M1b number
this adjudication cites against `results/m4_m1b_r_at_level/` (sigma_w, the four
gate widths, the ladder pair, the cell sd, both pilot contrasts with their SE
multiples). G1m'' PASS. **G2m'' PASS with no new pilot** — M1b's persisted pilot
is the pinned regime/liveness/noise source, verified bit-exactly. **G3m''
CONFIRMED BEFORE ANY WORLD**: from M1b's persisted sigma_w = `0.026889438327132725`,
at n = 192 and B_proj = 2000, projected q widths
{'1.0': 0.24923889216646022, '1.8528700746510731': 0.46602037304504784} against the `0.5` bar; boundary rule fired
**False**, escalation fired **False**. M1b's
B_proj=500 ladder had said [0.45033528452170346, 0.2531601642892628] at the same configuration — the
honest re-run at 4x the draws agrees inside its own MC error. SMOKE PASS: world
0 of all 20 cells finite and non-saturated, booleans only, worlds retained.
G4m'' PASS. RN-M1C-1's inheritance proof passed against **both** predecessor
harnesses.

**Two independent signals say the registered form family does not span this
truth. Both are readings; neither moves the slug.** (i) **The winner's epsilon
is pinned at its declared upper bound**: `0.049999999999999996` against the bound
`0.05`, with a bootstrap interval `[0.049999999999909624, 0.049999999999999996]` whose LOWER
endpoint is `9.037909309839165e-14` from the bound — essentially every draw sat
on the constraint, so F1e contributes no effective free fourth parameter and is
F1 with a fixed offset and a re-scaled lambda. (ii) **Appendix W's
quadratic-in-r discriminator FIRES while L-4 stays quiet** — per-share
Spearman(residual, phi) = [0.0, 0.8999999999999998, 0.6, -0.6], reading A 2/4 and reading B
0/4 against a 3/4 bar (both False/False),
against an r^2 coefficient of `-0.19911194958208703` with CI `[-0.2879978718649799, -0.10706476050455438]`
excluding zero on the within-share fixed-effects reading (pooled
`-0.04921007645583713` `[-0.10592459329331352, 0.007421060164540356]` does not fire). W.1's own rule:
that configuration is **FORM_GAP evidence**, and the prescribed follow-up is a
registered form extension, NOT a phi-channel claim.

**An honest correction to appendix W.** W.1 predicted the span gap would appear
as a U-shape — "residuals positive at both r-extremes, negative mid-range" — a
POSITIVE r^2 coefficient. The measured coefficient is NEGATIVE: residuals low at
both r-extremes, high in the middle, an inverted U. The discriminator fires in
KIND exactly as W.1 specified and its prescription stands; the missing shape is
on the OPPOSITE side of the family's span from the "positive floor plus positive
power" W.1 hypothesised. The epsilon-at-bound finding points the same way
independently: the fit wants a MORE negative constant than the box allows, not a
positive floor.

**Tie rule and rule 13.** The tie rule FIRED: F1e beats
F1 by `1.2480116602522386e-05` = 0.39176024882147187% of the winner's LOO,
inside the 5% band, so every verdict had to agree across both forms — and every
verdict does, so nothing reports SPLIT. This matters more than usual here,
because F1 is UNBOUNDED and reaches the same three verdicts from
q = `-0.1888182542137735` CI `[-0.22686946646111852, -0.14900957557344477]`, kappa = `0.7596295789070726`, lambda =
`0.17505204234174737`: the negative-q, kappa~0.76 picture is not an artifact of a
clipped parameter. Rule 13 fired (True) on the near-tie and
the B=20000 re-run left every verdict unchanged (all stable:
True).

**What is settled that was not.** K2f could not separate an intercept from
lambda*r^q and reported q' straddling zero at `[-0.3792124136721057, 0.5313115708778163]` (width
`0.910523984549922`). M1c's interval is `[-0.18322395953281184, -0.11871900002844447]`, width `0.06450495950436737`,
and excludes zero. The r-term at level is present, small and NEGATIVE.

**What must not be claimed.** q = `-0.15040108849226472` is the best exponent within a
family the residuals demonstrate is too small — not the exponent of the world.
The claim is scoped to the tested grid (shares [0.1, 0.25, 0.4, 0.6], phi
[0.05, 0.3, 0.6, 0.85, 0.98], this instrument, this carrier), and phi at {0.05, 0.30,
0.60, 0.85} remains a regime extension beyond the exercised {0.90, 0.98}.

**Anomalies, with timing.** The hypothesis-relevant boundary is the `fit` stage;
Part 0, the smoke booleans and the world CSVs all precede any level being read
or aggregated, and every RN note was pinned in Part 0 before the first world.
**A-1/A-2 (before Part 0):** the M4-M1 environment reused verbatim (CPython
3.12.12 venv from `requirements-lock-main.txt`, numpy `2.4.4`,
pandas `3.0.2`, scipy `1.17.1`), and no `timeout(1)` on macOS so
each stage ran as its own foreground command. **A-3 (Part 0, before any
world):** the binding projected width landed just BELOW the `[0.47, 0.53]`
re-decide band, so the 10000-draw rule did not fire — disclosed because a
slightly noisier draw would have triggered it, and the call was made by the
registered rule rather than by preference. **A-4 (at the fit):** the form tie.
**A-5 (at the fit):** epsilon pinned at its bound. **A-6 (at finalize):**
appendix W's discriminator firing with the opposite sign to W.1's prediction.
**A-7:** no stage approached its 2x threshold — Part 0 `155.4262659549713` s
against 240 s, every world chunk inside its 480 s estimate.

**Registration-defect candidates: ONE, non-blocking.** The registration selects
by LOO across four forms, one of which has a BOUNDED parameter, and it does not
say what happens when a bounded form wins WITH ITS BOUND ACTIVE — whether the
boundary parameter's CI is interpretable, whether the form should be demoted to
its effective dimension, or whether the boundary is itself a reportable finding
(it is treated as one here). Nothing turned on it, because the tie rule
independently forced agreement with the unbounded F1. Flagged so a
successor can settle it in advance rather than in the report.

**For M2.** The winner's LOO-RMSE is `0.0031856515917748638`, so a +-2xLOO band is
+-`0.0063713031835497275`. Two cautions before registering the seal: sealing
F1e seals a form whose fourth parameter is a boundary artifact, and
sealing ANY of the four seals a family the residuals say is incomplete. Sealing
F1 — unbounded, 0.39176024882147187% behind, same verdicts — may be
the more honest object. That is a registration decision; the executor takes no
position beyond reporting that the two are statistically indistinguishable here.
M3 (the one-kappa question) is now materially better supplied: kappa's fourth
appearance arrives with the tightest interval in the record.

### Planner adjudication of M1c (2026-08-11, appended after the run) — THE EXPONENT IS MEASURED, ITS SIGN IS NEGATIVE, AND THE FAMILY IS CONVICTED

**LEVEL_RESPONSE_DISSOCIATION + KAPPA_FOURTH_APPEARANCE accepted.**
L-1 HOLD at width 0.06450495950436737 against a 0.60 bar — the
decollinearized factorial identified at the first attempt what two
corpora could not. L-2 BELOW — and not merely below the band:
**below zero** (winner F1e q = −0.15040108849226472
[−0.18322395953281184, −0.11871900002844447]; unbounded F1 agrees,
q = −0.1888182542137735 [−0.22686946646111852, −0.14900957557344477]).
The registered lean (.55 below) pays out, and the both-truth gate that
was hardest under the disfavored truth is what makes this payout
worth something. L-3 overlap: κ = 0.7601952008701406
[0.7356727662590873, 0.7846243216827854] — the tax's FOURTH
independent appearance (0.722 difference-fitted; ≈0.715 9-pair refit;
0.750 K2f level; 0.760 here). The substantive core, stated plainly:
**at exactly fixed V = 0.18, r falls 0.657 → 0.454 and the measured
field RISES 0.0541 → 0.0638 (4.03 pooled SEM) — the level law's
r-dependence has the OPPOSITE SIGN to the response-grade exponent.**

**The family is convicted by two independent readings.** (i) F1e's ε
sits ON its declared bound (CI collapsed to within 9.04e-14 of 0.05)
— the family wants more constant than the T4 species floor allows.
(ii) Appendix W's discriminator FIRED IN KIND WITH INVERTED SIGN:
within-share r² residual coefficient −0.19911194958208703
[−0.2879978718649799, −0.10706476050455438] with the Spearman probe
quiet (2/4, 0/4) — FORM_GAP, not φ-leak; but W.1 predicted a positive
U for the (c>0, q>0) corner and the world sits in the (c pressed,
q<0) corner. Appendix W's sign prediction was WRONG and the record
says so (theory appendix X.5); the prescription it attached (a
registered form extension, not a φ-channel claim) survives and is
executed as M1d below.

**Rule 26 (enacted, rule-13-style — paid for by M1c's non-blocking
candidate, on which nothing turned only because the tie rule forced
F1 agreement by luck):** when a bounded form wins selection with any
bound ACTIVE at its optimum, every verdict is co-adjudicated on its
unbounded relaxation (or nearest registered unbounded form);
disagreement reports SPLIT; bound-activity is itself a reported
finding, and the active-bound CI is flagged one-sided-by-construction.

**Theory consequence (dated, appendix X):** the response-grade band
q = 1.83 [1.71, 1.98] was measured on K2b's share-driven sweeps, where
r and V move in lockstep — no V-clean positive-exponent measurement
exists anywhere in the record. M1c's V-clean measurement is negative.
The re-attribution — **the "quadratic over-response" was mostly the
variance tax wearing r's clothes** — is recorded in appendix X with
its yield clause, and M1d carries the in-corpus demonstration
(the V-omitted shadow fit, pre-signed positive).

**The executor's M2 flag is answered by sequencing.** Cell 4's route
("M2 seals the measured law") STANDS; what M1d determines is WHICH
measured law is the sealable object — sealing F1e would seal a
boundary artifact and sealing an incomplete family would seal glue
(the D-open lesson). M1d is artifact-space (no new worlds), then M2
seals its winner at exterior configurations. Projection honesty note
carried: the binding width 0.46602037304504784 sat just below the
re-decide band and the registered rule decided — disclosed, correct.

---

## M4-M1d — the completion and the coordinate (artifact-space; the leg appendix W's discriminator prescribed)

**REGISTERED 2026-08-11, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. NO NEW WORLDS: the data are M1c's
persisted 3840-world corpus (`results/m4_m1c_r_at_level/`), the 20
cell means re-derived round-trip from the rawest per-world artifacts.
Optimizer pins, start-grid conventions, LOO-cell selection,
within-cell world-block bootstrap (B = 2000; 20000 at rule-13
boundaries), tie rule at 5%, and rule 26 are inherited/in force.

### Question

Two, sharp: (1) **Completion** — does one free intercept complete the
family (the exact gap appendix W.1 named)? (2) **Coordinate** — is
the level law's second argument r (card space) or φ (state dynamics)?
Within a share stratum r and φ are re-parametrizations; ACROSS shares
they differ exactly where r(share, ·) moves — the factorial can tell
them apart. The mechanism hypothesis being tested (named in appendix
X.4, not claimed): the deployed gauge is an occasion-structure
consumer, not a card consumer — slower state (higher φ) means more
coherent occasion blocks and a more readable person×frame
interaction, RAISING the field even as the card attenuates (K-R1's
scaffold finding and K2b's G4b supply the frame-carried substrate).

### Forms (six; the four incumbents frozen in their M1c roles, plus)

- **F0: field = c + λ·r^q − κ·V** — free intercept, ALL parameters
  unbounded (the W.1 gap made a form);
- **Fφ: field = c + a·φ^m − κ·V** — the coordinate alternative,
  unbounded; φ = the design φ of each cell.

Start grids (pinned): F0 — c ∈ {0, 0.05, 0.1}; λ ∈ {0.05,
0.17417497661611914, 0.5}; q ∈ {−1, −0.5, −0.15, 0, 0.5,
1.8528700746510731}; κ ∈ {0, 0.7220359963712748, 2.0} (162 starts).
Fφ — c ∈ {0, 0.05, 0.1}; a ∈ {0.01, 0.05, 0.15}; m ∈ {0.5, 1, 2, 4};
κ as F0 (108 starts). Nesting stated: F0 nests F1 at c = 0; Fφ nests
no incumbent. Selection: leave-one-CELL-out RMSE across all SIX
forms; bootstrap CIs for the winner(s); rule 26 co-adjudication
wherever a bound is active (none is declared for F0/Fφ).

### Leans (sides declared)

- **L-1d [.70]:** an extension (F0 or Fφ) beats ALL FOUR incumbents
  on LOO. One-sided.
- **L-2d [conditional on L-1d; .50/.50]:** Fφ vs F0 as LOO winner —
  the coordinate question; either answer re-types theory (X.4).
- **L-3d [.75]:** the winner's κ CI overlaps M1c's
  [0.7356727662590873, 0.7846243216827854] — the fifth appearance.
  Two-sided overlap.
- **L-4d (reading → routing):** the winner's within-share r²-residual
  CI contains 0 ⟹ the family is COMPLETE (routes M2); fires ⟹ M2 is
  DEFERRED (sealing an incomplete family is sealing glue).

### Readings (adjudicate nothing; both pre-signed)

- **The V-shadow demonstration:** fit field = λ·r^q with NO V term
  and NO intercept on M1c's own 20 cells. Pre-signed: q_shadow > 0 —
  omitting the tax flips the exponent's sign in-corpus, the
  re-attribution of the response-grade band made visible in one
  number. Report q_shadow with CI.
- **Legacy retrodiction:** the winner's plain RMSE on K2f's 26
  compiled rows (re-derived round-trip) vs the sealed form's
  0.11259090547752257 and K2f's refit LOO 0.0061559195350209.
  Scoped: same-instrument extrapolation across corpora, descriptive.

### Gates

- **G0d (bit-exact).** (i) Re-derive M1c's 20 cell means from the
  rawest persisted per-world artifacts; match the persisted means
  bit-exactly. (ii) Verify every M1c number quoted in the
  adjudication above against `results/m4_m1c_r_at_level/` at full
  precision (both q CIs, κ CI, λ CI, ε boundary gap, LOO values, tie
  margin 1.2480116602522386e-05, r² CIs, Spearman vector, field/SEM
  ranges, the share-.60 rise 0.05410832013119198 →
  0.063796931786496 and its 4.03-SEM multiple, projection widths).
  (iii) The theory-doc band [1.71, 1.98] quote unchanged. Any
  mismatch → STOP (citation defect).
- **G1d.** Six-form table with nesting statement written before any
  fit; rule-22 sides; rule-24 generated tables. No feasibility gate:
  no new worlds — the estimand is form comparison on existing data
  and LOO is its guard (rule 25 note recorded).
- **G3d.** Stage estimates: part0 120 s, fit 240 s, finalize 60 s;
  2× stop-and-report.

### Routing (rule 16 — every combination to exactly one outcome)

| # | condition | outcome |
|---|---|---|
| 1 | any G0d mismatch | **STOP** (citation defect; no fit) |
| 2 | incumbents stand AND winner r² quiet | **FAMILY_STANDS** — M2 seals the M1c pair (F1e+F1 co-sealed per rule 26) |
| 3 | incumbents stand AND r² fires | **INCOMPLETE_UNREPAIRED** — M2 deferred; M1e (shape study) named |
| 4 | F0 wins AND r² quiet | **COMPLETED_IN_R** — T4 keeps the r-coordinate with an intercept; M2 seals F0 |
| 5 | Fφ wins AND r² quiet | **COORDINATE_RETYPED_TO_PHI** — the level law's second argument is state dynamics, not card readability; M2 seals Fφ |
| 6 | an extension wins AND r² fires | **COMPLETED_BUT_INCOMPLETE** — M2 deferred; M1e named |
| — | F0/Fφ tie (<5% LOO) | **CO_WINNERS** — both sealed in M2 (multiple predictions inside one hashed file, K2f precedent); verdicts co-adjudicated, disagreements SPLIT |
| — | L-3d disjoint | modifier **TAX_SHIFT** → M3's charter |

### Deliverables and budget

`scripts/run_suica_m4_m1d_form_completion.py`;
`results/m4_m1d_form_completion/` (gitignored);
`reports/SUICA_M4_M1D_FORM_COMPLETION_REPORT.md` (generated tables);
outcome appended HERE; one ledger row (EXPLORATORY); exactly ONE
commit `feat(m4-m): M1d — the completion and the coordinate —
<SLUG>`, never amended, never pushed; suite green first.
Artifact-space only: target < 10 min wall, every stage < 600 s.

### Outcome (appended 2026-08-11 by the executing agent; append-only)

**`COMPLETED_BUT_INCOMPLETE` — rule-16 cell 6** (an extension wins AND the r²
residual fires). L-1d **HOLD**, L-2d **F0**, L-3d **overlap**,
L-4d **fires**; modifier none. Report:
`reports/SUICA_M4_M1D_FORM_COMPLETION_REPORT.md`; harness
`scripts/run_suica_m4_m1d_form_completion.py`; artifacts
`results/m4_m1d_form_completion/`. **No new worlds**: M1c's persisted
3840-world corpus, its 20 cell means re-derived
round-trip and matched bit-exactly.

**G0d PASSES.** All 20 cell means AND SEMs re-derived from the rawest per-world
artifacts match M1c's persisted values bit-for-bit; 30 enumerated
adjudication citations, the Spearman vector, both L-4 readings, the share-.60
rise (full precision rounding to the adjudication's 2-dp quote) and the theory
band all verify. The four incumbents were PROVEN frozen in their M1c roles —
Part 0 imports the M1c harness and demands bit-exact agreement on every
parameter and SSE.

**L-1d HOLDS — the intercept is real.** `F0` (`field = c + lambda*r^q - kappa*V`) wins
leave-one-cell-out at **`0.0030682764618814033`** against the best incumbent
`F1e`'s `0.0031856515917748638` — a factor of
1.03825441786347. Full LOO ranking: {'F0': 0.0030682764618814033, 'F1': 0.003198131708377386, 'F1e': 0.0031856515917748638, 'F2': 0.0034019365713125944, 'F3': 0.003877604046883495, 'Fphi': 0.0032498223469787663}. In-sample RMSE:
{'F0': 0.0025054232543959215, 'F1': 0.0026264051166751978, 'F1e': 0.002621078709438027, 'F2': 0.002591249722764473, 'F3': 0.0033903747201612703, 'Fphi': 0.002538699623241356}. Appendix W.1 named the gap; one free intercept closes most of
it.

**L-2d answers `F0` — the coordinate is r, not φ.** `Fφ` does not merely lose to
`F0` (`0.0032498223469787663` vs `0.0030682764618814033`), it loses to the best incumbent as
well. Appendix X.4's occasion-structure hypothesis — the gauge as a consumer of
state dynamics rather than card readability — gets **no support** from this
corpus. **But the call is close and must not be over-quoted:** the two separate
by `0.000181545885097363` = 5.916868553169516% of the smaller LOO against a
5.0% CO_WINNERS bar — 0.92 percentage
points outside it. Under 5% this leg would have sealed both coordinates. `Fφ` is
outperformed, not refuted.

**L-3d overlap — κ's FIFTH appearance.** `0.7766770259880144`, CI `[0.7482226203832176, 0.8064115044591174]`,
overlapping M1c's `[0.7356727662590873, 0.7846243216827854]`; no TAX_SHIFT. At width
`0.05818888407589973` it is the ONLY sharply identified parameter in the winning
form — which is the next paragraph's subject.

**THE FINDING THE REGISTRATION DID NOT ASK FOR: the winning form cannot report
an exponent.** `F0`'s intercept buys its LOO win by trading against the power
term, and the three are jointly non-identified: `c` CI `[0.20818746052333, 1.6803368132111625]` (width
`1.4721493526878324`), `λ` CI `[-1.5059828481846496, -0.04256154549067277]` (width `1.4634213026939769`),
`q` CI `[0.021913588793404413, 2.6445200496694605]` — **width `2.622606460876056`**. And the point
estimates arrive from the opposite corner to M1c's: `λ = -0.055190882521519`
(NEGATIVE) with `q = 1.372031438858951` (POSITIVE) describes the SAME falling-in-r
field that M1c described as `q = -0.15040108849226472` with positive λ. **M1c's negative
exponent was the family's only way to bend the field downward in r without an
intercept.** Given the intercept, it re-parameterises. The monotone direction is
robust across every form tried; the exponent is not a structural constant of
this world, it is a coordinate on a ridge. **M1c's `q = -0.15040108849226472`
`[-0.18322395953281184, -0.11871900002844447]` should be re-scoped in the record accordingly** — the
`LEVEL_RESPONSE_DISSOCIATION` verdict survives, because it rests on the SIGN of
the r-dependence and on the V-shadow contrast, not on the exponent's value.

**L-4d FIRES — M2 is deferred, M1e is named.** Within-share r² residual
`-0.12563681892698172`, CI `[-0.1772060912696028, -0.07219090437007022]`, excluding zero and stable at B = 20000
(`[-0.17935555262608965, -0.07097803090981235]`, stable True). The intercept shrank it
36.90141692114467% from M1c's `-0.19911194958208703` — real progress, not closure.
Two concrete inputs for M1e: the leftover curvature is **within-share only**
(pooled r² quiet at `0.001054754288525179` `[-0.010339263255536255, 0.014815973510193174]`), and it survived
the intercept at reduced amplitude, so the missing term is a within-stratum
shape in r rather than a between-share effect. The φ² companion
(`-0.012377098889152529` `[-0.021250484030500706, -0.0035455613757951142]`, executor-added, routes nothing) also fires and
carries NO coordinate information: within a stratum r and φ are monotone
re-parametrisations, so curvature in one implies curvature in the other — the
coordinate question is settled across shares, which is what the F0/Fφ LOO
comparison does.

**Reading 1, the V-shadow — pre-signed positive, and decisively so.** Fitting
`field = λ·r^q` on M1c's own 20 cells with NO tax and NO intercept returns
**q_shadow = `2.24488769944643`**, CI `[2.1768337883424214, 2.318980336007031]` — positive, confirmed
(True), and *above* the response band [1.71, 1.98] on the high
side. The winner's own exponent on the same cells is `1.372031438858951`. Omit the
variance tax and the same data produce a large positive exponent: the
re-attribution recorded in appendix X, demonstrated in-corpus in one number. The
shadow fits badly on its own terms (RMSE `0.02157946817434354` against the
winner's `0.0025054232543959215`) — a demonstration, not a rival.

**Reading 2, the legacy retrodiction — and it beats K2f's own refit.** The
winner, with NO refit, predicts K2f's 26 legacy compiled rows at RMSE
`0.0059526106645589934` against the sealed T4 composite's `0.11259090547752257`
— a factor of **18.914542176909535**. It also comes in BELOW K2f's own refit LOO
of `0.0061559195350209` (True), which is the
stronger statement: parameters estimated on a decollinearized factorial transfer
to a different corpus on a different design, unadjusted, and beat what that
corpus achieved by fitting itself. Per-form: {'F0': 0.0059526106645589934, 'F1': 0.0056954308382002605, 'F1e': 0.005699989746733798, 'F2': 0.005604357102432524, 'F3': 0.006428578997454841, 'Fphi': 0.006280786394748213}. Scoped as
registered — same-instrument extrapolation across corpora, descriptive.

**Rule 26 fired on its first opportunity, exactly as enacted.** `F1e`
reaches the bootstrap set as runner-up with its ε bound ACTIVE, so its unbounded
relaxation F1 was co-adjudicated AUTOMATICALLY rather than by the tie rule's
luck — the failure mode M1c's non-blocking candidate flagged. F0 and Fφ carry no
declared bounds; the numerical-limit test (RN-M1D-5: |param| >= 1e3 or
termination at max_nfev) was checked on every bootstrapped form and did not fire
(largest |parameter| `1.372031438858951`). **Tie rule** also fired
(`0.00011737512989346043` = 3.8254417863469983% between F0 and F1e); every
verdict agrees across the pair — L-3d overlap under both, and L-4d fires under
both (M1c measured F1e's own within-share r² at `-0.19911194958208703`
`[-0.2879978718649799, -0.10706476050455438]` under the same estimator) — so nothing reports SPLIT and the
routing is unchanged whichever member is read. **Rule 13** triggered on both
proximities; the B = 20000 re-run left L-3d and L-4d unchanged.

**Anomalies, with timing.** No worlds were drawn, so the hypothesis-relevant
boundary is the `fit` stage and Part 0 is pure verification of published
numbers; every RN note was pinned there. **A-1/A-2 (before Part 0):** the
inherited CPython 3.12.12 environment from `requirements-lock-main.txt`
(numpy `2.4.4`, pandas `3.0.2`, scipy `1.17.1`), and no
`timeout(1)` on macOS so each stage ran as its own foreground command. **A-3 (at
the fit):** the winner's joint non-identification — the headline caveat above,
found when its bootstrap CIs were first read; it changes no verdict because
L-4d defers M2 on independent grounds. **A-4 (at the fit):** the CO_WINNERS call
landing 0.92 points outside its bar. **A-5 (at the fit):**
rule 26 firing on its first opportunity — recorded because a rule enacted one leg
earlier changed this leg's bootstrap set automatically. **A-6:** no stage near
its 2× threshold (Part 0 `1.1947739124298096` s against 120 s).

**Registration-defect candidate: ONE, non-blocking, and it is the same shape as
the ones that bought rules 25 and 26.** Cells 4 and 5 route a winning extension
straight to "M2 seals F0 / seals Fφ" **with no identification requirement on the
sealed parameters**. Had the r² probe come back quiet, this leg would have
routed to `COMPLETED_IN_R` and handed M2 a form whose exponent interval is
`[0.021913588793404413, 2.6445200496694605]` — a seal on a ridge. Nothing turned on it because L-4d fired
independently. But the routing selects on predictive accuracy (LOO) while the
downstream consumer (a prospective seal) needs identified parameters, and those
are different properties — a gate that does not check what its consumer
requires. A successor should either add an identification clause to the sealing
cells or state explicitly that a seal may issue on a non-identified
parameterisation.

**Line state.** M2 deferred pending M1e. M3 is the best-supplied question in the
line: κ now has five independent appearances and is the only sharply identified
parameter in the winning form.

### Planner adjudication of M1d (2026-08-11, appended after the run) — THE EXPONENT DISSOLVES INTO A SLOPE; THE SHAPE QUESTION REPLACES THE COORDINATE QUESTION

**COMPLETED_BUT_INCOMPLETE accepted.** L-1d HOLD (F0 beats all four
incumbents; LOO 0.0030682764618814033), L-2d = F0 (Fφ loses even to
the incumbents — the naive state-dynamics representation is NOT the
better cross-share account), L-3d overlap (κ fifth appearance,
0.7766770259880144 [0.7482226203832176, 0.8064115044591174] — "the
only sharply identified parameter", the executor's words, correct),
L-4d FIRES (within-share r² = −0.12563681892698172
[−0.1772060912696028, −0.07219090437007022], stable at B = 20000) →
**M2 stays deferred; M1e is chartered below.**

**The scientific correction this leg forces (theory appendix Y).**
F0's (c, λ, q) are JOINTLY NON-IDENTIFIED — q [0.021913588793404413,
2.6445200496694605], λ entirely negative, c wide — and λ<0 with q>0
describes the SAME falling-in-r field that M1c's F1 described as λ>0,
q<0. **M1c's "negative exponent" was a parameterisation artifact of
the missing constant.** What survives, parameterisation-free: the
SIGN OF THE SLOPE — ∂field/∂r < 0 at fixed V (the model-free 4.03-SEM
contrast; both parameterisations agree). The DISSOCIATION verdict
stands (it was slope-sign vs the response attribution); the exponent
VALUE is withdrawn as a structural claim by dated note. The
level-law's identified content on this corpus is: a level constant, a
weak negative r-slope, and the tax.

**What strengthened.** (i) The V-shadow demonstration confirmed its
pre-signed flip: q_shadow = 2.24488769944643 [2.1768337883424214,
2.318980336007031] — omit the tax and the exponent snaps positive,
ABOVE the response band; appendix X.2's re-attribution now has its
one-number in-corpus proof. (ii) Legacy transfer: with NO refit, F0
retrodicts K2f's 26 rows at 0.0059526106645589934 — 18.91× better
than the sealed form and below K2f's own refit LOO; every M1d form
lands 0.0056–0.0064 — the constant-minus-tax-with-weak-negative-slope
structure is CROSS-CORPUS. (iii) Rule 26 fired on its first
opportunity, automatically. (iv) The five κ routes now read 0.715 /
0.722 / 0.750 / 0.760 / 0.777 — chainwise-overlapping with a mild
upward drift in level-space routes; recorded UNADJUDICATED as M3's
opening observation.

**Defect #45 (mine; rule 27 enacted).** M1d's cells 4/5 routed a
winning extension straight to "M2 seals it" with NO identification
requirement on the sealed parameters — had r² come back quiet, M2
would have been handed a ridge (q width 2.62). Same genus as the
defects behind rules 25/26: selection by LOO, consumption requiring
identification. **Rule 27:** a route that hands a fitted object to a
downstream consumer (a seal, an adoption, a cross-leg comparison)
carries an explicit identification budget on every parameter the
consumer will quote; selection wins (LOO or otherwise) alone never
qualify an object for consumption.

---

## M4-M1e — the shape: additive or r-mediated (artifact-space)

**REGISTERED 2026-08-11, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. NO NEW WORLDS: M1c's persisted
corpus, the 20 cell means re-derived bit-exactly. Optimizer pins,
LOO-cell selection, bootstrap B = 2000/20000, tie rule 5%, rules
26/27 in force.

### Question

Does the level field SEPARATE — field(share, φ) = share-margin +
φ-margin — or does the r-channel carry the cross-structure? These are
distinguishable HERE because r(share, φ) is strongly non-additive:
the within-share r-spans run 0.04714885082631204 →
0.20320393707216905 (4.31× by share), so any r-mediated field with a
material r-coefficient must show share×φ interaction, while a truly
additive field kills r-mediation. If additivity wins with the share
margin linear in V, the level law's arguments are (V, φ) as SEPARATE
channels — the tax channel and a state-dynamics channel — and card
readability was never an argument, only a correlate.

### Part 0 (before any fit)

- **G0e.** (i) Re-derive the 20 cell means bit-exactly. (ii) Verify
  every M1d number quoted in the adjudication above at full precision
  against `results/m4_m1d_form_completion/` (F0 params and all three
  CIs, LOO table, tie margins 0.00011737512989346043 and the 5.92%
  Fφ separation, κ CI, r² CIs at both B, q_shadow triple, legacy
  retrodiction values). (iii) Quote-check [1.71, 1.98] unchanged.
  Mismatch → STOP.
- **Model-free monotonicity table (Part-0 object, before fits):** the
  four within-share extreme contrasts field(φ=.98) − field(φ=.05)
  with pooled SEs — the parameterisation-free record of the slope
  sign per share.

### Models (five; pinned)

- **E-add:** field = α_s + g_φ (free margins, sum-to-zero pinning
  stated; 8 identifiable params);
- **E-tax-add:** field = c − κ·V + g_φ (share margin forced through
  the tax; 7 params);
- **E-rlin:** field = α_s + s·r (5);
- **E-rq:** field = α_s + λ·r^q (6; starts: λ ∈ {−0.5, −0.055, 0.05,
  0.5}, q ∈ {0.5, 1.0, 1.372031438858951, 2.0, 3.0});
- **F0 (frozen incumbent):** M1d's winner as the baseline to beat.

### Leans (sides declared, rule 22)

- **L-1e [.45]:** an additive form (E-add or E-tax-add) wins LOO
  outright over all r-mediated forms AND F0. Complement: r-mediated
  wins [.35]; F0 stands [.20].
- **L-2e [.60]:** the share margin is the tax — E-tax-add within 5%
  LOO of E-add AND its κ CI overlaps M1d's [0.7482226203832176,
  0.8064115044591174] (the sixth appearance).
- **L-3e (reading → routing):** the winner's within-share r²-probe —
  quiet ⟹ SHAPE settled; fires ⟹ shape remains open.

### Rule-27 identification budgets (for any object routed to M2)

κ width ≤ 0.15; c width ≤ 0.05; each g_φ point width ≤ 0.01; s width
≤ 50% of |point|; E-rq's (λ, q) jointly: q width ≤ 1.0. An object
missing its budget is NOT sealable regardless of routing.

### Routing (rule 16)

| # | condition | outcome |
|---|---|---|
| 1 | any G0e mismatch | **STOP** (citation defect) |
| 2 | additive wins AND winner's probe quiet AND budgets met | **ADDITIVE_SHAPE_SETTLED** — r-mediation dead at level on this family; the arguments are (V, φ); M2 seals the winner at exterior-share × interior-φ cells (a pinned-form φ-extrapolation may ride as a SECONDARY sealed prediction) |
| 3 | r-mediated wins AND probe quiet AND budgets met | **R_MEDIATED_SETTLED** — M2 seals it |
| 4 | F0 stands (nothing beats it) | **NO_BETTER_SHAPE** — the M1-series closes at its measured limit: identified = level band + negative slope + tax; shape = named open; M2 re-charters on the SCOPED object (κ-channel + model-free cell predictions), not on a shape |
| 5 | any winner AND probe fires | **SHAPE_OPEN_NAMED** — same scoped-M2 route as cell 4 |
| 6 | routing would seal but budgets unmet | **IDENTIFIED_INSUFFICIENTLY** — scoped-M2 route |
| — | L-2e κ disjoint | modifier **TAX_SHIFT** → M3 |

Tie (<5% LOO) between an additive and an r-mediated form:
**REPRESENTATION_TIE** — both reported, verdicts co-adjudicated,
disagreement SPLIT; routing takes the SCOPED route (a tie on
representation is not a settled shape).

### Deliverables and budget

`scripts/run_suica_m4_m1e_shape.py`; `results/m4_m1e_shape/`
(gitignored); `reports/SUICA_M4_M1E_SHAPE_REPORT.md` (generated
tables); outcome append HERE; one ledger row (EXPLORATORY); exactly
ONE commit `feat(m4-m): M1e — the shape — <SLUG>`, never amended,
never pushed; suite green first. Artifact-space: target < 10 min,
stages < 600 s.

### Outcome (appended 2026-08-11 by the executing agent; append-only)

**`IDENTIFIED_INSUFFICIENTLY` — rule-16 cell 6** (routing would seal but the rule-27
budgets are unmet), modifier **`TAX_SHIFT`**. L-1e **MISS (r-mediated wins)**, L-2e
**MISS**, L-3e **quiet**. Sealable: **False**. Report:
`reports/SUICA_M4_M1E_SHAPE_REPORT.md`; harness
`scripts/run_suica_m4_m1e_shape.py`; artifacts `results/m4_m1e_shape/`. No new
worlds; M1c's 20 cell means re-derived bit-exactly.

**G0e PASSES**: all 20 cell means and SEMs bit-exact from the rawest per-world
artifacts, 32 M1d adjudication citations bit-exact, the theory
band unchanged, and **F0 re-fitted here reproduces M1d's fit bit-exactly** — the
frozen incumbent really is the same object.

**The Part-0 model-free table decided the leg before any model was fitted.**
Within-share extreme contrasts field(φ=0.98) − field(φ=0.05) with pooled SEs:
[0.0012820301142057455, 0.010391443071199338, 0.01143698383536769, 0.009688611655304012] against SEs [0.002662818947918231, 0.0025394961062406947, 0.0024676318457931964, 0.002401953516046843], i.e. [0.4814559830318264, 4.091931090448552, 4.634801522303819, 4.03363828257993] in SE units.
All 4 are positive; 3/4 exceed 2 SE. The
exception is share 0.10 at **0.4814559830318264 SE** — flat — and share 0.10 is
exactly where the r-span is smallest (0.04714885082631204 against
[0.04714885082631204, 0.10864636443860598, 0.15863752656482977, 0.20320393707216905]). **A strictly additive field predicts the same contrast at
every share; this one does not, and it fails precisely where r stops moving.**

**L-1e MISSES — the field does not separate.** LOO across the five:
{'E-add': 0.002706675155983591, 'E-rlin': 0.0026942709003566117, 'E-rq': 0.0024079360107794926, 'E-tax-add': 0.003579020306723271, 'F0': 0.0030682764618814033}. Best additive E-add at `0.002706675155983591`
loses to best r-mediated E-rq at `0.0024079360107794926`; F0 at
`0.0030682764618814033`. The winner is **E-rq** (`field = alpha_s + lambda*r^q`) at
`0.0024079360107794926`, ahead of E-rlin by 11.89129978102812% — outside the
tie band, and both are r-mediated, so no REPRESENTATION_TIE. The .45 additive
prior does not pay; the .35 r-mediated complement does. Note `E-add`
(`0.002706675155983591`) and `E-rlin` (`0.0026942709003566117`) are nearly level — the
additive representation is not bad, it is beaten, and beaten decisively only by
the model carrying BOTH free share margins and an r-power.

**L-3e QUIET — the shape probe goes silent for the first time in three legs.**
Under the winner the within-share r² residual is `-0.007427848773582237`, CI
`[-0.03672898793443594, 0.018353437794254]` — containing zero. M1c fired at −0.19911194958208703, M1d at
−0.12563681892698172, M1e is quiet. Free share margins plus an r-power absorb
the leftover curvature. **The M1-series' shape question closes.**

**And nothing is sealable, because rule 27 says so.** The winner's exponent is
`q = 3.863625377453229`, CI `[2.0529339475688055, 5.921369905297595]`, **width `3.86843595772879` against the
registered `1.0` budget — over by a factor of
3.86843595772879**. Routing is cell 6, IDENTIFIED_INSUFFICIENTLY, the
scoped-M2 route. This is rule 27 doing exactly the job defect #45 bought it for:
under the pre-rule-27 routing this leg would have read "r-mediated wins, probe
quiet → R_MEDIATED_SETTLED, M2 seals it" and handed M2 an exponent spanning
`[2.0529339475688055, 5.921369905297595]`. The object is not uniformly vague — the share margins run
[0.015203974958038241, 0.015209021910085607, 0.01511553251962218, 0.012928411315645191] wide and λ = `-0.057625974791364554` `[-0.0843564122153383, -0.042724477794351616]` is NEGATIVE and
well identified at width `0.04163193442098668` — it has ONE bad coordinate and the
budget catches precisely that one. For contrast, the REJECTED `E-tax-add` meets
every budget it has: identifiability and correctness are independent properties,
and this leg exhibits both directions in one table.

**L-2e MISSES on both clauses, and this is the finding to carry.** (i) Forcing
the share margin through the linear tax costs 32.229399557283564% of LOO —
`E-tax-add` at `0.003579020306723271` against a bar of `0.002842008913782771` — making it
**the worst of the five models**. The four share margins are not on a straight
line in V; they carry curvature that −κ·V cannot represent. (ii) Its κ is
`0.6761549415814`, CI `[0.6619291032569563, 0.6901486195533926]`, **disjoint BELOW** M1d's
`[0.7482226203832176, 0.8064115044591174]`. There is no sixth appearance (False); there is a
shift, and the **`TAX_SHIFT` modifier fires to M3**.

**The qualification M3 must read with it.** The κ that shifted belongs to the
model this same leg ranks LAST, and the WINNING model contains no κ at all — its
share direction is four free margins and the tax is implicit in them. So the
finding is not "κ is 0.6761549415814". It is: **κ's point value depends on the
representation it is embedded in, and the dependence
([0.715, 0.722, 0.75, 0.76, 0.777] previously, `0.6761549415814` here) exceeds the width of
any of its own intervals.** M3 was chartered to ask whether κ is ONE instrument
constant; this leg supplies a sharp prior answer and a design instruction —
**vary the surrounding form deliberately rather than accumulate more appearances
within one family.**

**Method notes.** Three of the five models are linear in their parameters and
were ALSO solved in closed form by lstsq; the largest optimizer-vs-lstsq
disagreement across all three is `3.311532159600006e-10` (RN-M1E-3), so no routing
decision rests on optimizer behaviour. `E-add` and `E-tax-add` return IDENTICAL
g_φ margins (True) — expected on a balanced design, where the
sum-to-zero φ margins are orthogonal to both the share margins and V, so the two
models differ ONLY in how they represent the share direction and their LOO gap
measures exactly that. `E-add`'s margin is [-0.00482194717663085, -0.002949742497307577, 0.0008027742599145105, 0.0035910954216355657, 0.003377819992388351] across φ
[0.05, 0.3, 0.6, 0.85, 0.98], span `0.008413042598266415`. Rule 13 did not trigger
(False); rule 26's bound trigger cannot fire (no model declares
bounds) and the numerical-limit surveillance was clean (False).

**Anomalies, with timing.** No worlds drawn; Part 0 is verification plus the
model-free table, and every RN note was pinned there before any model was
fitted. **A-1/A-2 (before Part 0):** the inherited CPython 3.12.12 lock
environment (numpy `2.4.4`, pandas `3.0.2`, scipy `1.17.1`);
no `timeout(1)` on macOS. **A-3 (Part 0, before any fit):** the registration's
parameter counts are inconsistent — "8 identifiable" for E-add is post-pinning,
"7 params" for E-tax-add is its RAW count and its identifiable count under the
same pinning is 6; resolved by RN-M1E-1 before any fit and reported both ways.
**A-4 (at the fit):** κ shifts on a REJECTED representation — reported as
registered, qualified above, neither suppressed nor amplified. **A-5 (at the
fit):** the winner misses its only budget by 3.86843595772879x — the
routing-determining number. **A-6:** no stage near its 2x threshold (Part 0
`4.5435521602630615` s against 120 s).

**Registration-defect candidates: four, all non-blocking.** (1) **The rule-16
table OVERLAPS** — (F0 stands AND probe fires) matches cells 4 and 5, and
(winner AND probe fires AND budgets unmet) matches cells 5 and 6; precedence was
pinned in Part 0 as RN-M1E-5 and did not bite here (probe quiet, budgets unmet →
only cell 6 matched), and the overlap is immaterial to consequence since cells
4/5/6 share the scoped-M2 route — but it is a rule-15/16 partition failure.
(2) **Rule 27's budget list is incomplete** — the share margins α_s and E-rq's λ
are consumed by any seal and carry no declared budget; they are well identified
here, so nothing turned on it, which is the same "nothing turned on it" that
preceded defect #45. (3) **The `TAX_SHIFT` modifier is not conditioned on its
host model being competitive** — it fires on E-tax-add's κ regardless of that
model ranking last, and the evidentiary weight of a shift measured on a rejected
representation differs materially from one measured on a winner; the
registration does not say which it means. (4) The parameter-count wording above.

**For M2, and for the line.** M2 must take the scoped route: the κ-channel and
model-free cell predictions, not a shape. Worth the planner's attention —
**this is the SECOND CONSECUTIVE leg whose winner carries a non-identified
exponent while its slope is identified.** Appendix Y's reading now has three
independent parameterisations behind it (M1c's λ>0/q<0, M1d's λ<0/q>0, M1e's
λ<0/q≈3.863625377453229), and a successor should consider registering the SLOPE itself —
∂field/∂r at fixed V, or the within-share contrast, which is already the
model-free Part-0 object of this leg — as the sealable quantity, rather than any
exponent in any parameterisation.

### Planner adjudication of M1e (2026-08-11, appended after the run) — THE SHAPE IS REAL, THE SEAL WAS RIGHTLY BLOCKED, AND κ BECOMES REPRESENTATION-INDEXED

**IDENTIFIED_INSUFFICIENTLY accepted; the TAX_SHIFT modifier is
RETYPED under new rule 28 (below) as REPRESENTATION-CONDITIONED.**
L-1e MISS — the field does NOT separate additively (E-rq wins at
0.0024079360107794926, 11.89% clear of E-rlin; best additive
0.002706675155983591): my .45 additive lean lost, on the record.
L-3e QUIET for the first time in three legs (r² −0.007427848773582237
[−0.03672898793443594, 0.018353437794254]): the within-share shape is
carried by a steep negative r-power. Rule 27 then did exactly what it
was enacted for: the winner's exponent (3.863625377453229
[2.0529339475688055, 5.921369905297595], width 3.87× budget) is not a
quotable object, so no seal — while the rejected tax-additive model
met every budget, the cleanest demonstration yet that **selection and
identification are different licenses.**

**The shape is quantitatively real (planner verification arithmetic,
run at adjudication).** Propagating the winner's (λ, q) =
(−0.057625974791364554, 3.863625377453229) through λ·Δ(r^q) over each
share's φ-extremes reproduces the Part-0 model-free monotonicity
table: predicted +0.00545 / +0.00991 / +0.01097 / +0.00866 against
measured +0.00128 / +0.01039 / +0.01144 / +0.00969 (SEs ≈ 0.0025) —
three of four inside 0.5 SE, INCLUDING the non-monotone hump (peak at
share .40, downturn at .60, which a linear-in-r channel cannot
produce); the share-.10 cell sits −1.57 SE (flat where the r-span is
smallest). The r-channel lives at HIGH r: steep power ⟹ the level
price concentrates where cards are most readable. Named (not
claimed): the READABILITY PENALTY.

**κ is representation-indexed (the leg's deepest finding).** Within
each representation κ is sharply identified; across them it spans
0.6761549415814 [0.6619291032569563, 0.6901486195533926]
(tax-forced-additive, the leg's WORST model, +32.23% LOO) to 0.777
(M1d F0). The executor's candidate (3) is upheld: an unconditioned
cross-leg κ comparison on a rejected representation would have
polluted M3's charter. **Rule 28 (enacted, paid by #48):** cross-leg
parameter comparisons (leans or modifiers) are licensed only on
representations competitive in the comparing leg (the winner or
within its tie band); on any other representation the comparison is
typed REPRESENTATION-CONDITIONED and adjudicates nothing. M3's
question is hereby REFINED: not "is κ one number" but "in WHICH
representation class is the tax an invariant — and is the V-margin
even linear" (the free share-margins α = [0.185608, 0.145649,
0.109349, 0.066676] at V = [0.03, 0.075, 0.12, 0.18] sit CONVEX
below their chord by ~2–3 cell-SEMs; named for M3).

**Defects (mine).** **#46 (rules 15/16 violated):** M1e's routing
table overlapped (cells 4/5 and 5/6 both reachable by one outcome) —
the enumeration was not a partition; immaterial only because the
overlapping cells shared the scoped-M2 route; the executor's
precedence pin (RN-M1E-5) was the correct in-leg repair. Enforcement
extension of the #43 convention: the planner VERIFIES
disjoint-and-covering mechanically before committing a routing table.
**#47 (rule 27 misapplied at first use):** the budget list omitted
α_s and λ — parameters any seal consumes; the budget attaches to
WHATEVER the consumer quotes, parameters or predictions. **#48 → rule
28** (above). Candidate (4) (parameter-count wording) noted,
unnumbered — RN-M1E-1 pinned it correctly.

**Route.** Cell 6's scoped-M2: the ridge lives in PARAMETER space,
not prediction space over the trained r-window [0.4541409476972356,
0.8189581462487876] — so the sealable objects are PREDICTIONS at
configurations whose realized r stays interior: α-free within-share
contrasts at exterior share, full levels at interior-new φ, and the
rejected model's extrapolation as a pre-signed stress reading. M2
registered below, at last.

---

## M4-M2 — the scoped extrapolation seal (predict, hash, then run)

**REGISTERED 2026-08-11, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. The K2f Stage-2 pattern governs:
predictions written AND HASHED before any fresh world exists
(ordering enforced in code, G1f-style permit gate; the rule-17 pilot
runs AFTER the stamp, RN-K2F-4 precedent); salt embedded INSIDE the
sealed bytes (D3 convention). Instrument, seeds policy, σ_w =
0.026889438327132725, and all M-line conventions inherited. Salts
`m4m2-world` / `m4m2-pilot`, master_seed 20260811, 192 worlds/cell.

### Design (planner arithmetic run at registration — #43 convention; executor reproduces bit-exactly)

Trained r-window [0.4541409476972356, 0.8189581462487876]. New cells
(5, all fresh; 960 worlds):

| cell | share | φ | r (planner-computed) | V | role |
|---|---|---|---|---|---|
| C1 | 0.70 | 0.05 | 0.5967380569813433 | 0.21000000000000005 | P1 contrast side; share EXTERIOR (envelope top 0.6634207990183637) |
| C2 | 0.70 | 0.60 | 0.5197539933932338 | 0.21000000000000005 | P1 contrast side; P4 stress level |
| C3 | 0.40 | 0.45 | 0.7131718346406168 | 0.12000000000000004 | P2 level; φ interior-new |
| C4 | 0.25 | 0.05 | 0.785015540293945 | 0.07500000000000002 | P3 contrast side; config duplicates an M1c cell on a FRESH salt → the seed-replication reading |
| C5 | 0.25 | 0.995 | 0.6701862156520305 | 0.07500000000000002 | P3 contrast side; φ EXTERIOR (above .98) |

Every realized r is interior to the window (that is the scope).
Share-.70 cells at φ ≥ 0.85 were REJECTED at registration because
their r exits the window (0.44410111322601925 / 0.384884059649622 —
computed, excluded, disclosed).

### The sealed predictions (computed in Part 0, then hashed, then worlds)

The predictor: E-rq refit bit-identically from M1c's persisted 20
cell means (deterministic; G0 verifies the point parameters equal
M1e's persisted values bit-exactly), with the winner's within-cell
world-block bootstrap (B = 2000, pinned seed) recomputed and each
draw propagated through the formulas. Band for each prediction =
[boot 2.5% − 2·SE_meas, boot 97.5% + 2·SE_meas], SE_meas = σ_w/√192
for levels and √2·σ_w/√192 for contrasts; two-sided containment
(rule 22).

- **P1 (contrast, α-free):** field(C2) − field(C1) = λ·(r_C2^q −
  r_C1^q). Planner sanity value ≈ +0.0032. **L-3m2 [.50]** inside.
- **P2 (level):** α(0.40) + λ·r_C3^q. Planner sanity ≈ +0.0937.
  **L-2m2 [.60]** inside.
- **P3 (contrast, the flagship — high-r region where the channel
  lives, φ-exterior):** field(C5) − field(C4) = λ·(r_C5^q − r_C4^q).
  Planner sanity ≈ +0.0103 (≈ 3.8 contrast-SE). **L-1m2 [.60]**
  inside.
- **P4 (stress reading, NO gate, pre-signed):** the REJECTED
  tax-additive model's level at C2: c − κ·V(0.70) + g_φ(0.60),
  planner sanity ≈ +0.0382. Pre-signed lean [.55]: measured ABOVE
  (the free margins are convex-below-chord, so the linear-tax
  extrapolation over-falls). Reported either way; feeds M3.
- **Replication reading (NO gate):** |mean(C4) − M1c's persisted
  (0.25, 0.05) cell mean| vs 2·√2·SEM → quiet [.85] or
  SEED_INSTABILITY flagged.

### Rule-27 budgets (on the CONSUMED objects — the predictions; #47's repair)

Band total widths, computed in Part 0 BEFORE the stamp: P1 ≤ 0.04;
P2 ≤ 0.05; P3 ≤ 0.04. A prediction whose band exceeds its budget is
**VOID_FOR_WIDTH** before measurement (a seal too wide to miss seals
nothing); voids are disclosed and excluded from the promotion count.

### Gates

- **G0m2 (bit-exact).** (i) The design table above reproduced from
  the pinned maps; (ii) every M1e number quoted in the adjudication
  above verified at full precision against `results/m4_m1e_shape/`
  (winner params + CIs, LOO table, monotonicity table, α vector, g_φ
  vector, tax-add κ CI); (iii) M1c cell means re-derived bit-exact;
  (iv) σ_w and the r-window endpoints against their persisted
  sources; (v) the E-rq refit's point parameters equal M1e's
  persisted values bit-exactly. Mismatch → STOP.
- **G1m2 (ordering, enforced not asserted).** K2f's G1f pattern: the
  permit to generate ANY fresh world (pilot included) is issued only
  by re-reading predictions.sha256 from disk and re-hashing
  predictions.json; zero pre-stamp generations; ordering log
  persisted.
- **G2m2 (pilot, AFTER the stamp).** 4 worlds each at C1 (share
  exterior) and C5 (φ exterior), salt `m4m2-pilot`:
  finite/non-saturated/nonzero-variance only. Failure → the leg
  stops with the predictions on the record unresolved (RN-K2F-4's
  accepted cost, stated).
- **G3m2.** Sides declared (all containments two-sided); stage
  estimates: part0 300 s (refit + bootstrap + stamp), pilot 30 s,
  worlds 3 chunks (2/2/1 cells) × ~420 s, measure 120 s, finalize
  60 s; 2× stop-and-report.
- **G4m2.** Routing table below reproduced verbatim; planner has
  verified it is disjoint and covering (#46's enforcement); rule 24
  generated tables.

### Routing (rule 16 — disjoint and covering, planner-verified)

| # | condition | outcome |
|---|---|---|
| 1 | any G0m2 mismatch | **STOP** (citation defect; nothing sealed) |
| 2 | any band budget exceeded in Part 0 | that prediction **VOID_FOR_WIDTH**; continue with the rest; if fewer than 3 predictions remain valid, the promotion cell is unreachable (best available = cell 5 grades) |
| 3 | pilot regime failure | **UNRESOLVED_SEAL** — predictions stand on the record unmeasured; leg ends |
| 4 | 3 valid predictions AND 3/3 inside | **LEVEL_LAW_PREDICTIVE_SCOPED** — the scoped level law (free share-margins + steep negative r-power, r interior) is graded PREDICTIVE in its scope: sealed-then-hit at share-exterior and φ-exterior configurations; the scope IS the claim |
| 5 | exactly 2 of the valid predictions inside | **BOUNDARY_NAMED** — the missing prediction names the boundary of validity; theory note required |
| 6 | ≤ 1 of the valid predictions inside | **NO_TRANSFER** — the shape does not leave its corpus; the line closes at the measured limit |
| — | P4 above/below/inside its own reported ±2·SE_meas of the tax-linear value | modifier **STRESS_{ABOVE,BELOW,MET}** (pre-signed ABOVE) — feeds M3 |
| — | replication reading exceeds 2·√2·SEM | modifier **SEED_INSTABILITY** |

### Deliverables and budget

`scripts/run_suica_m4_m2_scoped_seal.py`; `results/m4_m2_scoped_seal/`
(gitignored); `reports/SUICA_M4_M2_SCOPED_SEAL_REPORT.md` (generated
tables); outcome append HERE; one ledger row (EXPLORATORY); exactly
ONE commit `feat(m4-m): M2 — the scoped extrapolation seal — <SLUG>`,
never amended, never pushed; suite green first. 960 worlds + 8 pilot;
target < 30 min wall, every stage < 600 s.

### Outcome (appended 2026-08-11 by the executing agent; append-only)

**`LEVEL_LAW_PREDICTIVE_SCOPED` — rule-16 cell 4; modifier(s) `['STRESS_ABOVE']`.**
**3/3 sealed predictions inside their bands**; voided
for width: none. Report:
`reports/SUICA_M4_M2_SCOPED_SEAL_REPORT.md`; harness
`scripts/run_suica_m4_m2_scoped_seal.py`; artifacts `results/m4_m2_scoped_seal/`.
960 fresh worlds plus 8 pilot.

**The ordering is the leg's spine and it is ENFORCED, not asserted.**
predictions.json (salt embedded inside the sealed bytes, D3) hashed
`d03e180919e2e2b1f08c7bde77c835d48b8c59177220085f1de1d39765f46ef2` at `2026-08-11T06:09:46.594113+00:00` with **0 fresh-world
generations in existence** — the pilot included, per RN-K2F-4. The permit for
the first world was issued `176.076157` s later by re-reading that
hash off disk and re-hashing the file to a match; guards wrap
build_k2b_world / run_field_world / emit_panel on **3** reachable
k2b instances (9 entry points).

**G0m2 PASSES on all five clauses**, including the one that matters most for a
seal: the E-rq refit from M1c's persisted cell means reproduces M1e's persisted
winner **bit-exactly**, so the predictor is the object the previous leg selected
and not a re-estimate of it. The planner's design table and the cells REJECTED
at registration (share .70 at phi >= 0.85, whose realized r exits the window)
both reproduce bit-exactly. 23 exact M1e citations plus the
rounded quotes, M1c's 20 cell means, sigma_w and the r-window all verify.

**The sealed predictions and what they met** (bands =
[boot 2.5% - 2*SE_meas, boot 97.5% + 2*SE_meas]; SE_meas = `0.00194057805706599`
level, `0.0027443918071463533` contrast, from sigma_w `0.026889438327132725`):

- **P2 (level, phi interior-new .45)** — predicted `0.09373871103378001`, band
  `[0.08823967502779759, 0.09925314694437559]`, measured **`0.09243571683982166`** CI `[0.08911835139816911, 0.09586975298752529]`, signed error
  `-0.0013029941939583511`, at -0.23801652307153248 of centre-to-edge. **INSIDE**, the cleanest
  of the three.
- **P3 (contrast, THE FLAGSHIP — phi EXTERIOR at .995, high-r where the
  readability penalty lives)** — predicted `0.010341381827303441`, band
  `[0.001953199000512147, 0.01853930794939698]`, measured **`0.007195006853810276`** CI `[0.0020157045910120137, 0.012381011229066962]`, error
  `-0.003146374973493165`, at -0.3679279607468678. **INSIDE.** The shape was fitted on
  phi <= 0.98 and asked to reach past it; it reached.
- **P1 (contrast, share EXTERIOR at .70) — INSIDE, and the weakest of the
  three.** Predicted `0.003242277707985443`, band `[-0.0035678414719505433, 0.010029212745051013]`, measured
  **`0.009126239258272953`** CI `[0.005255150305427958, 0.013425943230790644]`: inside, but at **0.8671810125388784** of the
  way to the upper edge, signed error `0.00588396155028751` = 2.1439947222425624
  contrast-SE_meas, and a measured value 2.814761744743775x the predicted
  point. A hit that would not have survived a much tighter band, stated because
  a seal that reports only its comfortable hits is not a seal.

**All three bands came in under their rule-27 budgets** (P1 `0.013597054217001556` <=
`0.04`; P2 `0.011013471916578005` <= `0.05`; P3
`0.016586108948884834` <= `0.04`), so nothing was VOID_FOR_WIDTH and the
promotion cell stayed reachable — defect #47's repair working as intended, the
budget now attached to what the consumer quotes. The bootstrap is the method's
point: M1e's winner carries a (lambda, q) RIDGE, so each of 2000 draws re-fits
E-rq on that draw's cell means and the SAME draw's (alpha, lambda, q) is pushed
through every prediction, so the bands inherit the parameter covariance
(RN-M2-3).

**P4, the pre-signed stress reading (no gate): `STRESS_ABOVE`, confirmed
(True).** The REJECTED tax-additive model extrapolates to
`0.03823746224897045` at C2 with a +-2*SE_meas envelope `[0.03435630613483847, 0.04211861836310243]`; measured
`0.04354391388413511` CI `[0.04031779189040825, 0.046724068686637095]` — ABOVE, at 1.3672347823997413 of the half-width,
error `0.005306451635164661`. The planner's reasoning (free share margins sit
convex-below-chord, so a linear-tax extrapolation must over-fall past the trained
shares) is vindicated at share .70, well outside the envelope. **This is a
second, independent line of evidence that the V-margin is not linear**, and it
goes to M3 beside M1e's kappa representation-indexing.

**The replication reading (no gate): QUIET, no SEED_INSTABILITY.** C4 duplicates
an M1c configuration on a FRESH salt: `0.12239759528671845` against M1c's persisted
`0.12162744485545209`, delta `0.0007701504312663671` against a bar of `0.005647466456046939`
(literal 2*sqrt(2)*SEM_C4, the gating reading) or `0.005346994263573509`
(exact two-sample) — quiet under both, which agree (True). The
instrument reproduces across independent seed streams at 192 worlds; the M-line's
cell means are a property of the design, not of the draw.

**G2m2, and the one judgement that changed this leg's outcome — stated first,
not buried.** The pilot first ran with the check written as "strictly inside
(0, 1)", the form K2f's G2f, M1b's G2m' and M1c's smoke all used. Under that form
C1 FAILS — one of its four pilot worlds reads
`-0.0007988006295671071` — and the leg ends at **UNRESOLVED_SEAL** with the
predictions unmeasured. The registration's word is **"non-saturated"**, not
"positive". Source check: `recovery_b_only` is a weighted mean of
`_matrix_cosine` (scripts/run_suica_m4_e1_convention_gap.py:250-264), so the
statistic's range is [-1, 1] and **zero is its NULL, not its floor** —
saturation means abs(value) -> 1. The "(0,1)" form is an UNREGISTERED import,
harmless in three prior legs only because their fields sat far above zero; C1 is
share .70 at V = 0.21, the most person-variance-dominated cell the line has ever
run, and its b-only field is near zero BY DESIGN, so a positivity gate there
tests the HYPOTHESIS rather than the regime — precisely the error M1b's own
registration corrected ("an outcome-side flat field is cell-2 EVIDENCE, not
channel death"). RN-M2-8 pins the registered wording as the gate and reports the
inherited form beside it. **The pilot data are IDENTICAL under both readings —
same eight worlds, same seeds — so only the gate reading differs**, the
first-reading artifact is preserved as `g2m2_pilot_FIRST_READING.json`, and a
planner who prefers the inherited form can read UNRESOLVED_SEAL off the report
without recomputing anything. **Found AFTER the pilot ran; disclosed as such.**

**Anomalies, with timing.** The hypothesis-relevant boundary is THE STAMP;
everything before it is verification and arithmetic on published numbers, and
every fresh world came after it. **A-1/A-2 (before Part 0):** the inherited
CPython 3.12.12 lock environment; no `timeout(1)` on macOS. **A-3 (BEFORE
Part 0 ran):** four of the harness's own embedded constants — the M1e
monotonicity contrasts, first transcribed from the M1e report's rounded prose —
were wrong in their trailing digits; corrected by reading the artifacts directly
and a rounded-quote cross-check added. Had this not been caught, G0m2 would have
raised a FALSE citation defect and stopped the leg: rule 24's discipline applied
to the harness's constants rather than the report's tables. **A-4 (AFTER the
pilot):** the G2m2 reading above. **A-5 (after `measure`):** `finalize` crashed
once on a stale key left by A-4's repair; `decision.json` was already written,
the fix touched only the table writer, no number changed. **A-6:** P1 is a hit
near the edge (0.8671810125388784). **A-7:** no stage near its 2x threshold — Part 0
`2.3444879055023193` s against 300 s, the three world chunks inside their 420 s
estimates.

**Registration-defect candidate: ONE, and it is consequential.** G2m2 says
"finite/non-saturated/nonzero-variance" without pinning what saturation means
for this statistic, while three prior legs establish a conflicting "(0,1)"
convention IN CODE. The two readings route to DIFFERENT OUTCOMES here —
`LEVEL_LAW_PREDICTIVE_SCOPED` versus UNRESOLVED_SEAL — so this is not the usual "nothing turned
on it" flag: everything turned on it. The executor pinned the registered wording
against the statistic's source-verified range and reported both; a successor
should pin the saturation test explicitly, and the program should decide once
whether the inherited "(0,1)" form is a convention or a bug in three prior gates.

**What this establishes, and what it does not.** ESTABLISHED, prospectively: the
scoped level law — free share margins plus a steep negative r-power, r interior
to `[0.4541409476972356, 0.8189581462487876]` — predicted three configurations nobody had run, at a share
above the trained envelope and a phi above the trained ladder, and hit all three,
with the predictions hashed before the first world existed and the ordering
enforced in code. **The M-line's object is PREDICTIVE in its scope, and the
scope IS the claim.** NOT established: (i) nothing extends to configurations
whose realized r leaves the window — the excluded share-.70 / phi >= 0.85 cells
are the boundary, computed and named at registration; (ii) the exponent remains
a ridge coordinate, not a constant (appendix Y stands — what transferred is the
PREDICTION, not q); (iii) P1's hit sits at 0.8671810125388784 of the way to its edge
on a band the ridge made wide, and it is the one a successor should tighten.

### Planner adjudication of M2 (2026-08-11, appended after the run) — THE SEAL LANDS 3/3; THE SATURATION CALL IS UPHELD; THE TAX BECOMES A CURVE HYPOTHESIS

**LEVEL_LAW_PREDICTIVE_SCOPED + STRESS_ABOVE accepted.** Three sealed
predictions, hashed at 06:09:46Z with ZERO pre-stamp generations and
the permit re-read from disk +176.076 s later, all three INSIDE
(positions +0.867 / −0.238 / −0.368 of half-width), all bands under
budget, replication quiet (Δ 0.0007701504312663671 against a
0.005647466456046939 bar). **The scoped level law — free share-margins
plus the steep negative r-power, r interior to [0.4541409476972356,
0.8189581462487876] — is the M-line's first law at a sealed
PREDICTIVE grade. The scope is the claim.** The graded-laws table of
record updates at line synthesis, after M3.

**The judgement that changed the outcome is UPHELD — RN-M2-8.** The
registered word is "non-saturated"; the statistic is a weighted mean
of matrix cosines on [−1, 1], so ZERO IS ITS NULL, NOT ITS FLOOR. C1
(share 0.70, V = 0.21) is by design the most person-variance-taxed
cell the program has ever run: at its measured mean
0.034417674625862156 and world-sd 0.018618930632302133, the expected
MINIMUM over 192 draws sits near mean − 2.7σ ≈ −0.016 — a single
world at −0.0007988006295671071 is EXPECTED UNDER THE HYPOTHESIS
ITSELF, and a strict-positivity gate there would have gated on the
hypothesis, not the regime (planner arithmetic, confirming the
executor's domain argument). The first reading is preserved in
`g2m2_pilot_FIRST_READING.json`; the pilot data are identical under
both readings.

**Defect #49 (mine; rule 29 enacted).** G2m2 left the saturation
predicate unpinned while three prior legs carried a conflicting
strictly-inside-(0,1) convention in code — and the two readings route
to DIFFERENT OUTCOMES (PREDICTIVE_SCOPED vs UNRESOLVED_SEAL);
everything turned on an unpinned word. **Rule 29:** every
regime/sanity gate pins its predicate in the measured statistic's OWN
DOMAIN (bounds, null, saturation points), named per statistic; any
divergence between registered text and an inherited code-form is
disclosed and adjudicated BEFORE an outcome-relevant consequence. The
retroactive question — did any prior pilot/smoke world in
M1b/M1c/M2's predecessors breach (0, 1), making the latent
convention consequential — is assigned to M3's G0 as a mechanical
check (REOPEN flag if breached; expected quiet: those cells' means
sit far from 0).

**What the seal's margins say (feeding M3).** P1 landed at +0.867 of
its half-width: the r-channel at exterior share ran STRONGER than the
transferred prediction (measured 0.009126239258272953 vs predicted
0.003242277707985443). Under the fixed channel this splits the two
α(0.70) readings: α from C1 = 0.0423, from C2 = 0.0481 (planner
values; executor recomputes at full precision) — the triangle closes
in M3. STRESS_ABOVE confirmed its pre-signed direction: measured C2
0.04354391388413511 sits above the tax-linear envelope
[0.03435630613483847, 0.04211861836310243] — the THIRD independent
sign that the α-margin is convex, i.e. the local tax DECLINES in V.

**The κ(V) hypothesis (planner arithmetic from persisted artifacts,
recorded BEFORE M3 runs).** Local tax slopes −Δα/ΔV from M1e's free
margins α = [0.18560847593788873, 0.1456494891347315,
0.10934916761257428, 0.06667603971206824] at V = [0.03, 0.075, 0.12,
0.18]:

    0.8879774845146051   (V .03 → .075)
    0.8066738116034938   (V .075 → .12)
    0.7112187983417673   (V .12 → .18)
    [~0.62, ~0.81]       (V .18 → .21, via M2's two α(0.70) readings)

— a monotone decline of ~0.18 over ΔV ≈ 0.11. A pre-signed mechanism
for the representation spread rides with it: a representation that
OMITS the r-channel (M1e's tax-additive) loads cov(−|λ|·r^q, V) > 0
into its slope and reads LOW (its 0.6761549415814 against local
slopes ≥ 0.71) — so if the tax is a curve and the channel is real,
the six published κ̂ values should be RETRODICTABLE from one law.
That closure test is M3's centerpiece.

---

## M4-M3 — the tax curve (is κ a constant, a curve, or an artifact of averaging?)

**REGISTERED 2026-08-11, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. The line's final chartered leg,
refined per rule 28 and appendix Z.2: not "is κ one number" but **is
the local tax κ(V) = −dα/dV constant, and does ONE law retrodict
every published κ̂ through each representation's own estimator?**
All M-line conventions inherited; rules 25–29 in force; salts
`m4m3-world` / `m4m3-pilot`, master_seed 20260811, 192 worlds/cell.

### Design (planner arithmetic RUN at registration — #43 convention; all 16 r INTERIOR to the trained window)

Grid: share ∈ {0.10, 0.175, 0.25, 0.325, 0.40, 0.50, 0.60, 0.70} ×
φ ∈ {0.05, 0.60} = 16 cells × 192 worlds = 3072 worlds. V = 0.3·share
∈ [0.03, 0.21] (ratio 7.0); executor reproduces this table bit-exactly
(G0m3(i)):

| share | V | r(φ=.05) | r(φ=.60) |
|---|---|---|---|
| 0.10 | 0.03000000000000001 | 0.8189581462487876 | 0.8075174172340943 |
| 0.175 | 0.05250000000000001 | 0.8029938537206762 | 0.7827569526268938 |
| 0.25 | 0.07500000000000002 | 0.785015540293945 | 0.7558507450373838 |
| 0.325 | 0.0975 | 0.7645994805478157 | 0.7264504152667802 |
| 0.40 | 0.12000000000000004 | 0.7411873080384952 | 0.6941115392115328 |
| 0.50 | 0.15000000000000002 | 0.7039654030974909 | 0.6453873930804982 |
| 0.60 | 0.18000000000000005 | 0.6573448847694047 | 0.5883719155687073 |
| 0.70 | 0.21000000000000005 | 0.5967380569813433 | 0.5197539933932338 |

### Estimation (pinned)

PRIMARY: the channel FIXED at the M2-sealed transfer point θ* =
(λ, q) = (−0.057625974791364554, 3.863625377453229); per share,
α̂_s = mean over both φ-cells of (per-world field − λ·r^q), with
world-bootstrap SEs (B = 2000, pinned seed; 20000 at rule-13
boundaries). SECOND READING: the joint refit (α×8, λ, q) free —
L-3m3's transfer check. Local slopes: κ̂_i = −Δα̂/ΔV at the seven
adjacent pairs, bootstrap CIs. Curve family over the 8 (V, α̂)
points, THREE forms, no others: **A-lin** α = c − κ·V; **A-quad**
α = c − κ0·V + (κ2/2)·V²; **A-sat** α = c + A·exp(−V/τ). Selection
by leave-one-SHARE-out RMSE; bootstrap through the whole pipeline
(worlds → α̂ → curve) for parameter CIs; tie rule 5%; rule 26 (no
declared bounds; numerical-limit surveillance stays on).

### The retrodiction closure (the centerpiece; rule 14 link pinned)

From the winning curve + the fixed channel, generate NOISELESS
predicted fields at each legacy estimator's OWN persisted design
points and run EACH ESTIMATOR'S OWN PIPELINE on them (the pinned
link: "the estimator's own code on law-generated data"; Monte-Carlo
noise version as a second reading). Six targets, verified at source
in G0m3(iv):

1. sealed difference-fit κ = 0.7220359963712748;
2. the K2e 9-pair refit (≈ 0.715; executor pins its persisted source
   and full precision);
3. K2f F2 κ′ = 0.750086268225045 [0.5202855978239498,
   0.8612166024267973];
4. M1c F1e κ = 0.7601952008701406 [0.7356727662590873,
   0.7846243216827854];
5. M1d F0 κ = 0.7766770259880144 [0.7482226203832176,
   0.8064115044591174];
6. M1e E-tax-add κ = 0.6761549415814 [0.6619291032569563,
   0.6901486195533926].

HIT = predicted κ̂ inside the target's published CI; for point-only
targets (1, 2 if no CI persists), |Δ| ≤ 0.03. Pre-signed direction
(appendix AA): target 6 is predicted to be retrodicted LOW via the
channel-covariance loading.

### Leans (sides declared)

- **L-1m3 [.65] TAX_IS_A_CURVE:** A-lin loses LOO to a curved form
  AND A-quad's κ2 CI excludes 0 (joint clause; satisfiability via
  the projection gate). One-sided each.
- **L-2m3 [.50] CLOSURE:** ≥ 5/6 retrodiction hits.
- **L-3m3 [.70] TRANSFER:** the joint-refit λ CI overlaps M1e's
  [−0.0843564122153383, −0.042724477794351616]; q reported.
- Readings (no gates): the α(0.70) triangle (M2's two cell-derived
  values vs M3's fresh estimate); the seven-slope table with CIs;
  the E-tax-add mechanism decomposition (α-slope term vs channel
  term, arithmetic).

### Gates

- **G0m3 (bit-exact).** (i) the design table above from the pinned
  maps; (ii) every M2 number quoted in the adjudication above
  against `results/m4_m2_scoped_seal/` (predictions, measured,
  positions, hash, stamp/permit timestamps, per-cell means, stress
  envelope, replication numbers); (iii) α vector and θ* against
  `results/m4_m1e_shape/fits.json`; (iv) the six retrodiction
  targets at their persisted sources at full precision (any target
  lacking a persisted CI is typed point-only, tolerance 0.03,
  disclosed); (v) **the #49 retro-check:** mechanically scan every
  persisted pilot/smoke per-world value in M1b, M1c and M2's
  predecessors for breaches of (0, 1); any breach → modifier
  **REOPEN** (planner adjudicates; M3's own verdicts unaffected).
  Mismatch anywhere → STOP.
- **G1m3.** Rule-29-compliant regime predicate, pinned per statistic:
  saturation = |per-world recovery_b_only| ≥ 0.995; finiteness;
  nonzero within-cell variance. No positivity clause (the M2 lesson,
  in code).
- **G2m3 (pilot).** Corners (0.10, 0.05) and (0.70, 0.60) × 4 worlds
  on `m4m3-pilot`, G1m3 predicate only. No seal exists in this leg;
  ordering is the standard Part-0-before-worlds discipline.
- **G3m3.** (a) sides above; (b) **projection gate (rule 25):** σ_w =
  0.026889438327132725 (pinned, G0-verified), parametric replication
  B_proj = 2000, seed = master, at truths A-quad (c, κ0, κ2 fitted to
  the four M1e α points: κ0 ≈ 0.983, κ2 ≈ 1.815 — executor computes
  exactly) and A-lin (κ = the same points' chord 0.7929, computed
  exactly); PASS iff P(κ2 CI excludes 0 | A-quad) ≥ 0.8 AND
  P(κ2 CI excludes 0 | A-lin) ≤ 0.1 at n = 192; once-only escalation
  to 384 worlds/cell (6144 worlds, budget ×2, declared); fail →
  **NON_PROJECTABLE** handback. (c) stages: part0 300 s (projection +
  target verification), pilot 30 s, worlds 4 chunks × 4 cells
  (~460 s each; 8 chunks at escalation), alpha+curves 300 s,
  retrodiction 300 s, finalize 60 s; 2× stop-and-report.
- **G4m3.** Routing below verified disjoint-and-covering by the
  planner (#46 convention); rule 24 generated tables.

### Rule-27 consumption budgets

For the curve to enter the theory table: κ0 width ≤ 0.25, κ2 width ≤
1.5, c width ≤ 0.03, each α̂ width ≤ 0.012. Budgets unmet → the curve
is reported DESCRIPTIVE-ONLY (closure still runs — its predictions
are deterministic); the outcome cell is unchanged and the consumption
flag is recorded.

### Routing (rule 16 — disjoint and covering; L-1's two clauses partition cells 3–5)

| # | condition | outcome |
|---|---|---|
| 1 | any G0m3 mismatch | **STOP** (citation defect) |
| 2 | projection fails after escalation | **NON_PROJECTABLE** (handback; no worlds) |
| 3 | LOO prefers a curved form AND κ2 CI excludes 0 | **TAX_IS_A_CURVE** — the constant-κ era closes by dated note; the curve is the object |
| 4 | LOO prefers A-lin AND κ2 CI contains 0 | **TAX_CONSTANT_RETAINED** — the representation spread needs a different owner (named) |
| 5 | the two clauses disagree | **CURVATURE_UNSETTLED** — which clause failed is stated (power vs form); no curve claim |
| — | closure hits ≥5/6 / 3–4 / ≤2 | modifier **CLOSURE_EXPLAINED / CLOSURE_PARTIAL / CLOSURE_FAILED** (runs in cells 3–5 regardless) |
| — | adjacent slopes non-monotone beyond joint 95% CIs | modifier **NON_MONOTONE** |
| — | L-3m3 disjoint | modifier **TRANSFER_BREAK** |
| — | #49 retro-check breach | modifier **REOPEN** |

### Deliverables and budget

`scripts/run_suica_m4_m3_tax_curve.py`; `results/m4_m3_tax_curve/`
(gitignored); `reports/SUICA_M4_M3_TAX_CURVE_REPORT.md` (generated
tables); outcome append HERE; one ledger row (EXPLORATORY); exactly
ONE commit `feat(m4-m): M3 — the tax curve — <SLUG>`, never amended,
never pushed; suite green first. 3072 worlds (6144 at escalation) +
8 pilot; target < 75 min wall, every stage < 600 s.

### Outcome (appended 2026-08-11 by the executing agent; append-only)

**`TAX_IS_A_CURVE` — rule-16 cell 3; modifier(s) `['CLOSURE_EXPLAINED']`.**
Curve consumption: **CONSUMABLE**. 3072 fresh worlds
(192/cell x 16 cells). Report:
`reports/SUICA_M4_M3_TAX_CURVE_REPORT.md`; harness
`scripts/run_suica_m4_m3_tax_curve.py`; artifacts `results/m4_m3_tax_curve/`.

**THE TAX IS A CURVE, AND THE CURVE WAS PREDICTED BEFORE IT WAS MEASURED.**
Both of L-1m3's clauses fire together (True): A-quad wins
leave-one-share-out at `0.001405398973367856` against A-lin's `0.003599294048156043`
(2.5610478706490034x worse), and its curvature term
**kappa2 = `1.562877770472943`, 95% CI `[1.0324533419318935, 2.119753814549891]`, EXCLUDES ZERO** — stable at
B = 20000 (`[1.0132571446499121, 2.1165781162197774]`). The winning curve is
**alpha(V) = c - kappa0*V + (kappa2/2)*V^2** with c = `0.21247398265278816`,
kappa0 = `0.9601680204204508`. The planner's pre-run arithmetic from M1e's four
alpha points predicted (kappa0, kappa2) ~ (0.983, 1.815); the fresh eight-point
measurement returns `0.9601680204204508` CI `{'c': [0.2089174011834685, 0.21623033411776654], 'kappa0': [0.8935297536704152, 1.0313114338425122], 'kappa2': [1.0324533419318935, 2.119753814549891]}`'s kappa0 and
`1.562877770472943` CI `[1.0324533419318935, 2.119753814549891]` — **both intervals contain the pre-run
prediction.** The seven local slopes run [0.8470350335482366, 0.8186459292853376, 0.9062733099585221, 0.803704394783968, 0.7522331399898755, 0.6305583963376168, 0.7111114837980396], declining from
`0.8470350335482366` to `0.7111114837980396` (minimum `0.6305583963376168`), with
NON_MONOTONE = False (0 adjacent increases
surviving their joint 95% CIs). **The reader's marginal price for
person-variance is highest where person-variance is scarcest.**

**AND ONE LAW RETRODICTS FIVE OF THE SIX PUBLISHED kappa-hats.**
5/6 -> **CLOSURE_EXPLAINED**, computed by generating NOISELESS
law-fields at each legacy estimator's OWN persisted design points and running
EACH ESTIMATOR'S OWN PIPELINE on them (rule 14's pinned link; no estimator was
re-implemented in a common form):

| # | target | published | predicted by the law | hit |
|---|---|---|---|---|
| 1 | sealed 6-pair difference-fit | 0.7220359963712748 | 0.746239389222837 | True |
| 2 | K2e 9-pair refit | 0.7145934082034173 | 0.7490807810533479 | False |
| 3 | K2f F2 | 0.750086268225045 | 0.7679919131618126 | True |
| 4 | M1c F1e | 0.7601952008701406 | 0.7626119165106452 | True |
| 5 | M1d F0 | 0.7766770259880144 | 0.7813000334879925 | True |
| 6 | M1e E-tax-add | 0.6761549415814 | 0.6795535093252197 | True |

**Target 6's pre-signed direction HELD (True):** the representation
that OMITS the r-channel is retrodicted at `0.6795535093252197`, the LOWEST of all
six, exactly as appendix AA predicted from the channel-covariance loading.
**The spread among the six published "constants" is not six taxes; it is one
curve seen through six estimators with different amounts of channel leakage.**
The single miss is target 2 at `0.7490807810533479` against `0.7145934082034173` — delta
`0.03448737284993053` against a `0.03` point-only tolerance, missing by
`0.004487372849930532`, while the same law hits its 6-pair sibling and every
CI-typed target. The closure is 5/6 and is not rounded up.

**G0m3 PASSES on all five clauses**, including two that matter especially: the
six targets verified at their persisted sources at full precision (targets 1 and
2 typed point-only, tolerance 0.03, disclosed — neither persists a CI), and the
pair reconstruction VERIFIED bit-exactly before any retrodiction ran (every
persisted dvar reproduced as V_a - V_b and every persisted D as
level_a - level_b, with the sign convention -- published kappa = MINUS the
through-origin slope -- verified rather than assumed).

**G0m3(v), the #49 retro-check: REOPEN = False.** Mechanical scan of
46 persisted pilot/smoke worlds across 23 sources in
the M-line and K2f found **0 breaches of (0, 1) outside
M2's own already-adjudicated case**. The latent strictly-inside-(0,1) convention
was never consequential before M2 walked into the one cell extreme enough to
expose it. Rule 29's retroactive question is closed.

**G3m3(b), the projection gate, passed BEFORE any world:**
P(kappa2 CI excludes 0) = **0.9935** under the A-quad truth (bar >= 0.8)
and **0.0575** under the A-lin truth (bar <= 0.1); escalation fired
False. Both truths were computed from M1e's own four alpha points and
reproduce the planner's sanity values (A-quad [0.21429311564963677, 0.9833070071661382, 1.813410134121013] against
(0.983, 1.815); A-lin [0.2071001875517445, 0.7928829081721366] against a chord of 0.7929). This is
rule 25 at its sharpest: the gate demanded not only power to DETECT curvature
but a bounded false-positive rate AGAINST the linear truth, and both bars were
met before a single world existed.

**Rule 27: all budgets met**, so the curve is CONSUMABLE and may enter the theory
table. Rule 13 fired on the A-quad/A-sat LOO proximity (the tie is between the
two CURVED forms, which cannot touch L-1m3's clause) and the B = 20000 re-run
left every verdict unchanged (True). Rule 26: no declared
bounds, numerical-limit surveillance clean (False). G1m3, the
first gate written under rule 29, held at all 16 cells with its predicate pinned
in the statistic's own domain (finite, abs(x) < 0.995, nonzero variance, NO
positivity clause).

**L-3m3 passes, and passes WEAKLY — stated, not claimed.** The joint refit's
lambda CI `[-30.615714683654733, -0.039429381031983984]` overlaps M1e's `[-0.0843564122153383, -0.042724477794351616]`, so the
verdict is overlap and no TRANSFER_BREAK fires — but the interval runs
to `-30.615714683654733` at its lower end. Freeing (lambda, q) alongside eight
alpha margins on 16 cell means re-creates exactly the ridge appendix Y named,
which is WHY the primary estimation holds the channel fixed at the M2-sealed
point. A successor wanting a real transfer test needs a design that identifies
the channel independently.

**The alpha(0.70) triangle does NOT close, and it fails informatively.** M2's two
cell-derived readings BRACKET M3's fresh estimate: C1 gives `0.04225793772553886`
(below), C2 gives `0.04814189927582637` (above), M3 measures `0.044932685277189494` CI
`[0.0430021644650165, 0.046821180917192055]`, and neither M2 reading is inside (False /
False); spread `0.00588396155028751`. This is the same discrepancy M2's
P1 flagged at 0.8671810125388784 of its band: at share 0.70 the channel does
slightly more work than the fixed theta* assigns it. A reading, adjudicating
nothing, and the clearest surviving pointer for a successor.

**Anomalies, with timing.** The boundary is the first world; Part 0 is arithmetic
on published numbers and every RN note was pinned there. **A-1/A-2 (before Part
0):** the inherited CPython 3.12.12 lock environment; no `timeout(1)` on
macOS. **A-3 (after the noiseless closure, before anything was written):** the MC
second reading re-ran each estimator's FULL start grid on all 200
replicates, overran its stage and was killed with nothing persisted; rewritten to
refit MC replicates from the noiseless optimum (the program's standing bootstrap
convention), and the MC means agree with the killed run's partial output to ~9
decimal places, so the change is a speed-up and not a different estimator —
**the noiseless run, which is what scores, was never affected.** **A-4 (after
`alpha`, before `finalize` consumed it):** a rule-13 stage was missing from the
first harness; the registration says "B = 2000; 20000 at rule-13 boundaries" and
the A-quad/A-sat LOO separation falls inside the 5% tie band. The omission was
the executor's; the B = 20000 re-run was performed and every verdict is
unchanged. **A-5:** L-3m3's overlap is nearly vacuous (above). **A-6:** the
alpha(0.70) triangle does not close (above). **A-7:** no stage near its 2x
threshold — Part 0 `0.05594515800476074` s against 300 s, the four world chunks
inside their 460 s estimates.

**Registration-defect candidates: NONE.** Every clause was satisfiable, the
projection gate did its job before any world was drawn, the routing table was
disjoint and covering as the #46 convention now requires, the rule-27 budgets
attached to the consumed object as #47 requires, and rule 29's domain-pinned
predicate worked without incident at all 16 cells. The two anomalies that
mattered were both the executor's and both were repaired before any
outcome-relevant number was written. This is the first M-line leg to close with
no defect candidate of any kind.

**What is NOT established.** (i) The closure is 5/6, not 6/6 — target 2 misses by
`0.004487372849930532`. (ii) The transfer check passes only weakly. (iii) The
alpha(0.70) triangle does not close, and its failure direction says the channel
does slightly more work at exterior share than theta* assigns. (iv) Everything
remains scoped to this world family, this instrument, and r interior to the
trained window.

### Planner adjudication of M3 (2026-08-11, appended after the run) — THE TAX IS A CURVE; FIVE ROUTES WERE ONE LAW; THE LINE CLOSES

**TAX_IS_A_CURVE + CLOSURE_EXPLAINED accepted; the curve is
CONSUMABLE (all rule-27 budgets met); zero registration defects — the
line's first fully clean leg.** A-quad wins (LOO 0.001405398973367856,
2.56× better than A-lin) with κ2 = 1.562877770472943
[1.0324533419318935, 2.119753814549891] excluding zero, stable at
B = 20000 — and BOTH fresh intervals contain the planner's pre-run
appendix-AA values (κ0 ≈ 0.983 in [0.8935297536704152,
1.0313114338425122]; κ2 ≈ 1.815 in the interval above). The local
tax declines 0.913 → 0.632 across V ∈ [0.03, 0.21]. **The constant-κ
era closes by dated note (theory appendix BB): κ was never one
number; each published value was its estimator's average over its
V-support, plus channel loading.**

**The closure: 5/6 through each estimator's OWN pipeline**, including
the pre-signed mechanism hit — target 6 (the channel-omitting
tax-additive) retrodicted LOW at 0.6795535 vs published
0.6761549415814, lowest of all six, exactly as appendix AA signed
before the leg ran. The one MISS is honest and narrow: target 2 (the
K2e 9-pair refit, the only point-only-tolerance target) at
0.7490808 vs 0.7145934082034173 — 0.0045 over the 0.03 tolerance.
Named, not queued: whether the 9-pair pipeline's weighting departs
from the compilation's (a target-2 forensic is a small artifact-space
question for any future leg).

**Honest residuals typed.** (i) The A-quad/A-sat tie (3.2%, inside
the 5% band; rule 13 at B = 20000 left verdicts unchanged): both are
CURVED, so TAX_IS_A_CURVE is tie-robust; the FORM between them is
UNSETTLED and matters only beyond V = 0.21 — A-quad's κ(V) crosses
zero at V ≈ 0.614 (out-of-scope arithmetic, NOT a prediction), A-sat
stays positive; the curve is scoped to V ∈ [0.03, 0.21]. (ii) The
α(0.70) triangle does not close (M2's two cell-derived readings
0.0423 / 0.0481 bracket M3's fresh 0.044933 [0.043002, 0.046821]
without containing it) — the transferred channel's within-share slope
at exterior share carries a small bias, consistent with q's
looseness; named. (iii) L-3m3's joint-refit overlap is nearly vacuous
(λ CI [−30.6, −0.039]) — the transfer evidence remains M2's seal,
not this check. (iv) #49 retro-check CLEAN (46 worlds, 23 sources, 0
breaches beyond M2's known case): the latent (0,1) convention never
bit a prior leg; defect #49's retroactive question closes.

**THE LINE IS CLOSED.** Seven legs (M1, M1b, M1c, M1d, M1e, M2, M3),
seven registrations before run, seven single commits, zero amends,
zero agent pushes, suite green at every commit. Laws, corrections,
rules (25–29) and defects (#43–#49) are compiled in
`docs/SUICA_M4_M_LEVEL_LAW_LINE_SYNTHESIS.md`; theory consequences in
appendices W–BB; the graded-laws table of record updates in the V8
integration document's dated addendum. The successor question — WHY
is the tax quadratic — is chartered as the M4-N line (its own plan
document), opened by the planner's square-agreement derivation
(appendix CC) whose response-transport prediction is N1's sealed
test.
