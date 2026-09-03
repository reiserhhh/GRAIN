# GRAIN M4-P — The Penalty-Mechanism Line

Line opened 2026-08-14, on the N-line's close. Question: **WHY does
the level price concentrate where cards read best?** The readability
penalty (appendix Z.1: the steep negative r-channel, λ identified,
sealed in scoped form by M2) has two named mechanisms that predict
OPPOSITE SIGNS under frame-side manipulation at fixed (share, φ):

- **H-SCAFFOLD** — the reader reads the person THROUGH the
  person×frame interaction (K-R1's scaffold corollary; K2b G4b: the
  b-only field's within-author occasion variation IS the frame).
  More common-frame content → more scaffold → b-only recovery RISES.
- **H-CONTAMINATION** — common-frame content is nuisance the
  deployed gauge AMPLIFIES (K1: +0.0925 at a 1× shift = 3.54× the
  F2 composition effect). More → recovery FALLS.

A sign is a cheap, decisive first cut. Tier: EXPLORATORY, label-free,
synthetic. The ledger controls. Registrations before run, append-only.
Standing rules 1–32 bind (K-line header; registry; D5; M-line 25–29;
N-line 30–32 — originating texts control); all conventions in force
(planner-run registration arithmetic incl. every gate-consumed
quantity; ladders checked against every gate sharing a knob; routing
tables verified disjoint-and-covering AND consequence-entailed;
domain-pinned predicates; executed provenance; estimator-noise
floors; two-sided bar reachability). Execution conventions as the
M/N lines (ONE commit per leg `feat(m4-p): ...`, never amended,
never pushed; chunked foreground stages < 600 s; `suica_core/`
READ-ONLY).

## Line charter

- **P1 — the frame-injection sign probe** (registered below).
- **P2 — the dose curve** (named; register only after P1 lands a
  sign): shift-size sweep, functional form, interaction with r.
- **P3 — the de-frame interplay** (named): does K1b/K1c′'s estimated
  de-framing invert or null the P1 sign, connecting the penalty to
  the certified-unadopted repair's deployment caution?

---

## M4-P1 — the frame-injection sign probe

**REGISTERED 2026-08-14, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent (implementation and execution
only; this text is binding). No seal — the leans are split, so no
sealable point prediction exists; ordering is the standard
Part-0-before-worlds discipline.

### Facts cited to motivate leans (rule 8 — sources named; executor re-verifies at full precision)

- K1's common-shift finding: the deployed relation-field gauge
  amplifies common-frame content, +0.0925 at a 1× shift = 3.54× the
  F2 composition effect (`docs/SUICA_V8_IDT_INTEGRATION.md` §1;
  K1's plan-doc outcome controls).
- K1's calibration recipe: "pre-map occasion-level common
  perturbation δ(o) added to every author's response on occasion o;
  sizes … of the world's author-deviation RMS at the response level
  (calibration reported)" (`docs/SUICA_M4_K_IDENTITY_LINE_PLAN.md`
  — K1 registration, Common-shift arms).
- K-R1's scaffold corollary and K2b G4b's frame-carried substrate
  (K-line plan, K-R1 outcome/adjudication).
- The unperturbed anchor: M2's C4 = (share 0.25, φ 0.05, n = 192)
  measured mean 0.12239759528671845
  (`results/m4_m2_scoped_seal/decision.json:per_cell.C4.mean`).

### Machinery (rules 9/12 — delegation with decision rule, K1 precedent)

K2b world family, K2F-FRESH carrier verbatim (int_share = 0), ONE
loader chain. NEW manipulation, transplanted from K1's recipe into
this family: a pre-map occasion-level common perturbation δ(o) added
to EVERY author's response on occasion o, drawn once per world per
occasion (seeded from the leg's lineage), scaled so its realized
response-level RMS equals s × (the world's author-deviation RMS at
the response level), s the arm's shift size; calibration persisted
per world. **The injection point is the LAST common per-occasion
object every author's response shares before the frozen map; the
executor identifies it with file:line in Part 0, and the choice rule
is this sentence** (rule 12; the K1 R-abs precedent pattern). The
frozen map and `suica_core/` are untouched — the perturbation lives
in world construction, upstream of the reader.

### Design

Arms: shift s ∈ {0, 0.5, 1.0} × base cells {(share 0.25, φ 0.05),
(share 0.25, φ 0.60)} = 6 arms × 192 worlds = 1152 fresh worlds.
Base cells sit in the high-r region where the penalty lives
(r = 0.785015540293945 / 0.7558507450373838, planner-computed,
interior); two φ levels give a within-leg replication. Salts
`m4p1-world` / `m4p1-pilot`, master_seed 20260814. Field statistic:
per-world `recovery_b_only`, as everywhere.

### Part 0 (all before any world)

- **G0p1 (bit-exact).** (i) The two base-cell r values above from
  the pinned maps; (ii) the cited K1/K-R1 sentences located and
  quoted verbatim from their controlling documents (rule 24); (iii)
  M2's C4 mean at full precision from its persisted source.
  Mismatch → STOP.
- **G1p1 (rule 10 non-degeneracy + rule 3 liveness, executed).** On
  pilot worlds: (a) the injection provably moves the pre-map object
  (norm delta > 0 at machine precision, reported); (b) the realized
  response-level RMS of δ matches its calibration target within 5%
  per world; (c) at s = 0 the pipeline is BIT-IDENTICAL to the
  unperturbed construction (the injection path proven inert at zero
  — no accidental contamination).
- **G2p1 (pilot).** 4 worlds × {0, 1.0} × both base cells on
  `m4p1-pilot` (16 worlds), rule-29 predicate
  (|recovery_b_only| ≥ 0.995 saturation, finiteness, nonzero
  variance; NO positivity clause).
- **G3p1 (rule 25 projection).** σ from the pilot's s = 0 worlds
  (df-inflated per the 4-world convention); the verdict quantity is
  the OLS slope b̂ of per-world recovery on s over {0, 0.5, 1.0}
  (world-level, 192/arm): SE(b̂) computed exactly for the design;
  declared minimum-interesting slope |b| = 0.01 per 1× shift (the
  within-share penalty's own magnitude scale, appendix Z.1
  arithmetic). Gate: MDE(2·SE) ≤ 0.01; once-only escalation to
  384/arm; still failing → NON_PROJECTABLE handback.
- **G4p1.** Routing below planner-verified disjoint-and-covering
  AND consequence-entailed (#46+#54 conventions); rule 24 generated
  tables; stage estimates: part0 90 s, pilot 40 s, worlds 6 chunks
  (1 arm each, ~120 s), fit 120 s, finalize 60 s; 2×
  stop-and-report.

### Verdicts (rule 22; equivalence per rule 4)

Per base cell: b̂ with world-bootstrap 95% CI (B = 2000; 20000 at
rule-13 boundaries). Classification: POSITIVE (CI > 0) / NEGATIVE
(CI < 0) / NULL (CI inside the ±0.01 equivalence margin) /
UNDERPOWERED (CI straddles 0 and exits the margin). A quadratic
coefficient in s is reported descriptively (no gate). Cross-cell
agreement is the second verdict.

### Leans (sides declared)

**L-1p1 [scaffold(+) .45 / contamination(−) .40 / null-or-split
.15].** Registered honestly split — this is a genuine fork: K-R1's
scaffold says the frame CARRIES the reading; K1's amplification says
the gauge DRINKS the frame. Either sign re-types the readability
penalty's owner.

### Routing (rule 16 — disjoint, covering, consequences entailed)

| # | condition | outcome |
|---|---|---|
| 1 | any G0p1/G1p1 failure | **STOP** (citation/instrument defect; no arms) |
| 2 | projection fails after escalation | **NON_PROJECTABLE** (handback) |
| 3 | both cells POSITIVE | **SIGN_SCAFFOLD** — the penalty's owner is frame-scaffolding; P2 doses it |
| 4 | both cells NEGATIVE | **SIGN_CONTAMINATION** — the gauge's amplification owns the penalty; P2 doses it; connects to K1's +0.0925 quantitatively |
| 5 | both cells NULL | **FRAME_INSENSITIVE** — neither named mechanism owns the penalty at this scale; a third mechanism is named, not invented |
| 6 | cells disagree in classification (any mix incl. UNDERPOWERED on one side) | **SPLIT_OR_UNDERPOWERED** — the φ-dependence or the power shortfall is named; P2's design inherits the diagnosis |

### Deliverables and budget

`scripts/run_suica_m4_p1_frame_injection.py`;
`results/m4_p1_frame_injection/` (gitignored);
`reports/SUICA_M4_P1_FRAME_INJECTION_REPORT.md` (generated tables);
outcome append HERE; one `docs/CLAIMS_LEDGER.md` row (EXPLORATORY);
exactly ONE commit `feat(m4-p): P1 — the frame-injection sign probe
— <SLUG>`, never amended, never pushed; suite green first. 1152
worlds (2304 at escalation) + 16 pilot; target < 30 min wall, every
stage < 600 s.

### M4-P1 outcome (appended after execution; registration above unedited)

**SIGN_SCAFFOLD — routing cell 3.** Both base cells POSITIVE and in
agreement (B1=POSITIVE, B2=POSITIVE; cross-cell agreement True).
**1152 fresh worlds** (192/arm × 6 arms). No seal — the
leans were registered honestly split, so no sealable point prediction existed.
Harness `scripts/run_suica_m4_p1_frame_injection.py`; report
`reports/SUICA_M4_P1_FRAME_INJECTION_REPORT.md`; artifacts
`results/m4_p1_frame_injection/` (gitignored).

**The sign is positive and it is not close.** b̂ = **0.3217400470171695**
[0.3125381894506993, 0.3316075999536289] at B1 (φ 0.05) and **0.34654236945734695** [0.3383230195793485, 0.3553320781635022] at B2
(φ 0.60) — 32× and 35× the declared minimum-interesting slope of
0.01, with both CIs far outside the ±0.01 equivalence margin.
**H-SCAFFOLD wins; H-CONTAMINATION is refuted at this scale in this family.**
More common-frame content makes b-only recovery RISE, steeply.

| cell | φ | s | n | mean | SEM | mean realized δ RMS |
|---|---|---|---|---|---|---|
| B1 | 0.05 | 0.0 | 192 | 0.11873302020953876 | 0.0017776895611196061 | 0.0 |
| B1 | 0.05 | 0.5 | 192 | 0.1794583899390327 | 0.0021742290750443875 | 0.03720787207513443 |
| B1 | 0.05 | 1.0 | 192 | 0.44047306722670826 | 0.0046458635448118155 | 0.07443049350172602 |
| B2 | 0.6 | 0.0 | 192 | 0.12904388025689947 | 0.001732157030280773 | 0.0 |
| B2 | 0.6 | 0.5 | 192 | 0.18906423691764004 | 0.0021547765918366528 | 0.03864141541522801 |
| B2 | 0.6 | 1.0 | 192 | 0.47558624971424646 | 0.004089574666712558 | 0.077341447932803 |

**The dose response is convex, not linear** (descriptive, no gate). Arm means
run 0.11873302020953876 → 0.1794583899390327 → 0.44047306722670826 at B1 and 0.12904388025689947 →
0.18906423691764004 → 0.47558624971424646 at B2: the half-dose moves recovery barely, the
full dose moves it enormously. Quadratic coefficients 0.40057861511636317 and
0.4530033122717317, both positive and both larger than the linear slope itself.
P2's dose-curve design should not assume linearity — the interesting structure
is between s = 0.5 and s = 1.0.

**The injection point, identified as the registration requires.** The object is
`world['common']` (4, 16, 64), built at `scripts/run_suica_m4_k2b_t4_branch.py:337`, returned at
`scripts/run_suica_m4_k2b_t4_branch.py:349`, and **last read before the frozen map at `scripts/run_suica_m4_k2b_t4_branch.py:377`**
— the `emit_panel` line that folds the common channel into every author's
response. That call is at `scripts/run_suica_m4_k2b_t4_branch.py:615`, one line before the map's entry at
`scripts/run_suica_m4_k2b_t4_branch.py:616`. Nothing between them is both common and per-occasion. δ(o) is
added to `common[c, o, :]` for EVERY context, so every author on occasion o
receives the identical response-level shift w["common"]·δ(o). `suica_core/` and
k2b were not edited: the harness perturbs the returned array between k2b's own
`build_k2b_world` and k2b's own `run_field_world`.

**G0p1 PASS.** Both base-cell r values bit-exact from the pinned maps
(0.785015540293945 / 0.7558507450373838); M2's C4 mean bit-exact at source; all 7
citation anchors located and quoted verbatim **by code** — each anchor
substring is found in its controlling document and the containing paragraph
lifted, so rule 24 covers the quotations as well as the tables.

**G1p1 PASS on all three clauses.** (a) The injection moves the pre-map object:
‖Δcommon‖_F ∈ [14.227782891799846, 14.883213260576998], every array bit-changed.
(b) Calibration lands to **1.8675090942348197e-16** relative error against a
0.05 band — the scale is solved in closed form and the realized RMS
recomputed from the scaled δ before persisting, so this tests an executed
number. (c) **s = 0 is BIT-IDENTICAL** to the unperturbed construction on
8 worlds (max |difference| 0.0): the injection
path is proven inert at zero.

**G3p1 PASS.** σ from the pilot's s = 0 worlds pooled within cell (df
6), df-inflated ×1.6498974741130894 on M1b's χ²(0.10) convention:
0.02686397191341203 → 0.04432279940458349. RN-P1-4 makes SE(b̂) exact for this design
(σ/√(n/2)), giving SE 0.004523676771373484 and **MDE(2·SE) = 0.009047353542746968 ≤
0.01**. Escalation to 384/arm did not fire (False).

**The unperturbed arm replicates M2's anchor.** P1's B1 s = 0 mean
0.11873302020953876 against M2's C4 0.12239759528671845 on the same base cell:
difference -0.0036645750771796964, z = -1.3707691436547715, within two pooled SEM
(True). Different salt, so a distributional replication, not a
bit-identity claim.

**Two findings about the shared machinery, both caught before any arm ran.**

1. **The corpus label is part of the frozen map's input (RN-P1-8).** k2b builds
   `corpus = f"m4k2b-{arm_id}-w{world_index}"` (k2b:613) and the map's output
   depends on it: the SAME world read under two different arm labels returns
   recovery_b_only 0.13207502282291814 vs 0.11984192209116917 — **0.0122 apart,
   larger than this leg's 0.01 bar**. Because `world_index` is also in the
   string this is exchangeable per-world noise rather than a per-arm offset, and
   it is already inside the σ the projection uses; but the arm tag was made
   s-INDEPENDENT (`P1-{cell}`) so the three s-arms share the identical corpus at
   matched world index and any corpus effect cancels exactly in b̂. World seeds
   still differ per arm, so the registered 1152 fresh worlds stand. **G1p1(c)
   failed on the first pilot for exactly this reason — the control had been
   given a different label — which is the gate doing its job on the harness.**
   Recorded for successors: any leg whose arm label varies with the manipulated
   factor should confirm the label is not confounded with the contrast.
2. **K1's own implementation defines the calibration, and it is not the obvious
   one (RN-P1-2).** The registration transplants K1's recipe, so K1's code
   controls (rules 9/12). `_author_deviation_rms`
   (`scripts/run_suica_m4_k1_issuer_theorems.py:337-361`) centres each response
   on its **(context, occasion)** cell mean, averages each author's deviations
   **over that author's occasions** to one vector per author, and takes the RMS
   over (author × dim). That is a much smaller quantity than a raw per-response
   deviation RMS and therefore sets a much smaller dose at the same s. It was
   transplanted literally; the two alternative readings are computed and
   persisted per world but never used to calibrate. One deliberate departure is
   disclosed as RN-P1-7: K1 SETS δ's per-component σ to s·rms, while this
   registration says the REALIZED RMS must EQUAL the target, so δ is rescaled
   exactly — the two agree to ~2% at 1024 draws and both clear G1p1(b), but the
   P1 text is the binding one.

**Rule events.** Rule 13: 0 boundary events — no B = 20000 re-run
(both CIs sit far from 0 and from the equivalence margin). Rule 26: no bounded
winner. Rule 29: in force as G2p1, held at every pilot arm and every full arm.
Rule 30: every constant verified against its persisted source before Part 0 and
every quoted sentence extracted by code. The slope identity b̂ = mean(s=1) −
mean(s=0) was verified against a full OLS witness on all 3n points at both
cells (True / True).

**Executor self-report.** Two standing anomalies, both resolved before any
hypothesis-relevant number existed: A-1 the dispatched interpreter is absent (a
pinned CPython 3.12.12 venv was built from the lockfile); A-2 `timeout(1)`
is absent on macOS (foreground stages under explicit sub-600 s timeouts).

**Registration-defect candidate (1).** The verdict classification is **not
disjoint**: POSITIVE (CI > 0) and NULL (CI inside ±0.01) overlap — a CI
like (0.001, 0.005) satisfies both. Rule 16 is met on the routing table but not
on the classification feeding it. Pinned before any number as RN-P1-5
(equivalence wins, per rule 4), with the sign-first ordering also computed; the
two agree at both cells here (True / True), so nothing
turned on it this time. Non-blocking.

### Planner adjudication of P1 (2026-08-14, appended after the run) — SCAFFOLD OWNS THE PENALTY; THE QUANTIFICATION QUESTION IS NAMED

**SIGN_SCAFFOLD accepted.** Both base cells POSITIVE at 32–35× the
minimum-interesting bar (b̂ = +0.3217400470171695 /
+0.34654236945734695, CIs nowhere near zero or the margin), s = 0
proven BIT-IDENTICAL to the unperturbed construction, calibration
exact to 1.9e-16, the injection point pinned to the last common
per-occasion object before the frozen map (k2b:377) with k2b and
`suica_core/` diff-verified untouched. My .45/.40 split resolves
decisively: **frame-scaffolding owns the readability penalty;
the contamination mechanism is refuted at this scale.** Two
precisions so the record is not misread: (i) K1's amplification
finding is NOT refuted — it concerns the gauge's issuer-leakage
quantity; P1 concerns b-only recovery; the gauge can amplify common
content AND read the person better through it — indeed that
conjunction IS the scaffold picture. (ii) Per the executor's caveat
(K-R1's F-1: the b-only truth panel is itself frame-carried), the
sign names the OWNER; it does not quantify how much of the boost is
genuine person-reading versus frame-vs-frame agreement — that
decomposition is P2, registered below.

**Theory consequence (appendix FF):** the level law's r-channel is
re-read as THE SCAFFOLD GRADIENT — at fixed share, r is a bijection
of φ, and the causal content of λ·r^q is frame-block coherence, not
card readability per se. M1e's cross-share victory of the
r-REPRESENTATION stands untouched (best predictive coordinate ≠
causal owner); the penalty is re-typed from "a price of
readability" to "the loss of scaffold that co-varies with
readability at fixed share."

**Machinery findings, recorded:** (i) k2b's corpus string embeds
`arm_id` and the frozen map's output depends on it — same world
under two labels differs by ~0.0122 (per-world noise via the label
path, not a per-arm offset; the executor made the arm tag
s-independent, and G1p1(c) caught its own first pilot control on
exactly this — the gate working on the harness). Standing
machinery note: k2b-family REPLICATION anchors are distributional
across salts/labels, bit-level only at matched corpus strings —
prior legs' cross-arm contrasts are unaffected (different configs =
different worlds regardless). (ii) K1's calibration is a
(context, occasion)-centred author-mean deviation RMS — the
literal transplant (rules 9/12) with both readings persisted
(RN-P1-7, ~2% apart) was correct. (iii) The dose response is
CONVEX (quadratic > linear at both cells; s = 1.0 lifts recovery
to ~0.44–0.48, a regime far above anything the family has produced)
— P2 must not assume linearity and must disclose the
common-to-author variance ratio per arm.

**Defect #55 (mine, non-material here):** the verdict
classification was not disjoint — POSITIVE (CI > 0) and NULL (CI
inside ±0.01) overlap; the executor's RN-P1-5 pin (equivalence
wins, rule 4) was correct and both orderings agreed. The #46+#54
mechanical check is extended to CLASSIFICATION predicates feeding a
routing, not only the routing cells.

---

## M4-P2 — the dose–decomposition (genuine reading vs frame-vs-frame agreement)

**REGISTERED 2026-08-14, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. No seal (decomposition, not
prediction). Question: **how much of the scaffold boost survives
when the truth side is held frame-fixed — and what is the dose
law's shape?**

### The decomposition (registered algebra)

For each world at dose s, build TWO variants at the same index:
the dosed world (δ at scale s) and the zero-dose world (δ = 0) —
bit-identical except the common channel (G1p1(c)'s proven
inertness is the constructive guarantee). Score the DOSED world's
gauge output twice: **R_nat(s)** against its own truth panel
(P1's quantity) and **R_cf(s)** against the ZERO-DOSE world's
truth panel (the counterfactual, frame-fixed truth). Then per
cell: genuine improvement G(s) = R_cf(s) − R_cf(0); frame-vs-frame
share F(s) = [R_nat(s) − R_nat(0)] − G(s); at s = 1.0 the
FRAME-SHARE fraction f = F(1)/[R_nat(1) − R_nat(0)] (well-defined
here: P1 measured the denominator at ~0.32 ≫ noise).

### Design

s ∈ {0, 0.25, 0.5, 0.75, 1.0} × base cells {(0.25, 0.05),
(0.25, 0.60)} × 192 worlds = 10 arms, 1920 fresh worlds (each world
built in dosed and zero-dose variants; the zero-dose builds of the
s = 0 arm ARE the arm). Salts `m4p2-world` / `m4p2-pilot`,
master_seed 20260814. Arm tags s-independent in the corpus string
(the P1 machinery note, now a pinned requirement). Field
statistics: per-world R_nat and R_cf as defined.

### Gates

- **G0p2 (bit-exact).** P1's six arm means, both slopes and CIs,
  the calibration record, and the injection-point pin verified
  against `results/m4_p1_frame_injection/`; the two base-cell r
  values from the pinned maps. Mismatch → STOP.
- **G1p2 (rule 10).** (a) The zero-dose variant of a dosed world is
  BIT-IDENTICAL to the plain construction (P1(c) re-proven on this
  leg's salt); (b) at s > 0, T-cf differs from T-nat provably
  (norm delta > 0 reported); (c) R_cf(0) ≡ R_nat(0) bit-exactly
  (the decomposition's zero-point identity).
- **G2p2 (pilot).** 4 worlds × s ∈ {0, 1.0} × both cells, both
  scorings, rule-29 predicate; REGIME disclosure: the
  common-to-author response-level variance ratio per arm
  (reported at every dose; ratio > 10 at any arm → modifier
  **REGIME_NOTE**, adjudicating nothing but scoping the claim).
- **G3p2 (rule 25).** Verdict quantity: the OLS slope of per-world
  R_cf on s (per cell). σ from pilot zero-dose worlds
  (df-inflated); MDE(2·SE) ≤ 0.01 at 192/arm (P1 realized 0.00905
  on the same design); once-only escalation to 384/arm; else
  NON_PROJECTABLE.
- **G4p2.** Routing verified disjoint-and-covering, consequences
  entailed, CLASSIFICATION disjointness verified (#55's extension:
  equivalence-wins ordering pinned IN the registration — NULL is
  tested first, then sign); rule-27 consumption budget: f's 95% CI
  width ≤ 0.30 (else f is reported DESCRIPTIVE-ONLY and cell
  labels carry UNQUANTIFIED); rule 24; stages: part0 90 s, pilot
  60 s, worlds 10 chunks (~150 s each; ×2 at escalation), fit
  120 s, finalize 60 s.

### Verdicts and leans (rule 22; equivalence per rule 4, NULL first)

Per cell, on the R_cf slope: NULL (CI inside ±0.01) / POSITIVE
(CI > 0) / NEGATIVE (CI < 0) / UNDERPOWERED. Leans: **L-1p2
[genuine-survives(+) .55 / all-frame(NULL) .30 / other .15]**;
**L-2p2 [f > 0.5 at s = 1.0: .55]** (reported with CI under its
budget). Dose-form fits (linear/quadratic on R_nat and R_cf) are
DESCRIPTIVE (no verdict — nothing downstream consumes the form
yet; N-line lesson).

### Routing (rule 16 — disjoint, covering, consequences entailed)

| # | condition | outcome |
|---|---|---|
| 1 | any G0p2/G1p2 failure | **STOP** |
| 2 | projection fails after escalation | **NON_PROJECTABLE** |
| 3 | both cells POSITIVE on R_cf | **GENUINE_SCAFFOLD** — frame content genuinely improves person-reading against a frame-fixed truth; f quantifies the split |
| 4 | both cells NULL on R_cf | **PURE_FRAME_AGREEMENT** — the P1 boost is entirely frame-vs-frame; the gauge does not read the person better, it reads the frame; the penalty's mechanism re-types accordingly (major theory note) |
| 5 | both cells NEGATIVE on R_cf | **SCAFFOLD_TRADEOFF** — frame content actively costs person-reading while inflating apparent recovery; theory note |
| 6 | cells disagree or any UNDERPOWERED | **SPLIT_OR_UNDERPOWERED** — named; P3 inherits the diagnosis |
| — | any arm's variance ratio > 10 | modifier **REGIME_NOTE** |
| — | f's budget unmet | modifier **UNQUANTIFIED** |

### Deliverables and budget

`scripts/run_suica_m4_p2_dose_decomposition.py`;
`results/m4_p2_dose_decomposition/` (gitignored);
`reports/SUICA_M4_P2_DOSE_DECOMPOSITION_REPORT.md` (generated
tables); outcome append HERE; one ledger row (EXPLORATORY); exactly
ONE commit `feat(m4-p): P2 — the dose–decomposition — <SLUG>`,
never amended, never pushed; suite green first. 1920 worlds (×2
variants where dosed) + 32 pilot; target < 45 min wall, every
stage < 600 s.

### M4-P2 outcome (appended after execution; registration above unedited)

**GENUINE_SCAFFOLD — routing cell 3; modifiers: none.** Both
cells POSITIVE on the R_cf slope (B1=POSITIVE, B2=POSITIVE; agreement True).
1920 fresh worlds (192/arm × 10 arms), each built
in two variants. Harness `scripts/run_suica_m4_p2_dose_decomposition.py`;
report `reports/SUICA_M4_P2_DOSE_DECOMPOSITION_REPORT.md`; artifacts
`results/m4_p2_dose_decomposition/` (gitignored).

**The routing says the scaffold is genuine. The magnitude says it is almost all
frame.** f = **0.9584700070215529** [0.9386859990245562, 0.9784391977625343] at B1 and **0.971270002747466**
[0.9535253481648153, 0.9882217251532477] at B2 — CI widths 0.03975319873797811 and 0.03469637698843242, far inside
the 0.3 rule-27 budget, so f is QUANTIFIED and no UNQUANTIFIED
modifier applies. **95.8% and 97.1% of P1's boost is
frame-vs-frame agreement**; only 4.2% / 2.9% is
genuine person-reading. L-2p2 (f > 0.5) resolves YES decisively; L-1p2
(genuine-survives) resolves YES but narrowly.

| cell | φ | s | n | R_nat mean | R_nat SEM | R_cf mean | R_cf SEM | var ratio (max) |
|---|---|---|---|---|---|---|---|---|
| B1 | 0.05 | 0.0 | 192 | 0.1235343574492067 | 0.0017830787914910914 | 0.1235343574492067 | 0.0017830787914910914 | 0.14084544987425832 |
| B1 | 0.05 | 0.25 | 192 | 0.13229270870621257 | 0.0019986603033000884 | 0.11948277133284631 | 0.0019780529555968418 | 0.15183012179894473 |
| B1 | 0.05 | 0.5 | 192 | 0.17943166001336808 | 0.002061808096229932 | 0.12571327258079104 | 0.0018291084505268455 | 0.19030082365256712 |
| B1 | 0.05 | 0.75 | 192 | 0.27503495194865474 | 0.003065331907046549 | 0.13202722932071947 | 0.0021537585109931215 | 0.2500386168970141 |
| B1 | 0.05 | 1.0 | 192 | 0.4351349145429551 | 0.004542105569862805 | 0.13647512639739026 | 0.002699456881334954 | 0.335303817037589 |
| B2 | 0.6 | 0.0 | 192 | 0.127872827136617 | 0.0018585633345693322 | 0.127872827136617 | 0.0018585633345693322 | 0.14083397198919556 |
| B2 | 0.6 | 0.25 | 192 | 0.14541045794189092 | 0.0020401799836485737 | 0.13056657450706122 | 0.001963009775790961 | 0.15296490962037035 |
| B2 | 0.6 | 0.5 | 192 | 0.1925347409657412 | 0.0023120481778482072 | 0.1314681944399098 | 0.002038903387649453 | 0.19263856149213643 |
| B2 | 0.6 | 0.75 | 192 | 0.29050291216454177 | 0.0029302846520945103 | 0.1381829995603561 | 0.0021377005486191924 | 0.2582398056127467 |
| B2 | 0.6 | 1.0 | 192 | 0.4699706496085552 | 0.004041069917533832 | 0.13770129663633365 | 0.0025541772719360075 | 0.35097712055652686 |

| cell | s | R_nat | R_cf | G(s) | F(s) |
|---|---|---|---|---|---|
| B1 | 0.0 | 0.1235343574492067 | 0.1235343574492067 | 0.0 | 0.0 |
| B1 | 0.25 | 0.13229270870621257 | 0.11948277133284631 | -0.0040515861163603895 | 0.012809937373366262 |
| B1 | 0.5 | 0.17943166001336808 | 0.12571327258079104 | 0.002178915131584344 | 0.053718387432577036 |
| B1 | 0.75 | 0.27503495194865474 | 0.13202722932071947 | 0.008492871871512778 | 0.14300772262793526 |
| B1 | 1.0 | 0.4351349145429551 | 0.13647512639739026 | 0.012940768948183565 | 0.2986597881455648 |
| B2 | 0.0 | 0.127872827136617 | 0.127872827136617 | 0.0 | 0.0 |
| B2 | 0.25 | 0.14541045794189092 | 0.13056657450706122 | 0.002693747370444227 | 0.014843883434829702 |
| B2 | 0.5 | 0.1925347409657412 | 0.1314681944399098 | 0.0035953673032928235 | 0.06106654652583138 |
| B2 | 0.75 | 0.29050291216454177 | 0.1381829995603561 | 0.010310172423739106 | 0.15231991260418568 |
| B2 | 1.0 | 0.4699706496085552 | 0.13770129663633365 | 0.009828469499716663 | 0.33226935297222154 |

**The verdict quantity.** b_cf = 0.015370398353696113 [0.009829531824123106, 0.02099405067675933] at B1 and
0.010909345621091282 [0.00559827343632706, 0.016261954399254843] at B2, against b_nat = 0.3063773429719756
[0.29798309345778956, 0.3145094094028631] and 0.3317152396666109 [0.32408785857830824, 0.3389402818801945]. b_nat reproduces P1
on a fresh salt (P1: 0.3217400470171695 / 0.34654236945734695), so the object being
decomposed is the object P1 measured. G(1) = 0.012940768948183565 [0.006585015895802276, 0.01931243058373269] and
0.009828469499716663 [0.004013107408011507, 0.0160057999823994]; F(1) = 0.2986597881455648 [0.29076271113972507, 0.30688958479858763] and
0.33226935297222154 [0.32362912650735165, 0.3406333136533709]; denominators 0.3116005570937484 and
0.3420978224719382.

**Honest magnitude caveat, stated plainly.** b_cf is POSITIVE by the registered
NULL-first classification — the CI excludes zero and is not wholly inside
±0.01 — but at B1 the CI STRADDLES the equivalence margin and at B2 it
sits mostly inside it. The genuine scaffold is real and signed; its size is at
the very edge of what this leg declared interesting. Both classification
orderings agree at both cells (True / True).

**The dose forms come apart, and that is the cleanest statement of the
finding.** R_nat is strongly convex (quadratic 0.4013115009173945 / 0.4282332588027763)
while R_cf is essentially linear-to-flat (0.01952276786062412 / -0.0006145318872409868).
**P1's convexity was a property of F, not of G.** R_cf runs 0.1235343574492067 →
0.13647512639739026 at B1 and 0.127872827136617 → 0.13770129663633365 at B2 while R_nat
runs 0.1235343574492067 → 0.4351349145429551 and 0.127872827136617 →
0.4699706496085552. Descriptive only (RN-P2-10).

**G1p2 PASS on all four clauses.** (a) The zero-dose variant of every pilot
world is bit-identical to a fresh plain `build_k2b_world` at the same seed
(16 worlds, max |difference| 0.0). (b) At s > 0 the two
truth panels provably differ: ‖T-nat − T-cf‖ ∈ [50.7748011282207,
52.72388384718147]. (c) The zero-point identity R_cf(0) ≡ R_nat(0) holds
bit-exactly on all 8 worlds — the hinge that makes G(0) = F(0) = 0.
(d) **This harness's reassembled gauge path returns exactly what k2b's own
`run_field_world` returns** (True, max |difference|
0.0) — the check that licenses reassembling the path at all.

**G0p2 PASS**: all twelve P1 citations bit-exact (both slopes, both CIs, four
arm means, the verdict slug, the calibration record and the injection pin),
both base-cell r values bit-exact, M2's C4 bit-exact. The injection pin
re-derives to `scripts/run_suica_m4_k2b_t4_branch.py:377` and matches P1 (True); k2b and
`suica_core/` unedited.

**G3p2 PASS**: σ from the pilot's zero-dose worlds (df 6), df-inflated
×1.6498974741130894: 0.031389273730078826 → 0.051789083441501405. Five equally-replicated
levels give Σ(s_j − s̄)² = 0.625, SE(b̂) = 0.004727674872237933, **MDE(2·SE) =
0.009455349744475866 ≤ 0.01**; escalation did not fire (False).

**Regime disclosure (G2p2).** The largest common-to-author response-level
variance ratio over every arm is 0.35097712055652686, far below the 10.0 bar
— **no REGIME_NOTE**. Even at s = 1.0 the author-specific variance still
dominates the common component, so the dosed regime is not degenerate and the
claim is not scoped away.

**Rule events.** Rule 13: 0 boundary events, no B = 20000 re-run.
Rule 26: no bounded winner. Rule 27: f's CI widths 0.03975319873797811 /
0.03469637698843242 against the 0.3 budget — QUANTIFIED. Rule 29: applied
to BOTH R_nat and R_cf at every arm. Rule 30: every cited constant verified
against its persisted source before Part 0.

**Executor self-report.** Two standing anomalies, both resolved before any
hypothesis-relevant number existed: A-1 the dispatched interpreter is absent (a
pinned CPython 3.12.12 venv built from the lockfile); A-2 `timeout(1)` is
absent on macOS (foreground stages under explicit sub-600 s timeouts). No
registration-defect candidates: the registration pinned the classification
order in its own text (#55's convention applied), declared the rule-27 budget
on f, and specified the regime disclosure — all three of the things that needed
pinning were pinned before the run.

**What this licenses and what it does not.** LICENSED: the scaffold mechanism
P1 named is genuine — holding the truth side frame-fixed, more common-frame
content still improves person-reading at both φ levels, CIs excluding zero. P1's
SIGN_SCAFFOLD stands as a SIGN. NOT LICENSED, and this qualifies P1
substantially: the MAGNITUDE P1 reported was overwhelmingly frame-vs-frame
agreement — about 96–97% of the boost disappears
the moment the truth panel is held frame-fixed. Any reading of P1 that treated
+0.32 as "how much better the gauge reads people under more frame" is wrong by
roughly a factor of twenty.

### Planner adjudication of P2 (2026-08-14, appended after the run) — THE SCAFFOLD IS GENUINE AND SMALL; P1'S MAGNITUDE IS RE-TYPED ÷20

**GENUINE_SCAFFOLD accepted; zero registration defects — the
registration pinned in advance exactly the three things the leg
needed (#55's classification order, the rule-27 budget on f, the
regime disclosure).** Both cells POSITIVE on the counterfactual-truth
slope (b_cf = +0.015370398353696113 [0.009829531824123106,
0.02099405067675933] and +0.010909345621091282
[0.00559827343632706, 0.016261954399254843]) — and the headline is
the fraction: **f = 0.9584700070215529 [0.9387, 0.9784] and
0.971270002747466 [0.9535, 0.9882]** (widths 0.040/0.035 against a
0.30 budget — QUANTIFIED eight times over). **Dated note,
controlling P1's reading: the sign stands; the magnitude does not —
~96–97% of P1's +0.32 was frame-vs-frame agreement, and reading it
as improved person-reading is wrong by roughly a factor of twenty.**
P1's convexity is a property of F (the frame component, quad coefs
+0.40/+0.43), not of G (+0.020/−0.001). The genuine component's
CIs escape the ±0.01 equivalence margin NARROWLY — typed honestly:
the genuine scaffold EXISTS (CIs exclude zero, both cells) and its
size sits at the declared interest threshold. The dosed regime is
not degenerate (max common-to-author variance ratio 0.351 against
the 10 bar).

**The coherent picture, stated once:** reading is frame-MEDIATED
(K-R1: remove the frame and reading dies), the mediation's marginal
value is small at this regime (P2: add frame and genuine reading
rises by ~0.013 per 1×), and the RECOVERY STATISTIC is dominated by
frame-agreement co-movement of gauge and truth (P2's f) — K-R1's
F-1 caveat is now a measured number, not a caution.

**The question this forces (P3):** if injected frame content moves
the recovery statistic ~96% through frame-agreement, how much of
the NATURAL φ-gradient — the M-line's sealed r-channel — is
frame-agreement too? The M-line's law is untouched as a law of the
statistic (PREDICTIVE-SCOPED is a property of predictions); what P3
tests is its DECOMPOSITION, using the program's own licensed
counter-operation (T9: frame REFRESHMENT — the T6″ pattern) rather
than injection.

---

## M4-P3 — the natural gradient under frame refreshment

**REGISTERED 2026-08-14, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. Question: **what fraction of
the natural φ-gradient of b-only recovery survives when the truth
side's frame is REFRESHED — same authors, fresh frame?** No seal
(the estimand is a ratio with an honest inferential gap from P2's
injection-f; leans are stated, not sealed).

### Device (rules 9/12 — delegation with decision rules)

Paired worlds per index at share 0.25, each φ: world A (evaluation)
and world B (refresh) built with THE SAME author/trait channel
draws and FRESH state/frame channel draws. The gauge's field from A
is scored against A's own b-only truth (R_nat — the M1c quantity)
and against B's b-only truth (R_refresh — the frame-refreshed
reading; same persons, different frame). **Seed-splitting decision
rule:** the author/trait channel comprises every random object that
persists per author across occasions (b-draws and any per-author
carrier); the state/frame channel comprises every per-occasion or
per-context object (the slow state, the common channel, occasion
assignments); the executor identifies the generator's seed
consumption file:line in Part 0 and splits streams so clause (a)
below is PROVABLE. If the generator's seed structure cannot be
split without touching k2b's code, STOP as INFEASIBLE_SPLIT (an
instrument finding, not a failure; `suica_core/` and k2b remain
READ-ONLY — the split must be achievable by seeding alone through
the existing constructor interface).

### Design

share 0.25 × φ ∈ {0.05, 0.30, 0.60, 0.85, 0.98} (M1c's ladder) ×
192 world-pairs = 960 pairs (1920 worlds). Salts `m4p3-worldA` /
`m4p3-worldB` / `m4p3-pilot`, master_seed 20260814; the A/B pair at
index i shares its author-stream seed and differs in its
frame-stream seed BY CONSTRUCTION (persisted per pair). Arm tags
φ-independent in the corpus string where the machinery permits;
where the corpus string must differ across φ, that is the natural
regime and is disclosed (the P1 label-noise note applies to
REPLICATION anchors only). SECONDARY READING (no gate, answers the
charter's original P3 clause): the estimated de-framing repair
(K1b/K1c′ machinery, diagnostic-only license) applied to A's
estimate and scored against A's truth, per φ — reported as
R_deframe(φ) descriptively.

### Quantities and verdicts (rule 22; NULL-first classification per #55)

Per φ: R_nat(φ), R_refresh(φ) with world-pair bootstrap SEs
(B = 2000; 20000 at rule-13 boundaries). The gradient objects:
range_nat = R_nat(φ=.05) − R_nat(φ=.98) side-signed as in M1c
(the natural gradient; G0 anchors it against M1c's share-.25 row),
range_ref likewise on R_refresh. The estimand: **g_ratio =
range_ref / range_nat**, bootstrap CI.

- **V-P3a (anchor):** R_nat's five levels replicate M1c's share-.25
  row within 2·√2·SEM each (distributional, fresh salt — the P1
  label-noise note applies); a failure here is an instrument STOP,
  not a finding.
- **V-P3b (the decomposition):** classification of g_ratio,
  NULL-first: NO_TRANSPORTABLE_GRADIENT (CI inside ±0.25·(SE-scaled
  equivalence band declared in Part 0 from the realized range_nat));
  MOSTLY_FRAME (CI < 0.25); INTERMEDIATE (CI ⊂ [0.25, 0.5] or
  straddling 0.25/0.5 with width under budget); SUBSTANTIALLY_GENUINE
  (CI > 0.5); UNDERPOWERED (width over budget). Rule-27 budget:
  g_ratio CI width ≤ 0.30, else UNQUANTIFIED modifier and the leg
  reports levels only.
- **V-P3c (levels):** if R_refresh is itself within the rule-29
  noise floor of ZERO at every φ (no transportable reading at all),
  sub-case **NO_TRANSPORTABLE_READING** fires and V-P3b is recorded
  N/A (a ratio of noise is not a ratio).

### Leans (sides declared)

**L-1p3 [MOSTLY_FRAME .55 / INTERMEDIATE .20 /
NO_TRANSPORTABLE_READING .15 / other .10].** The inferential gap is
stated: P2's f concerns INJECTED frame at 1×; the natural gradient's
frame share need not equal it — that is exactly what is being
measured, with the program's own licensed instrument.

### Gates

- **G0p3 (bit-exact).** (i) M1c's share-.25 row (five cell means +
  SEMs) from `results/m4_m1c_r_at_level/`; (ii) P2's f values, b_cf
  CIs and the ten-arm table from `results/m4_p2_dose_decomposition/`;
  (iii) the two base r values and the five ladder r values from the
  pinned maps; (iv) the T6″/frame-refreshment lineage located
  (file:line of the published frame-refreshed scoring pattern, cited
  not copied). Mismatch → STOP.
- **G1p3 (rule 10, the split proven).** (a) A/B author-channel
  objects BIT-IDENTICAL per pair; (b) A/B frame-channel objects
  differ (norm delta > 0 per pair); (c) at a control pair with the
  SAME frame seed, R_refresh ≡ R_nat bit-exactly (the refreshment
  operator's zero-point identity).
- **G2p3 (pilot).** 4 pairs × φ ∈ {0.05, 0.98} with the rule-29
  predicate on BOTH scorings; the equivalence band for V-P3b's NULL
  and V-P3c's floor computed here from realized noise (df-inflated)
  and written into the report BEFORE the main arms.
- **G3p3 (rule 25).** Projection for g_ratio's CI width at truths
  {g = 0.04 (P2's 1−f), g = 0.5} from pilot noise: width ≤ 0.30
  under both; once-only escalation to 384 pairs/φ (attached to THIS
  budget gate); else NON_PROJECTABLE.
- **G4p3.** Routing verified disjoint-and-covering,
  consequence-entailed, classifications NULL-first; rule 24; stages:
  part0 120 s, pilot 60 s, worlds 5 chunks (1 φ each, ~240 s;
  ×2 at escalation), score+fit 180 s, finalize 60 s.

### Routing (rule 16)

| # | condition | outcome |
|---|---|---|
| 1 | G0p3/G1p3 failure or seed-split impossible via seeding alone | **STOP / INFEASIBLE_SPLIT** |
| 2 | projection fails after escalation | **NON_PROJECTABLE** |
| 3 | V-P3a fails | **ANCHOR_BREAK** (instrument stop; nothing adjudicated) |
| 4 | V-P3c fires | **NO_TRANSPORTABLE_READING** — the natural gradient carries no frame-refreshed person signal at all; K-R1's picture at natural regimes; V-P3b N/A |
| 5 | g_ratio MOSTLY_FRAME | **NATURAL_GRADIENT_MOSTLY_FRAME** — the M-line law's r-channel is dominated by frame-agreement; the law stands as a law of the statistic; the theory's mechanism section re-types |
| 6 | g_ratio INTERMEDIATE | **MIXED_GRADIENT** — quantified split; theory carries the number |
| 7 | g_ratio SUBSTANTIALLY_GENUINE | **GENUINE_GRADIENT** — the r-channel transports across frames; the scaffold-gradient reading strengthens |
| 8 | UNDERPOWERED / budget unmet | **UNDERPOWERED** (+ UNQUANTIFIED modifier; levels reported) |

### Deliverables and budget

`scripts/run_suica_m4_p3_refresh_gradient.py`;
`results/m4_p3_refresh_gradient/` (gitignored);
`reports/SUICA_M4_P3_REFRESH_GRADIENT_REPORT.md` (generated tables);
outcome append HERE; one ledger row (EXPLORATORY); exactly ONE
commit `feat(m4-p): P3 — the natural gradient under frame
refreshment — <SLUG>`, never amended, never pushed; suite green
first. 1920 worlds (×2 at escalation) + 16 pilot; target < 45 min,
every stage < 600 s.

### M4-P3 outcome (appended after execution; registration above unedited)

**INFEASIBLE_SPLIT — routing cell 1.** STOP / INFEASIBLE_SPLIT. Stopped at
G1p3(a) -- provably unsatisfiable through the existing constructor interface. **0 worlds drawn, no seal.** Ladder:
none: the registration legislates STOP as INFEASIBLE_SPLIT and calls it an instrument finding, not a failure. Harness `scripts/run_suica_m4_p3_refresh_gradient.py`;
report `reports/SUICA_M4_P3_REFRESH_GRADIENT_REPORT.md`; artifacts
`results/m4_p3_refresh_gradient/` (gitignored).

**The impossibility is PROVEN over the constructor's entire input space, not
inferred from reading the code.** `build_k2b_world` has exactly
2 parameters — `build_k2b_world(world_seed: 'int', phi_slow: 'float') -> 'dict[str, np.ndarray]'` — with no varargs (False) and
no defaults (False), verified by `inspect` at run time. Its input
space is therefore the pair (world_seed, phi_slow) and nothing else, so
enumerating the effect of each argument is exhaustive. 8 trials:

| axis | point | trait (A) | a_load (A) | slow (F) | slow_latent (F) | common (F) | int (F) | author identical | SPLIT? |
|---|---|---|---|---|---|---|---|---|---|
| vary world_seed (fixed phi) | {'phi_slow': 0.05, 'world_seed': 1002} | False | False | False | False | False | False | False | no |
| vary world_seed (fixed phi) | {'phi_slow': 0.05, 'world_seed': 20260814} | False | False | False | False | False | False | False | no |
| vary world_seed (fixed phi) | {'phi_slow': 0.05, 'world_seed': 7} | False | False | False | False | False | False | False | no |
| vary phi_slow (fixed world_seed) | {'phi_slow': 0.3, 'world_seed': 1001} | True | True | False | False | True | True | True | no |
| vary phi_slow (fixed world_seed) | {'phi_slow': 0.6, 'world_seed': 1001} | True | True | False | False | True | True | True | no |
| vary phi_slow (fixed world_seed) | {'phi_slow': 0.85, 'world_seed': 1001} | True | True | False | False | True | True | True | no |
| vary phi_slow (fixed world_seed) | {'phi_slow': 0.98, 'world_seed': 1001} | True | True | False | False | True | True | True | no |
| vary both | {'phi_slow': 0.98, 'world_seed': 1002} | False | False | False | False | False | False | False | no |

Two facts do the work. **Changing `world_seed` changes every channel, the
author channel included** — `trait` is built from `z` and `loadings`, both drawn
from `default_rng(world_seed)` BEFORE any frame object, so G1p3(a) (author
objects bit-identical per pair) fails at every seed pair. **Changing `phi_slow`
holds the author channel bit-identical but does not refresh the frame** — only
slow, slow_latent move; **common, int are bit-identical across φ**,
and `common`, the frame channel proper and the object this leg is about, is
among them. The registration's third frame item, occasion assignments, lives in
`k2b.layout()`, which takes no arguments and is memoised in a module-private
global — not seed-driven at all.

**A predicate subtlety, caught and pinned BEFORE any verdict existed
(RN-P3-6), disclosed because a looser reading would have flipped the outcome.**
Scoring "any frame object differs" makes the φ axis look like a split (at fixed
seed, φ leaves the author channel identical and moves `slow`). That reading is
wrong on two independent grounds, both read off the registration: **A and B must
sit at the SAME φ** — φ is the ladder variable being decomposed, so it cannot
also be the refresher — and **`common` is bit-identical across φ**, with only
the recombination of the SAME innovation draws changing, which is not a fresh
draw. The tight predicate (at fixed φ, a seed pair holding every author object
identical while `common` differs) is the one that routes. Both are computed and
persisted; the loose one is satisfied only by
['vary phi_slow (fixed world_seed)', 'vary phi_slow (fixed world_seed)', 'vary phi_slow (fixed world_seed)', 'vary phi_slow (fixed world_seed)'].

**Routes that would work, all excluded by the registration's own words.**
(a) Editing `build_k2b_world` to accept split seeds — k2b is READ-ONLY.
(b) Channel surgery on the returned dicts — not "seeding", and independently
incoherent: the constructor does not return `loadings`, so splicing A's trait
onto B's frame would silently mix two orthonormal bases with no way for a caller
to detect it from the public return value. (c) Mutating k2b's memoised private
`_LAYOUT` to re-key the frame shocks — module-private state mutation, not
seeding, and it cannot refresh the slow state. Each is recorded as a route the
planner could authorise; none was taken.

**What would make P3 feasible.** build_k2b_world would need to accept the frame stream's seed separately from the author stream's -- e.g. an optional frame_seed defaulting to world_seed, which would leave every existing call bit-identical — small and backward-compatible,
since `trait` and `a_load` are drawn before any frame object, but it is an edit
to k2b that this leg is forbidden to make. **The alternative needing no edit
deserves the planner's attention:** reader-level refreshment (K1b / K1c-prime's reader A vs A', disjoint norm sub-pools) is the programme's OWN published form of the T6-double-prime operation and is fully supported by existing machinery -- a different estimand, named for the planner, not chosen here (RN-P3-5)

**A lineage finding inside G0p3(iv) (RN-P3-5, reported not routed).** All
5/5 T6″ anchors were located and quoted by code.
**Every published frame-refreshment in this programme — K1b's and K1c′'s reader
A vs A′ — refreshes the READER's norm/issuer sample via disjoint author
sub-pools; none refreshes the GENERATOR's frame channel.** The operation P3
registers is generator-level refreshment, which has no precedent in the repo
and, as proven above, no interface. The T6″ pattern the registration cites is
real and published; it simply lives at a different layer than the one P3 needs.

**What Part 0 established anyway — all reusable verbatim on re-dispatch.**
G0p3 PASS on all four clauses.

| cell | φ | r_pred | M1c field mean | SEM | n |
|---|---|---|---|---|---|
| s0.25_p0.05 | 0.05 | 0.785015540293945 | 0.12162744485545209 | 0.0017778785791358425 | 192 |
| s0.25_p0.30 | 0.3 | 0.7761302864207245 | 0.12295515685269942 | 0.001799144985921348 | 192 |
| s0.25_p0.60 | 0.6 | 0.7558507450373838 | 0.12714790436588774 | 0.0017053615065044834 | 192 |
| s0.25_p0.85 | 0.85 | 0.7168731389294273 | 0.13204663807737851 | 0.0018782693723257374 | 192 |
| s0.25_p0.98 | 0.98 | 0.6763691758553391 | 0.13201888792665142 | 0.0018133362157806163 | 192 |

The realized natural range across the ladder is **-0.010391443071199338** —
note the sign: the field mean RISES with φ while r_pred falls, M1c's
side-signing convention, and what P3's `range_nat` would have anchored against.
All five ladder r values recompute bit-exactly from the pinned map against
M1c's persisted `r_pred` column. P2's headline verified: verdict
GENUINE_SCAFFOLD, f = 0.9584700070215529 / 0.971270002747466, so the
registration's g = 0.04 projection truth is P2's 1 − f = 0.04152999297844706 /
0.028729997252534.

**Rule events.** Rule 13: not reached — no verdict boundary exists without a
measurement. Rule 26: no bounded winner. Rule 29: not reached; the predicate was
pinned in Part 0 and stands ready. Rule 30: exercised throughout — every cited
constant read from its persisted source, every quoted sentence extracted by
code.

**Executor self-report.** Two standing anomalies, both resolved before any
number existed: A-1 the dispatched interpreter is absent (a pinned CPython
3.12.12 venv built from the lockfile); A-2 `timeout(1)` is absent on
macOS (foreground stages under explicit sub-600 s timeouts). No
hypothesis-relevant number was ever computed — the leg stopped before its first
measured world.

**Registration-defect candidates: none.** The registration anticipated this
outcome precisely, named it, legislated the response, and called it an
instrument finding rather than a failure. It also pinned the channel decision
rule finely enough that the determination could be made mechanically. The one
judgement the executor had to make — what exactly counts as a split — is a
consequence of the design being under-determined only in a corner the
registration had no reason to foresee, and it is pinned as RN-P3-6 with both
readings reported.

### Planner adjudication of P3 (2026-08-14, appended after the run) — AN INSTRUMENT THEOREM, NOT A FAILURE

**INFEASIBLE_SPLIT accepted — zero worlds, zero defects, and the
registration's anticipated-STOP design did its job.** The proof is
EXHAUSTIVE over the published interface (`build_k2b_world` takes
exactly two parameters, verified by inspection): varying
`world_seed` moves the author channel (trait derives from z and
loadings, drawn before any frame object); varying `phi_slow` leaves
`common` — the frame channel proper — bit-identical (only the
recombination of the same innovations changes). **Instrument
theorem: generator-side frame refreshment is impossible on the
published k2b interface by seeding alone.** RN-P3-6 (the φ-axis
pseudo-split — "same draws recombined" is not "fresh draws") is
recorded as the canonical predicate lesson of the rule-29 family;
RN-P3-5 (every published refreshment in the programme acts on the
READER's norm sample; none on the generator's frame channel) is
elevated to a named theory-instrument gap (appendix HH): T9's
counter-operation existed at only one of its two layers.

**Route.** The executor named two remedies; the k2b edit is barred
(published machinery stays published), and reader-level refreshment
measures a DIFFERENT decomposition (norm-sample dependence, not
truth-side frame co-movement) — recorded, not chosen. The licensed
path is the K1/K2a clause itself: **minimal extraction with
file:line provenance** — a split-seed builder extracted from
k2b:321-349 with two RNG streams, certified before use. P3b merges
the instrument and the measurement into one leg, below. All of P3's
Part-0 anchors were verified and are reusable verbatim (the M1c
share-.25 row, the natural range −0.01039144307119933, the ladder r
values, P2's f pair).

---

## M4-P3b — the refresh gradient on a certified split-seed instrument

**REGISTERED 2026-08-14, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. P3's question, estimand,
quantities, verdicts, leans, budgets and routing are INHERITED
VERBATIM (g_ratio = range_ref/range_nat; V-P3a anchor / V-P3b
NULL-first classification with the 0.30 rule-27 budget / V-P3c
NO_TRANSPORTABLE_READING sub-case; leans MOSTLY_FRAME .55 /
INTERMEDIATE .20 / NO_TRANSPORTABLE_READING .15 / other .10;
projection at truths {0.04, 0.5} with the once-only escalation ON
THE BUDGET GATE; R_deframe secondary reading; the routing table
with INFEASIBLE_SPLIT replaced by INSTRUMENT_DEFECT). Exactly the
following is new.

### The instrument (rules 9/12 — minimal extraction with provenance)

`build_split_world(author_seed, frame_seed, phi_slow)`, implemented
INSIDE `scripts/run_suica_m4_p3b_refresh_gradient.py` by extraction
from `run_suica_m4_k2b_t4_branch.py:321-349` with a per-line
provenance table in the report (k2b and `suica_core/` untouched,
diff-verified). **Channel taxonomy (pinned):** author stream =
{loadings, z→trait, a_load} — the per-author-persistent objects
PLUS the shared basis (loadings must be common to an A/B pair or
cross-scoring silently mixes two orthonormal bases — the trap P3
named); frame stream = {slow state, noise, common via
f2().shock_vector(frame_seed,…), shocks via
k2a().shock_int_matrix(frame_seed,…)}. Draw ORDER within each
stream preserves k2b's sequence (documented in the provenance
table). Layout (`k2b.layout()`, argument-free) is shared by
construction.

### Certification battery (ALL before any measurement world)

- **C2a (split proof):** 8 probe pairs (same author_seed, different
  frame_seed): author-stream objects BIT-IDENTICAL; every
  frame-stream object differs (norm deltas > 0, reported per
  object).
- **C2b (determinism):** same (author_seed, frame_seed, φ) rebuilt →
  entire world bit-identical.
- **C2c (basis shared):** loadings bit-identical across each pair.
- **C3 (sanity pilot):** 4 pairs × φ ∈ {0.05, 0.98}, rule-29
  predicate on BOTH scorings; the V-P3b equivalence band and V-P3c
  floor computed here (df-inflated) and written into the report
  before the main arms.
- **C1′ (the anchor as certificate):** V-P3a — R_nat's five levels
  on the NEW instrument replicate M1c's share-.25 row within
  2·√2·SEM each (distributional; the two-stream plumbing changes
  seeds, not the law). Failure → **INSTRUMENT_DEFECT** (a stop
  about the extraction, never a finding about the world).

### Design deltas

share 0.25 × φ ∈ {0.05, 0.30, 0.60, 0.85, 0.98} × 192 A/B pairs
(A: author=s_i, frame=s_i′; B: author=s_i, frame=s_i″; all three
streams hash-derived, persisted per pair). Salts `m4p3b-author` /
`m4p3b-frameA` / `m4p3b-frameB` / `m4p3b-pilot`, master_seed
20260814. R_nat = A-gauge vs A-truth; R_refresh = A-gauge vs
B-truth. Stages: part0 (extraction + provenance + C2 battery)
180 s, pilot 60 s, worlds 5 chunks (~260 s each; ×2 at escalation),
score+fit 180 s, finalize 60 s; every stage < 600 s; target
< 50 min. Six deliverables; ONE commit
`feat(m4-p): P3b — the refresh gradient, certified instrument —
<SLUG>`, never amended, never pushed; suite green first.

### M4-P3b outcome (appended after execution; registration above unedited)

**NON_PROJECTABLE — routing cell 2; modifiers: none.**
NON_PROJECTABLE. Stopped at G3 (rule 25), after the registered once-only escalation. **0 measurement worlds.**
Harness `scripts/run_suica_m4_p3b_refresh_gradient.py`; report
`reports/SUICA_M4_P3B_REFRESH_GRADIENT_REPORT.md`; artifacts
`results/m4_p3b_refresh_gradient/` (gitignored).

**Two things happened, and they should be recorded separately.** First, **the
instrument P3 said was needed now exists and is certified** — that is the leg's
durable asset. Second, the registered estimand turns out to be unmeasurable on
it at any feasible size, for a structural reason the projection makes exact.

**The instrument.** `build_split_world(author_seed, frame_seed, phi_slow)`,
extracted from `run_suica_m4_k2b_t4_branch.py:321-349` in
**18 mapped entries with exactly 4 edits** (rng → rng_a /
rng_f; world_seed → author_seed at the a_load site; world_seed → frame_seed at
the common and shocks sites; the return dict gains `loadings`). Constants and
helpers are IMPORTED from k2b, never copied. k2b and `suica_core/` untouched,
diff-verified.

| k2b lines | stream | note |
|---|---|---|
| 321 | SPLIT | the one edit that makes the instrument: one stream becomes two |
| 322 | author | the shared basis; author-stream so an A/B pair shares it (RN-P3B-4) |
| 323 | author | the b-draw |
| 324 | author | unused in k2b and here; holds k2b's stream order (RN-P3B-3) |
| 325-326 | frame | the state's initial condition |
| 327 | derived | phi enters here exactly as in k2b |
| 328-329 | frame | the AR recursion |
| 330 | frame | pinned to the frame stream by the registration's taxonomy |
| 331 | author | author draw through the shared basis |
| 332 | frame | frame state through the SHARED basis |
| 333-336 | frame | keyed call site #2: the frame channel proper |
| 337 | frame | frame content through the SHARED basis |
| 338-340 | author | keyed call site #3: the per-author interaction carrier |
| 341 | author | per-author-persistent |
| 342 | frame | keyed call site #4: per-occasion interaction shocks |
| 343 | mixed | author carrier x frame shocks -- the interaction, correctly mixed |
| 344 | mixed | through the SHARED basis |
| 345-353 | return | the one ADDITION: k2b withholds loadings, which is what made P3's channel surgery undetectable; exposing it here is what makes C2c possible |

**The one addition earns its place:** k2b does not return `loadings`, and that
omission is precisely what made P3's channel-surgery route undetectable — a
caller splicing two worlds could not see it had mixed two orthonormal bases.
Exposing `loadings` is what makes C2c checkable at all.

**The k2b comparison is a POSITIVE verification, not a caveat.** At equal seeds
the author half reproduces k2b **bit-exactly** (True): identical
on a_load, common, int, trait — `loadings` and `z` are the first draws of the author stream
just as they are k2b's first draws, so `trait` matches, and `a_load`, `common`
and `int` are keyed on a seed rather than on stream position. The ONLY
divergence is noise, slow, slow_latent (True), the three objects the frame
stream draws in sequence, which differ because that stream restarts at the
seed's first draw. **The divergence is localised to the stream restart itself.**

**Certification (all before any measurement world).** C2a = True across
8 probe pairs: every author object bit-identical, every frame object
differing.

| object | stream | norm delta range across the probe pairs |
|---|---|---|
| common | frame | [15.521348769823044, 16.262406409804694] |
| slow | frame | [250.9186220787883, 251.6310585032786] |
| slow_latent | frame | [1229.0471669041528, 1232.475921191497] |
| noise | frame | [250.76248392701416, 251.26126326265268] |
| int | frame | [248.05669190330656, 251.93853774424574] |

C2b = True (same triple rebuilt bit-identically); C2c = True
(loadings bit-identical on every pair — no basis mixing); C3 = True
(rule-29 predicate on BOTH scorings at both pilot φ).

**Bands, computed df-inflated and written before any arm** (df 6,
inflation 1.6498974741130894): σ_R_nat = 0.032930580570026624, σ_R_refresh =
0.04663904004781301; V-P3b equivalence band **0.9161532348267848**, V-P3c floor
0.006731765581587645. The band is already the warning: it spans nearly the whole
interesting range of g_ratio, so at the registered size almost any ratio would
have been indistinguishable from zero.

**Why the projection fails — structural, and exact.** The estimand is a ratio
whose DENOMINATOR is the natural gradient itself. M1c's realized natural range
is 0.010391443071199338 and SE(range_nat) at 192 pairs is 0.0033609633054239685, so **the
denominator sits at 3.091804975802469 SE.** Dividing by a quantity known to
three standard errors gives a wide ratio however precisely the numerator is
measured:

| pairs/φ | truth g | projected g_ratio 95% CI | width | within 0.3 |
|---|---|---|---|---|
| 192 (registered) | 0.04 | [-1.0205225392166732, 1.3027058206019817] | 2.323228359818655 | False |
| 192 (registered) | 0.5 | [-0.4999284805257762, 1.9306216588125371] | 2.4305501393383135 | False |
| 384 (escalated) | 0.04 | [-0.6267371454071723, 0.7850089391170982] | 1.4117460845242706 | False |
| 384 (escalated) | 0.5 | [-0.15255945826573986, 1.3445368651754255] | 1.4970963234411654 | False |

The once-only escalation fired on the budget gate and did not rescue it
(True). **12288 pairs/φ — 64.0× the registered design —
would be required** at both truths. Per rule 25 and the routing, no measurement
world is spent against a failed feasibility gate.

**A named alternative, so the handback is actionable.** The DIFFERENCE
(range_ref itself, or range_nat − range_ref) does not divide by a small noisy
denominator: at 192 pairs/φ SE(range_ref) = 0.0047600770920988265,
so a range_ref equal to the natural range would sit at
2.1830409193262694
SE, and at 384 at
3.0872860753266402
SE. **NAMED FOR THE PLANNER, NOT CHOSEN** — the executor does not substitute
estimands.

**Not reached.** C1′ (the M1c anchor as certificate) and the five measurement
arms: the rule-25 gate fires before them. C1′ is therefore the right first test
on any re-dispatch, and the instrument is ready for it. R_deframe's stride was
measured and pinned at 1 in Part 0 (0.5978600978851318 s/pair
plain, 0.9944519996643066 s/pair with de-framing, using K-R1's transcription of
K1b's estimated repair), so the secondary reading would have run on every
measurement world.

**Rule events.** Rule 13: not reached. Rule 25: fired as designed — this is the
gate that stopped the leg. Rule 26: no bounded winner. Rule 27: the g_ratio
budget is what rule 25 projected against, unmet at both sizes. Rule 29: ran on
BOTH scorings at both pilot φ. Rule 30: every cited constant read from its
persisted source; the provenance table is generated from the extraction's own
line map.

**Executor self-report.** Three anomalies, all resolved before any
hypothesis-relevant number existed. A-1 the dispatched interpreter is absent (a
pinned CPython 3.12.12 venv built from the lockfile). A-2 `timeout(1)` is
absent on macOS (foreground stages under explicit sub-600 s timeouts). **A-3 a
wrong claim in my own draft, caught before any gate consumed it:** the first
version of this harness asserted the split builder would differ from k2b on
EVERY object at equal seeds. It does not — the author half matches bit-exactly.
The assertion was replaced with a generated per-object comparison, and the
finding is strictly stronger than the claim it replaced. No number changed.

**Registration-defect candidate (1), and it is the substantive one.** The
registration inherited P3's estimand VERBATIM — including its rule-27 budget of
0.30 on a RATIO whose denominator is M1c's natural range. That range
(0.010391443071199338) and the per-world noise that determines SE(range_nat) were
both already persisted and citable at registration time, so the projection's
failure was **computable before the leg was dispatched**: the denominator's
3.091804975802469-SE size fixes the ratio's width almost independently of the
numerator. This is the #43/#44/#51/#53 genus once more — a gate-consumed
quantity computable at registration and not computed — with the specific twist
that inheriting an estimand verbatim also inherits its feasibility arithmetic,
which changes when the instrument changes. Non-blocking: the leg's registered
routing handled it correctly and the instrument survives.

### Planner adjudication of P3b (2026-08-14, appended after the run) — THE INSTRUMENT IS DELIVERED; THE RATIO WAS THE WRONG ESTIMAND; INHERITANCE IS NOT EXEMPTION

**NON_PROJECTABLE accepted — rule 25 fired exactly as designed, and
the leg still delivered its standing asset: the certified split-seed
instrument.** Four edits over an 18-entry provenance map; C2a/C2b/C2c
all pass; and a STRONGER property than required: at equal seeds the
author half reproduces k2b BIT-EXACTLY (a_load, common, int, trait
identical; divergence localised to the frame stream's restart, which
is inherent to splitting). Exposing `loadings` closes the
undetectable-basis-mixing hole P3 named. The instrument is a
program asset independent of this leg's estimand.

**Defect #56 (mine — the #43/#44/#51/#53 genus, fifth appearance,
with a new twist).** Inheriting P3's estimand VERBATIM inherited a
rule-27 budget on a RATIO whose infeasibility was computable from
persisted artifacts before dispatch: the denominator is M1c's
natural range 0.010391443071199338, whose SE at 192 pairs
(0.0033609633054239685) puts it at 3.09 SE — a ratio with a 3-SE
denominator is wide no matter how precise the numerator (projected
widths 2.32/2.43 at 192; 1.41/1.50 at 384; 64× the design to meet
0.30). **Convention sharpened: INHERITANCE IS NOT EXEMPTION — a
verbatim-inherited estimand's feasibility arithmetic is re-run under
the new leg's conditions before the registration commits.** The
executor's named alternative is ADOPTED: DIFFERENCE estimands avoid
the noisy denominator entirely.

**Method note.** The executor's A-3 (its own draft claim about
all-objects-differing replaced by a GENERATED per-object comparison
before any gate consumed it, yielding the stronger bit-exact
finding) is the rule-30 culture operating executor-side.

---

## M4-P3c — the transportable gradient, by differences

**REGISTERED 2026-08-14, BEFORE RUN.** Planner: this document's
author; executor: dispatched agent. Instrument: P3b's certified
`build_split_world` (imported from
`scripts/run_suica_m4_p3b_refresh_gradient.py` by file with
provenance; the C2 battery re-run on 4 fresh probe pairs; C2c
included). Question unchanged (GG.3); ESTIMANDS changed to
differences:

- **range_nat** = R_nat(φ=.98) − R_nat(φ=.05) (M1c side-signing;
  co-measured in this leg);
- **range_ref** = R_refresh(φ=.98) − R_refresh(φ=.05) (the
  TRANSPORTABLE gradient);
- **D_grad** = range_nat − range_ref (the FRAME-OWNED component);
  all three with JOINT world-pair bootstrap (R_nat and R_refresh
  share the A-worlds — the correlation is part of the estimator,
  B = 2000; 20000 at rule-13 boundaries).

### Feasibility (planner arithmetic, rule 30 — executed inputs cited, scaling stated, exact gate in Part 0)

P3b persisted SE(range_ref) ≈ 0.0047580 and SE(range_nat) ≈
0.0033609633054239685 at 192 pairs/φ; both scale as 1/√n. At **768
pairs/φ** (the registered size): SE(range_ref) ≈ 0.0024,
SE(D_grad) ≤ √(SE_nat² + SE_ref²) ≈ 0.0029 (upper bound — the
shared-A correlation only helps), so a MOSTLY_FRAME truth (D_grad ≈
0.0104) sits at ≥ 3.5 SE and a TRANSPORTS truth (range_ref ≈
0.0104) at ≥ 4.3 SE. These are approximate planner values from
persisted inputs; **G3p3c's projection recomputes exactly (B_proj =
2000, both truths {FULLY_FRAME, TRANSPORTS}), PASS iff both
discriminations ≥ 0.8 power at 2·SE with the null-truth false-fire
≤ 0.1; once-only escalation to 1152 pairs/φ ON THIS GATE; else
NON_PROJECTABLE.**

### Design

share 0.25; φ ENDPOINTS {0.05, 0.98} × **768 A/B pairs** each +
φ = 0.60 × 192 pairs (a shape-sanity READING, no gate); total 1728
pairs = 3456 worlds. Salts `m4p3c-author` / `m4p3c-frameA` /
`m4p3c-frameB` / `m4p3c-pilot`, master_seed 20260814. R_deframe
secondary reading at stride 1 (P3b's measured cost 1.0 s/pair
absorbed in the budget). C1′ anchor now RUNS: R_nat's two endpoint
levels (and the 0.60 reading) vs M1c's share-.25 row,
distributional 2·√2·SEM — failure routes INSTRUMENT_DEFECT.

### Verdicts (rule 22; NULL-first per #55; equivalence bands from Part-0 realized noise, df-inflated)

- **V-A (frame-owned component):** D_grad vs 0 — POSITIVE (CI > 0)
  / NULL (CI inside ±ε_D) / NEGATIVE / UNDERPOWERED.
- **V-B (transportable component):** range_ref vs 0 — NULL-first
  (CI inside ±ε_r) / POSITIVE / NEGATIVE / UNDERPOWERED.
- Descriptive (no gate, no budget): the fraction
  range_ref/range_nat quoted ONLY as a point with its honest CI,
  labeled UNBUDGETED (the P3b lesson made visible in the report).

### Leans (sides declared)

**L-1p3c [FULLY_FRAME .50 / MIXED .25 / TRANSPORTS .10 / INVERSION
.05 / underpowered .10]** — P2's f carried across the stated
inferential gap.

### Routing (rule 16 — disjoint, covering, consequences entailed)

| # | condition | outcome |
|---|---|---|
| 1 | G0/C2 battery/citation failure | **STOP / INSTRUMENT_DEFECT** |
| 2 | projection fails after escalation | **NON_PROJECTABLE** |
| 3 | C1′ anchor fails | **INSTRUMENT_DEFECT** (never a world finding) |
| 4 | V-A POSITIVE and V-B NULL | **GRADIENT_FULLY_FRAME** — the natural φ-gradient is frame-owned; the M-line law stands as a law of the statistic; the mechanism section re-types (the r-channel reads frame-agreement) |
| 5 | V-A POSITIVE and V-B POSITIVE | **GRADIENT_MIXED** — both components real; the split is quoted descriptively |
| 6 | V-A NULL and V-B POSITIVE | **GRADIENT_TRANSPORTS** — the reading crosses frames; the scaffold-gradient strengthens |
| 7 | V-B NEGATIVE | **INVERSION_NAMED** — refreshment reverses the gradient; new phenomenon, theory note |
| 8 | any UNDERPOWERED among V-A/V-B (and no higher cell fires) | **UNDERPOWERED** (levels and bands reported) |
| 9 | V-A NEGATIVE (and V-B not NEGATIVE) | **FRAME_COMPONENT_NEGATIVE** — refreshed gradient exceeds natural; named, theory note |

### Deliverables and budget

`scripts/run_suica_m4_p3c_transportable_gradient.py`;
`results/m4_p3c_transportable_gradient/` (gitignored);
`reports/SUICA_M4_P3C_TRANSPORTABLE_GRADIENT_REPORT.md` (generated
tables); outcome append HERE; one ledger row (EXPLORATORY); exactly
ONE commit `feat(m4-p): P3c — the transportable gradient, by
differences — <SLUG>`, never amended, never pushed; suite green
first. 3456 worlds (+escalation ×1.5) + pilot; stages: part0 180 s,
pilot 60 s, worlds 6 chunks (~350 s each), score+fit 240 s,
finalize 60 s; target < 60 min, every stage < 600 s.

### M4-P3c outcome (appended after execution; registration above unedited)

**UNDERPOWERED — routing cell 8; modifiers: none.**
UNDERPOWERED (levels and bands reported). Harness
`scripts/run_suica_m4_p3c_transportable_gradient.py`; report
`reports/SUICA_M4_P3C_TRANSPORTABLE_GRADIENT_REPORT.md`; artifacts
`results/m4_p3c_transportable_gradient/` (gitignored). 3456 worlds
(768 A/B pairs per endpoint + 192 at the shape reading).

**V-A: D_grad = 0.00852887414363202 [0.005277986571389943, 0.011952276850539912] → POSITIVE.
V-B: range_ref = 0.0011121987595310255 [-0.001020386706397168, 0.003228730521677211] → UNDERPOWERED.**
Against range_nat = 0.009641072903163045 [0.00715774671522028, 0.01223467530439654].

**The frame-owned component is real and large; the transportable component
cannot be certified either way.** D_grad is 0.00852887414363202 — most of range_nat —
with a CI comfortably clear of zero. range_ref is 0.0011121987595310255, straddling
zero, and its CI is only *slightly* too wide to sit inside the
±0.001993500543087015 equivalence band, so it classifies UNDERPOWERED rather than
NULL. Cell 8 takes any UNDERPOWERED verdict, so the leg lands one notch short of
GRADIENT_FULLY_FRAME — which would have required V-B certifiable as NULL.

**The levels say more than the verdict does.** R_refresh is essentially zero at
every φ against an R_nat of 0.121–0.131:

| φ | role | n | R_nat mean | R_nat SEM | R_refresh mean | R_refresh SEM | R_deframe mean | within-pair corr |
|---|---|---|---|---|---|---|---|---|
| 0.05 | endpoint | 768 | 0.12141191457796047 | 0.0009692408241353337 | -0.0009178005907745429 | 0.0007756226258312586 | -0.0006611068355035473 | 0.007142710928294203 |
| 0.6 | shape reading | 192 | 0.127054822864402 | 0.0019654438197426417 | -0.0014144445255356564 | 0.0015808013006755244 | 0.0006918693173260006 | -0.0027490284889659074 |
| 0.98 | endpoint | 768 | 0.1310529874811235 | 0.0009013957608274527 | 0.00019439816875648275 | 0.0008406974743652825 | -0.0005750845212524437 | 0.0036923299323797887 |

The gauge scored against a DIFFERENT frame's truth agrees with it barely at all.
That is the substance; UNDERPOWERED reflects only that certifying
"indistinguishable from zero" is a stronger demand than observing "near zero".

**The instrument.** P3b's certified `build_split_world` IMPORTED by file from
`scripts/run_suica_m4_p3b_refresh_gradient.py:279` (function sha256 `a7618a321752fc50…`, file sha256
`b06c36bc3ff5a757…`) — one certified builder in the programme, nothing to drift.
Its C2 battery was re-run on 4 FRESH probe pairs and passed
(True): author objects bit-identical, every frame object differing
(`common` by a norm delta in [15.456355013034589, 15.926464179341583]),
determinism, shared basis. k2b and `suica_core/` untouched.

**The side-signing REVERSES from P3b, deliberately (RN-P3C-2).** P3b reported
range_nat = R(.05) − R(.98) = −0.0104; this registration defines
range = value(.98) − value(.05), so range_nat is **+0.010391443071199338** and
matches the registration's own truths. Stated because a silent sign flip
between sibling legs is exactly what survives review and then poisons a
comparison.

**C1′ anchor: PASS 3/3** (largest |pooled z| =
0.4769833641131134):

| φ | P3c R_nat | M1c mean | difference | tolerance | inside | pooled z |
|---|---|---|---|---|---|---|
| 0.05 | 0.12141191457796047 | 0.12162744485545209 | -0.00021553027749161846 | 0.005028599997733033 | True | -0.10643918235484917 |
| 0.6 | 0.127054822864402 | 0.12714790436588774 | -9.308150148573668e-05 | 0.004823490742495307 | True | -0.035770884964947285 |
| 0.98 | 0.1310529874811235 | 0.13201888792665142 | -0.0009659004455279119 | 0.005128889338998506 | True | -0.4769833641131134 |

The extracted two-stream instrument reproduces M1c's measured levels, so any
downstream miss would have been a finding rather than a plumbing artefact.
range_nat also co-measures M1c's range to -0.0007503701680362934.

**The joint bootstrap is what made this leg feasible where P3b's ratio was
not.** R_nat and R_refresh come from the SAME A-world (one gauge pass, two truth
panels), so the bootstrap resamples pair indices once per replicate and
recomputes both ranges from them. Realized bootstrap correlation between
range_nat and range_ref: 0.017673408969719043; realized SE(D_grad) 0.0017252853220331757
against 0.0017405070975400247 under independence.

**Projection (G3p3c) PASSED at the registered 768 pairs/φ, no escalation
(False):**

| truth | detected | power at 2 SE | null quantity | false fire | PASS |
|---|---|---|---|---|---|
| FULLY_FRAME | D_grad | 1.0 | range_ref | 0.05 | True |
| TRANSPORTS | range_ref | 1.0 | D_grad | 0.0445 | True |

**The pilot's luck, audited (RN-P3C-9, added before any verdict was read).** The
pilot is 4 pairs per endpoint, and an n = 4 correlation carries a standard error
of roughly 0.5. It shows: the pilot's ρ = 0.761471882823515
against a realized 0.0054175204303369955 — a miss of 0.756054362393178 — and a
pilot sd_R_refresh of 0.019532236530097353 against a
realized 0.022396388278681817. The projection used the
optimistic ρ and so understated SE(D_grad); it did not matter, because D_grad
sits far from its band either way. **Both verdicts are unchanged under bands
recomputed from the realized arms (True):** ε_D 0.0030995661223025234 →
0.0034505706440663515, ε_r 0.001993500543087015 → 0.002287673666505379, V-A POSITIVE and
V-B UNDERPOWERED under both. The Part-0 bands route, as registered; the realized
ones are reported so the luck is visible and auditable.

**The fraction, quoted and labelled.** range_ref/range_nat = 0.115360476028154
[-0.10768346212929444, 0.3468040491435675], width 0.4544875112728619 — **UNBUDGETED**, gating nothing and
routing nothing. It is shown because a reader will compute it anyway, and
showing its width is the P3b lesson made visible.

**Rule events.** Rule 13: 0 boundary events, B = 2000. Rule 25:
the gate passed at the registered size; its correlation input was badly
estimated by the 4-pair pilot, which is a power calculation and not a verdict.
Rule 26: no bounded winner. Rule 27: the only budgeted quantities are the
verdict CIs against their Part-0 bands; the fraction carries its UNBUDGETED
label. Rule 29: the domain-pinned predicate ran on BOTH scorings at every arm.
Rule 30: every cited constant read from its persisted source; the instrument
carries file, line and two sha256s.

**Executor self-report.** Two standing anomalies, both resolved before any
hypothesis-relevant number existed: A-1 the dispatched interpreter is absent (a
pinned CPython 3.12.12 venv built from the lockfile); A-2 `timeout(1)` is
absent on macOS (foreground stages under explicit sub-600 s timeouts).

**Registration-defect candidate (1).** The registration specifies a 4-pair
pilot and then feeds the pilot's realized noise — **including a correlation** —
into both the equivalence bands and the projection. A correlation estimated
from n = 4 has a standard error near 0.5, and here it missed by
0.756054362393178; the same pilot's sd_R_refresh understated the realized value
by roughly a factor of two even after the registered χ²(0.10) inflation, which
is why V-B lands UNDERPOWERED rather than NULL by a small margin. The df
inflation guards a variance, not a correlation, and nothing in the registration
guards the correlation at all. Non-blocking here — the verdicts are robust to
the band source and the leg reports both — but the convention worth adding is
that a pilot-estimated CORRELATION entering a projection needs either its own
inflation or a minimum n. This is the estimator-noise-floor family (rule 32)
applied to a second moment rather than a first.

### Planner adjudication of P3c (2026-08-14, appended after the run) — ONE NOTCH SHORT BY ROUTING, DECISIVE BY BOUNDS; THE LINE CLOSES

**UNDERPOWERED accepted as routed — and the bounds carry the
science.** V-A POSITIVE: the frame-owned component D_grad =
0.00852887414363202 [0.005277986571389943, 0.011952276850539912] is
real and is ~88% of the co-measured natural gradient
(0.009641072903163045 [0.00715774671522028, 0.01223467530439654]).
V-B UNDERPOWERED by a hair: range_ref = 0.0011121987595310255
[−0.001020386706397168, 0.003228730521677211] straddles zero with a
CI marginally wider than the ±ε_r band. The transportable share
(UNBUDGETED, as registered): 0.115360476028154 [−0.10768346212929444,
0.3468040491435675] — **at most about a third, point about a
ninth.** C1′ 3/3 (largest |z| 0.477) — the split instrument's
certificate now includes a live anchor.

**The reading the routing could not give (defect #58, mine).** The
LEVELS are the loudest fact: R_refresh ≈ 0 at every φ (−0.000918 /
−0.001414 / +0.000194 against R_nat 0.121–0.131), all inside P3b's
0.006731765581587645 floor — exactly P3's V-P3c sub-case
(NO_TRANSPORTABLE_READING). The P3c pivot to differences DROPPED
that still-measurable sub-case, so the most informative reading of
the leg exists only descriptively. **Convention: an estimand pivot
carries forward every sub-case still measurable under the new
estimands.** Had it been carried, the leg lands the named cell; as
registered, the record shows UNDERPOWERED plus this dated note.

**Defect #57 (mine).** The registration fed a 4-pair pilot's
CORRELATION into the projection and bands; realized ρ missed the
pilot's by 0.756 and the pilot sd understated by ~2× despite the
df-inflation — the executor's RN-P3C-9 recomputation left both
verdicts unchanged, disclosed before any verdict was read.
**Convention (rule-32 family, second moments): pilot-estimated
correlations are never consumed — projections and bands use
independence with a stated margin or worst-case sign; the df
inflation licenses variances only.**

**P3d, priced and not queued.** Certifying V-B NULL needs SE ≈
0.0006 on range_ref — ≈ 3.2× the pairs (≈ 2,470/φ, planner scaling
from the persisted SE 0.00108, stated approximate) for one label
upgrade on an already-bounded quantity. Named, not funded.

**THE LINE CLOSES.** Five legs: P1 SIGN_SCAFFOLD (+0.32/+0.35), P2
GENUINE_SCAFFOLD (f = 0.958/0.971 — the boost was 96–97%
frame-vs-frame), P3 INFEASIBLE_SPLIT (the instrument theorem), P3b
NON_PROJECTABLE (the certified split-seed instrument delivered; the
ratio estimand retired), P3c UNDERPOWERED (the bounds above; the
near-zero cross-frame levels observed). Synthesis:
`docs/SUICA_M4_P_PENALTY_MECHANISM_LINE_SYNTHESIS.md`; theory
appendix II. Defects #55–#58, all planner's, all non-blocking, each
buying a convention. Successor: the M4-Q transport line — the same
certified device pointed at the CARD readers.
