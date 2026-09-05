# GRAIN: Individual Language Texture Analysis

**Group-Referenced Analysis of Individual Natural-Language Expression**

**English** | [日本語](README.ja.md) | [简体中文](README.zh.md)

GRAIN is a foundational analysis method for text-based measurement research.
It treats a person's natural-language events as local grains observed at
multiple resolutions, then studies their reproducible distribution,
orientation, conditional response, and path structure relative to an explicit
reference population. Their organized multiscale pattern is the person's
**language texture**. GRAIN is not a scale and does not require the current
corpora to become one. Future text-based assessments may be constructed on
this foundation only after their own construct, population, validity, and use
evidence is supplied.

The earlier watermelon/rind metaphor remains part of the method's development
history: topic and situation are not packaging to strip away, because
condition selection and condition-specific response can carry person-relevant
information. The current GRAIN name generalizes that insight beyond one
slicing metaphor. See the [binding naming and compatibility policy](docs/GRAIN_NAMING_MIGRATION.md).
The public migration is recorded in the [v1.1.0 release notes](docs/RELEASE_NOTES_V110.md).
The tag-gate correction is recorded in [v1.1.1](docs/RELEASE_NOTES_V111.md).

## Vector inference research loop (2026-09-06)

The active research program studies how an LLM's internal representations use
author context and combine evidence during personality inference. The
[project retrospective](docs/GRAIN_PROJECT_RETROSPECTIVE_20260905.md) connects
Project Persona, MCD, the historical method versions, and the current GRAIN
theory. The [experiment plan](docs/GRAIN_VECTOR_INFERENCE_MECHANISM_PLAN_V1.md)
specifies layer measurements, context contrasts, activation interventions,
external links, resource limits, and deliverables. The first instrument cycle
is complete: 192 synthetic fact prompts, layer probes, 3,264 activation-patch
forwards, and an execution-shape numerical audit. See the
[research log](docs/GRAIN_VECTOR_RESEARCH_LOG.md) and
[VM01 report](reports/grain_vector_mechanisms/VM01_RESEARCH_REPORT.md).
The first natural-text pilot is also complete: 24 authors, matched histories,
shared-current controls, and nonoverlapping-history follow-ups. See the
[context report](reports/grain_vector_mechanisms/VM02_VM03_CONTEXT_REPORT.md).
It separates retained context from cross-comment author correspondence; it
does not establish personality validity. Historical release and lockbox
results retain their original scope.

The [same-material baseline comparison](reports/grain_vector_mechanisms/VM02_SAME_MATERIAL_BASELINES.md)
does not establish LLM superiority over E5 or TF-IDF. A new
[relational-binding pilot](reports/grain_vector_mechanisms/VM03_RELATIONAL_BINDING_REPORT.md)
separates additive pooling, contextual interaction, output calibration, and
distributed causal use. Explicit product kernels recover the controlled relation
without language-model training. These are synthetic mechanism results, not
improved real-person personality prediction. The full VM02-VM07 program remains open.

## Theory manuscript (2026-09-01)

The completed theory is now expressed as a journal-neutral theoretical and
methodological article in [English](docs/paper/GRAIN_THEORY_MANUSCRIPT_EN_V1.md),
[Japanese](docs/paper/GRAIN_THEORY_MANUSCRIPT_JA_V1.md), and
[Chinese](docs/paper/GRAIN_THEORY_MANUSCRIPT_ZH_V1.md). The
[trilingual crosswalk](docs/paper/GRAIN_THEORY_MANUSCRIPT_CROSSWALK.md) fixes
section, equation, terminology, and claim equivalence across the three drafts.
[Evidence provenance](docs/paper/GRAIN_THEORY_EVIDENCE_PROVENANCE.md) maps each
headline number to its primary repository report.
These drafts do not claim that GRAIN is a validated scale or clinical tool.

## What is in the legacy sealed release (v0.2.1)

The v0.2.1 release and its identifiers were sealed under the former project
name. They remain unchanged as historical evidence and are interpreted under
the current GRAIN theory.

- **V7 technical core** — operator-indexed relative text geometry, frozen
  geometry bundles, refusal rules, multiview calibration, uncertainty and
  future-data gates. The highest supported claim is
  `OPERATOR_INDEXED_RELATIVE_TEXT_GEOMETRY_WITHIN_DECLARED_DOMAIN`; it is not a
  personality, reliability, clinical, universal-language, or market claim.
- **Method** — `docs/THEORY.md` (rind model, three channels: choice / style /
  react), `docs/RULEBOOK.md` (binding experiment-design rules, each traced to
  measured evidence), `docs/VALIDATION_PLAN.md` (the P0-P5 falsification
  framework).
- **Worked example** — `docs/WORKED_EXAMPLE_MANUAL.md`: the complete,
  audited construction of a 19-construct battery + 12 choice axes on PANDORA
  (Reddit), with development-tier anchor performance (e.g., MBTI
  thinking-feeling ridge CV r = 0.346; Essays Big5 transport mean r ~ 0.144
  from 19 interpretable scores) and score-interpretation rules.
- **Audit trail** — `docs/CLAIMS_LEDGER.md`: every claim with status, seven
  adversarial audit rounds, retractions included. The ledger is the
  authoritative record; prose may not exceed ledger status.
- **AI operating standard** — `docs/AI_ANALYST_GUIDE.md`: role separation
  (scorer/builder/coder/auditor/interpreter/human), fixed prompts, guardrails
  G1-G11 (each traceable to a caught failure).
- **Sealed preregistration** — `docs/PREREGISTRATION.md`: the seal is the
  initial commit hash of this repository. **Opening #1 was performed on
  2026-07-07** under the frozen rules (commit-pinned script, adversarial
  pre-audit): the preregistered success rule **FAILED** (2/7 hypotheses at
  BH-FDR q<.05; the rule required >=4/7) — recorded in full in
  `reports/suica_lockbox_opening_1.md`. H2 (first-person -> Neuroticism,
  r=+0.111, q=.002) and H6 (politics/news choice -> Openness, r=+0.096,
  q=.006) confirmed at lockbox tier. One opening remains; the Essays
  confirm-half labels are still untouched, though its text was read by V8
  realtext loaders on 2026-07-29..30 and it is no longer a text-blind cohort
  (`docs/V8_PUSH_REMEDIATION_20260803.md`, item 2).
- **Code** — new work imports `grain_core/` and `grain_sim/`; the
  `suica_core/`, `suica_sim/`, and `run_suica_*` names remain compatibility
  identifiers for frozen experiments. The V7 release lockbox is independently
  verifiable without restricted corpora.

## State of main after the sealed release (2026-08-03)

> **Update (2026-08-10) — the identity era.** Main now also carries the
> Identity Theory (IDT, T1–T10 with appendices A–V:
> `docs/SUICA_IDENTITY_THEORY_V1.md`), its two empirical lines (M4-K,
> twelve legs; M4-L, three legs), a completed defense phase
> (prospective seal → adversarial verification with zero refutations →
> portable artifact lockbox → real-text governance R-G1..R-G8 → rule
> replay → the seal OPENED measure-first at 3/5 predicted), and the
> repaired level law. Two laws are now PREDICTIVE grade (the η-floor
> law hit 3/3 at unseen dimensions; the taxometer read a novel cell to
> 0.0042); the F2/F4/F5 attributions were corrected by dated notes; the
> map of it all is `docs/SUICA_V8_IDT_INTEGRATION.md`. Standing rules
> 1–24; defect registry #1–#42. Everything EXPLORATORY, synthetic,
> instrument-world — no person claims, ever.

> **Theory synthesis (2026-08-17) — the 4W person-interpreter map.**
> `docs/SUICA_4W_PERSON_INTERPRETER_THEORY.md` integrates the development
> lineage and the latest selection work as `Where` (opportunity/selection),
> `What` (conditional expression), `Who` (reference-relative reproducible
> deviation), and `When` (state/path dynamics). The four terms are analytical
> roles, not four factors. The document also incorporates M4-T1's exploratory
> finding that selection identity contains nested conditional innovations plus
> a continuous or higher-order tail; neither object is licensed as personality.

Main has moved past v0.2.1 in three tracked layers (the sealed release above
is unchanged and remains the verification anchor):

- **V8 unified theory system** — `docs/SUICA_UNIFIED_THEORY_SYSTEM_V8.md`
  (typed measurement chain with V/R/P output types and refusal-first
  governance), with the M3 micro–meso–macro event-kernel layer and the M4
  condition-manifold / composition-grammar layer;
  `docs/SUICA_THEORY_ROUTE_INDEX.md` is the navigable map,
  `docs/V8_THEORY_PAPER.md` the paper-form draft, `docs/V8_MANUAL.md` the
  practitioner manual, and `docs/V8_DEVELOPMENT_REPORT_JA.md` the development
  history (Japanese).
- **M4-D/E/F experimental line (closed)** — 16 registered experiments
  dissecting the composite-loop transport wall into four parts, with two
  constructive repairs (two-stage route-then-refit; lambda~1/n de-biasing),
  a typed R→V relation bridge with a perfect designed-null refusal record,
  and the D3 panel sizing law (size-only rescue infeasible at ~10^14×;
  composition must change). Ledger rows M4-D.1..M4-F1 in
  `docs/CLAIMS_LEDGER.md`.
- **Governance** — the V8 drop was adversarially reviewed before publication;
  all findings and their dispositions are in
  `docs/V8_PUSH_REMEDIATION_20260803.md` (notably: the PANDORA cross-scale
  bridge headline is POST_HOC_OPERATOR_SELECTED_EXPLORATORY and must be cited
  with its same-day null; the Essays confirm-half is no longer text-blind). A
  v0.3.0 numeric lockbox over V8-era headline numbers is queued.

## Quickstart (no data required)

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_suica.py          # 39 passed
python -m pytest -q -p no:cacheprovider          # release audit: 970 passed (2026-08-03; 318 at the v0.2.1 tag)
python scripts/verify_grain_v021_lockbox.py       # portable v0.2.1 seal
python scripts/run_grain_synthetic_ground_truth_v2.py   # P0: estimator
python scripts/run_grain_p0b_thin_cell_regime_v3.py     # P0-B: thin cells
```

These synthetic harnesses verify the entire estimator layer against planted
ground truth without any real data. To reproduce the worked example, obtain
the datasets per `docs/DATA_ACCESS.md` (this release ships no user text, no
user IDs; SHA-256 manifests allow byte-identical preparation checks).

## Current development route and foundation contract

The canonical integrated statement of the current system is
`docs/SUICA_UNIFIED_THEORY_SYSTEM_V8.md`. It defines GRAIN's type system,
reference-relative measurement spine, micro/meso/macro formulas, residual and
applicability theory, version lineage, evidence boundaries, and the procedure
for building future study-specific natural-language measurement systems.

The development-first map is
`docs/SUICA_THEORY_ROUTE_INDEX.md`. It reconstructs the MCD/H-MCD/V-HMCD,
V3/V4, V6, V7, V8, and M3 lineage; links every retained mathematical object;
and defines the next mechanism program around composition algebra, endogenous
opportunity, condition-manifold discovery, dynamic generators, multiscale
coarse-graining, residual discovery, and the individual-to-relation bridge.
The evidence constitution below constrains those experiments, but it does not
select the scientific development route.

The first mechanism-program result, M4-A, is now implemented as a finite
third-order mechanism-composition grammar. Across 8 repetitions of 9
synthetic worlds it
separated genuine synergy from redundancy and suppression, recovered sparse
pair/triple hyperedges (support F1 1.00), preserved planted author-parameter
geometry (mean Spearman 0.736), identified gate direction, refused perfect
aliases, and remained null-calibrated. The new object is an author
mechanism-composition signature rather than another factor score. This is a
matched multilinear discovery, not yet a general joint-kernel algebra. See
`docs/SUICA_M4_MECHANISM_COMPOSITION_THEORY.md`.

The M4-A dynamic extension then estimated condition-only and history-only
transition kernels and compared both composition orders on held-out joint
paths. It recovered resolvable order with 0.828 accuracy, planted commutator
geometry at 0.940, temporal gate strength at Spearman 1.00, and an independent
panel dynamic-signature AUC of 0.856. A commuting control retained marginal
author structure but correctly lost order and gate effects; a calibration-free
role alias refused. This is a designed synthetic kernel result, not automatic
identification from uncontrolled text.

M4-B then made the opportunity set endogenous. Its three-role estimator fits
selection, opportunity generation, expression response, return, and history
gate kernels on calibration data, selects a registered family on a separate
panel, and opens final evaluation only once. Across 8 repetitions, mechanism
macro F1 was 0.991, feedback-loop geometry was 0.920, and selection versus
creation remained 0.998 distinguishable despite matched raw-menu
half-\(L_1=0.006\) and choice TV 0.013. Return and recovery geometry exceeded
0.94, active author-signature AUC was 0.830, hidden-source aliases refused,
and null false positives were zero. The scalar spectral-radius ranking failed
(Spearman 0.473), so it remains diagnostic while the full ecology loop is
retained. See `docs/SUICA_M4_ENDOGENOUS_OPPORTUNITY_ECOLOGY_THEORY.md`.

M4-C now removes the planted-condition-coordinate assumption for one bounded
conversion edge. A response-safe chart is selected only from pre-response
context on independent reference panels, represented through source-specific
PCA/Isomap or a landmark geodesic atlas, and frozen before any response is
opened. Across 8 repetitions of 8 worlds, identifiable-chart resolution was
22/24, held-out geodesic geometry was 0.913, local-neighbour Jaccard was
0.774, and cross-source relation geometry was 0.963. The frozen chart retained
0.925 of oracle conditional-response \(R^2\) and 0.960 author-response
geometry. Declared author/response leakage, no-manifold worlds, condition
aliases, and a branched topology conflict were all refused or routed
correctly; response permutation changed no chart output. Topology labels
remain a registered graph/MST diagnostic (0.833 match), not a general topology
theorem. Full chart-covariant transport of M4-B selection, creation, return,
recovery, and history-gate operators was tested next in M4-C.2. See
`docs/SUICA_M4_CONDITION_MANIFOLD_THEORY.md`.

M4-C.2 V2 corrected two defects exposed by its preserved V1 development
NO-GO: realized rather than expected matched menus, and truth-open condition
alias information loss rather than an absolute skill cutoff. Across 8 fresh
repetitions of 15 worlds, the corrected alias audit passed 8/8; matched
condition, environment, and union-menu paths were exactly equal; source alias,
null, basis invariance, and response safety all passed. Atomic action transport
was supported for choice (.790), creation (.774), return (.845), and recovery
(.977). The composite loop failed (.652 versus the frozen .70 gate), however,
and only 2/8 repetitions passed every substantive gate. The frozen decision is
`M4_C2_NO_GO_CHART_TRANSPORT`: atomic edge recovery does not imply recovery of
their plug-in product. M4-D remains blocked. M4-C.3 now audits the physical
creation, response, and choice edges and nonlinear loop definition without
reopening V2. See
`docs/SUICA_M4_CHART_COVARIANT_ECOLOGY_THEORY.md` and
`docs/SUICA_M4_PHYSICAL_EDGE_COMPOSITION_THEORY.md`.

M4-C.3 showed that the physical product is algebraically correct and that
Jacobian-versus-finite geometry exceeds .998, but an absolute edge-error
budget did not match the author-relation endpoint. M4-C.3.1 repaired that
statistical-object mismatch: its centered relational Möbius budget predicted
composition loss at Spearman .9868 and allocated about 80% of symmetric loss
to creation. A shared RBF chart remapping then failed all 12 development arms.
The sealed cross-fitted Fisher-Wiener creation intervention gave a real but
incomplete improvement: loop geometry .7198 -> .7699 (gain .0501, clustered
LCB .0051), creation geometry .7347 -> .8054, hazard degradation 1.30%,
permutation gain negative, and null false success zero. It recovered only
25.4% of oracle headroom and qualified in 5/8 repetitions, so the frozen
decision remains `M4_C32_NO_GO_FISHER_WIENER_CREATION`. This supports a
partial author-derivative estimation-noise mechanism while redirecting the
next experiment to opportunity-budget scaling rather than more kernel or
embedding replacement.

M4-C.3.3 then increased observable creation information through nested
passive occasions and balanced orthogonal response excitation. The high/low
Fisher-information ratio was 13.04 (LCB 12.41), geometry rose with
log-information (slope LCB .0222), and the frozen endpoint improved by .0802
(LCB .0611) to geometry .7523. This establishes a real information-dose
effect, but not sufficiency: only 44.0% of oracle headroom was recovered and
5/8 repetitions passed, below the frozen 50% and 6/8 gates. Passive
opportunity supplied most of the gain (+.0611); excitation added +.0192 at
K=8; performance saturated by K=4 and varied sharply by mechanism world.
The decision is `M4_C33_NO_GO_INFORMATION_LIMIT`. The next bounded target is
creation/history heterogeneity, not more `K`, kernels, or embeddings. M4-D
remains blocked. See
`docs/SUICA_M4_RELATIONAL_ERROR_BUDGET_THEORY.md`,
`docs/SUICA_M4_CREATION_ESTIMATOR_INTERVENTION_THEORY.md`, and
`reports/SUICA_M4_C33_MECHANISM_DECOMPOSITION.md`.

M4-C.3.4 then held the gate-zero creation estimand fixed and crossed condition
chart (`C`), history/source observation likelihood (`S`), and cross-author
pooling (`P`) in an eight-cell attribution cube. The full cube recovered only
34.1% of oracle headroom, so the frozen decision is
`M4_C34_NO_GO_THREE_FACTOR_INCOMPLETE`. The decomposition is nevertheless
informative: `C` alone recovered 78.5% of headroom and was the separated
Shapley leader (`+.0727`, leading-margin LCB `+.0396`); `P` was small
(`+.0090`); and `S` was harmful (`-.0434` overall, target-world effect
`-.0844`, LCB `-.1581`, positive in `0/8` repetitions). Thus omitted
history/source likelihood is rejected as the dominant residual mechanism,
while condition-chart distortion becomes the next bounded target. The
positive `C` cell is truth-open and nondeployable, and the alias audit reached
only `7/8`, so no chart repair or transport closure has yet been demonstrated.
M4-C.3.5-R1 then confirmed a response-safe replicated-CCA quotient chart using
pre-response source/author views only. R2B tested that frozen chart in a
strictly separated downstream experiment. The conditional chart effect was
large and stable in the three registered accessible worlds (eligible gain
`.1227`, cluster LCB `.0934`; oracle-headroom recovery `.8146`, LCB `.6798`;
accessible efficiency `.9874`, LCB `.9801`). It also beat both equal-rank
controls, passed all CKA/permutation, history, compensation, null, alias,
source, gauge, and hazard gates, and produced zero false successes in 120 null
cells (Wilson upper `.0310`).

The frozen decision is nevertheless
`M4_C35_R2B_PARTIAL_CONDITIONAL_EFFECT`, not GO. Two of 160 native cells
triggered the preregistered `SUPPORT_SHIFT` refusal, including one of 40 main
cells, so the zero-refusal gate failed.

R2C then implemented the exact pre-response policy
`Pi = A R + (1-A) B0`. Its fresh 16-cell development replay was exact, but
all cells were accepted, so `Pi=R` and routing utility was unidentified. A
separate paired boundary-ecology intervention then crossed the frozen `.80`
coverage boundary while holding response truth fixed. The scalar gate was
counterproductive: refused cells had larger forced-`R` gain than accepted
cells (`.0880` versus `.0301`), and fallback lost `.0528` relative to always
using `R` (development interval `[-.0824,-.0232]`). Harm discrimination was
only AUC `.554`. Coverage remains an out-of-support warning, but no longer
licenses automatic `R/B0` routing.

A post-seal 24-component pre-response applicability signature improved
leave-one-world-out harm AUC from `.450` to `.588`, but its OOF
`R2=-.691` and routing value remained negative (`-.0616`). The signature is
therefore not resolved. The fresh fixed-coverage experiment then crossed
radial-dispersed, high-leverage-concentrated, tangential, and
source-asymmetric departures at exact counts `12/16` and `13/16`. Its audited
432-cell result is `M4_C35_PRE_RESPONSE_DOMAIN_NOT_IDENTIFIED`: all 72 null
gains and all replay errors were zero, but LOWO `Q` harm AUC was only `.533`
(LCB `.464`), gain `R2=-1.616`, and routing value over always-`R` was `-.0243`
(LCB `-.0417`). LORO likewise failed (AUC `.512`, `R2=-.195`, routing value
`-.0134`). The scalar learner selected `R` for every cell, while the historical
threshold lost `.0503` relative to always-`R`.

The current observables therefore do not identify a cell-level `R/B0` utility
gate. Applicability is frozen as a study-design license: inside an independently
supported design class use the registered chart; outside it report `B0/R`
sensitivity or refuse. No outcome-driven threshold tuning is authorized, and
M4-D remains blocked. See
`docs/SUICA_M4_C35_R2B_CONDITIONAL_CHART_PROTOCOL.md`,
`docs/SUICA_M4_C35_R2B_LOCKBOX_SEAL.md`, and
`reports/SUICA_M4_RESPONSE_SAFE_CHART_REPLACEMENT_R2B_CONFIRMATION.md`,
`reports/SUICA_M4_RESPONSE_SAFE_BOUNDARY_ECOLOGY_DEVELOPMENT.md`, and
`reports/SUICA_M4_VECTOR_APPLICABILITY_SIGNATURE_PILOT.md`,
`docs/SUICA_M4_C35_R2C_DIRECTIONAL_APPLICABILITY_DEVELOPMENT_PROTOCOL.md`,
`reports/SUICA_M4_DIRECTIONAL_APPLICABILITY_DEVELOPMENT.md`, and
`reports/SUICA_M4_DIRECTIONAL_APPLICABILITY_AUDIT_NOTE.md`.

The binding pre-experiment constitution is
`docs/SUICA_FOUNDATIONAL_CLOSURE_STANDARD.md`; known empirical openings are
tracked in `docs/SUICA_FOUNDATION_GAP_LEDGER.md`. It separates human process,
expression behavior, stored text, representation, anonymous technical
objects, reference-relative measurements, psychological interpretation, and
use/decision into eight licensed layers. Unknowns may remain open, but no
study may cross an open edge implicitly.

Validate a study contract before data or compute access:

```bash
python scripts/validate_suica_foundation_study.py \
  --contract configs/suica_foundation_study.smoke.json
```

Copy `configs/suica_foundation_study.template.json` for a new study. The
unfilled template intentionally returns `REFUSE_EXPERIMENT_START`.

The first bounded L4-to-L5 implementation is documented in
`docs/SUICA_L45_REFERENCE_FRAME_PROTOCOL.md`. Validate its contract and
reproduce the opened synthetic discovery with:

```bash
python scripts/validate_suica_foundation_study.py \
  --contract configs/suica_foundation_study.l45_reference_frame.json
python scripts/run_suica_l45_reference_frame.py \
  --config configs/suica_l45_reference_frame.discovery_v3.json
```

V3 passed all registered opened-discovery gates, including a minimum
condition-specific uncertainty-coverage lower bound of 0.925 and five
mandatory refusal rates of 1.0. The maximum licensed layer remains L4:
this is not a validated human score, personality construct, or application.

## Honest summary of evidence status

Confirmed at held-out grade (T3): choice-axis stability (5/5 axes,
shrinkage 0.027), 15/15 legacy operational score families on unseen users,
react signatures for 2 operational families under the stranger null.
Falsified and retired:
condition-mean centering (three independent ways), affect word-rates as
trait or occasion-state measures, attention-weight interpretation as
measurement evidence. Confirmatory (T4, lockbox opening #1, 2026-07-07):
the preregistered success rule FAILED (2/7; recorded in full, no
re-analysis) — first-person -> Neuroticism (r=+0.111) and politics/news
choice -> Openness (r=+0.096) are the two lockbox-confirmed relations;
tension, novelty, directive, venue entropy and gaming choice did not
confirm. One opening remains sealed. Scope: English, one platform + student
essays; dialogue and clinical use are designed but untested (OPEN_PROBLEMS
OP-7/8/14 — OP-7a register-proxy and OP-8 stage 1 closed 2026-07-07, see
CLAIMS_LEDGER).

V6 raw-text factor rediscovery (2026-07-13) did **not** promote a new factor
inventory. A discovery-only, label-free 24-dimensional text basis retained
distributed author information after opportunity control (Static
own-vs-stranger AUC 0.691), but Static, Hybrid, and Dynamic factor spaces all
failed author-disjoint confirmation. This is recorded as a boundary result,
not converted into post-hoc named constructs; see
`reports/V6_FACTOR_DISCOVERY_REPORT.md`.

V6 nonlinear-path audit (2026-07-14): a phase-coupled synthetic world proves
that linear mean/covariance/lag summaries need not exhaust path information;
however, neither the degree-two/three path-signature screen nor a held-out RFF
conditional-transition embedding detected an order-stable signal in unseen
PANDORA authors. The nonlinear object is therefore a validated **theoretical
candidate**, not a detected human-text construct. See
`docs/V6_NONLINEAR_PATH_OBJECTS.md`.

V7.2 operator-indexed multiview audit (2026-07-15): fixed cross-comment
slicing did not show a robust universal precision advantage, so slicing is
retained as one observation view rather than promoted as a universal best
operator. On a fresh 600-author, label-free condition/opportunity audit, total
source-disjoint author geometry (AUC .916 [.890, .938]), observed context
selection recurrence, and a **within-stratum exchangeable** matched partial
screen (AUC .834 [.766, .896], 86/109 retained authors) reproduced. The
matched screen’s selected Jaccard caliper was 0.0, so it is conditional on
metadata balance rather than proof of literal same-topic equivalence. The
recorded primary subreddit/time surface itself explained only R2=.0045
[.00024, .00768] and did not robustly transport to the total configuration.
This is evidence for operator-indexed author-relative text geometry with
unresolved condition accounting, **not** a personality factor, causal context
effect, or clinical score. See
`docs/V7_OPERATOR_INDEXED_MULTIVIEW_V71.md` and
`reports/V7_CONDITION_OPPORTUNITY.md`.

V7.2 registered operator family (2026-07-15): on 240 new authors excluding
all 1,020 earlier V7 authors, five operators and word/char representations
showed same-source transport in a synchronized max-T family. The decisive
boundary is source-disjoint replication: fine-grained linear coordinate maps
did not transport, but source-disjoint own-vs-stranger position alignment did
(AUC .719--.805; 48-comment sensitivity .791--.862). GRAIN therefore has
evidence for reproducible **relative text geometry**, not fixed named axes or
personality measurement. See `reports/V7_OPERATOR_FAMILY.md`.

V7.3 theoretical closure audit (2026-07-15): the primary object is no longer
an unidentified factor coordinate. It is a frozen, identifier-free landmark
distance profile under an operator- and reference-indexed regularized
Mahalanobis geometry. A real held-out smoke scored 98.6% of discovery, 92.2%
of calibration, and 79.2% of confirmation authors; out-of-reference rows are
explicit radial-envelope refusals, not a general OOD diagnosis. Corrected W2 simulations separate nonlinear geometry from
linear coordinates, W3 uses 25 random partitions, W4 uses split-specific null
spectra and paired method-difference intervals, and W7 proves that unpaired
cross-domain coordinates are not identified. All six corrected evidence
bundles passed their original run-manifest/inventory audit and are distributed
as deidentified aggregate snapshots with source hashes. Version 0.2.1 seals
`V7_THEORETICAL_CORE_CLOSED_WITH_EMPIRICAL_GATES`. Reliability, external
construct validity, and paired cross-domain transport remain separate
future-data gates. See `reports/V7_THEORY_CLOSURE_AUDIT.md`,
`docs/V7_EVIDENCE_STATUS_LATTICE.md`, and `docs/V7_LOCKBOX_V020.md`.

## V8 development line

V8 development has started on `main` after the sealed `v0.2.1` remediation
release. Its target is an **LLM-native, evidence-carrying assessment
framework**, not direct LLM personality judgment. A frozen LLM may become an
observation-level semantic transducer, while deterministic code remains the
final aggregator. Human-readable explanations must bind to numerical
artifacts, positive and counterevidence, counterfactual tests, support status,
joint uncertainty, and an explicit claim license. Adaptation agents may
propose new versioned candidates but cannot mutate or validate the active
instrument.

The authoritative integration map for the complete V8 mathematical line,
including V3.7G/H, the H.4D/R2 residual-completion sequence, and M3, is
`docs/V8_MATHEMATICAL_RESEARCH_ROUTE.md`. It distinguishes opened development,
sealed confirmation, real-text technical evidence, psychological construct
validity, and clinical use. The current integrated status is
`V8_EVENT_TO_RELATION_SEAM1B_SUPPORTED_SYNTHETIC__M3_V4_FORMAL_CONFIRMATION_OPEN__HUMAN_CONSTRUCT_VALIDITY_OPEN`.

M3-V4 mathematical development (2026-07-28) has reached
`M3_CROSS_FAMILY_V4_READY_FOR_CLEAN_SEAL`. A generator-blind battery recovered
author-dependent distribution, nonlinear response-operator, renewal-hazard,
time-reversal-current, and nonlinear-dynamics objects across a 12-seed
synthetic preflight while low-order controls, target knockouts, exact
observational aliases, and null worlds remained controlled. This is a
development-power result only: the 32-repeat external-randomness confirmation
is not sealed or opened, and it makes no human-text or psychological claim.
See `docs/SUICA_M3_CROSS_FAMILY_CONFIRMATION_PROTOCOL_V4.md` and
`reports/SUICA_M3_CROSS_FAMILY_PREFLIGHT_V4.md`.

Full-PANDORA exploratory external connection (2026-07-28): frozen,
label-blind V8 scoring covered 10,090 authors and retained 7,523
geometry-ready profiles. The invariant canonical score did not support
direct Big Five prediction (official-fold mean r=-.005) and was weak for
MBTI (mean AUC=.544). It did recover the held-out Big5-to-MBTI relation
matrix (element r=.498, permutation p=.0086, bootstrap 95% CI
[.383,.582]), exceeding a text-volume/format nuisance view by .289
[.125,.430]. This is an exploratory cross-scale structural connection on
previously opened labels, not a named construct or confirmatory personality
validity result. See `reports/V8_PANDORA_EXTERNAL_CONNECTION_REPORT.md`.

The hierarchical follow-up keeps the upstream author representation,
canonical relation geometry, and opportunity surface separate. Source-only
nested routing matched the upstream MBTI head (AUC .608 versus .607) but was
unstable and noninferior-failing for Big Five (mean r=.049 versus .062).
Its bridge relation was nonzero (element r=.463; source-head/bridge-author
refit CI [.140,.621]) but did not outperform upstream. The current decision is
`LAYER_SPECIALIZATION_PARTIAL_EXPLORATORY`: task-local layer access is a
supported architecture candidate, while hard winner-take-all routing and a
universal combined person score are rejected. See
`docs/V8_HIERARCHICAL_EVIDENCE_STACK_THEORY.md` and
`reports/V8_HIERARCHICAL_EVIDENCE_STACK_PANDORA.md`.

The subsequent label-free synthetic identifiability line now treats V8 as a
typed graph rather than a single score chain. `V8-HJIC-1` passed recovery,
commutation, coverage, alias refusal, frozen-map information order,
route-nonontology, confound refusal, and reference specificity, but remained
partial because its shared-relation gate was logically too strong. HJIC-1A
then separated a relation-matrix license from a unique-mode license; it
preserved a partial result after `SIMPSON_MIXTURE` was falsely licensed in
8/60 repetitions. The prospective HJIC-1B correction passed all 10 gates:
shared relation/mode license rates were .983/.800, low-gap relation/mode rates
were 1.000/.000, both global and localized Simpson effects were detected in
all repetitions with zero final licenses, high-dimensional irrelevant
nuisance caused zero vetoes, collider sensitivity was always vetoed, and
synthetic truth was never read by licensing. HJIC-1B licenses only a
condition-invariance veto, not causal deconfounding or personality validity.
HJIC-1C then implemented the context-fibered relation field and an explicit
ecological decomposition. In 60 frozen repetitions with 999 null permutations
and 999 author bootstraps, all 13 gates passed. The covariance identity
\(J_T=J_W+J_B\) held to q95 error \(1.94\times10^{-15}\); pooled nominal 95%
coverage was .943. Global invariance, sign-reversal cancellation, ecological-
only structure, composition reweighting, in/out-of-sieve nuisance behavior,
low-singular-gap relation without a unique mode, and the context-reliability
frontier were all classified at or above their registered gates. Ecological-
only worlds received zero individual-relation licenses, and truth usage by
licensing was zero.

The subsequent event-to-relation seam preserved two informative partial
results before reaching a pass. SEAM-1 passed 9/10 gate families but detected
ecological structure in only .850 of repetitions because it incorrectly
reused the max-context local null for the single between-context operator.
SEAM-1A introduced a separate D0 between-null and raised ecological detection
to .967 with zero individual, held-out-null, or mismatch false ecological
licenses; it remained partial because the simulation had also silently
changed the latent-to-event measurement support between D0 and confirmation.
SEAM-1B made that assumption observable. It calibrates the worst principal-
angle alignment of raw replicated M3 supports, separately refuses support
non-invariance and support underresolution, and accepts rotations inside the
same support. In the independent 60x999 formal run, all 13 gate families
passed: clean seam and ecological detection were 1.000, both support-drift
worlds were detected 1.000 with zero final seam licenses, support
underresolution was detected 1.000, within-support-gauge false refusal was
.017, population coverage was .924, and truth usage remained zero.

This closes the registered synthetic event-to-M3-to-macro relation seam under
an auditable support-invariance requirement. It does not close M3-V4's
separate formal confirmation, identify a transport map after support drift,
or establish psychological validity. See
`docs/V8_HIERARCHICAL_IDENTIFIABILITY_PROTOCOL.md`,
`docs/V8_RELATIONAL_LIFT_SUPPORT_PROTOCOL.md`,
`docs/V8_NUISANCE_INVARIANCE_VETO_PROTOCOL.md`,
`docs/V8_CONTEXT_FIBERED_RELATION_PROTOCOL.md`,
`docs/V8_M3_HJIC_SEAM_PROTOCOL.md`,
`docs/V8_M3_HJIC_SEAM_1A_PROTOCOL.md`,
`docs/V8_M3_HJIC_SEAM_1B_PROTOCOL.md`, and the corresponding reports.

The current mathematical measurement object is formalized in
`docs/V8_CONDITIONAL_RESPONSE_GEOMETRY_THEORY.md`. It separates C1 opportunity
selection, C2 shared-condition response, population hierarchy/topology,
stable residual structure, uncertainty, and evidence. Anonymous technical
objects are not personality constructs until a later external-validity
mapping is supported.
The complete event-to-relation type graph, assumptions, estimands, and refusal
branches are consolidated in
`docs/V8_MICRO_MESO_MACRO_SEAM_CLOSURE.md`.

The initial V8.0 theory and executable refusal contract are documented in
`docs/V8_LLM_NATIVE_ASSESSMENT_THEORY.md` and
`docs/V8_EXECUTION_PLAN.md`. This is a technical development scaffold; it
adds no personality, state, reliability, clinical, or external-validity
result. The dev3 contract binds the prompt, semantic event schema,
deterministic aggregator, active/candidate manifests, promotion ledger, and
auditor attestation to verified hashes and an external active-bundle trust
root. Auditor, language-license, evidence-ledger, and clinical-signoff
artifacts require Ed25519 verification against a caller-supplied authority
registry, with signatures scoped to the exact contract, bundle, evidence root,
certificate, and claims they authorize. Its explanation certificate uses
typed evidence nodes, closed provenance links, paired counterfactual tests,
controlled language templates, and artifact-bound evidence licenses.
The V8.0 scaffold passed its independent adversarial contract audit; this
closes the technical boundary only and does not establish psychological or
clinical validity.

The full currently executable V8.1-V8.4 battery was completed on 2026-07-24.
The overall decision is `V8_TECHNICAL_CORE_NOT_CLOSED`. V8.2 explanation
fidelity and V8.3 planted-world component identification passed. V8.1 achieved
200/200 valid outputs, macro-F1 0.9981, minimum ICC 0.9734, and cross
prompt/model CKA 0.9992, but failed the frozen run-drift gate
(0.1196 > 0.10), so the LLM remains renderer-only. In V8.4, the semantic
channel did not improve frozen PANDORA V7 geometry
(\(\Delta\)AUC=-0.0368, 95% CI [-0.1658, 0.0895]) and all four real-text run
CKAs were below 0.70. MEPS showed a local +0.1061 AUC increment
[0.0076, 0.2273], but its run CKA was only 0.5534 and the result is not
promoted. See `reports/V8_FULL_TECHNICAL_EXPERIMENT_REPORT.md` and
`reports/V8_FULL_PROVENANCE_AUDIT.md`; the provenance audit passed 70/70
checks. See also `docs/V8_EXECUTION_PLAN.md`. V8.5 human repeated-measurement
and external construct validation were not run.

A subsequent constrained-interpreter experiment implemented the corrected
role split in which GRAIN is the sole numerical channel and the LLM only
observes explicit behavior, renders compiled evidence-bound atoms, and audits
them. The 200-profile planted formal gate passed (observer macro-F1 0.943,
Fleiss kappa 0.988, interpretation coverage 0.988, evidence-edge F1 lower
bound 0.967). PANDORA runtime and repeated-input checks also passed, but the
formal real-text decision remained `V8_INTERPRETER_RENDERER_ONLY`: all 14
discovery links collapsed to `novelty_expression`, only 3/80 held-out
half-profiles had candidates, same-author AUC was 0.500, and neighbor AUC was
0.525. The interpreter exactly passed through the deterministic candidate set,
so it added no demonstrated selection information. Secondary corpora were
stopped after the primary gate failed. See
`reports/V8_INTERPRETER_STABILITY_REPORT.md`.

The next label-free method audit localized an upstream geometry bottleneck.
The frozen sorted landmark-distance profile had effective rank 1.446 and
discarded which structurally distinct landmark generated each distance. A
permutation- and Euclidean-isometry-invariant canonical coordinate was built
from intrinsic landmark-distance fingerprints; all 16 current landmarks had
unique fingerprints. Its scale-residual form reached source-disjoint
same-author cosine AUC 0.774, with author-cluster estimate 0.796
[0.687, 0.886], versus 0.498 for sorted distances. The paired delta was
+0.280 [0.154, 0.387], scale-matched AUC was 0.754, and topology-coupling
shuffle reduced AUC by 0.271 to 0.503. This is an opened-panel exploratory
mechanism result, not a personality or clinical-validity result. A fresh
registered author panel is required before changing the frozen V7 geometry or
reconnecting the behavior interpreter. See
`reports/V8_CANONICAL_ORBIT_GEOMETRY_REPORT.md`.

That fresh internal gate has now been run. All 240 V7 eligible authors were
excluded before deterministic sampling from the PANDORA raw-comment pool.
Among 156 source-eligible authors, 89 passed frozen geometry support.
Canonical scale-residual cosine AUC was 0.691, with author-cluster estimate
0.689 [0.636, 0.740] and permutation p=0.00010; sorted distance was 0.497.
The paired gain was +0.190 [0.118, 0.261], the gain over spectral energy was
+0.109 [0.056, 0.161], and topology shuffling removed 0.190 AUC. Decision:
`V8_CANONICAL_SCALE_RESIDUAL_INTERNAL_FRESH_PASS`. This confirms the
author-geometry mechanism within PANDORA; it still does not establish a
personality construct, cross-corpus transport, or clinical validity. The next
bounded step is to reconnect this frozen geometry to the existing behavior
cache without new LLM calls.

That reconnection is complete. Canonical geometry self-author AUC was 0.735,
but the unchanged three-segment/six-event behavior view remained at 0.556.
Cross-modal AUC was 0.547 [0.447, 0.616], distance alignment was 0.143
(p=0.073), and no stable registered links survived. Decision:
`V8_CANONICAL_GEOMETRY_PASS_BEHAVIOR_VIEW_STOP`. The geometry mechanism is no
longer the active bottleneck; the current explicit behavior object is. Prompt
or renderer tuning is not justified. See
`reports/V8_CANONICAL_BEHAVIOR_BRIDGE_REPORT.md`.

The behavior bottleneck has now been decomposed. A 24-author,
12-comment-per-half high-resolution observer pilot achieved repeated atomic
event macro-F1 0.849, but the preregistered condition-residual object failed
(AUC 0.385 at twelve comments). No-call diagnostics showed that strict
observer intersection and sparse exact-subreddit residualization removed the
stable author structure. A deterministic five-family hierarchy over the same
explicit evidence reached opened-panel AUC 0.699 [0.554, 0.798], p=0.0010;
the two observer repetitions were 0.669/0.683 and cross-model macro-F1 was
0.771. Its nested 3/6/9/12-comment curve rose
0.589/0.639/0.679/0.699.

Decision:
`V8_BEHAVIOR_V21_CANDIDATE_FROZEN_CONDITION_UNRESOLVED`. The result supports
a technical author-relative choice-and-expression candidate, not personality,
emotion, diagnosis, or condition-invariant response. Human coding and a fresh
author panel are required before any geometry bridge. See
`reports/V8_BEHAVIOR_V21_CANDIDATE_REPORT.md`.

The next gate is executable without more model calls. Its blind human-coding
packet contains 48 training, 336 natural-prevalence, and 192
boundary-enriched segments; each coder receives 528 opaque items, and 192
segments have hidden independent-model predictions for audit. The evaluator
has explicit waiting, adjudication, pass, and stop states and cannot calculate
observer accuracy from blank files. Fresh-author and C1/C2 thresholds are
frozen in `configs/v8_behavior_v21_fresh_protocol.json`. Human coding has not
yet been performed, so the scientific result remains the opened technical
candidate. See `docs/V8_BEHAVIOR_V21_HUMAN_AND_FRESH_PROTOCOL.md`.

A no-call fresh-panel preflight also passed the author/thread support gate:
300/300 scanned candidates supported two `link_id`-disjoint 32-comment
halves, and 190 authors were frozen as 60/20/80/30
discovery/calibration/confirmation/reserve. However, exact-condition C2 is not
identifiable in Reddit: only five supported subreddits covered 15.8% of
discovery segments, and confirmation structural completeness under those
shared conditions was 0%. The fresh run may therefore test only the joint/C1
object after human coding; C2 requires fixed-condition material. See
`reports/V8_BEHAVIOR_V21_FRESH_PREFLIGHT_REPORT.md`.

The available 46-person MEPS material was also audited without reading raw
text. It has three shared prompts and exactly three direct-answer slots per
participant/condition, but no independent repeat of a condition. Zero
participants meet the six-opportunity minimum across all prompts; only 2/46
meet six AI-chat turns across all prompts. MEPS is therefore an ontology and
procedure pilot only, not C2 confirmation. See
`reports/V8_BEHAVIOR_V21_MEPS_C2_SUPPORT_AUDIT.md`.

The C2 estimator itself has now passed an independent final planted-world
battery: `V8_BEHAVIOR_C2_PLANTED_WORLD_PASS` (70/70 registered checks).
The frozen binary score used exactly six observations per condition and
excluded author-specific standard errors. At SNR 0.50 it achieved same-author
AUC 0.951, while the separate Firth link-scale inference track recovered the
planted operator with median cosine 0.843, author-distance Spearman 0.660,
recovery slope 0.993, analytic CI coverage 0.968, and bootstrap-t coverage
0.904. C1-only information remained null on the C2 score (AUC 0.499), and
unshared, insufficient, or inconsistent condition coordinates refused
output. This is a technical identification result only; human observer
accuracy and real fixed-condition C2 remain open. See
`reports/V8_BEHAVIOR_C2_PLANTED_WORLD_REPORT.md`.

A further hierarchical limit experiment tested whether apparent author
structure survives when within-group individuality tends to zero:

\[
B_u=B_0+G_{k(u)}+\varepsilon D_u.
\]

All registered core checks passed. In the group-only world, ordinary
same-author AUC remained 0.864 but within-group same-author AUC was 0.500,
proving that ordinary author matching can be entirely explained by group
membership. In the individual-only world, within-group AUC was 0.866. Stable
residual energy followed the expected square law at the oracle level
(\(\varepsilon^{1.999}\)) and a compressed but monotone law in the estimated
score (\(\varepsilon^{1.687}\)); the current design's minimum detectable
individuality was \(\varepsilon=0.50\). C1-only group structure and
single-observer artifacts were both correctly separated from C2 individual
structure. That experiment left the discrete-versus-continuous topology
comparison unresolved. See `reports/V8_VANISHING_INDIVIDUALITY_REPORT.md`.

That topology question has now been closed in planted space, but not in real
text. The first topology-margin rule stopped after ring-to-curve errors; the
revised independent-seed open-set rule passed at its registered low-noise
margin. Minimum clear-world accuracy was 0.894 at N=320 and 0.926 at N=640,
all registered ambiguous primary worlds were refused, and the primary
wrong-decisive/null one-sided 95% upper bounds were 0.00597. Conditional
affine/permutation invariance was 0.974-0.982. Sensitivity is deliberately
reported as a boundary: at noise ratio 0.25 every clear world was refused.
Decision: `V8_C2_TOPOLOGY_MARGIN_PLANTED_PASS`. This distinguishes technical
population geometries in simulation; it does not discover personality types
or establish real-text topology. See `reports/V8_TOPOLOGY_MARGIN_REPORT.md`
and `reports/V8_TOPOLOGY_MARGIN_INDEPENDENT_AUDIT.md`.

The next affine response-set experiment represented each author as a bounded
line, surface, or volume under shared conditions rather than one point. Its
audited code gate passed: same-condition/free-image AUC 1.000/1.000, relation
balanced accuracy 0.9657, finite-scale dimension error 0.0863,
map-equivalence graph F1 1.000, cross-view Jaccard 0.9979, and attack refusal
1.000. A shared condition anchor generated a complete 120-edge raw graph but
zero pair-specific map-equivalence edges, so geometric intersection alone
cannot be interpreted as interpersonal affinity. Independent methodology
decision: `PARTIAL`; the gate covers fixed low-noise affine templates only,
and nonlinear persistence, transversality, real text, and psychological
interpretation remain closed. See
`reports/V8_RESPONSE_SET_INCIDENCE_REPORT.md`.

The nonlinear incidence-multiplicity extension has now passed its final
V2.3 planted battery. Four clear response-group worlds had minimum median
AUC/F1/ARI 1.000/1.000/1.000 and minimum claim rate 0.950; five null worlds
each had 0/160 group claims (worst one-sided upper 0.0185), isolated
transverse contacts had 0/160 claims, and all registered attacks refused.
A global common anchor connected all 24 author trajectories but produced no
group claim, while true groups were recovered at multiplicity 6. Thus
intersection degree alone is not similarity; the candidate object is a
condition-matched, scale-persistent, cross-view-stable, population-specific
incidence hyperedge. Decision:
`V8_INCIDENCE_MULTIPLICITY_PLANTED_PASS`. This remains a simulated technical
code-gate result, not evidence that real-text intersections are personality
similarity. Independent methodology decision is `PARTIAL`: a fitted
whole-map distance baseline also obtained AUC 1.0 in every clear and
challenge world, so incidence has not yet shown incremental information.
See `reports/V8_INCIDENCE_MULTIPLICITY_REPORT.md` and
`reports/V8_INCIDENCE_MULTIPLICITY_INDEPENDENT_AUDIT.md`.

The distance-matched V3/V3.1 battery has now isolated what that object adds.
Positive and counterfactual path populations were constructed with identical
whole-map distance geometry: whole-map AUC was 0.501-0.504, while
condition-aligned persistent-subset AUC, F1, and ARI were all 1.000 in three
registered positive worlds. Positive group claims were 600/600;
asynchronous, overlap-chain, rotating-membership, and two null worlds each
had 0/200 group claims (one-sided 95% upper 0.01487). V3.1 also repaired
hidden persistent-subset recovery and required cross-view support at the same
condition and radius. Independent decision:
`PASS_TECHNICAL_PLANTED_ONLY`.

The result is narrower than “intersections are personality similarity.”
Minimum same-condition distance and a soft local kernel also achieved AUC
approximately 1.0 in the positive worlds. What is established is that local
condition-aligned structure can survive after integrated whole-path distance
has been neutralized, and that a no-chaining hypergraph can express a
simultaneous multi-author core. Raw line count is not informative: a common
anchor can connect all authors without a group-specific claim. See
`reports/V8_INCIDENCE_INCREMENTAL_V31_REPORT.md`.

The V3.2/V3.3 strong-chain sequence then separated higher-order geometry from
simple information loss. V3.2 showed that collapsing condition-indexed
recurrence into one aggregate pair matrix destroys concurrency information.
V3.3 gave three pairwise comparators the complete condition-radius adjacency
tensor; all three recovered 500/500 simultaneous cores and rejected 500/500
strong chains. The accepted theorem is therefore about retaining condition
indices, not generic hypergraph superiority. See
`reports/V8_INCIDENCE_STRONG_CHAIN_V32_INDEPENDENT_AUDIT.md` and
`reports/V8_INCIDENCE_GRAPH_FAIRNESS_V33_INDEPENDENT_AUDIT.md`.

V3.4 isolated one genuinely higher-order object. Two paired worlds had
exactly the same author-pair adjacency in every condition, radius, and
observer view. In one, each planted six-author set fit inside one common
ball; in the other, all pairs were still adjacent but the six points had no
common ball. Over 500 fresh pairs, tensor mismatches were 0, pairwise outputs
matched 500/500, the common-ball estimator recovered the feasible six-author
sets 500/500, and made 0/500 six-author claims in the infeasible world
(one-sided 95% upper 0.00597). All four 200-run controls passed.
Decision: `V8_COMMON_BALL_INCREMENTAL_V34_PASS`.

This gives a precise meaning to an intersection: it is a recurring
multi-author joint-feasibility relation, not a count of incident lines.
Multiplicity alone is not similarity, and even this stronger relation is
not yet personality similarity. Full continuous pairwise distances already
determine minimum-enclosing-ball geometry; the incremental result is only
beyond a finite thresholded binary adjacency tensor in the registered
two-dimensional simulator. Real-text existence and psychological validity
remain closed. See `reports/V8_COMMON_BALL_INCREMENTAL_V34_REPORT.md` and
`reports/V8_COMMON_BALL_INCREMENTAL_V34_INDEPENDENT_AUDIT.md`.

V3.5 executed the previously missing planted-world geometry controls.
Exact common-ball birth scales agreed with numerical truth to
`8.88e-16`; continuous decisions agreed with 9/17/33-point radius grids in
all cases; persistent groups were recovered 200/200 while matched nulls
produced 0/200 group claims. Local curve/surface relations were correctly
classified 2200/2200, all 400 boundary cases were refused, and held-out
nonlinear sheets had AUC 1.000. Decision:
`V8_SCALE_NULL_TRANSVERSAL_MANIFOLD_V35_PASS`; independent decision:
`PARTIAL_TECHNICAL_PASS`. The nonlinear AUC is a local transversality score,
not general manifold reconstruction, and intersection dimension was not
independently recovered. These perfect low-noise planted results validate a
narrow estimator battery, not real-text geometry.
See `reports/V8_SCALE_NULL_TRANSVERSAL_MANIFOLD_V35_REPORT.md`.

V3.6 then separated a junction's geometry from its temporal routing kernel.
The same spacetime junctions supported pass-through, random, and cue-guided
routing, each recovered 500/500. Static marginals could not identify the
world (accuracy 0.316), while event-level transition information did.
Spatial coordinates alone collapsed three repeated stages (accuracy 0.333);
adding time recovered their order perfectly. Cue-guided routing made all 27
depth-three leaves addressable; random routing visited almost the whole tree
but had near-zero channel capacity and goal accuracy near `1/27`. A guided
near-miss retained perfect cue routing while failing the common-junction
gate. The tree/addressability result is partly entailed by the deterministic
generator, and explicit time recovery is a construction check. The supported
statement is that the registered common-ball criterion and cue routing are
logically independent; it is not a thought or intelligence claim. Frozen
decision: `V8_SPACETIME_JUNCTION_FLOW_V36_PASS`; independent decision:
`REGISTERED_DGP_TECHNICAL_PASS / TREE_CAPACITY_PARTIAL`. See
`reports/V8_SPACETIME_JUNCTION_FLOW_V36_REPORT.md`. The V3.6.1 post-seal
scope audit removed sparse path-MI bias: random raw conditional MI 0.411
became -0.0001 after matched permutation correction, while cue-guided routing
retained adjusted MI 0.451 and generalized perfectly to held-out cue-path
combinations. This remains deterministic simulator evidence, not planning or
intelligence; see
`reports/V8_SPACETIME_JUNCTION_FLOW_V361_SCOPE_CORRECTION_REPORT.md`.

V3.7A then made routing author-specific. In a sealed hidden-low-rank synthetic
world, the anonymous author-relative routing operator reached truth
correlation 0.862, split-session reliability 0.650, unseen-context reliability
0.824, and within-group AUC 0.963 across 200 confirmation populations.
Group-only structure raised ordinary author AUC to 0.848 but left within-group
AUC at 0.500, demonstrating why group-conditioned scoring is necessary.
Independent decision: `PASS_WITH_MAJOR_CAVEATS`. The core low-rank recovery
result stands, but variance-source attribution, formal single-author
confidence intervals, fully group-free recovery, and an information-theoretic
sample limit remain open. See
`reports/V8_AUTHOR_ROUTING_OPERATOR_V37A_REPORT.md`.

V3.7B removed the oracle junction window. A geometry-and-order-only locator
recovered the registered pause-plus-perpendicular-cusp family with F1 0.9996
and P95 location error one sample. The blind operator correlated 0.959 with
the oracle-window operator and retained 1.001 of its held-out predictive gain;
cue-leak author claims were 0/200. Canonical decision:
`V8_BLIND_JUNCTION_LOCALIZATION_V37B_PASS`; independent decision:
`PASS_WITH_CAVEATS`. This is a narrow synthetic signature-localization
result. The oracle/blind comparison reused the same events, mask stress was
negligible, and the planted signature closely matched the locator. It is not
open-world event discovery or evidence for thought nodes, intelligence,
personality, clinical meaning, or real text. See
`reports/V8_BLIND_JUNCTION_LOCALIZATION_V37B_REPORT.md` and
`reports/V8_BLIND_JUNCTION_LOCALIZATION_V37B_INDEPENDENT_AUDIT.md`.

V3.7C then removed true-group centering, separated blind and oracle event
panels, added transported path families, and registered MAR/MNAR analyses.
All 20 sealed canonical gates passed. In the core arm, group-free operator
truth correlation was 0.963, independent-oracle correlation 0.935,
split-session reliability 0.740, unseen-context reliability 0.936, and
hard-neighbor author AUC 0.932. A 32-event arm fell to truth correlation
0.526 and unseen-context reliability 0.363, establishing a finite-budget
boundary. Out-of-family and high-noise locator F1 remained 0.984 and 0.962.
Under registered MAR, AIPW excess NRMSE was 0.021 versus 0.103 for
available-case estimation.

The key V3.7C result is a separation between operator recovery and individual
identification. Across true ranks 2/4/6/8, truth correlation stayed near 0.96,
while hard-neighbor AUC changed from 0.777 to 0.961, 0.992, and 0.998.
Low-dimensional author spaces can therefore be measured accurately yet remain
locally crowded. Independent decision: `PASS_WITH_CAVEATS`. Rank 8 and lambda
0.3 are working hyperparameters, MNAR remains a broad sensitivity envelope,
and transport is limited to registered synthetic path families. See
`reports/V8_GROUP_FREE_ROUTING_TRANSPORT_V37C_REPORT.md` and
`reports/V8_GROUP_FREE_ROUTING_TRANSPORT_V37C_INDEPENDENT_AUDIT.md`.

V3.7D mapped the finite-sample boundary between recovery and identity. An
initial in-cohort analysis was rejected after showing mean hard-AUC optimism
of 0.098. The corrected exploration and sealed 40-world confirmation fitted
the stable basis on 96 authors and evaluated entirely disjoint, nested author
sets. All 13 confirmation gates passed. At 128 events, rank-2 truth
correlation stayed near 0.98 while hard-neighbor AUC fell from 0.892 to 0.605
as evaluation authors increased from 32 to 192; the paired AUC loss was
0.288 [0.281, 0.294]. Conversely, a fixed rank-8 estimator retained rank-12
identity AUC above 0.93 while truth correlation stayed near 0.77; a
true-rank sensitivity restored truth correlation to 0.940. At rank 8,
recovery and identity both passed in every 128-event population. Recovery and
identity are therefore bidirectionally separable, not nested, in the
registered synthetic design. This is not personality or real-text evidence.
See
`reports/V8_DIMENSION_DENSITY_EVENT_BUDGET_V37D_CONFIRMATION_REPORT.md` and
`reports/V8_DIMENSION_DENSITY_EVENT_BUDGET_V37D_INDEPENDENT_AUDIT.md`.

V3.7E then tested the strongest alternative explanation for the reverse
rank-12 result: fixed estimator under-capacity. A four-fold author-level
one-standard-error selector used 128 calibration authors, while 256
external-reference and 96 evaluation authors remained disjoint. All 16
sealed gates passed. In 60 flat-spectrum worlds, rank 4/8/12/16 was selected
exactly. At rank 12, fixed rank8 truth correlation was 0.783 and
identity-without-recovery occurred in 93.3% of worlds; adaptive rank raised
truth correlation to 0.952 and changed all worlds to
recovery-plus-identity. Opportunity standardization reduced eight-panel
systematic drift by 0.686 [0.669, 0.703], and a planted population shift was
recovered with direction cosine 0.998 and projected amplitude 1.006. Decision:
`PASS_MISSPECIFICATION_EXPLAINED`. The V3.7D reverse example was a valid
fixed-rank counterexample but did not persist under adaptive capacity. The
external router supports contrast transport; because the projection center
still comes from calibration authors, absolute external norms and
single-author uncertainty remain open. See
`reports/V8_ADAPTIVE_RANK_FIXED_REFERENCE_V37E_REPORT.md` and
`reports/V8_ADAPTIVE_RANK_FIXED_REFERENCE_V37E_INDEPENDENT_AUDIT.md`.

V3.7F replaced the calibration-cohort origin with disjoint router and
zero/norm reference panels, then added event/reference/rank-subspace
uncertainty and an explicit residual-sufficiency test. All 25 sealed gates
passed across 30 fresh four-world populations. Event-only and full-pipeline
Monte Carlo coverage were 0.944 and 0.952. In an exact rank-12 control, the
scorer-only omitted-space AUC was 0.498; in a 48-direction power-law world,
the finite-budget residual AUC was 0.863 [0.844, 0.882]. A registered
full-space-or-soft rule reduced dense hard capacity floor by 35.0%, while
event-only asymptotic error was 0.0035 and a planted state-alias world retained
a 0.243 positive floor. Decision:
`PASS_EVENT_NOISE_ZERO_TOTAL_FLOOR_POSITIVE`; independent ruling:
`PASS_WITH_MAJOR_CAVEATS`. This supports information-preserving reliability
weighting, not the claims that every token contains personality or that total
prediction error is zero. See
`reports/V8_EXTERNAL_ZERO_UNCERTAINTY_V37F_REPORT.md` and
`reports/V8_EXTERNAL_ZERO_UNCERTAINTY_V37F_INDEPENDENT_AUDIT.md`.

## V8 real-text density-resolution update (2026-07-29)

A label-free three-corpus audit now distinguishes local replicated density,
capacity resolution, cross-corpus spectral order, and psychological meaning.
PANDORA and Essays each confirmed scale-free resolution regions for anonymous
order-free marginal M and order-specific transition K families. PANDORA
confirmed 671/835 M and 464/755 K capacity-sharpness cells; Essays confirmed
700/835 M and 393/755 K cells. All four D0 internal Haar tests had p=.01 and
effective ranks were about 35--41. X was refused because its D1/D2 author
counts, 46/42, were below the frozen 48-author gate.

The apparent opposite PANDORA/Essays mask nesting did not survive a
pair-symmetric shared-gauge replay with equal D0/D1/D2 counts of
420/296/269. The registered M direction reversed at the point-estimate level,
and the K direction remained only a weak trend whose simultaneous band did
not clear zero. The accepted result is therefore
`LOCAL_SCALE_FREE_RESOLUTION_CONFIRMED__CROSS_CORPUS_SPECTRAL_ORDER_NOT_SUPPORTED`.
This is technical real-text structure, not a factor inventory, personality,
emotion, state, diagnosis, or clinical validity. See
`reports/V8_SUPPORT_RESOLUTION_SCALE_FREE.md` and
`reports/V8_SPECTRAL_ORDER_REPLAY.md`.

## V8 permutation-orbit correction (2026-07-29)

The K family is no longer treated as one presumed dynamic axis. Every
four-event path is enumerated over all 24 permutations and decomposed into:

- event-set susceptibility \(V\): what the fixed utterance multiset permits
  under reordering;
- observed phase \(q\): where the author's actual order lies after
  regularized within-orbit whitening.

The apparent common PANDORA--Essays event-set geometry is robust against a
block-Haar null that preserves the four K feature blocks and all spectra,
D1/D2 HS excess is .093/.159 and fidelity excess is .116/.170; all four
max-family \(p=.002\), with positive bootstrap lower bounds.

Observed-order phase is not supported. The full raw-K statistic-level null
has maxT \(p=.762\)--.966. After orbit whitening, D1 remains only a trend
(\(p=.211/.249\)) and D2 is below its random-order mean
(\(p=.967/.955\)). The current headline is therefore:

An additional composition knockout preserved corpus, split, context, event
position, every event vector marginal, and local text-length rank while
breaking natural within-author co-occurrence. These pseudo sets reproduced
the common geometry: D1 maxT \(p=.635/.405\), D2
\(p=.750/.755\). The final headline is therefore:

`K_OPERATOR_MARGINAL_BACKGROUND_IDENTIFIED__NATURAL_COMPOSITION_AND_OBSERVED_PHASE_NOT_DETECTED`

The final corpus-local signed residual test also finds no stable author
composition increment. PANDORA D1/D2 maxT \(p=.865/.800\); Essays
\(p=.995/.995\). No positive-part projection was used. The K branch is
therefore closed at the **Conditional Marginal Orbit-Susceptibility
Operator (CMOSO)**:

\[
\mathcal T_{K,c}
=
E_Z E_{\prod_j P_c(x_j\mid Z,j)}
\operatorname{Cov}_{\pi\in S_4}[K(\pi\{x_1,\ldots,x_4\})].
\]

This is an operator calibration/background object, not an author
condition/opportunity response, personality factor, time-arrow result,
cognitive mechanism, diagnosis, or clinical score. See
`reports/V8_ORDER_STATISTIC_NULL.md`,
`reports/V8_PERMUTATION_ORBIT_PHASE.md`, and
`reports/V8_EVENT_SET_COMPOSITION_KNOCKOUT.md`, and
`reports/V8_CORPUS_LOCAL_COMPOSITION_RESIDUAL.md`.

## V8 background-quotient concordance update (2026-07-30)

The order-free M family was then tested against pseudo authors that preserve
corpus, split, context, event slot, local length rank, and complete event
vectors while perturbing natural author grouping. An audit found that the
historical nonzero cyclic shift was a valid generative knockout but not an
exact permutation null: it excluded fixed points and the observed identity
assignment. The complete M analysis was therefore rerun with uniform
within-block permutations whose support includes both.

Under the corrected exchangeable null, normalized signed concordance survives
in all three corpora. D1/D2 link excess is .01650/.01533 in PANDORA,
.06299/.07299 in Essays, and .20973/.22887 in X-market; every paired lower
bound is positive. The corresponding same-author AUCs (.560/.559,
.691/.710, and .788/.786) remain descriptive technical statistics.

Declared opportunity, template, and collection profiles were then fitted on
D0 and applied unchanged. Signed concordance remains positive in all three
corpora after this sensitivity filtration: PANDORA D1/D2
\(.01354/.01741\), Essays \(.05526/.06491\), and X-market
\(.07391/.10854\), with positive lower bounds and corrected significance.
This is not subtraction of noise; it is survival relative to a declared
opportunity model.

The filtered spectrum resolves no stable global linear axis in any corpus.
The earlier X rank-3 M and rank-1 strict-shape axes are therefore retained
only as historical raw-quotient findings; they do not survive the declared
opportunity layer. A subsequent axis-free relation audit finds:

- PANDORA: short-scale RBF relation correspondence at multiplier .5;
- X-market: multi-scale RBF relation correspondence at .5 and 1.0;
- Essays: signed scalar concordance only;
- no corpus: a replicated linear axis or a closed RBF-minus-linear nonlinear
  increment.

The result is a stronger **separation theorem for the empirical claims**:
signed concordance, stable factor axes, and reproducible author-relation
geometry are different objects. PANDORA and X preserve pairwise author
relations in context-stratified block aggregates without furnishing a common
coordinate chart. These aggregates weight retained within-context author
pairs and contain no cross-context relation.

The same-author cross-context follow-up remains underresolved. A subsequent
strictly nested audit fixed the same 120 authors, the same twelve-event source
pool, and one twelve-event-fitted operator while revealing 4/6/8/10/12-event
subsets. In the primary twelve-minus-eight contrast, five of six relation
cells increased, but D1 AskReddit decreased by .0848 and every pointwise and
simultaneous interval crossed zero. Across all 15 possible eight-event
pair-compositions, cross-context excess increased in 15/15 schedules in both
held panels (median deltas .0620/.0637), whereas within-context directions
varied by composition. This replaces the earlier informal “all cells rise”
reading: event count and event composition are separate measurement facets,
and neither transport nor a universal minimum text budget is established.
The entire RGC/transport/budget chain remains opened-panel development. See
`docs/V8_EXCHANGEABLE_BACKGROUND_AUDIT_PROTOCOL.md`,
`reports/V8_EXCHANGEABLE_BACKGROUND_AUDIT.md`,
`docs/V8_CBQ_CONCORDANCE_SPECTRUM_THEORY.md`,
`reports/V8_NUISANCE_FILTRATION.md`,
`reports/V8_FILTERED_CONCORDANCE_SPECTRUM.md`,
`reports/V8_RESIDUAL_GEOMETRY_CORRESPONDENCE.md`,
`reports/V8_CONTEXT_GEOMETRY_TRANSPORT.md`, and
`reports/V8_CONTEXT_TRANSPORT_RESOLUTION_FRONTIER.md`,
`docs/V8_NESTED_INFORMATION_BUDGET_PROTOCOL.md`, and
`reports/V8_NESTED_INFORMATION_BUDGET.md`.

A historical scale-development alignment audit stress-tests GRAIN against
representative design, Generalizability Theory, construct validation,
state-trait theory, invariance, and argument-based validation. GRAIN is a
methodology foundation rather than a scale-development project. Its most important
unclosed technical requirements are a declared generalization universe,
frozen reference-population scoring, person-by-facet error decomposition,
conditional score uncertainty, response-process evidence, and invariance.
See `docs/SUICA_SCALE_DEVELOPMENT_ALIGNMENT_AUDIT.md`.

Current real-text state:

`K_BRANCH_CLOSED_AT_CMOSO__M_SIGNED_CONCORDANCE_ALL_THREE__FILTERED_GLOBAL_AXES_NONE__PANDORA_SHORT_SCALE_AND_X_MULTI_SCALE_CONTEXT_STRATIFIED_RELATION_CORRESPONDENCE__CONTEXT_TRANSPORT_INFORMATION_BUDGET_UNDERRESOLVED__HUMAN_CONSTRUCT_VALIDITY_OPEN`

## Provenance

Frozen from the private development repository `project persona` at commits
`154822a`, `05be394`, `cad83d5`, `c27727b`, `1c417fa`, `8447541`, `5189168`,
`b9f65a6`, `0650936`, `5485a02` (+ the freeze commit recorded in
`docs/FREEZE_NOTES.md`). Built with AI-assisted research under the
builder/auditor protocol documented in the guide; seven audit rounds caught
and corrected real artifacts in five of them — the audit trail is part of
the method.
