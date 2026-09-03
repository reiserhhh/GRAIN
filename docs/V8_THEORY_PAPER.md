# GRAIN V8: A Typed, Refusal-First Foundation for Text-Based Measurement

**Status: DRAFT (2026-08-03) — internal paper draft, not submitted.**
Claim tiers in this draft are bound by `docs/CLAIMS_LEDGER.md`; where wording
here and the ledger differ, the ledger controls. The formal reference list is
deliberately not yet written (no placeholder citations are used anywhere in
this draft); Section 8 positions the framework conceptually and the reference
pass is queued work.

## Abstract

Text produced spontaneously by a person is an attractive measurement medium
and a treacherous one: the same corpus supports author identification,
stylistics, and topic inference, and it is easy to relabel any of these as
"personality assessment". GRAIN (Group-Referenced Analysis of Individual Natural-Language Expression) is a
foundational analysis method that treats a person's writing as many small
behavioral observations measured inside their naturally occurring contexts.
Their reproducible multiscale organization is the person's **language
texture**: distributions, orientations, condition choices, conditional
responses, interactions, and paths defined relative to an explicit reference
frame. GRAIN specifies — as a typed chain with machine-enforced refusals — what may be
constructed from them and what may not. Its central design commitment is that
topic and situation are not noise to strip away but the "rind" that carries
person signal through self-selection; its central epistemic commitment is
falsification-first development, in which every universal claim the program
ever held is either backed by a registered test or recorded as refuted in a
public ledger. We present the V8 system: the measurement chain
(H,F)→B→X→Z→𝔄→𝔐→Θ→D with three non-interchangeable output types (individual
vector V, relation R, population field P); the author object separating
selection, position, response, dynamics, interaction, and path; a
micro–meso–macro mechanism hierarchy; residual theory; chart applicability as
a design-level license; and the empirical program that disciplines all of it,
including designed-null refusal records (200/200), a demonstration that
author-identification AUC .864 arises in worlds with zero individual
structure, a four-part decomposition of a composite-transport failure with
two constructive repairs, a measured panel sizing law showing that a key
real-text objective cannot be rescued by scale (≈10^14× event budget), and
honestly null direct-prediction results on real text. We argue that the
governance layer — typed refusals, claim tiers, registration-before-run,
append-only adjudication — is not overhead but the method itself.

## 1. Introduction

Language-based assessment research faces a standing inferential temptation:
models that separate authors exist, correlations between text features and
questionnaire scores exist, and the distance from these to "we measure
personality from text" is routinely crossed in prose rather than in evidence.
The failure modes are well known in outline — construct-irrelevant variance
(topic, platform, opportunity), reference-population dependence, unstable
coordinates, post-hoc selection — but they are usually treated as caveats
rather than as objects the measurement system itself must represent.

GRAIN starts from the opposite end. It is not a scale; it is a specification
of the conversions that stand between a human process and a defensible
assessment output, with each conversion typed, each type equipped with the
evidence it requires, and each unlicensed shortcut turned into an explicit
refusal. Two commitments organize everything:

1. **The rind thesis.** A person's choice of topics, venues, and situations
   (the rind) is itself person signal transported by self-selection.
   The naive corollary — remove context statistically to purify the signal —
   was tested first and refuted: condition mean-centering destroys person
   signal in three independent implementations. The surviving design
   principle is to control context by design and to measure choice as its own
   channel, beside style-under-condition and reaction-to-condition-change.

2. **Falsification-first development.** Each generation of the theory was
   required to kill part of its predecessor's universal claims with
   registered tests before its own additions were admitted. The development
   lineage (V8 canonical document, §11) records for every stage both the
   retained contribution and the refuted claim; the claims ledger records
   every adjudication with its tier, including retractions.

This paper presents the V8 integration: the type system (§2), the author
object and the reference-relative measurement spine (§3), the layered
mechanism theory (§4), chart geometry and applicability (§5), the empirical
program that disciplines the theory (§6), governance as method (§7),
conceptual positioning (§8), and limitations with open problems (§9).

## 2. The measurement chain and type system

GRAIN forbids silent conversion along the chain

(H,F) → B → X → Z → 𝔄^(τ) → 𝔐^(τ) → Θ^(τ) → D^(τ),  τ ∈ {V, R, P},

where (H,F) is unobserved human history and current process; B is behavior
that became expressible under available opportunity; X is the recorded
text/event stream; Z a technical representation; 𝔄 an author-level technical
object; 𝔐 a measurement candidate in an explicit reference frame; Θ a
psychologically interpreted construct; and D a use-specific decision output.

Each arrow names an inferential gap with its own evidence class and its own
refusal while open (the Foundation Gap Ledger types all of them, F01–F17,
with statuses CONSTITUTIONALLY_CLOSED / EMPIRICALLY_BOUNDED_OPEN /
DOWNSTREAM_INACTIVE). Two properties are distinctive:

- **Output types.** The superscript τ distinguishes an individual vector or
  score candidate (V), a relation between people, conditions, or constructs
  (R), and a population-level distribution or relation field (P). Evidence
  for R does not license V; evidence for P does not identify an individual.
  These are not style guidelines: the released code raises exceptions on
  out-of-license comparisons (comparison licenses L0–L4), and the typed
  R→V bridge of §6.2 is the only sanctioned promotion path, itself gated.

- **Refusal as an output.** Every stage may return refusal instead of a
  value, and several protocols are scored partly on whether they refuse
  correctly on designed nulls. A system that cannot refuse cannot carry the
  rest of the architecture.

The construct layer Θ and decision layer D are deliberately INACTIVE in the
current program: no coordinate is given a personality name, no diagnosis is
produced, and the highest supported claim for the sealed technical core is
operator-indexed relative text geometry within a declared domain.

## 3. The author object and the reference-relative spine

Observation is operator-indexed. A corpus and a chosen slicing/representation
pair (S_λ, R_v) define one observation plane; no theory step requires a
globally optimal cut, and slice-vs-uncut superiority is an empirical,
construct-local question. This dissolves an older internal conflict:
slicing remains central as an experimental operation while losing its status
as an axiom.

The minimal author object separates channels that common pipelines collapse:

𝔄_u = (π_u, μ_u, R_u, K_u, C_u, P_u)

— selection over available conditions (π), expressed position under
conditions (μ), response to controlled condition change (R), temporal/path
dynamics (K), interaction coupling (C), and higher-order path information not
preserved by marginals (P). At event resolution, position decomposes as
w = μ_condition/opportunity + a_u + B_u z + g_u, and none of a_u (reference
position), B_u (response operator), or g_u (unrepresented structure) is
automatically personality, state, or noise.

A score has no standalone meaning: 𝔐_u = Ψ(𝔄̂_u; P₀, Q₀, S_λ, R_v, C) is a
position relative to a declared reference population P₀, opportunity
distribution Q₀, observation operator, and calibration version, all of which
must travel with every result. Consequences that the program enforces rather
than merely states: batch-standardized exploratory scores are not reusable
individual scores; reference and support metadata are mandatory; axes may
rotate while relational geometry is stable, so relations are the more
defensible object; and a future instrument requires its own
generalizability/decision study, uncertainty, minimum detectable difference,
and external validation. The current L4.5 line supports a finite-synthetic
vector score candidate only.

## 4. Layered mechanism theory (M3) and residuals

**Micro.** Five nonexhaustive mechanism families — distribution/selection,
conditional response, temporal dynamics, interaction, higher-order path —
organize event-level structure; an individual path kernel
K_u(ds′, dc, dξ | H_t, s, o) separates state change, condition transition,
and expression given history and opportunity. Synthetic recovery shows these
families can generate distinct author geometries; the program explicitly does
not claim they span human psychology.

**Meso.** Replicated author objects lift to context-local relation objects
J_s with within/between/transported components kept separate; the
event-to-relation seam is licensed only when replicated supports are
observably comparable, otherwise transport is refused.

**Macro.** Population structure is a field over reproducible contexts, not
one universal correlation matrix; a macro relation can be real while
individual prediction is weak, and back-projecting it into individual
diagnosis requires a separate bridge (this is exactly the license the typed
R→V bridge formalizes).

**Coarse graining.** token → slice → utterance → occasion → author is a
filtration; averaging preserves position while destroying order, transitions,
hysteresis, and coupling (the V6 theorems), so resolution is treated as a
parameter whose invariants must be established per property.

**Residual theory.** R = R_resolution + R_alias + R_omitted + R_private +
R_technical, without orthogonality or unique identification. The goal is not
residual zero but a statement of which components become negligible under
which limits — a discipline that §6.3's wall decomposition vindicated, since
two of the four wall components turned out to live in the instruments, not
the world.

## 5. Charts and applicability as a design-level license

The response-safe RCCA chart is a gauge-quotient coordinate system whose
defensible objects are distances and subspaces, not raw coordinates. A sealed
finite-synthetic result (R2B) shows conditional downstream benefit; the same
program refused the tempting generalizations: unconditional replacement of
the baseline operator is not licensed; a scalar support-coverage threshold
(.80) was falsified as an automatic routing gate (Boundary Ecology: similar
support counts contained both beneficial and harmful departures); and a
24-component pre-response signature failed gain prediction and routing
utility with every replay control exactly zero
(M4_C35_PRE_RESPONSE_DOMAIN_NOT_IDENTIFIED). The standing conclusion is
architectural: **applicability is a design-level license** — a chart may be
used only inside prospectively declared study classes with separate evidence;
otherwise expose baseline/chart sensitivity or refuse. No further threshold
tuning is licensed on these outcomes.

## 6. The empirical program

All numbers below are artifact-bound (per-report reproduction commands;
results directories with decision records); tiers are the ledger's.

### 6.1 Deflation instruments

Two results discipline interpretation program-wide. First, designed-null
refusal: across all arms and batteries of the relation-bridge line, worlds
constructed with vanishing individuality were refused 200/200 — the bridge
never promotes group-only structure to an individual claim. Second, the
author-AUC deflator: in synthetic worlds with zero individual structure,
author-identification AUC .864 is still attainable from shared machinery;
author separability therefore cannot serve as personality validity, and no
GRAIN claim is allowed to lean on it.

### 6.2 The typed R→V bridge and its operating curve

The bridge licenses promotion of relation evidence toward vector candidates
via a rigidity index (spectral margin × noise-matched identity stability)
with a fixed threshold. Its heteroscedastic calibration study mapped an
operating frontier rather than a single winner: the baseline dominates at its
registered threshold; a variance-weighted variant uncaps one battery (60/60)
while degrading deep-noise ordering; a permutation variant repairs deep-noise
ordering (.944→.995); the two repairs anti-correlate, so variant choice is a
declared design decision, not a tunable. The C2 saturation mechanism was
exactly identified (exact-EDM fields make the negative-spectrum floor
collapse to ~2e-9), converting an anomaly into a understood boundary of the
instrument.

### 6.3 Anatomy of a transport wall (16 registered experiments)

V8 shipped with an honest failure: atomic mechanism actions transported
(.77–.98) while composite loops did not (.6519), across four registered
NO-GOs. The M4-D/E/F line decomposed this wall into four parts:
(i) route misidentification under chart overspan, repaired constructively by
a two-stage route-then-refit estimator (route flips 196→73; pooled loop
geometry .6519→.7605); (ii) estimator self-infliction — a ridge penalty's
non-vanishing bias manufactured an apparent budget-invariant floor; λ~1/n
de-biasing restored textbook n^(−1/2) scaling (fitted slope −.005→−.521) and
forced a retroactive correction of an earlier conclusion; (iii) common-mode
structure of paired truth-referenced diagnostics, which reward shared
shrinkage and can be subsidized by contamination (a cleaning intervention
lowered an attribution correlation .7264→.6677 while improving every loop
statistic — a property of the metric, not a bug in the world); and (iv) a
genuine object-level deficit: the discovery objective's residual is a smooth
quadratic-basin direction-content deficit whose common offset is distributed
(max single-term share .38) and world-specific in direction — registered as
the program's open problem, with a named lead suspect family
(freeze-whitening 1/√eig scale amplification, .39–.48 mass).

Two lessons generalize beyond this program: half of a "wall" lived in the
instruments (penalty bias; paired-metric structure), and the honest
accounting was only possible because leans and pivot conditions were
committed before each run (registered pivots fired as pre-committed in eight
experiments; two of the planner's own interpretations were killed by their
own registered tests; one retroactive correction was applied append-only).

### 6.4 Real text: honest nulls, an adjudicated headline, and a sizing law

On real corpora the program reports its nulls as results. The V8 invariant
canonical score does not directly predict Big Five (official-fold mean
r=−.005, below a text-volume/format nuisance comparator) and is weak for
MBTI (AUC .544). One apparently strong result — recovery of the held-out
5×4 Big5-to-MBTI relation matrix at element r=.498 (permutation p=.0086) —
was adjudicated POST_HOC_OPERATOR_SELECTED_EXPLORATORY after review
established that an earlier same-day run under a different operator returned
r=−.103 (p=.673) and was not disclosed; the citation rule now binds the
number to its null and to the selection order. The deeper real-text
constraint was then measured rather than lamented: the internal split-half
agreement of the deployed relation field is nil at every budget under every
penalty convention (a "saturated yardstick", with the two conventions
near-projectively equivalent, so penalty choice is immaterial there), and a
calibrated synthetic sizing law puts the cost of rescuing agreement by scale
alone at ≈10^14× the current event budget (events exponent .153; a held-out
16× cell validates the law within factor 2). The design consequence is
sharp: the next real-text relation panel must change composition
(shared-context events, within-author condition pairing), not size.

## 7. Governance as method

The program's central methodological claim is that the governance layer is
inseparable from the measurement theory. Concretely: a claims ledger in which
prose may not exceed ledger tier; registration-before-run with leans and
pivot conditions committed to version control; append-only, dated
adjudication (originals preserved; corrections are new dated notes);
machine-enforced comparison licenses and refusal paths; sealed releases with
hash-bound lockboxes verified in CI against the immutable tag tree; and
adversarial review of large drops before publication. The V8 push itself is
the case study: review found the substance trustworthy (15/15 headline
numbers traced, 9 recomputed exactly) and the governance of one campaign
deficient (an unregistered full-corpus label join and an outcome-contingent
operator switch), and the remediation — demotion, full timeline disclosure,
consumption records, process rules — was published together with the theory
(`docs/V8_PUSH_REMEDIATION_20260803.md`). In this architecture a governance
failure is representable, adjudicable, and survivable without rewriting
history; we regard that property as a requirement for any AI-accelerated
research program, where generation is cheap and verification is the scarce
resource.

## 8. Conceptual positioning (reference pass pending)

GRAIN's spine is deliberately continuous with the classical measurement
tradition: the reference-relative score, the demand for
generalizability-style decomposition before reuse, and minimum detectable
difference requirements correspond to reliability/G-study discipline; the
insistence that construct interpretation is a separate, later evidence layer
is classical validity theory taken literally. What GRAIN adds is
(i) types with enforcement — the V/R/P distinction and stage-wise refusals
exist as raised exceptions, not prose; (ii) opportunity and self-selection as
first-class measured channels rather than confounds; (iii) path/dynamics
retained through the coarse-graining filtration instead of averaged away; and
(iv) governance artifacts (ledger, seals, registered pivots) as part of the
method's identity. Relative to the language-based assessment literature that
optimizes prediction of questionnaire scores, GRAIN inverts the default: it
treats predictive correlation without construct, invariance, and use evidence
as the *beginning* of an argument, and its own strongest real-text
direct-prediction result is currently a null that it reports as such. A
proper engagement with specific prior work, with citations, is queued for the
first external version of this draft.

## 9. Limitations and open problems

- **Scope.** Human evidence is English, one platform plus student essays;
  dialogue and clinical registers are designed but unvalidated; language and
  culture transport is typed as open (F13).
- **Construct layer inactive.** No coordinate has a psychological name; the
  program's endpoint (a personality/state interpreter) remains distant by
  design until Θ-layer evidence classes are met.
- **Mechanism families are not a basis.** M3's five families generate
  distinct geometries in synthetic worlds; human completeness is explicitly
  not claimed.
- **Open problem (registered).** The discovery objective's distributed,
  world-specific frame displacement (§6.3-iv), with the freeze-whitening
  scale family as lead suspect.
- **Real-text relation program.** Blocked on panel composition redesign
  (§6.4); size cannot rescue it.
- **Reference pass.** This draft's related-work section is conceptual; the
  citation pass has not been done and no citations were invented to fake it.

## 10. Conclusion

GRAIN V8 is a measurement foundation whose primary artifacts are a typed
chain with enforceable refusals, an author object that separates what text
pipelines habitually collapse, a mechanism hierarchy disciplined by synthetic
ground truth, and a governance layer that makes the program's own failures —
statistical, architectural, and procedural — first-class, citable objects.
Nothing in it yet measures personality, and it says so; what it provides is
the part of that goal which usually goes missing: the machinery by which such
a claim could one day be earned, audited, and — if wrong — refused.

## Appendix: canonical documents

- Theory: `docs/SUICA_UNIFIED_THEORY_SYSTEM_V8.md`;
  map: `docs/SUICA_THEORY_ROUTE_INDEX.md`
- Evidence: `docs/CLAIMS_LEDGER.md` (controlling);
  gap typing: `docs/SUICA_FOUNDATION_GAP_LEDGER.md`
- M4-D/E/F line: `docs/SUICA_M4_D_CURVATURE_AND_RELATION_BRIDGE_PLAN.md`
  (registrations + outcomes), `docs/SUICA_M4_D_LOOP_WALL_SYNTHESIS.md`
- Governance: `docs/V8_PUSH_REMEDIATION_20260803.md`,
  `docs/V7_PROCESS_AUDIT_20260717.md`
- Practice: `docs/V8_MANUAL.md`; sealed worked example:
  `docs/WORKED_EXAMPLE_MANUAL.md`
- Development history (Japanese): `docs/V8_DEVELOPMENT_REPORT_JA.md`

---

## Part II supplement (dated 2026-08-10; DRAFT status unchanged): the identity program and the defense results

This dated supplement summarizes, in paper form, what the identity era
added after Part I's sections were drafted. The integration map is
`docs/SUICA_V8_IDT_INTEGRATION.md`; the theory of record is
`docs/SUICA_IDENTITY_THEORY_V1.md` (T1–T10, appendices A–V). No
reference-pass citations are introduced here either; the positioning
anchors of IDT appendix P.6 await the same verified pass as §8.

**II.1 The frame calculus.** The owner's conjecture — that the
individual "error" within a shared group and topic IS identity, with
indeterminate magnitude, functioning as an identification card — was
formalized as a frame F = (P, O, h, U) with a deviation field and a
reproducible component. The magnitude indeterminacy became a theorem
(three gauge channels with distinct signatures); the certificate
metaphor became exact (issuer, jurisdiction, expiry — and, by T10, the
proof that no informative reading is anchor-free: the reference-relative
spine of §3 is a necessity, not a preference).

**II.2 The reader as mechanism.** The deployed relation-field gauge
amplifies common-frame content into apparent agreement (3.54× the F2
composition effect at a 1× shift), reads the trait like the square of
the card's attenuation, and taxes raw person-state variance (κ ≈ 0.72
by three independent fitting routes). Consequently the panel-line wall
of §6 — occasion-bound, state-inclusive, never a trait — now has a
mechanism, and three of its headline attributions were corrected by
dated retrospective notes (composition = frame content; the author axis
= a replicate axis; truth recovery = mixture recovery).

**II.3 Identity, typology, and the audit.** The discriminator that
defines identity had to be hardened twice against measured forgeries
(split-scheme dependence; issuer-error forgeries that pass plain
reproducibility) — the licensed form is frame-refreshed reproducibility.
The reverse reading (remove identity to recover types) was resolved as a
conditional theorem set: true removal works but is circular (EM);
label-free projection provably cannot buy the boundary-normal component;
the governing object is a projection-invariant floor curve in the
aligned fraction η — and that curve, sealed, later PREDICTED at foreign
dimensions. "Error = 0" became an audit criterion with a calibrated
meter rather than an assumption.

**II.4 Defense as science.** The phase sealed five quantitative
predictions before their configurations existed, verified the headline
table adversarially from raw artifacts (zero refutations), made
verification portable (content-addressed lockbox), bound a real-text
governance rule (no person claims; no cross-corpus linkage; forgery-safe
statistics only), and then opened the seal measure-first: three
predictions hit (two laws promoted to predictive), one conjecture died,
and one glued formula was falsified and re-fitted into an honestly
scoped level law whose only stable content is the tax term. The method
ledger closed at standing rules 1–24 with a defect registry of 42
entries and a replay showing zero would now survive to post-hoc.

**II.5 Limitation, restated.** Everything in Part II is synthetic and
instrument-relative at EXPLORATORY tier. The one sentence that must
survive any future summary: these are laws about a reader–world pair,
licensed for design, refusal, and audit — never about persons.
