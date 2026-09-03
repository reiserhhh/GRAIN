# GRAIN Theory Formal Notes v2 — F7–F10: invariance, aggregation geometry, spectral nulls, factor fluidity

Created 2026-07-11. Extends F1–F6 (SUICA_THEORY_FORMAL_NOTES_V1.md). Source of the
mathematical apparatus: the field program's T-geometry reformulation
(trading-agent-claude, `docs/SUICA_V5_TGEOMETRY_REFORMULATION_20260711.md`, layers I–III),
ported and instantiated on PANDORA. Discipline inherited from that document: **each formal
layer must emit at least one new falsifiable instrument; anything that only renames existing
results is cut.** Empirical instantiations: results/suica_tgeo_p1_phi_linearity,
results/suica_tgeo_p2_time_reversal, results/suica_tgeo_p3_mp_edge.

## F7 — Inference as invariance: the null suite is a subgroup lattice

**Definition.** Let G act on the labeled data (user labels, condition labels, occasion
orders, half labels). A statistic carries G-signal iff its sampling distribution is not
invariant under the G-action. Every null this program has run is a choice of G:

| Our null (historical row) | Group G | Invariance asserted under H0 |
|---|---|---|
| Stranger null (OP-32, E6 v3) | derangement class of S_N | person-exchangeability of profile pairings |
| Leak-mask permutation (W-E) | S_N on survivor labels | mask sufficiency |
| Shuffled-fit null (E11) | S_N on fit targets | direction-fit exchangeability |
| Per-world seeds / sign conventions (WW suite) | (Z/2)^N sign flips | conditional symmetry of residuals |
| Contiguous-halves rule | trivial group (forbidden shuffle) | refusal to assert time-exchangeability |

**F7.1 (artifact, formal definition).** An artifact is a statistic that is constant on a
larger orbit than the analyst believed. (The field program diagnosed a live artifact this
way; upstream's E4 stranger-null lesson — react signatures absorbed by the stranger baseline
— is the same object: the "signature" statistic was nearly constant on the pairing orbit.)

**F7.2 (new instrument, ported I-1 — time-reversal null).** Author-level stationarity
asserts invariance under per-author occasion-sequence reversal (an involution). Reversal
maps each author's time slope to its negative, so the per-author sign-flip group (Z/2)^N
yields an exact null for the mean slope. PANDORA instantiation: T-GEO-P2 (K=4 equal-count
blocks, span >= 180 days; arms: pooled conditions vs top-1 condition to separate style drift
from venue-mix drift).

## F8 — Aggregation as Möbius flow: the phi-coordinate

**F8.1 (phi-linearization; theorem, ported with proof).** Let phi(r) = r/(1-r). The
Spearman-Brown family r_W = W·rho / (1 + (W-1)·rho) satisfies exactly

    phi(r_W) = W · phi(rho).

Proof: phi(r_W) = W·rho / (1 + (W-1)rho - W·rho) = W·rho/(1-rho) = W·phi(rho). ∎
Reliability lives on the projective line; rho_1 is the infinitesimal generator of a
one-parameter Möbius subgroup; the SB step-up S(r) = 2r/(1+r) is phi(S(r)) = 2·phi(r).

**F8.2 (curvature = clustered noise).** Under slice-clustered noise with cluster-shock
variance V_c (the field program's day shocks; our within-condition venue shocks), the
designed-aggregation curve is phi(W) = a·W/(1 + b·W): concave with asymptote a/b, and b
estimates the clustered-noise share. Under the parallel model b = 0.

**F8.3 (new falsifier — asymptote consistency check).** The saturating fit is itself
falsifiable: the fitted asymptote phi_max must not fall below the realized phi at large
aggregation (the full-half retest). Violation rejects the simple saturation form for that
construct (its small-W curvature is not clustered noise).

**PANDORA instantiation (T-GEO-P1, n = 2,767, label-free, nested designed subsets,
W in {1,2,4,8}, Spearman primary).** Directional expectation (F near-linear, C saturating)
did NOT hold as a family split; what the geometry actually found is finer:

| construct | fam | phi ratios W2/W4/W8 (pred 2/4/8) | fitted phi_max | full-half phi (realized) | verdict |
|---|---|---|---|---|---|
| tension_core | und | 1.29 / 1.70 / 3.63 | 0.49 | 0.46 (r=.316) | saturation model CONSISTENT — tension's ceiling is real; "no-trait" gets a geometric signature |
| first_person | F | 1.84 / 3.27 / 5.20 | 3.58 | 2.16 (r=.684) | mild curvature, consistent |
| wcl_23 | F | 2.14 / 3.90 / 6.45 | 2.18 | 1.16 (r=.537) | near-linear at small W, consistent |
| wcl_60 | F | 1.38 / 1.69 / 2.27 | 0.79 | **2.44 (r=.709) — VIOLATES F8.3** | saturation form REJECTED: small-W curvature is not clustered noise; consistent with slice-level ZERO-INFLATION (apostrophe opportunities are sparse per 128 tokens) — the v5 hurdle regime exists INSIDE PANDORA, at slice granularity, for sparse form habits |
| wcl_13 | F | 3.08 / 6.10 / 10.59 | — | — | SUPER-linear (ratios exceed the parallel bound) — W=1 noise heavier-tailed than the model admits; flagged for follow-up |
| novelty | C | 2.04 / 4.38 / 9.10 | — | — | near-linear — see F8.4 |
| wcl_02 / wcl_25 / wcl_35 | C | sub-linear (5.27 / 4.45 / 4.20 at W8) | 3.0 / 1.5 / 1.4 | 1.22 / 1.10 / 0.82 | consistent mild saturation |

**F8.4 (scope note — what phi-curvature can and cannot see).** Stable venue choice is
person-stable across halves, so for C-family constructs the choice-mediated component m_u
behaves as SIGNAL in a retest, not as clustered noise; phi-linearity therefore does NOT
detect the choice channel (novelty runs near-linear while being .59 choice-mediated). The
phi instrument reads the NOISE geometry; the purity decomposition (F3) reads the SIGNAL
provenance. They are complementary, not redundant.

**F8.5 (standing rule, adopted).** Composite constructs must be blended in phi-space, not
r-space: under independent-noise composition the additive object is phi. The frozen v4
composite weights are r-space blends; any composite refit for opening #2 derives weights in
phi first (rule recorded; frozen weights untouched).

## F9 — Spectral nulls: factor claims against random-matrix edges

**F9.1 (instrument).** For a user x construct matrix (n x p, standardized), eigenvalues of
the correlation matrix under pure noise concentrate below the Marchenko-Pastur edge
lambda_+ = (1 + sqrt(p/n))^2. A factor claim requires (i) eigenvalue above the edge —
empirical within-column-shuffle edge preferred over the analytic constant
(heteroscedasticity caveat, field round-13 N2); (ii) cross-half replication of the
supra-edge eigenvector (loading congruence).

**PANDORA instantiation (T-GEO-P3, n = 2,941 x 19).** Analytic MP edge 1.167; empirical
within-column-shuffle edge 1.169 (500 draws — heteroscedasticity mild here). **Exactly 4
components exceed the edge, and all 4 replicate across halves** (early/late eigenvector
congruence .985 / .980 / .932 / .939). The battery's factor structure is real and stable by
the RMT criterion. Follow-up target: OP-33's 48 co-selection axes (requires re-scoring).

## F10 — Factor fluidity: the frame as a field (proposed 2026-07-11, user conjecture)

**Motivating conjecture (user, verbatim intent).** Factors are not fixed; they FLOW with
conditions (time, position within a text, venue — to be explored). If so, factors likely
cannot be NAMED, but they take on a stronger role as AXES OF INDIVIDUALITY.

**Context from F9.** T-GEO-P3 already constrains this: the population between-person frame
is time-stable at half scale (4 supra-MP factors, early/late congruence .93–.99). So if
flow exists, it must live (a) at finer condition scales than the half split, or (b) at the
PERSON level — each person carrying their own frame — which is exactly where the conjecture
points ("individuality as axes"). The population average frame can be rigid while the
person-level frames disperse and drift.

**F10.1 (frame field).** For a condition variable t (time block, text position, venue), let
C(t) be the between-person correlation operator of the battery and V_k(t) ∈ Gr(k, p) the
span of its top-k eigenvectors — a point on the Grassmann manifold. Factor FIXITY (H0) is
the assertion that t ↦ V_k(t) is constant up to sampling noise; per F7 this is invariance
of the chordal distance d(V_k(t), V_k(t')) = sqrt(k − ||V_tᵀV_t'||²_F) under the group that
permutes condition labels within person. Factor FLOW is rejection of that invariance.

**F10.2 (nameability criterion).** An axis is eligible for a NAME only if the frame is
(i) condition-invariant (F10.1 H0 holds over the tested t) and (ii) person-shared (the
person-indexed frames U_u concentrate near a common point of the Grassmannian). If (i) or
(ii) fails, the surviving invariants are the eigenvalue spectrum and the person-level frame
itself: the axis stops being a population construct and becomes a personal quantity. This
gives mathematical form to the program's standing rule that vectors are not entitled to
names by default (vector interpretation unlicensed until calibrated).

**F10.3 (individuality as axes).** Person frame: U_u = leading eigenvector (or top-k frame)
of person u's WITHIN-person slice correlation. The conjecture's positive content: U_u is
trait-like — orientation, not only scores, is a personal signature. If it holds, the GRAIN
person representation extends from the score vector f_u to the pair (f_u, U_u) ∈ R^p ×
Gr(1, p), and interperson distance gains a principal-angle term. Deployment is AI-native by
construction: an axis needs no name to be used.

**F10.4 (instruments and falsifiers — predictions REGISTERED BEFORE the runs).**

| instrument | design | registered prediction | falsifier |
|---|---|---|---|
| T-GEO-P4 time flow | P2 block design (K=4, span ≥180d), per-block between-person corr, top-4 chordal distance d(1,4) + path length; null = within-user block-label permutations (500); arms pooled/top1 | NO detectable flow beyond null at block scale (P3 constraint); if pooled flows but top1 does not → venue-mix flow, mirroring P2 | flow detected on top1 arm would OVERTURN the fixed-population-frame reading of F9 |
| T-GEO-P5 position flow | comments ≥288 tokens: open = first 128, close = last 128 tokens; per-construct paired mean shift (user-level primary) + open/close frame distance; null = per-comment open/close swaps, (Z/2)^N | mean-level shifts exist (hunches, stated for honesty: directive_action higher at close; first_person higher at open); frame rotation small | large frame rotation would relocate "style trait" claims to position-conditional claims |
| T-GEO-P6 personal axes | per user per half: leading eigenvector of within-person slice corr (gate ≥12 slices/half; top1 arm ≥8); self early↔late |cos| vs stranger null (cyclic-shift derangements, F7 row 1); noise ceiling = within-half odd/even split; SHARP arm: residualize slices against the population top-4 frame first | self > stranger raw AND residualized (γ-channel precedent: E6 v3 profile signatures); early↔late ≈ odd/even ceiling (frames stable, no strong within-person drift) | **the conjecture DIES if residualized self ≤ stranger** (personal frames = noisy copies of the shared frame); early↔late ≪ odd/even ceiling would instead CONFIRM temporal flow of personal frames |

**F10.5 (stated caveats, before results).** (a) P4 block means have heteroscedastic noise
across blocks (slice counts vary within user); the exchangeability null is approximate.
(b) P6 person frames are estimated from ~21 slices per half; estimation noise is symmetric
between self and stranger pairings, so the comparison remains valid, but congruence
magnitudes are attenuated — the odd/even ceiling is the honest yardstick. (c) P5 "position
in text" is confounded with discourse function (openings and closings do different jobs);
this instrument reads the confound deliberately, as a first probe. (d) All three are
label-free Tier-U; no seals, no frozen objects touched.

## F10.6 — Continuization: from cut to flow (user directive 2026-07-11, registered before run)

**Motivating observation (user, verbatim intent).** The "cut" (slicing) is an interval
variable; the finding turns it into "flow," a continuous variable.

**Formalization.** P5 measured position as a two-point contrast (open vs close). The
conjecture's natural completion treats position as a continuous flow parameter: each
window of a text carries t = relative position in [0, 1] (the frozen slicer has always
computed this coordinate — token_mid_frac — which the frozen prep then discarded), the
frame becomes a path t ↦ V(t) on the Grassmannian, and each construct gains a mean path
t ↦ μ_c(t). Two readings then separate, and only a continuous instrument can separate
them:

- BOUNDARY-REGISTER reading: openings and closings are discrete micro-registers; the
  interior is homogeneous. Movement is edge-localized; the "flow" is two zone jumps.
- TRUE-FLOW reading: movement accumulates along t; interior steps are comparable to edge
  steps.

**Instrument (T-GEO-P7, flow curve).** Non-overlapping 128-token windows within long
comments (>= 288 tokens; comments capped at 12 windows by endpoint-preserving even
subsampling), t = j/(m−1), five bins over t. Per bin: between-person frames (pairwise
user-matched to bin 0) and mean curves. Null: within-comment permutation of window order
(the group S_m per comment). Key property: this group preserves each comment's SET of t
values, hence its bin memberships — so bin composition (which comment lengths feed which
bins) is held fixed under the null, and every p-value is composition-matched by
construction. Statistics: adjacent-step frame distances d(b, b+1); per-construct adjacent
mean steps; EDGE SHARE = (first step + last step) / (sum of all adjacent steps), which is
≈ .5 under uniform flow and → 1 under edge localization; interior flatness = max interior
step vs null. Sanity anchor: bin4 − bin0 must reproduce P5's open/close direction for
first_person before anything is reported.

**Registered predictions (honest hunches, BEFORE the run).** Edge-localized: edge share
above null for first_person and for the PC3/PC4 rotation; interior steps within null;
PC1/PC2 flat everywhere. The conjecture's strongest form — true continuous flow — would
instead show interior movement at edge-comparable size. Either verdict refines F10; a
boundary verdict would RENAME P5's finding from "position flow" to "boundary
micro-registers," and a flow verdict would license t as a genuine continuous condition
variable in F10.1.

**Exchangeability note (F8 × F10).** Position structure violates within-comment window
exchangeability. Pass-A concatenation mixes positions across comments, so the user-level
residue is an opening-share composition effect (each cell's first slice always starts at
a comment opening; users with shorter comments carry more openings per slice). Flagged as
a pending robustness check for first_person (opening-share covariate).

## F10.7 — Functionalization of the flow (user directive 2026-07-11, registered before run)

**Motivating conjecture (user, verbatim intent).** If factors are fluid, the flow can be
FUNCTIONALIZED: within the observed range, specifying the position specifies the height;
outside the observed range, the height remains predictable to a degree.

**Formalization.** Promote the position profile from a description to a predictive
function μ_c(coordinate) with held-out tests. Two claims, two instruments:

- **F10.7a (in-range interpolation).** Given a comment's endpoints and a population
  functional form, the interior height is determined up to noise. Predictor ladder for
  the m = 3 mid window: (0) cross-fitted grand mean; (1) PERSON = leave-one-comment-out
  user mean; (2) FUNCTION-LINEAR = the comment's own endpoint average; (3)
  FUNCTION-QUADRATIC = (2) + cross-fitted population curvature. User-level MAE; the
  conjecture's in-range half holds for a construct iff a function rung beats the person
  rung.

- **F10.7b (out-of-range coordinate race).** CRITICAL IDENTIFICATION FACT: on m = 3
  support, a monotone flow in relative position t and an additive boundary-zone model are
  indistinguishable — they separate only in the DEEP INTERIOR of long comments, i.e., in
  (start-distance ≥ 2, end-distance ≥ 2) cells that short comments never realize. Three
  coordinate models, all fitted ONLY on short strata (m ≤ 3; five observed (τ, δ) cells,
  where τ = windows from start, δ = windows from end): CONSTANT; RELATIVE FLOW μ = a +
  b·t + c·t²; ADDITIVE ZONES μ = a + s(τ-zone) + e(δ-zone), zones {0, 1, 2+} — exactly
  identified with 5 parameters on 5 cells. Each model's shape is transported to long
  comments (m ≥ 4) with an intercept recalibrated on long ENDPOINT windows only; models
  are then judged by MAE and signed bias on the long INTERIOR windows — for the deep
  cells this is genuine out-of-range prediction. Bootstrap over comments for MAE-
  difference CIs. Thin evaluation set expected (~200 interior windows, ~100 comments):
  PILOT status declared in advance.

**Registered predictions (BEFORE the run).** P8a: for first_person the comment's own
endpoints beat the person rung (within-text information is real); the curvature
correction helps tension and directive but not first_person. P8b: tension is END-ANCHORED
(zone model wins via e(δ)); directive is START-ANCHORED (via s(τ)); for first_person the
race is genuinely open — registered lean: ZONES (the m = 3 "monotone flow" is an additive
boundary effect in disguise; the decline is opening-anchored settling, not a planned
relative-position allocation). Verdict semantics: if RELATIVE FLOW wins the first_person
race on deep interiors, the conjecture's strongest form holds (a transportable function
of t); if ZONES wins, functionalization still holds but the flow's true coordinate is
boundary distance, not relative position. Either way the in/out-of-range predictability
the conjecture asserts is quantified by the MAE ladders.

## F10.8 — Flow-only factors: coherence that exists only in motion (user conjecture 2026-07-11, registered before run)

**Motivating conjecture (user, verbatim intent).** Beyond the factors extracted so far,
do there exist "factors" visible ONLY in flow — existing only in the line (motion), such
that the moment you stop, the coherence disappears?

**Formalization (multivariate extension of F8's noise geometry).** Decompose a comment's
window-k score vector:

    w_k = p + k·s + g_k,   Cov(p) = Π (stable),  Cov(s) = Σ_flow (ordered flow),
                            Cov(g) = Γ (gust: transient window-level co-fluctuation)

Then LEVEL aggregation reads Cov_L = Π + Γ/m (+ small flow term if k uncentered): stopping
(averaging) kills gust coherence at rate 1/m — the exact mechanism of "once you stop, the
togetherness disappears," and the multivariate face of F8.2's clustered-noise parameter.
DIFFERENCING reads motion: D = w_last − w_first has Cov_D = (m−1)²Σ_flow + 2Γ. On m = 3
support the two motion layers SEPARATE by contrast algebra, with no permutation needed:

    linear contrast    ℓ = (w_2 − w_0)/2        Cov_ℓ = Σ_flow + Γ/2
    curvature contrast q = (w_0 − 2w_1 + w_2)/√6  Cov_q = Γ
    ⇒  Σ̂_flow = C_ℓ − C_q / 2   (up to scale; possibly indefinite — bootstrap the spectrum)

A FLOW-ONLY FACTOR (the conjecture's strong object, "line factor") is a component v with
(i) significant variance in Σ̂_flow, (ii) cross-user-half replication, and (iii) level-space
invisibility: vᵀC_L v inside the L-bulk (below the empirical shuffle edge) and low
congruence with all supra-edge L-components. A GUST FACTOR (weak reading) is the same
gates applied to C_q — coherence carried by transient co-fluctuation, erased by averaging.

**Instruments (T-GEO-P9, three phases, all on the P8 window cache; label-free).**
P9-A existence at scale: D on ALL m ≥ 2 comments (n ≈ 21.5k; Cov_D mixes flow+gust) —
supra-edge components (column-shuffle edge), user-half replication, static-visibility vs
C_L on the same comments. P9-B separation on m = 3 (n = 618): C_ℓ vs C_q, Σ̂_flow spectrum
with bootstrap-over-comments CIs, gates (i)–(iii). P9-C person level: per-user mean D over
their long comments (gate ≥ 3) — personal flow-STYLE factor structure C_uD, its edge, and
congruence with the same cohort's person-level static factors C_uL.

**Registered predictions (BEFORE the run, honest leans).** A: YES — at least one
replicating, statically invisible D-component exists (differencing isolates Γ, whose
structure has no reason to mirror Π). B: most motion structure is GUST (Γ); at n = 618 the
Σ̂_flow spectrum is likely too noisy to certify a flow-only factor — lean NO-at-this-power,
existence left OPEN rather than refuted. C: one person-level flow-style component, led by
co-movement of the first_person/directive declines; its static invisibility is OPEN.
Artifact caveats registered: (a) residual scorer-vocabulary overlap (v2 lexicons vs wcl
clusters) could mechanically couple gusts — any found factor gets a loading-pair overlap
screen before being believed; (b) sparse constructs make D heavy-tailed; (c) m = 2
comments cannot separate flow from gust (stated).

## F10.8 results (T-GEO-P9, run AFTER the registration commit a493ff6, same day)

**Answer to the conjecture: YES — motion-only coherence exists, with the cleanest
specimen at the gust layer.**

**Phase A (n = 21,498 comments; edge 1.059).** Four supra-edge slope components.
Components 1–2 (first_person/wcl_36 co-decline, λ=1.403, rep .990; directive/wcl_13,
λ=1.284, rep .973) are statically VISIBLE (level-visibility 1.97 / 1.59 vs null95 ≈1.01)
— motion factors with level shadows. **Component 3 (λ=1.145, replication .778) is
STATICALLY INVISIBLE: level-visibility 0.993 vs null95 1.014 — exactly at noise.**
Loadings: wcl_07 −.49, tension −.46, wcl_02 −.36, wcl_20 +.34. Component 4 fails
replication (.482) — discarded.

**Phase B (n = 618 m=3 comments).**
- ORDERED FLOW factor CERTIFIED (against my registered lean): λ₁(Σ̂_flow) = 0.486,
  bootstrap CI [0.42, 0.66] excludes 0, attenuation-permutation p < .002, person-disjoint
  half replication .809. Loadings: tension +.43, wcl_07 +.42, wcl_22 −.37, directive
  −.34. BUT level-visibility 2.75 — statically visible: a FLOW factor, not a flow-ONLY
  factor.
- **GUST factor 1 = the conjecture's textbook specimen, ALL THREE GATES PASSED:**
  λ = 2.223 ≫ edge 1.386; replication .829; **level-visibility 0.972 vs null95 1.096 —
  statically at noise.** Loadings: wcl_02 −.47, wcl_11 +.45, wcl_07 −.40, wcl_20 +.31 —
  all four are wcl clusters, whose vocabularies are DISJOINT by k-means construction, so
  the coupling cannot be shared-token mechanical (the registered artifact screen passes
  for this component). Gusts 2–3 fail replication (.293 / .235) — discarded.

**Phase C (n = 2,166 users, gate ≥ 3 long comments; edge 1.192).** Components 1–2
(first_person-led λ=1.432 rep .675 vis 2.32; directive-led λ=1.358 rep .602 vis 1.65)
replicate moderately but are statically visible/entangled (static congruence .85 / .73).
**Component 3 — barely supra (1.224), statically invisible (1.015; static congruence
.337) — FAILS replication (.150): the person-level flow-only factor is NOT established;
remains open.**

**Cross-phase congruence.** The same motion direction recurs: A-comp3 × B-flow = .786;
A-comp3 × C-comp3 = .817; B-flow × C-comp3 = .547. A {tension, wcl_07, wcl_02/20/22}
"motion cluster" appears at comment scale, in the ordered-flow estimator, and as the
(unreplicated) person-level candidate.

**F10.8 verdict.** The conjecture is CONFIRMED in the gust reading and partially in the
ordered reading: (i) a transient co-fluctuation factor exists that is invisible at rest —
supra-edge, person-disjoint replicating, level-space at noise, vocabulary-disjoint (B
gust1); averaging erases it at rate 1/m, which is the precise sense of "stop and the
togetherness disappears"; (ii) an ordered flow factor exists (slopes co-move along texts)
but casts a level shadow — flow, not flow-only; (iii) the fully personal flow-only factor
fails replication at current power — open. Registered-lean scorecard: A ✓; B half-wrong
(flow factor DID certify — stronger than my lean); C's invisible candidate died at the
replication gate as standards require. Measurement consequence: the battery's construct
space is not exhausted by level factors — motion coordinates (slopes, gusts) carry
replicating structure orthogonal to every static factor, and any future battery revision
should consider motion-space constructs as first-class candidates (registry gate: the
three F10.8 gates + vocabulary-overlap screen for mixed v2×wcl loadings, which remains
unmeasured — flagged for A-comp3, whose tension loading is a v2 lexicon).

## F10.7 results (T-GEO-P8, run AFTER the registration commit c5a9b4f, same day)

**P8a — in-range interpolation ladder (412 m=3 comments, 70 users, user-level MAE,
bootstrap-over-users CIs).** The ladder splits by the P7b shape taxonomy:

| construct | grand | person | func-linear | func-quad | reading |
|---|---|---|---|---|---|
| first_person | 2.166 | 1.902 | **1.516** | 1.504 | endpoints beat person by +0.378 (CI [−0.13, +0.90], direction as registered, ns at this n); quadratic adds nothing — flow-consistent |
| wcl_15 | 0.793 | 0.933 | **0.555** | 0.638 | endpoints beat person, +0.390 CI [+0.03, +0.83] — SIGNIFICANT in-range functionalization |
| wcl_36 | 1.056 | 0.986 | **0.773** | 0.805 | endpoints better, +0.217 (ns) |
| tension | 0.414 | **0.377** | 0.485 | 0.568 | PERSON best — registered "curvature helps" prediction WRONG |
| directive | 0.608 | **0.635/0.608** | 0.718 | 0.844 | person/grand best — registered prediction WRONG |
| novelty, wcl_02/54 | — | **best** | — | — | person best |

Post-hoc reading (recorded as such): for BOUNDARY-type constructs the interior is the
person's baseline and the endpoints are the contaminated zones — so endpoint
interpolation overshoots and the correct in-range function is "person baseline + zone
offsets," not endpoint averaging. For FLOW-type constructs the comment's own endpoints
genuinely carry the interior. In-range "position + function → height" holds, but the
right function differs by shape class.

**P8b — out-of-range coordinate race (fit m ≤ 3, predict m ≥ 4 interiors; 69 long
comments, 54 users, 204 interior / 66 deep windows — PILOT as declared).** Deep-interior
MAE (constant / relative-t / zones), user-level, intercept recalibrated on long endpoints:

| construct | const | relative | zones | winner |
|---|---|---|---|---|
| first_person | 1.365 | **1.114** | 4.167 | **RELATIVE — decisively** (zones−rel +3.05, CI [+3.03, +3.08]); registered lean (zones) WRONG |
| tension | 0.179 | **0.110** | 0.380 | relative |
| wcl_36 | 0.613 | **0.330** | 2.198 | relative (+1.87) |
| novelty | 0.323 | 0.277 | **0.079** | zones (−0.198) |
| wcl_54 | 2.184 | 1.674 | **1.093** | zones (−0.581) |
| wcl_22 | 3.301 | 2.642 | **1.542** | zones (−1.109) |
| wcl_25 / wcl_15 | **best** | — | — | nothing transports (2 of 10) |

**F10.7 verdict.** The conjecture HOLDS quantitatively, in both halves: (in-range) a
function rung beats or matches the person rung for flow-type constructs, significantly
for wcl_15; (out-of-range) for 8 of 10 highlighted constructs some transported function
beats the constant baseline on genuinely never-observed deep cells — and the WINNING
COORDINATE follows the P7b shape taxonomy: flow-type constructs transport as functions of
relative position t (first_person's win is decisive — the conjecture's strongest form:
specify t, predict the height, even out of range), plateau/dip-type constructs transport
as zone/plateau functions. Two constructs do not transport at all (wcl_25, wcl_15 deep).
My registered lean for first_person (zones) was WRONG — the m=3 monotone pattern is a
genuine relative-position flow, not boundary settling in disguise.

**Stated limitations.** (a) The zones model's short-support identification is confounded
(its τ≥2 parameter comes from a δ=0 cell and vice versa), so its deep-cell failure for
flow-type constructs partly reflects the parameterization, not the falsity of boundary
anchoring where P7b established it; a cleaner zone fit needs m=4 in the fit set at the
cost of evaluation data. (b) Evaluation cohort is thin (69 comments / 66 deep windows) —
pilot. (c) P8b CIs are tight because systematic model bias dominates user noise; they are
CIs of the MAE difference, not of construct effects.

**Consequence for measurement.** Position-conditional scoring now has a concrete recipe
per shape class: flow-type constructs should be scored with a t-regression (or t-norming)
inside long texts; boundary-type constructs with zone offsets (open/close/interior);
plateau-type with interior-only scoring. This is new battery-design material for the
SUICA-native corpus (design doc cross-reference) and a candidate robustness fix for the
pass-A opening-share confound flagged in F10.6.

## F10.6 results (T-GEO-P7/P7b, run AFTER the registration commit e839ed9, same day)

**Data-shape discovery first.** PANDORA's long-comment length distribution is too steep
to fill a five-point curve: 43,818 windows from 21,498 comments (3,622 users), but median
m = 2, so bins 0/4 carry 21.5k windows each while bin 2 gets 659 (294 users, m = 3
comments) and bins 1/3 get 80 windows (54 users, m >= 4). Interior FRAME estimates are
therefore underpowered by construction; interior MEAN tests retain usable power. The
within-comment permutation null preserves each comment's bin set, so all p-values are
composition-matched; bin VALUES, however, mix composition (interior bins = long comments
only) and must not be read as trajectories — the m = 3 paired stratum (P7b) is the
composition-free readout.

**T-GEO-P7 (five-bin curve).** Sanity anchor held: first_person bin4 − bin0 = −0.741,
reproducing P5's direction on the perfectly matched endpoint pair (bins 0 and 4 receive
one window from EVERY comment). Registered edge-localization prediction: SUPPORTED where
powered, for means — tension edge share .742 (p = .027), novelty .868 (p = .040),
first_person .774 (p = .053, marginal); ALL interior mean steps within null (interior-max
p = .64–.96 for those constructs). No wcl construct shows significant edge structure.
TRUE-FLOW reading: no support anywhere. Frames: no adjacent step beyond null (p =
.08–.77), edge share .488 (p = .74) — verdict WITHHELD for frames (54–294 users in the
interior cannot estimate a 19-dim frame); the well-powered endpoint congruence
[.947, .942, .835, .815] replicates P5's PC1/PC2-hold / PC3-PC4-move pattern.

**T-GEO-P7b (m = 3 paired stratum, composition-free).** 412 comments with exactly three
windows, 70 users (gate >= 2 comments; thin cohort, flagged), user-level statistics, S_3
within-comment null (2,000 draws). Every position estimated from the SAME comments, so
shape claims are clean; this test takes PRECEDENCE over P7 curve steps for shape claims.

| construct | open / mid / close | linear (p) | curvature (p) | shape verdict |
|---|---|---|---|---|
| first_person | 2.908 / 2.043 / 1.720 | −1.188 (< .0005) | −0.271 (.156, ns) | **TRUE FLOW — monotone decline, mid ON the endpoint segment** |
| tension | 0.239 / 0.216 / 0.618 | +0.379 (< .0005) | −0.212 (.001) | closing-zone jump (boundary) |
| directive | 0.890 / 0.442 / 0.500 | −0.390 (.001) | −0.253 (.008) | opening elevation (boundary) — invisible to the two-point contrast; NOTE P5's all-strata two-point was +0.026 ns → stratum-dependence flagged |
| novelty | 0.163 / 0.242 / 0.145 | −0.018 (.782) | +0.088 (.100) | no clean verdict (P7's edge share p=.040 not corroborated here; underpowered) |
| wcl_22 | 4.328 / 3.032 / 3.671 | −0.657 (.029) | −0.968 (< .0005) | mid-text dip (a THIRD shape) |
| wcl_54 | 2.435 / 2.049 / 2.470 | +0.035 (.861) | −0.404 (.041) | mid-text dip |

**F10.6 verdict.** The registered edge-localization hunch was PARTLY WRONG, and the
continuization earned its keep: position structure is CONSTRUCT-SPECIFIC. The flagship
first_person genuinely FLOWS (gradual monotone decline along the text — the user's
continuous-variable reading holds for it); tension and directive live in boundary
micro-registers (closing jump / opening elevation); two style clusters dip mid-text.
Neither pure reading (all-flow / all-boundary) survives — t is a licensed continuous
condition variable in F10.1, and each construct carries its own position profile, which
is itself new measurement material (position-conditional norms; opening/closing/interior
as micro-registers for the boundary-type constructs). Follow-ups: directive's stratum
dependence (m=2 vs m=3); position profiles as user-level signatures (does the SHAPE of a
person's first_person decay individuate? — would be F10.3 x F10.6); Essays' long documents
as the natural corpus for dense interior coverage (frozen benchmark governance applies —
Tier-U-equivalent text only, no labels).

## F10 results (T-GEO-P4/P5/P6, run AFTER the registration commit 60ce99c, same day)

**P4 — time flow of the population frame: PREDICTION HELD (rigid).** Pooled arm
(n = 4,454): D14 = .2848 vs null mean .2603 (p = .342); PATH = .8248 vs .7624 (p = .244).
Venue-held arm (n = 3,278): D14 = .2266 vs .2457 (p = .662); PATH p = .096. No movement
beyond the within-user block-permutation null at block scale — consistent with F9's
early/late congruence .93–.99. λ spectra flat across blocks.

**P5 — position flow: prediction PARTLY WRONG (frame rotation is real).** 21,496 comments
≥ 288 tokens (3,614 users; frame cohort 1,380 users ≥ 5 comments). Frame: chordal(open,
close) = .4919 vs swap-null p95 .3475 (p < .002); per-PC best congruence [.965, .940,
.732, .780] — PC1/PC2 position-invariant, PC3/PC4 REORIENT, while eigenvalue spectra
barely move (orientation shifts, strength does not). Mean shifts (user level, close −
open): first_person −0.687 (p < .0005; registered hunch CONFIRMED — openings are
first-person territory), tension +0.055 (p < .0005), directive +0.026 (p = .104 —
registered hunch FAILED, recorded); wcl_36/54/22/15/25 lower at close, wcl_02 higher.

**P6 — personal axes: CONJECTURE SUPPORTED; kill condition NOT triggered.** Pooled arm
(n = 2,430 users, ≥ 12 slices per half):

| variant | self | stranger | p | ceiling early/late (odd-even) |
|---|---|---|---|---|
| raw | .2567 | .2237 | < .0033 | .2554 / .2508 |
| residualized (pop top-4 projected out) | .2769 | .2155 | < .0033 | .2805 / .2917 |

Self-congruence sits AT the odd/even ceiling → within-person frame rotation is at most
mild (the ceiling uses half-sized estimates, so exactly zero is not claimable — a
φ-correction of frame congruence is the natural F8×F10 follow-up). The decisive row is
the residualized one: the self−stranger gap WIDENS (.061 vs .033) after the shared
four-factor frame is projected out — personal orientation is not a noisy copy of the
population factors. Venue-held arm (n = 938, ≥ 8 slices): raw ns (.204 vs .199,
p = .173) but residualized significant (.2435 vs .2078, p < .0033) — within a single
venue the personal orientation appears once the shared frame is removed. Alignment
dispersion: median best |cos| of personal axes vs population PC1–4 = .4929 (IQR
.394–.598) against a random-direction median .3332 (p95 .5475); best-match shares
.42/.27/.17/.14 across the four PCs — the population frame describes individual frames
poorly.

**F10 verdict.** The population frame is rigid in time (P4, confirming F9); position
within a text bends the third and fourth axes while leaving the first two and the
spectrum intact (P5); and the person-level frame is a stable, individually informative
orientation that the population factors do not capture (P6). Nameability (F10.2) fails at
clause (ii) — person-sharedness — exactly as the conjecture proposed: the axes' role
shifts from named population constructs to carriers of individuality. The F10.3
representational consequence now has empirical backing: (f_u, U_u) with a principal-angle
term is a licensed exploratory object. Naming individual axes remains UNLICENSED (the
E11-v2 calibration rule applies unchanged); all congruence magnitudes are attenuated by
~21-slice-per-half estimation and only ORDER claims (self vs stranger, self vs ceiling)
are made.

**F10 follow-ups (ledger-noted).** (a) φ-correct the odd/even ceiling to bound true
temporal rotation of personal frames; (b) identify WHICH loadings move in the PC3/PC4
position reorientation; (c) position-conditional scoring (open/close as micro-registers)
as parallel-form material; (d) whether personal-axis geometry carries Tier-L signal —
requires a NEW seal (NOT opening #2's registered candidates); deferred.

## Status

F7–F9 are formal apparatus plus first instantiations, all three computed label-free on
Tier-U (T-GEO-P1/P2/P3; ledger rows carry the authoritative numbers). F10 was registered
as conjecture + instruments + falsifiers at commit 60ce99c BEFORE the runs; T-GEO-P4/P5/P6
results above landed the same day (ledger rows TGEO-P4/P5/P6). The manuscript carries the
compact version (exploratory subsection + A.7(iv), v5.6, EN/JA/ZH).

**T-GEO-P2 result (time-reversal drift, K=4 blocks, span >= 180 days, sign-flip null 2000
draws).** Pooled arm (n = 4,454): first_person mean slope −0.224/yr (p < .0005), directive
−0.036 (p < .0005), tension −0.016 (p < .0005), novelty ns. Venue-constant arm (top-1
condition, n = 3,278): **first_person −0.247/yr (p < .0005) — the drift survives holding
the venue fixed**, i.e., it is within-person change, not venue-mix change; tension and
directive lose significance (their pooled drift WAS venue-mix drift — the two-arm design
separated the channels as intended); novelty +0.015 (p = .049, borderline). Reading:
first_person shows the classic maturation pattern — rank-order stable (raw .68) with
mean-level decline — which is what a genuine trait-like quantity does in longitudinal
personality data; it also marks a boundary condition for within-person monitoring (L1):
drift must be modeled, not assumed away. Nothing here touches frozen batteries, seals, or
labels. The wcl_60
F8.3 violation is the headline theoretical yield: it independently rediscovers, from pure
aggregation geometry, the zero-inflation phenomenon that motivated the v5 hurdle formula —
two instruments, two registers, one mechanism.
