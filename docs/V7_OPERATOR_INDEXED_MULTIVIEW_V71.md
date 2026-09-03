# GRAIN V7.1 Operator-Indexed Multiview Measurement System

## Status

V7.1 is an engineering and mathematical audit layer for competing text
observation operators. It does not define a psychological construct, a
clinical score, or an external-scale prediction system. It exists to establish
whether a reported slicing advantage survives strict source-level controls.

The prior V7 smoke run demonstrated that several operators could create frozen,
author-relative label-free scores. It did not establish that their raw
factor-score SEMs were comparable across operators, and it used an even/odd
unit split that can leak source text across cross-comment fixed windows.

## Measurement Model

For author \(u\), operator \(j\), and source-disjoint technical replicate
\(r\), V7.1 uses the provisional decomposition

\[
X_{u j r} = \mu_j + \Lambda_j z_u + \Delta_j h_{u j}
            + \Gamma_j c_{u j r} + \varepsilon_{u j r}.
\]

- \(z_u\): anonymous shared author configuration.
- \(h_{u j}\): operator-conditioned author structure. It is not error only
  if it recurs under source-disjoint repetitions.
- \(c_{u j r}\): condition, opportunity, and selection path such as source,
  length, time, prompt, turn, or community. It may covary with \(z_u\):
  \(c_{u j r}=\alpha_j+H_jz_u+v_{u j r}\). It is therefore not automatically
  residualized or treated as noise.
- \(\varepsilon_{u j r}\): finite-text sampling error.

Saved-model replay error is an engineering issue and is reported separately;
it is not part of \(\varepsilon\).

An operator is a lossy transformation, not a demonstrated coordinate chart.
The word *atlas* is reserved for a later result in which directed maps, shared
subspaces, and cycles are empirically tested.

## Slice-Family Principle

Different slicing rules are treated as parallel **observation operators**
\(\Phi_j\), not as contestants in a single universal-preprocessing race. For a
source-comment panel \(C_u\), each view is

\[
X_{u j}=\Phi_j(C_u),\qquad j\in\{\text{native},\text{sentence-pack},
\text{fixed-within},\text{fixed-cross},\text{whole},\ldots\}.
\]

The empirical questions are therefore: which coordinates recur within a view,
which subspaces or directed maps carry across views, and which discrepancies
remain after provenance and opportunity controls. A shared component may be
more robust than any one view; a view-private component is retained as a
candidate text-geometry object rather than erased as error. It is promoted
only when it has a frozen scoring rule, source-disjoint stability, support
coverage, and an interpretation/validation path. E0 rules out the shortcut of
calling one fixed cross-comment window universally "best"; it does not rule
out fixed windows as a distinct observational plane.

## E0: Boundary and Fixed-Window Audit

Every E0 operator receives the same deterministic source-comment panel:

- same reference authors;
- same capped number of source comments;
- same source-token budget;
- one discovery-fitted common TF-IDF/SVD representation;
- no Big Five, MBTI, or other external labels.

The compared views are native comments, sentence packing, whole-author text,
and the following fixed variants: cross-comment, no-cross-comment,
boundary-marker, a fixed 64-token phase shift, deterministic author-hash
phase, and 64/128/256-token widths.

Each fixed-window unit records the source rows it actually overlaps. The old
behavior that marked every fixed window as containing all capped comments is
invalid and is not reused for V7.1 evidence.

### E0 metrics

1. Calibration-normalized source-cluster bootstrap SEMs. Raw factor-score
   SEMs are not compared across independently fitted factor spaces.
2. Source-disjoint half agreement. Splits occur before slice construction;
   no source comment may occur in both halves.
3. Author-correspondence permutation nulls for the half agreement.
4. Error-to-between-author variance ratio, score coverage, source coverage,
   boundary-crossing rate, and window/marker perturbation sensitivity.
5. Exact frozen-replay check for representation and factor bundle artifacts.

### E0 decision rule

Call a fixed-window precision advantage **robust** only if its advantage over
native comments remains positive with a bootstrap lower confidence bound above
zero for the required boundary controls: no-cross, marker, and both offset
controls. A gain limited to ordinary cross-comment windows is reported as a
possible boundary-smoothing artefact. Missing support produces an
*inconclusive* result rather than a favorable interpretation.

E0 reuses the current 180-author engineering cohort. Its output is an
engineering audit, not a fresh confirmation result. The next multiview
decomposition must exclude this cohort or be explicitly marked exploratory.

## V7.2 Closure Work Packages

1. **E0 / complete:** boundary, provenance, and normalized precision audit.
2. **E1 / rerun under V7.2:** full admissible-rank search, calibration one-SE
   selection, bootstrap loading-subspace alignment, and exact artifact replay.
3. **E2 / complete:** condition/opportunity contribution accounting on
   PANDORA, without automatically subtracting endogenous author choice.
4. **E3 / exploratory only:** directed operator maps, canonical relations,
   cycle error, and asymmetry reuse E1 confirmation authors. They are retained
   as `HOLDOUT_REUSED_EXPLORATORY` until a new registered family is run.
5. **E2 follow-up / not yet licensed:** improve the condition coordinate
   system before attempting a stronger transport claim. Do not blanket-
   residualize text or treat the observed primary surface as causal.
6. **E4 / protocolized:** fixed/free text is a pure holdout, not training input
   or external validity evidence. The new validator refuses q estimation
   without event-level opportunity logging and refuses B estimation without
   randomized repeated fixed conditions. The existing MEPS material remains a
   feasibility source, not a substitute for that design.

## Executed Evidence Ledger (2026-07-14)

### E0: Boundary audit, reused engineering cohort

The source-provenance correction was required: the older fixed-window code
recorded every capped comment as the source of every window. V7.1 now records
only the comments actually crossed by each window. On 180 engineering authors
with 50 confirmation authors, all 240 source-disjoint factor comparisons had
zero source overlap and all 2,000 source-cluster bootstrap SEMs were observed.

The predeclared result was `FIXED_ADVANTAGE_NOT_ROBUST`.

- `fixed_128_cross` did not improve normalized precision over native comments:
  delta native-minus-fixed = -0.021, 95% bootstrap CI [-0.064, 0.034].
- No-cross, boundary-marker, and offset controls all had confidence intervals
  crossing zero or favoring native comments.
- Thus fixed slicing remains an observation view, not a promoted universal
  precision improvement.

Artifacts: `reports/V7_OPERATOR_BOUNDARY_AUDIT.md` and
`results/v7_operator_boundary_audit/e0_full_20260714/`.

### E1: Fresh-author shared/private projection (V7.2 replayable rerun)

E1 excluded all 180 E0 authors and selected 240 fresh authors (141 discovery,
51 calibration, 48 confirmation). It used a discovery-fitted common source
comment representation, no external labels, and five observation views.

- V7.2 searched every admissible rank 0--24. Calibration's raw maximum was
  rank 24, but the predeclared author-bootstrap one-SE rule selected the
  smaller rank 23 (alpha 10.0); it did **not** hit the admissible boundary.
- Confirmation mean shared-subspace cross-view R2 was 0.501; the broken-
  correspondence direct-map reference averaged -0.365.
- Every operator's loading subspace recurred under discovery-author bootstrap:
  lower congruence 0.790--0.957 versus random-subspace upper bounds
  0.503--0.515.
- No anonymous axis is identified: alignment lower bounds are 0.463--0.525
  and median assignment margins are negative. All five views are therefore
  `FACTOR_AXES_NOT_IDENTIFIED_SUBSPACE_ONLY`, not a new factor inventory.
- Artifact inventory and reload audit passed exactly. This is a reproducible
  subspace result, not a claim about named factors or psychological traits.

### E2: Fresh-author condition/opportunity contribution accounting

E2 selected 600 fresh PANDORA authors after excluding the E0/E1 cohorts. Each
author contributed exactly 48 chronologically spaced source comments, divided
into source-disjoint alternating A/B halves. The discovery/calibration/
confirmation split was 379/112/109 authors. The pipeline read no external
labels and persisted neither raw text nor author IDs.

For each comment vector \(y_{ui}\) and recorded context surface \(d_{ui}\),
the discovery-fitted analysis used

\[
\hat m(d) = \widehat{E}[y \mid d],\qquad
T_{uh}=\bar y_{uh}-\bar y_D,\qquad
K_{uh}=\overline{\hat m(d)}_{uh}-\overline{\hat m(d)}_D,\qquad
A_{uh}=\overline{y-\hat m(d)}_{uh}-\overline{y-\hat m(d)}_D,
\]

so that \(T_{uh}=K_{uh}+A_{uh}\) exactly. The primary context surface was
subreddit plus calendar-time coordinates; visible length/format variables were
strictly sensitivity-only. Context vocabulary, scales, penalties, selection
shrinkage, and matching caliper were fitted before confirmation.

The valid primary-native confirmation result is
`TOTAL_SELECTION_AND_PARTIAL_STRUCTURE_SUPPORTED_WITH_UNRESOLVED_CONTEXT_TRANSPORT`.

- Total A/B author alignment: AUC 0.916, 95% CI [0.890, 0.938], Holm-adjusted
  permutation q=.005.
- The author selection profile recurred across halves after empirical-Bayes
  shrinkage: log-score gain 0.894 [0.741, 1.042], q=.005.
- **Audit correction:** the earlier r4 partial screen used a nonexchangeable
  unrestricted-author null and is not used for inference. The r6 screen freezes
  disjoint 4--8-author strata from calibration-time metadata only, then
  permutes B identities *within each stratum*. It retained 86/109 confirmation
  authors in 11 strata (sizes 6--8) and gave partial AUC 0.834
  [0.766, 0.896], Holm q=.005. The selected numeric caliper was 4.0; the
  feasible Jaccard caliper was 0.0, so this is not literal cross-author
  matched-subreddit equivalence.
- Recorded primary context predicted only 0.0045 of comment-vector variance
  [0.00024, 0.00768], q=.005: statistically nonzero, but below the registered
  material threshold 0.02.
- The calibration-selected map from context-carried \(K\) to total \(T\) had
  held-out R2 0.0209 with CI [-0.0144, 0.0339]. It does not meet the positive
  lower-bound transport rule.
- The algebraic decomposition held numerically (maximum identity error
  \(1.10\times10^{-15}\)). Variance shares are not reported as standalone
  fractions because \(K\) and \(A\) are not orthogonal.

Thus E2 supports recurring author-conditioned selection and residual
author-relative structure under declared recorded controls. It does not show
that subreddit/time causes the geometry, that the primary context surface
accounts for the total structure, or that any component is a personality
factor. A high K retrieval AUC and an unresolved K-to-T R2 are not a
contradiction: recurrence of a low-variance author-aligned component does not
establish that it reconstructs the higher-variance total configuration.

Artifacts: `reports/V7_CONDITION_OPPORTUNITY.md` and
`results/v7_condition_opportunity/e2_full_20260715_r7/`. The run retains the
caliper grid, strata, balance table, and within-stratum permutation null.

### E3: Directed operator atlas (holdout-reused exploratory)

The V7.2 atlas reloads frozen E1 artifacts and is explicitly
`HOLDOUT_REUSED_EXPLORATORY`, because it reuses the E1 confirmation cohort.
Its descriptive maps report mean direct R2=0.626, broken-correspondence
mean=-0.353, and mean three-edge cycle R2=0.641. These numbers are not a new
confirmation family. Held-out directed maps describe a topology of the tested
transformations:

- Native, sentence-pack, and within-comment fixed views are near-equivalent in
  this representation (bidirectional R2 about 0.94 to 0.97).
- Cross-comment fixed is more distinct (roughly 0.33 to 0.46 to/from those
  fine-grained views), while still far above broken correspondence.
- Whole-author aggregation is asymmetric: fine-grained views predict whole
  features at R2 about 0.70 to 0.85, but whole-to-fine maps are only about 0.27
  to 0.42. This is consistent with an information-compression relation.
- Three-edge cycle R2 remains below one for every source view. The views are
  therefore not established as interchangeable coordinate charts under this
  linear map family.

These are operator-indexed text-geometry results. They do not identify
psychological factors or demonstrate personality validity.

### E4: Registered cross-representation family and source-disjoint control

E4 selected 240 new authors after reconstructing and excluding all E0, E1,
and E2 cohorts (1,020 prior authors). It predeclared five operators and word
1--2-gram / character 3--5-gram TF-IDF-SVD24 representations. All 52
same-source shared-subspace, directed-edge, and cross-representation endpoints
passed one synchronized 199-permutation max-T family (all adjusted p=.005).

The source-disjoint result has two distinct layers:

- **Linear coordinate transport:** only the whole-author view passed (2/10
  representation-by-operator endpoints). Fine-grained native, sentence, and
  fixed views had negative held-out R2. Raising source support from 32 to 48
  comments did not reverse this.
- **Relative-position alignment (POST-HOC EXPLORATORY; relabeled 2026-07-17,
  see dated correction below):** a separate own-vs-stranger source-disjoint AUC
  family passed for all ten views at 32 comments
  (AUC=.719--.805, max-T p=.005). In the exploratory 48-comment support arm it
  rose to .791--.862, again all max-T p=.005.

Thus the current data support an operator-indexed **metric relation** between
independent text samples, but do not support a stable common linear coordinate
map for fine-grained views. This is compatible with a rotating/high-dimensional
configuration and is not evidence for named latent factors, personality, or a
universal slice choice.

**Process-audit correction (2026-07-17).** The own-vs-stranger alignment family
was previously described in this document as "registered". The 2026-07-15
process audit established that it was designed and added AFTER the 8/10
fine-view transport failures were on disk. Timeline (2026-07-15 UTC): r1
17:10:53Z contained same-source endpoints only; the registry config file was
modified at 17:12:39Z; r2 17:15:31Z added the source-disjoint transport family
and 8/10 endpoints FAILED; r3 17:21:17Z added the alignment family, which
passed 10/10. The alignment result stands as computed — every number above
(.719--.805, p=.005, and the 48-arm .791--.862) is unchanged, and the max-T
machinery was independently verified — but its evidentiary tier is
EXPLORATORY-POST-HOC, not registered-confirmatory. A genuinely registered
own-vs-stranger confirmation on a fresh cohort, with the family committed
before any source-disjoint run, is the listed follow-up.

## Claim Boundary

V7.2 can at most show reproducible, operator-indexed, author-relative text
structure with quantified technical text-sampling uncertainty. It cannot yet
claim a personality factor, stable human trait, clinical construct, or
cross-language universal score.
