# GRAIN Cross-Level Inference Contract

Status: `CANONICAL_SCOPE_CONTRACT__TEMPORAL_AUDIT_COMPLETE`

Opened: 2026-08-22. This document corrects the inference scope of the M4-X
measurement-series line without changing any historical artifact, verdict
name, estimate, or release seal.

## 1. Why this contract exists

The X line established that one dense longitudinal event stream can be read
through three measurement series: between-person, average within-person, and
person-specific. Three different claims were then compressed too aggressively:

1. common physical units were called measurement invariance;
2. a difference between two registered slopes was called process ergodicity;
3. cross-half covariance was called attenuation-free true variance without
   carrying its independence and stability assumptions.

The corrected principle is:

> A common unit makes outputs commensurable. It does not make their meanings,
> response functions, or temporal laws invariant.

The formal paper by Saegusa and Geshi (2025) motivates the separation of
measurement series. The later conference discussion of an ideal "Level 5"
was an oral conceptual extension, not a five-level taxonomy stated in that
paper. Here it is represented only as an asymptotic ideal in Section 8.

## 2. Observed event system

Let an observed corpus be

\[
\mathcal D
=
\{(u,t,c,x_{ut},y_{ut},q_{ut})\},
\]

where \(u\) is author, \(t\) is event order or time, \(c\) is condition,
\(x,y\) are registered technical channels, and \(q\) records support and
observation quality. Every statement below is indexed by:

\[
I=(\mathcal D,P,Q,S,R,r,\tau),
\]

the corpus, reference population, opportunity distribution, slice operator,
representation, relation, and observation horizon. Omitting this index does
not make an estimand universal.

For relation \(r\), define author means \(\bar x_u,\bar y_u\). The level-1
between-person projection is

\[
\beta_{B}^{(r)}
=
\frac{\operatorname{Cov}_u(\bar x_u,\bar y_u)}
     {\operatorname{Var}_u(\bar x_u)}.
\]

The level-2 average within-person projection is

\[
\beta_{W}^{(r)}
=
\frac{\sum_{u,t}(x_{ut}-\bar x_u)(y_{ut}-\bar y_u)}
     {\sum_{u,t}(x_{ut}-\bar x_u)^2},
\]

with the actual X5 implementation evaluating the registered half-specific
versions and averaging them. The level-3 object is not one population slope:

\[
\beta_{u,h}^{(r)}
=
\frac{\sum_{t\in h}(x_{ut}-\bar x_{u,h})(y_{ut}-\bar y_{u,h})}
     {\sum_{t\in h}(x_{ut}-\bar x_{u,h})^2}.
\]

Its legitimate outputs are a population distribution, an estimability map,
and a reproducibility certificate. It is not automatically a psychological
score for author \(u\).

## 3. Unit commensurability is not measurement invariance

Let \(T_L:\mathcal D\rightarrow\mathcal U\) be a readout at measurement
series \(L\). If every \(T_L\) maps into the same physical unit space
\(\mathcal U\), then subtraction and dimensional comparison are defined:

\[
T_{L_1}(\mathcal D)-T_{L_2}(\mathcal D)\in\mathcal U.
\]

This is **unit commensurability**. It does not imply

\[
T_{L_1}=T_{L_2},
\qquad
P(Y\mid Z,L_1)=P(Y\mid Z,L_2),
\]

nor equality of factor structure, loadings, intercepts, residuals, causal
meaning, or response functions. Those are empirical invariance questions.

X4's sign reversal is possible precisely because the same units permit a
meaningful comparison while the relations themselves differ.

## 4. The transport gap

The historical artifact name is retained:

\[
\Delta_{\mathrm{erg}}^{(r)}
=
\beta_B^{(r)}-\beta_W^{(r)}.
\]

Its canonical interpretation is now the **projection-specific cross-level
transport gap**

\[
\Delta_T^{(r)}:=\Delta_{\mathrm{erg}}^{(r)}.
\]

Decision language:

| Evidence | Licensed statement |
|---|---|
| CI for \(\Delta_T\) excludes 0 | this registered slope does not transport between these two series at this resolution |
| detected slopes have opposite signs | transporting the slope reverses direction |
| CI for \(\Delta_T\) includes 0 | no gap resolved; not equivalence |
| CI lies inside a preregistered equivalence margin | slope transport is supported within that margin |

No row alone establishes full process ergodicity. That stronger claim would
require explicit conditions on temporal stationarity, person homogeneity,
distributional equality, horizon, and the class of functionals being
transported. A slope is one projection of a process, not the process.

## 5. Persistent covariance and its assumptions

For a person-specific parameter measured in two partitions, write

\[
\widehat\theta_{u,h}
=
\theta_u+d_{u,h}+e_{u,h},
\]

where \(\theta_u\) is the stable component at the declared horizon,
\(d_{u,h}\) is structured temporal or condition drift, and \(e_{u,h}\) is
measurement error. Then

\[
\operatorname{Cov}_u
(\widehat\theta_{u,1},\widehat\theta_{u,2})
=
\operatorname{Var}_u(\theta_u)
+\operatorname{Cov}_u(d_{u,1},d_{u,2})
+\operatorname{Cov}_u(e_{u,1},e_{u,2})
+\text{cross terms}.
\]

Only under adequate stability, zero-mean half-specific errors, negligible
cross terms, and

\[
\operatorname{Cov}(e_{u,1},e_{u,2})=0
\]

does the cross-half covariance identify \(\operatorname{Var}(\theta_u)\).
Without those conditions it is a **persistent covariance**, which may contain
shared drift or correlated technical error.

Likewise,

\[
\rho_{\mathrm{own}}
=
\operatorname{Corr}_u
(\widehat\theta_{u,1},\widehat\theta_{u,2})
\]

is reproducibility at one partition rule and horizon. It licenses a
population-level ownership distribution, not a timeless person attribute.

## 6. Claim gates

The gates are conjunctive. Passing a later-looking statistic cannot skip an
earlier gate.

| Gate | Question | Minimum evidence | Current X-line status |
|---|---|---|---|
| G0 Observability | Is the channel recorded with provenance and common units? | frozen transforms, support, horizon | pass for the five metadata relations |
| G1 Estimability | Is zero calibrated on the realized design and the estimator precise? | synthetic recovery, support floors, precision ceilings | pass after defects #85-#93 for the certified objects |
| G2 Temporal persistence | Does the estimand survive more than one arbitrary partition? | multisegment stability and drift audit | projection-specific: at K=8 the disjoint cohort has 4/5 equivalent and 1 unresolved; the Big5-name-list cohort has 1 equivalent, 2 detected heterogeneous, and 2 unresolved |
| G3 Cross-level transport | Does the registered projection agree across series? | paired gap CI or equivalence test | heterogeneous across five projections |
| G4 Construct connection | Does the technical object connect to an external psychological or behavioral instrument? | frozen label join on held-out or controlled data | X3 primary coupling not detected; exploratory cells unconfirmed |
| G5 Person applicability | Is an individual reading calibrated, stable, fair, and decision-valid? | prospective repeated-person and use-specific validation | not licensed |

This gate order preserves GRAIN's personality-interpreter objective. Metadata
and text geometry can be strong technical objects without yet being named
personality constructs.

## 7. Same-corpus recurrence and replication

The following terms are distinct:

- **re-expression**: the same data and estimand under another formula;
- **analytical recurrence**: overlapping corpus/labels under another pool or
  procedure;
- **internal replication**: disjoint people or events from the same source;
- **external replication**: an independently sampled corpus and population.

SR1, U3, and X3 observations on PANDORA are not three independent external
replications. They are useful robustness evidence, but their shared source and
labels must remain visible.

## 8. The professor-discussion ideal as a limit

Represent total estimation error as

\[
\widehat\theta-\theta
=
e_{\mathrm{sample}}
+e_{\mathrm{technical}}
+b_{\mathrm{model}}
+d_{\mathrm{target}}
+a_{\mathrm{construct}}.
\]

With identifiability, adequate support, independent repetitions, and a stable
target, the first two terms may satisfy

\[
\lim_{n,T\rightarrow\infty}
E[(e_{\mathrm{sample}}+e_{\mathrm{technical}})^2]=0.
\]

More observations do not automatically remove model misspecification,
construct ambiguity, or genuine target change. Therefore the oral "Level 5"
idea is useful as an ideal boundary: every remaining error term must be named
and assigned either a vanishing condition or an irreducible status. It is not
a promise that social-scale aggregation makes total error zero.

## 9. X5-T temporal transport audit — registered before execution

Status: `REGISTERED_NOT_RUN` at the commit introducing this document.

This is a post-closure diagnostic of X5, not a new personality claim and not a
reopening of frozen X5 verdicts. It uses the existing label-free metadata
cache and never opens text bodies or external labels.

### 9.1 Data and partitions

- Source: `results/m4_x5_ergodicity_atlas/event_cache.npz`.
- Relations and transforms: X5 R1-R5, unchanged.
- Cohorts: disjoint primary and Big5-name-list replication, unchanged.
- Partitions: \(K\in\{2,4,8\}\) contiguous event-order segments.
- The original early/late boundary is preserved. For K=4 or 8, each original
  half is subdivided into K/2 equal-count contiguous usable-event blocks.
- Minimum events per segment:
  \(m_K=\lceil50/(K/2)\rceil\), giving 50, 25, and 13.
- Denominator floor per segment:
  \(d_K=1/(K/2)\), giving 1, 0.5, and 0.25.
- The K=8 eligible author set is the common comparison pool for K=2/4/8, so
  changes across K cannot be attributed to changing authors.
- Pre-run label-free feasibility counts observed before any x-y estimand:
  disjoint 7,353-7,955 and Big5 941-1,105 across relations.

### 9.2 Segment estimands

For each segment k on the fixed K=8 pool:

\[
\beta_{W,k}
=
\frac{\sum_u\sum_{t\in k}
(x_{ut}-\bar x_{u,k})(y_{ut}-\bar y_{u,k})}
{\sum_u\sum_{t\in k}(x_{ut}-\bar x_{u,k})^2},
\]

\[
\beta_{B,k}
=
\frac{\operatorname{Cov}_u(\bar x_{u,k},\bar y_{u,k})}
{\operatorname{Var}_u(\bar x_{u,k})},
\qquad
\Delta_{T,k}=\beta_{B,k}-\beta_{W,k}.
\]

An author-cluster bootstrap resamples authors and recomputes every segment
statistic. B=500, seed=20260822. Pointwise CIs are descriptive. Familywise
heterogeneity uses a max-|t| simultaneous 95% interval over segment-to-mean
contrasts.

### 9.3 Registered temporal classifications

Resolution margin: \(\epsilon=0.02\) in each relation's registered slope
units, inherited from X5's transport-gap resolution floor.

| Class | Rule |
|---|---|
| `TEMPORALLY_EQUIVALENT_AT_0.02` | every simultaneous CI for \(\beta_{W,k}-\bar\beta_W\) lies inside [-0.02, 0.02] |
| `TEMPORAL_HETEROGENEITY_DETECTED` | at least one simultaneous CI excludes 0 and its point contrast has absolute magnitude > 0.02 |
| `TEMPORAL_HETEROGENEITY_UNRESOLVED` | neither rule holds |

Co-reported, never silently promoted:

- segment sign sequence for \(\beta_W\);
- segment transport-gap sequence \(\Delta_{T,k}\);
- whether the X5 aggregate cell family is stable over segments. Segment
  families use pointwise 95% bootstrap intervals:
  `SIGN_FLIP` if both slopes exclude zero with opposite signs and the gap
  excludes zero; `SAME_SIGN_GAP` if both slopes exclude zero with the same
  sign and the gap excludes zero; `GAP_SIGN_UNRESOLVED` if only the gap
  excludes zero; otherwise `GAP_UNRESOLVED`. Historical X5 labels map to these
  four families by removing the `NONERGODIC_`/`LEVELS_` naming layer;
- raw and normalized linear trend across segment index. With
  \(z_k\in[-1,1]\), raw trend is the OLS slope of \(\beta_{W,k}\) on \(z_k\),
  and normalized trend is
  \(b_{time}/\max(|\bar\beta_W|,0.02)\);
- retained author and event counts;
- K=2 reproduction against X5/X4 artifacts on their original pools.

### 9.4 Gates and stop rules

1. Cache anchors and relation transforms must match X5.
2. K=2 within-person slopes must reproduce the committed early/late values on
   the original pools to tolerance 1e-9; R1 uses X4's original pool.
3. Hand-constructed and synthetic sufficient-statistic tests must recover
   segment between/within slopes and preserve author-cluster resampling.
4. A relation/cohort arm with fewer than 500/150 authors on the fixed K=8
   pool is `INSUFFICIENT_SUPPORT` and receives no temporal classification.
5. No threshold, partition, or relation may be changed after the first real
   segment x-y statistic is written.

### 9.5 Interpretation contract

- Heterogeneity means the aggregate level-2 slope is horizon-averaged and
  should be indexed by time segment. It does not imply personality change.
- Equivalence at 0.02 is limited to this relation, horizon, partition family,
  and pool. It does not prove process stationarity or full ergodicity.
- A stable transport class strengthens the X5 projection claim; an unstable
  class narrows X5 to a corpus-horizon average. Neither outcome creates a
  psychological construct.

### 9.6 Executed outcome (2026-08-22)

The cache and K=2 reproduction gates passed for all ten relation/cohort arms at
maximum absolute error \(2.36\times10^{-16}\). All support gates passed. No
text body or questionnaire score was opened.

At the finest registered resolution, K=8:

- disjoint primary: four of five within-person slope paths were equivalent at
  0.02; R1 was unresolved; only R5 retained its historical transport family in
  every segment;
- Big5-name-list cohort: R3 was equivalent, R1 and R2 had detected temporal
  heterogeneity, R4 and R5 were unresolved; R3-R5 retained their historical
  families in every segment.

The K=2/4/8 rows are nested analyses of the same events, not fifteen
replications. The main result is therefore not "14 of 15 stable." It is:

> average within-person temporal persistence and cross-level transport-family
> persistence are separate empirical properties.

R5 is the strongest joint example in the disjoint cohort: its within slope is
equivalent at K=8 and all eight segment families remain `SAME_SIGN_GAP`. R3 is
the clearest separation: its disjoint within slope is nearly constant, while
its between-person path moves enough that the segment transport family does
not remain fixed.

Segments are equal usable-event-count blocks, not equal calendar-time blocks.
The result concerns event-order evolution at this corpus horizon. Because
usable events differ by relation, segment boundaries are nested within a
relation but need not align across relations. Segment-family retention uses
pointwise intervals and is descriptive; the registered familywise decision is
the within-slope temporal class.

Full aggregate evidence:
`reports/SUICA_M4_X5_TEMPORAL_TRANSPORT_AUDIT.md`.

## 10. Consequence for future GRAIN theory

The three-series system remains a useful methodological contribution, but its
canonical output is now typed:

\[
\text{shared unit}
\rightarrow
\text{estimable series-specific projections}
\rightarrow
\text{temporal stability certificate}
\rightarrow
\text{cross-level transport certificate}
\rightarrow
\text{external construct test}
\rightarrow
\text{use-specific person claim}.
\]

No arrow is automatic. This makes the theory stronger: it preserves the
observed sign reversal and three-series machinery while preventing a precise
technical result from being silently renamed as invariance, ergodicity, or
personality.

The executed audit adds a path geometry for cross-level transport. For segment
\(k\), center the between- and within-person slope paths:

\[
b_k=\beta_{B,k}-\bar\beta_B,
\qquad
w_k=\beta_{W,k}-\bar\beta_W.
\]

The centered transport path is \(d_k=b_k-w_k\), so its energy obeys the exact
identity

\[
E_T=\lVert d\rVert^2
=\lVert b\rVert^2+\lVert w\rVert^2
-2\lVert b\rVert\lVert w\rVert\cos\phi,
\]

where \(\cos\phi=\langle b,w\rangle/(\lVert b\rVert\lVert w\rVert)\). Thus a
stable within-person response does not imply stable transport: group geometry
may move while the within response remains fixed. Conversely, between and
within paths may move together and partially cancel in the transport gap.
These are technical path properties, not psychological constructs, but they
give the previously vague "dynamic axis" three explicit coordinates:
within-response energy, between-geometry energy, and their coupling angle.
