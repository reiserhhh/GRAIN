# HM64: source-fixed joint author coupling

2026-10-09. Finite mechanism protocol, frozen before outcome inspection.
The parent goal and original 18 protocols / 100 cards / 44 questions remain open.
HM45 fidelity work stays deferred; HM59 template diagnostics stay closed.

## Question and scope

Do the within-comment co-location profiles of four frozen lexical GRAIN
coordinates and fixed E5 query-configuration proxies recur across the same
author's disjoint material panels? Does exact comment pairing contribute beyond
the two channels' mean locations? This is a descriptive author-measurement
interface question, not a personality-validity or prediction-accuracy contest.
Earlier NEXT27 already computed Cov(G,psi) and external correlations: covariance
itself is not a new finding. Here the new object is its source-calibrated,
coordinate-invariant, cross-panel author recurrence and matched-stranger contrast.

Use exactly the HM63 sealed numeric inputs: 276 authors, 9,165 comments,
213 Big5-only and 63 MBTI-only authors separately. Each author has three technical
panels with 7--12 eligible comments. They are not dates, scenarios or repeated
psychological occasions. No raw text, questionnaires, labels, new model calls,
API, generation, neural training or new proxy fitting. Previously frozen
coordinates/maps are inherited, not claimed to have been newly learned on a
completely independent discovery population.

## Common units, preserved levels

For each cohort and existing author fold f, apply the SAME old query-map
coefficient matrix W_f to reference and held-out comment embeddings:

    y = h W_f;  C_uv = mean[(g-mu_g)(y-mu_y)^T].

The reference population consists only of authors with fold != f; each author
and panel receives equal mass and each comment equal mass within its panel.
Freeze source means, full marginal covariances Sg/Sy, Cref, and block RMS units
before any held-out distance calculation. Retain both mean-location channels;
centering C against Cref is relative scoring, not deletion of a common signal.
No author-specific normalization, chosen ridge, dimension, kernel or rank.
Reject non-SPD marginal reference units at a fixed machine tolerance; report the
failure rather than selecting a successful regularizer. A zero-energy profile
gets an explicitly recorded neutral unit and remains zero, never a fake signal.

    D_uv = Sg^(-1/2) (C_uv-Cref) Sy^(-1/2)
    d_C(a,b)^2 = tr[Sg^(-1) (Ca-Cb) Sy^(-1) (Ca-Cb)^T].

This finite rectangular-tensor metric is invariant under jointly transported
invertible coordinates and reference covariances. It is not an SPD manifold
distance, an infinite-kernel independence criterion, or the person's causal B_u.

## Fixed contrasts and evaluation

Seven arms, all retained: G_location, Q_location, joint_location, coupling,
shifted_coupling, joint_with_coupling, joint_with_shifted. Joint blocks are
source-RMS normalized with equal energy weights (1/sqrt(number of blocks)).

The shift control rotates y by floor(n/2) comment positions within each panel,
preserving both marginal means/covariances while disrupting exact pairing.
Sorted cache positions are not chronological time. Shifted C has its own
source reference/RMS, frozen by the same rule; no application tuning.

For all six directed panel pairs, compare the correct author's destination
distance to every other held author in the same cohort/fold. Report each anchor's
stranger-win fraction (ties within 1e-12 count half), then average the six tasks
per author and authors per cohort. This is not pooled dyad AUC.
The hard control uses the three nearest strangers under joint_location, with
the identical stranger set for every arm. It is not topic matching.
Report paired contrasts coupling-minus-shifted and joint_with_coupling-minus-
joint_location; retain the shifted joint contrast too. Bootstrap whole author
summaries 2,000 times (seed 20261009), conditional on the frozen references.

199 destination-identity permutations per fold, shared across arms/panel pairs,
are a descriptive null. For a permuted nominated identity exclude that identity
from its stranger set; recompute its three nearest location strangers. The
all-stranger null averages .5 in expectation; the hard-stranger null generally
does NOT, and must be reported empirically. No population psychological p-value.

## Failure-capable checks and delivery

Synthetic worlds: identical means with author-specific repeatable g-y relations;
constant shared relation with no author information; independently changing
relations between panels. Check shift marginal preservation, coordinate
invariance, source/held isolation, singular reference failure, ties and complete
pair counts. A negative natural result is retained without parameter rescue.

Canonical output: results/grain_math_response_v1/joint_author_coupling_r1.
prepare -> run -> verify; no overwrite/retry. Max 300 CPU wall seconds.
Save source units before application, private numeric profiles/distances,
author summaries, aggregate contrasts/nulls, hashes and recoverable checkpoint.
Verifier replays raw cross-products and trace distances independently of the
whitened-distance implementation. Same-agent arithmetic verification is not
independent replication. Neither file/test counts nor author recurrence closes
psychological meaning, human response parameters or independent new-text scoring.

## Mathematical provenance

Fukumizu et al., Kernel Measures of Conditional Dependence (NIPS 2007),
Section 2.1 equations 1--5, motivates separating cross-covariance and units:
https://papers.nips.cc/paper_files/paper/2007/file/3a0772443a0739141292a5429b952fe6-Paper.pdf
The present GLOBAL source units do not equal each author's NOCCO marginals;
operator norm <=1 and kernel conditional-independence theorems are not imported.
The trace identity/invariance are standard tensor/Mahalanobis algebra.

OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a, metric/connection
inverse-problem introduction, was reread for observable geometry versus gauge:
preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/01-introduction.tex.
Only a design analogy is used; no smooth-manifold/PDE uniqueness theorem applies
to these caches. New contribution, if any, is the measured coupling interface
and its cross-material author organization, not invention of these formulas.
