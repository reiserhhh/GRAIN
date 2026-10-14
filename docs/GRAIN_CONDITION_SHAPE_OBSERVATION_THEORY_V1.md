# Conditional shape as a separate author observation

HM187, 2026-10-12 JST. Construction, cold execution and alternate numerical
verification are complete. The original 18/100/44 scope remains OPEN.

## Scientific object, not another prediction fit

HM186 reads expression level, condition prevalence and conditional mean
pairing. These do not exhaust how expressions vary together. HM187 adds a
finite conditional covariance contrast on the SAME observations, without
deleting context, fitting a prediction head or replacing the original means.
Global second moments and conditional moments already exist in earlier GRAIN
work; covariance geometry is established mathematics. The empirical question
here is whether THIS author-conditioned shape repeats across disjoint texts
and whether its SOURCE target orientation repeats across disjoint authors.

The space is the fixed orthonormal 15-dimensional span of the existing
1024x15 GRAIN map, obtained once by QR. It is not full E5 covariance, learned
PCA, whitening, a selected rank or a test of all GRAIN capabilities.

## Complete finite account

For an author's observed vectors z_t and binary recorded condition c_t, define

    p = mean(c); o = p(1-p); mu = mean(z)
    rho = mean((c-p)z)
    C = mean((z-mu)(z-mu)^T)
    Qc = mean((c-p)(z-mu)(z-mu)^T)

When both conditions are observed, let delta = mu_1-mu_0 and let Sigma_j
be the finite covariance within condition j, with denominator n_j. Then

    rho = o delta
    B = o delta delta^T = rho rho^T/o
    D = (1-2p) B
    T = Qc-D = o(Sigma_1-Sigma_0)
    C = (1-p)Sigma_0 + p Sigma_1 + B

The proof is a conditional expansion: z-mu equals its within-condition
deviation plus (1-p)delta in condition 1 or -p delta in condition 0.
Within-condition deviations average to zero, so the cross terms vanish in
this finite account. No independent-noise or Gaussian assumption is needed.

Qc=T+D is an accounting decomposition, not denoising by discarding D.
C, Qc, D, T, p, mu, rho and original GRAIN coordinates all remain available.
A difference of covariance matrices need not be positive semidefinite: do
not clip its negative eigenvalues or call them noise.

D is NOT a complete response-orientation operator. At p=1/2, D=0 even when
delta is nonzero. D therefore couples choice prevalence with mean-displacement
geometry. B is the positive-semidefinite between-condition displacement tensor;
it is algebraically identified here, but no separate natural B experiment was
run in HM187. Neither observed delta nor B is a causal individual response law.

Absent conditions leave delta and Sigma_1-Sigma_0 unobserved, not neutral zero.
Weighted accounts are zero in the no-opportunity case by algebra. A singleton
condition has finite empirical covariance zero but no dispersion replication;
the cold interface records this separately as dispersion_support=false.

## Distinguishable at unchanged first observations

Consider the vector bag {+e1,-e1,+e2,-e2}. Assign condition 1 to the e1 pair and
condition 0 to the e2 pair, then exchange the assignments. The vector bag,
choice proportion, overall mean, conditional mean difference and pooled
covariance remain unchanged. T reverses sign. This is a mathematical witness
that conditional shape is a different observation, not human validity evidence.

Translation of every z leaves the covariance accounts unchanged. Under a fixed
orthogonal coordinate rotation, each operator is conjugated and its dual field
is conjugated too, preserving Frobenius readings. General invertible affine
changes require a transported metric; arbitrary affine invariance is not claimed.

## Registered comparisons and observed distinction

The frozen SOURCE population contains 386 authors, two disjoint 12-comment
views and fixed author folds of 203/183. Condition is the inherited TOP20 venue
bit, not complete topic/situation, state, intervention or expression opportunity.

Author recurrence compares <T_u0,T_u1> with ALL other authors having exactly the
same late-view condition-1 count. All 386 have an eligible stranger. The contrast
is +1.1750874354920191e-7, with upper empirical exchange rank .010 over the
199 predeclared late-author permutations. Count matching is not full topic
matching. Recurring author structure need not be a psychological trait.

For each of the five SOURCE binary target labels, construct high-minus-low
operator fields within each frozen author fold and text view. The registered
target primary is the sum of cross-fold products, averaged over four view pairs,
for T. It is -3.462993677399221e-8. Whole-label-profile and within-author condition
relink upper ranks are .985/.995. THIS general target shape bank is not promoted.

The prelisted secondary D account has a positive five-field SUM,
+3.701885068255948e-8, with .010/.005 upper ranks. E contributes
+2.0024831931782933e-8 (.010/.005). This is an unadjusted exploratory clue,
not a substituted primary, proof for each of five traits or accuracy improvement.
All fields and negative outcomes remain in the complete tables.

## Post-outcome geometry, separate from the primary

The separately sealed companion describes all operators and fields using

    A_iso = tr(A) I/15; A_aniso = A-A_iso
    <A,B> = tr(A)tr(B)/15 + <A_aniso,B_aniso>

This is standard symmetric-matrix geometry under orthogonal conjugation.
The traceless part includes directional variance AND cross-coordinate coupling;
it does not uniquely identify off-diagonal coordination or psychological content.

Author shape recurrence: isotropic +4.583306361196396e-9 (upper rank .235),
traceless +1.1292543718800547e-7 (.005). Secondary D target sum: isotropic
+7.719815628942002e-9 (.100/.040), traceless +2.9299035053617474e-8 (.010/.005).
The empirical direction is therefore not adequately described as scalar strength.
This description was chosen after results and is not confirmatory promotion.

One floating-order artifact is retained rather than rescued. Pooled covariance
is strictly unchanged by condition relinking. Both companion pooled reference
ranges are exactly zero; splitting before versus after averaging leaves a
1.7205356741102976e-22 traceless offset. The raw CSV's .005 pooled-traceless
relink rank has NO scientific interpretation. All pooled relink ranks are
invariant/undefined comparisons, not effects. Original primary invariance error
is zero and the primary pooled relink rank is 1. No code or sealed result was
edited to manufacture corrected precision.

## Construction interface and evidence boundary

The frozen cold bundle contains only version, fixed map/basis and SOURCE fields.
Addressed numeric events produce all four operator readings, original coordinate
means, choice counts and availability flags. Historical 480 frames yield 86,720
scalar readings: 248 conditional contrasts are observed, 232 missing, and only
142 frames support both within-condition dispersions. These are 24 enriched
historical authors in cue/raw/subset frames, not 480 independent new people.
No application labels, native answers or new model reads enter scoring.

Alternate direct conditional-group computation verifies source operators, all
fields, all three 199-reference families and every cold reading. Maximum errors:
operators 2.60e-18, fields 4.34e-18, cold 5.93e-20, original coordinates 7.63e-17.
The same agent performed this audit; it is not independent blind replication.

Canonical outputs: results/grain_math_response_v1/condition_shape_response_r1.
The post-outcome description has its own plan, manifest and prepared seal.
Both stay immutable; no rerun, eigentruncation, sign repair or metric rescue.

## Prior art and next scientific decision

[Hoff and Niu (2012), A Covariance Regression Model](https://www3.stat.sinica.edu.tw/sstest/oldpdf/A22n213.pdf)
already separates conditional means and covariance geometry; pp1-6 and
Sections 2.1-2.4 were consulted. Their parametric covariance model, Gaussian
likelihood and EM/MCMC estimators are not implemented in this packet.
[Fukumizu, Bach and Jordan (2004)](https://www.jmlr.org/papers/volume5/fukumizu04a/fukumizu04a.pdf),
Section 3.2, Equations 8-10, supplies prior conditional covariance-operator
theory. This finite two-condition contrast is not its inverse RKHS operator or
conditional-independence theorem.

Pinned OpenAI/math adc7f124, Signed-tensor-densities..., 03b-isotypic.tex
lines150-245, supplied design inspiration about specified operator actions,
support and complete component retention. Its positive-factor/density bounds
are NOT applied to indefinite T or transferred to personality measurement.

The useful theoretical distinction is: author-conditioned co-variation can
recur without reusable target orientation, while choice prevalence coupled to
displacement geometry has a separate SOURCE-target clue. Next inventory should
separate p modulation from B geometry on SAME supported evidence and connect
qualified target meaning/action interfaces where they actually exist. Do not
launch rank, head, weight, predictor, condition-granularity or higher-moment grids.
This is an empirical mechanism/construction increment using mature mathematics,
not a new covariance theorem or completion of text-based personality theory.
