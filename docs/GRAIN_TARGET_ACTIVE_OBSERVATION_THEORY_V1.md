# Target-Active Observation Geometry

## What HM36 Adds

Natural source displacements and target-active source covectors are different
objects. In the fixed reader and addressed query layout, HM36 source covector
spaces contain97.75%-98.60% of cold covector squared norm while containing only
0.28%-0.51% of cold displacement squared norm. Matching random ranks do not
reproduce the action coverage. This is an empirical finite observation-family
transport result, not a new general theorem about personality or neural models.

The closest established method is gradient-based active-subspace reduction,
not unsupervised input PCA. Zahm et al. explicitly use an output metric and an
input measure in their gradient construction. Their Gaussian/Poincare bounds
are not assumed for this finite designed collection. HM36 retains the full
source-observed numerical span instead of optimizing a held-out spectral rank.
[Primary paper](https://arxiv.org/abs/1801.07922).

## Fixed Objects and Units

Let F(H,B,P,Q) be the actual fixed decoder readout after the zero-based
layer-index16 output hook. H is the four-token property span, B the preserved
other context/layout, P all earlier query tokens, and Q the final token state.
For the existing physical H perturbation deltaH define

    phi(H,B,P,Q;deltaH) = <d_H F,deltaH>
    v(H,B,P,Q;deltaH) = d_P phi.

The source collection v_s is a family of P covectors. In the declared original
Euclidean metric, their Riesz representatives form the columns of V. This
identification is metric-dependent; arbitrary coordinate changes must transport
the metric and dual action, not just rotate one feature matrix.

From only source observations construct thin orthonormal U_V and the numerical
projector Pi_V=U_V U_V^T. The raw source columns and physical norms remain
archived. Normalized columns select numerical rank only. This is not a
psychological-factor rotation or a claim that each retained basis vector is a
construct. Different targets retain their own ordered token spaces.

## Exact Action Split and Finite Observation

For each actual cold covector v and P displacement d,

    <v,d> = <U_V^T v,U_V^T d> + <v,(I-Pi_V)d>.

The first term is a pairing of target-reading coefficients and active input
coordinates. The second remains a signed residual, not automatically noise.
For the fixed gamma(t)=P1+t(P0-P1) path, the original half-contrast is

    [phi(P0)-phi(P1)]/2
      = (1/2) integral_0^1 <v(gamma(t)),P0-P1> dt.

HM35/HM36 use predeclared three-node Simpson and trapezoid approximations.
Their node/half-integral error is measured, not replaced by exact calculus.
The action RMS contributions need not sum: the squared-energy identity has
the signed cross term 2 E[a_supported*a_missing]. Even a larger geometric
space need not monotonically reduce error on a particular signed action.

## Shared Carrier Versus Reading Law

Write the observed action as

    <v,d> = a(H,B,P,Q;deltaH)^T z(d) + remainder,
    a=U_V^T v, z=U_V^T d.

HM36 establishes a useful source-learned carrier U_V in this finite design.
It evaluates a using actual cold derivatives; it does not yet compile a.
HM35 separately shows that holding reading fixed along P is inadequate.
The next mechanism is therefore a source-only conditional law for a, not a
larger unconditional feature matrix. A successful law would remove the need
for new upper-decoder derivatives on these held conditions; the existing
upstream neural representation remains distinct from decoder-free arithmetic.

## Numerical Rank Is Part of the Observation Definition

V ranks are87/88/86 and DV ranks94/96/94 (negative affect, low mood, AI trust).
Gram eigenvalue cutoff is100*float64eps*column_count*largest_eigenvalue.
Because normalization/cutoff changes with the declared column family, the
numerical DV space is not asserted to contain every truncated V direction
exactly. Thus tiny covector-norm nonmonotonicity is not a new cognitive effect.
Both untruncated spans and the numerical truncation definition must be
distinguished. There is no application-selected rank, alpha or kernel.

Primary Gram-vector construction failed1e-8 geometric verification despite
accurate actions. Separate r2 keeps every learned rank but constructs bases
with QR/SVD and verifies with direct tall SVD. Original failure and all values
remain archived. It does not become accepted by rounding or loosening limits.

## Measurement Link

This observation law belongs to the text-evidence -> vector action part of
GRAIN. Event identity, actor eligibility, conditions, source units and author
aggregation are still necessary parts of the measurement theory. A large
reader sensitivity is not an author's trait score, and no context/choice/state
component is discarded merely because it lies outside the observed carrier.
