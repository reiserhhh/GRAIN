# Source endpoint composition and an observable-specific error geometry

2026-09-11. NEXT77 / pass66. Ordinary chain rule and bilinear algebra applied
to the actual same-material GRAIN/query interface. The combination is our local
implementation; the mathematical identities are not claimed as new theorems.

## 1. A complete function for a declared component

Let H be cached pre-block16 states on the exact rendered material. The source
reference S fixes qbar, selector coefficients B, value coefficients C and rbar.
For each attention head h:

    ell_qh(H) = ell_identity_qh(H) + Phi_s(H) B_qh
    a_qh = softmax(ell_qh+mask)
    Vhat_q(H) = V_identity(H) + Phi_v(H) C_q
    p_q(H) = O concat_h E_a[Vhat_kv(h)]
    Fhat_q(H) = rbar_q^T p_q(H).

The same fixed source reader is used on every new material. With direction D_f
aligned to a frozen GRAIN factor and the same token addresses:

    u_qf = Dp_q[D_f]
         = O concat_h {Cov_a(Vhat,dell_qh[D_f]) + E_a[DVhat[D_f]]}
    D Fhat_q[D_f] = rbar_q^T u_qf.

Both terms are vector-valued before projection, preserving information that a
single head-projected action would discard. The content and selection terms
are derivatives of one scalar, not independently chosen direction regressions.
Numerically this agrees with projecting rbar through O into g and applying the
previous grouped-head formula. This agreement and central finite differences
are separately tested.

Q and rbar are fixed during differentiation. No claim is made that rbar is the
gradient of a complete nonlinear endpoint at every H. The formula measures one
K/V component; it omits Q variation, residual identity and other layer changes.
No native remainder restores these omissions in application. Cached H16 still
requires the original encoder or a separately validated alternative.

## 2. Why separately better components can make a worse composition

Compare native vector u/current reader r and candidate uhat/source reader rbar:

    rbar^T uhat - r^T u
      = r^T(uhat-u) + (rbar-r)^T uhat.

Squaring introduces a signed cross term. Two squared component errors decreasing
does not order the squared sum. NEXT77 shows that exact pattern on the frozen
new8 response-selector contrast: .028774->.028029 transport MSE and
.046370->.046264 reader MSE, but cross .030924->.034696 and total
.106069->.108989. These changes were not fitted or selected on new8.

The identity chooses native r for the transport term and candidate uhat for the
reader term. Another bilinear expansion gives another allocation. Therefore the
cross-account identifies numerical interaction, not uniquely assigned causal
percentages, psychological traits or a count of recovered mechanisms.

This is also the observable-norm distinction from NEXT76: the same vector error
E is measured by ||E||^2 in hidden space and by E^T rr^T E for a fixed endpoint.
Query-specific and query-mean readers supply different seminorms and different
targets. Their smaller errors cannot be ranked as a single accuracy measure.

## 3. Two ways that the same text differentiates personality queries

The result raises a more informative question than another average-reader fit.
For two queries N/C, define r+=(r_N+r_C)/2, dr=r_N-r_C, p+=(p_N+p_C)/2,
dp=p_N-p_C. Bilinearity gives the exact symmetric identity

    F_N-F_C = r+^T dp + dr^T p+.

The first term is different represented/selected content under a common reader.
The second is a different reader acting on common content. Both can carry query
contrast; one need not locate disjoint tokens for every personality word.
For fixed source readers the same identity holds for compatible directional
vectors u and derivatives. For native state-dependent readers, it is only an
identity between frozen local components, not the full derivative of changing r.

NEXT77 already shows strong mode differences, but has not attributed the whole
contrast to these terms. NEXT78 is a bounded fixed-array test of that question,
with signed covariances and source/new8 separately. Its purpose is to decide
whether to develop query-differential reading or upstream evidence geometry,
not to proclaim that a two-query decomposition establishes all personality axes.

## 4. Connection to interpretation

GRAIN describes a material's position/direction relative to a reference. This
query function describes how moving along that same-material direction changes
one specified model observable. The usable interface is the signed differential
D F_q[D_f], plus the unchanged GRAIN score, evidence address and query definition.
Relational/interaction coding is another possible observable, not a reason to
rename the direction before interpreting or externally relating it.

See [report](../reports/grain_vector_mechanisms/SOURCE_ENDPOINT_ASSEMBLY_REPORT_V1.md),
[previous derivative theory](GRAIN_SOURCE_VALUE_CHANNEL_THEORY_V1.md),
[NEXT51 endpoint](GRAIN_SELECTION_ENDPOINT_THEORY_V1.md),
[next card](GRAIN_ENDPOINT_QUERY_CONTRAST_PLAN_V1.md).
