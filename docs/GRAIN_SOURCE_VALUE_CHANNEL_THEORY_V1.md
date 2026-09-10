# Content and selection are two derivative routes through one local function

2026-09-11. NEXT76 / pass65. Finite sums, chain rule and source least squares;
not a new general differentiation theorem or evidence of human mental causality.

## 1. The integrated function

For fixed query anchor q_S, support, source maps and readout g:

    ell(H)=ell_identity(H;q_S)+Phi_selector(H) B_Sq
    a=softmax(ell+mask)
    Vhat(H)=V_identity(H)+Phi_value(H) C_Sq
    v_mh,t(H)=g_mh^T Vhat_t,kv(h)(H)
    F_qm(H;g)=sum_h,t a_h,t(H) v_mh,t(H).

Selector logits retain their softmax gauge; value features do not have that
per-sequence gauge. A constant logit shift leaves F unchanged. A constant value
shift changes F, although its derivative contribution is zero with fixed g.
Thus the two constant terms cannot be treated with the same centering rule.

The actual derivative along aligned GRAIN direction D_f is

    dF[D_f] = sum_h Cov_(a_h)(v_mh,dell_h[D_f])
               + sum_h E_(a_h)[dv_mh[D_f]]
             = K_f + C_f.

Proof: differentiate the finite sum. For softmax,
da_t=a_t(dell_t-E_a[dell]); insert this and the ordinary product rule. Fixed
masked tokens have zero weight and no support-boundary derivative. This identity
is now checked by64 finite differences of the implemented scalar, not separate
unrelated regressions for values and tangents.

The two routes match the distribution-change/direct-function distinction in
[Schulman et al.,2015, Theorem1](https://arxiv.org/pdf/1506.05254). Their paper
handles stochastic graphs and estimators; ours is an exact finite softmax sum,
without Monte Carlo sampling or a claim that attention is a mental probability.
The borrowed distinction concerns computational dependence, not personality.

## 2. An appropriate error metric depends on the observable

Hold a, dell and g fixed and compare Vhat with native V. With e=Vhat-V and
de=D Vhat-D V, each projected value error e_h=g_h^T e_kv(h) obeys

    error_F = sum_h E_(a_h)[e_h]
    error_Kf = sum_h E_(a_h)[(dell_hf-E_(a_h)[dell_hf]) e_h]
    error_Cf = sum_h E_(a_h)[de_hf]
    error_total = error_Kf + error_Cf.

Collect these linear functionals in M_(a,dell,g). The observable loss is
||M E||^2=E^T M^T M E, whereas raw vector loss uses E^T E. Lowering the latter
does not order the former. A Euclidean improvement may leave a highly weighted
component worse. This is a mathematical explanation of how NEXT76's11% V-MSE
improvement and worsening K+V response error can coexist, not an empirical claim
that one specific hidden coordinate has already been identified as responsible.

Likewise E[(K+C)^2]=E[K^2]+E[C^2]+2E[KC]. The observed positive cross term for
the learned value channel and cancellation in the source-Q/native-KV comparator
must remain in the account. Dropping either term or declaring the cross term
noise would change the measurement. No scalar shrinkage was fitted to new8.

## 3. Source construction and retained dependencies

C_Sq minimizes the normalized source residual V and directional dV losses, with
one alpha=.01 ridge penalty and unpenalized explicit constant. Sources alone
define the earlier response basis, feature scales and query anchor. Ten fits
cover source4LOO and new8/all4. H16 -> selector/value -> scalar/derivative can
execute numerically without current native K/Q/V, current target scores or a
current suffix execution. H16 itself and fixed g remain neural-derived inputs.

The totals are K+V at fixed Q/g. They do not include DQ, MLP, residual-identity
or later-layer changes, so are not the derivative of the complete original L.
This restriction is explicit in both target reconstruction and table schemas.
Native K/V partials independently replay old NEXT49 computations to1e-15.

## 4. Reconnect to the actual query endpoint

NEXT51 already compiled source-only later readers rbar_q; it did not yet have
the current numerical selector/value functions. Reuse that result rather than
inventing another nominally new average-reader theorem. Define

    g_source_q,h = (rbar_q^T W_O)_h
    F_source_q(H) = rbar_q^T W_O concat_h E_(ahat_qh)[Vhat_kv(h)]
    dF_source_q[D_f] = rbar_q^T W_O concat_h (K_vector_f+C_vector_f).

This composes source-built parts into an endpoint-directed *component* without
a material-specific current readout. It is not full L; no native remainder may
be added back and called an independent prediction. The next card freezes the
existing candidates, applies the already compiled source reader and evaluates
the matching native endpoint component. It changes a real input dependency and
returns to the personality-query endpoint, instead of a V-accuracy tuning loop.

See [NEXT51 theory](GRAIN_SELECTION_ENDPOINT_THEORY_V1.md),
[experiment](../reports/grain_vector_mechanisms/SOURCE_VALUE_CHANNEL_REPORT_V1.md),
[NEXT77 plan](GRAIN_SOURCE_ENDPOINT_ASSEMBLY_PLAN_V1.md).
