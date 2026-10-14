# Response geometry beyond scalar positions

HM103, 2026-10-10. A selected, independently computable response construction.
Do not identify this object with a natural neural controller or human dynamics.

## 1. Object and declared units

For fixed source queries q and covectors d, the frozen carrier is x=(K,V).
The six observed functions are

    p_qha = exp(q_qh . k_a,g(h)) / sum_b exp(q_qh . k_b,g(h))
    F_q(x) = sum_h sum_a p_qha (d_qh . v_a,g(h)).

They are actual attention functions of the HM101/HM102 numeric carriers,
not six psychological ground-truth scores. One score position F(x) discards
the internal fiber F^-1(F(x)). No global smooth-bundle assertion is required:
on any neighborhood with rank(dF)=6 the regular-level-set theorem supplies
local fibers and the tangent split below.

Choose original Euclidean K/V coordinates, not an unannounced whitening.
J=dF, G=JJ^T and U=J^T G^-1 define the minimum-norm solution to J u=b.
This inverse requires all six directions to be identifiable. A dropped-rank
pseudoinverse does not implement the same declared measurement/control law.

## 2. Why output coordinates can commute while controls do not

The i-th column U_i satisfies J U_i=e_i. Under its flow, F changes only in
coordinate i, at unit speed. Let the bracket convention be

    C_ij = D U_j[U_i] - D U_i[U_j].

Differentiating J U_j=e_j along U_i gives

    dJ[U_i] U_j + J D U_j[U_i] = 0.

Subtract the equation with i/j exchanged. The Hessian of each F_q is
symmetric, hence dJ[U_i] U_j=dJ[U_j] U_i and J C_ij=0.
Thus curvature lies in the score fiber: it is not an additional first-order
score direction. This is a standard connection/commutator identity.

For the ordered rectangle +i,+j,-i,-j, locally

    x_end-x_start = epsilon^2 C_ij(x_start) + O(epsilon^3).

Each F_q returns exactly in exact flow, regardless of C. Numerical RK4 closure
checks the implementation. Reverse the ORDER to reverse the leading bracket;
negating both axes keeps the rectangle orientation. A scalar exact
differential still has zero integral on a closed STATE path. Here the score
path is closed while the state path need not be; no contradiction or quantum
measurement claim is involved.

## 3. A latent difference becomes observable through another functional

For any additional fixed functional R,

    R(x_end)-R(x_start) = epsilon^2 dR[C_ij] + O(epsilon^3).

The finite experiment uses a seventh source-mean query/covector, fixed before
outcomes. It is not a trait. Nonzero dR[C_ij] is a constructive, falsifiable
way to show that identical six-score positions need not exhaust the carrier's
later observable behavior. No target-label regression is needed.

For an explicit author evidence law P_u, define

    K_u,ij,R = E_(e~P_u) dR_e[C_ij(x_e)].

This scalar has declared response/metric units and can be accumulated by the
HM102 expectation interface. It is distinct from F's average level and from
mass-weighted merging. The present P+/- are synthetic equal-margin accounts,
not measured human authors. Current-score identity alone does not determine
K; whether it is psychologically meaningful requires subsequent evidence.

## 4. Why the linear control matters

Holding K fixed makes F linear in V. Its Jacobian and minimum-norm controller
are constant, so all brackets vanish and every rectangle closes internally.
Joint K/V permits the response field to turn as key selection and content
change together. A nonzero joint bracket is therefore a coupled response
property, not inevitable merely because the hidden dimension exceeds six.

Every common translation of all keys in a KV head leaves all local normalized
functions, Jacobians and brackets unchanged. Horizontal key motions sum to
zero over tokens, so their loops do not merely drift along the common-key
translation ambiguity constructed in HM102. Probe/partition/metric objects
remain separate; no blanket deletion of context or mass is justified.

## 5. Scope and research consequence

The controls are deliberately constructed. Curvature depends on the metric,
chosen targets and allowed K/V intervention coordinates. A different vertical
component added to U can change it. It is not a uniquely recovered LLM path,
and perturbed K/V need not correspond to a possible natural text encoding.
It measures the local response geometry of the declared observer.

This addresses a mechanism gap rather than predictive precision: a dynamic
object can be defined by multi-target response operations instead of naming
an unstable PCA axis. The subsequent problem is which of this geometry can
be reached by text-supported changes and which is shared across carriers,
contexts and target families. Do not optimize a classifier to answer it.

The minimum-norm inverse/nonrepeatability idea has established precedents in
robotics (Roberts/Maciejewski 1991; Shamir/Yomdin 1988). OpenAI/math's metric/
unitary-connection introduction inspired separating observations, internal
fibers and transport. Its inverse PDE theorem is neither required nor applied.
HM103 is a GRAIN-specific construction/test, not a new geometry theorem.

Execution specification: GRAIN_RESPONSE_CONNECTION_PLAN_HM103.md. Numerical
results, including failures and small effects, belong to the separate report.
