# Encoded sectors and functional observation operations

2026-10-09. Theory continuation from HM73 and the frozen HM74 experiment.
HM74 collection, primary analysis, alternate verification and result-after
consequence accounts are complete. The verdict below comes from actual fixed
interventions, not code or successful unit tests. It is not an accuracy leaderboard.

## 1. Three separate objects

For one fixed query q and its eight designed evidence conditions z=(x,y,g),
let B_z be the retained body states at a cut and S_z the complete query suffix.
The remaining decoder is a context-indexed function R_q:

    F_q(z) = R_q(B_z,S_z).

The finite coefficients of S, coefficients of F and a functional intervention
on S are three distinct objects. A large coefficient in S need not have a large
effect on F. A small coefficient can matter strongly. An output interaction can
arise even when that character is absent from S, because R is nonlinear or B
provides a second route. This is why a better vector cosine is not a mechanism.

## 2. Character projection does not generally commute with reading

Define P_J on finite functions of z using orthogonal characters:

    c_J[S] = E_uniform[chi_J(z) S_z]
    (P_J S)_z = chi_J(z) c_J[S].

At fixed B, an affine reader commutes with the nonconstant projection after
its affine offset is removed. For a general nonlinear context-indexed reader,
there is no such equality. The elementary counterexample S=(y,g) and R(S)=y*g
has P_yg S=0 but P_yg R(S)=y*g. This is established mathematics, not a newly
discovered neural theorem. It supplies a falsifiable mechanism distinction.

HM74 measures the actual family of operations

    F_q^{-J}(z)=R_q(B_z,S_z-(P_J S)_z)
    A_{q,K;J}=c_K[F_q-F_q^{-J}].

This is an observation-action matrix: removed encoded sector J versus affected
output sector K. In the present design J is yg,xyg,xy or their fixed combination;
all output K are retained. Off-diagonal changes and changed main effects are
scientifically meaningful, not silently classed as noise.

## 3. Exact operation formulas, not fitted reproduction

For a differentiable readout and a fixed z, write d_J=(P_J S)_z. Then

    F_q(z)-F_q^{-J}(z)
      = integral_0^1 D_S R_q(B_z,S_z-t*d_J)[d_J] dt.

For two removed directions a=d_yg and b=d_xyg, the joint intervention term is

    I_q(z)=R(B,S)-R(B,S-a)-R(B,S-b)+R(B,S-a-b)
      = integral_0^1 integral_0^1
          D_S^2 R(B,S-s*a-t*b)[a,b] ds dt.

These are ordinary fundamental-theorem-of-calculus identities. HM74 measures
the finite four-arm left side, not the derivative integral or its causal path
inside individual heads. Do not infer a unique Hessian mechanism from one
finite rectangle. Body, query positions and decoder remain part of the object.
Signed interaction is retained; summing individual ablation losses can misstate
the joint loss because their effects need not add.

## 4. From a common observer to evidence distributions

For a synthetic author evidence distribution P_u over this designed support,

    Theta_{u,q}=sum_z P_u(z) F_q(z)
    Delta_{u,q;J}=sum_z P_u(z) (F_q(z)-F_q^{-J}(z)).

The first object is the person's evidence composition under a common observer;
the second is its change under a declared observation operation. This puts
the experiment back into GRAIN's individual-versus-group measurement logic.
It does not make F a personal response parameter, P an opportunity-corrected
preference, or a synthetic profile a human psychological score.

HM74's parity profiles have known probabilities, rather than probabilities
learned from patched outcomes. They isolate exactly how a removed operation
changes a measurement contrast between compositions sharing lower marginals.
This is the finite bridge to author measurement, not a demand that the reader
produce a higher accuracy or that every query have the same strength.

## 5. Useful outcomes and next decisions

Selective attenuation relative to focal-query, generic-xy, random and neutral
controls motivates a portable event-conditioned operation, with qualification
kept explicit. Generic attenuation suggests a shared evaluation/readout action.
Preserved output interaction despite removed state interaction motivates
testing retained-body/upper-reader recomputation or redundancy. None motivates
an automatic stronger patch, another chosen layer or a regression accuracy grid.
An observed mixture motivates a conditional operator, not deleting shared
information or forcing six independent axes.

This uses [entity binding](https://arxiv.org/html/2310.17191v1) and
[activation interventions](https://arxiv.org/html/2404.15255v1) as established
tools. OpenAI/math's fixed finite observation-family and scalar/matrix distinction
is a design analogy; no elliptic uniqueness theorem has been transferred.
The novel question is which event-qualified operation gives stable target
consequences and can be separated from generic judgment in the GRAIN chain.
Its actual answer must come from recorded intervention results, not terminology.

## 6. Actual HM74 finite result and the measurement bridge

All 4224 declared calls completed (528 native, 3696 patches), zero retries.
For 30 predeclared paired queries the marginal matched ownership action falls
from .137805 to .024223 when removing the yg suffix sector. The matched-minus-
focal contrast falls from .104780 to .017024; its RMS decreases 72.33%, with
29/30 absolute contrasts smaller. Ordinary xy and norm-matched random controls
do not have the same reduction. The exact-owner suffix donor reverses the mean
contrast to -.098314. Neutral and off-target actions remain, and anthropomorphism
is an explicit exception to positive target selectivity. This is mixed functional
participation, not a pure psychological gate or a global six-query causal map.

Removing the xyg suffix sector leaves 91.39% of the matched score xyg coefficient
RMS, 70.97% of its log-odds RMS and 75.11% of centered-logit RMS. Removing both
yg and xyg leaves 51.29% of the centered-logit xyg RMS. Therefore the explicit
suffix xyg term is not the sole carrier of this output interaction. Retained body,
other suffix sectors, downstream recomputation and redundancy are not separated
by this intervention; do not label the residual pure noise or uniquely identify
one of these routes. These are within-unit finite effect ratios, not percentages
of personality information. This does not justify a stronger patch or fidelity grid.

The author-relative measurement interface, under a declared common observer and
fixed reference evidence distribution, is exactly

    m_u,K = E_Pu[chi_K]
    Theta_u,q - Theta_ref,q = sum_K (m_u,K - m_ref,K) c_K[F_q]
    Delta(Theta_u,q - Theta_ref,q; J)
        = sum_K (m_u,K - m_ref,K) A_q,K;J.

Here Delta is native minus declared-ablated observation. This finite identity
keeps individual evidence organization and the common observation operation
separate. In the frozen parity populations the yg contrast RMS changes from
.038756 to .008373 while the xyg contrast changes from .009103 to .008320 under
its corresponding single-sector removal. All 3168 profile records, probability
vectors, claimed matching lower marginals and score consequences replay exactly.
The theory increment is an executable evidence-composition/action interface;
orthogonal characters and expectation identities are established mathematics.

Canonical root: results/grain_math_response_v1/causal_ownership_sectors_r1.
Primary verification and separate 04_interpretation are preserved. The operation
requires the designed counterfactual cube and is not standalone natural-text
scoring. Human validity, individual B_u and original 18/100/44 scope remain open.
