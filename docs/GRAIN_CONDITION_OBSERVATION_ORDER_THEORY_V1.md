# Condition accounting and observation order

2026-10-10 JST. HM83. This note specifies a measurement-architecture consequence
of established projection theory. It does not claim a new two-projection theorem,
a universal personality decoder, or transfer of OpenAI's amenability result.

## Objects and estimands

Keep the distinction from HM82:

    P_u: distribution of an author's qualified reported events
    T_q: common query-conditioned observer
    B_u: personal response to a real condition (a separate target)
    f_uv = (P_u-P_v)/R: reference-relative author-law contrast
    theta_t(u)-theta_t(v) = <f_uv,t>_R

HM83 observes a flat source lexical coordinate4 plus six local Qwen readings,
not the historical complete GRAIN hierarchy. Its source targets t are the
controlled material codes c,x,cx,xc*actor. The companion-event polarity c is
known design metadata. It is not a causal environmental intervention.

## Two operations that should not be conflated

In H=L2(R), let V=span(1,B) and P be its orthogonal projection. Let
E:h -> E_R[h|c] be the condition-accounting projection. Both choice E f and
within-condition (I-E) f are retained; nothing is deleted as nuisance.

Let M:h -> c*h, with c in {-1,+1}. M is a self-adjoint unitary involution.
E is generally NOT M; E averages within a cell, M changes signs by cell.

For a mean-zero/unit-norm c under a balanced reference, put v=||Pc||_R^2.
Since P preserves constants and is self-adjoint,

    E Pc = <Pc,c>_R*c = <Pc,Pc>_R*c = v*c
    ||(I-E)Pc||_R^2 = ||Pc||_R^2-||EPc||_R^2 = v-v^2.

This is an exact accounting identity. A PURE condition-choice contrast f=c,
after reconstruction in an incomplete observation span, can acquire a component
in the within-condition subspace. That is a representation/accounting artifact,
not evidence that this person truly has within-condition variation. Its absolute
energy v(1-v) need not decrease as visibility v improves. In particular a better
observer alone is not a guarantee of a better construct assignment.

The statement concerns a projected contrast FIELD, not a new measured effect
in real authors. It does not say every scalar score is biased by v(1-v), or that
context is psychological noise. It changes when/how we may name score components.

## Anchor and conditional products solve different problems

If V includes 1 and c, range(E)=span(1,c) is a subspace of V. For orthogonal
projections onto nested subspaces, EP=PE=E. The condition account then commutes
with observation projection. The explicit anchor is sufficient for this property.

However V=span(1,c,B) need not contain c*B. Therefore P need not commute with M.
For V*=span(1,c,B,c*B), multiplication by c sends each generator into V*;
it also preserves V*'s orthogonal complement, so PM=MP. For targets x and cx,
this gives ||Px||_R=||Pcx||_R under the fixed balanced source design.

Thus:

    anchor: prevent choice/within mixing under source condition accounting
    conditional products: preserve the function space under condition gating

These are structural requirements, not two successive rounds of hyperparameter
optimization. Having the first does not imply having the second. Additional
coordinates have finite-sample and transportation costs, so no global winner
is selected on the strength of either algebraic property.

## From a source table to an executable function

Let normalized centered B give weighted A=sqrt(R)*(B-center)/scale=U*S*Vt.
The fixed machine-rank source calibration compiles each source target t as

    beta_norm = V*S^-1*Ut*sqrt(R)*t
    beta_raw = beta_norm/scale
    offset = E_R[t]-center*beta_raw
    s_t(e_new) = B(e_new)*beta_raw+offset.

This is ordinary least-squares function calibration with a declared source
law and target. Source replay equals projection to numerical precision. Unlike
HM82's vocabulary lookup, s_t can be executed on new observation values. This
algebra does NOT guarantee that a semantically new text lies on the same evidence
law. The scorer does not read new target codes or a new counterfactual table.

The lexical and query units are frozen; application batches cannot change the
normalization. Unknown c cannot be replaced by zero: conditional arms return
UNOBSERVED_CONDITION, while unconditioned arms remain executable. A missing
condition is not a zero psychological response.

The numeric function stage runs without a language model, but its six input
query readings still require the frozen local Qwen. This is NOT achievement
of the goal of replacing LLM personality inference with an independent vector
measurement method. Public design-code response is not questionnaire validity.

## Prior art and what this packet adds

- [Ay et al., Theorem3.11/Remark3.13](https://arxiv.org/html/1207.6736v3): sufficient
  statistics and information monotonicity. Used to keep targets and lost contrast
  explicit, not to label every finite moment fraction Fisher information.
- [Raginsky, Eq2.5 and Theorem3.2 proof](https://arxiv.org/html/1411.3575): conditional
  variance and adjoint channel operators. A feature span is weaker than the full
  sigma-algebra generated by the feature vector; this distinction is retained.
- [Luo, Moslehian and Xu, introduction](https://arxiv.org/pdf/1904.10742): recalls
  established Halmos two-projection geometry. We only use the finite real Hilbert
  special case, not their novel Hilbert C*-module extension. The original Halmos
  publisher PDF returned403 during this reading; no claim of reading that full PDF.
- OpenAI/math, pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a, Strong-Ulam-
  Stability introduction, lines84-160: comparisons using different inner products
  need a separate composition-defect account. This is a design heuristic, not a
  theorem about language/personality or a guarantee that better precision helps.

F16 already demonstrated condition-dependent calibration mixing level/response.
Older source-misclassification experiments already rejected a global epsilon
fix. This packet does not rediscover those general facts. Its addition is an
explicit E-versus-M criterion and measured leakage of the actual frozen HM81/82
observation span, followed by a source-calibrated function on unseen bodies.

Canonical result directory:
`results/grain_math_response_v1/condition_observation_order_r1`.
The source algebra and cold transport tables must be read separately. Final
numeric acceptance is recorded in verification.json, not inferred from this note.
Original18 protocols/100 cards/44 computational questions remain open.

## Executed consequences (read source and new-material accounts separately)

Source choice-to-within energy is .138663551 for marginal10 and .223936852 for
joint49. Explicit c removes it to roundoff, but anchor_joint50 still has
||[M,P]||F=8.378821988. Conditional_joint99 removes both E and M commutators.
The source conditional-target visibility is .377868844 with the anchor versus
.906828850 with conditional products. These are different structural benefits.

All120 new public bodies/720 reads completed in175.405 model seconds, no retry.
New-text mean signed x action is .497426/.442389/.475915/.444346 across the four
fixed arms. Conditional99's Extraversion-material action is +1.126407 (5/5 pairs)
but Neuroticism-material action is -.280415 (1/5); pooled source calibration
does not make a universal high/low psychological axis. New conditional99 x and
cx RMSEs coincide because s_cx=c*s_x is a constructive identity, error2.20e-14,
not two independent transport successes. All poor values remain in the tables.

Primary12260 alternate checks max2.5091e-14; terminal1984 checks max4.441e-16;
29 focused tests/Ruff/format passed. Optional terminal reporting's int64 JSON
failure is retained in04; a zero-call output-type repair produced sibling04_r2.
The scientific primary was not modified. No universal/high accuracy promotion.
