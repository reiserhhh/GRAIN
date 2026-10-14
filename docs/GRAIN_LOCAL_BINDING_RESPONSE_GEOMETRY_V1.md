# Local Binding Response Geometry: HM10B3

## Objects and Units

This companion develops an already frozen experiment; it does not replace its
primary design. The materials are controlled expressions, not human trait
truth. Fix a receiver prompt, decoder block output, property-token positions
and a query q. Let X be the receiver's live property matrix and Y the same-word
donor matrix, under a different declared owner. Downstream computation with
this replacement defines a function F_q(X). Other upstream states remain those
of the receiver. F_q is not a psychological response parameter B_u.

The property matrix lies before the query in a causal decoder. Distinguish a
common prefix representation X from the target-conditioned map F_q. A query
can change the latter without changing prefix states. Numeric prefix checks
remain necessary because execution kernels depend on sequence shape.

## Two Context Directions and Their Complement

The source families alone define mean owner contrasts for affect and trust:

    D_l = [E_source(mean_tokens(Y_A-X_A)),
           E_source(mean_tokens(Y_T-X_T))]
    U_l = orthonormal_column_basis(D_l)
    Pi_l = U_l U_l^T

The dimension is at most two by construction, not because this proves binding
has only two dimensions. The Frobenius geometry here is the model's fixed
activation-coordinate geometry. Orthogonal changes of coordinates preserve
the projector decomposition. Arbitrary rescaling does not preserve it unless
the metric and dual maps are transported too.

For d=Y-X, write p=d Pi_l and c=d-p. Then:

    d = p+c, <p,c>_F = 0, ||d||_F^2 = ||p||_F^2+||c||_F^2.

These are exact vector identities. They do not imply that score effects add,
that c is nuisance, or that a high projection energy explains psychology.

## Finite Nonlinear Interaction

All three arms were fixed before calls, so their existing scalar outputs can
also expose a finite mixed difference without extra model calls:

    Delta_full = F_q(X+p+c)-F_q(X)
    Delta_p    = F_q(X+p)-F_q(X)
    Delta_c    = F_q(X+c)-F_q(X)
    I_q(p,c)   = Delta_full-Delta_p-Delta_c.

If the suffix map is C2 on this rectangle, the fundamental theorem of calculus
gives the exact conceptual identity

    I_q(p,c) = integral_0^1 integral_0^1
               D2 F_q(X+s p+t c)[p,c] ds dt.

Observed forwards use float32 interventions, so actual rounding is recorded;
the analytic identity describes the continuous map, not a bitwise equality of
finite-precision inference. A nonzero finite difference is evidence of joint
response at this intervention, not an intrinsic curvature tensor or a unique
mechanistic decomposition. Fixed prompt, layer and token positions are part
of the object. No tiny-denominator transfer ratio is needed.

The polarity contrast of each Delta or I removes a shared signed level effect
and tests owner-dependent high-minus-low action. It still includes the model's
lexical, positional and contextual response, not verified human disposition.

## Group Operations Versus Local Mechanisms

HM10B2's declared bit flips form a finite product group on material addresses.
Walsh characters split their response table into main and interaction terms.
This is a controlled-design decomposition, not independence of real authors.
OpenAI math's Fourier manuscript explicitly builds orthogonality under a
product law; its Boolean influence/hypercontractive bounds are not asserted
for our continuous neural score. The transferable reasoning is to distinguish
coordinate contrasts from joint supports before attributing an effect.

HM10B3 then asks whether a local intervention realizes part of that table's
owner action. An address-group identity alone cannot establish an activation
mechanism. Likewise a loop sum of scalar endpoint differences telescopes to
zero and is not evidence of flat geometry. Genuine local interaction above
requires independent forwards at the intermediate activation points.

## Separating Downstream Action from the Score Link

The zero-forward companion `scripts/diagnose_grain_math_span_readout.py` uses
the saved categorical logits at the four existing intervention corners.
This is a result-after diagnostic; it does not change any primary arm.
Write z00, z10, z01 and z11 for receiver, projection, complement and full
logits. Work in the zero-sum logit gauge, which removes only the exact
softmax equivalence z ~ z+c*1. Define:

    z_add = z10+z01-z00
    S(z) = sum_j v_j softmax(z)_j
    I_total = S(z11)-S(z10)-S(z01)+S(z00)
    I_pre_link = S(z11)-S(z_add)
    I_link = S(z_add)-S(z10)-S(z01)+S(z00)
    I_total = I_pre_link+I_link.

The pre-link term measures the action of actual logit nonadditivity against
the declared three-corner reference. It includes the recomputed downstream
network, final normalization and answer-head observation. It is not a
storage-only mediator or a unique allocation to a particular neural layer.
The link term is present even when the four logits are exactly additive.
The signed terms can cancel; do not report either as an information fraction.
Nonadditive logits can also change the answer distribution without changing
its ordinal mean. Preserve both centered logit norm and categorical KL.

For p(z)=softmax(z), the score gradient is exact:

    grad S(z)_j = p_j(z)*(v_j-S(z))
    S(z+d)-S(z) = integral_0^1 grad S(z+t*d) dot d dt.

The companion independently evaluates probabilities, recomputes the existing
four-corner helper, and numerically integrates both pre-link and link paths.
It also checks the KL/Bregman identity with A(z)=log(sum_j exp(z_j)):

    KL(p(z11) || p(z_add))
      = A(z_add)-A(z11)-grad A(z11) dot (z_add-z11).

These are standard finite categorical/exponential-family identities, not
new psychological laws. Wainwright and Jordan (2008), section 3.5.1 and
section 5.2.2, supply the partition/covariance and KL geometry. Their
mean-field algorithm is not used here. All corners share the exact receiver;
high/low owner contrasts are then formed within each material family.
Source, validation, application, both clause orders and all fixed layers
remain separate. Existing scalar values are never used to choose a new frame.

## Conditional Paths and Jointly Preserved Structure

Additional OpenAI math reading expands the question rather than supplying a
neural theorem. The pinned coordinate-sweep manuscript's introduction and
setup distinguish an unconditional marginal from a law conditioned on the
other objects' paths; it proves a prefix-measurable flow in its independent
coin shuffle. Our model has no such independent-coin process. The useful
design lesson is to retain order and the other event's context when asking
whether a binding action transports. Averaging all orders before inspecting
them could conceal conditional cancellation.

The pinned backward-intertwiner manuscript's introduction distinguishes an
invariant subspace of one operator from one preserved by an entire operator
family. Its infinite-dimensional shifts, Haar base and proposed theorem are
not properties of our finite neural activations. A source frame that acts for
one query need not be a common mechanism for all queries; each action and
its complement must be observed. These two ideas motivate conditional
multi-query action checks, not an imported claim of hyperinvariance or mixing.

## Literature Alignment

- Feng and Steinhardt (2024), sections 2-4, identify binding representations
  using causal interventions. Their storage-only analysis freezes downstream
  text activations. This experiment instead recomputes downstream states: a
  local total perturbation effect, not that specific mediation estimand.
- Dai et al. (2026), section 3, locate entity-relation binding through
  property-token states and PLS. Our two source mean contrasts use an explicit
  orthonormal frame, not their fitted indexed-cell construction. Whether the
  frame carries action is an experimental result, not an assumed theorem.

Primary sources:
[Feng and Steinhardt](https://proceedings.iclr.cc/paper_files/paper/2024/file/9d1b7fc578c0d2d6431fc26d736ecaf3-Paper-Conference.pdf),
[Dai et al.](https://arxiv.org/html/2604.19052v1),
[Wainwright and Jordan](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf),
[OpenAI math pinned collection](https://github.com/openai/math/tree/adc7f1241b42e322a6451854ab7e4b4c146bf78a).

Read scope: Fourier subsection in the cached influence manuscript; Feng
sections 2-4.2; Dai sections 1-3.3 and limitations. No claim of full appendix
review, imported code reproduction or all-manuscript proof verification.
New read scope: Wainwright/Jordan sections 3.5.1 and 5.2.2; coordinate-sweep
`00-introduction.tex` 1-220 and all 141 lines of `01-setup.tex`;
backward-intertwiner all 188 lines of `introduction.tex`.
No whole-paper or newly verified main-theorem claim.
