# Author contrasts and observation laws

2026-10-10 JST. HM82 finite construction; established mathematics applied to
the same-event GRAIN/query interface. This is not a new information-geometric
theorem, a completed personality instrument, or an author-specific causal model.

## 1. Distinguish the objects

An event e is a qualified reported account, including its condition/role
address. P_u is an author's distribution of such accounts. A common reader
T maps accounts into observations. An individual response B_u would describe
how that person reacts to changes in a real condition. P_u, T and B_u are
different objects; recurrent joint moments do not by themselves identify B_u.

For a strictly positive declared reference R on a finite event vocabulary,
define the relative contrast between two author laws:

    f_uv(e) = [P_u(e) - P_v(e)] / R(e).
    <f,h>_R = sum_e R(e) f(e) h(e).

The field has zero reference mean. Its squared norm is a reference-relative
distribution contrast, not a psychological variance or Pearson correlation.
Changing R changes the geometry; HM82 uses uniform mass on 480 designed accounts,
not an estimated natural population.

## 2. Conditions are retained, not subtracted as noise

For a declared condition c(e):

    f_choice = E_R[f | c]
    f_within = f - f_choice
    ||f||_R^2 = ||f_choice||_R^2 + ||f_within||_R^2.

The proof is conditional expectation orthogonality. Choice is a difference in
reported-condition mass. The within part can include conditional expression,
role associations and other differences; it is not automatically a response
parameter or a personality trait. Neither part is discarded.

HM82's c is reported companion-event polarity, not a randomized environmental
stimulus. Moving to real topic/situation opportunities requires separately
declaring encounter availability, author selection and evidence eligibility.

## 3. A shared observation channel loses specific contrasts

Let K(y|e) be a known common Markov channel. With joint reference R(e)K(y|e),

    R_out(y) = sum_e R(e)K(y|e)
    f_out(y) = E_R[f(E) | Y=y].

Then the total-variance identity gives

    ||f||_R^2 = ||f_out||_Rout^2
                 + E_RK[(f(E) - f_out(Y))^2].

An invertible re-addressing retains the field. An observation that forgets a
condition can retain expression level while erasing a condition-expression
interaction. This is a joint-space identity; a stochastic channel is not an
idempotent orthogonal projection on the original event space.

For local parametric perturbations P_theta=R(1+theta*f), the Fisher information
at theta=0 is ||f||_R^2 and its observed value is ||f_out||_Rout^2. This connects
the finite contrast account to standard information geometry. It does not make
the actual GRAIN reader a known invertible channel.

## 4. A moment dictionary is weaker than the full event observation

Let B(e) contain registered observation functions, and let V_B be the centered
span of its columns in L2(R). Pi_B is the weighted orthogonal projection.
For any centered declared target function t:

    <f,t>_R = <Pi_B f, Pi_B t>_R
               + <(I-Pi_B)f, (I-Pi_B)t>_R.

Compile t_visible=Pi_B t using source observations and source target definitions.
Application scores average t_visible over actual sampled events. The target
remainder remains an explicitly unobserved contribution; setting it to zero
is an assumption, not a noise-removal theorem.

For unrestricted probability laws on this finite full-support vocabulary, the
target contrast is identified by the dictionary moments exactly when t lies
in V_B (up to a constant). To see the converse, let r=t-Pi_B t be nonzero.
P_+=R(1+epsilon*r/||r||_infty) and P_-=R(1-epsilon*r/||r||_infty), with
0<epsilon<1, have identical B moments but differ on t by
2*epsilon*||r||_R^2/||r||_infty. This is a standard annihilator argument, not
an additional HM82 acquisition or a first discovery of score fibers.

Full per-event vectors may contain information that these moments omit. The
finite table compilation is also not a predictor for unrestricted unseen text:
the cold scorer receives addresses in the already registered vocabulary.

## 5. Why condition-indexed dictionaries change the object

With binary c in {-1,+1}, use the function space

    V_tilde = span(1, c, B, c*B).

Multiplication M_c:h -> c*h is a unitary involution in L2(R), and V_tilde is
invariant under it. Therefore its orthogonal projector commutes with M_c.
For the balanced HM82 targets x and c*x, both are orthogonal to 1 and c;
their projected squared norms are equal. The executed values are .681451649
for the conditional marginal dictionary and .906828850 for conditional joint.
This symmetry explains those equalities without searching parameters.

This is a standard invariant-subspace argument and a design rule: preserve
the condition-indexed observation functions needed by the contrast. It is
not evidence that the words "condition" or "personality" have been discovered
inside an embedding. The c column was supplied by controlled material design.

## 6. Per-event invariance is not moment invariance

An invertible recoding q -> q^3 loses no individual q information. However,
reusing a fixed low-order moment recipe after this recoding generally changes
V_B, and thus the target information it retains. Reconstructing the original
q before applying the original recipe restores that recipe exactly.

HM82's conditional-polarity retained norm is .376810136 for joint49,
.297991262 for recoded_joint49, and .376810136 again after the exact inverse.
This is a change in the fixed summary, not an empirical loss of full-vector
information, a language-transport result, or a failed reader.

## 7. Information and finite-sample precision are separate

Source visibility uses exact functions on the declared vocabulary. Cold
application scores use 128 iid events per simulated author. Larger visible
functions can increase sampling variance while reducing omitted contrast.
Neither quantity alone is a personality-accuracy measure. Natural temporal
dependence would require a different sampling error model.

## 8. Executed artifacts and next use

Canonical: results/grain_math_response_v1/author_contrast_observability_r1.
480 public source bodies/2880 existing Qwen scores, eight fixed dictionaries,
four declared design targets, 120 new simulated laws/15360 sampled events.
No new lexical/neural fit or forward. Eight source span calibrations were
computed; do not call this "no estimation of any kind".

Theoretical criterion for the next mainline packet: specify the desired author
contrast and its observation law first. Modify a reading tool only if that
modification separates competing mechanisms or restores a declared contrast.
Do not infer B_u from a shared reader's nonlinearity or optimize teacher fidelity.
Natural evidence qualification, historical GRAIN hierarchy integration and
psychological/behavioral target calibration remain distinct tasks.

## Sources and limits of transfer

- Ay, Jost, Le, Schwachhofer, *Information geometry and sufficient statistics*,
  [primary paper](https://arxiv.org/abs/1207.6736). Theorem3.11/Remark3.13 provide
  information-monotonicity/sufficiency context; the finite identity above is
  derived directly, not a new infinite-dimensional proof.
- Raginsky, *Strong data processing inequalities and Phi-Sobolev inequalities
  for discrete channels*, [primary paper](https://arxiv.org/abs/1411.3575).
  Eq2.5 and Theorem3.2/proof connect conditional expectation, variance and
  contraction. No channel constant for natural GRAIN data is estimated here.
- [OpenAI/math, pinned isotypic/coefficient-integral section](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Signed-tensor-densities-and-diagram-budgets-for-coordinate-sweeps-September-26-2026/build/sections/03b-isotypic.tex),
  targeted lines185-244. Retaining declared projection support is a design
  inspiration only; its signed tensor bounds are not transferred to psychology.
