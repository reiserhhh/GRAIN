# NEXT64: conditional scalar development and context geometry

2026-09-11. Computational development, not a new human questionnaire result.
Primary code `be55c8f` preceded four source-only fits. Original frozen plan:
[NEXT64](GRAIN_CONTEXT_MODULATED_POTENTIAL_PLAN_V1.md). Its parameter choices
were followed, not retroactively replaced after the outcome.

## 1. What was implemented

Let z_t=(H_t U-mu)/s be the unchanged source BODY coordinates, p_t the unchanged
position design, and l the original last coordinate in the SAME BODY reference.
For each frozen token landmark m, phi_m consists of its Gaussian feature and
eight center-derivative/Hermite features. The independent context coordinate c
comes from the four ORIGINAL source last states, not gradients or labels.

\[
\widehat L_q(H,l,c)=b+a^T\bar z+\beta^Tl
 +\frac1T\sum_{t,m}[\theta_m+\gamma_m(c)e_m]^T\phi_m(z_t,p_t),
\quad
\gamma_m(c)=e^{-\|c-c_m\|^2/(2\sigma_c^2)}
             -e^{-\|c_m\|^2/(2\sigma_c^2)}.
\]

The additive program has593 coefficients; the conditional extension1169. This
is a nested comparison with matched teaching data, NOT a matched-capacity test.
Setting e=0 recovers the old family. Setting c=0 removes the added scalar and
BODY-gradient term, but does not require the mixed derivative to be zero.

Both outputs come from the same scalar:

\[
\widehat\tau_f=\sum_t\langle\nabla_{H_t}\widehat L_q,D_{f,t}\rangle,
\quad
\nabla_c\nabla_{z_t}\widehat L_q
 =\frac1T\sum_m\nabla_{z_t}(e_m^T\phi_m)\nabla_c\gamma_m^T.
\]

The actual last-state derivative includes both beta and the context chain rule.
BODY and last finite differences, mixed-derivative/Hessian symmetry and path
integration agree in synthetic tests;384 real-input BODY directional checks
agree within6.74e-10. This is an implementable function, not an arbitrary
independently fitted vector field.

## 2. What the matched experiment says

Four source authors each contribute original and NEXT63 shifted-last views.
Each author has total weight1/4, each view1/2 within author; original source
score/density scales and alpha=.01 are unchanged. All4 teach; the existing
exposed new8 is source-disjoint application. Donor-mean targets are not used in
a fake source leave-one-out experiment. No new neural calls, humanY or MEPS.

The same source intervention now produces nonzero mathematical GRAIN responses.
However, predicted change RMS is only7.54%/8.20% of native change RMS, with
change correlations .143/-.371 (N/confirmation). This is not a successful
recovery of both measured mixed actions. Source objective falls, yet the source
interaction itself remains weak; do not attribute everything to new-author shift.

New8 full-action RMSE falls .757%/1.092% against the augmented additive control,
but both are worse than the old original-source-only reference. Whole-model
score r falls .228->.108 and .381->.250. The conditional model's full-action
SSE/zero1.119/1.106 and scoreSSE/sourceconstant1.107/.945 fail whole promotion.
N allocation and the paired-query allocation contrast retain local utility;
the contrast r=.499, SSE/zero=.789. All negative factors and full responses stay.

## 3. A post-outcome algebraic finding: source-context whitening forms a simplex

This account was added AFTER the primary results. It changes no fit or prediction.
For n centered source rows X of rank n-1, thin SVD gives X=U S V^T. PCA coordinates
divided by their population SD are C=sqrt(n) U. Hence

\[
CC^T=nI-\mathbf1\mathbf1^T,\qquad
\|C_i-C_j\|^2=2n\quad(i\ne j).
\]

For n=4, EVERY source pair is distance sqrt(8), regardless of the original
distance differences. In this particular Gaussian context gate, source-reference
weights therefore distinguish same-source versus other-source but discard the
relative closeness among different source authors. This does not make the whole
token function constant: BODY content still varies, and new8 projected locations
can vary. It identifies a specific small-sample reference distortion, not a
universal argument against PCA, standardization, or conditional functions.

Verified source distances before whitening are3.211--4.046 for N and3.559--3.985
for confirmation; CV .07744/.04784 becomes <9e-16. Simplex Gram error<=1.16e-14.
Separately, the source3 context span covers only9.40%/11.03% of new8 displacement
energy about the source mean. Energy outside the span is not automatically
psychological signal, but this implementation does not let it modulate BODY.
These facts motivate a geometry-preserving comparison, not a causal conclusion
that whitening explains all observed prediction loss.

## 4. Connections to prior work

[FiLM, Perez et al.](https://arxiv.org/html/1709.07871v2), equations1--2, conditions
feature-wise scale/shift on another input. It supplies a precedent for input-dependent
feature reading. Our frozen finite kernel scalar is not their visual-reasoning
architecture, nor a claim that Qwen literally contains a FiLM layer.

[Solak et al., NIPS2002](https://papers.nips.cc/paper_files/paper/2002/file/5b8e4fd39d9786228649a8a8bec4e008-Paper.pdf),
section2.2/equations2--3, combines function and derivative observations by
differentiating a common covariance function. It supports keeping value/derivative
compatibility rather than fitting unrelated outputs. Here we solve a finite
ridge problem, not a Gaussian-process posterior or uncertainty calculation.

The simplex result above follows directly from centered full-row-rank whitening;
it is an elementary linear-algebra identity, not a newly discovered theorem.
Its relevance to our current context-reference failure is the project finding.

## 5. Mainline implication

The native interaction from NEXT63 remains established. The first small-source
conditional extraction is closed without promotion. The next intervention is
on the INPUT GEOMETRY, not a donor/dose/alpha search: use full original last-state
distances in the context kernel, keeping the same source views, BODY dictionary,
objective and parameter count. See [NEXT65](GRAIN_FULL_CONTEXT_GEOMETRY_PLAN_V1.md).

GRAIN supplies fixed author-text perturbation directions; the conditional
reader supplies query-dependent action on those directions. Both remain linked
to the same material and compatible scalar. Evidence-linked psychodynamic/TA
interpretation remains a separate candidate-coding interface, not an inference
from these kernel coefficients to clinical constructs. H16 is still neural.
