# Curvature Factor Transport and Conditional Reading

2026-10-09 JST. HM41 is verified on the frozen HM39 midpoints. Goal ACTIVE.
The pre-execution plan and all code, source bundles and numerical results remain
sealed; this document records their interpretation, not a changed experiment.

## Measured operation

Let the frozen reader's ten digit logits be l(H,P,B,Q), with the original
digit values a and softmax expectation F=m(l). H is the four-token property
state, P the ordered early-query states, and B/Q the other fixed receiver
states. For an actual physical H displacement d,

```text
g = gradient_l m, C = Hessian_l m
J_H = derivative_H l, J_P = derivative_P l
v_link(d) = J_P^T C J_H d
v_decoder(d) = sum_j g_j derivative_P <derivative_H l_j, d>
v_full(d) = v_link(d) + v_decoder(d) + numerical_remainder
A(d,e) = <v_full(d), e>, e = physical query-state displacement.
```

These quantities are local reader derivatives and actions, not personality
scores. C is the Hessian of a softmax-weighted numeric score, not cross-entropy;
it is symmetric but not generally positive semidefinite. Both Jacobians and C
depend on the receiver. The second-order composition rule is established
mathematics; see Dangel, Harmeling and Hennig (2020), Eq.7.
[Primary derivation](https://proceedings.mlr.press/v108/dangel20a/dangel20a.pdf).

## Source references and ordered substitutions

Each query and H/B/Q cell supplies two original source wordings. Their means
define C0, H0=mean(J_H), P0=mean(J_P); their original triples also define

```text
source_mean(d)   = P0^T C0 H0 d
source_paired(d) = mean_s[J_P,s^T C_s J_H,s d].
```

The two are different operators. Source-only coefficients do not make this
a raw-text, LLM-free system: physical H/P inputs still come from the reader.
All 24 source bundles were saved and sealed before application arrays were
loaded. The two wordings' absolute token positions differ by one; substitutions
align the declared four H slots and per-query ordered P slots, preserve every
absolute address, and do not claim position-invariant neural intervention.

For S a subset of {C,H,P}, select actual factors for members of S and source
means for the others. Evaluate all eight L(S)=P(S)^T C(S) H(S)d. The exact
Boolean-lattice decomposition is

```text
M(S) = sum_{T subset S} (-1)^(|S|-|T|) L(T)
L({C,H,P}) = sum_{S subset {C,H,P}} M(S).
```

With DeltaC=C-C0, DeltaH=J_H-H0 and DeltaP=J_P-P0, M(S) is also the direct
ordered contraction using the delta factor for members of S and its source
factor otherwise. DeltaP here denotes a Jacobian difference, not displacement e.
This follows from multilinearity and inclusion-exclusion, not a new theorem.
Boolean-lattice inversion is described by Stanley, Enumerative Combinatorics,
Volume 1, Proposition 3.7.1 and Example 3.8.3.
[Primary text](https://math.mit.edu/~rstan/ec/ec1.pdf).

All substitutions using actual application factors are oracle diagnostics.
They locate a component whose variation matters; they do not constitute a cold
predictor or establish that a causal forward patch has the same effect. Every
singleton, pair and triple, decoder term and signed cross moment is retained.

## What the finite observations establish

The experiment reused all 96 midpoints: three queries, source/application,
two wordings, eight H/B/Q cells and two physical H directions. No model calls
or fits were added. Below, gain means reduction in mean squared *link-action
error* relative to source_mean, not accuracy, explained psychological variance
or removal of decoder error. U/M refer to unseen_source/known_mean H arms.

| Application query | Actual C only gain U/M | Actual J_H only gain U/M | Actual C and J_H gain U/M |
| --- | ---: | ---: | ---: |
| Negative affect | -26.41% / -102.00% | 95.44% / 97.25% | 91.38% / 79.36% |
| Low mood | 44.74% / 12.62% | 61.75% / 72.07% | 97.01% / 90.21% |
| AI trust | 36.75% / 32.95% | 58.79% / 58.24% | 94.28% / 93.73% |

Negative affect's wording-level transport difficulty lies substantially in
text-to-logit sensitivity, rather than output curvature alone. Low mood and
trust benefit from their joint replacement. A scalar confidence or entropy
calibration cannot be assumed to replace this joint object; the observed
entropy ranges overlap between source and application.

The result depends on what is contrasted. source_paired improves the low-mood
pair mean by 64.81%/82.16%, but worsens the pair contrast by 18.26%/23.57%.
For trust the analogous gains are 36.70%/37.02% and -1.85%/-2.09%. Preserving
source pairing alone does not recover local wording-specific reading.

Negative affect's known_mean contrast is an important counterexample: J_H-only
replacement increases link error MS by 140.01%, C-only by 1344.92% and P-only
by 1647.55%, despite the strong wording-level J_H gain. Baseline contrast error
RMS is 3.6613e-6, J_H-only 5.6721e-6 and numerical remainder 1.9659e-10;
the failure is not erased as an unresolved roundoff ratio. Other factor changes
and their cancellation matter. The unseen_source contrast instead retains a
98.01% J_H-only gain. Both arms remain in the record.

Finally, approximating link is not approximating the complete mixed action.
source_mean's error divided by full mixed-action RMS is 2.154/1.200 for
negative affect, .935/.999 for low mood and .728/.756 for trust. Retaining
decoder plus its cross terms is essential; no overall prediction improvement
is inferred from a successful oracle link substitution.

## Implication for GRAIN's mathematical compiler

The minimal local reading object is not just text evidence and a final scoring
curve. It also contains how evidence changes logits in the current query and
receiver. A prospective compiler needs an input-dependent law for this operation,
not merely a source average of final coefficients. HM41 identifies where to
look for that law without selecting a larger rank or kernel.

HM42 will use one fixed downstream graph cut to decompose J_H into text-body,
early-query and remaining-endpoint routes. The chain rule makes their sum exact;
the observed route allocation will decide whether to develop a query-conditioned
reading law or preserve a body/endpoint carrier. This is a planned finite neural
measurement, not an already discovered universal route or attention head.

OpenAI/math's noncommutative-formula encoding manuscript motivates respecting
ordered products, node domains and evaluations that cancellation can conceal.
Its rational-formula assumptions are not established for softmax readers, so
the hitting-list theorem is not applied here. The empirical contribution is
the measured transport/cancellation behavior of the declared reader operations.
[Fixed manuscript](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026/build/sections/encoding.tex).

## Reproduction and scope

Canonical result: results/grain_math_response_v1/curvature_factor_transport_r1.
Verification: curvature_factor_transport_verification_r1/summary.json.
All 247 output and 10448 ancestry/code hashes were checked; alternate direct
contractions, inclusion-exclusion and table replay have maximum error
7.105427357601002e-15. Source/evaluation/verification cost 44.2369/300 CPU wall
seconds. There are 13 new tests; the fresh 69-file math suite passes 622 tests.
This is same-agent alternate arithmetic review on existing texts and midpoints.

Original 18 protocols/100 cards/44 computational requirements, natural-author
response parameters B_u and psychological interpretation remain open. No
API, private MEPS, new neural training, Claude dispatch, automation, publication,
lockbox action or old queue was used. Goal ACTIVE; no experiment process remains.
