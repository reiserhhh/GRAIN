# Text-to-Query Routes and Observable-Specific Error

2026-10-09 JST. HM42 stopped at its original coordinate replay gate. Its code,
specification, failed node and original results remain unchanged. A separate
zero-call partial-source audit is verified. Goal ACTIVE; the full scope is open.

## Complete-state cut

For the block16 patched leaf X and full block17 output S, partition positions
into body, the exact ordered early-query slots, and every remaining position.
The masks M_G are disjoint and sum to the identity. For digit logit l_j,

```text
J_H,j,G = (D_H S)^T M_G gradient_S(l_j)
J_H,j = sum_G J_H,j,G
v_G(d) = J_P^T C J_H,G d
A_G(d,e) = e^T v_G(d).
```

The downstream gradient is detached before VJP through S. The operation is
first-order differentiation with fixed cotangent weights, not differentiating
the gradient a second time. Residual and nonlinear branches through S are
retained. A branch bypassing S would invalidate completeness and is covered
by a synthetic witness that must fail. The graph-cut chain rule is standard
mathematics, not a new personality theorem.

This allocates the H leg at one chosen cut. J_P stays whole. Masking backward
weights is not a forward intervention, and allocation is not a unique circuit.
The three components have signed effects: their squared magnitudes require
cross terms before summation. A large norm is not an information percentage.

## Actual finite result and its boundary

The source prefix contains 33 actual forwards and 1320 first-gradient calls,
zero second-gradient calls, zero fits and 78.9136 model wall seconds. The first
32 nodes satisfy the original 1e-8 coordinate gate; node32 has maximum error
1.1624138096522074e-8. All33 logits, native readouts and scores are identical
to HM39. The main packet remains STOPPED_FAILURE_OR_BUDGET, not accepted.
The last saved node remains explicitly failed. Fifteen source and all48
application nodes are missing; no source barrier was created.

The separate text_query_route_partial_audit_r1 includes the failed node without
promoting it. On all33, the maximum link-covector difference is 5.52085e-12,
and maximum physical-action difference is 8.67799e-12. Alternate contraction
error is 2.30393e-18. These compare the stored actual arrays, not independent
neural acquisition. They establish that this particular coordinate replay
discrepancy has very small action under the declared directions. They do not
prove the error's floating-point cause or certify every possible direction.

For E=sum_G J_H,G-J_H, the error witness is

```text
w = C^T J_P e
DeltaA(d,e) = w^T E d
|DeltaA| <= ||w||_2 ||E d||_2
         <= ||w||_2 ||E||_F ||d||_2.
```

An error large in coordinates can lie near the kernel of an observed action.
Conversely, changing d/e or their units can make the same error important.
Thus a coordinate norm and a measurement-action discrepancy are different
objects. This is duality and Cauchy-Schwarz, not a license to ignore error.
The audit keeps both. No tolerance was raised to complete HM42.

## Source route evidence

Only negative affect and low mood have their full16 source nodes. Trust has
one point and no balanced mean/contrast summary. The two complete source
queries retain both wordings, eight H/B/Q cells and both H physical arms.

Body action RMS is larger than the other two routes in these source groups.
Yet retaining body alone loses a wording-contrast action of RMS
3.0573e-6/3.7692e-6 for negative affect and 3.1700e-6/4.4878e-6 for low mood
(unseen_source/known_mean). Relative to complete link contrast RMS this is
49.09%/49.91% and 24.06%/24.70%, respectively. These are omission RMS ratios,
not percentages of personality variance or prediction accuracy.

Negative affect body/query signed cross terms are negative for word and mean
but positive for contrast. The remainder route also contributes to contrasts.
Therefore a route small in the overall level can matter for differences.
Neither an exclusively query-carried model nor a body-only model follows.
These are source-only controlled wording effects, not natural-author differences.

## Relation to prior work and mathematical reading

Ferrando, Gallego and Costa-jussa (2022), ALTI, considers contextual mixing
through attention blocks including residual/LN and follows layer-wise token
contributions. The current exact signed derivative cut is a different observable:
it does not implement ALTI's normalized distance attribution or evaluate its
relative faithfulness. This link supplies a prior framework for studying mixing,
not a claim that raw attention weights alone determine the route.
[Primary article](https://aclanthology.org/2022.emnlp-main.595.pdf).

OpenAI/math's fixed noncommutative generator manuscript motivates keeping the
complete certificate family and ordered evaluations rather than a single
attractive sum. Its algebraic hitting theorem is not applied to softmax or
floating-point neural derivatives. The contribution here is measured route
and observable-specific behavior, not a new proof of that theorem.
[Fixed generator manuscript](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026/build/sections/generator.tex).

## Next operation

HM43 is separately planned to measure the existing physical action directly.
For each unchanged midpoint, form frozen w=C^T J_P e and use the scalar
L_w=w^T l, with w detached. One gradient into S and three masked VJPs yield
all three routes for both H directions. This requires four first-gradient
calls rather than forty per node. Preserve weight quantization as a separate
term and compare actual action to the unchanged HM39 physical action.

This is not a retry of the full-Jacobian assertion or a relaxed coordinate gate:
it measures a smaller, explicitly declared observable. It cannot claim to
recover every logit direction. All96 paths, source48 before application48,
three routes, two arms, means, contrasts, decoder and roundoff remain required.
Actual frozen C/J_P weights are mechanism diagnostics, not a cold predictor.

The next decision is whether these carriers can support a source-dependent
conditional reading law. Direct action measurement must come before replacing
them by source averages. Natural authors, B_u, psychological interpretation,
the original18/100/44 scope and the standalone compiler remain open.
