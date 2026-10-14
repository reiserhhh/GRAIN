# Behavior-keyed forward observation on author evidence

HM261, 2026-10-13. Sections 1--6 are written BEFORE this packet's native
observations. This is a construction of a measuring function, not a new
likelihood theorem, a trait predictor optimization, or an author questionnaire.

## 1. A different observation law, not a more precise old answer

Let X denote an authenticated public author context, and let b_j be a complete
first-person behavioral self-description with an explicit end-of-turn token.
The neutral, fixed request asks for a plausible self-description by the writer.
It does not name a personality factor or tell the observer which pole to use.

    ell_j(X) = log p_theta(b_j | neutral_request, X, assistant_prefix_I)
             = sum_t {W_bjt . h_jt(X) + bias_bjt - A(h_jt(X))}.

The model, tokenization, request, all 50 public IPIP statements, and their keys
are fixed before outcomes. A is the full-vocabulary log partition. The candidate
events are distinct and complete, but not exhaustive. We therefore do not
renormalize their probabilities into an artificial questionnaire alphabet.

The author/foreign contrast lambda_uj = ell_j(own_u) - ell_j(foreign_u) measures
how these two evidence contexts change the model's support for that candidate.
It is NOT the probability that the author endorses the statement, the frequency
of an observed real behavior, or a human trait posterior. New author contexts
require new neural observation; exported accounts can then be scored numerically.

HM255 binary judgments and HM257--260 trait-conditioned continuations use other
observation laws. Changing the law is not an isolated wording intervention.
The old trait-conditioned memory cannot be silently reused under the new law.

## 2. Keep directional and common evidence as distinct coordinates

For each independently published ten-item key q, H_q and L_q contain its high
and low behavioral poles. With h and l their respective mean lambda values:

    B_q = (h - l)/2,
    U_q = (h + l)/2,
    Q_q = (sum_H lambda - sum_L lambda)/10 = B_q + a_q U_q,
    a_q = (|H_q| - |L_q|)/10.

This is an exact linear key identity. U_q remains a measurement coordinate,
not discarded noise. An equally weighted item score Q_q can contain common
support when its key is unbalanced. In particular the N orientation has eight
high and two low items, hence a_N = .6; it reverses the published Emotional
Stability key. The O anchor is Intellect/Imagination, not equivalence to every
Openness inventory.

If all ten candidates receive the same context gain c, B_q = 0 and U_q = c.
That finite counterexample explains what B rules out: a strictly common gain
on this key. It does not rule out generic language, register, content, or
opportunity effects that differ between the two pole vocabularies. High B or
positive SOURCE correspondence alone cannot resolve all those alternatives.

All items, both poles, B/U/Q and both directions are retained. Raw complete
sequence sums are the primary unit; per-candidate token normalization is a
fixed secondary account, never a replacement selected for a better result.

## 3. Measuring covectors depend on the observation contract

At fixed target-token addresses, the derivative of the complete likelihood
with respect to a differentiable representation K is the standard chain rule:

    D_K ell_j = sum_t (D_K h_jt)^T
                    [W_bjt - sum_v p_jtv W_v].

The bracket is a measuring covector on the output state; the Jacobian carries
it back through the candidate-conditioned reader. This is not the unqualified
input embedding direction of the sentence. Even with a shared neutral prefix,
different candidate continuations can produce different readers. This packet
does not estimate those Jacobians or identify a unique causal mediator.

Under an invertible state-coordinate change h' = A h and corresponding output
rows W' = A^(-T) W, the scalar pairing is unchanged. Arbitrary input cosine
similarity does not have that property. No new metric is tuned here to turn
negative observations positive. The function is defined by the declared model
law and complete candidate, not by a fitted angle in author-vector space.

## 4. GRAIN supplies a separate source-relative coordinate system

Use the inherited frozen SOURCE fields S_up on the SAME author evidence units,
with current 24 authors excluded from the 386-author source calibration.
The correspondence functional is:

    M_p[y] = mean_u (S_up - mean S_p)(y_u - mean y).

We inspect the complete 5x5 M_p[B_q] matrix, not only the named diagonal.
SOURCE full = choice + within is retained exactly; GRAIN15 coordinates are
also reported. Two directional uses belong to each of 24 historical authors.
They are not 48 independent people; 50 candidates are not independent trials.

The fixed marginal bootstrap resamples whole authors and is conditional on
the SOURCE estimate. It is exploratory, without family adjustment or source
estimation uncertainty. Anonymous representation discovery overlaps 17 authors,
and stranger matching has no exact subreddit match. These inherited properties
must stay visible when interpreting a correspondence, not become hidden zeros.

## 5. What the experiment changes if either mechanism is observed

H_contract: independently keyed forward behavior support has useful SOURCE
correspondence that trait-name/binary judgment contracts failed to express.
This would provide a usable behavioral observation interface without asking
the model to label a personality. It would not uniquely attribute the change
to omission of a trait word or establish actual author endorsement.

H_source: that correspondence is not recovered in this forward law. Then
another readout substitution is insufficient; the next construction must
address SOURCE meaning, evidence opportunity, or source-to-behavior linking.
It must not rescue this experiment with a name, head, layer, or precision grid.

Mixed signs and target-specific outcomes remain informative. A meaningful
construction consequence, not the count of positive cells or a higher r/AUC,
is the scientific deliverable. These are correspondence units, not r or AUC.

## 6. Primary ancestry and the actual project delta

- [Official IPIP keys](https://ipip.ori.org/newBigFive5broadKey.htm) and
  [public-domain permission](https://ipip.ori.org/newPermission.htm) supply the
  independent complete behavior descriptions and directions.
- [PICLe 2024, section 3.1](https://arxiv.org/html/2405.02501v2) already uses
  persona likelihood ratios. Neither likelihood ratios nor signed item keys
  are new mathematical tools introduced by this project.
- [Park et al. 2024](https://proceedings.mlr.press/v235/park24c.html) distinguishes
  input/context and output/word representations and their measurement geometry.
  Its causal-inner-product assumptions are not assumed for arbitrary authors.
- [Pinned OpenAI/math foundations](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex)
  motivate specifying the experiment before combining observations. Their
  conditional independence and tree reconstruction theorem do not transfer.

The project delta is the executed same-evidence bridge between SOURCE-relative
author coordinates and a neutral, behavior-keyed forward probability law.
It is a new construction in the local system, not a claim of field-wide priority
or a completed psychological measurement theory. Original18/100/44 remain open.

