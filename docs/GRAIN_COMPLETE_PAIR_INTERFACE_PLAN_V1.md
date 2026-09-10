# NEXT26: one complete conditional pair score without borrowed current endpoints

2026-09-10. Next finite mainline experiment after NEXT24/25. This plan chooses
the actual target/input contract, not a new predictive metric search.

## Why this interface

The positive empirical moment candidate has separated strength from mean, but
its effect on final N scores is tiny and it does not outperform the pair-energy
mean. Further second-moment calibration is not the next useful dependency.
The current signed mixed-logit reader already computes useful adjacent-pair
effects, but its score assembly borrows current z_add=z_i+z_j-z_0. Replace that
borrowed input with an actual source-only function while keeping the mixed
reader unchanged. Do not pick new pairs, traits, folds or human labels.

## Exact inherited target

Use the same24 N authors, four old author folds, both adjacent(0,1) and
distant(0,11) pairs, both inherited bag/position input families. All outputs
remain centered candidate logits followed by the ORIGINAL alphabet S.

    z_ij = z_0 + d_i + d_j + mixed_ij
    d_i = z_i-z_0; mixed_ij = z_ij-z_i-z_j+z_0.

z_0 is the all-comment mean-null, fixed-PQ endpoint. z_i restores only comment
i's within-comment configuration. This conditional pair score is not full
unmodified F(T,q) or human questionnaire truth. The old targets must replay.

## Inputs, no current-author teacher endpoint

For each author use existing block16 state, comment groups and query mask to
construct the complete mean-null context inputs:

- M: all12 comment mean vectors, in original slot order.
- R: every role-aligned nonbody/nonquery retained-state vector.
- L: comment starts/ends, retained token positions, query start/count and total
  length; verify a fixed coordinate layout, otherwise stop this constructor.
- phi_i: original frozen-fold Fourier contrast of comment i, for i=0,1,11.
- Fixed source geometry/PQ anchors, observer/alphabet and fitted source weights.

M,R,L determine the mean-null conditioned hidden input with the fixed PQ anchor.
Use all these observed inputs, not a reduced representation silently unable to
distinguish the target. This is an encoded-input interface; upstream block16
encoding remains a neural dependency and is not called raw-text LLM-free.

## One source-only construction, fixed .01 regularization

1. Context kernel k_C: training-centered linear kernels on M,R,L, each divided
   by mean training squared norm, averaged across nonconstant blocks. Retain
   any constant block in metadata but not a zero division. No feature selection.
2. Base z_0 head: vector kernel Ridge with an unpenalized training mean target
   and k_C. Center, scales, target mean and regularization use18 training authors.
3. Shared singleton increment head: kernel
   (1+k_C(u,v))*<phi_ui,phi_vj>, trained on those18 authors x3 singleton positions.
   Zero intercept. Thus zero comment contrast has exactly zero increment.
4. Keep the existing signed mixed-logit Ridge for each fold/family untouched.
   Assemble z_add_hat=z_0_hat+d_i_hat+d_j_hat, then z_pair_hat=z_add_hat+mixed_hat.
   Penalty is n*.01 times the kernel's training mean diagonal, as in old code.

This is one structured function family, not an arm/parameter sweep. Four base
heads plus eight singleton heads are the maximum new small numerical fits.
No new model calls, training neural weights, API, lockbox or questionnaire data.

## Freeze then evaluate four explicit assemblies

1. actual z_add + old mixed_hat: inherited oracle-context comparison.
2. predicted z_add + old mixed_hat: PRIMARY, no current-author native endpoints.
3. predicted z_add + actual mixed: diagnostic oracle only, to localize baseline
   error rather than interpret assembly cancellation as a new mechanism.
4. predicted z_add alone: predicted additive baseline.

All predictions and models freeze before native endpoint scoring. Report logit
base/singleton/additive/mixed/full errors, exact signed error cross terms, score
RMS/MAE/r and author-bootstrap paired differences. Show full-level score error
and local interaction error on different denominators. Never hide a failed
full interface behind good mixed-action correlation or high baseline dominance.

## Completion and next decision

- A real serialized predictor taking only declared numeric inputs must reproduce
  stored OOF predictions with endpoint files unavailable. Include a subprocess
  source-denial replay, not merely a manifest claiming independence.
- Tests: training-author exclusion; source-only centering; exact zero/swap;
  finite target algebra; constant-block handling; full error decomposition;
  source-denied predictions. Private IDs, inputs, models stay ignored.
- If fidelity is useful, the next development question is transporting this
  complete encoded interface to the existing early encoder. If it fails, locate
  which base/context or singleton function loses action before any new candidate.
- No claim that the full GRAIN/personality theory is finished. This removes one
  concrete endpoint dependency and gives a truthful complete subfunction.
