# R08 controlled, source-fitted predictive-state witness: protocol V1 (frozen before any application computation)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_15_HISTORY_AND_PRAGMATIC_READER.md` Part B; `SEMINAR_FABLE_13_DECISION.md`.
- **Runner and output.** `scripts/run_grain_r08_controlled_psr_witness.py`, writing to
  `results/grain_gptpro_foundation_v3/F15/controlled_psr_witness_r1/`.
- **Scope.** CPU ≤ 10 min, 2 workers × 1 BLAS thread, 0 neural calls, ≤ 1,000,000 generated observation steps.
- **Method grounding.**
  - A transformed PSR / observable-operator model with actions, estimated spectrally from observed action–observation
    windows (Boots, Siddiqi & Gordon 2011; Thon & Jaeger 2015 relate PSRs and OOMs).
  - Named honestly: a finite-horizon spectral OOM/PSR with fixed history and test sets, **not** an unrestricted PSR
    learner.

## Worlds (existing kernels only, from `F15/text_behaviour_quotient_r1`; imported read-only)

- **Worlds.** W1 (the documented collision world) and W6, both in the construct view o = (c, B). The view has 4
  symbols; the actions are a0 and a1; the register form f is marginalized.
- **Seeds.** 20261101, 20261102, 20261103.
- **Generator** (the only user of the kernel; the learner sees only (a, o) sequences):
  - x ~ uniform, and (h, r) is drawn from the stationary law of T_mix = ½(T_a0 + T_a1) given x, so the process is
    stationary under the policy.
  - Length 10.
  - Blind policy a_t ~ Bernoulli(.5), i.i.d.
- **Roles, disjoint sequences:** source 8,000; source-validation 1,500; application 2,500. That is 120,000 steps per
  world-seed and 720,000 in total.
- **Support variants** (W6, seed 20261101 only; source 8,000 and application 2,500 each; 210,000 steps):
  - V_unsup: the source policy is a0 only, so a1 is a novel **primitive** in application.
  - V_comb: in the source policy a1 is never followed by a1, so the pair (a1, a1) is a novel **combination** of
    supported primitives.
- **Total:** 930,000 steps.

## Learner (fitted from SOURCE windows only)

- **Histories.** All (a, o) sequences of length 2: 64 of them.
- **Tests.** All (a, o) sequences of length 1 or 2: 72 of them.
- **Windows.** Positions i = 0..5 of each sequence; stationarity makes windows exchangeable.
- **Blind-policy conditional estimates** (counts divided by counts of the same action sequence):
  - P_H[h] = P(h_o | h_a);
  - P_T[t] = P(t_o | t_a);
  - P_{T,H}[t, h] = P(h_o, t_o | h_a, t_a);
  - P_{T,ao,H}[t, h] = P(h_o, o, t_o | h_a, a, t_a).
  An action sequence with zero count makes its entries undefined: the operator is **not estimable** and the model
  refuses.
- **Spectral fit.**
  - U = the top-k left singular vectors of P_{T,H}.
  - b1 = Uᵀ P_T.
  - b∞ᵀ = P_Hᵀ (Uᵀ P_{T,H})⁺.
  - B_ao = Uᵀ P_{T,ao,H} (Uᵀ P_{T,H})⁺.
- **Rank rule.** k ∈ {1..8}, chosen by mean one-step log loss on source-validation sequences (filtered from b1,
  positions 2..9).
- **Prediction and update.**
  - P(o | a, b) = b∞ᵀ B_ao b. Negative values are clipped to 1e-6 and renormalized over o, with the clipping
    counted.
  - Update b ← B_ao b / (b∞ᵀ B_ao b). A non-positive denominator resets b to b1 and is counted.

## Baselines (same observable prefix and next action; fitted on source; hyperparameters chosen on source-validation)

1. **Static composition.** P(o | a) = (n_ao + α P_pop(o | a)) / (n_a + α), with n counted over the prefix and
   α ∈ {1, 3, 10, 30}.
2. **History PCA with a matched future decoder.** One-hot of the last 4 (a, o) steps (32 dimensions) → PCA with
   k = the learned PSR rank → a multinomial logistic decoder per next action (C = 1). Capacity difference: the window
   is fixed at 4 steps.
3. **Order-one.** P(o_t | o_{t−1}, a_t) from source counts, plus one.
4. **Oracle.** The exact belief filter with the true kernel. **Reference only**, computed after all fits, and never
   used for rank, parameters or selection.

## Scoring (application sequences)

- **Next-step prediction.** Log loss and Brier at t = 4..9, given the prefix and a_t.
- **Two-step continuation** at t = 4: joint (o_4, o_5) given (a_4, a_5), with each model updating on the hypothesized
  o_4.
- **Uncertainty.** Paired sequence bootstrap: 2,000 draws, seed 20260926.
- **History-collision example.** Application prefixes (t = 4) with identical static composition and the same last
  observation but oracle next-step TV > .1; report every model's predictions.
- **Diagnostics.**
  - Source reconstruction: model-implied P(t_o | t_a) from b1 against empirical source P_T. **Gate:** TV ≤ .02 for
    every test action sequence.
  - Normalization deviation, and clip and reset counts.
  - Recursive closure: two-step joint by direct operators against the product of recursive conditionals.

## Pre-declared falsifiers and stops

- **Q1 (support gate).** If source reconstruction fails for a world-seed, that arm is an **estimator/support
  limitation**. Its application comparison is not evidence, and no retuning follows.
- **Q2.** If the learned PSR does not beat static composition and order-one on application next-step log loss (a
  paired interval excluding 0 in its favour), that is reported as found. It is not a refutation of PSRs in general.
- **Q3.** In V_unsup, any learned model that emits an a1 prediction is an implementation error: STOP.
- **Q4.** Any oracle quantity influencing selection is an implementation error.
- **No sweeps.** No sample-size or rank sweep beyond the declared k ∈ {1..8} validation choice.
