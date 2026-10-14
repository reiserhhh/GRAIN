# R08 conditional information bottleneck, bounded finite witness: protocol V1 (frozen before computation)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_14_PRAGMATIC_IDENTIFIABILITY.md` Part C.
- **Source clause.** P03 pp.22–23: histories are equivalent iff P(X_future | h₁, do(C)) = P(X_future | h₂, do(C)); seek
  Z = f(H) maximizing I(Z; X_future | C) while minimizing I(Z; H).
- **Runner and output.** `scripts/run_grain_r08_conditional_ib.py`, writing to
  `results/grain_gptpro_foundation_v3/F15/conditional_ib_r1/`.
- **Scope.** 0 neural calls, cap 10 min CPU.
- **Worlds.** The existing frozen W1 and W6 of `F15/text_behaviour_quotient_r1`, with kernels imported read-only from
  the r1 runner. No new world.

## Joint law (exact enumeration)

- **Stationary start.** Hidden s_{t−2} ~ π, the passive stationary law (uniform initial state propagated 500 steps
  under a0), as in r1.
- **History schedule.** Passive a0 at t−2 and t−1.
- **Observable history** H = (c_{t−2}, c_{t−1}, B_{t−1}): **8 symbols, full support**. The surface form f and
  B_{t−2} are marginalized, so H is observed and never hidden.
- **Condition** C ∈ {a0, a1} at t, chosen by the measurer with probability ½ each, independent of H.
- **Future** F = (c_t, B_t) under action C (4 symbols). F is fixed across every comparison.
- **The objective as the source states it:** conditional prediction I(Z; F | C) = Σ_c P(c) I(Z; F | C = c), against
  unconditional compression I(Z; H). C is side information at prediction and is never revealed in the code.

## Encoders and outputs

- **Encoder family, restricted.** Every deterministic code Z = g(H): all 4,140 set partitions of the 8 history
  symbols (Bell(8)). This is a deterministic-encoder Pareto frontier, not a global stochastic-IB optimum.
- **Frontier.** The non-dominated set over (I(Z; H), I(Z; F | C)), and the best code per number of blocks.
- **Endpoints.** The constant code (0, 0) and the full code (H(H), I(H; F | C)).
- **Fixed summaries.**
  - mean content, c_{t−2} + c_{t−1};
  - last content c_{t−1};
  - last behaviour B_{t−1};
  - last construct observation (c_{t−1}, B_{t−1}).
  For each: its retained information, its gap to the best code with the same number of blocks, and its **future
  collisions** (history pairs merged by the summary whose conditional future laws differ, with TV per condition).
- **Oracle, labelled separately.** The latent state gives I(S_t; F | C), an upper bound that no observable code
  attains.
- **Independent checks.**
  - Mutual information by two formulas: entropy decomposition, and a KL sum over the joint.
  - Data processing: I(Z; F | C) ≤ I(H; F | C) ≤ I(S_t; F | C).
  - Refinement monotonicity on the lattice.
  - Joint-law normalization.
- **Not done.** No β grid, no new world, no claim of general human minimality.
