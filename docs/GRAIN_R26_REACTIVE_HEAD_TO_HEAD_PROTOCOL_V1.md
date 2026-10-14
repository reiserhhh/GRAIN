# R26 continuous-target head-to-head WITH reactivity: protocol V1

Frozen 2026-09-26 JST before computation.
- **Authority.** `NEXT_PACKET_19_RECONSTRUCTION_AND_REACTIVITY.md` Part B; `SUPERVISOR_CHECK_20260926_2042.md`.
- **Runner and output.** `scripts/run_grain_r26_reactive_head_to_head.py`, writing to
  `results/grain_gptpro_foundation_v3/F06/reactive_head_to_head_r1/`.
- **Execution.** CPU ≤ 20 min, one process, single-thread BLAS, 0 neural calls.
- **Scope.** The seminar-05 rival-identification CRN experiment is **not** rerun; its paired result stands. This
  packet fills the separate continuous-target reactivity gap in the F06/head_to_head_r1 design.

## Reused design (F06/head_to_head_r1, unchanged)

- **Bank.** 28 repeatable items: c ∈ {−1.5, −1, −.5, 0, .5, 1, 1.5}; natural (a = 1, art 0) or probe (a = 2, art 1);
  nuisance loading n ∈ {.2, 1.0}.
- **Persons and budget.** 400 persons; budget K = 8 turns.
- **Grids.** τ 41 points on [−3, 3]; κ 21 points on [−1, 3]; ν by 7-point Gauss–Hermite (integrated out).
- **Priors.** τ ~ N(0, 1); κ ~ N(1, .5²); ν ~ N(0, 1).
- **Policies.**
  - **F:** fixed natural low-nuisance order over c.
  - **S:** scalar CAT. Its model assumes κ = 1 and no nuisance, and it maximizes Fisher information at the posterior
    mean, so it is misspecified in κ and ν.
  - **O:** operator-aware. It maximizes the expected posterior-variance reduction of the planning target.
  - **OC:** O with costs.
- **Planning targets** (separate runs, as in F06):
  - **T1 = τ.**
  - **T2 = τ + 2κ**, the unperturbed condition c* = 2. This is **an extrapolation beyond the bank range**
    [−1.5, 1.5].

## Worlds (declared simulation parameters, not established psychological effects)

- **ZERO:** no reaction, r ≡ 0.
- **REACT:** r_0 = 0.
  - Response logit at turn t: a_j (τ + κ c_j + r_t) + n_j ν.
  - After the response: r_{t+1} = .6 r_t + .20 art_j. Natural items add 0.

## State clock (exact order at each turn t = 0…7)

1. The planner picks j_t using its posterior after y_0…y_{t−1}. The aware mode also uses r_t computed from its own
   actions j_0…j_{t−1}.
2. The response y_t = 1[U[u, t] < logistic(a (τ + κ c + r_t) + n ν)], using the state **before** this turn's
   reaction.
3. The posterior is updated with y_t. The aware likelihood uses r_t.
4. The reaction is applied: r_{t+1} = .6 r_t + .20 art_{j_t}.

The final state is r_8, after the last reaction. The planner never sees τ, κ, ν, U or future responses.

## Modes

- **Naive:** ignores r in the likelihood, in the expected-information planning, and in costs.
- **Known-law aware:** r_t is computed deterministically from the planner's **own** action history with the declared
  law. It enters both the likelihood and the expected information. This is a **best-case reference**: it has no
  access to τ, κ or ν, and it makes no claim that real reaction laws are known.
- **ZERO world:** naive and aware are identical by construction. Both are run, and their exact identity is checked
  and reported.

## Costs (OC; fixed; never re-weighted after results)

- utility = gain / (1 for T1, 1 + .25 c*² for T2) − .05 art − .05 n − .10 × Δr.
- Δr = .20 art is the immediate reaction increment. The Δr term applies only in the REACT world and only in aware
  mode, because naive assumes no reaction.
- For every policy, the saved cost components are Σ .05 art, Σ .05 n and Σ .10 Δr (under the true law).

## Randomness (stable integer seeds; no Python hash())

- **Latents.** Drawn once with `default_rng([19101, 1])`: τ, κ = 1 + .5 z, ν.
- **Uniform response stream.** U[u, t] from `default_rng([19101, 2])`, shared across all policies, modes and worlds.
  Different chosen items still produce different responses through their own probabilities.
- **Bootstrap.** 2,000 whole-person draws, `default_rng([19101, 3])`.

## Outputs (per person; every policy × mode × world × planning target)

- **Estimates and errors** for three targets, never interchanged:
  - **T1 = τ.**
  - **T2 = τ + 2κ**, unperturbed.
  - **Current-state target CS = τ + r_8**, a separate target. Naive predicts E[τ]; aware predicts E[τ] + r_8.
- **Posterior means.** Taken under each policy's own model; S predicts T2 as E[τ] + 2.
- **Reported per target:** bias, MSE and RMSE. 95% coverage is reported only where a grid posterior interval exists:
  O, OC and F on T1/T2, and S on T1. S on T2 is labelled NOT_SUPPORTED.
- **Actions and state:** item choices, probe count, accumulated Σr_t, final r_8, and the cost components.
- **Paired deltas** (2,000 whole-person bootstrap draws):
  - same-policy aware − naive (REACT), which isolates ignored reactivity;
  - REACT − ZERO within policy and mode;
  - within aware F vs O vs OC, the same inference family (a design comparison);
  - S vs O, a compound model/design comparison, not a pure policy win.
- **Cap rule.** If the cost exceeds the cap, a complete named subset is checkpointed and the remaining arms are left
  open. The design is not shrunk after seeing results.
- **Question answered.** Can active probes improve information yet distort the intended target? Does handling the
  action history or the cost change the preferred policy? No arm is required to win.
