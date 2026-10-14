# R01 source-crossed derivative design: protocol V1 (packet24 B)

Frozen 2026-09-27 JST before any fit or score.
- **Authority.** `NEXT_PACKET_24_SOURCE_CROSSED_ACTION.md` Part B; `SEMINAR_FABLE_19_DECISION.md`, including its
  corrections.
- **Code.** Module `grain_research/r01_source_crossed.py` (bound). Runner `scripts/run_grain_r01_source_crossed.py`.
- **Output root.** `results/grain_gptpro_foundation_v3/P05/source_crossed_action_r1/`, created in x-mode.
- **Budget.** ≤ 10 min CPU, one process, single-thread BLAS. 0 neural calls, no finite native patches. Cached arrays
  only.
- **Expected Jacobian tree hashes** (packet 18):
  - `private_jacobians` 35e1b6d4d305a8e0aacf0f47827e42bf99921515d0fcf295c19c2e8ea649d751;
  - `private_app_jacobians` 8601b303be50ad1b0279ab4d7aad89c831c25cd6401d703a27955bc36adb60cc.

## Objects

- **Per query.** Extraversion segment dh[:, 0:7552]; neuroticism segment dh[:, 7552:15232], as in the frozen native
  experiment. Same head, layer, query-suffix positions, LOW base and sign. The two queries are never crossed.
- **Gradients.** u_j = √s J_j, with s the frozen source-only metric scale from `fit.json`, checked equal to
  D / mean‖J‖².
- **Donors.** d_i is the cached donor activation difference.
- **Crossed derivatives.** T_ij = d_i · u_j, a 36 × 36 matrix per query: cached directional derivatives. They are NOT new
  independent samples, finite native patch outcomes or W-generated labels. The material count stays 36 pairs / 12
  cells.
- **Replay gates.**
  - diag(T) equals the old t_ii.
  - The old J4/J24/E4/E24 compiled actions are reproduced from the closed forms.
- **Diagnostics on the full 36-source panel, per input z4/z24:**
  - rank(Z), rank(U) and the full-cross rank rank(Z) · rank(U), with the numerical rank of the explicit primal
    operator z_i ⊗ v_j;
  - the product-Gram identity (singular values = products);
  - the diagonal-design rank, rank(ZZᵀ ∘ UUᵀ);
  - ‖P_Z T‖²_F / ‖T‖²_F with rank-aware pinv. T = Z A needs every COLUMN of T in col(Z). A low share rejects this fixed
    linear map, not all information in z.

## Source-cell-held-out comparison

- **Folds.** 12 folds, one per original source cell. The held cell's 3 seeds are removed as BOTH donor rows and
  recipient columns. n = 33. The CV is conditional on the frozen z and s transforms.
- **Loss for every design.** mean_rows (z_iᵀ W u_j − T_ij)² + λ‖W‖²_F with λ = 1/n, in the orthonormal basis Q of the
  training-recipient U row space (W = B Qᵀ, U = V Qᵀ).
  - D: n diagonal rows.
  - C: cyclic, donor cell k to recipient cell k + 1 in sorted training-cell order, same seed.
  - R: one fixed-seed derangement of the cells plus a within-cell seed permutation, seed [24011, fold]. Row lists are
    saved.
  - F: all n² rows. Equation: (ZᵀZ) B (VᵀV) + mλ B = Zᵀ T V, with m = n², so summed α = n; D/C/R have summed α = 1.
    It is solved by eigen factors and checked against explicit primal and dual ridge in fold 0.
  - E: the identity-metric donor regression reference, mean_rows ‖zᵀW − d‖² + (1/n)‖W‖² (summed α 1).
  - ZERO: the zero reference.
  - There is no rank, λ or seed grid.
- **Primary evaluation.** The held cells' own diagonal derivatives T_hh/√s (native score units), three per fold:
  pooled RMSE, correlation, per-cell MSE, paired cell-bootstrap deltas (2,000 draws) and failure cells (worse than
  zero). Intervals are descriptive: training folds overlap.
- **Supplement.** Held donor × training recipients, aggregated per held donor and then per cell.
- **Exposure.** The held diagonal product row's exposure under each design, reported separately from prediction loss.
- **Application panel (descriptive, after freezing).** Full-36 fits of every design score the old 72-pair application
  panel once, against the application diagonal DERIVATIVE J_a · d_a.
  - Finite native outcomes are NOT evaluated, because their cache semantics are not established in this packet.
  - No selection.

## Checksum bindings

| file | sha256 |
|---|---|
| `grain_research/r01_source_crossed.py` | `bb237e5865c2b5392c997ec554edfcbbd8beee5e75a857530c6ca7e6118b9769` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/action_arrays.npz` | `778a5acdd88649e31eb8b5bd0013ddab0e50bacf4cc3e0d0bbe2325b8f9c0a9f` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/pair_order.json` | `a942fd048c42ac89a241ee27929f29f37d7599e0c9c3fb114d3d935eeabaf17a` |
| `results/grain_gptpro_foundation_v3/P05/native_action_r1/private_prepared.npz` | `b9ce2b1c8de9365adf45a1cced3cd351dc83d24f230b663104d4aead92940338` |
| `results/grain_gptpro_foundation_v3/P05/native_action_r1/fit.json` | `03e164e64b4c125b86ae20f8a2b5d2e42dde07a19e4f0105b8967454ff10f0ad` |
| `docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json` | `4f3d38693cf8dd2024fc5a8be5a9ee2c7b4f80479bcf230d04e10a1ebe557b87` |
| `scripts/run_grain_r01_product_space.py` | `d8d41496d5b107afc050db26dfa3e30d3b5845b263495c4e373691e2c68a9e47` |
