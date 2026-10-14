# R01 common donor shift versus text-specific vector action: protocol V1 (packet 27)

Frozen 2026-09-27 05:03 JST, before any fit or score on cached arrays.
- **Authority.** `NEXT_PACKET_27_AFFINE_ACTION_DIAGNOSIS.md` Parts B–E; `SUPERVISOR_CHECK_20260927_0452.md`.
- **Code.** New module `grain_research/r01_source_affine.py` (bound). New runner `scripts/run_grain_r01_source_affine.py`
  (holds this protocol's sha). New tests `tests/test_grain_r01_source_affine.py` (bound). Frozen packet-24 module reused
  unchanged.
- **Output root.** `results/grain_gptpro_foundation_v3/P05/source_affine_action_r1/`, created in x-mode (refuse if it exists).
- **Budget.** Compute ≤ 10 min CPU, one process, single-thread BLAS. 0 neural/API calls. No text, labels, finite
  patches, training, remote work or new data. Cached numeric arrays only.
- **Expected Jacobian tree hashes** (packet 18): `private_jacobians`
  35e1b6d4d305a8e0aacf0f47827e42bf99921515d0fcf295c19c2e8ea649d751; `private_app_jacobians`
  8601b303be50ad1b0279ab4d7aad89c831c25cd6401d703a27955bc36adb60cc.

## Question

Does the missing common offset explain the packet-24 source linear map's misfit? Does any improvement encode
differences between texts, or only their shared HIGH-minus-LOW persona contrast?

## Objects (identical to packet 24)

- Per query: extraversion segment dh[:, 0:7552], neuroticism dh[:, 7552:15232]. Queries are never crossed.
- u_j = √s J_j with the frozen source-only metric scale s. d_i = cached donor difference. T_ij = d_i · u_j (36 × 36).
- T_ij are dependent cached DIRECTIONAL DERIVATIVES, not new persons, independent native interventions, finite patch
  outcomes or W-generated labels. Material stays 36 source pairs / 12 cells and 72 historical application pairs.
- Inputs z4 / z24: frozen HIGH-minus-LOW feature differences scaled by the source SD (not centered). z = 0 means no
  feature difference. CV is conditional on these frozen source transforms, not end-to-end.
- Reused source-cell and application panels are exploratory, not a lockbox.

## Four fixed models (no alpha / rank / kernel / nonlinear grid, no search, no required win)

In the orthonormal basis Q of the TRAINING-recipient row space: U = V Qᵀ, D_Q = D Q, T = D_Q Vᵀ. Full-cross mean loss
over all n × n training rows, slope penalty λ‖B‖² with λ = 1/n (packet-24 units).
1. **ZERO.** Zero action.
2. **COMMON.** d̂ = mean_rows(D_Q) Qᵀ: the training-mean donor action projected into the training-recipient span.
3. **LINEAR.** The packet-24 full-cross no-intercept ridge `fit_full`, replayed exactly (held-cell, supplement and
   application predictions must equal packet 24 within 1e-9).
4. **AFFINE.** Zc = Z − 1 z̄, Tc = T − 1 mean_rows(T). B solves (ZcᵀZc) B (VᵀV) + mλB = Zcᵀ Tc V; then
   c = mean_rows(D_Q) − z̄ B. Prediction T̂_ij = (c + z_i B) · v_j. **The intercept is unpenalized.** This is a
   documented change: c is not treated as an extra penalized feature.
   - Gate: the centered solution equals an explicit joint primal solve in fold 0 of every query / input (≤ 1e-8), and
     the general c formula mean_rows(T) V (VᵀV)⁻¹ − z̄B equals the D_Q form. Synthetic tests cover rank-deficient Z
     (with 1 in col(Z)), rank-deficient U, λ = 0 least squares and basis rotations.

## Source-cell-held-out evaluation (primary; 12 folds)

- Fold = one source cell. Its 3 seeds are removed from BOTH donor rows and recipient columns (n = 33). Q is fitted to
  training recipients only.
- **Panel 1: held own derivatives.** T_hh/√s in native units, 3 per fold, 36 per query / input.
- **Panel 2: held donor × training recipients.** T_h,tr/√s, MSE per held donor, then per cell.
- **Contrasts.** Per-cell MSE; paired 12-cell bootstrap, 2,000 draws, seed 20260927 (packet-24 scheme; descriptive,
  because training folds overlap). Contrasts: AFFINE − LINEAR, AFFINE − COMMON, AFFINE − ZERO, LINEAR − ZERO,
  COMMON − ZERO, plus LINEAR − COMMON. Failure cells: worse than zero; AFFINE worse than COMMON.
- **Query / input combinations stay separate.** No winner is selected.
- **Centered view.** Subtract the fold's COMMON prediction from both prediction and target. Report centered correlation
  and amplitude ratio sd(pred_c)/sd(target_c). Centered residuals equal raw residuals (gate); this is NOT an independent
  win. Its purpose is to show whether apparent association comes entirely from recipient-gradient response to one shared
  donor vector. **AFFINE − COMMON MSE is the actual incremental test.**

## Training-only source diagnostics (full 36 panel and per fold; T units)

- Left projection shares onto col(Z), span(1), col([1, Z]); affine-minus-linear projection gain; residual spectrum
  (top 5 singular values).
- Orthogonal decomposition: P_[1,Z] = P_1 + P_Zc, so common energy (span 1), centered-in-col(Zc) and residual add
  to 1. The P_Z and P_1 shares overlap and are never summed. The non-orthogonality of the COMMON and LINEAR training
  prediction matrices is reported as their Frobenius cosine.
- In-sample full-cross RMSE (native) of COMMON, LINEAR, AFFINE.
- **Change of origin.** Translate z by the training mean: AFFINE predictions must be invariant (gate ≤ 1e-8). The
  LINEAR prediction change is reported as origin dependence (diagnostic only, not scored as a fifth model).

## Orientation check (algebraic; not new data)

At the same fixed recipient gradient, reversing (d, z) → (−d, −z) must negate a derivative. The untagged AFFINE
oddness error is exactly 2c · v (identity gate); report its RMS against the RMS of own derivatives on the full-36 fit.
If a sign tag σ is used (σc + zB), σ is metadata supplied by the paired experiment, not an independently recovered
personality attribute. Swapping the reader's native LOW/HIGH base (which changes its Jacobian) is NOT tested.

## Application panel (descriptive, once, after freezing)

One full-36 source fit per model scores the old 72 application pairs against the application diagonal DERIVATIVE
J_a · d_a (native). RMSE, r, centered r. No parameter change, new model or ranking afterwards. Finite native outcomes
are not evaluated.

## Private outputs and cold replay

- Save coefficients (Q, B_linear, B_affine, c_affine, dbar, z̄, s), fold coefficients, pair identities, all predictions
  and per-cell errors privately.
- A separate process rescores the application panel from the saved coefficients plus the declared numeric arrays (gate
  ≤ 1e-10).
- This is not a standalone no-LLM system: recipient Jacobians are cached model-derived inputs, and targets are directional
  derivatives, not personality labels.

## Decision branches (from the packet; no win required)

- AFFINE fits the source and transfers better only through COMMON: the omitted shared contrast explains part of the
  failure; no individualized mapping advance.
- AFFINE beats COMMON on held differences: evidence of z-dependent action at this frozen finite derivative scope, with
  magnitude and failure cells.
- Source fit improves but transfer fails: source memorization / context support remains limiting; no new search.
- Neither improves: this simple origin / intercept explanation is weakened; representation, nonlinear and finite-action
  hypotheses are preserved.

## Checksum bindings

| file | sha256 |
|---|---|
| `grain_research/r01_source_affine.py` | `3f02539484da0077623ae7bf0e0c2f4561215299d5885e58e39d316755fd1d31` |
| `grain_research/r01_source_crossed.py` | `bb237e5865c2b5392c997ec554edfcbbd8beee5e75a857530c6ca7e6118b9769` |
| `tests/test_grain_r01_source_affine.py` | `7f3fcec29bf345a35eca65aac1f3d420e896346c8bb1bd1019a3fce29999b6a2` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/action_arrays.npz` | `778a5acdd88649e31eb8b5bd0013ddab0e50bacf4cc3e0d0bbe2325b8f9c0a9f` |
| `results/grain_vector_mechanisms/conditional_operator_v1/r2/pair_order.json` | `a942fd048c42ac89a241ee27929f29f37d7599e0c9c3fb114d3d935eeabaf17a` |
| `results/grain_gptpro_foundation_v3/P05/native_action_r1/private_prepared.npz` | `b9ce2b1c8de9365adf45a1cced3cd351dc83d24f230b663104d4aead92940338` |
| `results/grain_gptpro_foundation_v3/P05/native_action_r1/fit.json` | `03e164e64b4c125b86ae20f8a2b5d2e42dde07a19e4f0105b8967454ff10f0ad` |
| `docs/GRAIN_P05_NATIVE_ACTION_PACKET_IDS_V1.json` | `4f3d38693cf8dd2024fc5a8be5a9ee2c7b4f80479bcf230d04e10a1ebe557b87` |
| `results/grain_gptpro_foundation_v3/P05/source_crossed_action_r1/results.json` | `296a1c222dbf8a21e7eb54a3864b35202eee1ae358c8ac938ee1eecf4ac05602` |
| `results/grain_gptpro_foundation_v3/P05/source_crossed_action_r1/private_heldcell_predictions.csv` | `5ad14b55250ea7f737eb36207541bd46c64fc7a6fee66e799b7a3d811d287a8d` |
| `results/grain_gptpro_foundation_v3/P05/source_crossed_action_r1/private_supplement_heldonor_mse.csv` | `8542ac7018a90dd385d8aa6bf69d4f27e975afec20d779ca32244c149ca27a7d` |
