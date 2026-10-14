# R01 product-space identification of the frozen z → action estimators: protocol V1

Frozen 2026-09-26 JST before any quantity was computed.
- **Authority.** `NEXT_PACKET_18_PRODUCT_SPACE_AND_RELIABILITY.md` Part A; `SEMINAR_FABLE_16_DECISION.md`, whose
  corrections 1 and 2 are binding.
- **Runner and output.** `scripts/run_grain_r01_product_space.py`, writing to
  `results/grain_gptpro_foundation_v3/P05/product_space_identification_r1/`.
- **Budget.** CPU ≤ 5 min, 0 neural calls, no fitting, no remedy, no hyperparameter grid.
- **Status of the data.** Every panel was used before. This is a diagnostic, not fresh confirmation, and not human
  accuracy.

## Inputs (frozen; hash-pinned in the runner)

- **Standardized inputs.** z4 and z24: `native_action_r1/private_prepared.npz` (sha b9ce2b1c…), rows in r2 pair
  order, source-standardized, with every coordinate kept.
- **Source gradients.** J_i: 36 source LOW gradients (`native_action_r1/private_jacobians`, tree sha 35e1b6d4…).
- **Application gradients.** J_*: 72 application LOW gradients (`application_j_diagnostic_r1/private_app_jacobians`,
  tree sha 8601b303…).
- **Metric scale.** s_q = D_q / mean_i ‖J_i‖², recomputed and checked against `native_action_r1/fit.json` (sha
  03e164e6…): E 1,278,169.05; N 1,159,832.60. α = 1.
- **Donors.** Source donor vectors d_i = r2 `dh` rows (source), per query segment.
- **Reference tangents.** Packet-17 native tangents L(c) for E4, E24, J4 and J24 (`components_by_change.csv`, sha
  e3d0db8a…) and signed tangent mismatch T (`decomposition_vs_donor.csv`, sha 5d6e987e…).
- **Material cells.** axis|family|situation, from the packet-5 IDs file.

## Definitions (original standardized coordinates, Frobenius metric on W ∈ R^{p×D})

- **Map and source rows.** A linear map W acts as a(z) = Wᵀz, a native head-state change. U_i = √s J_i. The source
  observation rows are v_i = vec(z_i U_iᵀ).
- **Kernel.** K_ij = (z_iᵀz_j)(U_iᵀU_j), a 36 × 36 matrix. No Kronecker matrix is formed.
- **Rank rule.** Eigendecompose K and keep λ > λ_max · 1e-10. K⁺ uses the kept modes only.
- **Exposure of an application row** v* = vec(z* U*ᵀ):
  - g_i = (z_iᵀz*)(U_iᵀU*) and q = ‖z*‖²‖U*‖²;
  - exposure = gᵀK⁺g / q;
  - squared unsupported norm = q − gᵀK⁺g, in U-scaled W units.
  - If q = 0 the exposure is UNDEFINED, not perfect.
- **J-cell reconstruction (binding check).**
  - Native prediction L̂_J = gᵀ(K + I)⁻¹t / √s, with t_i = U_iᵀd_i.
  - It must match packet-17 L(J4) and L(J24) for all 72 application rows and both queries.
  - Scale-aware tolerance: max |L̂ − L| ≤ 1e-8 · RMS(L) + 1e-15.
- **Euclidean reconstruction (binding check).**
  - L̂_E = Σ_i w_i (J_*ᵀd_i), with w = Z(ZᵀZ + I)⁻¹z*.
  - It must match packet-17 L(E4) and L(E24) at the same tolerance.
- **Source leave-one-out reference.** For each source row i: g_{−i}ᵀK_{−i}⁺g_{−i} / K_ii, using the other 35 rows and
  the same rank rule.
- **Span coverages, reported separately from product-row exposure.**
  - J-span share: ‖P_J U*‖²/‖U*‖², with P_J projecting onto the source U rows.
  - Z-span share: ‖P_Z z*‖²/‖z*‖², with P_Z projecting onto the source z rows.
  - Both use SVD with singular values > s_max · 1e-10.
- **Crossed controls (descriptive; fixed; no new data).**
  - CZ: application z with source J. For each source k, v = vec(z* U_kᵀ). Report the mean exposure over k.
  - CJ: source z with application J. For each source k, v = vec(z_k U*ᵀ). Report the mean exposure over k.
- **Aggregation.** Material-cell means (3 seeds kept together). Cluster bootstrap by cell, 2,000 draws, seed
  20260926.

## Null witness

- **Construction.** β = K⁺g; ΔW = z*U*ᵀ − Σ_i β_i z_iU_iᵀ.
- **Source check (relative).** max_i |z_iᵀ ΔW U_i| / (‖z_i‖ ‖U_i‖ ‖ΔW‖_F) ≤ 1e-8.
- **Application check.** z*ᵀ ΔW U* = ‖ΔW‖²_F, to relative error ≤ 1e-8.
- **Unit witness.** Ŵ = ΔW / ‖ΔW‖_F. Its native effect on the application row is ‖ΔW‖_F / √s, with zero effect on
  every source row.
- **Comparison.** The frozen J24 map's native effect per unit Frobenius norm on the same row. The magnitude is fixed
  by unit normalization, not chosen.
- **Exactness.** If the rank rule drops positive eigenvalues, the witness is reported as a numerical or approximate
  null, not an exact algebraic non-identification.

## Prespecified sensitivity (one)

- **Source-Z whitening.** Σ_z = ZᵀZ/36. Keep modes with eigenvalue > max · 1e-10, and set z̃ = Λ^{-1/2}Vᵀz. Recompute
  K, exposure and leave-one-out.
- This reports metric dependence. Neither geometry is canonical.

## Euclidean weights (descriptive)

- **Per application row:** Σw, ‖w‖₁, ‖w‖₂, ridge leverage h = z*ᵀ(ZᵀZ + I)⁻¹z*, and max |z*| with the declared
  input-support flag (source max |z|: 2.94 for z4, 3.14 for z24).
- **Spread:** the spread of the source donor actions under the application tangent, J_*ᵀd_i.
- **Associations:** Spearman correlations of ‖w‖₁ and h with the signed tangent error T and with |T| (packet 17), per
  query.
- **Native units kept.** No per-pair ratios to tiny donor effects.

## Interpretation limits

- **What is shown.** Shared J span versus joint identification in the product space is a finite algebraic result.
- **What is not.** Neither low exposure nor leverage proves overfitting, irreducible input loss, or the unique cause.
- **No thresholds.** No verdict thresholds are used.
- **Acceptance.** Exact reconstructions and numerical consistency, plus an interpretable description. Stop after the
  diagnostic, even if it is inconclusive.
