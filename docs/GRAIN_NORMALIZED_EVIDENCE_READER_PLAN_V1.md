# NEXT70: relative evidence selection with mass retained

2026-09-11. PLANNED, not executed. NEXT68/69 pooled-quadratic family is closed.
Return to the native selection mechanism already observed in NEXT50, without
pretending the new kernel selector is an actual attention head.

## One matched functional comparison

Same source4/four views, two queries, source U8, original Gaussian bandwidth and
the16 frozen centers0,4,...,60. Keep the full local1169 feature prefix. No center,
rank, temperature, alpha or dose search. Do not recenter on new/intervened authors.

For each center k and current token t use the existing Gaussian w_kt and define:

    Z_k = mean_t w_kt
    m_k = mean_t w_kt*z_t
    r_k = m_k/Z_k
    ell_k = log Z_k

Two arms with exactly144 new coefficients each (16*(8+1)), total1313:

- unnormalized_mass: local + linear readout of concat(m_k,ell_k).
- normalized_mass: local + linear readout of concat(r_k,ell_k).

Center each arm's144 features on original source4 means only; no whitening or
extra standardization. Retain ell in BOTH arms: normalized selection must not
silently erase evidence mass/choice. The summaries are invertibly related when
Z>0, but a linear readout and isotropic Ridge are not invariant to that nonlinear
coordinate change. Thus this tests functional use, not creation of information.

Use stable log-sum-exp for ell and normalized weights. For unnormalized mass,
if exp(ell) cannot be represented or any output/derivative is nonfinite, stop
and record the numerical boundary rather than clipping values/derivatives.

## Source teaching and independent application

Four numeric fits total (two arms x two queries), max180CPU seconds,0newneural/Y.
Exactly the NEXT68/69 score, density, last-delta, untouched-B-delta losses and
their unchanged source units; each coefficient1, alpha=.01. Fit only source4.
Verify the frozen matched-local normal prefix; no refitting earlier controls.

Application only reads current H16/layout/GRAIN and frozen models. Freeze48
two-arm prediction packets before evaluating current native scores/gradients.
No new8 targets, questionnaires, MEPS, lockbox data or native intermediate
outputs may enter fitting or application. H16 remains an encoder dependency.

## Acceptance outputs and stop

- Primary development comparison: source B response error in coordinate space,
  complete hidden-gradient space and all four GRAIN directions; compare normalized
  against matched unnormalized, and retain NEXT68/69 outputs without a new fit.
- Exposed new8: whole native-score r/RMSE, all GRAIN actions and allocation error.
  Keep query-specific gains and losses, not only a pooled headline.
- Directly test scalar/gradient finite differences, mixed derivatives, stable
  softmax formula, mass-preserving invertibility and the source-only centering.
- Mathematically verify common log-weight shift: r unchanged, ell changes by that
  shift. Do not call the whole mass-retaining predictor invariant.
- Save code/lineage freeze, private predictions, audit, aggregate tables, report,
  theory and next decision. End after this contrast; do not search64 selector sets
  if it fails. Separate a usable local mechanism from a full predictor replacement.

Expected private output `results/grain_vector_mechanisms/logic_normalized_evidence_v1/`.
New module/runner/tests in existing research namespaces. Preserve old scorer,
release, lockbox and clinical/data-use rules. Overall goal ACTIVE.
