# VM05: compression and fixed-layer sensitivity closure

2026-09-06, round29. Scope repair before primary neural outcomes, not an additional
outcome-selected search. The original SAME_EVIDENCE_DESIGN_V1 explicitly includes
original-dimension Ridge and Qwen block04/block16/final_norm sensitivities. These
requirements were too implicit in the progress table; they are still required.

## Why the original-dimension contrast matters

Let z be the author vector standardized using training authors only, and let P
contain k orthonormal principal directions as rows. For any linear readout beta:

\[
\beta^\top z=(P\beta)^\top(Pz)
  +\beta^\top(I-P^\top P)z.
\]

A head fitted only to Pz cannot access the second term. PCA maximizes retained
feature variance, not covariance with a personality outcome. Consequently a weak
PCA16 head does not establish that the original representation lacks predictive
information. Conversely, a better uncompressed head may reflect a different
regularization/bias-variance tradeoff; it is not proof of a personality-specific
subspace, or that the LLM itself uses the recovered direction.

The operational contrast is paired source OOF performance with the same authors,
folds, complete text, observer, scaler and Ridge grid, changing only PCA16 versus
no PCA. It remains a sensitivity analysis rather than a new selected headline.

## Fixed follow-up grid

| Family | Conditions | Source and readout |
|---|---|---|
| Original-dimension Ridge | 3 arms x3 observers =9 | Big5-only624; BGE, E5, Qwen block10; same primary aggregation, `--no-pca` |
| Fixed-layer sensitivity | 3 arms x5 tasks x3 layers =45 | Big5-only624 or MBTI-only6294; Qwen block04/block16/final_norm; PCA16 |

Arms remain raw/cue_mask/placebo_mask. All source candidates exclude the99 old
reference authors. MBTI retains its four separate official fold assignments;
Big5 retains its own official five folds. Inner selection stays3-fold seed731,
with the existing fixed alpha/C grids. No layer is selected using bridge values.
Original-dimension MBTI classifiers are not silently added to the Ridge requirement.

These54 conditions are **not** in the45-primary-case queue at a5b91d9. Do not edit
that queue or its fingerprinted dependencies while waiting/running. Follow-up
execution must use a separate bounded output batch after the current CPU lease
is released; no model downloads, new LLM training or private data transfer is needed.

The existing entrypoint already supports these conditions:
`scripts/run_grain_vector_mechanism_external_links.py`. Reuse the common arguments
from GRAIN_VM05_FULL_NEURAL_SOURCE_QUEUE_V1, then set task/arm/feature as above;
add `--no-pca` only for the9 Ridge cases. The final layer's exact feature key is
`qwen_final_norm`. Each fit needs a fresh output,1h/8GiB supervised bound and a
saved-five-fold replay, not just an apparently successful process exit.

## Reporting and mechanism boundary

- Report all9 paired PCA contrasts and all fixed layers, including null/negative
  results. Use author-paired uncertainty, separate pooled and mean-fold estimates,
  and label fitted-head bootstrap versus refit uncertainty correctly.
- Do not replace a failed/partial output or expand ranks/layers to rescue it.
  Record resource/protocol failures and resolve them without inspecting bridge.
- The source Qwen body mean/std readout has3072 coordinates. A final decision
  state or its score gradient has1536 coordinates. They are different sites and
  different mathematical objects, even when produced by the same checkpoint.
  Compare their predictive outputs on common authors; do not subtract their
  coefficients or claim the same axis without a separately specified mapping.
- Internal history, equal-view/task channels, same-ready GRAIN increments,
  direct scoring and stronger source nulls remain separate requirements. Neither
  a compression advantage nor a layer advantage alone closes the mechanism claim.

Current status: protocol coverage explicit;54 follow-up fits **not run**. No new
human-outcome result is reported here. Bridge outcomes and MEPS scale values stay
closed under the existing source-head freeze gate.
