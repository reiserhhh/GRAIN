# NEXT30: shared native material, query-dependent reading

2026-09-10. Finite mechanism extension of the shared E5 author operator. Freeze
this plan before new cache statistics or neural calls. No human outcome reads.

## Decision

Can the GRAIN/native interface retain one material representation, and if so,
what does a single query direction omit? Distinguish a common causal prefix
from a common *reader*. Do not equate the compiled E5 reader with a native
Transformer's response or mix their coordinate systems.

## Existing material and targets

- Use exactly the four previously chosen authors, view0, in
  `full_state_edit_connection_v1/private_inputs.json`; preserve its two reference
  and two evaluation roles. These are an already exposed mechanism pilot, not
  a new population assessment or a release lockbox.
- Audit full and E1-edited cached block16 states for the existing three queries:
  TF, neuroticism and confirmation-seeking. Bind material and token addresses
  to `multiquery_sensitivity_s1_r1`; no new authors or edit selection.
- Measure all four frozen whole GRAIN coordinates on four lexical slices of
  each eligible full comment, using the old minimum48-token/12-per-slice rule.
  Bind complete-token body positions to slices by character-overlap weights.
  Exclude tokens crossing a comment boundary, retain eligible coverage counts.
  No new factor fit, factor naming, labels or test-score selection.
- Native target: original candidate-conditioned scalar score after block16.
  Different query alphabets remain different numerical functions. No clinical
  interpretation of their absolute magnitudes.

## Declared calculations

Verify identical token prefixes, positions, renderer, frozen weights and causal
mask. Under this query-last layout, a body state cannot attend to the later
question. Cache discrepancies are numerical/layout questions, not semantic
effects. Audit all8 material arms, all3 query pairs; tolerance2e-5 for max state
discrepancy, also retain exact RMS/max and final-state differences. Stop native
collection for a changed-prefix material; do not silently average it away.

For full material let H have n selected body-token rows, and G contain four
slice scores allocated to the same rows. Center both across these token rows:

    C = Gc.T @ Hc / n
    D_k[t,:] = Gc[t,k] * Hc[t,:]
    J_q[t,:] = d F_q / d H[t,:]
    T_qk = <J_q, D_k>
    w_q = n * mean_token(J_q)
    T_q = C w_q + R_q
    R_qk = sum_t Gc[t,k] * Hc[t,:] dot (J_q[t,:] - mean(J_q))

This is an exact local differential identity. Cw is the uniform-token reader;
R is token-specific reading, not statistical noise. For an affine mean-embedding
reader R=0. No finite change identity or common w across authors is presumed.
Here C uses native tokens, unlike the old equal-comment E5 C.

Normalize each nonzero D_k to the selected H's RMS for a declared perturbation
size; this is a numerical unit convention, not a psychological scale. Apply
epsilon=0.001 on either side, altering only selected body positions. Keep all
query/scaffold states fixed. Compare central finite difference with the local
Jacobian contraction, and retain all four directions for all12 author/query
units. This is a direct suffix intervention at block16, not a natural-text edit
or a total effect through the already encoded question states.

## Boundaries and outputs

- Maximum12 suffix forward/backward gradient evaluations,96 finite suffix
  forwards, plus3 fixed first-author gradient replays:111 calls,900 seconds.
  No full-prefix collection, optimizer, parameter gradients, network/API calls.
- Cache all gradients privately. Preserve pause/failure state and completed
  units; resume only within remaining declared budget. Hard stop on source or
  native score replay mismatch (>2e-6), nonfinite values or budget exhaustion.
- Report query-pair gradient-field cosine, final-state differences, T/Cw/R,
  uniform-reader error relative to T, finite-difference error and coverage.
  Small derivatives use absolute errors rather than unstable relative claims.
- Unit tests: prefix dependency, character-overlap allocation, constant reader
  R=0, varying reader counterexample, offset/paired-permutation identities,
  nonlinear toy finite difference. Repeat construction and numerical analysis.
- Public aggregate report, theory/interface update and local commits. Keep
  tokens, author IDs, texts, factor rows, native states and gradients ignored.
  NEXT24-26 remain separate. A positive native tangent is not external validity.

## Primary sources guiding this packet

[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762): causal decoder masking
and attention's query/key/value distinction support the prefix argument.
[Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html):
separates attention routing and value/output transformations. We use this as a
mechanism decomposition prompt, not a claim that their small-model analysis
already establishes our personality interpretation. The differential identity
above follows the chain rule and centering algebra; it is not a new theorem.
