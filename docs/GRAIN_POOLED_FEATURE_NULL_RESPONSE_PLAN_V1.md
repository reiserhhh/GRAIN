# NEXT59: native response in the pooled feature map's null space

2026-09-11. Planned, not executed. NEXT58 delivered a source compensation
mechanism; this packet changes the question to input observability, not another
score/penalty grid. Mainline decision: preserve token allocation, expand the
source mean basis, or investigate conditional readout on the existing features?

## 1. Fixed materials and cache contracts

- Same12 previously exposed authors, two queries (neuroticism and
  confirmation_seeking), four GRAIN directions. No new criterion labels.
- Reuse NEXT57 `logic_complete_function_extraction_v1/private_inputs/`,
  `private_bases/`, `private_jobs.json` and their manifest hashes. Source4 uses
  own-excluded reference; new8 uses all4. No duplicated source-fit cohort needed.
- The native H16 BODY gradient already exists in
  `logic_fixed_seed_attention_v1/private_packets/<query_key>.json/.npz` (NEXT39).
  Read only `body_gradient` and scalar contrast for accounting, not predictions.
  `run_grain_fixed_seed_attention.py::native_unit` differentiates the full suffix
  starting at layer17 with respect to the H16 leaf, at the canonical BODY positions.
- Before computing results, check original job input/common-prefix hashes,
  positions, query/alphabet/contrast and target linkage to NEXT57. Check all24
  packet hashes, not just the one inspected during plan preparation. Reconstruct
  all96 original tau contractions against NEXT51 targets with declared tolerances.
- NEXT51 `capture_source`/`capture_new` has `r` at the post-attention22 endpoint:
  that is NOT the H16 BODY gradient. NEXT55's bypassed suffix gradient is also a
  different function. Do not silently substitute either cache.

## 2. Exact decomposition, no model fitting

For each frozen BODY direction D and source orthonormal PCA32 basis U:

    mean_D = mean over BODY token rows
    D_visible[t] = mean_D @ U @ U.T
    D_allocation[t] = D[t] - mean_D
    D_outside[t] = mean_D - mean_D @ U @ U.T

All non-BODY rows are zero. Compute one direction at a time to bound memory.
These sum to D. Their pairwise Frobenius inner products are zero. The last two
are in the kernel of Dphi; visible retains the original feature derivative.
Use the original input dtype-to-FP64 convention, not newly perturbed token text.

With the cached original gradient g, compute all three signed responses
tau_j=<g,D_j>, their sum, source/new and query/factor aggregate RMS, signed
projection onto original tau, and every response cross-term. No clipping of
negative contributions or normalizing by near-zero individual tau. Publish raw
aggregate RMS as well as ratios. Undefined denominators stay undefined.

If DL[D_allocation + D_outside] is nonzero while Dphi is zero, no differentiable
function of this same phi can match that local native response. This proves a
feature-contract limitation, not absence/presence of human personality or
necessity of every native model sensitivity in a measurement instrument.

## 3. Analysis and numerical checks

- Tiny synthetic tests: uniform gradient; token-heterogeneous gradient with
  zero-mean allocation; basis-orthogonal gradient/mean; response cancellation;
  joint orthogonal re-expression; zero direction/undefined ratio.
- Source U orthonormality and direction reconstruction<=1e-6 relative; feature
  null residual<=1e-8 max(1,norm(Dphi)); pairwise energy identities<=1e-8 relative
  in FP64. Check observed cached basis precision before any outcome inspection.
- Native tau cache reconstruction uses atol1e-5, rtol3e-4 (existing full-chain
  precision scale); stop on a contract mismatch, do not relabel a gradient.
- Existence is a local numerical witness, not an arbitrary r promotion gate.
  Report the maximum numerical identity discrepancy alongside effect sizes.
  For descriptive branch priority use pooled squared contributions with signed
  cross-terms and all four factors, never choose a factor by human labels.
- One analysis plus one exact cached replay, independently verify projection
  and contraction on a fixed first unit. Zero fits/API/neural/Y,60s CPU cap.
  If cache identity fails, document the exact missing binding before considering
  new native calls; no implicit long run or alternate-target rescue.

## 4. Deliverables and branch rule

New module/runner/tests under `grain_research/vector_mechanisms/`, `scripts/`,
`tests/`; primary committed before results. Private packet:
`results/grain_vector_mechanisms/logic_pooled_feature_null_response_v1/`.
Aggregate report/tables and theory update record exact hashes and12 unique authors.

- Allocation carries response: next source-only input must retain token-addressed
  or conditional weighted statistics; nonlinear heads on unchanged means alone
  do not resolve this witness.
- Outside-basis mean carries response: consider a source response-informed basis,
  with output-sensitive subspace literature and its assumptions, not a held-rank grid.
- Both are negligible here: close the input-null question for these directions;
  prioritize source support/conditional value-response learning instead.
- Mixed/canceling result: retain both signed channels and choose a compatible
  source-only representation, not delete either as noise.

No publication scorer, clinical coding, MEPS use, lockbox or release change.
One local mechanism decision finishes this packet; the larger goal stays ACTIVE.
