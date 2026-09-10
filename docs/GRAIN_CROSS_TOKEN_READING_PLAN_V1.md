# NEXT67: does changing one BODY region change the reading of an untouched region?

2026-09-11. PLANNED, not executed. NEXT66 closes the response-weighting candidate:
even explicit supervision leaves source delta near zero quality; the576-column
conditional dictionary represents11.34%/8.66% of paired density target energy.
More local centers or another weight do not remove its exact cross-token zero.

## Mainline decision

Current scalar is a(h)+mean_t f(H_t,p_t,h). Holding last h fixed, changing token
region A cannot change its gradient on disjoint region B. Determine whether the
native personality-query function actually has that missing finite dependence,
then develop a coupled pooled readout instead of another local kernel family.
This is a new missing mechanism, not an accuracy or donor/dose sweep.

## Frozen material and one intervention

Use the same original source4, both queries and pure H16/state/layout/GRAIN caches.
No new8 targets or inputs, humanY, MEPS, natural-text regeneration or retraining.
Native observation reuses `context_reading_interaction.native_body_gradient`,
`context_carrier.cached_suffix_kwargs` and the existing frozen Qwen observer loader.
Preserve model/alphabet/contrast/token positions/mask/rotary hashes from NEXT63.

For each source/query context with T BODY tokens:

1. Split BODY indices by existing position: A=first floor(T/2), B=remainder.
   These are computational regions, not independent documents or trait/state clocks.
2. Restrict each of the four original GRAIN directions to A; normalize each by
   its Frobenius norm, then sum with equal weights. If any direction or the sum
   is effectively zero (<=1e-12 in native units), stop this cell as unsupported;
   do not choose a different winning factor. Set final unit direction D_A.
3. Set epsilon=.01*||H_A||_F and states H_plus/minus=H +/- epsilon*D_A, modifying
   only A; BODY B, last h and all non-BODY rows remain byte-identical after cast.
4. Capture native fullL and BODY gradients at plus/minus. Same cached baseline
   gradient may be used for normalization; one unchanged native replay must pass
   NEXT63 score1e-4 and relative gradient3e-4 tolerances before proceeding.
5. On B compute delta_g_B=(g_plus_B-g_minus_B)/2, relative norm against baseline,
   and its contractions against ALL FOUR ORIGINAL GRAIN directions restricted to B.
   Also report ordinary local A changes, fullL plus/minus changes and realized
   perturbation norm after cast. Do not recompute the GRAIN direction at changed H.
6. Apply frozen NEXT66 scalar on the same states; its B gradient should remain
   exactly unchanged up to1e-12. This is the parameter-independent null comparator.

Delta is a finite central response, not the exact Hessian at a point. A nonzero
native effect shows cross-region coupling at the H16 boundary; it does not alone
identify attention normalization as the unique cause or prove psychological change.

## Cost and checks

One unchanged native replay + 16 plus/minus forward/backwards =17 calls planned,
17 maximum,120s native execution/300s loading. Sequential, first2contexts cost
projection with20s reserve; preserve partial packets on a bounded stop.
No rerun because a p-value is weak; no additional epsilon/region grid.
GPU observer uses the existing local tested MPS path; no model/data transfer.

Track liveness only for this specific process; no restart of any old queue.
Source path guard denies current/new8 targets; model loading precedes guard using
existing credential-free local model files. Save exact state masks and input hashes.
Synthetic tests: local-sum zero, normalized-pooling nonzero, disjoint-region byte
invariance and fixed-direction contraction. Native arithmetic/replay uses caches.

## Deliverables and next branch

- `grain_research/vector_mechanisms/cross_token_reading.py`,
  `scripts/run_grain_cross_token_reading.py`, focused tests.
- Private `results/grain_vector_mechanisms/logic_cross_token_reading_v1/` with
  manifest, source membership, both native gradients, bounded call/cost ledger.
- Public aggregate `CROSS_TOKEN_READING_REPORT_V1.md`: both queries, all four
  directions, per-query source counts, gradient/score change magnitudes, current
  scalar zero witness, replay resolution and dependence diagram. No author text/IDs.
- Mechanism detection uses native B change compared with unchanged-replay error,
  not a new r>.5 gate; report every supported cell and resolved effect size.
  Never use same-author tokens as independent subjects for significance.
- If native nonlocal response exists, NEXT68 develops a compatible nonlinear
  post-pooling/normalized-selection scalar tied to this response. If negligible,
  examine source query-field dependence in the retained token representation;
  do not pretend the absence across one direction exhausts all coupling.

Keep release/scorer/lockbox/clinical boundaries unchanged. Source4 mechanism work
is not a new human-validation panel. See
[theory and primary sources](GRAIN_PAIRED_RESPONSE_OBJECTIVE_THEORY_V1.md).
