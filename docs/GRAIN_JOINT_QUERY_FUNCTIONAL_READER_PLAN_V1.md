# NEXT89: teach the missing paired-query observations

2026-09-11. **PLANNED, not executed.** Follows NEXT88/pass77. The frozen source12
expansion improved F02/F03 functional coordinates and reduced total action error
5.26%, while confirmation absolute reading worsened. Keep that progress; do not
restart a source-count/kernel grid or infer that fixed common m mathematically
forces failure. Contrast-only observation and its chosen null complement are
distinct restrictions.

## Decision and exact intervention

Does teaching both absolute query observations improve the same-material
effective reader, rather than only calibrating their contrast? One new joint
coordinate function, source12 only. Same upstream all4 programs, same input
features/scales, same row weight45.417633 and numerical lambda .003201, same
five row units from NEXT85. No new source selection or native captures.

For q=N/confirmation and j=scalar/F01..F04 use original source values only:

```
r0_q = m0 + sign_q*d0/2
R_q[j] = compiled_p_q[j] / old_unit[j]
c_star[q,j] = (native_p_q[j] @ native_r_q
              - compiled_p_q[j] @ r0_q) / old_unit[j]
```

One fixed Gaussian12-position solve produces10 coordinates (120 coefficients).
This deliberately doubles functional observations versus NEXT88's five contrast
coordinates; it is not a matched parameter-count superiority claim.
No native derivative labels, clamp positions, new normalization, tuning or Y.

Apply independent per-query minimal lifts with the old GRAIN-field baseline:

```
r_hat_q = r_base_q + R_q^+ (c_hat_q - R_q (r_base_q-r0_q))
```

Each five-row frame must have full numerical row rank under the existing
machine-precision rule; report conditions, stop on rank failure, do not adjust
rank or regularization using outcomes. The unobserved complement remains the
old baseline for each query. The common reader may now vary; it is an effective
gauge choice, not a unique recovered native mechanism. A fixed-common lift can
sometimes produce the same10 observations; this is already a synthetic witness,
not an additional model-selection branch.

Keep anchored K/V actions separate from the total derivative of the changing
reader function. For the latter include baseline, coordinate and frame changes.
Source coordinates encode anchored action **values**, not their second derivatives.

## Existing artifacts and isolated stages

1. Verify NEXT88 `audit.json`, all source12 teacher/capture/model hashes and
   current8 membership. Bind new script/module/test/plan before any new solve.
2. Source construction reads only
   `logic_source_diversity_teaching_v1/teachers/private_teachers.npz` and the
   old fixed baseline/units. This contains12 paired current/native five-row
   packets and native source readers, plus12 fixed input positions. Calculate
   the10targets directly; no recipient arrays or raw text needed.
3. One source fit. Recipient application reads H16, the same fixed GRAIN values,
   compiled upstream programs and the old baseline, never current native Q/K/V,
   readers, human labels or evaluation targets.
4. The cached NEXT85 frame derivatives average both queries, so they are **not**
   sufficient for the new per-query frames. Recompute numeric per-query frame
   finite differences using the unchanged compiler,4directions x2signs x8authors
   x2queries=128compiled query calls per application, epsilon1e-5 unchanged.
   Reuse first-author perturbations for anchored/full derivative checks; no
   additional model forward. Freeze prediction arrays before target access.
5. Evaluate once then exact numeric replay (another128compiled query calls).
   Keep all22 previous comparator rows, append the joint reader; no replacements.
   Audit role allowlists, units, derivative identities, count/cost and hashes.

Suggested new module `joint_query_reader.py`, runner
`scripts/run_grain_joint_query_functional_reader.py`, private
`results/grain_vector_mechanisms/logic_joint_query_functional_reader_v1/`, public
`JOINT_QUERY_FUNCTIONAL_READER_REPORT_V1.md` and aggregate-only tables.

## Outputs and budget

Retain the six core tables for23readers and all three existing query views;
expected counts552/1656/6624/299/23/2208 if the current schema remains unchanged.
Append separate source-fit10rows, coordinate-target20rows, per-query frame
geometry summaries, pair-mean scalar/action comparisons for native12,
compatible12 and joint12, and signed errors with the **new** common reader.
Do not reuse a decomposition formula that silently assumes m=m0 for the new
reader. Scalar contrast/mean, each absolute query, centered variation, amplitude
and all four anchored factors remain separate. Publish no author-level rows.

Hard bounds:1kernel solve/120coefficients,256compiled query calls including
replay,120cumulative numerical CPU seconds; preflight/hash/tests excluded. Zero
neural/API/LoRA/MEPS/lockbox/raw-text calls. Evaluate cost after the first two
application authors, never use new8 results for stopping or model adjustment.
If capped, preserve complete units and record the uncompleted stage; do not
report partial data as the full eight-author panel.

## Finish and subsequent mechanism

Retain observed improvements and query-specific costs. If joint observations
improve both absolute reading and useful GRAIN responses, develop this interface.
If not, source expansion and output observation have both been tested: the next
decision is input representation sufficiency/local conditional coverage, not
another loss-weight or query-choice search. Either outcome closes this finite
card. Reused new8 is discovery evidence; H16 remains a neural encoder input and
these are local model components, not questionnaire/clinical validity.

The same-material interpretation interface retains evidence addresses, query
definition, sign and numerical observation; independent relationship-event
coding remains a downstream option, never an LLM-generated self-validating label.

Basis: [NEXT88 theory](GRAIN_SOURCE_DIVERSITY_TEACHING_THEORY_V1.md),
[NEXT88 report](../reports/grain_vector_mechanisms/SOURCE_DIVERSITY_TEACHING_REPORT_V1.md),
[conditional vector regression](https://arxiv.org/abs/1205.4656),
[task-based learning](https://arxiv.org/abs/1703.04529).

## Implementation clarification before fitting

The old NEXT82 projection helper sets the shared-reader derivative to zero.
That was valid for its constant common reader, not for this joint function.
The new projection includes `mean(D r_N, D r_C)` in the shared mode; synthetic
whole-function checks cover both modes.64anchored and64diagonal FD cells are
expected for the one new reader, using existing128compiled application calls.

Add a three-row `reader_pair.csv`: native12/compatible12/joint12 whole paired
reader error split into `2*mean_error2 + difference_error2/2`. This is a numeric
account of the new common channel, not a new fitting objective. The13public table
counts are552/1656/6624/299/23/2208/10/20/4/96/48/1/3. Old common-mode comparator
rows remain unchanged; new common-mode rows include their actual derivative.
