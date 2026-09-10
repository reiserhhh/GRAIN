# NEXT77: assemble existing source readers with the numerical K/V functions

2026-09-11. PLANNED, not executed. Source-only endpoint composition, zero new
fits/neural/Y. This reuses NEXT51 rather than repeating its reader-discovery task.

## Decision

NEXT76 removes current native V but retains current fixed g. Its global V fit
improves while local action error worsens. The actionable next question is whether
the existing source-only endpoint readers can complete the current H16-to-endpoint
component path, and what component error remains after that composition. Do not
change value/selector parameters or use new8 to select a readout.

## Fixed inputs and functions

1. Inspect `scripts/run_grain_selection_endpoint.py`, its `source_reader` and
   frozen reader banks. Reuse exact split-specific `readers` from NEXT51, also
   carried as `r` in NEXT52 `private_fixed`. Verify source membership and model/
   query boundary hashes. Source4 excludes itself; new8 uses all4 sources.
2. Keep both queries and all4 GRAIN directions, all four NEXT76 selector/value
   combinations. Do not choose the lower-error cell after seeing outcomes.
3. Endpoint-specific source readout rbar_q is primary. As one matched specificity
   comparison, retain its query-pair mean (rbar_N+rbar_C)/2; these are two declared
   endpoint modes, not the previous material-specific own/pooled modes. Never
   equate numerically different target readouts because they share a mode name.
4. Compute g=(rbar^T W_O) by grouped heads, then the same local scalar and K/V/total
   derivatives as NEXT76. Application reads H16/layout/fixed matrices/source
   selector and value maps/source readers only. Current g, V/dV/K/Q, endpoint
   gradients and outcome files are forbidden. No native remainder add-back.

## Targets and accounts after prediction freeze

- Reuse existing NEXT51 source/new8 native endpoint-gradient captures; no new
  native runs. Contract matching native-Q K and V vectors from old NEXT49 with
  the corresponding current endpoint gradient (query-specific and query-pair mean).
  K+V excludes DQ and other layers; compare only the declared component.
- Retain source-Q/native-KV with source readers as an oracle-input comparator,
  not a deployable candidate. Reproduce the prior NEXT51 key component first.
- Score fidelity refers to a fixed-reader local linear functional, not full L.
  If no original scalar reference exists, reconstruct it from frozen native
  values/weights/readers, labeled as a component rather than inventing a full logit.
- Exact error account for increment vector u and reader r:
  e=r^T(uhat-u)+(rbar-r)^T uhat; retain the cross term in squared error. The
  two terms distinguish transport error from reader substitution. This account
  does not choose parameters or assign psychological causal percentages.

## Outputs and finite boundary

Suggested module `source_endpoint_assembly.py`, runner
`scripts/run_grain_source_endpoint_assembly.py`, focused tests and publisher.
Private `results/grain_vector_mechanisms/logic_source_endpoint_assembly_v1/`;
public component/query/factor/contrast/reader-error aggregate tables, report,
dependency/theory updates. No source author IDs or raw text in public outputs.

Zero fits/neural/API/Y/MEPS,60CPU seconds including one cached replay. Fixed
source exclusion, total scalar finite differences, old-key-target equivalence,
error-sum identity and complete table inventory. Use current endpoint targets only
after candidate freeze. Preserve all negative comparisons.

Finish after the one assembly. If useful, the next missing input is the native
H16 encoder / a more faithful complete endpoint program, not another readout-alpha
grid. If poor, retain the dependency-removal result and locate which already
measured link loses the endpoint signal before defining one new mechanism card.
Overall goal stays ACTIVE; no claim of universal/theory-global closure.
