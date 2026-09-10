# NEXT51: does fixed evidence selection reach the endpoint?

2026-09-10. Planned, not executed. NEXT50's anchor family is closed. Return to
the complete readout path; no further attention22/head/anchor tuning.

## Decision and fixed estimands

Can the fixed source-query selection component be read at the declared final
model endpoint, and can a source-only downstream reader preserve that component?
Same12 authors, N/confirmation, four GRAIN directions, fixed block22, exact
NEXT49 boundary/weights. Target L is the original fixed logit contrast c from
NEXT39/41, not free generation, a new query or human Y. Preserve L versus older
probability F distinction. Source4 exclusion/new8 all4 remain unchanged.

Let u_K be the **final-position** attention22 key-channel hidden increment,
and r be L's gradient at the post-attention22 final-position residual. Define:

    kappa = r^T u_K
    transported_candidate = r^T uhat_K
    compiled_candidate = rbar_q^T uhat_K
    compiled_error = r^T(uhat_K-u_K) + (rbar_q-r)^T uhat_K.

uhat_K uses exactly the three frozen NEXT50 source anchors; same-query is the
primary candidate chosen by NEXT50's mechanism comparison, not a new blind choice.
rbar_q is the unweighted same-query source mean, no pooled/alpha/layer search.
This tests one last-position component, not the key responses at every position.

## Implementation sequence and input boundaries

1. Add `selection_endpoint.py`, `run_grain_selection_endpoint.py` and focused
   tests. Extend in a new helper, rather than mutate NEXT50's frozen code:
   compute full hidden uhat_K using its same attention competition and actual
   W_O. All3 anchors/4 directions; no local w* compression before suffix reading.
   Unit tests: projection on both old w* modes recovers NEXT50; actual-Q vector
   recovers NEXT49; a nonidentity suffix exposes why local projection is not
   compositional; exact two-term error identity. Freeze before native outcomes.
2. Prepare from verified NEXT49/50 caches. Freeze **all candidate vectors**
   before opening actual K targets or suffix gradients. No current Q/gradient
   in construction. Prediction input whitelist contains source Q and current
   K/V/dK, mask, positions, W_O only. Upstream encoding/transport stays native.
3. Capture the8 source-query suffix gradients first, then compile rbar with
   source exclusion. Freeze candidate scalar outputs for all authors before
   reading any new8 suffix gradients. No model fit or human labels.
4. For each of24 cached full pre-attention22 states, run the actual block22
   and downstream suffix to fixed L. Hook its post-attention norm input; obtain
   gradients w.r.t. pre22 full state and post22 residual in one reverse pass.
   Keep full sequence computation and original final norm/head shapes; select
   final-position r only after capture. Reuse `load_observer`,
   `cached_suffix_kwargs` and `capture_source_cotangent` shape discipline.
   No prefix rerun and no treating a last-token cache as a full sequence.
5. Evaluation only: compute actual u_K from current Q and exact boundary
   differential, and actual full derivative tau=<grad_pre22 L,d_pre22>.
   Compare actual/candidate component and two diagnostic assemblies:

       tau_physical_linear_patch = tau + r^T(uhat_K-u_K)
       tau_compiled_component_swap = tau-kappa+compiled_candidate.

   Both borrow native tau/other contributions: label them **replacement audits**,
   not independent predictors of L or finite edits to a personality score.
   r^T uhat uses current evaluator gradients and is an oracle-transport
   diagnostic; only rbar^T uhat is the source-only component program.

## Outputs, checks and decision

Publish aggregate component and contrast r/RMS/SSE-over-omission/signs; report
all cohorts/queries/factors/anchors. One table separates selection transport
error, compiled-reader error and their cross-term. Report component magnitude
relative to tau with signed covariance, not a personality-information share.

Retain source-only same-query component if new8 SSE/zero<1 separately for both
queries; contrast and physical-patch results separately determine how to use it.
If local NEXT50 fidelity does not carry through, locate whether the difference
is selection error or readout transport using the fixed two-term decomposition.
Either result finishes this component family. Do not tune rbar on new8, add
anchors/heads/layers or repeat native calls to turn a negative result positive.
Next decision must reduce an actual upstream/runtime dependency or assemble
an existing light-input program, not initiate another local decimal chase.

At most24 suffix forward/reverse pairs,180s native compute; source8 first.
Two fixed JVP-versus-gradient checks (first source/new N, F01) plus at most4
central-difference forwards for final-position injection, within the same180s.
Loading separate; numerical phase60s plus one cache-only replay. If budget or
observer fails, preserve partial records and report the real boundary; no
automatic extension. No API, training, new labels, raw MEPS reads or lockbox use.

Private: `results/grain_vector_mechanisms/logic_selection_endpoint_v1/`.
Deliver report `SELECTION_ENDPOINT_REPORT_V1.md`, frozen component arrays,
source/evaluator capture manifests, aggregate tables, parity/replay checks,
dependency/theory update. Commit locally; no push/release/scorer changes.
