# NEXT50: one source-query anchor test of evidence selection

2026-09-10. Planned, not executed. NEXT49 finds that key-side selection is a
major transported GRAIN channel, while query responses remain meaningful.
Test a concrete query-trigger/navigation primitive, not another layer search.

## Fixed question and formula

Can source-derived Q replace the current **primal attention query** for reading
the key channel, on the same current evidence K,V,dK? The output target is
NEXT49's actual K-channel action, not the whole attention or personality score.

At attention22, export pre-RoPE normalized query projections only from source4.
For each recipient and head, use exactly three source averages:

    source_own_Q: same rendered query, permitted source authors
    source_other_Q: other rendered query, same permitted sources
    source_shared_Q: equal mean of both source-query averages.

Source4 leave-one-out, new8 all4. No fitted weights, centering, head deletion,
scale search or other layers. Rotate the averaged pre-RoPE Q at the recipient's
actual final position. Keep recipient K,V,dK, masks and fixed w* unchanged.

For each anchor, recompute its complete attention competition and centered value:

    a_Q = softmax(Q_anchor^T K / sqrt(h) + mask)
    o_Q = sum_t a_Q,t V_t
    key_action = sum_heads,t a_Q,t [g_h^T(V_t-o_Q)] [Q_anchor^T dK_t/sqrt(h)].

Do not freeze the old attention weights while substituting Q, or ignore the
softmax/centered-value dependence on it. Qhere is a local neural query vector,
not the query-position set P/Q or a human psychological intention.

## Minimal execution

1. Add a compact helper/runner/tests for this single anchor family. Reuse NEXT49
   full boundary captures, actual attention22 matrices and source readout contracts.
   Verify its explicit r2 reporting amendment using `verify_evaluation` and the
   original capture/prediction hashes; primary prepare/capture/predict ASTs are
   unchanged. Do not rewrite the old manifest to satisfy a current-file hash.
   Unit tests: actual-Q recovery of NEXT49 K channel, identical-source limiting
   case, source exclusion and rotary length placement. No new neural capture.
2. Prepare pure current K,V,dK,positions/mask and shared readouts; exclude current
   primal Q from prediction inputs. Derive anchors only from source4 final states.
   Current K/V can themselves contain query-context information; removing direct
   Q input is not removing all query conditioning. Keep this distinction explicit.
3. Freeze source anchors and predictions before opening NEXT49 target K outputs.
   Separate source-excluded and new8, both modes, both queries, all four factors.
4. Report target/action r, RMS, SSE/zero, same signs and query-contrast fidelity.
   Own/other/shared comparisons address specificity versus reusable selection.
   No target-dependent arm mixture or post-outcome threshold selection.
5. If an anchor is useful (SSE/zero<1 for each query in new8 own mode), retain
   its stated scope; show pooled/source/factor failures too. If none is useful,
   keep the exact current selection function and label current Q as a meaningful
   input dependency. Either outcome determines the next whole-chain integration.

## Budget and stopping

0 neural/scoring/backward/API/fit/Y,60s numeric plus one replay. Same12 authors,
two queries and four GRAIN directions only. This is one finite anchor family,
not a sequence of residual tuning experiments at attention22. After it, return
to an assembled mainline readout/dependency experiment; do not expand heads,
layers or fit to human labels merely to make this local primitive pass.

Private packet `results/grain_vector_mechanisms/logic_source_query_selection_v1/`;
aggregate `reports/grain_vector_mechanisms/SOURCE_QUERY_SELECTION_REPORT_V1.md`.
Keep original NEXT49 r1/r2 lineage and use corrected r2 identity columns. No
release/scorer/lockbox/MEPS changes. Local commits, not remote push.

The mathematical basis is the softmax differential and the QK/OV distinction in
[Elhage et al.](https://transformer-circuits.pub/2021/framework/index.html) and
[Kamath et al.](https://transformer-circuits.pub/2025/attention-qk/index.html).
The empirical question is whether real-author source anchors transfer, not
whether a standard identity is itself a novel personality law.
