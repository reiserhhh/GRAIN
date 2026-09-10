# NEXT50: fixed query anchors preserve evidence selection, not all query contrast

2026-09-10. Complete; primary code frozen in `e1512df` before preparation and
prediction. Same12 previously inspected authors, two queries, four frozen GRAIN
directions, unchanged own/pooled readers. Source4 uses leave-one-author-out;
new8 uses all4 sources. No human labels, neural calls or fitted parameters.

## Finding and decision

**A source-fixed query vector can recover a useful part of current key-channel
selection. Matching the query matters for the difference between queries.**
For new8, the same-query anchor retains N-minus-confirmation contrast r=.939345
and31/32 action signs. Exchanging the two anchors instead gives r=-.890468;
using their equal mean compresses the contrast to8.4% of its native RMS.
This connects source query conditioning to how the same GRAIN directions alter
evidence selection. It does not require a freshly generated query vector in
this local numerical reader.

| new8, own reader | Same-query source Q | Other-query source Q | Shared source Q |
|---|---:|---:|---:|
| N key-action r | .937931 | .787393 | .891434 |
| N SSE / zero | .503491 | .758160 | .643439 |
| Confirmation key-action r | .903473 | .965236 | .966759 |
| Confirmation SSE / zero | .361251 | .067702 | .164534 |
| Between-query contrast r | .939345 | -.890468 | .477162 |
| Contrast SSE / zero | .535449 | 1.448545 | .924270 |
| Predicted contrast RMS | .017467 | .013346 | .005048 |
| Actual contrast RMS | .059927 | .059927 | .059927 |
| Same contrast sign | 31/32 | 12/32 | 26/32 |

All three anchors meet the **original per-query** retention rule, SSE/zero<1
for both queries. Do not retroactively label the other/shared anchors as failing
that gate. The separately declared contrast comparison distinguishes their uses:
the other-query anchor reads confirmation well but reverses the paired contrast;
the shared anchor retains a smaller difference. Keep all results, no fitted blend.

The same-query anchor recovers only29.1% of contrast RMS and31.9%/49.8% of N/
confirmation action RMS. Correlation is not amplitude recovery. Its new8 pooled
contrast r=.939477, SSE/zero=.535174, signs32/32; the two reader modes agree here.
Source4 own contrast r=.895533, SSE/zero=.254070, signs16/16. These cohorts use
different source-exclusion contracts and have different native action amplitudes.

## Statistical units and full outcomes

Each new8 aggregate correlation pools8 authors x4 directions=32 actions, not32
independent people. Source4 aggregates contain16 actions. These are descriptive
model-response correlations, not a new Big5/MBTI questionnaire result or blind
replication. Both query contexts and these authors have been used before.

Same-query new8 per-factor contrast r is .946468/.981187/.994173/.982228 for
F01/F02/F03/F04; SSE/zero=.646408/.516187/.441121/.305262. Source4 F01 is much
weaker (r=.305642), despite same-sign4/4. Every arm, cohort, mode, query and
factor is available, including contrary cases, in these aggregate tables:

- [Query actions](source_query_selection_tables/query_summary.csv)
- [Factor actions](source_query_selection_tables/factor_summary.csv)
- [Between-query contrasts](source_query_selection_tables/contrast_summary.csv)
- [Factor contrasts](source_query_selection_tables/contrast_factors.csv)

## Exactly what was replaced

Source Q is averaged **before RoPE**, then placed at the recipient's final token
position. Recipient K,V,dK and mask are unchanged. For every anchor and head,
attention and the attended value are recomputed, not borrowed from native Q:

    a = softmax(Q_source^T K / sqrt(h) + mask)
    o = sum_t a_t V_t
    key_action = sum_head,t a_t [g_head^T(V_t-o)] [Q_source^T dK_t/sqrt(h)].

Actual input RMSNorm differentials, RoPE, GQA, projection biases and output
readout are included. g=W_O^T w* uses the previously fixed shared readout.
Target = NEXT49's analytical key-channel action whose Q/K/V sum was checked
against native attention22. This is not the whole attention increment or F.

Current Q/full hidden states/target files are blocked during prediction. The
current K,V,dK were still constructed from native query-conditioned states and
transported GRAIN directions. Removing direct Q input does not remove that
conditioning. An exchanged query anchor is a functional intervention on this
reader, not a proof that a named psychological concept has been uniquely isolated.

## Reproduction and cost

Preparation1.103012s; primary prediction.373379s, numerical replay.429862s.
0 neural/scoring/backward/API calls,0 fits,0 human-label reads. All576 actions
and288 contrasts completed; four explicit forbidden-file probes,0 extra denials.
12 NPZ hashes/arrays and6 CSV files replay exactly; summaries agree. Independent
metric recalculation differs by at most6.162e-14. Actual-Q recovery tests cover
different rotary lengths; source exclusion/identical-source limits are tested.
62 related tests pass,18 existing PyTorch/Python3.14 deprecation warnings.
Ruff and whitespace checks pass; this is not a full-repository test claim.

Private packet: `results/grain_vector_mechanisms/logic_source_query_selection_v1/`.
Inputs, source bank, per-author arrays and rows remain ignored. `manifest.json`,
both prediction freezes, `analysis/summary.json` and `replay_audit.json` preserve
the immutable implementation, source lineage and numerical checks. NEXT49 is
verified through its explicit r2 reporting amendment, not by rewriting old hashes.

```bash
PY="../project persona/.venv/bin/python"
# Existing completed packet: audit only, do not overwrite/restart it.
$PY scripts/audit_grain_source_query_selection.py
# Fresh private packet using the same existing NEXT49 assets:
$PY scripts/run_grain_source_query_selection.py prepare --output /private/new_packet
$PY scripts/run_grain_source_query_selection.py predict --output /private/new_packet
$PY scripts/run_grain_source_query_selection.py evaluate --output /private/new_packet --name analysis
```

## Mainline next step

End this attention22 anchor family as planned. **Test whether the source-fixed
selection's vector contribution reaches the final declared readout.** A good
projection on local w* need not remain good under the downstream transformed
readout. NEXT51 therefore measures endpoint transport of the frozen component
and a source-only suffix reader, not another local correlation grid.

See [theory](../../docs/GRAIN_SOURCE_QUERY_SELECTION_THEORY_V1.md) and
[NEXT51](../../docs/GRAIN_SELECTION_ENDPOINT_PLAN_V1.md). No release/scorer,
lockbox, MEPS or clinical interpretation contract changed. Whole-system goal
remains ACTIVE; this finite anchor family is complete.
